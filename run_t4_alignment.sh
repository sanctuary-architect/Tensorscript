#!/usr/bin/env bash
# ============================================================
#  TensorScript T4 - train Llama-3-8B (SFT + DPO) on a GPU
#  One command. No coding needed.
#
#  USAGE:
#    1. Put this script on a machine with an NVIDIA GPU
#       (Google Colab T4 works: open a terminal, paste this
#        file, then run it).
#    2. Make it runnable:   chmod +x run_t4_alignment.sh
#    3. Set your HF token:  HF_TOKEN=hf_xxx ./run_t4_alignment.sh
#
#  What it does: installs deps, writes the training spec,
#  generates the training code from it, then runs a full
#  SFT + DPO alignment pipeline on Meta-Llama-3-8B.
# ============================================================
set -euo pipefail

# ---------- config ----------
WORKDIR="${1:-$HOME/tensorscript-run}"
SEQ_LEN="${SEQ_LEN:-1024}"
EPOCHS="${EPOCHS:-1}"

echo ""
echo "===================================================="
echo "  TensorScript T4 - Llama-3-8B SFT + DPO"
echo "===================================================="
echo "Working dir : $WORKDIR"
echo "Seq length  : $SEQ_LEN   Epochs: $EPOCHS"
echo ""

# Remember where this script lives (the toolchain may sit next to it).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "$WORKDIR"
cd "$WORKDIR"

# ---------- 1. check GPU ----------
echo "[1/5] Checking GPU..."
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
else
  echo "WARNING: No NVIDIA GPU detected. Llama-3-8B really needs one."
fi
echo ""

# ---------- 2. install deps ----------
echo "[2/5] Installing dependencies (first run only)..."
pip install -q --upgrade transformers peft accelerate datasets trl bitsandbytes wandb 2>&1 | tail -2 || true
echo ""

# ---------- 3. write the spec ----------
echo "[3/5] Writing TensorScript spec..."
cat > align.spec.tensor <<'SPECEOF'
tensorscript v1.0

dataset SFTData streams {
    source: "hf://datasets/HuggingFaceH4/ultrachat_200k",
    tokenize: AutoTokenizer("meta-llama/Llama-3-8B"),
    sequence_length: 1024,
    split: [train: 95%, val: 5%]
}

model BaseLM {
    base: "hf://meta-llama/Meta-Llama-3-8B",
    quantize: 4bit,
    peft: LoRA(r=16, alpha=32, target_modules="all-linear")
}

monitor RunTelemetry {
    track: [loss, perplexity, memory_usage, reward_margin, kl_divergence],
    checkpoint_every: 20.steps,
    select_best_on: loss minimize,
    destination: "wandb://pioneer-space/sft-dpo-pipeline"
}

optimize BaseLM as SFTStage {
    using: dataset.SFTData.train,
    hardware: {profile: "t4"},
    telemetry: monitor.RunTelemetry,
    schedule: CosineAnnealing(max_lr=2e-4, min_lr=2e-5, warmup=0.03),
    epochs: 1,
    guardrails: [
        if loss > 2.5: rollback_and_scale_lr(0.5),
        if loss flat_lines(epsilon=1e-4) for 200.steps: terminate_early
    ]
}

dataset PreferenceData streams {
    source: "hf://datasets/Anthropic/hh-rlhf",
    tokenize: AutoTokenizer("meta-llama/Llama-3-8B"),
    sequence_length: 1024,
    split: [train: 90%, val: 10%]
}

model DPOModel {
    base: pipeline.SFTStage.best,
    quantize: 4bit,
    peft: LoRA(r=16, alpha=32, target_modules="all-linear")
}

optimize DPOModel as DPOStage {
    using: dataset.PreferenceData.train,
    hardware: {profile: "t4"},
    telemetry: monitor.RunTelemetry,
    schedule: CosineAnnealing(max_lr=5e-6, min_lr=5e-7, warmup=0.1),
    epochs: 1,
    guardrails: [
        if reward_margin < 0: rollback_and_scale_lr(0.3),
        if kl_divergence > 10: rollback_and_scale_lr(0.5)
    ]
}

pipeline AlignmentRun {
    stage one {
        run: optimize.SFTStage
    },
    stage two {
        run: optimize.DPOStage,
        depends_on: one,
        inherit_weights: best
    }
}

SPECEOF
echo "  wrote align.spec.tensor"
echo ""

# ---------- 4. toolchain + transpile ----------
echo "[4/5] Validating + generating training code..."
TOOLCHAIN_OK=0
for f in parser.py transpiler.py semantic_checks.py tsc_check.py lexer.py; do
  if [[ -f "$SCRIPT_DIR/$f" ]]; then TOOLCHAIN_OK=1; fi
done
if [[ "$TOOLCHAIN_OK" == "0" ]]; then
  echo "  (toolchain files not next to the script - fetching from GitHub)"
  REPO="${TENSORSCRIPT_REPO:-https://raw.githubusercontent.com/YOUR-ORG/tensorscript/main}"
  for f in parser.py transpiler.py semantic_checks.py tsc_check.py lexer.py; do
    curl -fsSL "$REPO/$f" -o "$f" || echo "  ! could not fetch $f (set TENSORSCRIPT_REPO)"
  done
else
  echo "  (using toolchain files from next to the script)"
  cp "$SCRIPT_DIR"/parser.py "$SCRIPT_DIR"/transpiler.py "$SCRIPT_DIR"/semantic_checks.py "$SCRIPT_DIR"/tsc_check.py "$SCRIPT_DIR"/lexer.py .
fi
python3 - <<'PYEOF'
import sys, os
sys.path.insert(0, ".")
from parser import parse
from semantic_checks import check
from transpiler import transpile

ast = parse(open("align.spec.tensor").read())
check(ast)
files = transpile(ast, out_dir=".")
for name, src in files.items():
    if name.endswith(".py"):
        with open(name, "w") as f:
            f.write(src)
for n in ["SFTStage.py", "DPOStage.py", "run_AlignmentRun.py"]:
    assert os.path.exists(n), f"missing {n}"
print("  generated SFTStage.py, DPOStage.py, run_AlignmentRun.py")
PYEOF
echo ""

# ---------- 5. run ----------
echo "[5/5] Running the pipeline (SFT then DPO)..."
echo "  -> downloads ~16GB of model + data, then trains"
echo "     $EPOCHS epoch(s). On a T4 expect ~30-60 min per stage."
echo ""
export HF_TOKEN="${HF_TOKEN:-}"
python3 run_AlignmentRun.py

echo ""
echo "===================================================="
echo "DONE. Outputs are in $WORKDIR/output/"
echo "  SFTStage/ ... best model checkpoint"
echo "  DPOStage/ ... final aligned model"
echo "===================================================="
