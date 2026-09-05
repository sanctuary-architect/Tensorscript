# TensorScript — Tiny Run Artifacts

Proof-of-life artifacts from the TensorScript v1.0 pipeline: a DSL spec
(`.tensor`) compiled to Python and **actually trained on this machine**
(512 MB RAM, CPU-only torch).

## Files

| File | What it is |
|---|---|
| `Tiny Example.tensor` | The DSL spec — scratch-initialized tiny LM, LoRA, `target: "manual-loop"` |
| `TinyStage Model.pt` | Trained model weights (completed run) |
| `TinyStage Tokenizer.json` | Tokenizer vocab (from `hf-internal-testing/tiny-random-gpt2`) |
| `TinyStage Tokenizer Config.json` | Tokenizer settings |

## How it was produced

1. `tsc_check.py` validated the spec (semantic checks incl. imports + guardrail metric tracking)
2. The transpiler compiled it to `TinyStage.py` — a compact PyTorch training loop
   (`target: "manual-loop"` path; no HF Trainer, so it fits 512 MB)
3. Ran on real torch: dataset loaded, model built from config, LoRA applied,
   several training steps executed, weights saved

## Guardrails

The generated stage carries a `GuardrailCallback` wired into the loop.
Verified working (both numpy harness and real-torch harness):

- `rollback_and_scale_lr` — spikes trigger LR scaling, weights roll back once a
  checkpoint exists
- flat-line detection + early termination
- Multi-rule composition: multiple triggers multiply their scale factors

## Status of the broader project

- `verify.sh` (in the project) passes all 5 layers: `tsc_check` on both examples,
  regeneration, numpy harness, real-torch harness, artifact inventory → `VERIFY OK`
- Real-run blockers fixed: `hf://` prefix stripping, wandb guard, transformers 5.x
  arg names (`eval_strategy`, `warmup_steps`), missing `labels` in tokenize, DPO
  dead guardrails (now emitted via `compute_metrics` + enforced by semantic check),
  `evaluate` docs-vs-implementation gap (stub now emitted)
- The full DPO/SFT 8B path needs a GPU box — harnesses prove guardrail logic; real
  scale needs real hardware

## Reproduce

```bash
python3 tsc_check.py Tiny\ Example.tensor
# then regenerate + run TinyStage.py in the project dir (torch + transformers installed)
```

## Running the real 8B pipeline on a Colab T4

`TensorScript T4 Colab.ipynb` is a ready-to-run Colab notebook (T4 GPU runtime)
that drives the full SFT + DPO pipeline on **Meta-Llama-3-8B**:

- Uses QLoRA (4-bit base via bitsandbytes), fp16, batch 1, gradient
  checkpointing, eager attention — all driven automatically by the
  `hardware: {profile: "t4"}` block in the spec (see `Colab Alignment Example.tensor`)
- Validates + transpiles in-notebook, then runs `run_AlignmentRun.py`
  (SFT first, then DPO on `pipeline.SFTStage.best`)
- Needs a Hugging Face read token (Llama-3-8B is gated — accept Meta's license)
- Sequence length 1024 fits ~12–14 GB VRAM; drop to 512 if you hit OOM

The T4 code paths (fp16 compute dtype, `attn_implementation='eager'`,
gradient checkpointing, batch 1, scratch-config fallback) are verified against
real transformers here on CPU — the notebook just needs the GPU to cross the
finish line.
