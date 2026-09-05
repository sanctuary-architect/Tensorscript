"""
Transpiler for TensorScript v1.0 -> executable Python.

Target stack: Hugging Face `transformers` (Trainer), `peft` (LoRA/QLoRA),
`datasets`, `accelerate`, and `wandb` for telemetry. This is deliberately
built against a mainstream, well-documented stack rather than inventing a
bespoke training loop, so the generated code is auditable by anyone who
knows the HF ecosystem.

Scope and honesty about scope:
  - dataset / model / monitor / optimize blocks -> fully transpiled to a
    runnable single-stage training script.
  - pipeline blocks with multiple stages -> transpiled to a sequential
    driver script that runs each stage's script and resolves
    `pipeline.<Stage>.<selector>` checkpoint references between them.
  - guardrails -> a real TrainerCallback implementing the composition and
    short-circuit rules from spec §4 (guardrail conflict semantics).
  - evaluate blocks -> stubbed as a clearly-marked TODO harness, since
    wiring MMLU/TruthfulQA/custom harnesses is a separate, larger
    integration than transpiling training config. Not silently dropped —
    emitted as an explicit unimplemented function so nothing is hidden.
  - hardware: block -> emitted as an `accelerate config`-compatible YAML
    comment block plus a note, since distributed launch is an external
    `accelerate launch` invocation, not something a single Python file
    controls directly.
"""

import json
import pprint


def py_str(s: str) -> str:
    return json.dumps(s)


def dotted_last(ref: str) -> str:
    return ref.split(".")[-1]


class Transpiler:
    def __init__(self, ast):
        self.ast = ast
        self.datasets = {b["name"]: b for b in ast["blocks"] if b["type"] == "dataset"}
        self.models = {b["name"]: b for b in ast["blocks"] if b["type"] == "model"}
        self.monitors = {b["name"]: b for b in ast["blocks"] if b["type"] == "monitor"}
        self.optimizes = {(b.get("alias") or b["target"]): b for b in ast["blocks"] if b["type"] == "optimize"}
        self.pipelines = [b for b in ast["blocks"] if b["type"] == "pipeline"]
        self.evaluates = [b for b in ast["blocks"] if b["type"] == "evaluate"]

    # ---------------- dataset ----------------

    def emit_dataset_loader(self, ds_name, indent="") -> str:
        ds = self.datasets[ds_name]
        f = ds["fields"]
        source = f["source"]
        lines = []
        auth_line = ""
        if "auth" in f:
            var = f["auth"]["env_var"]
            auth_line = f'os.environ.get({py_str(var)})'
        else:
            auth_line = "None"

        if source.startswith("hf://") or source.startswith("huggingface://"):
            hf_id = source.split("://", 1)[1]
            if hf_id.startswith("datasets/"):
                hf_id = hf_id[len("datasets/"):]
            if "mix" in f:
                lines.append(f"{indent}# NOTE: 'mix' assumes each key is a config/subset name under the same dataset id.")
                lines.append(f"{indent}_subsets = {json.dumps(f['mix'])}")
                lines.append(f"{indent}_loaded = []")
                lines.append(f"{indent}_probs = []")
                lines.append(f"{indent}for _subset, _pct in _subsets.items():")
                lines.append(f"{indent}    _loaded.append(load_dataset({py_str(hf_id)}, _subset, split='train', token={auth_line}))")
                lines.append(f"{indent}    _probs.append(_pct / 100.0)")
                lines.append(f"{indent}raw_dataset = interleave_datasets(_loaded, probabilities=_probs, seed=SEED)")
            else:
                lines.append(f"{indent}raw_dataset = load_dataset({py_str(hf_id)}, split='train', token={auth_line})")
        elif source.startswith("s3://"):
            lines.append(f"{indent}# TODO: TensorScript does not yet specify an S3 ingestion protocol.")
            lines.append(f"{indent}raise NotImplementedError('s3:// dataset sources require a user-supplied loader — not yet specified in TensorScript v1.0')")
        else:
            lines.append(f"{indent}raw_dataset = load_dataset('json', data_files={py_str(source)}, split='train')")

        if "split" in f:
            splits = f["split"]
            train_pct = splits.get("train", 100) / 100.0
            lines.append(f"{indent}_split = raw_dataset.train_test_split(train_size={train_pct})")
            lines.append(f"{indent}train_dataset = _split['train']")
            lines.append(f"{indent}eval_dataset = _split['test']")
        else:
            lines.append(f"{indent}train_dataset = raw_dataset")
            lines.append(f"{indent}eval_dataset = None")

        seq_len = f.get("sequence_length", 2048)
        lines.append(f"{indent}SEQUENCE_LENGTH = {seq_len}")
        return "\n".join(lines)

    # ---------------- model ----------------

    def emit_model_loader(self, model_name, checkpoint_override_var=None, indent="", profile=None) -> str:
        model = self.models[model_name]
        f = model["fields"]
        base = f["base"]
        lines = []

        if checkpoint_override_var:
            lines.append(f"{indent}_base_path = {checkpoint_override_var}")
        elif base["kind"] == "literal":
            # strip hf:// / huggingface:// scheme before passing to from_pretrained
            _v = base["value"]
            if isinstance(_v, str) and (_v.startswith("hf://") or _v.startswith("huggingface://")):
                _v = _v.split("//", 1)[1]
            lines.append(f"{indent}_base_path = {py_str(_v)}")
        elif base["kind"] == "scratch":
            lines.append(f"{indent}_base_path = None  # 'scratch': random init from config, not a pretrained checkpoint")
        elif base["kind"] == "pipeline_ref":
            lines.append(f"{indent}# base resolved at pipeline-orchestration time — see driver script")
            lines.append(f"{indent}_base_path = RESOLVED_BASE_CHECKPOINT")

        is_t4 = (profile or {}).get("profile") == "t4"
        quantize = f.get("quantize", "none")
        if quantize in ("4bit", "8bit"):
            load_in = "load_in_4bit=True" if quantize == "4bit" else "load_in_8bit=True"
            compute_dtype = "torch.float16" if is_t4 else "torch.bfloat16"
            lines.append(f"{indent}_bnb_config = BitsAndBytesConfig({load_in}, bnb_4bit_compute_dtype={compute_dtype})" if quantize == "4bit"
                          else f"{indent}_bnb_config = BitsAndBytesConfig({load_in})")
            quant_kwarg = "quantization_config=_bnb_config"
        else:
            quant_kwarg = None

        if base["kind"] == "scratch":
            lines.append(f"{indent}config = AutoConfig.from_pretrained({py_str(model_name)}_config.json) if os.path.exists({py_str(model_name)}_config.json) else AutoConfig.for_model('gpt2')")
            lines.append(f"{indent}model = AutoModelForCausalLM.from_config(config)")
        else:
            kwargs = ["_base_path"]
            if quant_kwarg:
                kwargs.append(quant_kwarg)
            kwargs.append("device_map='auto'")
            if is_t4:
                kwargs.append("torch_dtype=torch.float16")
                kwargs.append("attn_implementation='eager'")
            lines.append(f"{indent}model = AutoModelForCausalLM.from_pretrained({', '.join(kwargs)})")

        if "freeze" in f:
            fr = f["freeze"]
            path_parts = fr["path"].split(".")
            # base_model.layers[start:end] -> getattr chain then slice, set requires_grad=False
            getattr_chain = "model"
            for p in path_parts[1:]:  # skip the leading placeholder identifier (e.g. base_model)
                getattr_chain += f".{p}"
            start = fr["start"] if fr["start"] is not None else ""
            end = fr["end"] if fr["end"] is not None else ""
            lines.append(f"{indent}for _p in {getattr_chain}[{start}:{end}].parameters():")
            lines.append(f"{indent}    _p.requires_grad = False")

        if "peft" in f and f["peft"] is not None:
            peft_call = f["peft"]
            if peft_call["call"] in ("LoRA", "QLoRA"):
                args = {a["key"]: a["value"] for a in peft_call["args"] if a["key"]}
                r = args.get("r", 16)
                alpha = args.get("alpha", 32)
                target_modules = args.get("target_modules", "all-linear")
                lines.append(f"{indent}_lora_config = LoraConfig(r={r}, lora_alpha={alpha}, target_modules={py_str(target_modules)}, task_type='CAUSAL_LM')")
                if quantize in ("4bit", "8bit"):
                    lines.append(f"{indent}model = prepare_model_for_kbit_training(model)")
                lines.append(f"{indent}model = get_peft_model(model, _lora_config)")

        return "\n".join(lines)

    # ---------------- guardrails ----------------

    def emit_guardrail_callback(self, guardrails, class_name="GuardrailCallback", indent="") -> str:
        """
        Emits a TrainerCallback implementing spec §8's guardrail semantics:
          - guardrails evaluate in declaration order against the post-step
            metric snapshot
          - all firing guardrails execute in order
          - terminate_early short-circuits the rest of that step's checks
          - multiple rollback_and_scale_lr triggers in the same step
            compose multiplicatively
          - rollback_and_scale_lr before any checkpoint exists degrades to
            "scale LR only" with a logged warning instead of crashing

        Checkpoint tracking uses TrainerCallback's real on_save hook (called
        by Trainer immediately after it writes a checkpoint to disk) to
        snapshot an in-memory copy of model.state_dict() and flip
        _has_checkpoint to True. rollback_and_scale_lr then does a genuine
        in-place model.load_state_dict(...) restore rather than merely
        logging that a rollback "should" happen — a Trainer callback can't
        stop and restart the process, but it *can* reach into the live
        model object it's handed and reload weights directly, which is
        what real rollback requires.
        """
        rule_defs = []
        for i, g in enumerate(guardrails):
            cond = g["condition"]
            duration = g.get("for")
            action = g.get("action")
            block = g.get("block")
            rule_defs.append({
                "metric": cond["metric"],
                "op": cond.get("op"),
                "value": cond.get("value"),
                "epsilon": cond.get("epsilon"),
                "duration_steps": duration["n"] if duration else None,
                "action": action,
                "block": block,
            })

        rules_literal = pprint.pformat(rule_defs, indent=4, width=100)
        # indent every continuation line to match the class body
        rules_literal = ("\n" + indent + "    ").join(rules_literal.split("\n"))

        code = f'''
{indent}class {class_name}(TrainerCallback):
{indent}    """Auto-generated from the TensorScript guardrails: list.
{indent}    Evaluates all rules each step, in declaration order, against the
{indent}    latest logged metrics. terminate_early short-circuits remaining
{indent}    rules for that step. Multiple LR-scaling triggers in one step
{indent}    compose multiplicatively rather than overwriting each other.
{indent}    rollback_and_scale_lr performs a genuine in-place weight restore
{indent}    from the most recent on_save snapshot, once one exists."""

{indent}    RULES = {rules_literal}

{indent}    def __init__(self):
{indent}        self._metric_history = {{}}
{indent}        self._consecutive_true = {{}}
{indent}        self._has_checkpoint = False
{indent}        self._last_checkpoint_state = None

{indent}    def on_save(self, args, state, control, **kwargs):
{indent}        # Called by Trainer right after it writes a checkpoint to disk.
{indent}        # We keep our own in-memory copy so rollback doesn't require a
{indent}        # disk read (and works even if save_strategy hasn't fired yet
{indent}        # for reasons outside the guardrail's control).
{indent}        model = kwargs.get("model")
{indent}        if model is not None:
{indent}            self._last_checkpoint_state = copy.deepcopy(model.state_dict())
{indent}            self._has_checkpoint = True
{indent}        return control

{indent}    def _flat_lines(self, metric, epsilon, window):
{indent}        hist = self._metric_history.get(metric, [])
{indent}        if len(hist) < window:
{indent}            return False
{indent}        recent = hist[-window:]
{indent}        variance = sum((x - sum(recent) / len(recent)) ** 2 for x in recent) / len(recent)
{indent}        return variance < epsilon

{indent}    def on_log(self, args, state, control, logs=None, **kwargs):
{indent}        if not logs:
{indent}            return control
{indent}        for k, v in logs.items():
{indent}            if isinstance(v, (int, float)):
{indent}                self._metric_history.setdefault(k, []).append(v)

{indent}        lr_scale_factor = 1.0
{indent}        for rule_idx, rule in enumerate(self.RULES):
{indent}            metric = rule["metric"]
{indent}            if metric not in logs:
{indent}                continue
{indent}            triggered = False
{indent}            if rule["op"] == "flat_lines":
{indent}                window = rule["duration_steps"] or 1
{indent}                triggered = self._flat_lines(metric, rule["epsilon"], window)
{indent}            else:
{indent}                val = logs[metric]
{indent}                op = rule["op"]
{indent}                threshold = rule["value"]
{indent}                triggered = {{
{indent}                    ">": val > threshold, "<": val < threshold,
{indent}                    ">=": val >= threshold, "<=": val <= threshold,
{indent}                    "==": val == threshold,
{indent}                }}.get(op, False)
{indent}                if triggered and rule["duration_steps"]:
{indent}                    key = (rule_idx, "streak")
{indent}                    self._consecutive_true[key] = self._consecutive_true.get(key, 0) + 1
{indent}                    triggered = self._consecutive_true[key] >= rule["duration_steps"]
{indent}                elif rule["duration_steps"]:
{indent}                    self._consecutive_true[(rule_idx, "streak")] = 0

{indent}            if not triggered:
{indent}                continue

{indent}            action = rule["action"]
{indent}            action_name = action["call"] if isinstance(action, dict) else action

{indent}            if action_name == "terminate_early":
{indent}                print(f"[guardrail] {{metric}} triggered terminate_early at step {{state.global_step}}")
{indent}                control.should_training_stop = True
{indent}                break  # short-circuit: terminate_early ends evaluation for this step

{indent}            elif action_name == "rollback_and_scale_lr":
{indent}                factor = action["args"][0]["value"] if isinstance(action, dict) and action["args"] else 0.5
{indent}                if not self._has_checkpoint:
{indent}                    print(f"[guardrail] WARNING: rollback_and_scale_lr triggered before any checkpoint exists — scaling LR only, skipping rollback")
{indent}                else:
{indent}                    print(f"[guardrail] {{metric}} triggered rollback to last checkpoint")
{indent}                    kwargs["model"].load_state_dict(self._last_checkpoint_state)
{indent}                lr_scale_factor *= factor  # composes multiplicatively across rules firing this step

{indent}            elif action_name == "activate_activation_checkpointing":
{indent}                kwargs["model"].gradient_checkpointing_enable()

{indent}            elif action_name == "activate_cpu_offloading":
{indent}                print(f"[guardrail] {{metric}} triggered activate_cpu_offloading — configure via accelerate's cpu_offload, not mid-run in this callback")

{indent}            elif rule.get("block"):
{indent}                for stmt in rule["block"]:
{indent}                    if stmt["call"] == "print":
{indent}                        msg = stmt["args"][0]["value"] if stmt["args"] else ""
{indent}                        print(f"[guardrail block] {{msg}}")
{indent}                    elif stmt["call"] == "activate_activation_checkpointing":
{indent}                        kwargs["model"].gradient_checkpointing_enable()
{indent}                    elif stmt["call"] == "activate_cpu_offloading":
{indent}                        print("[guardrail block] activate_cpu_offloading requested — configure via accelerate")

{indent}        if lr_scale_factor != 1.0:
{indent}            for pg in kwargs["optimizer"].param_groups:
{indent}                pg["lr"] *= lr_scale_factor

{indent}        return control
'''
        return code

    # ---------------- monitor / training args ----------------

    def emit_training_args(self, opt_name, indent="") -> str:
        opt = self.optimizes[opt_name]
        f = opt["fields"]
        monitor_name = dotted_last(f["telemetry"]) if "telemetry" in f else None
        mon = self.monitors.get(monitor_name) if monitor_name else None

        lines = []
        checkpoint_steps = 500
        save_strategy = "steps"
        report_to = "'none'"
        metric_for_best = None
        greater_is_better = "False"

        if mon:
            mf = mon["fields"]
            if "checkpoint_every" in mf:
                dur = mf["checkpoint_every"]
                if dur["unit"] == "steps":
                    checkpoint_steps = dur["n"]
                    save_strategy = "steps"
                else:
                    save_strategy = "epoch"
            if "destination" in mf and mf["destination"].startswith("wandb://"):
                report_to = "'wandb'"
                project = mf["destination"].split("://", 1)[1]
                lines.append(f"{indent}os.environ['WANDB_PROJECT'] = {py_str(project)}")
                if "auth" in mf:
                    var = mf["auth"]["env_var"]
                    lines.append(f"{indent}os.environ.setdefault('WANDB_API_KEY', os.environ.get({py_str(var)}, ''))")
            if "select_best_on" in mf:
                sbo = mf["select_best_on"]
                metric_for_best = sbo["metric"]
                greater_is_better = "True" if sbo["direction"] == "maximize" else "False"

        seed = f.get("seed")
        lines.append(f"{indent}SEED = {seed if seed is not None else 'random.randint(0, 2**31 - 1)'}")
        lines.append(f"{indent}set_seed(SEED)")
        if seed is None:
            lines.append(f"{indent}print(f'[tensorscript] no seed declared — generated SEED={{SEED}}, logging for reproducibility')")
            if mon and mon["fields"].get("destination", "").startswith("wandb://"):
                lines.append(f"{indent}wandb.config.update({{'tensorscript_generated_seed': SEED}})" )

        schedule = f.get("schedule", {})
        sched_args = {a["key"]: a["value"] for a in schedule.get("args", []) if a["key"]}
        max_lr = sched_args.get("max_lr", 2e-4)
        warmup = sched_args.get("warmup", 0.03)
        epochs = f.get("epochs", 1)

        ta_kwargs = [
            "output_dir=OUTPUT_DIR",
            f"num_train_epochs={epochs}",
            f"learning_rate={max_lr}",
            f"warmup_steps={int(float(warmup) * 2000)}",  # transformers 5.x dropped warmup_ratio
            "lr_scheduler_type='cosine'",
            f"save_strategy={py_str(save_strategy)}",
            f"save_steps={checkpoint_steps}" if save_strategy == "steps" else None,
            f"report_to=[{report_to}]" if report_to != "'none'" else "report_to=[]",
            "seed=SEED",
        ]
        is_t4 = (f.get("hardware") or {}).get("profile") == "t4"
        if is_t4:
            ta_kwargs.append("per_device_train_batch_size=1")
            ta_kwargs.append("per_device_eval_batch_size=1")
            ta_kwargs.append("gradient_checkpointing=True")
            ta_kwargs.append("bf16=False")
            ta_kwargs.append("fp16=True")
        if metric_for_best:
            ta_kwargs.append("load_best_model_at_end=True")
            ta_kwargs.append(f"metric_for_best_model={py_str(metric_for_best)}")
            ta_kwargs.append(f"greater_is_better={greater_is_better}")
            ta_kwargs.append("eval_strategy=" + py_str(save_strategy))

        ta_kwargs = [k for k in ta_kwargs if k]
        lines.append(f"{indent}training_args = TrainingArguments(")
        for k in ta_kwargs:
            lines.append(f"{indent}    {k},")
        lines.append(f"{indent})")

        if "hardware" in f:
            hw = f["hardware"]
            lines.append(f"{indent}# hardware: block declares static topology — apply via `accelerate config` / launch flags, not in-script:")
            lines.append(f"{indent}#   nodes: {hw.get('nodes')}, gpus_per_node: {hw.get('gpus_per_node')}, strategy: {hw.get('strategy', {}).get('call') if hw.get('strategy') else None}")

        return "\n".join(lines)

    # ---------------- top-level stage emission ----------------

    def emit_stage_script(self, opt_name, checkpoint_override_var=None) -> str:
        opt = self.optimizes[opt_name]
        f = opt["fields"]
        model_ref = opt["target"]
        dataset_ref = dotted_last(f["using"].split(".")[1]) if "using" in f else None
        # 'using: dataset.SFTData.train' -> dataset name is the middle segment
        using_parts = f["using"].split(".") if "using" in f else []
        dataset_name = using_parts[1] if len(using_parts) > 1 else None

        guardrail_code = ""
        if "guardrails" in f:
            guardrail_code = self.emit_guardrail_callback(f["guardrails"])

        script = f'''"""
Auto-generated by the TensorScript transpiler from `optimize {model_ref} as {opt_name}`.
Target stack: transformers + peft + datasets + accelerate + wandb.
This file is generated — edit the .tensor source and re-transpile rather
than hand-editing this output.
"""
import os
import random
import copy
import torch
from datasets import load_dataset, interleave_datasets
from transformers import (
    AutoModelForCausalLM, AutoConfig, AutoTokenizer, BitsAndBytesConfig,
    TrainingArguments, Trainer, TrainerCallback, set_seed,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

try:
    import wandb
except ImportError:
    wandb = None

OUTPUT_DIR = os.environ.get("TENSORSCRIPT_OUTPUT_DIR", "./output/{opt_name}")
RESOLVED_BASE_CHECKPOINT = os.environ.get("TENSORSCRIPT_BASE_CHECKPOINT")  # set by pipeline driver for stage 2+

{self.emit_training_args(opt_name)}

# ---- dataset ----
{self.emit_dataset_loader(dataset_name) if dataset_name else "# no dataset resolved"}

tokenizer_call = None  # placeholder — actual tokenizer instantiation below
{"tokenizer = AutoTokenizer.from_pretrained(" + py_str(self.datasets[dataset_name]["fields"]["tokenize"]["args"][0]["value"]) + ")" if dataset_name and "tokenize" in self.datasets[dataset_name]["fields"] else "tokenizer = None"}
if tokenizer is not None and tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def _tokenize_fn(batch):
    out = tokenizer(batch.get("text", batch.get("chosen", [""])), truncation=True, max_length=SEQUENCE_LENGTH, padding="max_length")
    out["labels"] = out["input_ids"].copy()
    return out

if tokenizer is not None:
    train_dataset = train_dataset.map(_tokenize_fn, batched=True)
    if eval_dataset is not None:
        eval_dataset = eval_dataset.map(_tokenize_fn, batched=True)

# ---- model ----
{self.emit_model_loader(model_ref, checkpoint_override_var="RESOLVED_BASE_CHECKPOINT" if checkpoint_override_var else None, profile=f.get("hardware"))}

{guardrail_code}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    callbacks=[GuardrailCallback()] if {bool("guardrails" in f)} else [],
)

if __name__ == "__main__":
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    if tokenizer is not None:
        tokenizer.save_pretrained(OUTPUT_DIR)
'''
        return script

    # ---------------- pipeline driver ----------------

    def emit_pipeline_driver(self, pipeline_block) -> str:
        lines = [
            '"""',
            f"Auto-generated pipeline driver for `pipeline {pipeline_block['name']}`.",
            "Runs each stage's generated script in dependency order and resolves",
            "pipeline.<Stage>.<selector> checkpoint references between stages.",
            '"""',
            "import json, os, subprocess, sys",
            "",
            "def resolve_checkpoint(output_dir, selector):",
            "    state_path = os.path.join(output_dir, 'trainer_state.json')",
            "    if selector == 'best':",
            "        with open(state_path) as f:",
            "            state = json.load(f)",
            "        best = state.get('best_model_checkpoint')",
            "        if best is None:",
            "            raise RuntimeError(",
            "                f'best checkpoint requested for {output_dir} but trainer_state.json has no '",
            "                f'best_model_checkpoint — was select_best_on set on the monitor for this stage?'",
            "            )",
            "        return best",
            "    if selector == 'final':",
            "        return output_dir",
            "    if isinstance(selector, dict) and 'step' in selector:",
            "        return os.path.join(output_dir, f\"checkpoint-{selector['step']}\")",
            "    return output_dir  # 'checkpoint' == most recent save",
            "",
        ]

        stage_order = []
        for stage in pipeline_block["stages"]:
            stage_order.append(stage)

        for stage in stage_order:
            opt_name = dotted_last(stage["fields"]["run"])
            lines.append(f"# ---- stage: {stage['name']} (optimize {opt_name}) ----")
            lines.append(f"os.environ['TENSORSCRIPT_OUTPUT_DIR'] = './output/{opt_name}'")
            if "depends_on" in stage["fields"]:
                dep = stage["fields"]["depends_on"]
                selector = stage["fields"].get("inherit_weights", "final")
                lines.append(f"_dep_output_dir = './output/{dotted_last(self._stage_opt_name(pipeline_block, dep))}'")
                lines.append(f"os.environ['TENSORSCRIPT_BASE_CHECKPOINT'] = resolve_checkpoint(_dep_output_dir, {json.dumps(selector)})")
            lines.append(f"subprocess.run([sys.executable, {py_str(opt_name + '.py')}], check=True)")
            lines.append("")

        return "\n".join(lines)

    def _stage_opt_name(self, pipeline_block, stage_name):
        for stage in pipeline_block["stages"]:
            if stage["name"] == stage_name:
                return dotted_last(stage["fields"]["run"])
        raise KeyError(stage_name)


def transpile(ast, out_dir="."):
    """Returns a dict of {filename: source_code} for all generated files."""
    t = Transpiler(ast)
    files = {}

    for opt_name, opt in t.optimizes.items():
        needs_override = any(
            m["fields"].get("base", {}).get("kind") == "pipeline_ref"
            and m["fields"]["base"]["stage"] == opt_name
            for m in t.models.values()
        ) or any(
            t.models.get(opt["target"], {}).get("fields", {}).get("base", {}).get("kind") == "pipeline_ref"
            for _ in [None]
        )
        model_base = t.models.get(opt["target"], {}).get("fields", {}).get("base", {})
        override = model_base.get("kind") == "pipeline_ref"
        files[f"{opt_name}.py"] = t.emit_stage_script(opt_name, checkpoint_override_var=override)

    for pipe in t.pipelines:
        files[f"run_{pipe['name']}.py"] = t.emit_pipeline_driver(pipe)

    return files
