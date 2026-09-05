"""
Auto-generated pipeline driver for `pipeline AlignmentRun`.
Runs each stage's generated script in dependency order and resolves
pipeline.<Stage>.<selector> checkpoint references between stages.
"""
import json, os, subprocess, sys

def resolve_checkpoint(output_dir, selector):
    state_path = os.path.join(output_dir, 'trainer_state.json')
    if selector == 'best':
        with open(state_path) as f:
            state = json.load(f)
        best = state.get('best_model_checkpoint')
        if best is None:
            raise RuntimeError(
                f'best checkpoint requested for {output_dir} but trainer_state.json has no '
                f'best_model_checkpoint — was select_best_on set on the monitor for this stage?'
            )
        return best
    if selector == 'final':
        return output_dir
    if isinstance(selector, dict) and 'step' in selector:
        return os.path.join(output_dir, f"checkpoint-{selector['step']}")
    return output_dir  # 'checkpoint' == most recent save

# ---- stage: one (optimize SFTStage) ----
os.environ['TENSORSCRIPT_OUTPUT_DIR'] = './output/SFTStage'
subprocess.run([sys.executable, "SFTStage.py"], check=True)

# ---- stage: two (optimize DPOStage) ----
os.environ['TENSORSCRIPT_OUTPUT_DIR'] = './output/DPOStage'
_dep_output_dir = './output/SFTStage'
os.environ['TENSORSCRIPT_BASE_CHECKPOINT'] = resolve_checkpoint(_dep_output_dir, "best")
subprocess.run([sys.executable, "DPOStage.py"], check=True)
