"""
Semantic checks for TensorScript v1.0, implementing the rules in
tensorscript_v1_spec.md §6 that are cross-block and can't be enforced by
the grammar alone.

Checks implemented:
  1. select_best_on's metric must appear in that monitor's track: list.
  2. If any model.base references pipeline.<Stage>.best, the optimize
     block for <Stage> must have telemetry pointed at a monitor block
     that declares select_best_on. Missing it is a compile-time error,
     not a silent fallback to 'final'.
  3. (bonus, not yet in spec but a natural extension) depends_on targets
     in a pipeline must name a stage declared earlier in the same pipeline.
"""


class SemanticError(Exception):
    pass


def check(ast):
    errors = []

    monitors = {b["name"]: b for b in ast["blocks"] if b["type"] == "monitor"}
    optimizes = {b.get("alias") or b["target"]: b for b in ast["blocks"] if b["type"] == "optimize"}
    models = {b["name"]: b for b in ast["blocks"] if b["type"] == "model"}
    pipelines = [b for b in ast["blocks"] if b["type"] == "pipeline"]

    # Rule 1: select_best_on metric must be tracked
    for name, mon in monitors.items():
        sbo = mon["fields"].get("select_best_on")
        if sbo:
            track = mon["fields"].get("track", [])
            if sbo["metric"] not in track:
                errors.append(
                    f"monitor {name!r}: select_best_on metric {sbo['metric']!r} "
                    f"is not in track: {track}"
                )

    # Rule 2: model.base referencing pipeline.<Stage>.best requires that
    # stage's monitor to declare select_best_on.
    for mname, model in models.items():
        base = model["fields"].get("base")
        if base and base.get("kind") == "pipeline_ref" and base["selector"] == "best":
            stage_name = base["stage"]
            opt = optimizes.get(stage_name)
            if opt is None:
                errors.append(
                    f"model {mname!r}: base references pipeline.{stage_name}.best, "
                    f"but no optimize block named or aliased {stage_name!r} exists"
                )
                continue
            telemetry_ref = opt["fields"].get("telemetry")
            if not telemetry_ref:
                errors.append(
                    f"model {mname!r}: base references pipeline.{stage_name}.best, "
                    f"but optimize {stage_name!r} has no telemetry: monitor attached"
                )
                continue
            monitor_name = telemetry_ref.split(".")[-1]
            mon = monitors.get(monitor_name)
            if mon is None:
                errors.append(
                    f"model {mname!r}: telemetry reference {telemetry_ref!r} on "
                    f"optimize {stage_name!r} does not resolve to a known monitor block"
                )
                continue
            if "select_best_on" not in mon["fields"]:
                errors.append(
                    f"model {mname!r}: base references pipeline.{stage_name}.best, "
                    f"but monitor {monitor_name!r} (used by that stage) has no "
                    f"select_best_on — 'best' is undefined without it"
                )

    # Rule 3 (extension): depends_on must name an earlier stage in the same pipeline
    for pipe in pipelines:
        seen = set()
        for stage in pipe["stages"]:
            dep = stage["fields"].get("depends_on")
            if dep is not None and dep not in seen:
                errors.append(
                    f"pipeline {pipe['name']!r}: stage {stage['name']!r} "
                    f"depends_on {dep!r}, which is not an earlier stage in this pipeline"
                )
            seen.add(stage["name"])

    return errors
