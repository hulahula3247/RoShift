"""Legacy plan-only dependency preview; use the accuracy CLI for execution."""

from .catalog import METHODS, asset, model_config, recipe
from .io import fingerprint


def make_plan(target, model, method, sparsity, allocation="uniform", tasks=None):
    if target not in ("accuracy", "speedup") or method not in METHODS:
        raise ValueError("Unknown target or method")
    config = model_config(model)
    protocol = asset("protocols", target + ".json")
    if method == "dense":
        if sparsity != 0:
            raise ValueError("Dense is measured only at sparsity 0")
    elif sparsity not in protocol["sparsities"]:
        raise ValueError("Sparsity is not part of this reproduction cohort")
    if target == "speedup" and (method not in ("dense", "teal", "roshift") or allocation != "uniform"):
        raise ValueError("Frozen speed measurements cover Dense/TEAL/RoShift with uniform allocation only")
    if target == "accuracy":
        available = asset("benchmarks.json")
        chosen = list(tasks) if tasks else list(available)
        if any(t not in available for t in chosen) or len(chosen) != len(set(chosen)):
            raise ValueError("Unknown or duplicate benchmark")
    else:
        if tasks:
            raise ValueError("Speedup uses the frozen timing prompts, not accuracy tasks")
        chosen = []
    stages = [{"id": "download", "depends_on": [], "status": "implemented_not_network_tested"},
              {"id": "data", "depends_on": ["download"], "status": "fixed_inputs_catalogued"}]
    previous = "data"
    if method != "dense":
        for name in ("statistics", "allocation", "rotation", "sidecar", "validation") if method == "roshift" else ("official_prepare", "validation"):
            stages.append({"id": name, "depends_on": [previous], "status": "declared_not_executed"})
            previous = name
    stages.append({"id": target, "depends_on": [previous], "status": "declared_not_executed"})
    frozen = recipe(model, allocation, sparsity) if target == "accuracy" and method == "roshift" else None
    payload = {"schema": 1, "target": target, "model": model, "method": method, "sparsity": sparsity,
               "allocation": allocation, "tasks": chosen, "model_identity": config,
               "protocol": protocol, "recipe": frozen, "stages": stages,
               "end_to_end_ready": False,
               "missing": (["use the accuracy command to execute; see docs/VALIDATION.md for tested coverage"]
                           if target=="accuracy" else ["use roshift speedup; see docs/SPEEDUP.md for native CUDA execution"])}
    payload["fingerprint"] = fingerprint(payload)
    return payload


def execute_plan(plan):
    raise RuntimeError("This legacy command only previews a plan: " + "; ".join(plan["missing"]))
