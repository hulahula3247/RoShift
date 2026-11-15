"""Reuse the audited worker's measurement loop, but never reload a resident model."""
import argparse
import gc
import json
import os
import shutil
from pathlib import Path
import sys
import time
import traceback
from types import SimpleNamespace
from support import DATASETS, MODELS, PERCENTS, ResidentModel, dump, open_runtime, preflight, sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--method", choices=("ours", "teal", "dense"), required=True)
    parser.add_argument("--percents", nargs="+", type=int, choices=PERCENTS, default=list(PERCENTS))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--timing-only", action="store_true", help="Explicitly omit separate diagnostic replays")
    args = parser.parse_args()
    if args.repeats < 1 or args.percents != sorted(set(args.percents)):
        parser.error("Use positive repeats and unique ascending sparsities")
    common, source, root = open_runtime(args.model, args.artifact_root)
    datasets = tuple(common.bench.DATASETS)
    problems = preflight(common, root, args.method, args.percents)
    report = dict(model=args.model, method=args.method, problems=problems,
                  status="not_ready" if problems else "ready", source=str(source), artifact_root=str(root))
    dump(args.output / args.method / "preflight.json", report)
    print(json.dumps(report, indent=2), flush=True)
    if args.preflight_only:
        return
    if problems:
        raise RuntimeError("Missing or incompatible calibration; no GPU model loaded")

    import torch
    import worker
    worker.OUTPUT = args.output / args.method
    reference = args.output / "dense/dense_prefill_logits.pt"
    if reference.exists() and args.method != "dense":
        worker.OUTPUT.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(reference, worker.OUTPUT / "dense_prefill_logits.pt")
    resident = ResidentModel(common, source, args.method)
    worker.load_model = resident
    worker.config_for = common.config_for if args.method == "ours" else lambda _: SimpleNamespace(manifest=Path("unused"))
    worker.threshold_path = common.threshold_path
    # Preserve the old worker's hash and validation of the full original runtime.
    protocol = dict(id="method_once_v1", model=args.model, method=args.method,
                    device=torch.cuda.get_device_name(),
                    capability=list(torch.cuda.get_device_capability()),
                    torch=torch.__version__, cuda=torch.version.cuda,
                    percents=[0] if args.method == "dense" else args.percents,
                    repeats=args.repeats, diagnostics=not args.timing_only,
                    teal_autotune="reselect each target; compiled binaries remain cached",
                    order="ascending sparsity; all four datasets per sparsity",
                    full_model_loads_per_process=1, kv_cache=1160,
                    historical_worker=str(source / "worker.py"), worker_sha256=sha(source / "worker.py"),
                    source_sha256={p.name:sha(p) for p in source.glob("*") if p.suffix in (".py", ".cu", ".json")},
                    launcher_sha256={p.name:sha(p) for p in Path(__file__).parent.glob("*.py")},
                    note="Not independent A/B runs. Do not pool with historical results.")
    protocol["prompt_sha256"] = sha(common.PROMPTS / "prompts.pt")
    protocol["dense_prefill_reference_sha256"] = sha(reference) if reference.exists() else None
    protocol["runtime_sha256"] = {str(p.relative_to(common.RUNTIME)):sha(p)
        for p in common.RUNTIME.rglob("*") if p.suffix in (".py", ".cu", ".h", ".cpp")}
    if args.method == "ours":
        config = common.config_for(.5)
        protocol["lm_head"] = dict(first_dim=config.lm_head_prefix_dim, middle_dim=config.lm_head_middle_dim,
                                   z1=config.lm_head_scaled_margin, z2=config.lm_head_second_scaled_margin)
    protocol["calibration_sha256"] = {}
    for percent in (() if args.method == "dense" else args.percents):
        files = ([common.config_for(percent/100).manifest, root / f"centers_s{percent}.json"]
                 if args.method == "ours" else [common.threshold_path(d, percent/100) for d in datasets])
        protocol["calibration_sha256"][str(percent)] = {str(p):sha(p) for p in files}
    state_path = args.output / args.method / "state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state["protocol"] != protocol:
            raise ValueError("Output belongs to a different protocol/source; use a new output directory")
    else:
        state = dict(protocol=protocol, completed=[], attempts=[])
    started = time.perf_counter()
    state["attempts"].append(dict(started=time.strftime("%Y-%m-%d %H:%M:%S"), pid=os.getpid()))
    old_argv = sys.argv
    try:
        for percent in protocol["percents"]:
            if percent in state["completed"]:
                previous = json.loads((worker.OUTPUT / f"s{percent}" / "result.json").read_text())
                if previous.get("status") != "complete" or previous.get("method_once", {}).get("protocol") != protocol:
                    raise ValueError("Completed checkpoint/result mismatch")
                continue
            tag = f"s{percent}"
            path = worker.OUTPUT / tag / "result.json"
            if path.exists():
                stamp = time.time_ns()
                path.replace(path.with_name(f"interrupted_{stamp}.json"))
            state.update(status="running", current=percent)
            dump(state_path, state)
            sys.argv = [str(source / "worker.py"), "--method", args.method,
                        "--sparsity", str(percent/100), "--tag", tag, "--repeats", str(args.repeats)]
            if not args.timing_only:
                sys.argv.append("--diagnostics")
            import kernels.sparse_gemv as sg
            originals = (sg.splitk_sparse_gemv, sg.qkv_gemv)
            if args.method == "teal":
                # Threshold is not in the old autotuner key. Do not let the
                # first sparsity silently dictate geometry for every later one.
                for name in ("qkv_kernel", "splitk_sparse_gemv_kernel"):
                    getattr(sg, name).cache.clear()
            try:
                worker.main()
            finally:
                # A captured diagnostic graph must never survive a runtime swap.
                torch.cuda.synchronize()
                torch._dynamo.reset()
                resident.restore_diagnostics()
                sg.splitk_sparse_gemv, sg.qkv_gemv = originals
                gc.collect()
                torch.cuda.empty_cache()
            payload = json.loads(path.read_text())
            if payload["status"] != "complete" or len(payload["rows"]) != len(datasets) * args.repeats:
                raise RuntimeError("Incomplete measurement")
            assert all(not v["invalid_steps"] for v in payload["compiled_validity"].values())
            if args.method == "ours":
                config = common.config_for(percent/100)
                first, middle = config.lm_head_prefix_dim, config.lm_head_middle_dim
                vocab = json.loads((common.MODEL / "config.json").read_text())["vocab_size"]
                for diag in payload.get("diagnostics", {}).values():
                    counts = diag["head_counts"]
                    diag["head_survival_stage1"] = sum(c[0]/vocab for c in counts)/len(counts)
                    diag["head_survival_stage2"] = sum(c[1]/vocab for c in counts)/len(counts)
                    diag["head_effective_sparsity"] = sum(
                        1-(first+c[0]/vocab*middle+c[1]/vocab*(common.DIM-first-middle))/common.DIM
                        for c in counts)/len(counts)
            payload["method_once"] = dict(protocol=protocol, load_events=list(resident.switches),
                                          full_model_load_count=resident.full_load_count)
            dump(path, payload)
            state["completed"].append(percent)
            state.update(load_events=list(resident.switches), seconds_this_process=time.perf_counter()-started)
            dump(state_path, state)
        state.update(status="complete", current=None)
    except BaseException:
        state.update(status="interrupted_or_failed", traceback=traceback.format_exc())
        raise
    finally:
        state.update(full_model_load_count=resident.full_load_count,
                     seconds_this_process=time.perf_counter()-started)
        dump(state_path, state)
        sys.argv = old_argv


if __name__ == "__main__":
    main()
