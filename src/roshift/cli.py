"""Small entry point with explicit readiness rather than silent placeholders."""

import argparse
import json
import os
from pathlib import Path

from .catalog import METHODS, MODELS, asset, validate_catalog
from .data import download_model, download_plan, verify_model
from .plan import execute_plan, make_plan


def main(argv=None):
    import sys
    values = sys.argv[1:] if argv is None else argv
    if values and values[0] == 'speedup':
        from .speedup.execution import main as speedup_main
        return speedup_main(values[1:])
    if values and values[0]=='score-mbpp':
        from .accuracy.scoring import main as score_main
        return score_main(values[1:])
    if values and values[0] == 'accuracy':
        from .accuracy.execution import main as accuracy_main
        return accuracy_main(values[1:])
    parser = argparse.ArgumentParser(description="RoShift: accuracy evaluation and native CUDA decoding. Use 'roshift accuracy --help' or 'roshift speedup --help' to get started.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("accuracy", help="Prepare, search and evaluate accuracy experiments")
    sub.add_parser("speedup", help="Prepare, validate and measure native CUDA decoding")
    sub.add_parser("catalog", help="List supported models, methods, and benchmark settings")
    sub.add_parser("check", help="Validate all packaged accuracy recipes without importing torch")
    dl = sub.add_parser("download", help="Plan checkpoint download; use --execute to download and verify")
    dl.add_argument("--model", choices=MODELS, required=True)
    dl.add_argument("--data-dir", type=Path, default=Path(os.environ.get("ROSHIFT_DATA", "artifacts")))
    dl.add_argument("--execute", action="store_true")
    vm = sub.add_parser("verify-model", help="Hash a locally downloaded checkpoint")
    vm.add_argument("--model", choices=MODELS, required=True)
    vm.add_argument("--directory", type=Path, required=True)
    rp = sub.add_parser("reproduce", help="Legacy plan preview; execution uses 'roshift accuracy'")
    rp.add_argument("target", choices=("accuracy", "speedup"))
    rp.add_argument("--model", choices=MODELS, required=True)
    rp.add_argument("--method", choices=METHODS, default="roshift")
    rp.add_argument("--sparsity", type=int, required=True)
    rp.add_argument("--allocation", choices=("uniform", "nonuniform"), default="uniform")
    rp.add_argument("--tasks", nargs="+")
    rp.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "catalog":
            result = {"models": MODELS, "methods": METHODS, "benchmarks": asset("benchmarks.json"),
                      "status": "Accuracy and CUDA speedup; see docs/VALIDATION.md for tested coverage"}
        elif args.command == "check":
            result = validate_catalog()
        elif args.command == "download":
            if args.execute:
                from .accuracy.execution import work_environment
                _,env=work_environment(args.data_dir)
                os.environ.update(env)
            result = (download_model if args.execute else download_plan)(args.model, args.data_dir)
        elif args.command == "verify-model":
            result = verify_model(args.model, args.directory)
        else:
            result = make_plan(args.target, args.model, args.method, args.sparsity, args.allocation, args.tasks)
            if args.execute:
                execute_plan(result)
        print(json.dumps(result, indent=2, allow_nan=False))
        if args.command == "verify-model" and not result["ok"]:
            raise SystemExit(1)
    except (ValueError, RuntimeError, FileNotFoundError) as error:
        parser.exit(2, f"error: {error}\n")
