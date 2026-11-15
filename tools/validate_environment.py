"""Write machine-readable CPU-only checks, including explicitly skipped tests."""

import ast
import importlib.metadata
import io
import json
from pathlib import Path
import platform
import unittest

from roshift.io import atomic_json


def main():
    root = Path(__file__).resolve().parents[1]
    count = 0
    for path in root.rglob("*.py"):
        if any(part in ("build", "dist", "__pycache__") or part.endswith(".egg-info") for part in path.relative_to(root).parts):
            continue
        ast.parse(path.read_text(encoding="utf-8-sig"))
        count += 1
    out = io.StringIO()
    tests = unittest.defaultTestLoader.discover(str(root / "tests"))
    result = unittest.TextTestRunner(stream=out, verbosity=2).run(tests)
    print(out.getvalue())
    summary = {"cpu_tests_run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
               "skipped": [{"test": str(t), "reason": reason} for t, reason in result.skipped],
               "syntax_files": count, "gpu_tests_run": False, "model_download_tested": False,
               "fresh_machine_end_to_end": False, "python": platform.python_version()}
    atomic_json(root / "validation/cpu_checks.json", summary)
    versions = {}
    for name in ("torch", "transformers", "triton", "numpy", "safetensors", "huggingface-hub", "lm-eval", "datasets"):
        try: versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: versions[name] = None
    atomic_json(root / "environments/local-observed.json", {"python": platform.python_version(), "packages": versions})
    print(json.dumps(summary, indent=2))
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
