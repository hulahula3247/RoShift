#!/usr/bin/env python3
"""Run one unchanged search entrypoint with explicitly scoped runtime adapters."""
import argparse
import importlib
import json
import hashlib
import os
import sys
import time
from importlib.metadata import version
import torch
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--module', required=True)
    parser.add_argument('--shared_median_dir', required=True)
    parser.add_argument('--model_type', required=True)
    args, rest = parser.parse_known_args()
    module = importlib.import_module(args.module)
    from rental_runtime_20260916 import install
    install(args.shared_median_dir, args.model_type)
    output = Path(rest[rest.index('--output_dir')+1])
    source_root = Path(__file__).resolve().parent
    sources = {}
    for loaded in list(sys.modules.values()):
        path = vars(loaded).get('__file__') if loaded is not None else None
        if path:
            path = Path(path).resolve()
            if path.is_relative_to(source_root) and path.suffix == '.py' and path.exists():
                sources[str(path.relative_to(source_root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    provenance = output/'provenance'/f'run_{time.time_ns()}_{os.getpid()}.json'
    provenance.parent.mkdir(parents=True, exist_ok=True)
    provenance.write_text(json.dumps(dict(module=args.module,argv=rest,
        shared_median_dir=args.shared_median_dir,model_type=args.model_type,
        source_sha256=sources,versions={name:version(name) for name in ('torch','transformers','optuna')},
        tf32=False,threads=4,pid=os.getpid()),indent=2)+'\n')
    sys.argv = [args.module, *rest]
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    with torch.inference_mode():
        module.main()


if __name__ == '__main__':
    main()
