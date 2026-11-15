"""Portable, process-isolated latency CLI. Planning never launches GPU work."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from ..catalog import MODELS
from ..accuracy.execution import work_environment, run_child
from .runtime import PERCENTS, DATASETS, ENGINE


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('prepare','run','validate','summarize'))
    p.add_argument('--model', choices=MODELS, required=True)
    p.add_argument('--model-path', type=Path)
    p.add_argument('--data-dir', type=Path, required=True)
    p.add_argument('--artifact-root', type=Path, help='Optional existing latency sidecars; never silently substitutes accuracy artifacts')
    p.add_argument('--accuracy-work', type=Path, help='Optional completed public accuracy work directory for regeneration')
    p.add_argument('--method', choices=('dense','teal','roshift','all'), default='all')
    p.add_argument('--percents', nargs='+', type=int, choices=PERCENTS, default=list(PERCENTS))
    p.add_argument('--datasets', nargs='+', choices=DATASETS, default=list(DATASETS))
    p.add_argument('--repeats', type=int, default=4)
    p.add_argument('--stage', choices=('all','prompts','weights','centers','thresholds'), default='all')
    p.add_argument('--check', choices=('all','kernels','model'), default='all', help='Validation scope; kernel-only does not require a sidecar')
    p.add_argument('--output', type=Path)
    p.add_argument('--execute', action='store_true')
    return p


def make_request(args):
    if args.repeats < 1 or len(set(args.percents)) != len(args.percents) or len(set(args.datasets)) != len(args.datasets):
        raise ValueError('Positive repeats and distinct targets/datasets required')
    if args.action=='validate' and args.check=='kernels' and args.method!='roshift':
        raise ValueError('Kernel-only validation is for --method roshift')
    root = args.data_dir.resolve()
    record = {k:str(v.resolve()) if isinstance(v,Path) else v for k,v in vars(args).items() if k!='execute'}
    record['artifact_root'] = str(args.artifact_root.resolve() if args.artifact_root else root/'speedup/artifacts'/args.model)
    record['model_path'] = str(args.model_path.resolve() if args.model_path else root/'models'/args.model)
    record['output'] = str(args.output.resolve() if args.output else root/'speedup/results'/args.model)
    record['percents'] = sorted(args.percents)
    record['smoke_subset'] = args.repeats != 4 or tuple(args.datasets) != DATASETS
    return record


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    try:
        request = make_request(args)
        if not args.execute:
            print(json.dumps(dict(plan=request, execution_started=False), indent=2))
            return
        if os.name != 'posix':
            raise RuntimeError('CUDA execution requires Linux/WSL; Windows supports planning')
        writable=[request['output']]
        if request['action']=='prepare':
            writable.append(request['artifact_root'])
        package=Path(__file__).resolve().parents[1]
        for directory in map(Path,writable):
            if str(directory).startswith('/mnt/c/') or directory.is_relative_to(package):
                raise ValueError('Generated files must be outside C: and the installed source tree')
        root, env = work_environment(args.data_dir)
        env.update(MAX_JOBS='1', TORCHINDUCTOR_CACHE_DIR=str(root/'cache/inductor_speedup'))
        driver_cache=root/'cache/cuda_driver'
        driver_cache.mkdir(parents=True,exist_ok=True)
        env['CUDA_CACHE_PATH']=str(driver_cache)
        env.setdefault('CUDA_HOME','/usr/local/cuda')
        env['PATH']=os.pathsep.join([str(Path(sys.executable).parent),
                                    str(Path(env['CUDA_HOME'])/'bin'),env.get('PATH','')])
        # Do not hardcode the GPU architecture; PyTorch queries the recipient.
        env.pop('TORCH_CUDA_ARCH_LIST', None)
        stamp = __import__('datetime').datetime.now().strftime('%Y%m%dT%H%M%S%f')
        path = root/'requests'/f'speedup_{args.model}_{stamp}.json'
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(request,indent=2)+'\n')
        env['ROSHIFT_SPEED_REQUEST'] = str(path)
        env['PYTHONPATH'] = os.pathsep.join([str(Path(__file__).resolve().parents[2]), env['PYTHONPATH']])
        run_child([sys.executable,'-u','-m','roshift.speedup.worker',str(path)],env)
    except (ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        p.exit(2, f'error: {error}\n')


if __name__ == '__main__':
    main()
