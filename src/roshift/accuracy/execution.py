"""Disk-bounded, process-isolated accuracy entry points."""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys

from ..catalog import MODELS, METHODS, asset

ENGINE = Path(__file__).resolve().parent / '_engine'


def run_child(command,env):
    child=subprocess.Popen(command,env=env,start_new_session=True)
    try:
        code=child.wait()
    except BaseException:
        if child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGKILL)
                child.wait()
        raise
    if code:
        raise subprocess.CalledProcessError(code,command)


def work_environment(data_dir):
    root = Path(data_dir).expanduser().resolve()
    if root == ENGINE or root.is_relative_to(ENGINE):
        raise ValueError('Data must be outside the installed source tree')
    if (os.name == 'nt' and root.drive.upper() == 'C:') or str(root).startswith('/mnt/c/'):
        raise ValueError('This release refuses new experiment/cache data on C:. Choose D: or another data volume.')
    root.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(root).free < 10 * 2**30:
        raise RuntimeError('Less than 10 GiB free on the data volume; no work started')
    env = os.environ.copy()
    # Loading-only reservation is optional and can OOM before useful GPU work.
    env.setdefault('ROSHIFT_SKIP_ALLOCATOR_WARMUP', '1')
    old_home=Path(env['HF_HOME']) if env.get('HF_HOME') else (
        Path(env['XDG_CACHE_HOME'])/'huggingface' if env.get('XDG_CACHE_HOME') else Path.home()/'.cache/huggingface')
    old_token=Path(env.get('HF_TOKEN_PATH',str(old_home/'token')))
    if old_token.is_file():
        # Reuse the login without copying its secret into the data directory.
        env.setdefault('HF_TOKEN_PATH',str(old_token.resolve()))
    dirs = {'HF_HOME':'cache/huggingface', 'HF_DATASETS_CACHE':'cache/datasets',
            'HUGGINGFACE_HUB_CACHE':'cache/huggingface/hub', 'TMPDIR':'tmp',
            'TEMP':'tmp', 'TMP':'tmp', 'XDG_CACHE_HOME':'cache/xdg',
            'TORCH_HOME':'cache/torch', 'TORCH_EXTENSIONS_DIR':'cache/extensions',
            'TRITON_CACHE_DIR':'cache/triton', 'MPLCONFIGDIR':'cache/matplotlib',
            'PIP_CACHE_DIR':'cache/pip', 'NUMBA_CACHE_DIR':'cache/numba'}
    for key, relative in dirs.items():
        path = root / relative
        path.mkdir(parents=True, exist_ok=True)
        env[key] = str(path)
    env.update(PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
               TOKENIZERS_PARALLELISM='false', MPLBACKEND='Agg',
               HF_DEACTIVATE_ASYNC_LOAD='1', CUBLAS_WORKSPACE_CONFIG=':4096:8',
               ROSHIFT_ENGINE_ROOT=str(ENGINE), ROSHIFT_DATA=str(root),
               ROSHIFT_INPUT_ROOT=str(root/'external_data/inputs'))
    entries = [str(ENGINE/'code/vendor'), str(ENGINE/'code/analysis'),
               str(ENGINE/'code/tools'), str(ENGINE/'ops'), str(Path(__file__).resolve().parents[2])]
    env['PYTHONPATH'] = os.pathsep.join(entries)
    return root, env


def parser():
    p = argparse.ArgumentParser(description='RoShift accuracy preparation, optional search, and evaluation')
    p.add_argument('action', choices=('prepare', 'search', 'ppl', 'evaluate', 'run', 'audit'))
    p.add_argument('--model', choices=MODELS, required=True)
    p.add_argument('--model-path', type=Path, help='Reuse a checkpoint without copying; otherwise download to the data volume')
    p.add_argument('--data-dir', type=Path, default=os.environ.get('ROSHIFT_DATA'), required=not bool(os.environ.get('ROSHIFT_DATA')))
    p.add_argument('--method', choices=METHODS, default='roshift')
    p.add_argument('--allocation', choices=('uniform','nonuniform'), default='nonuniform')
    p.add_argument('--sparsity', type=int, choices=(0,40,50,60), default=50)
    p.add_argument('--search-stage', choices=('allocation','q','all'), default='all')
    p.add_argument('--use-search', action='store_true', help='Evaluate independently re-searched profiles instead of frozen paper recipes')
    p.add_argument('--tasks', nargs='+', choices=list(asset('benchmarks.json')))
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--limit', type=int, help='Explicit smoke subset, never reported as full benchmark')
    p.add_argument('--execute', action='store_true')
    return p


def make_request(args):
    if args.method not in METHODS:
        raise ValueError('Unsupported method in this distribution: '+args.method)
    if args.method == 'dense' and args.sparsity != 0:
        raise ValueError('Dense requires --sparsity 0')
    if args.method != 'dense' and args.sparsity == 0:
        raise ValueError('Sparse methods require 40, 50 or 60')
    if args.batch_size <= 0 or (args.limit is not None and args.limit <= 0):
        raise ValueError('Batch size and smoke limit must be positive')
    if args.tasks and len(args.tasks) != len(set(args.tasks)):
        raise ValueError('Duplicate benchmark names')
    if args.action == 'search' and args.method != 'roshift':
        raise ValueError('RoShift allocation/Q search never applies to official baselines')
    if args.use_search and (args.method!='roshift' or args.action not in ('evaluate','ppl','audit')):
        raise ValueError('--use-search is only for evaluating or auditing a completed RoShift re-search')
    if args.action=='ppl' and args.method!='roshift':
        raise ValueError('ppl is the full RoShift profile audit; use evaluate for baseline benchmarks')
    return {key: str(value.resolve()) if isinstance(value, Path) else value
            for key,value in vars(args).items() if key != 'execute'}


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    try:
        request = make_request(args)
        if not args.execute:
            print(json.dumps({'plan':request,'execution_started':False},indent=2))
            return
        if os.name != 'posix':
            raise RuntimeError('Execute from Linux/WSL; Windows supports planning only')
        root, env = work_environment(args.data_dir)
        stamp = __import__('datetime').datetime.now().strftime('%Y%m%dT%H%M%S%f')
        request_file = root/'requests'/f'{args.model}_{stamp}.json'
        request_file.parent.mkdir(exist_ok=True)
        request_file.write_text(json.dumps(request,indent=2)+'\n')
        run_child([sys.executable,'-m','roshift.accuracy.worker',str(request_file)],env)
    except (ValueError,RuntimeError,subprocess.CalledProcessError) as error:
        p.exit(2,f'error: {error}\n')


if __name__ == '__main__':
    main()
