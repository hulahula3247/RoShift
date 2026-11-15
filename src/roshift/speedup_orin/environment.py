"""Private child-process environment for the Orin speed backend."""
import os
from pathlib import Path
import shutil
import signal
import subprocess


def run_child(command, env):
    child = subprocess.Popen(command, env=env, start_new_session=True)
    try:
        code = child.wait()
    except BaseException:
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        raise
    if code:
        raise subprocess.CalledProcessError(code, command)


def work_environment(data_dir):
    root = Path(data_dir).expanduser().resolve()
    package = Path(__file__).resolve().parents[1]
    if root.is_relative_to(package) or str(root).startswith('/mnt/c/'):
        raise ValueError('Choose a data volume outside the installed source tree and C:')
    root.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(root).free < 10 * 2**30:
        raise RuntimeError('Less than 10 GiB free on the data volume; no work started')
    env = os.environ.copy()
    for key in ('ROSHIFT_ENGINE_ROOT', 'ROSHIFT_INPUT_ROOT', 'ROSHIFT_SPEED_REQUEST'):
        env.pop(key, None)
    env.setdefault('ROSHIFT_SKIP_ALLOCATOR_WARMUP', '1')
    old_home = Path(env.get('HF_HOME', str(Path(env.get('XDG_CACHE_HOME', str(Path.home()/'.cache')))/'huggingface')))
    old_token = Path(env.get('HF_TOKEN_PATH', str(old_home/'token')))
    if old_token.is_file():
        env.setdefault('HF_TOKEN_PATH', str(old_token.resolve()))
    cache = root/'cache/speedup_orin'
    directories = {
        'HF_HOME': 'huggingface', 'HF_DATASETS_CACHE': 'datasets',
        'HUGGINGFACE_HUB_CACHE': 'huggingface/hub', 'XDG_CACHE_HOME': 'xdg',
        'TORCH_HOME': 'torch', 'TORCH_EXTENSIONS_DIR': 'extensions',
        'TORCHINDUCTOR_CACHE_DIR': 'inductor', 'TRITON_CACHE_DIR': 'triton',
        'CUDA_CACHE_PATH': 'cuda_driver', 'MPLCONFIGDIR': 'matplotlib',
        'PIP_CACHE_DIR': 'pip', 'NUMBA_CACHE_DIR': 'numba',
    }
    for key, relative in directories.items():
        path = cache/relative
        path.mkdir(parents=True, exist_ok=True)
        env[key] = str(path)
    temporary = root/'tmp/speedup_orin'
    temporary.mkdir(parents=True, exist_ok=True)
    env.update(TMPDIR=str(temporary), TEMP=str(temporary), TMP=str(temporary),
               PYTHONPATH=str(package.parent), PYTHONDONTWRITEBYTECODE='1',
               PYTHONNOUSERSITE='1', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
               OPENBLAS_NUM_THREADS='4', TOKENIZERS_PARALLELISM='false',
               MPLBACKEND='Agg', HF_DEACTIVATE_ASYNC_LOAD='1',
               CUBLAS_WORKSPACE_CONFIG=':4096:8', ROSHIFT_DATA=str(root),
               MAX_JOBS='1', TORCHINDUCTOR_COMPILE_THREADS='1', CMAKE_BUILD_PARALLEL_LEVEL='1')
    return root, env
