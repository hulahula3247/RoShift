"""Time the unchanged core model-common beta0 Q_head preparation separately."""
import os as _roshift_os
import argparse
import fcntl
from pathlib import Path
import sys
from types import SimpleNamespace
ROOT = Path(_roshift_os.path.join(_roshift_os.environ['ROSHIFT_ENGINE_ROOT'], ''))
CODE = ROOT / 'code'
sys.path.insert(0, str(ROOT / 'ops'))
sys.path.insert(0, str(CODE / 'tools'))
from roshift_head_v2.worker import read, configure_environment, verify_sources

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=read(CODE / 'models.json'), required=True)
    parser.add_argument('--work', type=Path)
    parser.add_argument('--model-path', type=Path)
    parser.add_argument('--expected-source-identity', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    args.work = args.work or ROOT / 'work' / args.model
    args.model_path = args.model_path or ROOT / 'models' / read(CODE / 'models.json')[args.model]['directory']
    verify_sources(args.expected_source_identity)
    if not args.execute:
        print('Plan: unchanged core materialize_profile.head on one GPU; shared Q_head for40/50/60')
        return
    configure_environment()
    import torch
    from experiment_timing import Attempt, job, reused
    from rental_conditions_20260917 import require_single_gpu
    from prepare_remote import verify_model
    from materialize_profile import head, verify_export
    require_single_gpu(torch)
    verify_model(args.model, args.model_path)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    folder = args.work / args.model / 'common/lmhead'
    folder.mkdir(parents=True, exist_ok=True)
    lock = (folder / 'worker.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    log = folder / 'experiment_timing.jsonl'
    metadata = dict(model=args.model, method='RoShift', targets=[40, 50, 60], target=None, attribution='model_common_Q_head', gpu_reserved=True)
    with job(log, **metadata, category='job', substage='Q_head', label=args.model + ':Q_head'):
        marker = folder / 'q_head_meta.json'
        if marker.exists() and read(marker).get('status') == 'complete':
            verify_export(folder, read(marker)['files'])
            reused(log, model=args.model, method='RoShift', targets=[40, 50, 60], category='preparation', substage='Q_head', label=args.model + ':Q_head')
            return
        with torch.inference_mode(), Attempt(log, **metadata, category='preparation', substage='Q_head', label=args.model + ':Q_head'):
            head(SimpleNamespace(model=args.model, work=args.work, model_path=args.model_path))
if __name__ == '__main__':
    main()
