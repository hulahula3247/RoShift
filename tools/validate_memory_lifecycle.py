"""Check release of a cyclic CUDA tensor before the next model-load stage."""
import argparse
import os
from pathlib import Path
import weakref


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,required=True)
    args=p.parse_args()
    from roshift.accuracy.execution import work_environment
    root,env=work_environment(args.data_dir)
    os.environ.update(env)
    import torch
    from roshift.accuracy.evaluate import release_cuda
    from roshift.io import atomic_json
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        raise RuntimeError('Select one CUDA GPU')
    release_cuda()
    before=torch.cuda.memory_allocated()
    class CyclicOwner:
        pass
    owner=CyclicOwner()
    owner.tensor=torch.ones(32*1024*1024,device='cuda:0',dtype=torch.float16)
    owner.self=owner
    reference=weakref.ref(owner.tensor)
    peak=torch.cuda.memory_allocated()
    del owner
    release_cuda()
    after=torch.cuda.memory_allocated()
    passed=reference() is None and peak>before and after<=before
    result=dict(passed=passed,before_bytes=before,allocated_bytes=peak,after_bytes=after,
                scope='64 MiB synthetic cyclic tensor, not a full-model memory benchmark')
    atomic_json(root/'memory_lifecycle.json',result)
    print(result,flush=True)
    if not passed:
        raise RuntimeError('CUDA tensor remained alive after stage cleanup')


if __name__=='__main__':
    main()
