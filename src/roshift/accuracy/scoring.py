"""Run MBPP scoring in a credential-free Linux chroot/seccomp process."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid
from .execution import ENGINE,work_environment,run_child


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path)
    p.add_argument('--data-dir',type=Path,required=True)
    p.add_argument('--validate-only',action='store_true')
    p.add_argument('--sandbox-root',type=Path,help='Root-owned POSIX filesystem directory; NTFS without metadata is not sufficient')
    args=p.parse_args(argv)
    if os.name!='posix' or os.geteuid()!=0:
        p.error('Use a separate Linux root process: chroot/seccomp is mandatory')
    if not args.validate_only and not args.input:
        p.error('--input is required')
    root,cache=work_environment(args.data_dir)
    env={k:cache[k] for k in ('PYTHONPATH','ROSHIFT_ENGINE_ROOT','ROSHIFT_DATA','TMPDIR','PYTHONDONTWRITEBYTECODE')}
    env.update(PATH='/usr/bin:/bin',LANG='C.UTF-8',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    (root/'requests').mkdir(exist_ok=True)
    request=root/'requests'/f'scoring_{uuid.uuid4().hex}.json'
    request.write_text(json.dumps(dict(input=str(args.input.resolve()) if args.input else None,
                                     validate_only=args.validate_only,data=str(root),
                                     sandbox=str(args.sandbox_root.resolve()) if args.sandbox_root else str(root/'mbpp_sandbox'))))
    run_child([sys.executable,'-m','roshift.accuracy.scoring','--worker',str(request)],env)


def worker(path):
    value=json.loads(Path(path).read_text())
    sys.path.insert(0,str(ENGINE/'ops'))
    import score_mbpp
    root=Path(value['data'])
    (root/'ops').mkdir(exist_ok=True)
    score_mbpp.ROOT=root
    sandbox=Path(value['sandbox'])
    sandbox.parent.mkdir(parents=True,exist_ok=True)
    import tempfile
    import stat
    with tempfile.TemporaryDirectory(prefix='permission-probe-',dir=sandbox.parent) as probe:
        file=Path(probe)/'probe'
        file.write_text('ownership check')
        file.chmod(0o444)
        info=file.stat()
        if info.st_uid!=0 or info.st_mode & (stat.S_IWGRP|stat.S_IWOTH):
            raise RuntimeError('Sandbox requires root-owned POSIX permissions; use --sandbox-root on ext4 or tmpfs')
    sys.argv=['score_mbpp','--sandbox-root',str(sandbox)]
    sys.argv+=['--validate-only'] if value['validate_only'] else ['--input',value['input']]
    score_mbpp.main()


if __name__=='__main__':
    worker(sys.argv[2]) if len(sys.argv)>1 and sys.argv[1]=='--worker' else main()
