"""Guard and time one external LaRoSA scheduler job without changing core."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

from common import dump,sha
from experiment_timing import Attempt,job


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-manifest',type=Path,required=True)
    p.add_argument('--source-sha256',required=True)
    p.add_argument('--model',required=True)
    p.add_argument('--phase',choices=('prepare','audit','search'),required=True)
    p.add_argument('--target',type=int,choices=(40,50,60))
    p.add_argument('--timing-log',type=Path,required=True)
    p.add_argument('--receipt',type=Path,required=True)
    p.add_argument('command',nargs=argparse.REMAINDER)
    args=p.parse_args()
    command=args.command[1:] if args.command[:1]==['--'] else args.command
    if not command:
        p.error('A child command is required after --')
    if sha(args.source_manifest)!=args.source_sha256:
        raise RuntimeError('LaRoSA source manifest changed after scheduling')
    source=json.loads(args.source_manifest.read_text())
    for filename,expected in source['files'].items():
        if sha(filename)!=expected:
            raise RuntimeError(f'Frozen source changed: {filename}')
    visible=os.environ.get('CUDA_VISIBLE_DEVICES','').split(',')
    if len(visible)!=1 or not visible[0].strip():
        raise RuntimeError('Scheduler must expose exactly one GPU')
    metadata=dict(method='LaRoSA',model=args.model,label=f'larosa:{args.phase}',
        category='allocation' if args.phase=='search' else 'baseline_preparation' if args.phase=='prepare' else 'baseline_audit',
        substage=f'larosa_{args.phase}',targets=[args.target] if args.target else [40,50,60],
        target=args.target,attribution='exact_target' if args.target else 'shared_targets',gpu_reserved=True)
    def interrupted(signum,frame):
        raise KeyboardInterrupt(f'Signal {signum}')
    signal.signal(signal.SIGTERM,interrupted)
    with job(args.timing_log,**metadata):
        # Search owns its detailed allocation leaf. Other phases are timed here.
        scope='container' if args.phase=='search' else 'leaf'
        with Attempt(args.timing_log,scope=scope,**metadata):
            subprocess.run(command,check=True)
    dump(args.receipt,dict(status='complete',source_manifest_sha256=args.source_sha256,
        model=args.model,phase=args.phase,target=args.target,command=command,
        single_GPU=True,TP=False,PP=False,item_sharding=False))


if __name__=='__main__':
    main()
