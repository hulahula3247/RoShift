#!/usr/bin/env python3
"""CPU-only generation dataset staging; never imports MBPP code_eval utilities."""
from __future__ import annotations
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from generation_tasks import ROOT, build, write, sha, digest
from prepare_datasets import block_network, document_hash, now, sanitized


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tasks',nargs='+',default=['mbpp','minerva_math500','gsm8k','bbh_cot_fewshot'])
    p.add_argument('--output',type=Path,default=ROOT/'ops/datasets_generation.json')
    p.add_argument('--protocol',type=Path,default=ROOT/'ops/generation_protocol.json')
    p.add_argument('--offline-check',action='store_true')
    args=p.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES']=''
    os.environ.setdefault('HF_HOME','/repro/cache')
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
        os.environ[key]='4'
    os.environ['HF_HUB_DISABLE_PROGRESS_BARS']='1'
    if args.offline_check:
        os.environ['HF_HUB_OFFLINE']='1'
        os.environ['HF_DATASETS_OFFLINE']='1'
        block_network()
    protocol=json.loads(args.protocol.read_text())
    if protocol['fingerprint']!=build()['fingerprint']:
        raise RuntimeError('Generation source hashes changed; regenerate the manifest')
    if not set(args.tasks).issubset(protocol['task_configs']) or len(set(args.tasks))!=len(args.tasks):
        p.error('Unique generation task names are required')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    lock=args.output.with_suffix('.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    started=time.monotonic()
    result=dict(status='running',started_at=now(),cpu_only=True,gpu_visible=False,
        model_loaded=False,code_eval_imported=False,benchmark_scores_computed=False,
        offline_check=args.offline_check,protocol_path=str(args.protocol),protocol_sha256=sha(args.protocol),
        historical_exact_protocol_identity_verified=False,tasks={},cache_files={},
        document_hash_scope='raw dataset evaluation records; no prompt processing or generated-code execution',
        script_sha256=sha(Path(__file__)))
    write(args.output,result)
    try:
        import datasets
        import huggingface_hub
        from huggingface_hub import HfApi
        result['versions']=dict(datasets=datasets.__version__,huggingface_hub=huggingface_hub.__version__)
        revisions={}
        for name in args.tasks:
            entry=protocol['task_configs'][name]
            row=dict(status='running',config_sha256=entry['config_sha256'],subjects={})
            result['tasks'][name]=row
            for subtask,cfg in entry['configs'].items():
                print(f'[{now()}] loading generation dataset {subtask} offline={args.offline_check}',flush=True)
                repo,config=cfg['dataset_path'],cfg.get('dataset_name')
                kwargs=cfg.get('dataset_kwargs',{})
                data=datasets.load_dataset(repo,config,**kwargs)
                requested=kwargs.get('revision')
                if (repo,requested) not in revisions:
                    if args.offline_check:
                        revisions[(repo,requested)]=dict(requested_revision=requested,network_metadata_not_requested=True)
                    else:
                        info=HfApi().dataset_info(repo,revision=requested)
                        revisions[(repo,requested)]=dict(requested_revision=requested,observed_hub_sha_after_load=info.sha,
                            pin_status='package_default_revision; observed_sha_is_not_independent_pin_proof' if not requested else 'explicit_revision')
                split=cfg.get('test_split') or cfg.get('validation_split')
                identity=document_hash(data[split],str)
                splits={}
                for split_name,part in data.items():
                    paths=[]
                    for cache in part.cache_files:
                        path=Path(cache['filename']);key=str(path);paths.append(key)
                        if key not in result['cache_files']:
                            result['cache_files'][key]=dict(bytes=path.stat().st_size,sha256=sha(path))
                    splits[split_name]=dict(rows=len(part),fingerprint=part._fingerprint,cache_files=paths)
                row['subjects'][subtask]=dict(evaluation_documents=identity,splits=splits,dataset_revision=revisions[(repo,requested)])
                write(args.output,result)
            total=sum(x['evaluation_documents']['rows'] for x in row['subjects'].values())
            if total!=entry['expected_documents']:
                raise RuntimeError(f'{name}: expected {entry["expected_documents"]} documents, found {total}')
            row.update(status='complete',actual_documents=total,
                       ordered_eval_identity_sha256=digest({k:v['evaluation_documents'] for k,v in row['subjects'].items()}))
            write(args.output,result)
        if not args.offline_check:
            offline=args.output.with_name(args.output.stem+'.offline.json')
            subprocess.run([sys.executable,str(Path(__file__).resolve()),'--offline-check','--tasks',*args.tasks,
                            '--output',str(offline),'--protocol',str(args.protocol)],check=True,env=os.environ.copy())
            checked=json.loads(offline.read_text())
            if checked['status']!='complete':raise RuntimeError('Offline verification incomplete')
            for name in args.tasks:
                if result['tasks'][name]['ordered_eval_identity_sha256']!=checked['tasks'][name]['ordered_eval_identity_sha256']:
                    raise RuntimeError(f'{name}: offline document identity mismatch')
            result['offline_verification']=dict(status='complete',path=str(offline),sha256=sha(offline),network_blocked=True)
        result.update(status='complete',finished_at=now(),elapsed_sec=time.monotonic()-started)
        write(args.output,result)
        print(f'[{now()}] generation dataset staging complete: {args.output}',flush=True)
    except BaseException as error:
        result.update(status='paused' if isinstance(error,KeyboardInterrupt) else 'failed',error=sanitized(error),
                      exception_type=type(error).__name__,traceback=sanitized(traceback.format_exc()),
                      elapsed_sec=time.monotonic()-started)
        write(args.output,result)
        print(f'[{now()}] generation dataset staging failed: {sanitized(error)}',flush=True)
        raise SystemExit(1)


if __name__=='__main__':main()
