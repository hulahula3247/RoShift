#!/usr/bin/env python3
"""One-GPU generation from a passed RoShift or baseline runtime audit.

MBPP generation is deliberately recorded as generated_pending_scoring: its
generated programs must subsequently pass the separate sandbox scorer. The
model process never imports the bundled code_eval utility or executes programs.
"""
from __future__ import annotations
import argparse
import copy
import fcntl
import functools
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from generation_tasks import ROOT, CODE, COUNTS, build, digest, literal_fewshot_samples, sha, write
from prepare_datasets import sanitized


def environment_identity():
    import torch
    import transformers
    distributions=('datasets','huggingface-hub','evaluate','numpy','scipy','sympy',
                   'math-verify','latex2sympy2-extended','antlr4-python3-runtime')
    packages={name:importlib.metadata.version(name) for name in distributions}
    return dict(python=sys.version,torch=torch.__version__,transformers=transformers.__version__,
        packages=packages,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0),
        dtype='float16',attention='sdpa',tf32=False,
        mbpp_sandbox_sha256=sha(ROOT/'ops/mbpp_sandbox.py'),mbpp_scorer_sha256=sha(ROOT/'ops/score_mbpp.py'))


def finish_mbpp(path):
    from mbpp_sandbox import SAFE_ENV
    subprocess.run([sys.executable,str(ROOT/'ops/score_mbpp.py'),'--input',str(path)],
                   env=SAFE_ENV,check=True)
    result=json.loads(path.read_text())
    if result.get('status')!='complete' or result.get('benchmark_complete') is not True or result.get('samples')!=500:
        raise RuntimeError('Isolated MBPP scoring did not finish all 500 documents')
    return result


def require_head_gate(profile,validation_file=None):
    marker=json.loads((profile/'materialized.json').read_text())
    gate_path=validation_file or profile/'sparse_head_validation.json'
    if not gate_path.is_absolute():gate_path=profile/gate_path
    gate=json.loads(gate_path.read_text())
    if gate.get('status')!='complete' or gate.get('passed') is not True:
        raise RuntimeError('A passed sparse-head validation is required for generation')
    if gate.get('generation_probe',{}).get('passed') is not True:
        raise RuntimeError('Sparse-head dense-prompt/cached-decode probe must pass')
    if gate.get('profile_fingerprint')!=marker['fingerprint']:
        raise RuntimeError('Sparse-head validation belongs to a different body profile')
    q=gate['q_head']
    if sha(Path(q['path']))!=q['sha256']:
        raise RuntimeError('Validated Q_head is missing or changed')
    policy=gate['head_policy']
    if not 0<int(policy['first_dim'])<int(policy['second_dim']):
        raise RuntimeError('Invalid sparse-head prefix dimensions')
    if not all(math.isfinite(float(policy[k])) for k in ('z1','z2')):
        raise RuntimeError('Invalid sparse-head pruning thresholds')
    return gate,gate_path


def samples_for(raw,task):
    return [(name,row) for name,rows in sorted(raw.get('samples',{}).items())
            if name==task or name.startswith(task+'_') for row in rows]


def historical_prompt_audit(raw,task):
    identities=[]
    for name,row in samples_for(raw,task):
        identities.append((name,row.get('doc_id'),row.get('doc_hash'),row.get('prompt_hash'),row.get('target_hash')))
    return dict(current_identity_sha256=digest(identities),samples=len(identities),
                historical_identity_available=False,identical=None)


def load_tasks(manager,name,entry):
    """Instantiate official task definitions; MBPP uses a metric-free prompt proxy."""
    from lm_eval.tasks import get_task_dict
    from lm_eval.api.task import ConfigurableTask
    if name!='mbpp':
        task_tree=get_task_dict([name],task_manager=manager)
    else:
        cfg=copy.deepcopy(entry['configs']['mbpp'])
        cfg.pop('resolved_cap_origin',None)
        cfg['fewshot_config']['samples']=literal_fewshot_samples(cfg['fewshot_config']['samples'])
        # The unsafe flag describes its original executable scorer. This prompt
        # proxy uses predict_only and produces no accuracy metric.
        cfg['unsafe_code']=False
        cfg['metric_list']=[dict(metric='exact_match',aggregation='mean',higher_is_better=True)]
        task_tree={'mbpp':ConfigurableTask(config=cfg)}
    import run_likelihood as likelihood
    leaves=dict(likelihood.leaf_tasks(task_tree))
    if set(leaves)!=set(entry['configs']):
        raise RuntimeError(f'{name}: task/subtask definition mismatch')
    if name=='bbh_cot_fewshot' and len(leaves)!=27:
        raise RuntimeError('All 27 BBH subjects are required')
    for subtask,task in leaves.items():
        task.set_config('generation_kwargs',dict(entry['configs'][subtask]['generation_kwargs']))
    counts={subtask:len(task.eval_docs) for subtask,task in leaves.items()}
    if sum(counts.values())!=entry['expected_documents']:
        raise RuntimeError(f'{name}: full document count mismatch: {counts}')
    return task_tree,leaves,counts


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True)
    p.add_argument('--model-path',type=Path,required=True)
    p.add_argument('--provider',choices=['roshift','teal','larosa'],default='roshift')
    p.add_argument('--work',type=Path)
    p.add_argument('--profile')
    p.add_argument('--head-validation-file',type=Path,help='Versioned passed RoShift head marker; relative paths are within the profile')
    p.add_argument('--head-runtime-path',type=Path,default=ROOT/'ops/roshift_head/runtime.py')
    p.add_argument('--artifact',type=Path,help='TEAL histogram/lookup root or LaRoSA Q root')
    p.add_argument('--audit',type=Path,help='Matching passed baseline precision audit')
    p.add_argument('--recipe',type=Path,help='Completed LaRoSA paper allocation recipe')
    p.add_argument('--sparsity',type=float,choices=[.4,.5,.6])
    p.add_argument('--teal-source-root',type=Path,default=ROOT/'ops/teal')
    p.add_argument('--source-identity',type=Path)
    p.add_argument('--tasks',nargs='+',choices=list(COUNTS),default=['minerva_math500','gsm8k','bbh_cot_fewshot'])
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--protocol',type=Path,default=ROOT/'ops/generation_protocol.json')
    p.add_argument('--batch-size',type=int,default=1)
    p.add_argument('--execute',action='store_true')
    args=p.parse_args()
    if args.batch_size<=0 or len(set(args.tasks))!=len(args.tasks):p.error('Positive batch and unique tasks required')
    if args.provider=='roshift' and (args.work is None or args.profile is None):p.error('RoShift requires --work and --profile')
    if args.provider!='roshift' and any(x is None for x in (args.artifact,args.audit,args.sparsity)):
        p.error('Baseline generation requires --artifact, --audit and --sparsity')
    if args.provider=='larosa' and args.recipe is None:p.error('LaRoSA requires --recipe')
    protocol=json.loads(args.protocol.read_text())
    if protocol['fingerprint']!=build()['fingerprint']:
        raise RuntimeError('Generation source/configuration changed; regenerate its manifest')
    if not args.execute:
        print(json.dumps(dict(mode='plan_only',model=args.model,provider=args.provider,profile=args.profile,tasks=args.tasks,
            requirements=(['complete body PPL audit',f'passed matching {args.head_validation_file or "sparse_head_validation.json"}'] if args.provider=='roshift'
                          else ['complete matching baseline artifacts','passed matching precision audit','dense native head']),
            protocol_fingerprint=protocol['fingerprint'],historical_identity_verified=False,
            mbpp_status='generation and isolated scoring are separate stages'),indent=2))
        return
    sys.path.insert(0,str(CODE/'tools'))
    import run_likelihood as likelihood
    likelihood.configure_environment()
    import torch
    import lm_eval
    from lm_eval.models.huggingface import HFLM
    from lm_eval.tasks import TaskManager
    from runtime_token_scope import is_decode_forward
    import generation_providers
    if torch.cuda.device_count()!=1 or int(os.environ.get('WORLD_SIZE','1'))!=1:
        raise RuntimeError('Exactly one GPU and one ordinary Python process are required')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    current_environment=environment_identity()
    cap_confirmation=ROOT/'ops/generation_user_confirmation.json'
    confirmed=json.loads(cap_confirmation.read_text())
    if confirmed['generation_caps']!=protocol['generation_cap_confirmation']['caps']:
        raise RuntimeError('Output token limits do not match the generation settings')
    args.output.mkdir(parents=True,exist_ok=True)
    lock=(args.output/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    prepared=generation_providers.prepare(args,likelihood,require_head_gate)
    manager=TaskManager()
    pending=[]
    for name in args.tasks:
        entry=protocol['task_configs'][name]
        tree,leaves,counts=load_tasks(manager,name,entry)
        identity=dict(**prepared.identity,protocol_fingerprint=protocol['fingerprint'],
            environment=current_environment,user_cap_confirmation_sha256=sha(cap_confirmation),
            task=name,batch_size=args.batch_size,counts=counts,
            task_configs={key:task.dump_config() for key,task in leaves.items()},
            datasets={key:{split:dict(rows=len(data),fingerprint=getattr(data,'_fingerprint',None))
                           for split,data in task.dataset.items()} for key,task in leaves.items()},
            source_files=likelihood.source_files(),ops_sources={str(path):sha(path) for path in
                (Path(__file__),ROOT/'ops/generation_providers.py',ROOT/'ops/score_mbpp.py',ROOT/'ops/mbpp_sandbox.py')})
        fingerprint=likelihood.digest(identity)
        path=args.output/f'{name}.json'
        if path.exists():
            previous=likelihood.read(path)
            if previous.get('fingerprint')!=fingerprint:raise RuntimeError(f'Changed identity; use a new output directory: {path}')
            if previous.get('status') in ('complete','generated_pending_scoring'):
                if len(samples_for(previous.get('raw',{}),name))!=entry['expected_documents']:
                    raise RuntimeError('Saved generation result is incomplete')
                if previous.get('status')=='generated_pending_scoring':previous=finish_mbpp(path)
                print(f'[skip] {name}: {previous["status"]}',flush=True);continue
        pending.append((name,tree,counts,identity,fingerprint,path))
    if not pending:return
    loaded=prepared.load();net=loaded.model
    phases=dict(prefill_calls=0,prefill_token_positions=0,decode_calls=0,decode_token_positions=0)
    original_forward=net.forward
    @functools.wraps(original_forward)
    def count_scopes(*positional,**kwargs):
        ids=kwargs.get('input_ids',positional[0] if positional else None)
        phase='decode' if is_decode_forward(ids,kwargs.get('past_key_values')) else 'prefill'
        phases[phase+'_calls']+=1
        if ids is not None:phases[phase+'_token_positions']+=ids.numel()
        return original_forward(*positional,**kwargs)
    net.forward=count_scopes
    with torch.inference_mode():
        for name,tree,counts,identity,fingerprint,path in pending:
            start=time.monotonic();entry=protocol['task_configs'][name]
            for key in phases:phases[key]=0
            loaded.reset();torch.cuda.reset_peak_memory_stats()
            row=dict(status='running',task=name,fingerprint=fingerprint,provenance=identity,
                     cohort=prepared.cohort,
                     historical_exact_protocol_identity_verified=False,
                     protocol_note=protocol['protocol_note'])
            likelihood.write(path,row)
            try:
                lm=HFLM(pretrained=net,tokenizer=str(args.model_path),dtype='float16',device='cuda:0',
                        batch_size=args.batch_size,parallelize=False,trust_remote_code=False)
                original_loader=manager.load_task_or_group
                manager.load_task_or_group=lambda names:tree
                cache=args.output/'response_cache'/f'{name}_{fingerprint}';cache.parent.mkdir(parents=True,exist_ok=True)
                try:
                    raw=lm_eval.simple_evaluate(model=lm,tasks=[name],task_manager=manager,
                        num_fewshot=entry['protocol']['fewshot'],limit=None,use_cache=str(cache),
                        log_samples=True,cache_requests=False,apply_chat_template=False,
                        predict_only=(name=='mbpp'),confirm_run_unsafe_code=False,**protocol['approved_seeds'])
                finally:manager.load_task_or_group=original_loader
                for subtask,count in counts.items():
                    if raw['n-samples'][subtask]['effective']!=count or len(raw['samples'][subtask])!=count:
                        raise RuntimeError(f'{subtask}: incomplete generation result')
                row.update(raw=raw,samples=sum(counts.values()),prompt_audit=historical_prompt_audit(raw,name),
                    elapsed_sec=time.monotonic()-start,scope_statistics=dict(phases),provider_statistics=loaded.statistics(),
                    statistics_scope='computed rows including padding/EOS rows; cached responses excluded',
                    peak_vram_gib=torch.cuda.max_memory_allocated()/2**30)
                if name=='mbpp':
                    row.update(status='generated_pending_scoring',metric_name=entry['protocol']['metric'],
                               scoring_required='isolated no-network/no-credential Python sandbox',benchmark_complete=False)
                else:
                    metric=entry['protocol']['metric'];metrics=raw['results'].get(name) or raw.get('groups',{}).get(name)
                    value=float(metrics[metric])
                    if not math.isfinite(value):raise RuntimeError('Nonfinite generation metric')
                    row.update(status='complete',metric_name=metric,metric_value=value,score_percent=100*value,
                               benchmark_complete=True)
                likelihood.write(path,row)
                if name=='mbpp':row=finish_mbpp(path)
                print(f'[{row["status"]}] {name} samples={row["samples"]}',flush=True)
            except BaseException as error:
                status='generated_pending_scoring' if name=='mbpp' and row.get('samples')==500 else ('paused' if isinstance(error,KeyboardInterrupt) else 'failed')
                row.update(status=status,
                           error=sanitized(error),traceback=sanitized(traceback.format_exc()),elapsed_sec=time.monotonic()-start)
                likelihood.write(path,row);raise


if __name__=='__main__':main()
