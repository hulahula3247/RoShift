#!/usr/bin/env python3
"""Finish all 500 MBPP pass@1 scores outside the model/credential process.

Candidate + newline + the task's three reference assertions, one candidate per
document, 3 second timeout. The packaged MBPP task requests no code extraction.
These are package-derived current settings; historical scorer identity is absent.
"""
from __future__ import annotations
import argparse
import fcntl
import json
import os
from pathlib import Path
import sys
import time

import mbpp_sandbox as sandbox
from generation_tasks import ROOT,CODE,digest,sha,write

SOURCES={
    'metric':'https://github.com/huggingface/evaluate/blob/main/metrics/code_eval/code_eval.py',
    'executor':'https://github.com/huggingface/evaluate/blob/main/metrics/code_eval/execute.py'}

def validate_sandbox(root):
    cases={
        'correct':('def add(a,b):return a+b\nassert add(2,3)==5',True),
        'incorrect':('def add(a,b):return a*b\nassert add(2,3)==5',False),
        'timeout':('while True:pass',False),
        'no_workspace':('import os\nassert not os.path.exists("/workspace")\nassert not os.path.exists("/proc")\nassert not os.path.exists("/root")',True),
        'minimal_environment':('import os\nassert os.getuid()==65534 and os.getgid()==65534\nassert os.getgroups()==[]\nassert not any("TOKEN" in x or "KEY" in x or "PASS" in x for x in os.environ)',True),
        'network_denied':('import socket,errno\ntry: socket.socket()\nexcept OSError as e: assert e.errno==errno.EPERM\nelse: raise AssertionError("socket unexpectedly allowed")',True),
        'process_denied':('import os,errno\ntry: os.fork()\nexcept OSError as e: assert e.errno==errno.EPERM\nelse: raise AssertionError("fork unexpectedly allowed")',True),
        'runtime_readonly':('try: open("/python/json/__init__.py","w")\nexcept PermissionError: pass\nelse: raise AssertionError("runtime writable")',True),
        'numpy':('import numpy as np\nassert np.sum([1,2,3])==6',True),
        'sympy':('import sympy\nassert sympy.isprime(17)',True),
    }
    outcomes={name:sandbox.execute(root,code,timeout=.25 if name=='timeout' else 3.0)
              for name,(code,expected) in cases.items()}
    for name,(code,expected) in cases.items():
        row=outcomes[name]
        if row.get('kind')=='sandbox_error' or row['passed'] is not expected:
            raise RuntimeError(f'Sandbox validation failed: {name}: {row}')
    if outcomes['timeout']['kind']!='timed_out':raise RuntimeError('Per-candidate timeout not enforced')
    return outcomes

def samples(row):
    if row.get('task')!='mbpp' or row.get('status') not in ('generated_pending_scoring','complete'):
        raise RuntimeError('A completed MBPP generation result is required')
    if row.get('samples')!=500 or row['provenance']['counts']!={'mbpp':500}:
        raise RuntimeError('All 500 MBPP test documents are required')
    records=row['raw']['samples']['mbpp']
    if len(records)!=500 or len({x['doc_id'] for x in records})!=500:
        raise RuntimeError('Missing or duplicated MBPP generation documents')
    programs=[]
    for record in records:
        responses=record['filtered_resps']
        if len(responses)!=1 or not isinstance(responses[0],str):
            raise RuntimeError('Exactly one deterministic candidate per MBPP document is required')
        target=record['target']
        if target!='\n'.join(record['doc']['test_list'][:3]):
            raise RuntimeError('Reference assertions disagree with the frozen MBPP task')
        programs.append(dict(doc_id=record['doc_id'],task_id=record['doc']['task_id'],
            program=responses[0]+'\n'+target,candidate_sha256=digest(responses[0]),
            target_sha256=digest(target),prompt_hash=record.get('prompt_hash')))
    return programs

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path)
    parser.add_argument('--sandbox-root',type=Path,default=ROOT/'ops/mbpp_sandbox_runtime')
    parser.add_argument('--validate-only',action='store_true')
    args=parser.parse_args()
    if not args.validate_only and args.input is None:parser.error('--input is required for scoring')
    if any(key in os.environ for key in ('HF_TOKEN','HUGGING_FACE_HUB_TOKEN','HUGGINGFACE_HUB_TOKEN')):
        raise RuntimeError('Launch the scorer with the provided minimal environment')
    lock=args.sandbox_root.with_suffix('.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
    runtime=sandbox.prepare(args.sandbox_root)
    validation=validate_sandbox(args.sandbox_root)
    if args.validate_only:
        target=ROOT/'ops/mbpp_sandbox_validation.json'
        write(target,dict(status='complete',passed=True,tests=validation,
            runtime_sha256=sha(args.sandbox_root/'runtime.json'),
            scorer_sha256=sha(Path(__file__)),sandbox_sha256=sha(Path(sandbox.__file__))))
        print(f'Sandbox validation passed: {target}',flush=True);return
    row=json.loads(args.input.read_text())
    programs=samples(row)
    identity=dict(generation_fingerprint=row['fingerprint'],programs=[{k:v for k,v in x.items() if k!='program'} for x in programs],
        scorer_sha256=sha(Path(__file__)),sandbox_sha256=sha(Path(sandbox.__file__)),
        runtime_sha256=sha(args.sandbox_root/'runtime.json'),timeout_sec=3.0,candidates_per_document=1,
        candidate_processing='verbatim filtered response; no Markdown/code extraction',
        reference_processing='candidate + newline + three task reference assertions',
        reference_semantics=SOURCES,packaged_task_sha256=sha(CODE/'vendor/lm_eval/tasks/mbpp/mbpp.yaml'),
        packaged_metric_sha256=sha(CODE/'vendor/lm_eval/tasks/mbpp/utils.py'),
        metric_integrity_assumption='Standard non-adversarial benchmark candidates; execution and outcome reporting share an interpreter. OS isolation is enforced, but hostile candidate self-report manipulation is not claimed impossible.',
        historical_scorer_identity_verified=False)
    fingerprint=digest(identity)
    output=args.input.with_name('mbpp.scoring.json')
    if output.exists():
        result=json.loads(output.read_text())
        if result['fingerprint']!=fingerprint:raise RuntimeError('Scorer identity changed; preserve old results and use a new output')
    else:result=dict(status='running',fingerprint=fingerprint,provenance=identity,validation=validation,results={})
    start=time.monotonic()
    for item in programs:
        key=str(item['doc_id'])
        if key in result['results']:continue
        value=sandbox.execute(args.sandbox_root,item['program'],timeout=3.0)
        if value['kind']=='sandbox_error':raise RuntimeError(f'Sandbox infrastructure failed at document {key}: {value}')
        result['results'][key]=dict(task_id=item['task_id'],**value)
        write(output,result)
        if len(result['results'])%25==0:print(f'MBPP scored {len(result["results"])} / 500',flush=True)
    if set(result['results'])!={str(x['doc_id']) for x in programs}:raise RuntimeError('Scoring result coverage mismatch')
    correct=sum(x['passed'] for x in result['results'].values())
    result.update(status='complete',samples=500,passed_samples=correct,metric_name='pass_at_1,none',
        metric_value=correct/500,score_percent=correct/5,elapsed_this_attempt_sec=time.monotonic()-start)
    write(output,result)
    current=json.loads(args.input.read_text())
    if current['fingerprint']!=row['fingerprint'] or digest(samples(current))!=digest(programs):
        raise RuntimeError('Generation inputs changed during scoring')
    current.update(status='complete',benchmark_complete=True,metric_name='pass_at_1,none',
        metric_value=correct/500,score_percent=correct/5,
        scoring=dict(path=str(output),sha256=sha(output),fingerprint=fingerprint,
                     isolated=True,passed_samples=correct,samples=500))
    current.pop('scoring_required',None)
    write(args.input,current)
    print(f'MBPP pass@1 = {correct}/500 = {correct/5:.3f}%',flush=True)

if __name__=='__main__':main()
