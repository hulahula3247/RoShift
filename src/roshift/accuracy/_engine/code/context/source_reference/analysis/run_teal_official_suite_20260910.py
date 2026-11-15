"""Resumable official-policy TEAL baseline suite."""
from __future__ import annotations

import argparse
import csv
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import traceback

os.environ['HF_DEACTIVATE_ASYNC_LOAD'] = '1'
os.environ.setdefault('HF_HOME', '/repro/data/hf_cache')
os.environ.setdefault('HF_ALLOW_CODE_EVAL', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
REPO = Path(__file__).resolve().parents[1]
OFFICIAL = REPO/'official_method_eval_20260827'
sys.path.insert(0, str(OFFICIAL))
ROOT = Path('/repro/data/teal_official_12tasks_20260910')
DENSE = Path('/repro/data/dense_baseline_13tasks_20260909/results.json')
UPSTREAM = OFFICIAL/'repos/TEAL'
MODELS = {
    'llama31': '/repro/models/Meta-Llama-3.1-8B',
    'llama2': '/repro/models/NousResearch-Llama-2-7b-hf',
    'mistral': '/repro/models/Mistral-7B-v0.1',
    'qwen25': '/repro/models/Qwen2.5-7B',
    'qwen3': '/repro/data/hf_local/Qwen3-8B-Base',
    'gemma1': '/repro/data/hf_local/gemma-7b',
}
TASKS = {
    'mmlu': ('MMLU',5,14042), 'mbpp': ('MBPP',3,500),
    'boolq': ('BoolQ',0,3270), 'piqa': ('PIQA',0,1838), 'arc_easy': ('ARC-E',0,2376),
    'gsm8k': ('GSM8K',8,1319), 'arc_challenge': ('ARC-C',25,1172),
    'winogrande': ('WinoGrande',5,1267), 'commonsense_qa': ('CSQA',0,1221),
    'bbh_cot_fewshot': ('BBH',3,6511), 'minerva_math500': ('MATH-500',4,500),
}
GEN = ('mbpp','minerva_math500','bbh_cot_fewshot','gsm8k')
ORDER = ('piqa','boolq','arc_easy','arc_challenge','winogrande','commonsense_qa','mmlu') + GEN
RELEASED = {'llama2': UPSTREAM/'models/Llama-2-7B', 'mistral': UPSTREAM/'models/Mistral-7B',
            'llama31': Path('/repro/data/official_method_eval_20260827/teal_l31_greedy_reoptimized')}
DIAGNOSTIC_POLICY = {
    'version': 2,
    'documents_per_task': 32,
    'group_sampling': '32 documents per constituent task for MMLU and BBH',
    'generation': 'same saved Dense task generation settings as the full benchmark',
    'additional_generation_cap': None,
    'changes': 'Remove diagnostic-only 128-token cap; retain document limit and all benchmark settings',
    'benchmark_manifest': 'Frozen initial manifest retained to preserve accuracy response-cache fingerprints',
}


def read(path):
    return json.loads(Path(path).read_text())


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, default=str)+'\n')
    tmp.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def log(value):
    print(f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] {value}', flush=True)


def artifact(root, model):
    return RELEASED.get(model, root/'calibration'/model)


def reference(model, task):
    record = read(DENSE)['results'][model][task]
    assert record['status'] == 'complete'
    row = read(record['path'])
    return record, row, row.get('raw', row.get('output'))


def batch_for(model, task):
    record, row, raw = reference(model, task)
    batch = row.get('batch_size') or record.get('batch_size') or raw.get('config', {}).get('batch_size')
    if str(batch).isdigit():
        return int(batch)
    # Archived rows sometimes omit the explicit CLI batch. Do not invent one.
    return 4 if task in ('mmlu','arc_challenge','mbpp') or (model in ('gemma1','llama2') and task in GEN) else 8


def seeds_for(raw):
    cfg = raw.get('config', {})
    return dict(random_seed=cfg.get('random_seed',1234), numpy_random_seed=cfg.get('numpy_seed',1234),
                torch_random_seed=cfg.get('torch_seed',1234), fewshot_random_seed=cfg.get('fewshot_seed',1234))


def active_tasks(root):
    path=Path(root)/'queue_policy.json'
    policy=read(path) if path.exists() else {}
    deferred=policy.get('deferred_tasks',[])
    if not isinstance(deferred,list) or any(t not in TASKS for t in deferred):
        raise ValueError('queue_policy.json contains unknown deferred tasks')
    tasks={t:cfg for t,cfg in TASKS.items() if t not in deferred}
    if not tasks:
        raise ValueError('queue_policy.json must retain at least one task')
    return tasks


MMLU_REVISION = 'c30699e8356da336a370243923dbaf21066bb9fe'


def configure_dataset_access(task):
    if task!='mmlu':
        return None
    cache=Path(os.environ.get('HF_DATASETS_CACHE',str(Path(os.environ['HF_HOME'])/'datasets')))
    folder=cache/'cais___mmlu'
    subjects=sorted(p for p in folder.iterdir() if p.is_dir())
    if len(subjects)!=57:
        raise RuntimeError('MMLU offline cache must contain all 57 subjects')
    total=0
    rows={}
    for subject in subjects:
        infos=list(subject.glob('*/*/dataset_info.json'))
        if len(infos)!=1 or infos[0].parent.name!=MMLU_REVISION:
            raise RuntimeError(f'MMLU cache missing or ambiguous revision: {subject.name}')
        info=read(infos[0])
        if info['config_name']!=subject.name or info['splits']['dev']['num_examples']!=5:
            raise RuntimeError(f'MMLU cache metadata mismatch: {subject.name}')
        for split in info['splits']:
            files=list(infos[0].parent.glob(f'mmlu-{split}*.arrow'))
            if not files or any(p.stat().st_size==0 for p in files):
                raise RuntimeError(f'MMLU cache missing split: {subject.name}/{split}')
        count=info['splits']['test']['num_examples']
        total+=count
        rows[subject.name]={'test_rows':count,'metadata_sha256':sha(infos[0])}
    if total!=14042:
        raise RuntimeError(f'MMLU cache test count mismatch: {total}')
    # This runs before importing datasets/transformers in a new worker.
    os.environ['HF_HUB_OFFLINE']='1'
    os.environ['HF_DATASETS_OFFLINE']='1'
    return dict(dataset='cais/mmlu',revision=MMLU_REVISION,cache=str(cache),
                network='disabled',subjects=rows,test_rows=total,dev_rows=285)


def retry_dataset_load(fn,label,attempts=4,sleep=None):
    sleep=time.sleep if sleep is None else sleep
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as exc:
            current=exc
            seen=set()
            transient=False
            retry_after=0.0
            while current is not None and id(current) not in seen:
                seen.add(id(current))
                response=getattr(current,'response',None)
                status=getattr(response,'status_code',None)
                transient |= status in (429,500,502,503,504)
                kind=type(current)
                transient |= (kind.__module__.startswith(('httpx','httpcore','requests')) and
                              kind.__name__ in ('ConnectError','ReadError','RemoteProtocolError',
                                                'ConnectTimeout','ReadTimeout','WriteTimeout','PoolTimeout',
                                                'Timeout','ConnectionError'))
                header=getattr(response,'headers',{}).get('Retry-After','')
                try:
                    retry_after=max(retry_after,float(header))
                except (TypeError,ValueError):
                    if header:
                        from email.utils import parsedate_to_datetime
                        try:
                            retry_after=max(retry_after,parsedate_to_datetime(header).timestamp()-time.time())
                        except (ValueError,TypeError,OverflowError):
                            pass
                match=re.search(r'Retry after\s+(\d+(?:\.\d+)?)\s+seconds',str(current),re.I)
                if match:
                    retry_after=max(retry_after,float(match.group(1)))
                current=current.__cause__ or current.__context__
            if not transient or attempt+1==attempts:
                raise
            delay=max(30*2**attempt,retry_after)+1
            log(f'DATASET RETRY {label} {attempt+1}/{attempts-1} after {delay:.0f}s: {exc}')
            sleep(delay)


def diagnostic_policy(root):
    path=root/'diagnostic_policy.json'
    if path.exists():
        if read(path)!=DIAGNOSTIC_POLICY:
            raise RuntimeError('Unexpected diagnostic policy; audit before changing it')
    else:
        dump(path,DIAGNOSTIC_POLICY)
    return DIAGNOSTIC_POLICY.copy()


def configure_generation_from_dense(tree, configs):
    """Full evaluations and sampled diagnostics use identical generation rules."""
    for name,item in tree.items():
        if isinstance(item,dict):
            configure_generation_from_dense(item,configs)
        elif isinstance(item,tuple):
            configure_generation_from_dense({name:item[1]},configs)
        elif hasattr(item,'set_config'):
            cfg=configs.get(name,{})
            if cfg.get('generation_kwargs'):
                item.set_config('generation_kwargs',dict(cfg['generation_kwargs']))


def prepare(root):
    COMMIT = 'fb7373c93ac3594817c9ee64d4e08b47430a1822'
    commit = subprocess.check_output(['git','rev-parse','HEAD'], cwd=UPSTREAM, text=True).strip()
    assert commit == COMMIT, commit
    diff = subprocess.check_output(['git','diff','--ignore-cr-at-eol','HEAD','--','utils/utils.py','teal/greedyopt.py'],
                                   cwd=UPSTREAM, text=True)
    assert not diff
    source = {str(p.relative_to(UPSTREAM)):sha(p) for p in (UPSTREAM/'utils/utils.py', UPSTREAM/'teal/greedyopt.py', UPSTREAM/'teal/grab_acts.py')}
    reuse = {}
    for model, folder in RELEASED.items():
        layers = read(Path(MODELS[model])/'config.json')['num_hidden_layers']
        files = []
        for i in range(layers):
            files += [folder/'lookup'/f'layer-{i}'/'results.csv']
            files += [folder/'histograms'/f'layer-{i}'/p/'histograms.pt' for p in ('mlp','self_attn')]
        assert all(p.is_file() for p in files), model
        reuse[model] = dict(path=str(folder), files={str(p.relative_to(folder)):sha(p) for p in files},
                           provenance='fresh exact Llama-3.1 greedy' if model=='llama31' else 'upstream released artifact')
    audit = dict(artifact_reuse=reuse, scores_reused=[], scores_excluded=[
        dict(root='/repro/data/official_method_eval_20260827/benchmarks_l31_greedy_reoptimized',
             reason='upstream half-prefill in likelihood AND generation prompt; violates common scope'),
        dict(root='/repro/data/official_method_eval_20260827/benchmarks_l31_greedy_extended_20260905',
             reason='same verified half-prefill runtime; cannot relabel as full-input or dense-prompt'),
        dict(root='/repro/data/qwen25_7b/latest_benchmark_suite_20260903/baselines',
             reason='uniform thresholds, not official greedy allocation'),
        dict(root='/repro/data/mistral7b/baseline_mixed_adaptive_dp_40_50_20260818',
             reason='uniform thresholds, not official greedy allocation')])
    dump(root/'reuse_audit.json', audit)
    # Preserve the original accuracy/cache fingerprint. The separate diagnostic
    # policy below supersedes the initial diagnostic description in this snapshot.
    manifest = dict(models=MODELS, tasks=TASKS, sparsities=[0.5,0.4], commit=commit, source=source,
                    calibration='upstream Alpaca train first 300 records, 10x2048; released artifacts reused as published',
                    calibration_override='official Alpaca calibration',
                    likelihood='all input positions sparse; dense LM-head',
                    generation='entire prompt dense; subsequent cached decode sparse; dense LM-head',
                    ported_models=['qwen25','qwen3','gemma1'],
                    batch_policy='match saved Dense batch where available; halve on CUDA OOM',
                    actual_sparsity='separate deterministic diagnostic: first 32 docs/task (per BBH/MMLU subtask); generation limited to 128 new tokens',
                    dense_registry_sha256=sha(DENSE), source_reuse=audit,
                    runtime_sha256=sha(REPO/'analysis/teal_official_runtime_20260910.py'))
    previous=root/'manifest.json'
    if previous.exists() and read(previous)!=json.loads(json.dumps(manifest,default=str)):
        finished=list((root/'benchmarks').glob('*/*/*.json'))
        if any(read(p).get('status')=='complete' for p in finished):
            raise RuntimeError('Protocol changed after completed results; use a new output root')
    dump(previous, manifest)
    diagnostic_policy(root)
    elapsed = 0
    dense_records = read(DENSE)['results']
    for model in MODELS:
        for task in active_tasks(root):
            info = dense_records[model][task]
            elapsed += info.get('elapsed_sec') or 0
    dump(root/'eta.json', dict(dense_single_suite_hours=elapsed/3600,
                             two_sparsities_dense_reference_hours=2*elapsed/3600,
                             active_tasks=list(active_tasks(root)),
                             note='TEAL PyTorch masks may take longer; calibration and diagnostics extra; no speed kernels'))
    return manifest


def load_native(model):
    import torch
    from transformers import AutoModelForCausalLM
    torch.set_num_threads(4)
    torch.manual_seed(1234)
    return AutoModelForCausalLM.from_pretrained(MODELS[model], torch_dtype=torch.float16,
                device_map='cuda:0', attn_implementation='sdpa').eval().requires_grad_(False)


def calibrate(args):
    import torch
    from datasets import load_dataset
    from transformers import AutoTokenizer
    from teal_official_runtime_20260910 import calibrate_native
    folder = artifact(args.root, args.model)
    if args.model in RELEASED or (folder/'complete.json').exists():
        return
    folder.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(MODELS[args.model], use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id if tok.eos_token_id is not None else 0
    records = list(load_dataset('tatsu-lab/alpaca', split='train', streaming=True).take(300))
    text = ''.join(r['text']+'\n\n' for r in records)
    encoded = tok(text, truncation=True, max_length=2048, return_overflowing_tokens=True,
                  padding='max_length', return_tensors='pt')
    ids = encoded['input_ids'][:10]
    assert ids.shape == (10,2048)
    dump(folder/'dataset.json', dict(dataset='tatsu-lab/alpaca', split='train', sampling='first 300, no shuffle',
                 text_sha256=hashlib.sha256(text.encode()).hexdigest(), ids_sha256=digest(ids.tolist()),
                 shape=list(ids.shape), tokenizer=MODELS[args.model]))
    torch.save(ids, folder/'input_ids.pt')
    model = load_native(args.model)
    calibrate_native(model, args.model, folder, ids.cuda())


def result_path(root, model, sparsity, task, diagnostic=False):
    return root/('actual_sparsity' if diagnostic else 'benchmarks')/model/f's{round(sparsity*100)}'/f'{task}.json'


def audit_prompts(raw, reference_raw):
    from summarize_roshift_extended_20260907 import identities
    a,b = identities(raw),identities(reference_raw)
    return dict(identity_available=bool(a) and bool(b), identical=bool(a) and a==b,
                shared_identical=bool(a) and all(k in b and b[k]==v for k,v in a.items()),
                shots_equal=raw.get('n-shot')==reference_raw.get('n-shot'),
                versions_equal=raw.get('versions')==reference_raw.get('versions'),
                dense_samples=len(b), current_samples=len(a))


def worker(args, loaded=None):
    import lm_eval
    import torch
    from lm_eval.tasks import TaskManager
    from official_common import build_hflm, extract_metric
    from run_teal_extended_benchmarks import extract_metrics
    from teal_official_runtime_20260910 import TealRuntime
    out = result_path(args.root,args.model,args.sparsity,args.task,args.diagnostic)
    log(f'LOAD {args.model} s={args.sparsity} task={args.task} diagnostic={args.diagnostic}')
    if loaded is None:
        model = load_native(args.model)
        model.config.use_cache = True
        runtime = TealRuntime(model, artifact(args.root,args.model), args.sparsity)
        lm = build_hflm(model, MODELS[args.model], str(args.batch), 'cuda:0')
        log('Native model, fixed thresholds and harness loaded')
    else:
        model,runtime,lm = loaded
        runtime.reset_stats()
    def lengths(values, width):
        runtime.external_valid_mask = torch.arange(width, device='cuda')[None,:] < torch.tensor(values,device='cuda')[:,None]
    lm._input_length_observer = lengths
    runtime.mode = 'generation' if args.task in GEN else 'likelihood'
    runtime.enabled_stats = args.diagnostic
    ref_info, ref_row, ref_raw = reference(args.model,args.task)
    protocol = dict(model=args.model, sparsity=args.sparsity, task=args.task, batch=args.batch,
                    seeds=seeds_for(ref_raw), scope=runtime.mode, head='dense',
                    histogram_lookup=str(artifact(args.root,args.model)), diagnostic=args.diagnostic,
                    max_new_tokens='Dense task config',
                    source_manifest=digest(read(args.root/'manifest.json')))
    if args.diagnostic:
        protocol['diagnostic_policy']=diagnostic_policy(args.root)
    fingerprint = digest(protocol)
    row = dict(status='running', config=protocol, fingerprint=fingerprint,
               started=time.time(), pid=os.getpid(), fewshot=TASKS[args.task][1], batch_size=args.batch)
    dump(out,row)
    begin = time.monotonic()
    try:
        manager = TaskManager()
        log('Task index loaded')
        # Retry transient metadata failures without reloading model weights.
        from lm_eval.tasks import get_task_dict
        task_dict = retry_dataset_load(lambda:get_task_dict([args.task], task_manager=manager),args.task)
        log('Task dataset/config loaded')
        configure_generation_from_dense(task_dict,ref_raw.get('configs',{}))
        # Keep lm-eval group aggregation while supplying the configured tree.
        manager.load_task_or_group = lambda names: task_dict
        cache = out.parent/'response_cache'/f'{args.task}_{fingerprint[:16]}'
        cache.parent.mkdir(parents=True,exist_ok=True)
        result = lm_eval.simple_evaluate(model=lm, tasks=[args.task],
                    num_fewshot=TASKS[args.task][1], limit=32 if args.diagnostic else args.limit,
                    use_cache=None if args.diagnostic else str(cache), task_manager=manager,
                    confirm_run_unsafe_code=True, apply_chat_template=False, log_samples=True,
                    **seeds_for(ref_raw))
        count = sum(r['effective'] for r in result['n-samples'].values())
        if not args.diagnostic and args.limit is None:
            assert count == TASKS[args.task][2], (count,TASKS[args.task])
        if args.task in ('arc_challenge','winogrande','commonsense_qa','bbh_cot_fewshot','minerva_math500'):
            metric,value,_ = extract_metrics(result,args.task)
        else:
            metric,value = extract_metric(args.task,result)
        assert value is not None
        row.update(status='complete', raw=result, samples=count, metric_name=metric,metric_value=value,
                   prompt_audit=audit_prompts(result,ref_raw), elapsed_sec=time.monotonic()-begin)
        if not args.diagnostic and args.limit is None:
            audit=row['prompt_audit']
            if audit['identity_available'] and not audit['identical']:
                row['comparison_warning']='Dense and TEAL prompt identities differ; not a matched comparison'
            elif not audit['identity_available']:
                row['comparison_warning']='Historical Dense sample hashes unavailable; full identity audit is not possible'
            import resource
            row['peak_rss_gib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2
            row['peak_vram_gib']=torch.cuda.max_memory_allocated()/2**30
        if args.diagnostic:
            stats = runtime.statistics()
            assert stats['mac_weighted_pruned'] is not None
            row['actual_sparsity'] = stats
        dump(out,row)
        log(f'COMPLETE {args.model}/{args.task} s={args.sparsity}: {value:.6f}')
    except BaseException as exc:
        row.update(status='failed', error=repr(exc),elapsed_sec=time.monotonic()-begin)
        dump(out,row)
        raise
    if not args.diagnostic and args.limit is None:
        diagnostic_args=argparse.Namespace(**vars(args))
        diagnostic_args.diagnostic=True
        diagnostic_path=result_path(args.root,args.model,args.sparsity,args.task,True)
        if not diagnostic_path.exists() or read(diagnostic_path).get('status')!='complete':
            del result,ref_raw,ref_row,row
            gc.collect()
            torch.cuda.empty_cache()
            worker(diagnostic_args,loaded=(model,runtime,lm))


def report(root):
    result = {}
    tasks=active_tasks(root)
    deferred=[t for t in TASKS if t not in tasks]
    diagnostics_complete=0
    dense_records = read(DENSE)['results']
    lines = ['# TEAL official-policy baseline status', '',
             'Dense sources are reused. See manifest.json and reuse_audit.json.',
             'Calibration follows upstream Alpaca, not the older WikiText convention.',
             'Likelihood: all input tokens sparse. Generation: dense prompt, sparse cached decode. LM-head dense.',
             'Qwen/Gemma are native-architecture ports of TEAL rules, not upstream-supported models.', '',]
    if deferred:
        lines += ['Deferred by queue_policy.json: '+', '.join(deferred)+'. Existing artifacts are preserved.', '']
    for model in MODELS:
        rows = []
        vals = [dense_records[model][t]['score'] for t in tasks]
        rows.append(['Dense']+[f'{v:.2f}' for v in vals]+[f'{sum(vals)/len(vals):.2f}'])
        for s in (.5,.4):
            cells = [f'TEAL {s:.0%}']
            scores=[]
            for t,d in zip(tasks,vals):
                p = result_path(root,model,s,t)
                item = read(p) if p.exists() else {'status':'not_started'}
                dp=result_path(root,model,s,t,True)
                diagnostics_complete+=int(dp.exists() and read(dp).get('status')=='complete')
                result[f'{model}/s{round(s*100)}/{t}']={k:v for k,v in item.items() if k!='raw'}
                if item['status']=='complete':
                    v=item['metric_value']*100
                    scores.append(v)
                    cells.append(f'{v:.2f} ({v-d:+.2f})')
                else:
                    cells.append('-')
            cells.append(f'{sum(scores)/len(scores):.2f}' if len(scores)==len(tasks) else '-')
            rows.append(cells)
        lines += [f'## {model}', '', '| Method | '+' | '.join(x[0] for x in tasks.values())+' | Avg |',
                  '|---|'+'---:|'*(len(tasks)+1)]
        lines += ['| '+' | '.join(row)+' |' for row in rows]+['']
    dump(root/'results.json',dict(completed=sum(v['status']=='complete' for v in result.values()),
                                expected=len(result),diagnostics_complete=diagnostics_complete,
                                active_tasks=list(tasks),deferred_tasks=deferred,results=result))
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')


def suite(args):
    args.root.mkdir(parents=True,exist_ok=True)
    lock = (args.root/'suite.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    prepare(args.root)
    child = None
    def stop(*_):
        if child and child.poll() is None:
            child.send_signal(signal.SIGINT)
            try: child.wait(timeout=30)
            except subprocess.TimeoutExpired: child.terminate()
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,stop)
    signal.signal(signal.SIGINT,stop)
    def launch(phase,model,task=None,s=.5,diagnostic=False,batch=1):
        nonlocal child
        tag=f'{model}_s{round(s*100)}_{task or phase}'+('_actual' if diagnostic else '')
        logpath=args.root/'logs'/f'{tag}_b{batch}.log'
        logpath.parent.mkdir(parents=True,exist_ok=True)
        cmd=[sys.executable,'-u',str(Path(__file__).resolve()),'--phase',phase,'--root',str(args.root),
             '--model',model,'--sparsity',str(s),'--batch',str(batch)]
        if task: cmd+=['--task',task]
        if diagnostic: cmd+=['--diagnostic']
        with logpath.open('a') as handle:
            child=subprocess.Popen(cmd,stdout=handle,stderr=subprocess.STDOUT)
            dump(args.root/'progress.json',dict(status='running',controller_pid=os.getpid(),worker_pid=child.pid,
                      model=model,task=task,phase=phase,sparsity=s,diagnostic=diagnostic,batch=batch,log=str(logpath)))
            code=child.wait()
        return code,logpath
    failed_calibrations = set()
    for s in (.5,.4):
        # Fast tasks across all models first, then all generation tasks.
        for task in ORDER:
            for model in MODELS:
                if task not in active_tasks(args.root):
                    break
                if model in failed_calibrations:
                    continue
                folder=artifact(args.root,model)
                if model not in RELEASED and not (folder/'complete.json').exists():
                    code,path=launch('calibrate',model)
                    if code:
                        dump(args.root/f'{model}.calibration_failed.json',dict(code=code,log=str(path)))
                        failed_calibrations.add(model)
                        continue
                for diagnostic in (False,True):
                    out=result_path(args.root,model,s,task,diagnostic)
                    if out.exists() and read(out).get('status')=='complete':
                        continue
                    batch=batch_for(model,task)
                    while True:
                        code,path=launch('worker',model,task,s,diagnostic,batch)
                        if code==0: break
                        if not diagnostic and out.exists() and read(out).get('status')=='complete':
                            # Accuracy committed; only the separate diagnostic failed.
                            break
                        tail=path.read_text(errors='replace')[-16000:]
                        if batch>1 and ('CUDA out of memory' in tail or 'OutOfMemoryError' in tail):
                            batch=max(1,batch//2)
                            log(f'OOM retry {model}/{task}: batch={batch}')
                            continue
                        failure=read(out) if out.exists() else {}
                        failure.update(status='failed',worker_exit_code=code,log=str(path))
                        if not failure.get('error'):
                            failure['error']=tail[-4000:]
                        dump(out,failure)
                        log(f'FAILED {model}/{task}, retained log {path}')
                        break
                    report(args.root)
    report(args.root)
    status=read(args.root/'results.json')
    complete=status['completed']==status['expected']==status['diagnostics_complete']
    dump(args.root/'progress.json',dict(status='complete' if complete else 'partial',
                                      completed=status['completed'],expected=status['expected'],
                                      diagnostics_complete=status['diagnostics_complete'],finished=time.time()))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--phase',choices=['prepare','worker','calibrate','suite','report'],default='suite')
    p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--model',choices=MODELS,default='llama31')
    p.add_argument('--task',choices=TASKS,default='piqa')
    p.add_argument('--sparsity',type=float,default=.5)
    p.add_argument('--batch',type=int,default=4)
    p.add_argument('--diagnostic',action='store_true')
    p.add_argument('--limit',type=int)
    args=p.parse_args()
    if args.phase=='worker' and args.task not in active_tasks(args.root):
        log(f'DEFERRED {args.model}/{args.task}: queue_policy.json; no model loaded')
        return
    if args.phase=='worker':
        access=configure_dataset_access(args.task)
        if access:
            dump(args.root/'dataset_access_mmlu.json',access)
            log(f'MMLU offline cache verified: {access["test_rows"]} test rows, 57 subjects')
    if args.phase=='prepare': prepare(args.root)
    elif args.phase=='suite': suite(args)
    elif args.phase=='worker': worker(args)
    elif args.phase=='calibrate': calibrate(args)
    else: report(args.root)


if __name__=='__main__':
    main()
