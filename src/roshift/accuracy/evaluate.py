"""Frozen eleven-task evaluation; executable MBPP scoring is isolated."""
import gc
import hashlib
import importlib.metadata
import json
import math
import platform
from pathlib import Path
import time
from types import SimpleNamespace

from ..catalog import asset
from ..io import digest, fingerprint, atomic_json
from .execution import ENGINE
from .accounting import document_count


def release_cuda():
    import torch
    # Patched forwards can form cycles around a model. Drop those references
    # before the next preparation/gating stage loads another full checkpoint.
    gc.collect()
    torch.cuda.empty_cache()


def safe_json(value):
    if hasattr(value,'tolist'):
        return value.tolist()
    if hasattr(value,'item'):
        return value.item()
    if callable(value):
        return value.__module__+'.'+value.__qualname__
    return str(value)


def save(path, data):
    # Harness standard errors can be NaN for a one-example smoke test. The
    # actual score is checked separately; unavailable auxiliary values are null.
    def clean(value):
        if isinstance(value,float) and not math.isfinite(value):
            return None
        if isinstance(value,dict):
            return {k:clean(v) for k,v in value.items()}
        if isinstance(value,(list,tuple)):
            return [clean(v) for v in value]
        return value
    value=json.loads(json.dumps(data,default=safe_json))
    atomic_json(path,clean(value))


def ensure_ppl(request,path,work):
    import fcntl
    import materialize_profile
    from run_likelihood import check_ppl
    import torch
    model = request['model']
    profile = f"{model}_{request['allocation']}_s{request['sparsity']}"
    args = SimpleNamespace(model=model,model_path=path,work=work,profile=profile,condition=None)
    if request.get('use_search'):
        args.condition=work/model/'q'/f"{request['allocation']}_s{request['sparsity']}"
    folder = work/model/'rebuilt'/profile
    with (folder/'.accuracy_ppl.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        audit=folder/'ppl_audit.json'
        complete=audit.is_file() and json.loads(audit.read_text()).get('status')=='complete'
        if not complete:
            with torch.inference_mode():
                materialize_profile.ppl(args)
        check_ppl(folder)
    return args,folder


def ensure_head(request,path,work):
    from roshift_head_v2 import worker
    import torch
    args,folder=ensure_ppl(request,path,work)
    release_cuda()
    args.policy_json = None
    args.chunk = 32
    with torch.inference_mode():
        worker.run(args)
    gate_path = folder/'sparse_head_validation_v2.json'
    if not gate_path.exists():
        # A previous exact matching version-one gate may be retained by v2.
        candidates = [p for p in folder.glob('*head*validation*.json')
                      if json.loads(p.read_text()).get('passed') is True]
        if len(candidates)!=1:
            raise RuntimeError('No unique passed profile-specific head validation')
        gate_path = candidates[0]
    gate = json.loads(gate_path.read_text())
    if gate.get('passed') is not True or gate.get('generation_probe',{}).get('passed') is not True:
        raise RuntimeError('Sparse LM-head validation failed; generation is not authorized')
    return gate


def run(request,path,work,root):
    import torch
    import lm_eval
    from lm_eval.models.huggingface import HFLM
    from lm_eval.tasks import TaskManager,get_task_dict
    from transformers import AutoModelForCausalLM
    from .worker import load_roshift
    from .audit import statistics as read_statistics,reset_statistics
    import run_likelihood
    from run_generation import load_tasks,samples_for
    from generation_tasks import build
    tasks = asset('benchmarks.json')
    selected = request['tasks'] or list(tasks)
    selected = sorted(selected,key=lambda k:tasks[k]['kind']=='generation')
    generation = any(tasks[k]['kind']=='generation' for k in selected)
    gate=None
    if request['method']=='roshift':
        if generation:
            gate=ensure_head(request,path,work)
        else:
            ensure_ppl(request,path,work)
    release_cuda()
    if request['method']=='roshift':
        net,controller,statistics = load_roshift(request['model'],path,work,request['allocation'],request['sparsity'])
    elif request['method']=='dense':
        net = AutoModelForCausalLM.from_pretrained(path,dtype=torch.float16,
            device_map='cuda:0',attn_implementation='sdpa').eval().requires_grad_(False)
        controller = statistics = None
    else:
        from .baselines import load_baseline
        net,controller,statistics = load_baseline(request['method'],request['model'],path,root,
                                                  request['allocation'],request['sparsity'])
    manager = TaskManager(include_path=str(Path(__file__).with_name('tasks')))
    protocol = build()
    task_protocol = json.loads((ENGINE/'code/context/evidence/accuracy_inventory_20260916/task_protocol.json').read_text())
    lm = HFLM(pretrained=net,tokenizer=str(path),batch_size=request['batch_size'],device='cuda:0',
              max_length=None,logits_cache=True)
    head_installed = False
    summaries = []
    source = {str(p.relative_to(ENGINE.parent)):digest(p) for p in ENGINE.parent.glob('*.py')}
    source['engine_manifest'] = digest(ENGINE/'code/MANIFEST.json')
    from .integrity import profile_identity
    source['release_manifest'] = digest(ENGINE.parent/'release_sources.json')
    prepared=profile_identity(request,work,root)
    import transformers
    environment=dict(python=platform.python_version(),torch=torch.__version__,cuda=torch.version.cuda,
        device=torch.cuda.get_device_name(0),transformers=transformers.__version__,tf32=False,
        dependencies={name:importlib.metadata.version(name) for name in
                      ('datasets','numpy','tokenizers','accelerate','huggingface-hub','lm-eval','math-verify')})
    for name in selected:
        spec = tasks[name]
        native = spec['task']
        is_generation = spec['kind']=='generation'
        if controller is not None:
            if hasattr(controller,'task_policy'):
                controller.task_policy = 'decode_only' if is_generation else 'all'
            if hasattr(controller,'mode'):
                controller.mode = 'generation' if is_generation else 'likelihood'
        if is_generation and gate and not head_installed:
            from roshift_head_v2.runtime import install_validated_head
            install_validated_head(net,Path(gate['q_head']['path']),gate['head_policy'],controller)
            head_installed = True
        if is_generation:
            entry = protocol['task_configs'][native]
            # The main accuracy cohort is 5-shot GSM8K, not the separate
            # historical 8-shot LM-head ablation in the archived protocol.
            entry['protocol']['fewshot'] = spec['shots']
            for cfg in entry['configs'].values():
                cfg['num_fewshot'] = spec['shots']
            tree,leaves,counts = load_tasks(manager,native,entry)
        else:
            task_request=({'task':'mmlu','dataset_kwargs':{'revision':run_likelihood.MMLU_REVISION}}
                          if native=='mmlu' else native)
            tree = get_task_dict([task_request],task_manager=manager)
            leaves = dict(run_likelihood.leaf_tasks(tree))
            counts = {k:len(task.eval_docs) for k,task in leaves.items()}
        if sum(counts.values()) != spec['count']:
            raise RuntimeError(f'{name}: dataset count differs from the paper protocol')
        identity = dict(request=request,sources=source,environment=environment,prepared_profile=prepared,head_gate=gate,
            task_configs={key:task.dump_config() for key,task in leaves.items()},
            datasets={key:{split:dict(rows=len(data),fingerprint=getattr(data,'_fingerprint',None))
                       for split,data in task.dataset.items()} for key,task in leaves.items()})
        identity = json.loads(json.dumps(identity,default=safe_json))
        key = fingerprint(identity)
        output = root/'results'/request['model']/request['method']/f"{request['allocation']}_s{request['sparsity']}"/key[:16]/f'{native}.json'
        if output.exists():
            previous = json.loads(output.read_text())
            if previous.get('fingerprint')!=key:
                raise RuntimeError('Result identity mismatch')
            if previous.get('status') in ('complete','smoke_complete'):
                summaries.append(previous)
                continue
        row = dict(status='running',task=native,label=name,method=request['method'],model=request['model'],
                   fingerprint=key,provenance=identity,benchmark_complete=False)
        save(output,row)
        original_loader = manager.load_task_or_group
        manager.load_task_or_group = lambda names:tree
        start = time.monotonic()
        reset_statistics(statistics)
        try:
            raw = lm_eval.simple_evaluate(model=lm,tasks=[native],task_manager=manager,
                num_fewshot=spec['shots'],limit=request['limit'],use_cache=None,
                log_samples=True,cache_requests=False,apply_chat_template=False,
                predict_only=native=='mbpp',confirm_run_unsafe_code=False,
                random_seed=0,numpy_random_seed=1234,torch_random_seed=1234,fewshot_random_seed=1234)
            samples = samples_for(raw,native)
            documents = document_count(samples)
            expected = sum(min(request['limit'], count) for count in counts.values()) if request['limit'] is not None else spec['count']
            if documents != expected:
                raise RuntimeError(f'{name}: incomplete result, {documents} / {expected}')
            row.update(raw=raw,samples=documents,sample_records=len(samples),elapsed_sec=time.monotonic()-start,
                       runtime_statistics=read_statistics(statistics))
            if native=='mbpp':
                row.update(status='generated_pending_scoring',scoring_required='isolated MBPP scorer')
                row['provenance']['counts'] = counts
                save(output,row)
                # No generated program is executed by this model process.
                print(f'[MBPP generated] {output}; use the separate score-mbpp command',flush=True)
            else:
                metric = task_protocol[native]['metric']
                values = raw['results'].get(native) or raw.get('groups',{}).get(native)
                value = float(values[metric])
                if not math.isfinite(value):
                    raise RuntimeError('Nonfinite benchmark score')
                row.update(status='complete' if request['limit'] is None else 'smoke_complete',
                    metric_name=metric,metric_value=value,score_percent=value*100,
                    benchmark_complete=request['limit'] is None)
                save(output,row)
                print(f'[{row["status"]}] {name}: {value*100:.4f}',flush=True)
            summaries.append(row)
        except BaseException as error:
            row.update(status='failed',error=str(error),elapsed_sec=time.monotonic()-start)
            save(output,row)
            raise
        finally:
            manager.load_task_or_group = original_loader
    print(json.dumps({'tasks_finished':len(summaries),'full_accuracy_complete':
                      len(summaries)==11 and all(r['benchmark_complete'] for r in summaries)},indent=2))
