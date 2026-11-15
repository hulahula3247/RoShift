"""Official TEAL queue: six models, 50/40/60%, eleven tasks.

The previous worker, runtime and result fingerprints remain unchanged at
40/50%. The 60% path only supplies a separately verified official lookup.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import run_teal_official_suite_20260910 as base

SPARSITIES = (.5, .4, .6)
TASKS = tuple(base.TASKS)
EARLY = tuple(t for t in base.ORDER if t in TASKS and t not in base.GEN)
STAGES = (EARLY, ('mbpp', 'minerva_math500'), ('bbh_cot_fewshot', 'gsm8k'))
PROJS = ('q', 'k', 'v', 'o', 'gate', 'up', 'down')
ORIGINAL_ARTIFACT = base.artifact
STOP = False


def coverage(folder, layers, target):
    rows = []
    for layer in range(layers):
        path = folder/'lookup'/f'layer-{layer}'/'results.csv'
        if not path.is_file():
            return dict(covered=False, reason=f'missing {path}', rows=rows)
        with path.open(newline='') as handle:
            values = list(csv.DictReader(handle))
        if not values:
            return dict(covered=False, reason=f'empty {path}', rows=rows)
        effective = [float(v['Effective Sparsity']) for v in values]
        if any(b < a for a, b in zip(effective, effective[1:])):
            raise ValueError(f'Nonmonotone official lookup: {path}')
        nearest = min(range(len(effective)), key=lambda j: abs(effective[j]-target))
        rows.append(dict(layer=layer, maximum=max(effective), selected=effective[nearest],
                         recipe={p:float(values[nearest][p]) for p in PROJS},
                         lookup=str(path), sha256=base.sha(path)))
    return dict(covered=all(r['maximum'] >= target for r in rows), rows=rows)


def layers_for(model):
    return base.read(Path(base.MODELS[model])/'config.json')['num_hidden_layers']


def target_artifact(root, model, target):
    original = ORIGINAL_ARTIFACT(root, model)
    if target < .6 or coverage(original, layers_for(model), target)['covered']:
        return original
    extended = root/'calibration_s60'/model
    if not (extended/'complete.json').exists() or not coverage(extended, layers_for(model), target)['covered']:
        raise RuntimeError(f'Official greedy extension required before {model} at {target}')
    return extended


def link_directory(source, dest):
    if dest.is_symlink():
        if dest.resolve() != source.resolve():
            raise RuntimeError(f'Unexpected artifact link: {dest}')
    elif dest.exists():
        raise RuntimeError(f'Refusing to replace directory: {dest}')
    else:
        dest.symlink_to(source.resolve(), target_is_directory=True)


def extend_lookup(root, model_name):
    original = ORIGINAL_ARTIFACT(root, model_name)
    n = layers_for(model_name)
    if coverage(original, n, .6)['covered']:
        return
    output = root/'calibration_s60'/model_name
    output.mkdir(parents=True, exist_ok=True)
    link_directory(original/'histograms', output/'histograms')
    before = {str(p.relative_to(original)):base.sha(p)
              for p in (original/'histograms').glob('layer-*/*/histograms.pt')}
    base.dump(output/'provenance.json', dict(model=base.MODELS[model_name],
              algorithm='unchanged upstream TEAL process_layer; no WAS, no Grid-CD',
              original_artifact=str(original), original_histograms_sha256=before,
              target=.65, evaluated_target=.6, base_step_size=.05, last_fraction=.25,
              calibration='same upstream Alpaca calibration, not benchmark inputs',
              frozen_40_50_artifacts=True))
    started = time.monotonic()
    if model_name == 'llama31':
        # This artifact was generated with the pinned old TEAL model API. Use
        # that API and its saved layer inputs, not newly reconstructed inputs.
        sys.path[:0] = [str(base.OFFICIAL/'deps/hf446'), str(base.UPSTREAM),
                       str(base.UPSTREAM/'utils'), str(base.OFFICIAL)]
        import torch
        from official_common import install_sparse_sdpa
        from teal.greedyopt import process_layer
        from teal.model import LlamaSparseForCausalLM
        link_directory(original/'activations', output/'activations')
        torch.set_num_threads(4)
        torch.manual_seed(1234)
        model = LlamaSparseForCausalLM.from_pretrained(base.MODELS[model_name],
                    torch_dtype=torch.float16, device_map='cuda:0', low_cpu_mem_usage=True,
                    attn_implementation='sdpa', histogram_path=str(output/'histograms'))
        model.eval().requires_grad_(False)
        install_sparse_sdpa(model, 'teal')

        class ResidentLayer:
            def __init__(self, layer):
                self.layer = layer
                self.self_attn, self.mlp = layer.self_attn, layer.mlp
            def to(self, device):
                return self
            def __call__(self, *args):
                return self.layer(*args)

        with torch.inference_mode():
            for i, layer in enumerate(model.model.layers):
                marker = output/f'layer-{i}.complete.json'
                if marker.exists():
                    continue
                tick = time.monotonic()
                process_layer(ResidentLayer(layer), 'Llama-3-8B', i, .65, .05, .25, str(output))
                base.dump(marker, dict(status='complete', seconds=time.monotonic()-tick))
    else:
        import torch
        from teal_official_runtime_20260910 import upstream_utils, upstream_greedy, projections
        from teal_official_runtime_20260910 import sparse_functions, NativeGreedyLayer
        utils, greedy = upstream_utils(), upstream_greedy()
        ids_path = original/'input_ids.pt'
        if not ids_path.is_file():
            raise RuntimeError(f'Missing original calibration token IDs: {ids_path}')
        ids = torch.load(ids_path, map_location='cpu', weights_only=True)
        if tuple(ids.shape) != (10, 2048):
            raise ValueError(tuple(ids.shape))
        model = base.load_native(model_name)
        weights = {p:m.weight.numel()/projections(model.model.layers[0])['q'].weight.numel()
                   for p,m in projections(model.model.layers[0]).items()}
        greedy.weight_dict[model_name] = weights
        handles = []
        for i, layer in enumerate(model.model.layers):
            marker = output/f'layer-{i}.complete.json'
            if marker.exists():
                continue
            state = {}
            def pre(module, args, kwargs, state=state):
                state.update(hidden=args[0], kwargs=kwargs)
            def post(module, args, result, i=i, state=state, marker=marker):
                tick = time.monotonic()
                path = output/'activations'/f'act_{i}.pt'
                path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(state['hidden'].cpu(), path)
                fns = sparse_functions(module, output/'histograms'/f'layer-{i}', utils)
                masks = []
                for p, linear in projections(module).items():
                    def mask(_, a, fn=fns[p]):
                        return (fn(a[0]), *a[1:])
                    masks.append(linear.register_forward_pre_hook(mask))
                pre_hooks = module._forward_pre_hooks.copy()
                post_hooks = module._forward_hooks.copy()
                module._forward_pre_hooks.clear()
                module._forward_hooks.clear()
                try:
                    greedy.process_layer(NativeGreedyLayer(module, state['kwargs']), model_name,
                                         i, .65, .05, .25, str(output))
                finally:
                    for h in masks:
                        h.remove()
                    module._forward_pre_hooks.update(pre_hooks)
                    module._forward_hooks.update(post_hooks)
                    del module.self_attn.sparse_fns, module.mlp.sparse_fns
                base.dump(marker, dict(status='complete', seconds=time.monotonic()-tick))
                # The one scratch activation created above is not a shared artifact.
                path.unlink()
                state.clear()
                torch.cuda.empty_cache()
            handles += [layer.register_forward_pre_hook(pre, with_kwargs=True),
                        layer.register_forward_hook(post)]
        try:
            with torch.inference_mode():
                model.model(input_ids=ids.cuda(), use_cache=False)
        finally:
            for h in handles:
                h.remove()
        base.dump(output/'input_provenance.json', dict(input_ids_sha256=base.sha(ids_path), weights=weights))
    audit = coverage(output, n, .6)
    if not audit['covered']:
        raise RuntimeError('Greedy extension did not reach 60% for every layer')
    if before != {str(p.relative_to(original)):base.sha(p)
                  for p in (original/'histograms').glob('layer-*/*/histograms.pt')}:
        raise RuntimeError('Original histograms changed')
    base.dump(output/'complete.json', dict(status='complete', seconds=time.monotonic()-started,
                                          lookup_audit=audit, target=.65))


def active_tasks(root):
    return tuple(t for t in TASKS if t in base.active_tasks(root))


def jobs(root=None):
    for tasks in STAGES:
        for sparsity in SPARSITIES:
            for task in tasks:
                for model in base.MODELS:
                    if root is not None and task not in active_tasks(root):
                        continue
                    yield model, sparsity, task


def complete(path):
    return path.is_file() and base.read(path).get('status') == 'complete'


def report(root):
    records = {}
    diagnosed = 0
    tasks = active_tasks(root)
    dense_records = base.read(base.DENSE)['results']
    lines = [f'# Official TEAL: {len(tasks)} benchmarks, 40/50/60%', '',
             'Official greedy allocation and fixed histogram thresholds. No RoShift/WAS/Grid-CD.',
             'Likelihood: all input tokens sparse. Generation: prompt dense, cached decode sparse.',
             'LM-head dense. Qwen/Gemma use native architecture ports, not upstream-supported implementations.', '']
    for model in base.MODELS:
        lines += [f'## {model}', '', '| Method | '+' | '.join(base.TASKS[t][0] for t in tasks)+' | Avg |',
                  '|---|'+'---:|'*(len(tasks)+1)]
        dense = {t:dense_records[model][t]['score'] for t in tasks}
        lines += ['| Dense | '+' | '.join(f'{v:.2f}' for v in dense.values())+f' | {sum(dense.values())/len(tasks):.2f} |']
        for sparsity in SPARSITIES:
            cells, scores = [], []
            for task in tasks:
                path = base.result_path(root, model, sparsity, task)
                row = base.read(path) if path.exists() else dict(status='not_started')
                records[f'{model}/s{round(sparsity*100)}/{task}'] = {k:v for k,v in row.items() if k != 'raw'}
                diagnosed += int(complete(base.result_path(root, model, sparsity, task, True)))
                if row['status'] == 'complete':
                    value = row['metric_value']*100
                    cells.append(f'{value:.2f} ({value-dense[task]:+.2f})')
                    scores.append(value)
                else:
                    cells.append('-')
            avg = (f'{sum(scores)/len(scores):.2f} ({(sum(scores)-sum(dense.values()))/len(scores):+.2f})'
                   if len(scores)==len(tasks) else '-')
            lines.append(f'| TEAL {sparsity:.0%} | '+' | '.join(cells)+f' | {avg} |')
        lines.append('')
    value = dict(completed=sum(v['status']=='complete' for v in records.values()),
                 expected=len(records), diagnostics_complete=diagnosed, results=records,
                 active_tasks=list(tasks), sparsities=list(SPARSITIES),
                 excluded_tasks=[t for t in base.TASKS if t not in tasks])
    base.dump(root/'results_11tasks_s40_s50_s60.json', value)
    (root/'REPORT_11tasks_s40_s50_s60.md').write_text('\n'.join(lines)+'\n')
    return value


def prepare(root):
    base.prepare(root)
    audit = {m:coverage(ORIGINAL_ARTIFACT(root,m), layers_for(m), .6) for m in base.MODELS}
    value = dict(models=list(base.MODELS), sparsities=list(SPARSITIES), tasks=list(active_tasks(root)),
                 job_order=list(jobs(root)), upstream_commit=base.read(root/'manifest.json')['commit'],
                 allocation='official MSE greedy only; no WAS or Grid-CD',
                 original_accuracy_manifest_sha256=base.sha(root/'manifest.json'),
                 runtime_sha256=base.sha(base.REPO/'analysis/teal_official_runtime_20260910.py'),
                 controller_sha256=base.sha(Path(__file__)), coverage_s60=audit,
                 completed_results='reuse original 40/50 results and response-cache fingerprints',
                 calibration_s60='reuse original histograms and inputs; official greedy to .65 only if needed')
    base.dump(root/'resume_20260912_manifest.json', value)
    return report(root)


def run_worker(args):
    folder = target_artifact(args.root, args.model, args.sparsity)
    if args.sparsity >= .6:
        base.dump(args.root/'lookup_audits_s60'/f'{args.model}.json', coverage(folder,layers_for(args.model),.6))
        base.artifact = lambda root, model: folder
    access = base.configure_dataset_access(args.task)
    if access:
        base.dump(args.root/'dataset_access_mmlu.json', access)
    base.worker(args)


def suite(args):
    global STOP
    root = args.root
    with (root/'suite.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        prepare(root)
        def stop(*_):
            global STOP
            STOP = True
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        def launch(phase, model, sparsity, task=None, diagnostic=False, batch=1):
            if STOP:
                raise KeyboardInterrupt
            tag = f'{model}_s{round(sparsity*100)}_{task or phase}'+('_actual' if diagnostic else '')
            path = root/'logs'/f'{tag}_resume_b{batch}.log'
            path.parent.mkdir(parents=True, exist_ok=True)
            cmd = [sys.executable, '-u', str(Path(__file__).resolve()), '--phase', phase,
                   '--root', str(root), '--model', model, '--sparsity', str(sparsity), '--batch', str(batch)]
            if task: cmd += ['--task', task]
            if diagnostic: cmd += ['--diagnostic']
            gpu = subprocess.check_output(['/usr/lib/wsl/lib/nvidia-smi', '--query-compute-apps=pid',
                                          '--format=csv,noheader'], text=True).strip()
            if gpu:
                raise RuntimeError(f'Another GPU process is active: {gpu}')
            with path.open('a') as handle:
                child = subprocess.Popen(cmd, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
                base.dump(root/'progress.json', dict(status='running', controller_pid=os.getpid(),
                          worker_pid=child.pid, phase=phase, model=model, sparsity=sparsity,
                          task=task, diagnostic=diagnostic, batch=batch, log=str(path), started=time.time()))
                while True:
                    if STOP:
                        os.killpg(child.pid, signal.SIGINT)
                        try: child.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            os.killpg(child.pid, signal.SIGTERM)
                            try: child.wait(timeout=15)
                            except subprocess.TimeoutExpired:
                                os.killpg(child.pid, signal.SIGKILL)
                                child.wait()
                        raise KeyboardInterrupt
                    try:
                        return child.wait(timeout=1), path
                    except subprocess.TimeoutExpired:
                        pass
        failed_models = set()
        try:
            for model, sparsity, task in jobs(root):
                if STOP: raise KeyboardInterrupt
                if (model, sparsity) in failed_models: continue
                original = ORIGINAL_ARTIFACT(root,model)
                if model not in base.RELEASED and not (original/'complete.json').exists():
                    code,path = launch('calibrate',model,sparsity)
                    if code:
                        failed_models.update((model,s) for s in SPARSITIES)
                        base.dump(root/f'{model}.calibration_failed.json', dict(code=code,log=str(path)))
                        continue
                if sparsity >= .6:
                    try: target_artifact(root,model,sparsity)
                    except RuntimeError:
                        code,path = launch('extend',model,sparsity)
                        if code:
                            failed_models.add((model,sparsity))
                            base.dump(root/f'{model}.s60_failed.json', dict(code=code,log=str(path)))
                            continue
                for diagnostic in (False,True):
                    out = base.result_path(root,model,sparsity,task,diagnostic)
                    if complete(out): continue
                    batch = base.batch_for(model,task)
                    accuracy = base.result_path(root,model,sparsity,task)
                    if diagnostic and complete(accuracy):
                        batch = base.read(accuracy)['batch_size']
                    while True:
                        code,path = launch('worker',model,sparsity,task,diagnostic,batch)
                        if code == 0 or complete(out): break
                        with path.open('rb') as handle:
                            handle.seek(max(0,path.stat().st_size-16000))
                            tail=handle.read().decode(errors='replace')
                        if batch > 1 and ('CUDA out of memory' in tail or 'OutOfMemoryError' in tail):
                            batch = max(1,batch//2)
                            base.log(f'OOM retry {model}/{task}: batch={batch}')
                            continue
                        row = base.read(out) if out.exists() else {}
                        row.update(status='failed', worker_exit_code=code, log=str(path))
                        row.setdefault('error', tail[-4000:])
                        base.dump(out,row)
                        break
                    report(root)
        except KeyboardInterrupt:
            base.dump(root/'progress.json', dict(status='paused_by_user', time=time.time()))
            report(root)
            return
        value = report(root)
        done = value['completed'] == value['expected'] == value['diagnostics_complete']
        base.dump(root/'progress.json', dict(status='complete' if done else 'partial', finished=time.time(),
                  completed=value['completed'], expected=value['expected'],diagnostics_complete=value['diagnostics_complete']))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=base.ROOT)
    p.add_argument('--phase', choices=('suite','prepare','report','worker','extend','calibrate'), default='suite')
    p.add_argument('--model', choices=base.MODELS, default='llama31')
    p.add_argument('--sparsity', type=float, choices=SPARSITIES, default=.5)
    p.add_argument('--task', choices=TASKS, default='mmlu')
    p.add_argument('--batch', type=int, default=4)
    p.add_argument('--diagnostic', action='store_true')
    p.add_argument('--limit', type=int)
    args = p.parse_args()
    if args.phase == 'worker' and args.task not in active_tasks(args.root):
        base.log(f'DEFERRED {args.model}/{args.task}: queue_policy.json; no model loaded')
        return
    if args.phase == 'suite': suite(args)
    elif args.phase == 'prepare': prepare(args.root)
    elif args.phase == 'report': report(args.root)
    elif args.phase == 'worker': run_worker(args)
    elif args.phase == 'extend': extend_lookup(args.root,args.model)
    elif args.phase == 'calibrate': base.calibrate(args)


if __name__ == '__main__':
    main()
