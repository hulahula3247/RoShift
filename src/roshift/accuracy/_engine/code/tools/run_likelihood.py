"""Full likelihood tasks for one immutable regenerated RoShift profile/GPU.

Example (after the profile PPL audit has completed):
  CUDA_VISIBLE_DEVICES=0 python tools/run_likelihood.py --execute     --model llama31 --model-path /repro/project/models/Meta-Llama-3.1-8B     --work /repro/project/work/llama31 --profile llama31_nonuniform_s50     --tasks piqa boolq --output /repro/project/results/llama31_s50_likelihood

Without --execute, print an offline plan only. No generation, sharding, data
subsampling, or allocation/Q selection is performed by this entrypoint.
"""
from __future__ import annotations
import os as _roshift_os
import argparse
import csv
import fcntl
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
from types import SimpleNamespace
from prepare_remote import KIT, configure_environment, read, sha, verify_model
COUNTS = dict(mmlu=14042, boolq=3270, piqa=1838, arc_easy=2376, arc_challenge=1172, winogrande=1267, commonsense_qa=1221)
MMLU_REVISION = 'c30699e8356da336a370243923dbaf21066bb9fe'
SEEDS = dict(random_seed=0, numpy_random_seed=1234, torch_random_seed=1234, fewshot_random_seed=1234)
CONFIG_KEYS = ('dataset_path', 'dataset_name', 'dataset_kwargs', 'training_split', 'validation_split', 'test_split', 'doc_to_text', 'doc_to_target', 'doc_to_choice', 'target_delimiter', 'fewshot_delimiter', 'fewshot_config', 'output_type', 'process_docs', 'filter_list')

def safe_json(value):
    if isinstance(value, Path):
        return str(value)
    if callable(value):
        return f"{getattr(value, '__module__', '')}.{getattr(value, '__qualname__', type(value).__name__)}"
    if hasattr(value, 'tolist'):
        return value.tolist()
    if hasattr(value, 'item'):
        return value.item()
    if isinstance(value, set):
        return sorted(value)
    return str(value)

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=safe_json, ensure_ascii=False).encode()).hexdigest()

def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, default=safe_json, ensure_ascii=False, allow_nan=False) + '\n')
    tmp.replace(path)

def historical_rows(context, model, allocation, target, tasks):
    path = context / 'evidence/accuracy_inventory_20260916/benchmark_inventory.csv'
    rows = []
    with path.open(encoding='utf-8-sig') as handle:
        for row in csv.DictReader(handle):
            if row['model'] == model and row['task'] in tasks and (row['method'] == 'Dense' or (row['method'] == 'RoShift' and row['allocation'] == allocation and (int(row['target']) == target))):
                rows.append(row)
    return rows

def reference_for(directory, task):
    path = None if directory is None else directory / f'{task}.json'
    if path is None or not path.is_file():
        return ({}, None)
    row = read(path)
    raw = row.get('raw', row.get('output', row))
    if not isinstance(raw, dict) or 'configs' not in raw:
        raise ValueError(f'Historical Dense result lacks task configs: {path}')
    return (raw, dict(path=str(path), sha256=sha(path)))

def settings(reference, task, protocol, batch_override):
    config = reference.get('config', {})
    seeds = {key: config.get(old_key, default) for key, old_key, default in (('random_seed', 'random_seed', 0), ('numpy_random_seed', 'numpy_seed', 1234), ('torch_random_seed', 'torch_seed', 1234), ('fewshot_random_seed', 'fewshot_seed', 1234))}
    batch = batch_override or config.get('batch_size') or protocol['fallback_batch']
    if not str(batch).isdigit() or int(batch) <= 0:
        batch = protocol['fallback_batch']
    options = dict(truncation=False, logits_cache=True, max_length=None, add_bos_token=None, prefix_token_id=None, use_fast_tokenizer=True)
    model_args = config.get('model_args', {})
    if isinstance(model_args, str):
        model_args = dict((part.split('=', 1) for part in model_args.split(',') if '=' in part))
    if isinstance(model_args, dict):
        for key in options:
            if key in model_args:
                value = model_args[key]
                if isinstance(value, str):
                    try:
                        value = json.loads(value.lower())
                    except json.JSONDecodeError:
                        pass
                options[key] = value
    return dict(batch=int(batch), seeds=seeds, hf_options=options, historical_dense_config_available=bool(reference), historical_protocol_identity_verified=False)

def leaf_tasks(tree):
    for name, task in tree.items():
        if isinstance(task, dict):
            yield from leaf_tasks(task)
        else:
            yield (str(name), task)

def task_samples(raw, task):
    groups = raw.get('samples', {})
    if task != 'mmlu':
        return groups.get(task, [])
    return [dict(row, subtask=name) for name, rows in sorted(groups.items()) if name.startswith('mmlu_') for row in rows]

def prompt_audit(samples, reference, task):
    keys = ('doc_id', 'doc_hash', 'prompt_hash', 'target_hash')

    def identity(rows):
        if not rows or any((any((key not in r for key in keys)) for r in rows)):
            return None
        return sorted(((r.get('subtask', task), *(r[key] for key in keys)) for r in rows))
    current = identity(samples)
    previous = identity(task_samples(reference, task))
    return dict(current_identity_sha256=digest(current) if current else None, current_samples=len(samples), historical_identity_available=previous is not None, identical=current == previous if previous is not None else None, note='Absence of historical sample identities is not equality.')

def complete_result(path, fingerprint):
    if not path.is_file():
        return False
    data = read(path)
    if data.get('fingerprint') != fingerprint:
        raise RuntimeError(f'Result fingerprint differs; use a new output directory: {path}')
    if data.get('status') != 'complete':
        return False
    samples = task_samples(data.get('raw', {}), data['task'])
    if len(samples) != COUNTS[data['task']] or not math.isfinite(data['metric_value']):
        raise RuntimeError(f'Invalid complete result: {path}')
    return True

def source_files():
    paths = set((KIT / 'analysis').glob('*.py')) | set((KIT / 'tools').glob('*.py'))
    for module in list(sys.modules.values()):
        file = vars(module).get('__file__') if module is not None else None
        if file and str(file).endswith('.py'):
            path = Path(file).resolve()
            if path.is_relative_to(KIT):
                paths.add(path)
    return {str(p.relative_to(KIT)): sha(p) for p in sorted(paths) if p.is_file()}

def check_ppl(profile):
    result = read(profile / 'ppl_audit.json')
    if result.get('status') != 'complete':
        raise RuntimeError('The complete profile PPL audit is required before likelihood tasks')
    for name in ('mixed_calibration', 'wikitext', 'heldout_mixed'):
        row = result.get(name, {})
        values = [value for key, value in row.items() if key.lower() in ('ppl', 'perplexity')]
        if not values or not all((math.isfinite(float(value)) and float(value) > 0 for value in values)):
            raise RuntimeError(f'Missing/nonfinite PPL audit: {name}')
    return result

def runtime_audit(net, patched):
    """Verify the actual active likelihood implementation, not a prefill flag."""
    import torch
    import run_llama3_mixed_allocation_search_20260815 as allocation
    if type(net.lm_head) is not torch.nn.Linear:
        raise RuntimeError('Likelihood requires the native fully dense Linear LM head')
    if len(patched) != net.config.num_hidden_layers:
        raise RuntimeError('Not every decoder layer is patched')
    for layer in patched:
        if layer.controller.runtime_controller is not None or not layer.controller.enabled():
            raise RuntimeError('Likelihood body sparsity must have no prompt/decode gate')
        for site in ('q', 'k', 'v', 'o'):
            projection = getattr(layer.inner.self_attn, f'{site}_proj')
            if not isinstance(projection, allocation.DynamicInputLinear):
                raise RuntimeError(f'Missing sparse {site} projection')
            if projection.controller is not layer.controller:
                raise RuntimeError('Inconsistent sparse controller')
        if layer.weighted_output_selectors or layer.inner.self_attn.o_proj.scale_cpu is not None:
            raise RuntimeError('Unexpected weighted O/Up/Down selector')
    probe = torch.arange(1, 129, dtype=torch.float32).reshape(1, 1, 128).repeat(2, 3, 1)
    masked = allocation.blockwise_topk(probe, None, 50.0, 128, torch.arange(128))
    zeros = masked.eq(0).sum(-1)
    if not bool(zeros.eq(64).all()):
        raise RuntimeError('Blockwise masking failed full-input-position audit')
    return dict(body_scope='all_input_positions', lm_head='dense_native_linear', runtime_controller=None, sparse_start=None, masked_per_probe_position=zeros.tolist(), decoder_source_sha256=digest(inspect.getsource(allocation.AllocationDecoderLayer.forward)), input_linear_source_sha256=digest(inspect.getsource(allocation.DynamicInputLinear.forward)), padding_note='lm-eval right padding is computed but ignored in likelihood scores; fallback statistics include padded rows')

def run(args):
    configure_environment()
    import numpy as np
    import torch
    import transformers
    import lm_eval
    from lm_eval.models.huggingface import HFLM
    from lm_eval.tasks import TaskManager, get_task_dict
    from materialize_profile import verify_export
    from rental_runtime_20260916 import install
    import run_llama3_layerwise_q_paircd_fullcov_20260830 as paircd
    if int(os.environ.get('WORLD_SIZE', '1')) != 1 or int(os.environ.get('LOCAL_WORLD_SIZE', '1')) != 1:
        raise RuntimeError('Use one ordinary Python process per GPU; distributed launch is disabled')
    if torch.cuda.device_count() != 1:
        raise RuntimeError('Set CUDA_VISIBLE_DEVICES to exactly one GPU')
    torch.set_num_threads(args.cpu_threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(1234)
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output / 'worker.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    common = args.work / args.model / 'common'
    profile = args.work / args.model / 'rebuilt' / args.profile
    marker = read(profile / 'materialized.json')
    info = read(profile / 'profile.json')
    if marker.get('status') != 'complete' or info['model'] != args.model:
        raise RuntimeError('A complete matching regenerated profile is required')
    if info.get('cohort') != 'remote_regenerated_shared_dense_median':
        raise RuntimeError('This runner is restricted to the explicitly regenerated cohort')
    verify_export(profile, marker['files'])
    verify_model(args.model, args.model_path)
    check_ppl(profile)
    meta = read(common / 'common_ready.json')
    if meta['status'] != 'complete':
        raise RuntimeError('Common preparation is incomplete')
    install(common / 'dense_median', meta['model_type'])
    protocol = read(args.context / 'evidence/accuracy_inventory_20260916/task_protocol.json')
    manager = TaskManager()
    pending = []
    base_files = [profile / 'profile.json', profile / 'materialized.json', profile / 'ppl_audit.json', common / 'adaptive_dp_centers.json', common / 'dense_preparation_state.json', KIT / 'inputs' / args.model / 'base_model_identity.json', KIT / 'regeneration.json']
    provenance = dict(cohort=info['cohort'], historical_bitwise_equivalence=False, model=args.model, profile=args.profile, base_model_hashes_verified=True, artifacts={str(p): sha(p) for p in base_files}, protocol_file_sha256=sha(args.context / 'evidence/accuracy_inventory_20260916/task_protocol.json'), environment=dict(python=sys.version, torch=torch.__version__, transformers=transformers.__version__, lm_eval=getattr(lm_eval, '__version__', 'unknown'), cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0), tf32=False, attention_backend='sdpa'), historical_references=historical_rows(args.context, args.model, info['allocation'], info['target'], args.tasks))
    for name in args.tasks:
        reference, reference_file = reference_for(args.dense_results, name)
        setting = settings(reference, name, protocol[name], args.batch_size)
        request = {'task': 'mmlu', 'dataset_kwargs': {'revision': MMLU_REVISION}} if name == 'mmlu' else name
        task_dict = get_task_dict([request], task_manager=manager)
        leaves = dict(leaf_tasks(task_dict))
        if name == 'mmlu' and len(leaves) != 57:
            raise RuntimeError(f'MMLU must contain all 57 subjects, found {len(leaves)}')
        current_config, dataset, document_counts = ({}, {}, {})
        for subtask, task in leaves.items():
            config = task.dump_config()
            previous_config = reference.get('configs', {}).get(subtask, {})
            saved_shots = previous_config.get('num_fewshot')
            if saved_shots is not None and saved_shots != protocol[name]['fewshot']:
                raise RuntimeError(f'{subtask}: historical few-shot count differs from the approved protocol')
            mismatches = {key: dict(current=config.get(key), historical=previous_config[key]) for key in CONFIG_KEYS if key in previous_config and config.get(key) != previous_config[key]}
            if mismatches:
                raise RuntimeError(f'{subtask}: historical task config mismatch: {mismatches}')
            current_config[subtask] = config
            document_counts[subtask] = len(task.eval_docs)
            dataset[subtask] = {split: dict(rows=len(data), fingerprint=getattr(data, '_fingerprint', None)) for split, data in task.dataset.items()}
        if sum(document_counts.values()) != COUNTS[name]:
            raise RuntimeError(f'{name}: expected {COUNTS[name]} documents, found {sum(document_counts.values())}')
        if name == 'mmlu' and sum((splits.get('dev', {}).get('rows', 0) for splits in dataset.values())) != 285:
            raise RuntimeError('MMLU requires all 285 development examples for the fixed 5-shot protocol')
        identity = dict(provenance=provenance, task=name, protocol=protocol[name], setting=setting, task_config=current_config, dataset=dataset, document_counts=document_counts, reference_file=reference_file, sources=source_files())
        fingerprint = digest(identity)
        if complete_result(args.output / f'{name}.json', fingerprint):
            print(f'[skip complete] {name}', flush=True)
            continue
        pending.append((name, task_dict, reference, setting, identity, fingerprint))
    if not pending:
        return
    par = SimpleNamespace(model_id=str(args.model_path), root=args.work / 'inputs' / args.model / 'root', output_dir=profile, adaptive_json=common / 'adaptive_dp_centers.json', device_map='cuda:0', attn_implementation='sdpa', block_size=128, permutation_seed=11, fallback_rel=0.05)
    net, device, patched, medians, adaptive = paircd.load_eval_model(par, np.asarray(info['recipe']))
    net.eval().requires_grad_(False)
    net.config.use_cache = False
    if any((parameter.device != device for parameter in net.parameters())):
        raise RuntimeError('Model is not resident on a single GPU')
    audit = runtime_audit(net, patched)
    write(args.output / 'runtime_audit.json', audit)
    with torch.inference_mode():
        for name, task_dict, reference, setting, identity, fingerprint in pending:
            path = args.output / f'{name}.json'
            row = dict(status='running', task=name, fingerprint=fingerprint, provenance=identity, runtime_audit=audit, historical_comparison='separate regenerated cohort', protocol_warning=None if reference else 'Uses packaged task definitions and recorded benchmark settings.')
            write(path, row)
            started = time.monotonic()
            try:
                adaptive.reset()
                torch.cuda.reset_peak_memory_stats()
                lm = HFLM(pretrained=net, tokenizer=str(args.model_path), dtype='float16', batch_size=setting['batch'], device='cuda:0', parallelize=False, trust_remote_code=False, **setting['hf_options'])
                cache = args.output / 'response_cache' / f'{name}_{fingerprint}'
                cache.parent.mkdir(parents=True, exist_ok=True)
                original_loader = manager.load_task_or_group
                manager.load_task_or_group = lambda names: task_dict
                try:
                    raw = lm_eval.simple_evaluate(model=lm, tasks=[name], task_manager=manager, num_fewshot=protocol[name]['fewshot'], limit=None, use_cache=str(cache), cache_requests=False, log_samples=True, apply_chat_template=False, confirm_run_unsafe_code=False, **setting['seeds'])
                finally:
                    manager.load_task_or_group = original_loader
                count = 0
                for subtask, expected in identity['document_counts'].items():
                    actual = raw['n-samples'][subtask]['effective']
                    if actual != expected or len(raw['samples'][subtask]) != expected:
                        raise RuntimeError(f'Incomplete {subtask}: expected {expected}, found {actual}')
                    count += actual
                samples = task_samples(raw, name)
                if count != COUNTS[name] or len(samples) != COUNTS[name]:
                    raise RuntimeError(f'Incomplete {name}: {count} results/{len(samples)} samples')
                metric = protocol[name]['metric']
                metric_row = raw['results'].get(name) or raw.get('groups', {}).get(name)
                if metric_row is None:
                    raise RuntimeError(f'Missing official task/group aggregation: {name}')
                value = float(metric_row[metric])
                if not math.isfinite(value):
                    raise RuntimeError('Nonfinite likelihood metric')
                row.update(status='complete', raw=raw, samples=count, metric_name=metric, metric_value=value, score_percent=100 * value, prompt_audit=prompt_audit(samples, reference, name), elapsed_sec=time.monotonic() - started, effective_batch=setting['batch'], adaptive_k4=adaptive.stats(), fallback_stats_scope='computed rows including padding; excludes cache hits', peak_vram_gib=torch.cuda.max_memory_allocated() / 2 ** 30, hf_runtime=dict(max_length=lm.max_length, add_bos_token=lm.add_bos_token, tokenizer_class=type(lm.tokenizer).__name__))
                write(path, row)
                print(f'[complete] {name} {metric}={value:.8f} items={count}', flush=True)
            except BaseException as error:
                row.update(status='paused' if isinstance(error, KeyboardInterrupt) else 'failed', error=repr(error), traceback=traceback.format_exc(), elapsed_sec=time.monotonic() - started)
                write(path, row)
                raise

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='llama31')
    parser.add_argument('--model-path', type=Path)
    parser.add_argument('--work', type=Path)
    parser.add_argument('--profile', default='llama31_nonuniform_s50')
    parser.add_argument('--tasks', nargs='+', choices=sorted(COUNTS), default=['piqa', 'boolq'])
    parser.add_argument('--context', type=Path, default=KIT / 'context')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--dense-results', type=Path, help='Optional original Dense task.json files, not inventory CSV')
    parser.add_argument('--batch-size', type=int, help='Explicit override; otherwise recorded Dense batch or protocol fallback')
    parser.add_argument('--cpu-threads', type=int, default=4)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--check-imports', action='store_true')
    args = parser.parse_args()
    if len(set(args.tasks)) != len(args.tasks) or (args.batch_size is not None and args.batch_size <= 0):
        parser.error('Task names must be unique and batch size must be positive')
    if args.check_imports:
        configure_environment()
        import torch, lm_eval, transformers
        from lm_eval.models.huggingface import HFLM
        import run_llama3_layerwise_q_paircd_fullcov_20260830
        print(json.dumps(dict(torch=torch.__version__, transformers=transformers.__version__, lm_eval=str(Path(lm_eval.__file__).resolve()), status='imports_passed')))
        return
    if args.execute:
        if not all((args.model_path, args.work, args.output)):
            parser.error('--execute needs --model-path, --work, and --output')
        run(args)
        return
    protocol = read(args.context / 'evidence/accuracy_inventory_20260916/task_protocol.json')
    print(json.dumps(dict(mode='offline_plan_only', model=args.model, profile=args.profile, tasks={name: dict(**protocol[name], expected_documents=COUNTS[name]) for name in args.tasks}, seeds_when_original_configs_missing=SEEDS, body_scope='all_input_positions', lm_head='dense', requirements=['materialized profile', 'complete finite PPL audit', 'one visible GPU', 'full task datasets', 'no exact historical identity claim without original evidence']), indent=2))
if __name__ == '__main__':
    main()
