"""Pinned official TEAL calibration and full likelihood tasks; dry-run by default.

One model and one task per process/GPU. No benchmark documents are sharded or
sampled. Generation runtime scope exists but this entry point intentionally
handles only likelihood until saved Dense generation configs are available.
"""
from __future__ import annotations
import os as _roshift_os
import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
import shutil
from pathlib import Path
import subprocess
import sys
import time
ROOT = Path(_roshift_os.path.join(_roshift_os.environ['ROSHIFT_ENGINE_ROOT'], ''))
CODE = ROOT / 'code'
UPSTREAM = ROOT / 'baselines/teal'
COMMIT = 'fb7373c93ac3594817c9ee64d4e08b47430a1822'
REFERENCE = CODE / 'context/source_reference/analysis/teal_official_runtime_20260910.py'
TASK_PROTOCOL = CODE / 'context/evidence/accuracy_inventory_20260916/task_protocol.json'
MODELS = {'llama31': 'Meta-Llama-3.1-8B', 'qwen25': 'Qwen2.5-7B', 'llama2': 'NousResearch-Llama-2-7b-hf', 'mistral': 'Mistral-7B-v0.1', 'qwen3': 'Qwen3-8B-Base', 'gemma1': 'gemma-7b'}
NATIVE_PORTS = ('qwen25', 'qwen3', 'gemma1')
RELEASED = {'llama2': 'Llama-2-7B', 'mistral': 'Mistral-7B'}
GREEDY_LIMIT = 0.65
COUNTS = {'mmlu': 14042, 'piqa': 1838, 'boolq': 3270, 'arc_easy': 2376, 'arc_challenge': 1172, 'winogrande': 1267, 'commonsense_qa': 1221}
sys.path.insert(0, str(CODE / 'analysis'))
from experiment_timing import Attempt, FILENAME as TIMING_FILE, job as timed_job, reused

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()

def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()

def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, default=str, allow_nan=False) + '\n')
    temporary.replace(path)

def sources():
    observed = subprocess.check_output(['git', '-C', str(UPSTREAM), 'rev-parse', 'HEAD'], text=True).strip()
    if observed != COMMIT:
        raise RuntimeError(f'Wrong TEAL checkout: {observed}')
    names = ('utils/utils.py', 'teal/greedyopt.py', 'teal/grab_acts.py', 'utils/data.py')
    for name in names:
        committed = subprocess.check_output(['git', '-C', str(UPSTREAM), 'show', f'{COMMIT}:{name}'])
        if hashlib.sha256(committed).hexdigest() != sha(UPSTREAM / name):
            raise RuntimeError(f'Pinned TEAL source changed: {name}')
    files = [UPSTREAM / name for name in names] + [REFERENCE, Path(__file__), Path(__file__).with_name('runtime.py'), CODE / 'analysis/runtime_token_scope.py', TASK_PROTOCOL, CODE / 'tools/run_likelihood.py', CODE / 'analysis/experiment_timing.py']
    for model in ('qwen2', 'qwen3', 'gemma'):
        files.extend((CODE / f'vendor/transformers/models/{model}').glob('*.py'))
    return {'upstream_commit': COMMIT, 'files': {str(p): sha(p) for p in files}}

def verify_source_identity(path):
    expected = read(path)
    if expected.get('status') != 'verified':
        raise RuntimeError('TEAL source identity is not verified')
    if expected['core_manifest_sha256'] != sha(CODE / 'MANIFEST.json'):
        raise RuntimeError('Core source manifest changed after TEAL job creation')
    if expected['source_bundle_sha256'] != fingerprint(sources()):
        raise RuntimeError('TEAL source bundle changed after job creation')

def artifact_files(path):
    return {str(p.relative_to(path)): {'bytes': p.stat().st_size, 'sha256': sha(p)} for folder in ('histograms', 'lookup') for p in sorted((path / folder).rglob('*')) if p.is_file()}

def lookup_coverage(path):
    result = {}
    for file in sorted((path / 'lookup').glob('layer-*/results.csv')):
        with file.open() as handle:
            values = [float(row['Effective Sparsity']) for row in csv.DictReader(handle)]
        if not values:
            raise RuntimeError(f'Empty TEAL lookup: {file}')
        result[file.parent.name] = {'minimum': min(values), 'maximum': max(values), 'rows': len(values)}
    return result

def verify_artifacts(path, model=None, sparsity=None):
    complete = read(path / 'complete.json')
    manifest = read(path / 'asset_manifest.json')
    identity = read(path / 'identity.json')
    if manifest['identity_sha256'] != fingerprint(identity) or (model is not None and identity['model'] != model):
        raise RuntimeError('TEAL calibration identity/model mismatch')
    if complete.get('status') != 'complete' or manifest['files'] != artifact_files(path):
        raise RuntimeError('TEAL histogram/lookup manifest mismatch or incomplete calibration')
    for layer in range(complete['layers']):
        for name in (f'histograms/layer-{layer}/self_attn/histograms.pt', f'histograms/layer-{layer}/mlp/histograms.pt', f'lookup/layer-{layer}/results.csv'):
            if name not in manifest['files']:
                raise RuntimeError(f'Missing TEAL artifact: {name}')
    if sparsity is not None:
        coverage = lookup_coverage(path)
        if len(coverage) != complete['layers'] or any((row['maximum'] < sparsity for row in coverage.values())):
            raise RuntimeError(f'TEAL lookup does not cover requested sparsity {sparsity}')
    return manifest

def plan(args):
    complete = (args.artifact / 'asset_manifest.json').is_file()
    return dict(model=args.model, model_path=str(args.model_path), phase=args.phase, artifact=str(args.artifact), artifact_manifest_present=complete, sparsity=args.sparsity, task=args.task, batch=args.batch, source=sources(), calibration_required=not complete and args.model not in RELEASED, released_assets=str(UPSTREAM / 'models' / RELEASED[args.model]) if args.model in RELEASED else None, calibration='published official histogram/lookup' if args.model in RELEASED else 'official Alpaca first 300 ordered examples; 10 x 2048 saved token IDs', calibration_scope='unchanged upstream last-half sparse during greedy', evaluation_scope='all likelihood input positions sparse; fully dense LM head', generation_scope='all prompt positions dense, every populated-cache decode sparse, including one-token prompts', new_calibration_cohort=args.model not in RELEASED, historical_threshold_equivalence_claim=False, parallelism='single CUDA GPU; no TP/PP and no item sharding', supported_sparsities=[0.4, 0.5, 0.6], max_new_greedy_sparsity=GREEDY_LIMIT, native_architecture_port=args.model in NATIVE_PORTS, generation_entrypoint_ready=False, timing_definition='single-GPU wall time', warning='Use model-specific histograms and lookups.')

def environment():
    for entry in (CODE / 'tools', CODE / 'analysis', CODE / 'vendor', CODE):
        sys.path.insert(0, str(entry))
    os.environ.setdefault('HF_HOME', str(ROOT / 'hf_cache'))
    os.environ.setdefault('HF_DEACTIVATE_ASYNC_LOAD', '1')
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    os.environ.setdefault('OMP_NUM_THREADS', '4')
    os.environ.setdefault('MKL_NUM_THREADS', '4')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import torch
    from rental_conditions_20260917 import require_single_gpu
    require_single_gpu(torch)
    if any((int(os.environ.get(name, '1')) != 1 for name in ('WORLD_SIZE', 'LOCAL_WORLD_SIZE'))):
        raise RuntimeError('Distributed model execution is prohibited')
    torch.set_num_threads(4)
    torch.manual_seed(1234)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return torch

def load_model(args):
    import torch
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(str(args.model_path), torch_dtype=torch.float16, device_map='cuda:0', attn_implementation='sdpa').eval().requires_grad_(False)

def released_assets(args):
    """Copy and fingerprint the pinned published lookup, without CUDA/model IO."""
    if args.model not in RELEASED:
        raise ValueError('This model has no compatible published TEAL assets; run calibration')
    source = UPSTREAM / 'models' / RELEASED[args.model]
    if args.artifact.resolve().is_relative_to(UPSTREAM.resolve()) or args.artifact.resolve().is_relative_to(CODE.resolve()):
        raise ValueError('Write assets outside the immutable upstream/core directories')
    source_record = sources()
    relative = str(source.relative_to(UPSTREAM))
    subprocess.run(['git', '-C', str(UPSTREAM), 'diff', '--exit-code', COMMIT, '--', relative], check=True, stdout=subprocess.DEVNULL)
    tracked = subprocess.check_output(['git', '-C', str(UPSTREAM), 'ls-tree', '-r', '--name-only', COMMIT, '--', relative], text=True).splitlines()
    source_files = artifact_files(source)
    expected = {str(Path(p).relative_to(relative)) for p in tracked if Path(p).relative_to(relative).parts[0] in ('histograms', 'lookup')}
    if expected != set(source_files):
        raise RuntimeError('Published TEAL asset tree contains missing or extra files')
    identity = dict(model=args.model, asset_kind='published_official', upstream_commit=COMMIT, published_model=RELEASED[args.model], published_files_sha256=fingerprint(source_files))
    args.artifact.mkdir(parents=True, exist_ok=True)
    with (args.artifact / 'worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (args.artifact / 'identity.json').exists() and read(args.artifact / 'identity.json') != identity:
            raise RuntimeError('Existing TEAL artifacts have another identity; use a new directory')
        dump(args.artifact / 'identity.json', identity)
        if (args.artifact / 'asset_manifest.json').exists():
            verify_artifacts(args.artifact, args.model, args.sparsity)
            return
        for folder in ('histograms', 'lookup'):
            shutil.copytree(source / folder, args.artifact / folder, dirs_exist_ok=True)
        coverage = lookup_coverage(args.artifact)
        layers = read(CODE / 'models.json')[args.model]['layers']
        if len(coverage) != layers or artifact_files(args.artifact) != source_files:
            raise RuntimeError('Released TEAL asset copy verification failed')
        dataset = dict(origin='published official TEAL assets', underlying_token_ids_independently_verified=False)
        dump(args.artifact / 'complete.json', dict(status='complete', layers=layers, target=min((row['maximum'] for row in coverage.values())), calibration=dataset))
        dump(args.artifact / 'asset_manifest.json', dict(status='complete', identity_sha256=fingerprint(identity), files=source_files, dataset=dataset, sources=source_record, lookup_coverage=coverage, cohort='published_official_TEAL_assets_common_token_scope', historical_threshold_equivalence_claim=False))
        verify_artifacts(args.artifact, args.model, args.sparsity)

def calibrate(args):
    if args.model in RELEASED:
        return released_assets(args)
    torch = environment()
    from datasets import load_dataset
    from huggingface_hub import HfApi
    from transformers import AutoTokenizer
    from prepare_remote import verify_model
    from runtime import calibrate_native, native_architecture_audit
    verify_model(args.model, args.model_path)
    args.artifact.mkdir(parents=True, exist_ok=True)
    lock = (args.artifact / 'worker.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    identity = dict(model=args.model, model_path=str(args.model_path.resolve()), sources=sources(), checkpoint_identity_sha256=sha(CODE / 'inputs' / args.model / 'base_model_identity.json'), calibration='official_first300_alpaca_train_10x2048', max_greedy_sparsity=GREEDY_LIMIT, orchestration_change='extend unchanged greedy trajectory from legacy .55 cutoff to .65; earlier rows unchanged')
    identity_file = args.artifact / 'identity.json'
    if identity_file.exists() and read(identity_file) != identity:
        raise RuntimeError('Calibration identity changed; use a new artifact directory')
    dump(identity_file, identity)
    if (args.artifact / 'asset_manifest.json').exists():
        verify_artifacts(args.artifact, args.model, args.sparsity)
        reused(args.artifact / TIMING_FILE, model=args.model, method='TEAL_official', category='baseline_allocation', substage='frozen_calibration', label=args.model + ':frozen_calibration', targets=[40, 50, 60])
        print('[reuse] verified frozen TEAL calibration', flush=True)
        return
    ids_path, dataset_path = (args.artifact / 'input_ids.pt', args.artifact / 'dataset.json')
    if ids_path.exists() and dataset_path.exists():
        metadata = read(dataset_path)
        if metadata['file_sha256'] != sha(ids_path):
            raise RuntimeError('Saved TEAL calibration tokens changed')
        ids = torch.load(ids_path, map_location='cpu', weights_only=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(str(args.model_path), use_fast=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
        revision = HfApi().dataset_info('tatsu-lab/alpaca').sha
        records = list(load_dataset('tatsu-lab/alpaca', revision=revision, split='train', streaming=True).take(300))
        if len(records) != 300:
            raise RuntimeError('Official calibration requires the first 300 Alpaca examples')
        text = ''.join((row['text'] + '\n\n' for row in records))
        encoded = tokenizer(text, truncation=True, max_length=2048, return_overflowing_tokens=True, padding='max_length', return_tensors='pt')
        ids = encoded['input_ids'][:10]
        torch.save(ids, ids_path)
        metadata = dict(dataset='tatsu-lab/alpaca', revision=revision, split='train', sampling='first 300; no shuffle', text_sha256=hashlib.sha256(text.encode()).hexdigest(), shape=list(ids.shape), token_ids_sha256=fingerprint(ids.tolist()), file_sha256=sha(ids_path), tokenizer_identity=read(CODE / 'regeneration.json')['models'][args.model]['tokenizer_sha256'])
        dump(dataset_path, metadata)
    if tuple(ids.shape) != (10, 2048):
        raise RuntimeError(f'Unexpected calibration shape: {tuple(ids.shape)}')
    common = dict(model=args.model, method='TEAL_official', targets=[40, 50, 60], gpu_reserved=True)
    with Attempt(args.artifact / TIMING_FILE, **common, category='baseline_preparation', substage='model_load', label='model_load'):
        model = load_model(args)
    with Attempt(args.artifact / TIMING_FILE, **common, category='baseline_audit', substage='zero_threshold', label='zero_threshold'):
        native_audit = native_architecture_audit(model, ids[:1, :32].to('cuda:0'))
    dump(args.artifact / 'native_architecture_audit.json', native_audit)
    calibrate_native(model, args.model, args.artifact, ids.to('cuda:0'), greedy_limit=GREEDY_LIMIT, timing_log=args.artifact / TIMING_FILE)
    manifest = dict(status='complete', identity_sha256=fingerprint(identity), dataset=metadata, files=artifact_files(args.artifact), lookup_coverage=lookup_coverage(args.artifact), cohort='remote_recalibrated_official_TEAL_s40_s50_s60_common_token_scope', native_architecture_port=args.model in NATIVE_PORTS, native_architecture_audit=native_audit, historical_threshold_equivalence_claim=False)
    dump(args.artifact / 'asset_manifest.json', manifest)
    verify_artifacts(args.artifact, args.model, args.sparsity)

def task_bundle(name, manager, loader, historical, protocol):
    """Reuse the frozen runner's leaf structure/revision; keep official grouping."""
    from run_likelihood import CONFIG_KEYS, MMLU_REVISION, leaf_tasks
    request = {'task': 'mmlu', 'dataset_kwargs': {'revision': MMLU_REVISION}} if name == 'mmlu' else name
    tree = loader([request], task_manager=manager)
    leaves = dict(leaf_tasks(tree))
    if name == 'mmlu' and len(leaves) != 57:
        raise RuntimeError(f'MMLU requires 57 subjects; got {len(leaves)}')
    configs, datasets, counts = ({}, {}, {})
    for subtask, task in leaves.items():
        config = task.dump_config()
        previous = historical.get('configs', {}).get(subtask, {})
        mismatch = {key: {'current': config.get(key), 'historical': previous[key]} for key in CONFIG_KEYS if key in previous and config.get(key) != previous[key]}
        if mismatch or previous.get('num_fewshot', protocol['fewshot']) != protocol['fewshot']:
            raise RuntimeError(f'{subtask}: historical task config differs: {mismatch}')
        configs[subtask] = config
        counts[subtask] = len(task.eval_docs)
        datasets[subtask] = {split: {'rows': len(data), 'fingerprint': getattr(data, '_fingerprint', None)} for split, data in task.dataset.items()}
    if sum(counts.values()) != COUNTS[name]:
        raise RuntimeError(f'{name}: expected {COUNTS[name]} documents, found {sum(counts.values())}')
    if name == 'mmlu' and sum((row.get('dev', {}).get('rows', 0) for row in datasets.values())) != 285:
        raise RuntimeError('MMLU requires 285 development examples for fixed 5-shot evaluation')
    return (tree, dict(task_config=configs, dataset=datasets, document_counts=counts, revision=MMLU_REVISION if name == 'mmlu' else None))

def checked_task_result(raw, name, document_counts, metric):
    from run_likelihood import task_samples
    for subtask, expected in document_counts.items():
        if raw['n-samples'][subtask]['effective'] != expected or len(raw['samples'][subtask]) != expected:
            raise RuntimeError(f'Incomplete {subtask}: expected {expected} scored samples')
    samples = task_samples(raw, name)
    if len(samples) != COUNTS[name]:
        raise RuntimeError(f'Incomplete {name}: {len(samples)} scored samples')
    row = raw['results'].get(name) or raw.get('groups', {}).get(name)
    if row is None or not math.isfinite(float(row[metric])):
        raise RuntimeError(f'Missing/nonfinite official task/group metric: {name}')
    return (samples, float(row[metric]))

def audit(args):
    torch = environment()
    from prepare_remote import verify_model
    from runtime import TealRuntime, native_architecture_audit
    manifest = verify_artifacts(args.artifact, args.model, args.sparsity)
    verify_model(args.model, args.model_path)
    args.output.mkdir(parents=True, exist_ok=True)
    identity = dict(model=args.model, sparsity=args.sparsity, sources=sources(), manifest_sha256=fingerprint(manifest))
    key = fingerprint(identity)
    result = args.output / 'audit.json'
    if result.exists():
        previous = read(result)
        if previous.get('fingerprint') != key:
            raise RuntimeError('Audit provenance changed; use new output directory')
        if previous.get('status') == 'complete':
            reused(args.output / TIMING_FILE, model=args.model, method='TEAL_official', category='baseline_audit', substage='small_fixed_input', label='small_fixed_input', targets=[round(args.sparsity * 100)])
            return
    common = dict(model=args.model, method='TEAL_official', targets=[round(args.sparsity * 100)], gpu_reserved=True)
    with Attempt(args.output / TIMING_FILE, **common, category='baseline_preparation', substage='model_load', label='model_load'):
        model = load_model(args)
    ids = torch.load(CODE / 'inputs' / args.model / 'root/mixed_calibration/input_ids.pt', map_location='cpu', weights_only=True)[:1, :32].to('cuda:0')
    with Attempt(args.output / TIMING_FILE, **common, category='baseline_audit', substage='small_fixed_input', label='small_fixed_input'):
        native = native_architecture_audit(model, ids)
        runtime = TealRuntime(model, args.artifact, args.sparsity)
        runtime.mode = 'likelihood'
        with torch.inference_mode():
            logits = model(input_ids=ids, use_cache=False).logits
        if not bool(torch.isfinite(logits).all()):
            raise RuntimeError('Nonfinite TEAL sparse fixed-input audit')
        thresholds = [{key: value for key, value in row.items() if key != 'counts'} for row in runtime.rows]
    dump(result, dict(status='complete', fingerprint=key, provenance=identity, native_architecture_audit=native, sparse_logits_finite=True, logit_shape=list(logits.shape), thresholds_sha256=fingerprint(thresholds), tokens=32, benchmark_accuracy_validated=False, cohort=manifest['cohort']))

def likelihood(args):
    torch = environment()
    import lm_eval
    import transformers
    from lm_eval.models.huggingface import HFLM
    from lm_eval.tasks import TaskManager, get_task_dict
    from prepare_remote import verify_model
    from run_likelihood import prompt_audit, reference_for, settings, write
    from runtime import TealRuntime, native_architecture_audit
    manifest = verify_artifacts(args.artifact, args.model, args.sparsity)
    verify_model(args.model, args.model_path)
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output / 'worker.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    protocol = read(TASK_PROTOCOL)[args.task]
    historical, historical_file = reference_for(args.dense_results, args.task)
    setting = settings(historical, args.task, protocol, args.batch)
    manager = TaskManager()
    tree, bundle = task_bundle(args.task, manager, get_task_dict, historical, protocol)
    identity = dict(model=args.model, source=sources(), calibration_manifest_sha256=fingerprint(manifest), task=args.task, **bundle, protocol=protocol, settings=setting, historical=historical_file, sparsity=args.sparsity, dtype='float16', attention='sdpa', tf32=False, cohort=manifest['cohort'], scope='all_input_positions_sparse', lm_head='dense', native_architecture_port=args.model in NATIVE_PORTS, historical_protocol_identity_verified=False, environment={'torch': torch.__version__, 'transformers': transformers.__version__, 'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0)})
    key = fingerprint(identity)
    result_path = args.output / f'{args.task}.json'
    if result_path.exists():
        old = read(result_path)
        if old['fingerprint'] != key:
            raise RuntimeError('Existing result has a different fingerprint; use new output')
        if old['status'] == 'complete' and old['samples'] == COUNTS[args.task]:
            checked_task_result(old['raw'], args.task, bundle['document_counts'], protocol['metric'])
            reused(args.output / TIMING_FILE, model=args.model, method='TEAL_official', category='benchmark', substage=args.task, label=args.task, targets=[round(args.sparsity * 100)])
            print('[reuse] complete full-task result', flush=True)
            return
    row = dict(status='running', fingerprint=key, provenance=identity, task=args.task, started=time.time(), protocol_warning=None if historical else 'Uses packaged task definitions and recorded benchmark settings.')
    write(result_path, row)
    timing = dict(model=args.model, method='TEAL_official', targets=[round(args.sparsity * 100)], gpu_reserved=True)
    try:
        with Attempt(args.output / TIMING_FILE, **timing, category='baseline_preparation', substage='model_load', label='model_load'):
            model = load_model(args)
        model.config.use_cache = False
        ids = torch.load(CODE / 'inputs' / args.model / 'root/mixed_calibration/input_ids.pt', map_location='cpu', weights_only=True)[:1, :32]
        with Attempt(args.output / TIMING_FILE, **timing, category='baseline_audit', substage='zero_threshold', label='zero_threshold'):
            row['native_architecture_audit'] = native_architecture_audit(model, ids.to('cuda:0'))
        runtime = TealRuntime(model, args.artifact, args.sparsity)
        runtime.mode = 'likelihood'
        thresholds = [{k: v for k, v in entry.items() if k != 'counts'} for entry in runtime.rows]
        write(args.output / 'frozen_thresholds.json', {'rows': thresholds, 'sha256': fingerprint(thresholds)})
        lm = HFLM(pretrained=model, tokenizer=str(args.model_path), dtype='float16', batch_size=setting['batch'], device='cuda:0', parallelize=False, trust_remote_code=False, **setting['hf_options'])
        (args.output / 'response_cache').mkdir(exist_ok=True)
        with torch.inference_mode(), Attempt(args.output / TIMING_FILE, **timing, category='benchmark', substage=args.task, label=args.task):
            original_loader = manager.load_task_or_group
            manager.load_task_or_group = lambda names: tree
            try:
                raw = lm_eval.simple_evaluate(model=lm, tasks=[args.task], num_fewshot=protocol['fewshot'], task_manager=manager, apply_chat_template=False, log_samples=True, limit=None, cache_requests=False, use_cache=str(args.output / 'response_cache' / key), **setting['seeds'])
            finally:
                manager.load_task_or_group = original_loader
        samples, metric = checked_task_result(raw, args.task, bundle['document_counts'], protocol['metric'])
        row.update(status='complete', samples=len(samples), metric_name=protocol['metric'], metric_value=metric, raw=raw, prompt_audit=prompt_audit(samples, historical, args.task), elapsed_sec=time.time() - row['started'], thresholds_sha256=fingerprint(thresholds))
        write(result_path, row)
    except BaseException as exc:
        row.update(status='failed', error=repr(exc), elapsed_sec=time.time() - row['started'])
        write(result_path, row)
        raise

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['plan', 'assets', 'calibrate', 'audit', 'likelihood'], default='plan')
    parser.add_argument('--model', choices=MODELS, required=True)
    parser.add_argument('--model-path', type=Path)
    parser.add_argument('--artifact', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--dense-results', type=Path)
    parser.add_argument('--task', choices=COUNTS, default='mmlu')
    parser.add_argument('--sparsity', choices=[0.4, 0.5, 0.6], type=float, default=0.6)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--expected-source-identity', type=Path, help='Reject source changes against scheduler job identity before GPU loading')
    args = parser.parse_args()
    args.model_path = args.model_path or ROOT / 'models' / MODELS[args.model]
    args.artifact = args.artifact or ROOT / 'artifacts/teal' / args.model
    args.output = args.output or ROOT / 'results/teal' / args.model / f's{round(100 * args.sparsity)}' / ('audit' if args.phase == 'audit' else args.task)
    if args.expected_source_identity:
        verify_source_identity(args.expected_source_identity)
    print(json.dumps(plan(args), indent=2), flush=True)
    if args.execute and args.phase != 'plan':
        directory = args.output if args.phase in ('likelihood', 'audit') else args.artifact
        gpu = args.phase not in ('assets',) and (not (args.phase == 'calibrate' and args.model in RELEASED))
        targets = [40, 50, 60] if args.phase in ('assets', 'calibrate') else [round(args.sparsity * 100)]
        with timed_job(directory / TIMING_FILE, model=args.model, method='TEAL_official', category='job', substage=args.phase, label='TEAL_official:' + args.phase, targets=targets, gpu_reserved=gpu):
            {'assets': released_assets, 'calibrate': calibrate, 'audit': audit, 'likelihood': likelihood}[args.phase](args)
if __name__ == '__main__':
    main()
