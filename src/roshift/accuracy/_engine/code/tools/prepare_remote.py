"""No-tensor-transfer RoShift preparation. No GPU work without --execute.

One model per process/GPU. Work must be outside the immutable extracted kit.
This is preparation/search only, not an eight-GPU benchmark dispatcher.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys

KIT = Path(__file__).resolve().parents[1]


def data_inputs():
    if not os.environ.get('ROSHIFT_INPUT_ROOT'):
        raise RuntimeError('Use the public roshift accuracy launcher to restore external datasets first')
    return Path(os.environ['ROSHIFT_INPUT_ROOT'])


COMPONENTS = ('S_att', 'S_ffn', 'M_q', 'M_k', 'M_v', 'M_gate')
sys.path.insert(0, str(KIT/'analysis'))
from rental_conditions_20260917 import condition_selected, require_single_gpu, selection, selection_record
from experiment_timing import Attempt, FILENAME as TIMING_FILE, classify, job as timed_job, reused


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def artifact_manifest_sha():
    """Return the tensor artifact identity for compatible source updates."""
    legacy_path = KIT/'ARTIFACT_MANIFEST_RC5.json'
    legacy = read(legacy_path)['files']
    current = read(KIT/'MANIFEST.json')['files']
    operational = {'tools/prepare_remote.py', 'tools/materialize_profile.py',
                   'vendor/transformers/modeling_utils.py'}
    # Exact hashes identify nonnumerical source updates.
    text_cleanup = {
        'analysis/run_llama3_larosa_teal_lmeval_suite_20260604.py':
            '57019765c90b0352f1ee59613f1199b0daaa9b464d860ed8a6bc03c1a4143068',
        'context/source_reference/official_method_eval_20260827/official_common.py':
            '5ae6caad7ee070187d386c012e471be885c5819bfa6277aab02bd06bf32ceaf0',
        'context/source_reference/analysis/run_teal_resume_20260912.py':
            'df04c6d05f40b870349f76b25a02bdbaa1fae6c4c4e9b8ab4e924861f42cde98',
        'context/source_reference/analysis/run_teal_official_suite_20260910.py':
            'e55685cddb4f178e5fd1318b8735eac04c03a1320dffcba1ad16ebe7416f4384',
        'context/source_reference/larosa/README.md':
            '34f2fa9faa283d5835ef49a500f6d2fd2d5bcc6042b0e8f6ace8ff5a81d19b3a',
        'vendor/lm_eval/tasks/README.md':
            'dca343308a22627be47e7ba8cfcaab687da5f2ffd1f10fc6520690ae52c34cc8',
        'tools/build_package.py':
            '447b1b64ea3298dc178793c9fc7cc7891b6a09b0414b4631da187c95d4e73e29',
        'tools/run_likelihood.py':
            '278aac631fb9a35809bf66793b2b2a15b715e8717946c4c9d72cf8a39191973f',
    }
    if set(current) != set(legacy) | {'ARTIFACT_MANIFEST_RC5.json'}:
        raise RuntimeError('Artifact compatibility check failed: core file set changed')
    for name, expected in legacy.items():
        if name in text_cleanup and current[name]['sha256'] == text_cleanup[name]:
            continue
        if name not in operational and current[name] != expected:
            raise RuntimeError('Artifact compatibility check failed: '+name)
    return sha(legacy_path)


def scratch_requirement(work, label, identity):
    """Reserve only the next stage's additional storage, not all past outputs."""
    if completed_stage_marker(work, label, identity):
        return 10, 'completed stage; verify and reuse'
    if label == 'dense_statistics':
        return 80, 'fresh/incomplete dense statistics and scratch'
    if label == 'head_basis':
        return 10, 'small head basis; model remains in memory'
    weights = read(KIT/'inputs'/identity['model']/'base_model_identity.json')['weights']
    model_gib = sum(item['bytes'] for item in weights) / (1 << 30)
    # Profile replay can retain a changed pack plus a transactional backup.
    multiplier = 1.5 if label == 'common_pack' else 2
    return max(25, math.ceil(multiplier * model_gib + 8)), 'pending folded weights and scratch'


def check_scratch(work, label, identity):
    required, reason = scratch_requirement(work, label, identity)
    free = shutil.disk_usage(work).free / (1 << 30)
    if free < required:
        raise RuntimeError(f'{label}: {free:.1f} GiB free, {required} GiB required '
                           f'({reason}); keep completed artifacts; no automatic cleanup')
    print(f'[storage] {label}: {free:.1f} GiB free; guard={required} GiB ({reason})', flush=True)


def safe_child(root, relative):
    root = Path(root).resolve()
    child = (root / relative).resolve()
    if child == root or not child.is_relative_to(root):
        raise ValueError(f'Unsafe relative path: {relative}')
    return child


def validate_alpha(alpha, layers):
    if len(alpha) != layers:
        raise ValueError('Wrong layer count in mixture recipe')
    for row in alpha:
        if len(row) != 6 or any(not math.isfinite(v) or v < -1e-12 for v in row):
            raise ValueError('Mixture needs six finite nonnegative coefficients')
        if abs(sum(row) - 1.) > 1e-8:
            raise ValueError('Mixture coefficients do not sum to one')


def configure_environment():
    entries = [str(KIT/'vendor'), str(KIT/'analysis'), str(KIT/'tools'), str(KIT)]
    sys.path[:0] = entries
    os.environ['PYTHONPATH'] = os.pathsep.join(entries + [os.environ.get('PYTHONPATH', '')])
    for key, value in dict(OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', MALLOC_ARENA_MAX='2',
                           TOKENIZERS_PARALLELISM='false', MPLBACKEND='Agg',
                           ROSHIFT_SKIP_ALLOCATOR_WARMUP='1',
                           HF_DEACTIVATE_ASYNC_LOAD='1', CUBLAS_WORKSPACE_CONFIG=':4096:8').items():
        os.environ.setdefault(key, value)


def verify_model(key, model):
    identity = read(KIT/'inputs'/key/'base_model_identity.json')
    expected = read(KIT/'inputs'/key/'config.json')
    actual = read(model/'config.json')
    for field in ('model_type', 'hidden_size', 'intermediate_size', 'num_hidden_layers',
                  'num_attention_heads', 'num_key_value_heads', 'head_dim', 'vocab_size',
                  'rope_theta', 'rope_scaling', 'rms_norm_eps', 'hidden_act', 'tie_word_embeddings'):
        if actual.get(field) != expected.get(field):
            raise RuntimeError(f'{key}: model config mismatch: {field}')
    for item in identity['weights']:
        path = safe_child(model, item['path'])
        if not path.is_file() or path.stat().st_size != item['bytes'] or sha(path) != item['sha256']:
            raise RuntimeError(f'{key}: base checkpoint hash mismatch: {item["path"]}')
    # Calibration token IDs are meaningful only with the matching tokenizer.
    for name, digest in read(KIT/'regeneration.json')['models'][key]['tokenizer_sha256'].items():
        if sha(safe_child(model, name)) != digest:
            raise RuntimeError(f'{key}: tokenizer mismatch: {name}')


def download_model(key, model):
    from huggingface_hub import snapshot_download
    info = read(KIT/'models.json')[key]
    pinned = read(KIT/'regeneration.json')['models'][key]
    identity = read(KIT/'inputs'/key/'base_model_identity.json')
    files = [x['path'] for x in identity['weights']]
    files += ['config.json', 'generation_config.json', 'model.safetensors.index.json',
              *pinned['tokenizer_sha256']]
    # No tokens are stored in this package; HF_TOKEN/hf auth login is external.
    snapshot_download(repo_id=info['hf_id'], revision=pinned.get('download_revision') or 'main',
                      local_dir=str(model), allow_patterns=files, max_workers=2)
    verify_model(key, model)


def plan(key, work, allocation='nonuniform', targets=(40, 50, 60), skip_head=False):
    records = read(KIT/'regeneration.json')
    modes, targets = selection(allocation, targets)
    profiles = [p for p in records['profiles'] if condition_selected(p, key, modes, targets)]
    allocations = [a for a in records['allocations'] if condition_selected(a, key, modes, targets)]
    return dict(model=key, work=str(work), allocation_recipes_reused=len(allocations),
                selection=selection_record(modes, targets),
                q_profiles_rebuilt_from_coefficients=[p['id'] for p in profiles],
                missing_allocation_searches=(len(targets) if 'nonuniform' in modes else 0)-len(allocations),
                missing_q_searches=len(targets)*len(modes)-len(profiles),
                stages=['download_and_hash_check', 'dense_median_and_statistics',
                        'initial_Q_and_folded_pack_and_DP'] + ([] if skip_head else ['beta0_Q_head']) +
                       ['rebuild_completed_Q_profiles', 'optional_missing_searches'],
                q_head_deferred=skip_head,
                large_tensor_transfer_required=False,
                historical_bitwise_equivalence=False,
                sparse_head_validation_required=True)


def completed_stage_marker(work, label, identity):
    """Detect reuse for accounting; subprocess still verifies artifact hashes."""
    key=identity['model']
    common=work/key/'common'
    marker={'dense_statistics':common/'dense_preparation_state.json',
            'common_pack':common/'common_ready.json',
            'head_basis':common/'lmhead/q_head_meta.json'}.get(label)
    if marker is None:
        profile=label.removesuffix('_export')
        if not profile.startswith(key+'_'):
            profile=key+'_'+profile
        marker=work/key/'rebuilt'/profile/'materialized.json'
    try:
        return marker if read(marker).get('status')=='complete' else None
    except (OSError,ValueError,AttributeError):
        return None


def run_script(work, label, script, arguments):
    identity=read(work/'regeneration_selection.json')
    metadata=classify(label,[script,*arguments],model=identity['model'])
    metadata.update(command=[str(script),*map(str,arguments)],run_sha256=identity['run_sha256'])
    marker=completed_stage_marker(work,label,identity)
    # A reused subprocess only verifies its artifacts. Its wrapper reservation
    # is recorded separately; the reused numerical stage receives zero charge.
    with Attempt(work/TIMING_FILE,scope='container' if marker else 'leaf',**metadata):
        result=_run_script(work,label,script,arguments)
    if marker:
        reused(work/TIMING_FILE,**metadata,marker=str(marker),reuse_verified=True)
    return result


def _run_script(work, label, script, arguments):
    if (work/'STOP_AFTER_STAGE').exists():
        raise KeyboardInterrupt('STOP_AFTER_STAGE exists; no new stage started')
    identity = read(work/'regeneration_selection.json')
    check_scratch(work, label, identity)
    log = work/'regeneration_logs'/f'{label}.log'
    log.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, '-u', str(KIT/script), *map(str, arguments)]
    identity = read(work/'regeneration_selection.json')
    dump(work/'regeneration_progress.json', dict(status='running', stage=label, command=command,
                                               log=str(log), pid=os.getpid(),
                                               selection=identity['selection'],
                                               run_sha256=identity['run_sha256'],
                                               q_head_deferred=identity['q_head_deferred']))
    print('[stage]', label, 'log=', log, flush=True)
    with log.open('a') as stream:
        stream.write('\nCOMMAND ' + json.dumps(command) + '\n')
        stream.flush()
        child = subprocess.Popen(command, cwd=KIT, stdout=stream, stderr=subprocess.STDOUT)
        try:
            code = child.wait()
        except BaseException:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
            raise
    if code:
        raise RuntimeError(f'{label}: exit={code}; see {log}')


def configure_search(key, work, model_path, allocation='nonuniform', targets=(40, 50, 60)):
    import run_search
    records = read(KIT/'regeneration.json')
    namespace = argparse.Namespace(models=key, models_root=model_path.parent,
        model_path=[f'{key}={model_path.resolve()}'], work=work, allocation=allocation, targets=targets)
    suite, _ = run_search.configure(namespace)
    suite.DATA = work/'inputs'
    suite.ALLOC = {(key, r['target']): KIT/r['path'] for r in records['allocations'] if r['model'] == key}
    suite.Q_READY = {(key, r['allocation'], r['target']): work/key/'rebuilt'/r['id']
                     for r in records['profiles'] if r['model'] == key}
    original_publish = suite.publish_status
    def publish(state):
        original_publish(state)
        output = read(work/'progress.json')
        for row in output['rotation']:
            if (row['model'], row['allocation'], row['target']) in suite.Q_READY:
                row['status'] = 'regenerated_frozen_profile'
                row['tensors_present'] = True
                row['historical_bitwise_equivalence'] = False
        output['scope'] = 'Selected recipient regeneration and missing searches; no automatic benchmark approval'
        dump(work/'progress.json', output)
    suite.publish_status = publish
    return suite


def execution_identity(identity, modes, targets, skip_head=False):
    scoped = dict(identity, selection=selection_record(modes, targets), q_head_deferred=skip_head)
    scoped['run_sha256'] = hashlib.sha256(json.dumps(scoped, sort_keys=True).encode()).hexdigest()
    return scoped


def execute(args):
    _,targets=selection(args.allocation,args.targets)
    with timed_job(args.work.resolve()/TIMING_FILE,model=args.model,label='prepare:'+args.stage,
                   category='job',substage='prepare',targets=list(targets),gpu_reserved=args.stage!='download'):
        return _execute(args)


def _execute(args):
    import fcntl
    from verify_package import verify
    verify(KIT)
    def interrupt(signum, frame):
        raise KeyboardInterrupt(f'Signal {signum}; stop child and retain checkpoints')
    signal.signal(signal.SIGTERM, interrupt)
    key, work = args.model, args.work.resolve()
    modes, targets = selection(args.allocation, args.targets)
    scope = selection_record(modes, targets)
    work.mkdir(parents=True, exist_ok=True)
    with (work/'regeneration.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fingerprint = artifact_manifest_sha()
        marker = work/'regeneration_identity.json'
        identity = dict(package_sha256=fingerprint, model=key, model_path=str(args.model_path.resolve()))
        if marker.exists() and read(marker) != identity:
            raise RuntimeError('Work directory belongs to another package/model; use a new directory')
        dump(marker, identity)
        # Scope may safely expand after a first-condition audit. Common model
        # artifacts retain their identity; each selection is independently logged.
        run_identity = execution_identity(identity, modes, targets, args.skip_head)
        dump(work/'regeneration_selection.json', run_identity)
        dump(work/'selection_history'/f'{run_identity["run_sha256"]}.json', run_identity)
        try:
            if args.download:
                download_model(key, args.model_path.resolve())
            else:
                verify_model(key, args.model_path.resolve())
            if args.stage == 'download':
                dump(work/'regeneration_progress.json', dict(status='download_verified', model=key, selection=scope))
                return
            import torch
            require_single_gpu(torch)
            if not torch.cuda.is_available() or torch.cuda.get_device_properties(0).total_memory < 30*(1 << 30):
                raise RuntimeError('Preparation requires an assigned CUDA GPU with >=32GB VRAM')
            records = read(KIT/'regeneration.json')
            meta = read(KIT/'models.json')[key]
            root = work/'inputs'/key/'root'
            # Inputs contain token IDs and small DP JSON, never saved weights.
            shutil.copytree(data_inputs()/key/'root', root, dirs_exist_ok=True)
            common = work/key/'common'
            base = ['--model_id', args.model_path.resolve(), '--root', root, '--output_dir', common]
            run_script(work, 'dense_statistics', 'analysis/rental_prepare_dense_20260916.py', base)
            adaptive = root/meta['adaptive_relative'] if meta['adaptive_relative'] else None
            run_script(work, 'common_pack', 'analysis/rental_prepare_common_20260916.py',
                       base + (['--adaptive_source', adaptive] if adaptive else []))
            if not args.skip_head:
                run_script(work, 'head_basis', 'tools/materialize_profile.py',
                           ['--action', 'head', '--model', key, '--model-path', args.model_path.resolve(), '--work', work])
            if args.stage == 'common':
                dump(work/'regeneration_progress.json', dict(status='common_complete', model=key,
                     selection=scope, q_head_deferred=args.skip_head))
                return
            for profile in records['profiles']:
                if condition_selected(profile, key, modes, targets):
                    run_script(work, profile['id'], 'tools/materialize_profile.py',
                               ['--action', 'profile', '--model', key, '--profile', profile['id'],
                                '--model-path', args.model_path.resolve(), '--work', work])
            if args.stage == 'all':
                suite = configure_search(key, work, args.model_path.resolve(), args.allocation, targets)
                suite.run()
                # The legacy queue installs its own handlers. Restore ours for
                # the export subprocesses that follow it.
                signal.signal(signal.SIGTERM, interrupt)
                signal.signal(signal.SIGINT, interrupt)
                state = read(work/'progress.json')
                if state.get('status') not in ('complete',):
                    raise RuntimeError(f'Search did not finish: {state.get("status")}; checkpoints retained')
                # Search removes disposable packs. Reconstruct each accepted profile
                # from its coefficient recipe and frozen acceptance trace, no re-search.
                for target in targets:
                    for mode in modes:
                        condition = work/key/'q'/f'{mode}_s{target}'
                        if not (condition/'final_recipe.json').exists():
                            continue
                        run_script(work, f'{mode}_s{target}_export', 'tools/materialize_profile.py',
                            ['--action', 'profile', '--model', key, '--model-path', args.model_path.resolve(),
                             '--work', work, '--condition', condition])
            dump(work/'regeneration_progress.json', dict(status='preparation_complete', stage=args.stage,
                 model=key, benchmark_ready=False,
                 selection=scope, q_head_deferred=args.skip_head,
                 note='Run rebuilt-body PPL audit and sparse-head validation before new benchmark reuse'))
        except BaseException as exc:
            dump(work/'regeneration_progress.json', dict(status='paused' if isinstance(exc, KeyboardInterrupt) else 'failed',
                                                       error=str(exc), checkpoints_retained=True, selection=scope))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, choices=list(read(KIT/'models.json')))
    parser.add_argument('--model-path', required=True, type=Path)
    parser.add_argument('--work', required=True, type=Path)
    parser.add_argument('--stage', choices=['download', 'common', 'rebuild', 'all'], default='rebuild')
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--allocation', choices=['nonuniform', 'uniform', 'both'], default='nonuniform')
    parser.add_argument('--targets', default='40,50,60', help='Comma-separated subset of 40,50,60')
    parser.add_argument('--skip-head', action='store_true', help='Defer Q_head for likelihood-first preparation')
    args = parser.parse_args()
    try:
        selection(args.allocation, args.targets)
    except ValueError as exc:
        parser.error(str(exc))
    if args.work.resolve() == KIT or args.work.resolve().is_relative_to(KIT):
        parser.error('--work must be outside the immutable package')
    if args.model_path.resolve() == KIT or args.model_path.resolve().is_relative_to(KIT):
        parser.error('--model-path must be outside the immutable package')
    print(json.dumps(plan(args.model, args.work, args.allocation, args.targets, args.skip_head), indent=2), flush=True)
    if args.execute:
        configure_environment()
        execute(args)


if __name__ == '__main__':
    main()
