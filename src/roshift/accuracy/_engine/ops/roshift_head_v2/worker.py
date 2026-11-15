"""Calibrate and validate a sparse-head policy on one frozen body profile.

Only saved mixed-calibration selection tokens are used. No benchmark documents,
labels or accuracy measurements enter the gate or threshold selection.
"""
from __future__ import annotations
import os as _roshift_os
import argparse
import fcntl
import hashlib
import json
import math
import os
import shutil
from pathlib import Path
import sys
import time
from types import SimpleNamespace
ROOT = Path(_roshift_os.path.join(_roshift_os.environ['ROSHIFT_ENGINE_ROOT'], ''))
CODE = ROOT / 'code'
sys.path.insert(0, str(ROOT / 'ops'))
sys.path.insert(0, str(CODE / 'tools'))
from prepare_remote import read, sha, dump, configure_environment
from roshift_head_v2.search import PROTOCOL, known_seed, strict_search

def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()

def sources():
    files = [Path(__file__), Path(__file__).with_name('runtime.py'), Path(__file__).with_name('prepare_q.py'), Path(__file__).with_name('search.py'), CODE / 'MANIFEST.json', ROOT / 'ops/roshift_head/source_identity.json', CODE / 'analysis/llama3_ours_accuracy_runtime.py', CODE / 'analysis/runtime_token_scope.py', CODE / 'analysis/evaluate_llama3_mixed_calibration_q_lmhead_20260812.py', CODE / 'analysis/run_llama3_mixed_q_heldout_ppl_robust_lmhead_20260812.py', CODE / 'tools/run_likelihood.py', CODE / 'tools/materialize_profile.py']
    return {str(path): sha(path) for path in files}

def split_rows(metadata):
    """Historical alternating whole-sequence split; four nonempty domains."""
    domains = metadata['domains']
    if [int(d['blocks']) for d in domains] != [16, 6, 5, 5]:
        raise ValueError('Expected exact saved mixed-domain sequence allocation16/6/5/5')
    names = [d['name'] for d in domains]
    if len(set(names)) != 4 or any((d.get('split') != 'train' for d in domains)):
        raise ValueError('Calibration requires four distinct training domains')
    result = {'tune': [], 'validation': [], 'domain_rows': {}, 'domain_names': names}
    offset = 0
    for index, domain in enumerate(domains):
        rows = list(range(offset, offset + int(domain['blocks'])))
        tune = rows[1::2] if index == 3 else rows[::2]
        validation = rows[::2] if index == 3 else rows[1::2]
        if not tune or not validation:
            raise ValueError('Both splits require every domain')
        result['tune'].extend(tune)
        result['validation'].extend(validation)
        result['domain_rows'][domain['name']] = {'all': rows, 'tune': tune, 'validation': validation}
        offset += len(rows)
    if len(result['tune']) != 16 or len(result['validation']) != 16 or set(result['tune']) & set(result['validation']):
        raise ValueError('Invalid disjoint16/16 complete-sequence split')
    return result

def strict_pass(metrics, target=0.1):
    return all((name in metrics and metrics[name]['tokens'] > 0 and all((math.isfinite(float(value)) and float(value) < target for value in metrics[name]['per_domain_policy_changed_pct'].values())) and (len(metrics[name]['per_domain_policy_changed_pct']) == 4) for name in ('tune', 'validation')))

def fixed_policy(args):
    if args.policy_json:
        return read(args.policy_json)
    if args.model == 'qwen25':
        return dict(name='model_strict', first_dim=512, second_dim=1536, z1=7.75, z2=4.025, origin='historical_qwen25_s50_seed; unvalidated on current profile')
    return dict(name='mixed_strict', first_dim=512, second_dim=1536, z1=6.975, z2=3.85, origin='historical_llama31_s50_seed; unvalidated on current profile')

def verify_sources(path):
    expected = read(path)
    if expected.get('status') != 'verified' or expected['source_bundle_sha256'] != fingerprint(sources()):
        raise RuntimeError('Head worker/source identity changed after scheduler preparation')

def matching_v1_record(profile, identity):
    """Match complete passing/failing seed evidence without overwriting it."""
    path = profile / 'sparse_head_validation.json'
    if not path.is_file():
        return None
    old = read(path)
    if old.get('status') != 'complete':
        return None
    provenance = old.get('provenance', {})
    if old.get('fingerprint') != fingerprint(provenance):
        return None
    expected = read(ROOT / 'ops/roshift_head/source_identity.json')
    if expected.get('status') != 'verified' or provenance.get('sources') != expected['sources']:
        return None
    if any((sha(Path(p)) != value for p, value in expected['sources'].items())):
        return None
    if provenance.get('artifacts') != identity['artifacts'] or provenance.get('split_rows') != identity['split_rows']:
        return None
    if old.get('profile_fingerprint') != identity['profile_fingerprint'] or old.get('head_policy') != identity['seed_head_policy']:
        return None
    if not identity['known_seed'] or identity['chunk_rows'] != 32:
        return None
    metrics = old.get('validation', {}).get('metrics', {})
    if set(metrics) != set(('tune', 'validation')) or any((metrics[s].get('tokens') != 8192 for s in metrics)):
        return None
    if old.get('generation_probe', {}).get('passed') is not True:
        return None
    return (old, dict(path=str(path), sha256=sha(path), fingerprint=old['fingerprint'], reuse='exact matching complete fixed-policy calibration and cached-decode evidence'))

def compatible_v1_gate(profile, identity):
    previous = matching_v1_record(profile, identity)
    if previous is None:
        return None
    old, _ = previous
    return previous if old.get('passed') is True and strict_pass(old['validation']['metrics']) else None

def run(args):
    configure_environment()
    import numpy as np
    import torch
    from experiment_timing import Attempt, reused
    from materialize_profile import verify_export
    from prepare_remote import verify_model
    from rental_conditions_20260917 import require_single_gpu
    from rental_runtime_20260916 import install
    from runtime_token_scope import RuntimeController, install_runtime_policy
    import run_likelihood as likelihood
    import run_llama3_layerwise_q_paircd_fullcov_20260830 as paircd
    import evaluate_llama3_mixed_calibration_q_lmhead_20260812 as historical
    from roshift_head_v2.runtime import install_validated_head, set_validated_policy
    require_single_gpu(torch)
    if any((int(os.environ.get(k, '1')) != 1 for k in ('WORLD_SIZE', 'LOCAL_WORLD_SIZE'))):
        raise RuntimeError('TP/PP and distributed model execution are forbidden')
    torch.set_num_threads(4)
    torch.manual_seed(1234)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    common = args.work / args.model / 'common'
    profile = args.work / args.model / 'rebuilt' / args.profile
    calibration = args.work / 'inputs' / args.model / 'root/mixed_calibration'
    marker = read(profile / 'materialized.json')
    spec = read(profile / 'profile.json')
    if marker.get('status') != 'complete' or spec.get('model') != args.model:
        raise RuntimeError('Complete matching body profile required')
    verify_export(profile, marker['files'])
    likelihood.check_ppl(profile)
    verify_model(args.model, args.model_path)
    qpath = common / 'lmhead/q_head_beta_0.pt'
    qmeta = common / 'lmhead/q_head_meta.json'
    qm = read(qmeta)
    if qm.get('status') != 'complete' or qm.get('tokens') != 32 * 4096 or qm.get('beta') != 0:
        raise RuntimeError('Complete model-common native dense beta0 Q_head required')
    verify_export(qmeta.parent, qm['files'])
    if qm['token_sha256'] != sha(calibration / 'input_ids.pt'):
        raise RuntimeError('Q_head calibration tokens mismatch')
    meta = read(calibration / 'calibration_meta.json')
    splits = split_rows(meta)
    ids = torch.load(calibration / 'selection_input_ids.pt', map_location='cpu', weights_only=True)
    if tuple(ids.shape) != (32, 512):
        raise ValueError('Saved32x512 selection IDs required')
    for domain in meta['domains']:
        rows = splits['domain_rows'][domain['name']]['all']
        observed = hashlib.sha256(ids[rows].contiguous().numpy().tobytes()).hexdigest()
        if observed != domain['selection_sha256']:
            raise RuntimeError('Selection token hash mismatch: ' + domain['name'])
    policy = fixed_policy(args)
    identity = dict(model=args.model, profile=args.profile, profile_fingerprint=marker['fingerprint'], artifacts={str(p): sha(p) for p in (profile / 'materialized.json', profile / 'profile.json', profile / 'ppl_audit.json', qpath, qmeta, calibration / 'selection_input_ids.pt', calibration / 'calibration_meta.json')}, sources=sources(), seed_head_policy=policy, split_rows=splits, threshold_pct=0.1, chunk_rows=args.chunk, cohort='remote_regenerated_profile_head_mixed_selection_strict_grid_v2', search_protocol=PROTOCOL, known_seed=known_seed(args.model, spec['target'], args.policy_json is not None), historical_accuracy_identity_verified=False, benchmark_data_used=False, body_calibration_scope='all selection input positions sparse; final original-basis residual captured', generation_probe_scope='dense prompt of32 saved tokens, then4 teacher-forced cached decode tokens per domain', validation_protocol='known seed validation or original strict coarse/fine grid; final fixed metrics and cached decode gate')
    key = fingerprint(identity)
    out = profile / 'sparse_head_validation_v2.json'
    log = profile / 'head_validation_v2/experiment_timing.jsonl'
    log.parent.mkdir(parents=True, exist_ok=True)
    lock = (log.parent / 'worker.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if out.exists():
        saved = read(out)
        if saved.get('fingerprint') != key:
            raise RuntimeError('Head validation identity changed; preserve old result and use a separate profile/cohort')
        if saved.get('status') == 'complete':
            reused(log, model=args.model, method='RoShift', category='head_validation', substage='fixed_policy', label=args.profile + ':head_validation', targets=[spec['target']])
            print('[reuse] completed fixed-policy validation', flush=True)
            return
    result = dict(status='running', passed=False, fingerprint=key, profile_fingerprint=marker['fingerprint'], q_head={'path': str(qpath), 'sha256': sha(qpath)}, head_policy=policy, head_runtime={'path': str(Path(__file__).with_name('runtime.py')), 'sha256': sha(Path(__file__).with_name('runtime.py'))}, provenance=identity, started=time.time())
    v1_record = matching_v1_record(profile, identity)
    previous = compatible_v1_gate(profile, identity)
    if previous is not None:
        old, receipt = previous
        result.update(status='complete', passed=True, validation=old['validation'], generation_probe=old['generation_probe'], body_runtime_audit=old['body_runtime_audit'], reused_validation=receipt, elapsed_sec=0, calibration_search=dict(performed=False, reason='exact complete v1 fixed-policy gate reused', protocol=PROTOCOL))
        dump(out, result)
        reused(log, model=args.model, method='RoShift', category='head_validation', substage='validated_v1_evidence', label=args.profile + ':validated_v1_evidence', targets=[spec['target']])
        print('[reuse] exact complete v1 profile/Q/head/calibration gate', flush=True)
        return
    dump(out, result)
    timing = dict(model=args.model, method='RoShift', targets=[spec['target']], target=spec['target'], category='head_validation', gpu_reserved=True)
    try:
        cm = read(common / 'common_ready.json')
        install(common / 'dense_median', cm['model_type'])
        par = SimpleNamespace(model_id=str(args.model_path), root=args.work / 'inputs' / args.model / 'root', output_dir=profile, adaptive_json=common / 'adaptive_dp_centers.json', device_map='cuda:0', attn_implementation='sdpa', block_size=128, permutation_seed=11, fallback_rel=0.05)
        with Attempt(log, **timing, substage='frozen_body_load', label='frozen_body_load'):
            model, device, patched, medians, adaptive = paircd.load_eval_model(par, np.asarray(spec['recipe']))
            model.eval().requires_grad_(False)
            result['body_runtime_audit'] = likelihood.runtime_audit(model, patched)
        control = RuntimeController()
        for layer in patched:
            layer.controller.runtime_controller = control
        install_runtime_policy(model, control)
        cache = log.parent / 'selection_residuals.pt'
        cache_meta = cache.with_suffix('.json')
        cache_identity = dict(profile_fingerprint=marker['fingerprint'], materialized_sha256=sha(profile / 'materialized.json'), selection_sha256=sha(calibration / 'selection_input_ids.pt'), sources=sources())
        if not cache.exists() and v1_record is not None:
            prior_cache = profile / 'head_validation/selection_residuals.pt'
            prior_meta = prior_cache.with_suffix('.json')
            original_identity = dict(cache_identity, sources=v1_record[0]['provenance']['sources'])
            if prior_cache.exists() and prior_meta.exists() and (read(prior_meta).get('identity') == original_identity) and (read(prior_meta).get('sha256') == sha(prior_cache)):
                shutil.copyfile(prior_cache, cache)
                dump(cache_meta, dict(identity=cache_identity, sha256=sha(cache), reused_from={'path': str(prior_cache), 'sha256': sha(prior_cache)}))
                print('[reuse] exact v1 body-matched selection residuals; old file preserved', flush=True)
        if cache.exists() and cache_meta.exists() and (read(cache_meta).get('identity') == cache_identity) and (read(cache_meta).get('sha256') == sha(cache)):
            hidden = torch.load(cache, map_location='cpu', weights_only=True)
            reused(log, **{k: v for k, v in timing.items() if k != 'gpu_reserved'}, substage='selection_residuals', label='selection_residuals')
        else:
            captured = []
            handle = model.model.norm.register_forward_pre_hook(lambda module, a: captured.append(a[0].detach().to('cpu').contiguous()))
            original = model.lm_head
            model.lm_head = torch.nn.Identity()
            try:
                with torch.inference_mode(), Attempt(log, **timing, substage='selection_residuals', label='selection_residuals'):
                    for row in range(32):
                        token = ids[row:row + 1].to(device)
                        model(input_ids=token, attention_mask=torch.ones_like(token), use_cache=False)
                        print(f'[head selection] {row + 1}/32', flush=True)
            finally:
                handle.remove()
                model.lm_head = original
            hidden = torch.cat(captured, dim=0).contiguous()
            torch.save(hidden, cache)
            dump(cache_meta, dict(identity=cache_identity, sha256=sha(cache)))
        if tuple(hidden.shape[:2]) != (32, 512):
            raise RuntimeError('Incomplete profile-specific hidden capture')
        with torch.inference_mode(), Attempt(log, **timing, substage='head_fold', label='head_fold'):
            head = install_validated_head(model, qpath, policy, control)
        sequence_domains = torch.tensor([i for i, d in enumerate(meta['domains']) for _ in range(d['blocks'])])

        def measure(selected):
            metrics = {}
            for name in ('tune', 'validation'):
                rows = splits[name]
                with torch.inference_mode(), Attempt(log, **timing, substage='fixed_metrics_' + name, label='fixed_metrics_' + name):
                    metrics[name] = historical.fixed_policy_metrics(hidden_cpu=hidden[rows].reshape(-1, hidden.shape[-1]), domain_ids_cpu=sequence_domains[rows].repeat_interleave(512), domain_names=splits['domain_names'], raw_weight=head.inner.weight, gamma=head.final_gamma, q=head.qhead, folded=head.folded_weight, policy=historical.Policy(z1=selected['z1'], z2=selected['z2'], first_dim=selected['first_dim'], second_dim=selected['second_dim']), chunk=args.chunk, raw_residual=True, final_eps=float(model.config.rms_norm_eps))
                print('[head metrics]', name, json.dumps(metrics[name]), flush=True)
            return metrics
        if v1_record is not None:
            metrics = v1_record[0]['validation']['metrics']
            result['reused_seed_validation'] = v1_record[1]
            reused(log, model=args.model, method='RoShift', targets=[spec['target']], category='head_validation', substage='fixed_seed_metrics', label='fixed_seed_metrics')
            print('[reuse] complete v1 fixed-seed metrics; failed original marker preserved', flush=True)
        else:
            metrics = measure(policy) if identity['known_seed'] else {}
        seed_passed = bool(metrics) and strict_pass(metrics)
        result['seed_validation'] = dict(performed=bool(metrics), passed=seed_passed, metrics=metrics, policy=policy)
        searched = not seed_passed
        if searched:
            first_dim = min(512, int(hidden.shape[-1]) // 4)
            second_dim = min(1536, int(hidden.shape[-1]) // 2)

            def evaluate(stage, split, z1, z2):
                grid_identity = dict(validation_fingerprint=key, stage=stage, split=split, z1=z1, z2=z2, first_dim=first_dim, second_dim=second_dim, chunk_rows=args.chunk)
                grid_path = log.parent / f'grid_{stage}_{split}.json'
                if grid_path.exists():
                    saved = read(grid_path)
                    if saved.get('identity') != grid_identity:
                        raise RuntimeError('Saved calibration grid identity changed')
                    if saved.get('status') == 'complete':
                        if len(saved['rows']) != len(z1) * len(z2):
                            raise RuntimeError('Incomplete saved grid')
                        reused(log, model=args.model, method='RoShift', targets=[spec['target']], category='head_calibration', substage=stage + '_' + split, label=stage + '_' + split)
                        return saved['rows']
                chosen = splits[split]
                grid_timing = dict(timing, category='head_calibration')
                with torch.inference_mode(), Attempt(log, **grid_timing, substage=stage + '_' + split, label=stage + '_' + split):
                    rows, _ = historical.search_grid(hidden_cpu=hidden[chosen].reshape(-1, hidden.shape[-1]), domain_ids_cpu=sequence_domains[chosen].repeat_interleave(512), domain_names=splits['domain_names'], gamma=head.final_gamma, q=head.qhead, folded=head.folded_weight, z1_values=z1, z2_values=z2, chunk=args.chunk, first_dim=first_dim, second_dim=second_dim, target_pct=0.1, final_eps=float(model.config.rms_norm_eps))
                dump(grid_path, dict(status='complete', identity=grid_identity, rows=rows))
                return rows
            policy, evidence = strict_search(evaluate, splits['domain_names'], first_dim, second_dim)
            result['calibration_search'] = evidence
            set_validated_policy(head, policy)
            metrics = measure(policy)
        else:
            result['calibration_search'] = dict(performed=False, reason='known fixed seed passes full strict gate', protocol=PROTOCOL)
        result['head_policy'] = policy
        result['validation'] = dict(metrics=metrics, strict_per_domain_threshold_pct=0.1, all_domains_nonempty=True, split_sequence_rows=splits, rotation_error_reported_separately=True, threshold_search_performed=searched, validation_role=PROTOCOL['validation_role'])
        with torch.inference_mode(), Attempt(log, **timing, substage='cached_decode_probe', label='cached_decode_probe'):
            result['generation_probe'] = generation_probe(model, head, control, ids, splits, device)
        result.update(status='complete', passed=strict_pass(metrics) and result['generation_probe']['passed'], elapsed_sec=time.time() - result['started'])
        dump(out, result)
        print(json.dumps({'status': result['status'], 'passed': result['passed'], 'marker': str(out)}), flush=True)
    except BaseException as error:
        result.update(status='failed', passed=False, error=repr(error), elapsed_sec=time.time() - result['started'])
        dump(out, result)
        raise

def generation_probe(model, head, control, ids, splits, device):
    import torch
    from runtime_token_scope import is_decode_forward
    observed = []
    captured = []
    handles = [model.model.norm.register_forward_hook(lambda m, a, o: captured.append(o.detach()))]
    for layer in model.model.layers:
        handles.append(layer.register_forward_pre_hook(lambda m, a: observed.append(control.sparse_enabled())))
    rows = []
    try:
        for name in splits['domain_names']:
            row = splits['domain_rows'][name]['validation'][0]
            prompt = ids[row:row + 1, :32].to(device)
            control.task_policy = 'dense'
            captured.clear()
            observed.clear()
            expected = model(input_ids=prompt, use_cache=False).logits
            control.task_policy = 'decode_only'
            head.reset_stats()
            captured.clear()
            observed.clear()
            output = model(input_ids=prompt, use_cache=True)
            dense_ok = bool(torch.equal(expected, output.logits)) and head.rows == 0 and (not any(observed))
            cache = output.past_key_values
            details = []
            for offset in range(32, 36):
                token = ids[row:row + 1, offset:offset + 1].to(device)
                decode_scope = is_decode_forward(token, cache)
                captured.clear()
                observed.clear()
                before = head.rows
                output = model(input_ids=token, past_key_values=cache, use_cache=True)
                cache = output.past_key_values
                hidden = captured[-1]
                raw = head.inner(hidden)
                rotated = head._qhead_dense(hidden)
                sparse = output.logits
                full_winner = rotated.argmax(-1)
                pruning_lost = ~torch.isfinite(sparse.gather(-1, full_winner.unsqueeze(-1))).squeeze(-1)
                details.append(dict(token_offset=offset, decode_scope=decode_scope, body_sparse=bool(observed) and all(observed), head_sparse=head.rows - before == 1, no_nan=not bool(torch.isnan(sparse).any()), finite_candidate_per_row=bool(torch.isfinite(sparse).any(-1).all()), original_dense_vs_qhead_dense_top1_changed=int((raw.argmax(-1) != full_winner).sum()), pruning_lost_dense_winner=int(pruning_lost.sum()), staged_vs_qhead_dense_top1_changed=int((sparse.argmax(-1) != full_winner).sum()), original_vs_qhead_dense_relative_l2=float((raw.float() - rotated.float()).norm() / raw.float().norm().clamp_min(1e-12))))
            rows.append(dict(domain=name, selection_sequence_row=row, prompt_tokens=32, prompt_dense_exact=dense_ok, decode=details))
        prompt = ids[:1, :1].to(device)
        control.task_policy = 'dense'
        expected = model(input_ids=prompt, use_cache=False).logits
        control.task_policy = 'decode_only'
        head.reset_stats()
        observed.clear()
        output = model(input_ids=prompt, use_cache=True)
        one_token = bool(torch.equal(expected, output.logits)) and head.rows == 0 and (not any(observed))
    finally:
        for handle in handles:
            handle.remove()
        control.task_policy = 'all'
    passed = one_token and all((row['prompt_dense_exact'] and all((all((item[key] for key in ('decode_scope', 'body_sparse', 'head_sparse', 'no_nan', 'finite_candidate_per_row'))) for item in row['decode'])) for row in rows))
    return dict(passed=passed, one_token_prompt_dense_exact=one_token, rows=rows, tokens=16, selection_only=True, teacher_forced=True, benchmark_used=False, body_scope_observation='shared controller at every layer entry; body_runtime_audit verifies consuming wrappers', accuracy_threshold_selection=False)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=read(CODE / 'models.json'), required=True)
    parser.add_argument('--work', type=Path)
    parser.add_argument('--model-path', type=Path)
    parser.add_argument('--profile', required=True)
    parser.add_argument('--policy-json', type=Path)
    parser.add_argument('--chunk', type=int, default=32)
    parser.add_argument('--expected-source-identity', type=Path)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    args.work = args.work or ROOT / 'work' / args.model
    args.model_path = args.model_path or ROOT / 'models' / read(CODE / 'models.json')[args.model]['directory']
    if args.expected_source_identity:
        verify_sources(args.expected_source_identity)
    if args.chunk <= 0:
        raise ValueError('Positive chunk size required')
    if not args.execute:
        print(json.dumps(dict(model=args.model, profile=args.profile, work=str(args.work), head_policy=fixed_policy(args), source_bundle_sha256=fingerprint(sources()), scope='versioned profile-specific strict calibration grid', search_protocol=PROTOCOL, required=['materialized profile', 'body PPL audit', 'common beta0 Q_head', 'single visible GPU']), indent=2))
        return
    configure_environment()
    from experiment_timing import job
    target = int(args.profile.rsplit('s', 1)[1])
    log = args.work / args.model / 'rebuilt' / args.profile / 'head_validation_v2/experiment_timing.jsonl'
    with job(log, model=args.model, method='RoShift', category='job', substage='head_validation', label=args.profile + ':head_validation', targets=[target], gpu_reserved=True):
        run(args)
if __name__ == '__main__':
    main()
