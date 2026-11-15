"""Recipient GPU stages used by prepare_remote.py; never run by package tests."""
from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace
import gc
import json

from prepare_remote import KIT, COMPONENTS, configure_environment, read, dump, sha, validate_alpha, artifact_manifest_sha


def exported_files(root):
    return {str(p.relative_to(root)): dict(bytes=p.stat().st_size, sha256=sha(p))
            for p in sorted(root.rglob('*.pt')) if 'baseline_uniform_fullcov' not in p.parts}


def verify_export(root, manifest):
    for name, expected in manifest.items():
        p = root/name
        if not p.is_file() or p.stat().st_size != expected['bytes'] or sha(p) != expected['sha256']:
            raise RuntimeError(f'Rebuilt artifact missing/changed: {p}')


def inputs(args):
    common = args.work/args.model/'common'
    root = args.work/'inputs'/args.model/'root'
    meta = read(common/'common_ready.json')
    if meta['status'] != 'complete':
        raise RuntimeError('Common preparation must finish first')
    return common, root, meta


def head(args):
    import torch
    from run_llama2_ffn_gate_output_threshold_ppl import load_model, input_device
    from build_llama3_lmhead_qhead_20260806 import canonicalize_columns
    from rental_prepare_dense_20260916 import save
    from rental_validation import rms_epsilon
    common, root, meta = inputs(args)
    out = common/'lmhead'
    path = out/'q_head_meta.json'
    if path.exists() and read(path).get('status') == 'complete':
        verify_export(out, read(path)['files'])
        return
    model = load_model(SimpleNamespace(model_id=str(args.model_path), device_map='cuda:0',
                                      attn_implementation='sdpa'))
    model.eval().requires_grad_(False)
    device = input_device(model)
    norm = model.model.norm
    eps = rms_epsilon(norm)
    dim = model.config.hidden_size
    second = torch.zeros(dim, dim, device=device, dtype=torch.float32)
    ids = torch.load(root/'mixed_calibration/input_ids.pt', map_location='cpu', weights_only=True)
    if tuple(ids.shape) != (32, 4096):
        raise ValueError('Q_head must use all mixed32x4096 tokens')
    count = 0

    def hook(module, call):
        nonlocal count
        x = call[0].detach().reshape(-1, dim).float()
        x *= torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)
        second.addmm_(x.T, x)
        count += len(x)

    handle = norm.register_forward_pre_hook(hook)
    try:
        for index, row in enumerate(ids):
            batch = row[None].to(device)
            # Native decoder only; never allocate a 4096 x vocab logits tensor.
            result = model.model(input_ids=batch, attention_mask=torch.ones_like(batch), use_cache=False)
            del result
            print('[Q_head] sequence', index+1, '/ 32', flush=True)
    finally:
        handle.remove()
    if count != ids.numel():
        raise RuntimeError('Incomplete final-residual capture')
    second /= count
    second = .5 * (second + second.T)
    values, q = torch.linalg.eigh(second)
    order = torch.argsort(values, descending=True)
    q = canonicalize_columns(q[:, order]).contiguous()
    save(out/'q_head_beta_0.pt', q.half())
    save(out/'eigenvalues.pt', values[order])
    dump(path, dict(status='complete', beta=0, tokens=count,
         token_sha256=sha(root/'mixed_calibration/input_ids.pt'),
         objective='uncentered final no-gamma activation second moment; native dense model',
         files=exported_files(out), sparse_head_thresholds_validated=False,
         note='Q_head exists; this does not validate candidate pruning thresholds'))


def profile_spec(args):
    if args.condition:
        condition = args.condition.resolve()
        if not condition.is_relative_to((args.work/args.model/'q').resolve()):
            raise ValueError('Condition must belong to this model/work directory')
        recipe = read(condition/'final_recipe.json')
        mode, target = condition.name.rsplit('_s', 1)
        return dict(id=f'{args.model}_{mode}_s{target}', model=args.model, allocation=mode,
                    target=int(target), alpha=[[row[k] for k in COMPONENTS] for row in recipe['alpha']],
                    replay_layers=recipe['reconstruction']['apply_selected_q_in_layer_order'],
                    recipe=read(recipe['allocation'])['recipe'],
                    source='recipient_new_onepass_search',
                    calibration_reference=recipe['calibration_PPL'],
                    accepted_layers=[row['layer'] for row in recipe['alpha'] if row['accepted']])
    found = [r for r in read(KIT/'regeneration.json')['profiles']
             if r['id'] == args.profile and r['model'] == args.model]
    if len(found) != 1:
        raise ValueError('No unique frozen coefficient profile')
    return found[0]


def materialize(args):
    import torch
    import numpy as np
    import run_llama3_layerwise_q_tpe_block_l2_20260827 as old
    import run_llama3_layerwise_q_paircd_fullcov_20260830 as paircd
    from rental_prepare_dense_20260916 import save
    from rental_prepare_common_20260916 import link_files
    from rental_runtime_20260916 import install
    from rental_validation import logit_equivalence_audit
    common, root, meta = inputs(args)
    spec = profile_spec(args)
    validate_alpha(spec['alpha'], meta['layers'])
    rates = np.asarray(spec['recipe'])
    if rates.shape != (meta['layers'], 7) or not np.isfinite(rates).all() or (rates < 0).any() or (rates > 100).any():
        raise ValueError('Invalid frozen allocation')
    out = args.work/args.model/'rebuilt'/spec['id']
    out.mkdir(parents=True, exist_ok=True)
    marker = out/'materialized.json'
    import hashlib
    fingerprint = hashlib.sha256((json.dumps(spec, sort_keys=True) + artifact_manifest_sha() +
                                  sha(common/'dense_preparation_state.json')).encode()).hexdigest()
    if marker.exists():
        state = read(marker)
        if state['fingerprint'] != fingerprint:
            raise RuntimeError('Changed profile or source; use a new work directory')
        if state['status'] == 'complete':
            verify_export(out, state['files'])
            return
    else:
        state = dict(status='running', fingerprint=fingerprint, replay_done=[])
        dump(marker, state)
    install(common/'dense_median', meta['model_type'])
    for name in ('q', 'd', 'pack', 'median'):
        for prefix in ('baseline_uniform_fullcov', 'selected'):
            link_files(common/'baseline_uniform_fullcov'/name, out/prefix/name)
    candidate = out/'regenerated_q'
    for index, alpha in enumerate(spec['alpha']):
        dest = candidate/f'q_layer_{index:02d}.pt'
        if dest.exists():
            continue
        components = [torch.load(common/('components' if k.startswith('S_') else 'weight_metrics')/
                      f'{k}_layer_{index:02d}.pt', map_location='cuda:0', weights_only=True) for k in COMPONENTS]
        q = old.build_q(np.asarray(alpha), components)
        save(dest, q.half())
        del q, components
    par = SimpleNamespace(model_id=str(args.model_path), root=root, output_dir=out,
          adaptive_json=common/'adaptive_dp_centers.json', device_map='cuda:0',
          attn_implementation='sdpa', block_size=128, permutation_seed=11, fallback_rel=.05)
    net, device, patched, medians, adaptive = paircd.load_eval_model(par, rates)
    updater = paircd.LiveArtifactUpdater(args=par, model=net, patched=patched, medians=medians,
                                       recipe=rates, device=device)
    # The persistent updater reads immutable baseline weights, not the previous
    # candidate. Reapply even rejected/restored layers to retain its rounding path.
    for layer in spec['replay_layers']:
        if layer in state['replay_done']:
            continue
        q = torch.load(candidate/f'q_layer_{layer:02d}.pt', map_location='cpu', weights_only=True).float()
        updater.apply(layer, q)
        state['replay_done'].append(layer)
        dump(marker, state)
    del updater, net, patched, medians, adaptive
    gc.collect()
    torch.cuda.empty_cache()
    # A cheap native-vs-folded zero-sparsity audit catches architecture mistakes.
    ids = torch.load(root/'mixed_calibration/input_ids.pt', map_location='cpu', weights_only=True)[:1, :128]
    raw = old.load_model(par)
    raw.eval()
    reference = raw(input_ids=ids.to('cuda:0'), use_cache=False).logits.float().cpu()
    del raw
    gc.collect()
    torch.cuda.empty_cache()
    net, device, patched, medians, adaptive = paircd.load_eval_model(par, np.zeros_like(rates))
    actual = net(input_ids=ids.to(device), use_cache=False).logits.float().cpu()
    audit = dict(tokens=128, **logit_equivalence_audit(reference, actual))
    dump(out/'dense_equivalence_audit.json', audit)
    if not audit['passed']:
        raise RuntimeError(f'Native/folded audit failed: {audit}')
    dump(out/'recipe.json', dict(status='complete', recipe=spec['recipe'], target=spec['target']))
    dump(out/'profile.json', dict(spec, cohort='remote_regenerated_shared_dense_median',
        selected=str(out/'selected'), adaptive_json=str(common/'adaptive_dp_centers.json'),
        q_head=str(common/'lmhead/q_head_beta_0.pt'),
        lmhead_policy_status='requires_validation_on_regenerated_profile',
        historical_bitwise_equivalence=False, ppl_audit_required=True,
        note='Frozen alpha reused without selection; old-median profiles now have shared dense medians'))
    state.update(status='complete', files=exported_files(out), dense_equivalence=audit,
                 historical_bitwise_equivalence=False, benchmark_ready=False)
    dump(marker, state)
    # Registry marker for the original queue: this denotes reconstructed frozen
    # choices, NOT a newly performed sequential acceptance experiment.
    dump(out/'state.json', dict(status='complete', artifact_kind='regenerated_frozen_profile',
         accepted_layers=spec['accepted_layers'], current_calibration_ppl=None,
         historical_reference=spec.get('calibration_reference'), full_ppl_not_rerun=True))


def ppl(args):
    import torch
    import numpy as np
    import run_llama3_layerwise_q_paircd_fullcov_20260830 as paircd
    import run_llama3_layerwise_q_tpe_nonuniform_20260828 as experiment
    from rental_runtime_20260916 import install
    common, root, meta = inputs(args)
    spec = profile_spec(args)
    out = args.work/args.model/'rebuilt'/spec['id']
    marker = read(out/'materialized.json')
    if marker['status'] != 'complete':
        raise RuntimeError('Materialization incomplete')
    verify_export(out, marker['files'])
    install(common/'dense_median', meta['model_type'])
    par = SimpleNamespace(model_id=str(args.model_path), root=root, output_dir=out,
          adaptive_json=common/'adaptive_dp_centers.json', device_map='cuda:0',
          attn_implementation='sdpa', block_size=128, permutation_seed=11, fallback_rel=.05)
    model, device, patched, medians, adaptive = paircd.load_eval_model(par, np.asarray(spec['recipe']))
    ids = torch.load(root/'mixed_calibration/input_ids.pt', map_location='cpu', weights_only=True)
    result = dict(status='running', profile=spec['id'], cohort='remote_regenerated_shared_dense_median',
                  historical_reference_not_acceptance_threshold=spec.get('calibration_reference'))
    path = out/'ppl_audit.json'
    result['mixed_calibration'] = paircd.evaluate_search_ppl(model, np.asarray(ids), device, adaptive)
    dump(path, result)
    from prepare_remote import data_inputs
    evalset = data_inputs()/args.model/'evalset'
    if not evalset.exists():
        evalset = common/'evalset'
    wiki, _ = paircd.read_evalset(evalset)
    result['wikitext'] = paircd.evaluate_search_ppl(model, np.asarray(wiki[:4]), device, adaptive)
    dump(path, result)
    heldout = data_inputs()/args.model/'heldout/heldout_tokens.pt'
    if not heldout.exists():
        heldout = common/'heldout/heldout_tokens.pt'
    ids, _, _ = experiment.load_mixed_evalset(heldout)
    result['heldout_mixed'] = paircd.evaluate_search_ppl(model, ids, device, adaptive)
    result.update(status='complete', sparse_lmhead_validated=False,
                  result_reuse_authorized=False, note='Review differences; no automatic historical-cohort merge')
    dump(path, result)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--action', choices=['head', 'profile', 'ppl'], required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--model-path', type=Path, required=True)
    p.add_argument('--work', type=Path, required=True)
    p.add_argument('--profile')
    p.add_argument('--condition', type=Path)
    args = p.parse_args()
    configure_environment()
    import torch
    torch.manual_seed(0)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    with torch.inference_mode():
        {'head': head, 'profile': materialize, 'ppl': ppl}[args.action](args)


if __name__ == '__main__':
    main()
