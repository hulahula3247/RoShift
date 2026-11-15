"""WAS-style constrained TPE allocation for the current Llama-3.1 Ours.

The experiment keeps Ours' mixed-domain rotations, random-permuted B128
selector, lazy correction, adaptive-DP K4 centers, and dense LM head fixed.
Only inter-block sparsity is searched.  Each block's internal seven-projection
recipe comes from the established TEAL-style greedy lookup path.

WAS samples monotonically increasing block sparsities in a target +/- 5 point
window and optimizes WikiText perplexity with 50 Optuna TPE trials.  This
adaptation parameterizes non-negative inter-layer gaps and centers every
candidate at the exact target MAC sparsity.  It therefore retains the WAS
ordering constraint without allowing the +/- epsilon budget drift present in
the paper's rejection formulation.
"""
from __future__ import annotations
import os as _roshift_os
import argparse
import csv
import json
import math
import random
import time
from pathlib import Path
from typing import Any
import matplotlib
import numpy as np
import optuna
import torch
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import run_llama3_mixed_allocation_paper_equivalent_20260815 as paper
import run_llama3_mixed_allocation_search_20260815 as common
ROOT = Path('/repro/data/llama3_1_8b/mixed_calib_c4_alpaca_metamath_code_20260812')
PAPER_ROOT = Path('/repro/data/llama3_1_8b/mixed_allocation_paper_equivalent_20260815')
PAIR_ROOT = Path('/repro/data/llama3_1_8b/mixed_allocation_pair_cd_20260816')
DEFAULT_OUTPUT = Path('/repro/data/llama3_1_8b/mixed_allocation_was_tpe_20260817')
SITES = common.SITES
SITE_INDEX = common.SITE_INDEX
COSTS = common.COSTS

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model_id', default='/repro/models/Meta-Llama-3.1-8B')
    parser.add_argument('--device_map', default='cuda:0')
    parser.add_argument('--attn_implementation', default='sdpa')
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--evalset_dir', type=Path, default=Path(_roshift_os.path.join(_roshift_os.environ['ROSHIFT_ENGINE_ROOT'], 'eval_data')))
    parser.add_argument('--output_dir', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--pack_dir', type=Path, default=ROOT / 'ours_absorbed_pack')
    parser.add_argument('--d_dir', type=Path, default=ROOT / 'output_aware_q' / 'd_cache' / 's_plus_lam_0p2')
    parser.add_argument('--median_dir', type=Path, default=ROOT / 'ours_daware_sparse_gate_medians')
    parser.add_argument('--adaptive_json', type=Path, default=ROOT / 'adaptive_dp_mixed_20260814' / 'adaptive_dp_centers.json')
    parser.add_argument('--source_lookup', type=Path, default=PAPER_ROOT / 'teal_paper_paths.json')
    parser.add_argument('--paper_recipes', type=Path, default=PAPER_ROOT / 'recipes.json')
    parser.add_argument('--pair_recipes', type=Path, default=PAIR_ROOT / 'recipes.json')
    parser.add_argument('--target', type=float, default=50.0)
    parser.add_argument('--window', type=float, default=5.0)
    parser.add_argument('--trials', type=int, default=50)
    parser.add_argument('--startup_trials', type=int, default=10)
    parser.add_argument('--block_size', type=int, default=128)
    parser.add_argument('--permutation_seed', type=int, default=11)
    parser.add_argument('--fallback_rel', type=float, default=0.05)
    parser.add_argument('--costs', default='1,0.25,0.25,1,3.5,3.5,3.5', help='relative MAC costs in q,k,v,o,gate,up,down order')
    parser.add_argument('--lookup_base_step', type=float, default=5.0)
    parser.add_argument('--lookup_last_fraction', type=float, default=0.25)
    parser.add_argument('--max_rate', type=float, default=100.0)
    parser.add_argument('--final_samples', type=int, default=4)
    parser.add_argument('--seed', type=int, default=20260817)
    parser.add_argument('--smoke', action='store_true')
    return parser.parse_args()

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def save_json(path: Path, payload: Any) -> None:
    common.save_json(path, payload)

def load_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding='utf-8'))

def layer_sparsities(recipe: np.ndarray) -> np.ndarray:
    value = common.effective_array(np.asarray(recipe, dtype=np.float64))
    return (value * COSTS.reshape(1, -1)).sum(axis=1) / common.TOTAL_COST

def validate_recipe(recipe: np.ndarray, target: float, window: float, max_rate: float) -> None:
    value = np.asarray(recipe, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != len(SITES):
        raise ValueError(f'unexpected recipe shape {value.shape}')
    if not np.isfinite(value).all():
        raise ValueError('recipe contains a non-finite value')
    if float(value.min()) < -1e-08 or float(value.max()) > float(max_rate) + 1e-08:
        raise ValueError(f'recipe range [{value.min():.6f}, {value.max():.6f}] is invalid')
    if np.any(value[:, SITE_INDEX['up']] > value[:, SITE_INDEX['down']] + 1e-08):
        raise ValueError('recipe violates nested Up/Down support')
    effective = common.weighted_sparsity(value)
    if abs(effective - float(target)) > 1e-06:
        raise ValueError(f'weighted sparsity {effective:.9f} != {target:.9f}')
    blocks = layer_sparsities(value)
    if np.any(np.diff(blocks) < -1e-07):
        raise ValueError(f'block sparsities are not monotonic: {blocks.tolist()}')
    if blocks[0] < float(target) - float(window) - 0.5:
        raise ValueError(f'first block sparsity {blocks[0]:.6f} is too low')
    if blocks[-1] > float(target) + float(window) + 0.5:
        raise ValueError(f'last block sparsity {blocks[-1]:.6f} is too high')

def copy_lookup(source: Path, destination: Path) -> dict[int, list[dict[str, Any]]]:
    if destination.exists():
        payload = load_json(destination)
    else:
        payload = load_json(source)
        payload['definition'] = {'source': str(source), 'metric': 'TEAL last-25% decoder-layer output L2', 'purpose': 'WAS inter-block lookup extended to target + 5 points'}
        save_json(destination, payload)
    paths = payload.get('paths', {})
    if not paths:
        raise ValueError(f'lookup has no layer paths: {source}')
    return {int(key): list(value) for key, value in paths.items()}

@torch.no_grad()
def extend_lookup(model: torch.nn.Module, layers: list[common.AllocationDecoderLayer], controller: common.RecipeController, calibration_ids: np.ndarray, device: torch.device, paths: dict[int, list[dict[str, Any]]], target: float, base_step: float, last_fraction: float, max_rate: float, output: Path) -> dict[int, list[dict[str, Any]]]:
    """Continue each existing intra-block greedy path only when required."""
    zero = np.zeros_like(controller.recipe)
    started = time.perf_counter()
    for layer_idx, layer in enumerate(layers):
        entries = paths[layer_idx]
        if float(entries[-1]['effective']) >= float(target) - 1e-09:
            continue
        controller.set_recipe(zero)
        capture = paper.capture_one_layer(model, layer, torch.from_numpy(np.asarray(calibration_ids, dtype=np.int64)), device)
        rates = np.asarray(entries[-1]['rates'], dtype=np.float64)
        while common.weighted_sparsity(rates.reshape(1, -1)) < float(target) - 1e-09:
            current = common.weighted_sparsity(rates.reshape(1, -1))
            best: tuple[float, int, np.ndarray] | None = None
            for site_idx, site in enumerate(SITES):
                candidate = rates.copy()
                candidate[site_idx] += float(base_step) / float(COSTS[site_idx])
                if candidate[site_idx] > float(max_rate) + 1e-09:
                    continue
                if candidate[SITE_INDEX['up']] > candidate[SITE_INDEX['down']] + 1e-09:
                    continue
                effective = common.weighted_sparsity(candidate.reshape(1, -1))
                if effective <= current + 1e-12:
                    continue
                controller.set_layer(layer_idx, candidate)
                value = layer(*capture['args'], **capture['kwargs'])
                value = value[0] if isinstance(value, tuple) else value
                score = paper.teal_activation_error(value, capture['output'], last_fraction)
                if best is None or (score, site_idx) < (best[0], best[1]):
                    best = (score, site_idx, candidate)
                del value
            if best is None:
                raise RuntimeError(f'lookup extension exhausted layer {layer_idx}')
            score, chosen_idx, rates = best
            controller.set_layer(layer_idx, rates)
            entries.append({'effective': common.weighted_sparsity(rates.reshape(1, -1)), 'rates': rates.tolist(), 'activation_error': float(score), 'chosen': SITES[chosen_idx]})
        controller.set_recipe(zero)
        paths[layer_idx] = entries
        save_json(output, {'definition': {'metric': 'TEAL last-25% decoder-layer output L2', 'calibration': 'mixed-domain 10x2048', 'base_step': float(base_step), 'extended_target': float(target)}, 'elapsed_s': time.perf_counter() - started, 'paths': {str(key): value for key, value in paths.items()}})
        print(f"[lookup] layer={layer_idx:02d} max={entries[-1]['effective']:.6f} elapsed={time.perf_counter() - started:.1f}s", flush=True)
        del capture
        torch.cuda.empty_cache()
    controller.set_recipe(zero)
    return paths

def block_rates_from_trial(trial: optuna.Trial, num_layers: int, target: float, window: float) -> np.ndarray:
    """Map TPE-sampled non-negative gaps to an exact-mean monotonic schedule."""
    gaps = np.asarray([trial.suggest_float(f'gap_{index:02d}', 0.0, 1.0) for index in range(1, int(num_layers))], dtype=np.float64)
    if float(gaps.sum()) <= 1e-12:
        quantiles = np.linspace(0.0, 1.0, int(num_layers), dtype=np.float64)
    else:
        quantiles = np.concatenate(([0.0], np.cumsum(gaps) / gaps.sum()))
    center = float(quantiles.mean())
    left = float(window) / max(center, 1e-12)
    right = float(window) / max(1.0 - center, 1e-12)
    maximum_span = min(left, right)
    span_fraction = trial.suggest_float('span_fraction', 0.0, 1.0)
    span = float(span_fraction) * maximum_span
    result = float(target) + span * (quantiles - center)
    if abs(float(result.mean()) - float(target)) > 1e-10:
        raise RuntimeError('block-rate parameterization changed the target mean')
    if np.any(np.diff(result) < -1e-10):
        raise RuntimeError('block-rate parameterization is not monotonic')
    return result

def lookup_recipe(paths: dict[int, list[dict[str, Any]]], block_rates: np.ndarray, target: float, max_rate: float) -> np.ndarray:
    rows = []
    for layer_idx, block_rate in enumerate(np.asarray(block_rates, dtype=np.float64)):
        entry = min(paths[int(layer_idx)], key=lambda row: abs(float(row['effective']) - float(block_rate)))
        rows.append(np.asarray(entry['rates'], dtype=np.float64))
    recipe = np.stack(rows)
    residual = float(target) - common.weighted_sparsity(recipe)
    gate_delta = residual * common.TOTAL_COST / float(COSTS[SITE_INDEX['gate']])
    recipe[:, SITE_INDEX['gate']] += gate_delta
    gate = recipe[:, SITE_INDEX['gate']]
    if float(gate.min()) < -1e-09 or float(gate.max()) > float(max_rate) + 1e-09:
        raise ValueError(f'Gate budget correction {gate_delta:.6f} exceeds recipe bounds')
    return recipe

def enqueue_seed_trials(study: optuna.Study, num_layers: int) -> None:

    def params(gaps: np.ndarray, span_fraction: float) -> dict[str, float]:
        result = {f'gap_{index:02d}': float(gaps[index - 1]) for index in range(1, int(num_layers))}
        result['span_fraction'] = float(span_fraction)
        return result
    linear = np.ones(int(num_layers) - 1, dtype=np.float64)
    late = np.linspace(0.15, 1.0, int(num_layers) - 1, dtype=np.float64)
    early = late[::-1].copy()
    seeds = (params(linear, 0.0), params(linear, 0.5), params(linear, 1.0), params(late, 0.8), params(early, 0.8))
    for seed in seeds:
        study.enqueue_trial(seed)

def write_trial_csv(study: optuna.Study, output: Path) -> None:
    fields = ['trial', 'state', 'ppl', 'recipe_hash', 'weighted_sparsity', 'block_min', 'block_max', 'span', 'elapsed_s', 'block_rates_json']
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for trial in study.trials:
            attrs = trial.user_attrs
            rates = attrs.get('block_rates', [])
            writer.writerow({'trial': trial.number, 'state': trial.state.name, 'ppl': '' if trial.value is None else float(trial.value), 'recipe_hash': attrs.get('recipe_hash', ''), 'weighted_sparsity': attrs.get('weighted_sparsity', ''), 'block_min': min(rates) if rates else '', 'block_max': max(rates) if rates else '', 'span': max(rates) - min(rates) if rates else '', 'elapsed_s': attrs.get('elapsed_s', ''), 'block_rates_json': json.dumps(rates, separators=(',', ':'))})

def plot_results(study: optuna.Study, baseline_ppl: float, best_recipe: np.ndarray, output_dir: Path, target: float, window: float) -> None:
    completed = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
    values = np.asarray([float(trial.value) for trial in completed], dtype=np.float64)
    running = np.minimum.accumulate(values)
    fig, axis = plt.subplots(figsize=(8.0, 4.8))
    axis.scatter(range(1, len(values) + 1), values, s=26, color='#e68a2e', alpha=0.68)
    axis.plot(range(1, len(values) + 1), running, color='#b55220', linewidth=2.0, label='Best so far')
    axis.axhline(float(baseline_ppl), color='#666666', linestyle='--', linewidth=1.5, label='Uniform block budget')
    axis.set_xlabel('Completed TPE trial')
    axis.set_ylabel('Mixed-calibration PPL')
    axis.set_title('WAS-style constrained TPE convergence')
    axis.grid(axis='y', alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / 'tpe_convergence.png', dpi=180)
    plt.close(fig)
    blocks = layer_sparsities(best_recipe)
    fig, axis = plt.subplots(figsize=(8.0, 4.8))
    axis.plot(range(len(blocks)), blocks, marker='o', markersize=3.5, color='#d97721')
    axis.axhline(float(target), color='#666666', linestyle='--', linewidth=1.3, label=f'Sparsity target: {target:g}%')
    axis.fill_between(range(len(blocks)), float(target) - float(window), float(target) + float(window), color='#eeeeee', alpha=0.8, zorder=0)
    axis.set_xlabel('Layer')
    axis.set_ylabel('Layer sparsity (%)')
    axis.set_title('Best WAS-style block allocation')
    axis.set_xticks(range(0, len(blocks), 2))
    axis.grid(axis='y', alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / 'best_block_sparsity.png', dpi=180)
    plt.close(fig)

def write_final_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fields = ('method', 'weighted_sparsity', 'ppl', 'mean_nll', 'total_tokens')
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})

def main() -> None:
    global COSTS
    args = parse_args()
    common.set_costs(args.costs)
    COSTS = common.COSTS
    if args.smoke:
        args.trials = 3
        args.startup_trials = 1
        args.final_samples = 1
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    calibration_ids, calibration_meta = paper.build_mixed_10x2048(args.root, args.output_dir)
    if args.smoke:
        calibration_ids = calibration_ids[:1, :128]
    eval_ids, eval_meta = common.read_evalset(args.evalset_dir)
    eval_ids = eval_ids[:int(args.final_samples)]
    print(f'[load] model={args.model_id}', flush=True)
    model = common.load_model(args)
    device = common.input_device(model)
    original_dense = common.evaluate_ppl(model, eval_ids, device)
    print(f"[dense-original] ppl={original_dense['ppl']:.6f}", flush=True)
    common.apply_common_pack(model, args.pack_dir)
    num_layers = len(list(common.decoder_layers(model)))
    controller = common.RecipeController(num_layers)
    adaptive = common.AdaptiveK4(args.adaptive_json, args.fallback_rel)
    medians = common.MedianBank(args.median_dir, int(model.config.intermediate_size))
    layers = common.patch_model(model, controller, args.d_dir, args.pack_dir, medians, adaptive, args.block_size, args.permutation_seed)
    zero = np.zeros((num_layers, len(SITES)), dtype=np.float64)
    controller.set_recipe(zero)
    absorbed_dense = common.evaluate_ppl(model, eval_ids, device)
    dense_delta = abs(absorbed_dense['ppl'] - original_dense['ppl']) / max(original_dense['ppl'], 1e-12)
    print(f"[dense-absorbed] ppl={absorbed_dense['ppl']:.6f} relative_delta={100.0 * dense_delta:.4f}%", flush=True)
    if dense_delta > 0.01:
        raise RuntimeError('absorbed dense path differs from original by more than 1%')
    lookup_path = args.output_dir / 'was_greedy_lookup_to55.json'
    paths = copy_lookup(args.source_lookup, lookup_path)
    lookup_started = time.perf_counter()
    paths = extend_lookup(model, layers, controller, calibration_ids, device, paths, args.target + args.window, args.lookup_base_step, args.lookup_last_fraction, args.max_rate, lookup_path)
    lookup_elapsed = time.perf_counter() - lookup_started
    baseline_recipe = lookup_recipe(paths, np.full(num_layers, args.target, dtype=np.float64), args.target, args.max_rate)
    validate_recipe(baseline_recipe, args.target, args.window, args.max_rate)
    store = common.TrialStore(args.output_dir / 'ppl_trials.csv')
    baseline_calibration = store.score(model, controller, adaptive, device, 'was_uniform_block', args.target, 'mixed_calibration', baseline_recipe, calibration_ids)
    database = (args.output_dir / 'optuna.db').resolve()
    sampler = optuna.samplers.TPESampler(seed=args.seed, n_startup_trials=int(args.startup_trials))
    study = optuna.create_study(study_name=f'roshift_was_tpe_{num_layers}l_s{int(round(args.target)):02d}', direction='minimize', sampler=sampler, storage=f'sqlite:///{database}', load_if_exists=True)
    if not study.trials:
        enqueue_seed_trials(study, num_layers)

    def objective(trial: optuna.Trial) -> float:
        trial_started = time.perf_counter()
        requested = block_rates_from_trial(trial, num_layers, args.target, args.window)
        recipe = lookup_recipe(paths, requested, args.target, args.max_rate)
        validate_recipe(recipe, args.target, args.window, args.max_rate)
        realized = layer_sparsities(recipe)
        row = store.score(model, controller, adaptive, device, 'was_tpe', args.target, 'mixed_calibration', recipe, calibration_ids)
        elapsed = time.perf_counter() - trial_started
        trial.set_user_attr('requested_block_rates', requested.tolist())
        trial.set_user_attr('block_rates', realized.tolist())
        trial.set_user_attr('recipe', recipe.tolist())
        trial.set_user_attr('recipe_hash', common.recipe_hash(recipe))
        trial.set_user_attr('weighted_sparsity', common.weighted_sparsity(recipe))
        trial.set_user_attr('elapsed_s', elapsed)
        print(f"[tpe] trial={trial.number:02d} ppl={float(row['ppl']):.6f} blocks={realized[0]:.3f}->{realized[-1]:.3f} elapsed={elapsed:.1f}s", flush=True)
        return float(row['ppl'])
    completed = sum((trial.state == optuna.trial.TrialState.COMPLETE for trial in study.trials))
    search_started = time.perf_counter()
    attempts = 0
    while completed < int(args.trials):
        remaining = int(args.trials) - completed
        study.optimize(objective, n_trials=remaining, gc_after_trial=True, show_progress_bar=False, catch=(ValueError,))
        attempts += remaining
        completed = sum((trial.state == optuna.trial.TrialState.COMPLETE for trial in study.trials))
        if attempts >= 3 * int(args.trials) and completed < int(args.trials):
            raise RuntimeError(f'only {completed} valid trials after {attempts} attempts')
    search_elapsed = time.perf_counter() - search_started
    completed = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
    if len(completed) < int(args.trials):
        raise RuntimeError(f'only {len(completed)} of {args.trials} TPE trials completed')
    best_trial = study.best_trial
    best_recipe = np.asarray(best_trial.user_attrs['recipe'], dtype=np.float64)
    validate_recipe(best_recipe, args.target, args.window, args.max_rate)
    target_label = f's{int(round(args.target)):02d}'
    uniform_key = f'was_uniform_block_{target_label}'
    best_key = f'was_tpe_{target_label}'
    save_json(args.output_dir / 'recipes.json', {uniform_key: baseline_recipe.tolist(), best_key: best_recipe.tolist()})
    write_trial_csv(study, args.output_dir / 'tpe_trials.csv')
    plot_results(study, float(baseline_calibration['ppl']), best_recipe, args.output_dir, args.target, args.window)
    final_rows: list[dict[str, Any]] = [{'method': 'dense_original', 'weighted_sparsity': 0.0, **original_dense}, {'method': 'dense_absorbed', 'weighted_sparsity': 0.0, **absorbed_dense}]
    comparison_recipes: list[tuple[str, np.ndarray]] = [(uniform_key, baseline_recipe)]
    if args.paper_recipes.exists():
        source = load_json(args.paper_recipes)
        paper_uniform_key = f'uniform_{target_label}'
        paper_teal_key = f'teal_paper_{target_label}'
        if paper_uniform_key in source:
            comparison_recipes.append((paper_uniform_key, np.asarray(source[paper_uniform_key], dtype=np.float64)))
        if paper_teal_key in source:
            comparison_recipes.append((paper_teal_key, np.asarray(source[paper_teal_key], dtype=np.float64)))
    if args.pair_recipes.exists():
        source = load_json(args.pair_recipes)
        if 'pair_cd_final' in source:
            comparison_recipes.append(('pair_cd_final', np.asarray(source['pair_cd_final'], dtype=np.float64)))
    comparison_recipes.append((best_key, best_recipe))
    scored_by_digest: dict[str, dict[str, Any]] = {}
    for method, recipe in comparison_recipes:
        digest = common.recipe_hash(recipe)
        if digest in scored_by_digest:
            final_rows.append({**scored_by_digest[digest], 'method': method})
            continue
        row = store.score(model, controller, adaptive, device, method, args.target, 'wikitext_test_4x4096', recipe, eval_ids)
        result = {'method': method, 'weighted_sparsity': common.weighted_sparsity(recipe), 'ppl': float(row['ppl']), 'mean_nll': float(row['mean_nll']), 'total_nll': float(row['total_nll']), 'total_tokens': int(row['total_tokens'])}
        final_rows.append(result)
        scored_by_digest[digest] = result
    write_final_csv(final_rows, args.output_dir / 'final_ppl.csv')
    baseline_final = next((row for row in final_rows if row['method'] == uniform_key))
    best_final = next((row for row in final_rows if row['method'] == best_key))
    summary = {'status': 'complete', 'definition': {'method': f'WAS-style constrained {num_layers}-block Optuna TPE', 'paper_constraints': 'monotonically increasing block sparsity in target +/- 5 points', 'exact_budget_adaptation': f'non-negative gap parameterization centered at exactly {args.target:g}% MAC sparsity', 'intra_block': 'existing TEAL-style greedy seven-projection lookup', 'fixed': 'mixed Q, random-permuted B128, adaptive-DP K4, dense LM head', 'search_objective': 'mixed-domain 10x2048 PPL', 'final': 'WikiText-103 test 4x4096 full sparse PPL'}, 'args': {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}, 'calibration_meta': calibration_meta, 'eval_meta': eval_meta, 'trials': len(completed), 'best_trial': int(best_trial.number), 'best_calibration_ppl': float(best_trial.value), 'baseline_calibration_ppl': float(baseline_calibration['ppl']), 'best_block_sparsities': layer_sparsities(best_recipe).tolist(), 'best_projection_summary': paper.projection_summary(best_recipe), 'final_ppl': final_rows, 'improvement': {'calibration_ppl_absolute': float(baseline_calibration['ppl']) - float(best_trial.value), 'calibration_ppl_percent': 100.0 * (float(baseline_calibration['ppl']) - float(best_trial.value)) / float(baseline_calibration['ppl']), 'wikitext_ppl_absolute': float(baseline_final['ppl']) - float(best_final['ppl']), 'wikitext_ppl_percent': 100.0 * (float(baseline_final['ppl']) - float(best_final['ppl'])) / float(baseline_final['ppl'])}, 'timing_s': {'lookup_extension': lookup_elapsed, 'tpe_search': search_elapsed, 'total': time.perf_counter() - started}}
    save_json(args.output_dir / 'summary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)
if __name__ == '__main__':
    main()
