"""PPL-guided pairwise coordinate descent for Llama-3.1 Ours.

This is the corrected follow-up to the fixed-step pair search.  Up/Down are
parameterized as ``u = s_up`` and ``g = s_down - s_up``.  For each iteration,
local decoder-block error selects the top donor and receiver coordinates.  Each
of the resulting pair directions is then minimized along a dense MAC-budget
grid.  Short mixed-domain PPL finds the promising transfer amount; the best
amount and its neighbors for every pair are verified on the full 10x2048
mixed-domain calibration set.  The search stops only when no verified pair and
transfer amount improves calibration PPL.
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
import numpy as np
import torch
import run_llama3_mixed_allocation_paper_equivalent_20260815 as paper
import run_llama3_mixed_allocation_search_20260815 as common
ROOT = Path('/repro/data/llama3_1_8b/mixed_calib_c4_alpaca_metamath_code_20260812')
DEFAULT_START = Path('/repro/data/llama3_1_8b/mixed_allocation_paper_equivalent_20260815/recipes.json')
DEFAULT_OUTPUT = Path('/repro/data/llama3_1_8b/mixed_allocation_pair_cd_grid_20260819')
RAW_SITES = common.SITES
RAW_SITE_INDEX = common.SITE_INDEX
SEARCH_SITES = ('q', 'k', 'v', 'o', 'gate', 'u', 'g')
SEARCH_INDEX = {site: index for index, site in enumerate(SEARCH_SITES)}
SEARCH_COSTS = np.asarray([1.0, 0.25, 0.25, 1.0, 3.5, 7.0, 3.5], dtype=np.float64)

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
    parser.add_argument('--start_recipes', type=Path, default=DEFAULT_START)
    parser.add_argument('--start_key', default='teal_paper_s50')
    parser.add_argument('--target', type=float, default=50.0)
    parser.add_argument('--block_size', type=int, default=128)
    parser.add_argument('--permutation_seed', type=int, default=11)
    parser.add_argument('--fallback_rel', type=float, default=0.05)
    parser.add_argument('--costs', default='1,0.25,0.25,1,3.5,3.5,3.5', help='relative MAC costs in q,k,v,o,gate,up,down order')
    parser.add_argument('--max_rate', type=float, default=90.0)
    parser.add_argument('--proxy_len', type=int, default=128)
    parser.add_argument('--screen_len', type=int, default=512)
    parser.add_argument('--donor_top', type=int, default=8)
    parser.add_argument('--receiver_top', type=int, default=8)
    parser.add_argument('--line_grid', default='0.3125:10.0:0.3125', help='inclusive MAC-budget grid as lo:hi:step or comma-separated values')
    parser.add_argument('--full_neighbors', type=int, default=1, help='verify the screen-best grid point and this many adjacent points')
    parser.add_argument('--min_ppl_improvement', type=float, default=0.0001)
    parser.add_argument('--max_updates', type=int, default=12)
    parser.add_argument('--final_samples', type=int, default=4)
    parser.add_argument('--seed', type=int, default=20260819)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--restart', action='store_true', help='ignore state.json while retaining recipe-hash PPL caches')
    return parser.parse_args()

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def parse_grid(raw: str) -> list[float]:
    if ':' not in raw:
        values = [float(value.strip()) for value in raw.split(',') if value.strip()]
    else:
        lo, hi, step = (float(value) for value in raw.split(':'))
        if step <= 0.0 or hi < lo:
            raise ValueError(f'invalid line_grid={raw!r}')
        count = int(math.floor((hi - lo) / step + 1e-09))
        values = [lo + index * step for index in range(count + 1)]
        if values[-1] < hi - 1e-09:
            values.append(hi)
    values = sorted({round(value, 10) for value in values if value > 0.0})
    if not values:
        raise ValueError(f'invalid line_grid={raw!r}')
    return values

def load_recipe(path: Path, key: str) -> np.ndarray:
    payload = json.loads(path.read_text(encoding='utf-8'))
    if key not in payload:
        raise KeyError(f'recipe {key!r} not found in {path}; keys={sorted(payload)}')
    recipe = np.asarray(payload[key], dtype=np.float64)
    if recipe.ndim != 2 or recipe.shape[1] != len(RAW_SITES):
        raise ValueError(f'recipe shape {recipe.shape} is not [layers, {len(RAW_SITES)}]')
    return recipe

def raw_to_search(recipe: np.ndarray) -> np.ndarray:
    raw = np.asarray(recipe, dtype=np.float64)
    result = np.empty_like(raw)
    result[:, :5] = raw[:, :5]
    result[:, SEARCH_INDEX['u']] = raw[:, RAW_SITE_INDEX['up']]
    result[:, SEARCH_INDEX['g']] = raw[:, RAW_SITE_INDEX['down']] - raw[:, RAW_SITE_INDEX['up']]
    return result

def search_to_raw(coords: np.ndarray) -> np.ndarray:
    value = np.asarray(coords, dtype=np.float64)
    result = np.empty_like(value)
    result[:, :5] = value[:, :5]
    result[:, RAW_SITE_INDEX['up']] = value[:, SEARCH_INDEX['u']]
    result[:, RAW_SITE_INDEX['down']] = value[:, SEARCH_INDEX['u']] + value[:, SEARCH_INDEX['g']]
    return result

def valid_search(coords: np.ndarray, max_rate: float) -> bool:
    value = np.asarray(coords, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != len(SEARCH_SITES) or (not np.isfinite(value).all()):
        return False
    if float(value.min()) < -1e-09:
        return False
    raw = search_to_raw(value)
    return bool(float(raw.min()) >= -1e-09 and float(raw.max()) <= max_rate + 1e-09)

def search_weighted_sparsity(coords: np.ndarray) -> float:
    value = np.asarray(coords, dtype=np.float64)
    return float((value * SEARCH_COSTS.reshape(1, -1)).sum() / (value.shape[0] * float(common.TOTAL_COST)))

def layer_value(layer, capture: dict[str, Any]) -> torch.Tensor:
    value = layer(*capture['args'], **capture['kwargs'])
    return value[0] if isinstance(value, tuple) else value

@torch.no_grad()
def refreshed_local_margins(model, layers, controller: common.RecipeController, proxy_ids: torch.Tensor, device: torch.device, recipe: np.ndarray, probe_budget: float, max_rate: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Measure margins on activations generated by the current sparse recipe."""
    controller.set_recipe(recipe)
    captures = common.capture_layer_io(model, proxy_ids, device)
    coords = raw_to_search(recipe)
    donors: list[dict[str, Any]] = []
    receivers: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for layer_idx, layer in enumerate(layers):
        base_raw = recipe[layer_idx].copy()
        base_coords = coords[layer_idx].copy()
        dense_raw = base_raw.copy()
        dense_raw[:] = 0.0
        controller.set_layer(layer_idx, dense_raw)
        reference = layer_value(layer, captures[layer_idx])
        controller.set_layer(layer_idx, base_raw)
        base_output = layer_value(layer, captures[layer_idx])
        base_error = common.relative_block_error(base_output, reference, captures[layer_idx]['input'])
        for coord_idx, site in enumerate(SEARCH_SITES):
            delta = float(probe_budget) / float(SEARCH_COSTS[coord_idx])
            donor_delta = math.nan
            receiver_benefit = math.nan
            donor_coords = coords.copy()
            donor_coords[layer_idx, coord_idx] += delta
            if valid_search(donor_coords, max_rate):
                donor_raw = search_to_raw(donor_coords)[layer_idx]
                controller.set_layer(layer_idx, donor_raw)
                output = layer_value(layer, captures[layer_idx])
                donor_error = common.relative_block_error(output, reference, captures[layer_idx]['input'])
                donor_delta = float(donor_error - base_error)
                donors.append({'layer': layer_idx, 'site': site, 'site_idx': coord_idx, 'proxy_delta': donor_delta})
                del output
            receiver_coords = coords.copy()
            receiver_coords[layer_idx, coord_idx] -= delta
            if valid_search(receiver_coords, max_rate):
                receiver_raw = search_to_raw(receiver_coords)[layer_idx]
                controller.set_layer(layer_idx, receiver_raw)
                output = layer_value(layer, captures[layer_idx])
                receiver_error = common.relative_block_error(output, reference, captures[layer_idx]['input'])
                receiver_benefit = float(base_error - receiver_error)
                receivers.append({'layer': layer_idx, 'site': site, 'site_idx': coord_idx, 'proxy_benefit': receiver_benefit})
                del output
            controller.set_layer(layer_idx, base_raw)
            rows.append({'layer': layer_idx, 'site': site, 'base_value': float(base_coords[coord_idx]), 'probe_delta': delta, 'base_proxy': base_error, 'donor_delta': donor_delta, 'receiver_benefit': receiver_benefit})
        del reference, base_output
    controller.set_recipe(recipe)
    donors.sort(key=lambda row: (float(row['proxy_delta']), row['layer'], row['site_idx']))
    receivers.sort(key=lambda row: (-float(row['proxy_benefit']), row['layer'], row['site_idx']))
    return (donors, receivers, rows)

def transfer(recipe: np.ndarray, donor: dict[str, Any], receiver: dict[str, Any], budget: float, max_rate: float) -> np.ndarray | None:
    donor_key = (int(donor['layer']), int(donor['site_idx']))
    receiver_key = (int(receiver['layer']), int(receiver['site_idx']))
    if donor_key == receiver_key:
        return None
    coords = raw_to_search(recipe)
    coords[donor_key] += float(budget) / float(SEARCH_COSTS[donor_key[1]])
    coords[receiver_key] -= float(budget) / float(SEARCH_COSTS[receiver_key[1]])
    if not valid_search(coords, max_rate):
        return None
    candidate = search_to_raw(coords)
    if abs(common.weighted_sparsity(candidate) - common.weighted_sparsity(recipe)) > 1e-08:
        raise RuntimeError('pair transfer changed the MAC-weighted sparsity budget')
    return candidate

def append_rows(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open('a', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, '') for field in fields})

def score_value(row: dict[str, Any]) -> float:
    return float(row['ppl'])

def save_state(path: Path, recipe: np.ndarray, iteration: int, current_ppl: float, history: list[dict[str, Any]]) -> None:
    common.save_json(path, {'status': 'running', 'recipe': recipe.tolist(), 'recipe_hash': common.recipe_hash(recipe), 'weighted_sparsity': common.weighted_sparsity(recipe), 'iteration': int(iteration), 'current_calibration_ppl': float(current_ppl), 'history': history})

def serializable_args(args: argparse.Namespace) -> dict[str, Any]:
    return {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}

def main() -> None:
    global SEARCH_COSTS
    args = parse_args()
    common.set_costs(args.costs)
    raw_costs = common.COSTS
    SEARCH_COSTS = np.asarray([*raw_costs[:5], raw_costs[RAW_SITE_INDEX['up']] + raw_costs[RAW_SITE_INDEX['down']], raw_costs[RAW_SITE_INDEX['down']]], dtype=np.float64)
    if args.smoke:
        args.proxy_len = 32
        args.screen_len = 64
        args.donor_top = 2
        args.receiver_top = 2
        args.line_grid = '0.625,1.25'
        args.full_neighbors = 0
        args.max_updates = 1
        args.final_samples = 1
    line_grid = parse_grid(args.line_grid)
    probe_budget = min(line_grid)
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    state_path = args.output_dir / 'state.json'
    started = time.perf_counter()
    start_recipe = load_recipe(args.start_recipes, args.start_key)
    if not valid_search(raw_to_search(start_recipe), args.max_rate):
        raise ValueError('start recipe violates transformed-coordinate bounds')
    if abs(common.weighted_sparsity(start_recipe) - args.target) > 1e-06:
        raise ValueError('start recipe does not match target weighted sparsity')
    selection_ids = torch.load(args.root / 'mixed_calibration' / 'selection_input_ids.pt', map_location='cpu', weights_only=True).numpy()
    calibration_ids, calibration_meta = paper.build_mixed_10x2048(args.root, args.output_dir)
    if args.smoke:
        calibration_ids = common.select_domain_rows(selection_ids, args.screen_len)
    screen_ids = common.select_domain_rows(selection_ids, args.screen_len)
    proxy_ids = torch.from_numpy(common.select_domain_rows(selection_ids, args.proxy_len))
    eval_ids, eval_meta = common.read_evalset(args.evalset_dir)
    eval_ids = eval_ids[:int(args.final_samples)]
    print(f'[load] model={args.model_id}', flush=True)
    model = common.load_model(args)
    device = common.input_device(model)
    original_dense = common.evaluate_ppl(model, eval_ids, device)
    common.apply_common_pack(model, args.pack_dir)
    num_layers = len(list(common.decoder_layers(model)))
    if start_recipe.shape[0] != num_layers:
        raise ValueError(f'start recipe has {start_recipe.shape[0]} layers, model has {num_layers}')
    controller = common.RecipeController(num_layers)
    adaptive = common.AdaptiveK4(args.adaptive_json, args.fallback_rel)
    medians = common.MedianBank(args.median_dir, int(model.config.intermediate_size))
    layers = common.patch_model(model, controller, args.d_dir, args.pack_dir, medians, adaptive, args.block_size, args.permutation_seed)
    zero = np.zeros_like(start_recipe)
    controller.set_recipe(zero)
    absorbed_dense = common.evaluate_ppl(model, eval_ids, device)
    dense_delta = abs(absorbed_dense['ppl'] - original_dense['ppl']) / max(original_dense['ppl'], 1e-12)
    if dense_delta > 0.01:
        raise RuntimeError('absorbed dense path differs from original by more than 1%')
    screen_store = common.TrialStore(args.output_dir / 'screen_trials.csv')
    full_store = common.TrialStore(args.output_dir / 'calibration_trials.csv')
    history: list[dict[str, Any]] = []
    current_recipe = start_recipe.copy()
    iteration = 0
    current_ppl = math.nan
    if state_path.exists() and (not args.restart):
        state = json.loads(state_path.read_text(encoding='utf-8'))
        current_recipe = np.asarray(state['recipe'], dtype=np.float64)
        iteration = int(state['iteration'])
        current_ppl = float(state['current_calibration_ppl'])
        history = list(state.get('history', []))
        print(f'[resume] iteration={iteration} recipe={common.recipe_hash(current_recipe)}', flush=True)
    start_calibration = full_store.score(model, controller, adaptive, device, 'pair_cd_grid_start', args.target, 'mixed_calibration', start_recipe, calibration_ids)
    if not math.isfinite(current_ppl):
        current = full_store.score(model, controller, adaptive, device, 'pair_cd_grid_current', args.target, 'mixed_calibration', current_recipe, calibration_ids)
        current_ppl = score_value(current)
    margin_fields = ['iteration', 'layer', 'site', 'base_value', 'probe_delta', 'base_proxy', 'donor_delta', 'receiver_benefit']
    line_fields = ['iteration', 'donor_layer', 'donor_site', 'receiver_layer', 'receiver_site', 'budget', 'recipe_hash', 'screen_ppl', 'calibration_ppl', 'selected_for_full', 'accepted']
    while iteration < int(args.max_updates):
        print(f'[iteration] {iteration} current_ppl={current_ppl:.9f} recipe={common.recipe_hash(current_recipe)}', flush=True)
        donors, receivers, margin_rows = refreshed_local_margins(model, layers, controller, proxy_ids, device, current_recipe, probe_budget, args.max_rate)
        append_rows(args.output_dir / 'local_margins.csv', margin_fields, [{'iteration': iteration, **row} for row in margin_rows])
        directions = []
        for donor in donors[:int(args.donor_top)]:
            for receiver in receivers[:int(args.receiver_top)]:
                if (donor['layer'], donor['site_idx']) == (receiver['layer'], receiver['site_idx']):
                    continue
                directions.append((donor, receiver))
        current_screen = screen_store.score(model, controller, adaptive, device, 'pair_cd_grid_current', args.target, f'screen_{args.screen_len}', current_recipe, screen_ids)
        all_rows: list[dict[str, Any]] = []
        full_candidates: dict[str, dict[str, Any]] = {}
        for donor, receiver in directions:
            direction_rows = []
            for budget_index, budget in enumerate(line_grid):
                candidate = transfer(current_recipe, donor, receiver, budget, args.max_rate)
                if candidate is None:
                    continue
                digest = common.recipe_hash(candidate)
                screen = screen_store.score(model, controller, adaptive, device, 'pair_cd_grid_candidate', args.target, f'screen_{args.screen_len}', candidate, screen_ids)
                row = {'donor': donor, 'receiver': receiver, 'budget': float(budget), 'budget_index': budget_index, 'recipe': candidate, 'recipe_hash': digest, 'screen': screen}
                direction_rows.append(row)
                all_rows.append(row)
            if not direction_rows:
                continue
            direction_rows.sort(key=lambda row: score_value(row['screen']))
            best_index = int(direction_rows[0]['budget_index'])
            selected_indices = {index for index in range(best_index - int(args.full_neighbors), best_index + int(args.full_neighbors) + 1) if 0 <= index < len(line_grid)}
            for row in direction_rows:
                if int(row['budget_index']) in selected_indices:
                    full_candidates.setdefault(row['recipe_hash'], row)
        if not full_candidates:
            print('[stop] no feasible pair directions', flush=True)
            break
        for candidate in full_candidates.values():
            candidate['calibration'] = full_store.score(model, controller, adaptive, device, 'pair_cd_grid_candidate', args.target, 'mixed_calibration', candidate['recipe'], calibration_ids)
        best = min(full_candidates.values(), key=lambda row: score_value(row['calibration']))
        best_ppl = score_value(best['calibration'])
        improvement = float(current_ppl - best_ppl)
        accepted = improvement > float(args.min_ppl_improvement)
        full_hashes = set(full_candidates)
        csv_rows = []
        for row in all_rows:
            csv_rows.append({'iteration': iteration, 'donor_layer': row['donor']['layer'], 'donor_site': row['donor']['site'], 'receiver_layer': row['receiver']['layer'], 'receiver_site': row['receiver']['site'], 'budget': row['budget'], 'recipe_hash': row['recipe_hash'], 'screen_ppl': score_value(row['screen']), 'calibration_ppl': score_value(full_candidates[row['recipe_hash']]['calibration']) if row['recipe_hash'] in full_hashes else '', 'selected_for_full': int(row['recipe_hash'] in full_hashes), 'accepted': int(accepted and row['recipe_hash'] == best['recipe_hash'])})
        append_rows(args.output_dir / 'line_trials.csv', line_fields, csv_rows)
        event = {'iteration': iteration, 'current_calibration_ppl': current_ppl, 'best_calibration_ppl': best_ppl, 'improvement': improvement, 'accepted': accepted, 'donor': {'layer': int(best['donor']['layer']), 'site': str(best['donor']['site'])}, 'receiver': {'layer': int(best['receiver']['layer']), 'site': str(best['receiver']['site'])}, 'budget': float(best['budget']), 'old_recipe_hash': common.recipe_hash(current_recipe), 'new_recipe_hash': best['recipe_hash'], 'screen_current_ppl': score_value(current_screen), 'screen_best_ppl': score_value(best['screen']), 'directions': len(directions), 'screen_trials': len(all_rows), 'full_trials': len(full_candidates)}
        history.append(event)
        if not accepted:
            print(f'[converged] all verified pairs failed; best={best_ppl:.9f} improvement={improvement:.9f}', flush=True)
            save_state(state_path, current_recipe, iteration, current_ppl, history)
            break
        current_recipe = np.asarray(best['recipe'], dtype=np.float64)
        current_ppl = best_ppl
        iteration += 1
        print(f"[accept] donor=L{best['donor']['layer']:02d}/{best['donor']['site']} receiver=L{best['receiver']['layer']:02d}/{best['receiver']['site']} budget={best['budget']:g} ppl={best_ppl:.9f} improvement={improvement:.9f}", flush=True)
        save_state(state_path, current_recipe, iteration, current_ppl, history)
    start_final = full_store.score(model, controller, adaptive, device, 'pair_cd_grid_start', args.target, 'wikitext_4x4096', start_recipe, eval_ids)
    final_row = full_store.score(model, controller, adaptive, device, 'pair_cd_grid_final', args.target, 'wikitext_4x4096', current_recipe, eval_ids)
    common.save_json(args.output_dir / 'summary.json', {'status': 'complete', 'definition': {'search': 'pairwise coordinate descent with PPL line-grid minimization', 'coordinates': 'q,k,v,o,gate,u=s_up,g=s_down-s_up', 'direction_filter': f'top {args.donor_top} donors x top {args.receiver_top} receivers', 'line_grid': line_grid, 'screen': f'mixed-domain PPL on 4x{args.screen_len}', 'acceptance': 'best grid point and neighbors per pair on mixed-domain 10x2048 PPL', 'final': 'WikiText-103 test 4x4096 full sparse PPL', 'fixed': 'mixed Q, random-permuted B128, adaptive-DP K4, dense LM head'}, 'args': serializable_args(args), 'calibration_meta': calibration_meta, 'eval_meta': eval_meta, 'dense_original_ppl': float(original_dense['ppl']), 'dense_absorbed_ppl': float(absorbed_dense['ppl']), 'start': {'recipe_hash': common.recipe_hash(start_recipe), 'calibration_ppl': score_value(start_calibration), 'final_ppl': score_value(start_final)}, 'final': {'recipe_hash': common.recipe_hash(current_recipe), 'weighted_sparsity': common.weighted_sparsity(current_recipe), 'calibration_ppl': current_ppl, 'final_ppl': score_value(final_row)}, 'accepted_updates': sum((bool(row['accepted']) for row in history)), 'history': history, 'recipe': current_recipe.tolist(), 'elapsed_s': time.perf_counter() - started})
    common.save_json(state_path, {'status': 'complete', 'recipe': current_recipe.tolist(), 'recipe_hash': common.recipe_hash(current_recipe), 'weighted_sparsity': common.weighted_sparsity(current_recipe), 'iteration': iteration, 'current_calibration_ppl': current_ppl, 'final_ppl': score_value(final_row), 'history': history})
    print(f"[done] updates={sum((bool(row['accepted']) for row in history))} start_final={score_value(start_final):.9f} final={score_value(final_row):.9f} elapsed={time.perf_counter() - started:.1f}s", flush=True)
if __name__ == '__main__':
    main()
