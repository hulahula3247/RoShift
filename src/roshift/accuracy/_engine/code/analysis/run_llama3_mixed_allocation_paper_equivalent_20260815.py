"""Paper-scale TEAL and LaRoSA allocation search for current Ours.

Only the 32 x 7 body sparsity recipe is searched.  The mixed-domain Q,
random-permuted B128 selector, lazy correction, adaptive-DP K4 centers, and
dense LM head are held fixed.  Search calibration is exactly 10 x 2048 mixed
tokens for both methods; final PPL uses the existing 4 x 4096 WikiText-103
test protocol so results remain comparable with prior Ours experiments.
"""
from __future__ import annotations
import os as _roshift_os
import argparse
import csv
import hashlib
import json
import math
import random
import time
from pathlib import Path
from typing import Any
import numpy as np
import torch
import run_llama3_mixed_allocation_search_20260815 as common
SITES = common.SITES
SITE_INDEX = common.SITE_INDEX
COSTS = common.COSTS
TOTAL_COST = common.TOTAL_COST

def parse_args() -> argparse.Namespace:
    root = common.ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model_id', default='/repro/models/Meta-Llama-3.1-8B')
    parser.add_argument('--device_map', default='cuda:0')
    parser.add_argument('--attn_implementation', default='sdpa')
    parser.add_argument('--root', type=Path, default=root)
    parser.add_argument('--output_dir', type=Path, default=Path('/repro/data/llama3_1_8b/mixed_allocation_paper_equivalent_20260815'))
    parser.add_argument('--evalset_dir', type=Path, default=Path(_roshift_os.path.join(_roshift_os.environ['ROSHIFT_ENGINE_ROOT'], 'eval_data')))
    parser.add_argument('--pack_dir', type=Path, default=root / 'ours_absorbed_pack')
    parser.add_argument('--d_dir', type=Path, default=root / 'output_aware_q' / 'd_cache' / 's_plus_lam_0p2')
    parser.add_argument('--median_dir', type=Path, default=root / 'ours_daware_sparse_gate_medians')
    parser.add_argument('--adaptive_json', type=Path, default=root / 'adaptive_dp_mixed_20260814' / 'adaptive_dp_centers.json')
    parser.add_argument('--targets', default='40,50')
    parser.add_argument('--block_size', type=int, default=128)
    parser.add_argument('--permutation_seed', type=int, default=11)
    parser.add_argument('--fallback_rel', type=float, default=0.05)
    parser.add_argument('--costs', default='1,0.25,0.25,1,3.5,3.5,3.5', help='relative MAC costs in q,k,v,o,gate,up,down order')
    parser.add_argument('--teal_base_step', type=float, default=5.0)
    parser.add_argument('--teal_last_fraction', type=float, default=0.25)
    parser.add_argument('--larosa_grid', default='0.70:1.20:0.05')
    parser.add_argument('--final_samples', type=int, default=4)
    parser.add_argument('--seed', type=int, default=20260815)
    parser.add_argument('--force_teal', action='store_true')
    parser.add_argument('--skip_larosa', action='store_true', help='Build and evaluate only Uniform and TEAL recipes.')
    parser.add_argument('--smoke', action='store_true')
    return parser.parse_args()

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)

def build_mixed_10x2048(root: Path, output_dir: Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Pack the established 50/18.75/15.625/15.625 mix into 20,480 tokens."""
    source_path = root / 'mixed_calibration' / 'input_ids.pt'
    source = torch.load(source_path, map_location='cpu', weights_only=True).numpy()
    if source.shape[0] < 32 or source.shape[1] < 4096:
        raise ValueError(f'unexpected mixed calibration shape: {source.shape}')
    names = ('general_c4', 'instruction_alpaca', 'math_metamathqa', 'code_codealpaca')
    row_ranges = ((0, 16), (16, 22), (22, 27), (27, 32))
    token_counts = (10240, 3840, 3200, 3200)
    pieces = []
    domains = []
    for name, (start, end), count in zip(names, row_ranges, token_counts):
        flat = np.asarray(source[start:end], dtype=np.int64).reshape(-1)
        if flat.size < count:
            raise ValueError(f'not enough tokens for {name}: {flat.size} < {count}')
        piece = flat[:count].copy()
        pieces.append(piece)
        domains.append({'name': name, 'tokens': int(count), 'ratio': float(count / sum(token_counts)), 'source_rows': [int(start), int(end - 1)]})
    packed = np.concatenate(pieces).reshape(10, 2048)
    digest = hashlib.sha256(packed.tobytes()).hexdigest()
    artifact = output_dir / 'mixed_calibration_10x2048.pt'
    artifact.parent.mkdir(parents=True, exist_ok=True)
    torch.save(torch.from_numpy(packed), artifact)
    meta = {'shape': list(packed.shape), 'tokens': int(packed.size), 'sha256': digest, 'source': str(source_path), 'packing': 'domain streams concatenated at exact token budgets then reshaped', 'domains': domains}
    save_json(output_dir / 'mixed_calibration_10x2048.json', meta)
    return (packed, meta)

def detach_clone(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, tuple):
        return tuple((detach_clone(item) for item in value))
    if isinstance(value, list):
        return [detach_clone(item) for item in value]
    if isinstance(value, dict):
        return {key: detach_clone(item) for key, item in value.items()}
    return value

@torch.no_grad()
def capture_one_layer(model: torch.nn.Module, layer: common.AllocationDecoderLayer, input_ids: torch.Tensor, device: torch.device) -> dict[str, Any]:
    capture: dict[str, Any] = {}

    def pre_hook(_module, args, kwargs):
        capture['args'] = detach_clone(args)
        capture['kwargs'] = detach_clone(kwargs)

    def post_hook(_module, _args, _kwargs, output):
        value = output[0] if isinstance(output, tuple) else output
        capture['output'] = value.detach().clone()
    handles = [layer.register_forward_pre_hook(pre_hook, with_kwargs=True), layer.register_forward_hook(post_hook, with_kwargs=True)]
    try:
        ids = input_ids.to(device)
        mask = torch.ones_like(ids)
        backbone = getattr(model, 'model', model)
        result = backbone(input_ids=ids, attention_mask=mask, use_cache=False)
        del result, ids, mask
    finally:
        for handle in handles:
            handle.remove()
    if not {'args', 'kwargs', 'output'}.issubset(capture):
        raise RuntimeError('failed to capture decoder-layer input/output')
    return capture

def teal_activation_error(output: torch.Tensor, target: torch.Tensor, last_fraction: float) -> float:
    start = int(output.shape[1] * (1.0 - float(last_fraction)))
    difference = target[:, start:].float() - output[:, start:].float()
    return float(torch.linalg.vector_norm(difference, dim=1).mean().item())

@torch.no_grad()
def build_teal_paths(model: torch.nn.Module, layers: list[common.AllocationDecoderLayer], controller: common.RecipeController, calibration_ids: np.ndarray, device: torch.device, max_target: float, base_step: float, last_fraction: float, output: Path, force: bool) -> dict[int, list[dict[str, Any]]]:
    paths: dict[int, list[dict[str, Any]]] = {}
    if output.exists() and (not force):
        payload = json.loads(output.read_text(encoding='utf-8'))
        paths = {int(key): value for key, value in payload.get('paths', {}).items()}
        if len(paths) == len(layers):
            return paths
        if paths:
            print(f'[teal-paper] resume {len(paths)}/{len(layers)} completed layers', flush=True)
    started = time.perf_counter()
    calibration = torch.from_numpy(np.asarray(calibration_ids, dtype=np.int64))
    controller.set_recipe(np.zeros_like(controller.recipe))
    for layer_idx, layer in enumerate(layers):
        if layer_idx in paths:
            continue
        controller.set_recipe(np.zeros_like(controller.recipe))
        capture = capture_one_layer(model, layer, calibration, device)
        rates = np.zeros(len(SITES), dtype=np.float64)
        entries = [{'effective': 0.0, 'rates': rates.tolist(), 'activation_error': 0.0, 'chosen': 'start'}]
        while common.weighted_sparsity(rates.reshape(1, -1)) < max_target - 1e-09:
            best: tuple[float, int, np.ndarray] | None = None
            current_effective = common.weighted_sparsity(rates.reshape(1, -1))
            for site_idx, site in enumerate(SITES):
                candidate = rates.copy()
                candidate[site_idx] += float(base_step) / float(COSTS[site_idx])
                if candidate[site_idx] > 100.0 + 1e-09:
                    continue
                if candidate[SITE_INDEX['up']] > candidate[SITE_INDEX['down']] + 1e-09:
                    continue
                effective = common.weighted_sparsity(candidate.reshape(1, -1))
                if effective <= current_effective + 1e-12:
                    continue
                controller.set_layer(layer_idx, candidate)
                value = layer(*capture['args'], **capture['kwargs'])
                value = value[0] if isinstance(value, tuple) else value
                score = teal_activation_error(value, capture['output'], last_fraction)
                if best is None or (score, site_idx) < (best[0], best[1]):
                    best = (score, site_idx, candidate)
                del value
            if best is None:
                raise RuntimeError(f'TEAL search exhausted candidates at layer {layer_idx}')
            score, chosen_idx, rates = best
            controller.set_layer(layer_idx, rates)
            entries.append({'effective': common.weighted_sparsity(rates.reshape(1, -1)), 'rates': rates.tolist(), 'activation_error': float(score), 'chosen': SITES[chosen_idx]})
        controller.set_layer(layer_idx, np.zeros(len(SITES), dtype=np.float64))
        paths[layer_idx] = entries
        del capture
        torch.cuda.empty_cache()
        elapsed = time.perf_counter() - started
        print(f'[teal-paper] layer={layer_idx:02d} steps={len(entries) - 1} elapsed={elapsed:.1f}s', flush=True)
        save_json(output, {'definition': {'calibration': 'mixed-domain 10x2048', 'metric': 'TEAL last-25% decoder-layer output L2', 'base_step': float(base_step), 'max_requested_target': float(max_target), 'note': 'search stops at the largest requested sparsity; later path entries cannot alter earlier greedy choices'}, 'elapsed_s': elapsed, 'paths': {str(key): value for key, value in paths.items()}})
    controller.set_recipe(np.zeros_like(controller.recipe))
    return paths

def teal_recipe(paths: dict[int, list[dict[str, Any]]], target: float) -> np.ndarray:
    rows = []
    for layer_idx in sorted(paths):
        entry = min(paths[layer_idx], key=lambda row: abs(float(row['effective']) - target))
        rows.append(np.asarray(entry['rates'], dtype=np.float64))
    return np.stack(rows)

def larosa_recipe(target: float, alpha_att: float, alpha_ffn: float, layers: int) -> tuple[np.ndarray, dict[str, float]]:
    """Apply LaRoSA keep-ratio coefficients to Ours' exact MAC groups."""
    alpha_o = 2.5 - 1.5 * float(alpha_att)
    alpha_mid = 1.5 - 0.5 * float(alpha_ffn)
    keep = 1.0 - float(target) / 100.0
    coefficients = np.asarray([alpha_att, alpha_att, alpha_att, alpha_o, alpha_ffn, alpha_mid, alpha_mid], dtype=np.float64)
    rates = 100.0 * (1.0 - keep * coefficients)
    if np.any(rates < -1e-08) or np.any(rates > 100.0 + 1e-08):
        raise ValueError(f'invalid LaRoSA rates: {rates.tolist()}')
    recipe = np.repeat(np.clip(rates, 0.0, 100.0).reshape(1, -1), int(layers), axis=0)
    return (recipe, {'alpha_att': float(alpha_att), 'alpha_o': float(alpha_o), 'alpha_ffn': float(alpha_ffn), 'alpha_mid': float(alpha_mid)})

class TrialStore:
    FIELDS = ('method', 'target', 'phase', 'alpha_att', 'alpha_ffn', 'recipe_hash', 'weighted_sparsity', 'ppl', 'mean_nll', 'total_nll', 'total_tokens', 'elapsed_s', 'recipe_json')

    def __init__(self, path: Path) -> None:
        self.path = path
        self.cache: dict[tuple[str, str, str], dict[str, str]] = {}
        if path.exists():
            with path.open('r', encoding='utf-8', newline='') as handle:
                for row in csv.DictReader(handle):
                    self.cache[row['method'], row['phase'], row['recipe_hash']] = row

    def score(self, model: torch.nn.Module, controller: common.RecipeController, adaptive: common.AdaptiveK4, device: torch.device, method: str, target: float, phase: str, recipe: np.ndarray, ids: np.ndarray, alpha_att: float | str='', alpha_ffn: float | str='') -> dict[str, Any]:
        digest = common.recipe_hash(recipe)
        key = (method, phase, digest)
        if key in self.cache:
            row = self.cache[key]
            return {**row, 'ppl': float(row['ppl']), 'mean_nll': float(row['mean_nll']), 'total_nll': float(row['total_nll']), 'total_tokens': int(row['total_tokens']), 'elapsed_s': float(row['elapsed_s']), 'recipe': np.asarray(json.loads(row['recipe_json']), dtype=np.float64)}
        controller.set_recipe(recipe)
        adaptive.reset()
        started = time.perf_counter()
        result = common.evaluate_ppl(model, ids, device)
        row = {'method': method, 'target': float(target), 'phase': phase, 'alpha_att': alpha_att, 'alpha_ffn': alpha_ffn, 'recipe_hash': digest, 'weighted_sparsity': common.weighted_sparsity(recipe), **result, 'elapsed_s': time.perf_counter() - started, 'recipe_json': json.dumps(recipe.tolist(), separators=(',', ':'))}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        exists = self.path.exists() and self.path.stat().st_size > 0
        with self.path.open('a', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=self.FIELDS)
            if not exists:
                writer.writeheader()
            writer.writerow({field: row[field] for field in self.FIELDS})
        self.cache[key] = {field: str(row[field]) for field in self.FIELDS}
        print(f"[ppl] method={method} target={target:g} phase={phase} ppl={result['ppl']:.6f} elapsed={row['elapsed_s']:.2f}s", flush=True)
        return {**row, 'recipe': recipe.copy(), 'fallback': adaptive.stats()}

def projection_summary(recipe: np.ndarray) -> dict[str, Any]:
    value = np.asarray(recipe, dtype=np.float64)
    return {'weighted_sparsity': common.weighted_sparsity(value), 'projection_mean': {site: float(value[:, idx].mean()) for idx, site in enumerate(SITES)}, 'projection_min': {site: float(value[:, idx].min()) for idx, site in enumerate(SITES)}, 'projection_max': {site: float(value[:, idx].max()) for idx, site in enumerate(SITES)}}

def main() -> None:
    global COSTS, TOTAL_COST
    args = parse_args()
    common.set_costs(args.costs)
    COSTS = common.COSTS
    TOTAL_COST = common.TOTAL_COST
    if args.smoke:
        args.targets = '50'
        args.teal_base_step = 25.0
        args.larosa_grid = '0.9,1.0,1.1'
        args.final_samples = 1
    targets = common.parse_targets(args.targets)
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    calibration_ids, calibration_meta = build_mixed_10x2048(args.root, args.output_dir)
    eval_ids, eval_meta = common.read_evalset(args.evalset_dir)
    eval_ids = eval_ids[:args.final_samples]
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
    dense_delta = abs(absorbed_dense['ppl'] - original_dense['ppl']) / original_dense['ppl']
    print(f"[dense-absorbed] ppl={absorbed_dense['ppl']:.6f} relative_delta={100.0 * dense_delta:.4f}%", flush=True)
    if dense_delta > 0.01:
        raise RuntimeError('absorbed dense path differs from original by more than 1%')
    teal_started = time.perf_counter()
    paths = build_teal_paths(model, layers, controller, calibration_ids, device, max(targets), args.teal_base_step, args.teal_last_fraction, args.output_dir / 'teal_paper_paths.json', args.force_teal)
    teal_elapsed = time.perf_counter() - teal_started
    store = TrialStore(args.output_dir / 'ppl_trials.csv')
    alpha_grid = common.parse_float_grid(args.larosa_grid)
    recipes: dict[str, np.ndarray] = {}
    search: dict[str, Any] = {}
    larosa_started = time.perf_counter()
    for target in targets:
        label = f's{int(round(target)):02d}'
        uniform = np.full((num_layers, len(SITES)), target, dtype=np.float64)
        teal = teal_recipe(paths, target)
        recipes[f'uniform_{label}'] = uniform
        recipes[f'teal_paper_{label}'] = teal
        search[label] = {'teal': projection_summary(teal)}
        if not args.skip_larosa:
            candidates = []
            for alpha_att in alpha_grid:
                for alpha_ffn in alpha_grid:
                    recipe, coefficients = larosa_recipe(target, alpha_att, alpha_ffn, num_layers)
                    row = store.score(model, controller, adaptive, device, 'larosa_paper_grid', target, f'mixed_10x2048_{label}', recipe, calibration_ids, alpha_att, alpha_ffn)
                    candidates.append({**row, 'coefficients': coefficients})
            best = min(candidates, key=lambda row: float(row['ppl']))
            larosa = np.asarray(best['recipe'], dtype=np.float64)
            recipes[f'larosa_paper_{label}'] = larosa
            search[label]['larosa'] = {**projection_summary(larosa), 'calibration_ppl': float(best['ppl']), 'coefficients': best['coefficients'], 'evaluated_candidates': len(candidates)}
    larosa_elapsed = time.perf_counter() - larosa_started
    final_rows: list[dict[str, Any]] = [{'method': 'dense_original', 'target': 0.0, 'weighted_sparsity': 0.0, **original_dense}, {'method': 'dense_absorbed', 'target': 0.0, 'weighted_sparsity': 0.0, **absorbed_dense}]
    for method, recipe in recipes.items():
        target = float(method.rsplit('s', 1)[1])
        row = store.score(model, controller, adaptive, device, method.rsplit('_s', 1)[0], target, 'final_wikitext_4x4096_full_sparse', recipe, eval_ids)
        final_rows.append({'method': method, 'target': target, 'weighted_sparsity': common.weighted_sparsity(recipe), 'ppl': float(row['ppl']), 'mean_nll': float(row['mean_nll']), 'total_nll': float(row['total_nll']), 'total_tokens': int(row['total_tokens']), 'fallback': row.get('fallback', adaptive.stats())})
    with (args.output_dir / 'final_ppl.csv').open('w', encoding='utf-8', newline='') as handle:
        fields = ('method', 'target', 'weighted_sparsity', 'ppl', 'mean_nll', 'total_nll', 'total_tokens')
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in final_rows:
            writer.writerow({field: row.get(field, '') for field in fields})
    common.write_recipe_csv(args.output_dir / 'recipes_by_layer.csv', recipes)
    common.plot_recipes(args.output_dir / 'recipes_by_layer.png', recipes)
    save_json(args.output_dir / 'recipes.json', {key: value.tolist() for key, value in recipes.items()})
    summary = {'status': 'complete', 'definition': {'body': 'current Ours, full sparse, random-permuted B128, adaptive-DP K4', 'lm_head': 'dense', 'calibration': calibration_meta, 'teal': 'paper 5% cost-normalized greedy, last-25% layer-output L2', 'larosa': 'skipped' if args.skip_larosa else 'paper-size 0.70:1.20:0.05 keep-ratio grid, all 121 candidates directly scored by PPL', 'larosa_adaptation': "QKV/O and Gate/(Up+Down) coefficients preserve exact Llama-3.1-8B MAC budget and Ours' nested Up/Down support", 'final_ppl': 'WikiText-103 test 4x4096, every target token sparse'}, 'args': {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}, 'eval_meta': eval_meta, 'search': search, 'timing_s': {'teal': teal_elapsed, 'larosa': larosa_elapsed, 'total': time.perf_counter() - started}, 'final_ppl': final_rows}
    save_json(args.output_dir / 'summary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)
if __name__ == '__main__':
    main()
