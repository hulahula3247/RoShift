"""Deterministic layerwise Pair-CD for the six-component RoShift rotation.

The experiment deliberately keeps the algorithm small enough to describe in a
paper:

1. Build S_att and S_ffn from every row of the 32x4096 mixed calibration set.
2. Initialize every layer with the uniform-no-up mixture.
3. For one layer, enumerate all 15 component pairs.  Keep the other four
   coefficients fixed and evaluate 17 exact redistributions of the pair mass.
4. Propose the lowest decoder-block relative-L2 candidate.
5. Accept it only when full 32x4096 mixed calibration PPL improves.
6. Sweep the 32 layers until a complete sweep accepts no update.

The body sparsity recipe, random-permuted B128 selector, exact lazy median, and
adaptive-DP K4 correction are fixed.  All expensive stages are checkpointed.
"""
from __future__ import annotations
import os as _roshift_os
import argparse
import csv
import gc
import hashlib
import itertools
import json
import math
import os
import random
import shutil
import time
from pathlib import Path
from typing import Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import run_llama3_layerwise_q_tpe_block_l2_20260827 as old
import run_llama3_layerwise_q_tpe_nonuniform_20260828 as experiment
import run_llama3_larosa_absorbed_lazy_ppl as lazy_abs
import run_llama3_mixed_allocation_search_20260815 as allocation
from run_llama2_ffn_gate_output_threshold_ppl import decoder_layers, input_device, read_evalset
DEFAULT_OUTPUT = Path('/repro/data/llama3_1_8b/layerwise_q_paircd_fullcov_20260830')
DEFAULT_RECIPE = Path('/repro/data/llama3_1_8b/mixed_allocation_pair_cd_grid_20260819/state.json')
DEFAULT_MIXED_HELDOUT = old.ROOT / 'heldout_multidomain_20260812' / 'heldout_tokens.pt'
INITIAL_ALPHA = np.asarray(old.SEED_RECIPES['uniform_no_up'], dtype=np.float64)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model_id', default='/repro/models/Meta-Llama-3.1-8B')
    parser.add_argument('--device_map', default='cuda:0')
    parser.add_argument('--attn_implementation', default='sdpa')
    parser.add_argument('--root', type=Path, default=old.ROOT)
    parser.add_argument('--metric_dir', type=Path, default=old.DEFAULT_METRIC_DIR)
    parser.add_argument('--output_dir', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--recipe_state', type=Path, default=DEFAULT_RECIPE)
    parser.add_argument('--mixed_heldout', type=Path, default=DEFAULT_MIXED_HELDOUT)
    parser.add_argument('--adaptive_json', type=Path, default=old.ROOT / 'adaptive_dp_mixed_20260814' / 'adaptive_dp_centers.json')
    parser.add_argument('--evalset_dir', type=Path, default=Path(_roshift_os.path.join(_roshift_os.environ['ROSHIFT_ENGINE_ROOT'], 'eval_data')))
    parser.add_argument('--context_length', type=int, default=512)
    parser.add_argument('--cov_rows_per_sequence', type=int, default=4096)
    parser.add_argument('--fit_rows', default='0,16,22,27')
    parser.add_argument('--eval_rows', default='1,17,23,28')
    parser.add_argument('--pair_denominator', type=int, default=16)
    parser.add_argument('--max_sweeps', type=int, default=8)
    parser.add_argument('--layers', default='')
    parser.add_argument('--block_size', type=int, default=128)
    parser.add_argument('--permutation_seed', type=int, default=11)
    parser.add_argument('--fallback_rel', type=float, default=0.05)
    parser.add_argument('--median_chunk', type=int, default=512)
    parser.add_argument('--median_hist_bins', type=int, default=512)
    parser.add_argument('--wikitext_samples', type=int, default=4)
    parser.add_argument('--search_ppl_rows', type=int, default=0)
    parser.add_argument('--seed', type=int, default=20260830)
    parser.add_argument('--force_components', action='store_true')
    parser.add_argument('--force_captures', action='store_true')
    parser.add_argument('--force_prepare', action='store_true')
    parser.add_argument('--skip_search', action='store_true')
    parser.add_argument('--skip_final_ppl', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    return parser.parse_args()

def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))

def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, ensure_ascii=True, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)

def atomic_tensor(path: Path, value: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(value.detach().cpu().contiguous(), temporary)
    temporary.replace(path)

def alpha_key(alpha: np.ndarray) -> str:
    value = np.asarray(alpha, dtype=np.float64)
    return hashlib.sha1(np.round(value, 12).tobytes()).hexdigest()[:16]

def normalized_alpha(alpha: np.ndarray) -> np.ndarray:
    value = np.maximum(np.asarray(alpha, dtype=np.float64), 0.0)
    total = float(value.sum())
    if total <= 0.0:
        raise ValueError('alpha must have positive mass')
    return value / total

def parse_layers(raw: str, count: int) -> list[int]:
    if not str(raw).strip():
        return list(range(int(count)))
    result = sorted({int(item.strip()) for item in str(raw).split(',') if item.strip()})
    if not result or result[0] < 0 or result[-1] >= int(count):
        raise ValueError(f'invalid layer list: {result}')
    return result

def clone_tree_hardlink(source: Path, destination: Path) -> None:
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination, copy_function=os.link)

def load_recipe(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    return experiment.load_recipe(path)

def component_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(**vars(args), old_search_dir=args.output_dir, cov_sequences=0)

def load_components(args: argparse.Namespace, layer_idx: int, device: torch.device) -> list[torch.Tensor]:
    proxy = component_args(args)
    return experiment.load_components(proxy, int(layer_idx), device)

@torch.inference_mode()
def prepare_sources(args: argparse.Namespace, model: nn.Module, layers: list[nn.Module], layer_indices: list[int], device: torch.device) -> None:
    calibration = torch.load(args.root / 'mixed_calibration' / 'input_ids.pt', map_location='cpu', weights_only=True)
    selection = torch.load(args.root / 'mixed_calibration' / 'selection_input_ids.pt', map_location='cpu', weights_only=True)
    old.collect_split_components(args=component_args(args), model=model, calibration=calibration, layers=layers, layer_indices=layer_indices, device=device)
    old.collect_block_captures(args=component_args(args), model=model, calibration=calibration, selection=selection, layers=layers, layer_indices=layer_indices, device=device)
    del calibration, selection
    gc.collect()
    torch.cuda.empty_cache()

@torch.inference_mode()
def write_q_from_alpha(args: argparse.Namespace, alpha: np.ndarray, q_dir: Path, layer_indices: list[int], device: torch.device) -> None:
    q_dir.mkdir(parents=True, exist_ok=True)
    for layer_idx in layer_indices:
        path = old.q_path(q_dir, layer_idx)
        if path.exists() and (not args.force_prepare):
            continue
        components = load_components(args, layer_idx, device)
        q = old.build_q(normalized_alpha(alpha[layer_idx]), components)
        atomic_tensor(path, q.to(dtype=torch.float16))
        del components, q
        torch.cuda.empty_cache()

def baseline_paths(args: argparse.Namespace) -> dict[str, Path]:
    root = args.output_dir / 'baseline_uniform_fullcov'
    return {'root': root, 'q': root / 'q', 'd': root / 'd', 'pack': root / 'pack', 'median': root / 'median'}

def selected_paths(args: argparse.Namespace) -> dict[str, Path]:
    root = args.output_dir / 'selected'
    return {'root': root, 'q': root / 'q', 'd': root / 'd', 'pack': root / 'pack', 'median': root / 'median'}

@torch.inference_mode()
def prepare_baseline_artifacts(args: argparse.Namespace, model: nn.Module, recipe: np.ndarray, device: torch.device) -> None:
    count = len(list(decoder_layers(model)))
    alpha = np.repeat(INITIAL_ALPHA.reshape(1, -1), count, axis=0)
    base = baseline_paths(args)
    write_q_from_alpha(args, alpha, base['q'], list(range(count)), device)
    old.outq.build_d_cache(model, base['q'], base['d'], force=bool(args.force_prepare))
    controller = old.RmsAlphaBucketController(npz_path=args.root / 'rms_inv_rms_selection.npz', num_layers=count, bucket_k=4, fallback_rel=float(args.fallback_rel), bucket_mode='fixed', bucket_avg_k=4.0)
    if args.force_prepare and base['pack'].exists():
        shutil.rmtree(base['pack'])
    old.build_absorbed_pack(model=model, pack_dir=base['pack'], q_dir=base['q'], median_dir=base['root'] / 'unused', alpha_controller=controller, k_list=[])
    median_args = argparse.Namespace(**vars(args), force_medians=bool(args.force_prepare))
    experiment.build_exact_recipe_medians(args=median_args, model=model, q_dir=base['q'], d_dir=base['d'], recipe=recipe, output_dir=base['median'], device=device)
    selected = selected_paths(args)
    for name in ('q', 'd', 'pack', 'median'):
        clone_tree_hardlink(base[name], selected[name])

def load_search_ids(args: argparse.Namespace) -> np.ndarray:
    ids = torch.load(args.root / 'mixed_calibration' / 'input_ids.pt', map_location='cpu', weights_only=True).numpy()
    if int(args.search_ppl_rows) > 0:
        ids = ids[:int(args.search_ppl_rows)]
    return np.asarray(ids)

def initial_state(args: argparse.Namespace, recipe: np.ndarray, recipe_meta: dict[str, Any], count: int) -> dict[str, Any]:
    alpha = np.repeat(INITIAL_ALPHA.reshape(1, -1), int(count), axis=0)
    return {'status': 'prepared', 'protocol': {'model': str(args.model_id), 'components': list(old.COMPONENTS), 'initial_alpha': INITIAL_ALPHA.tolist(), 'covariance': 'all 32x4096 mixed calibration rows per stream', 'local_objective': 'decoder-block update-relative L2 on 4x512', 'pair_search': f'all 15 pairs; pair mass split on {int(args.pair_denominator) + 1} uniform grid points', 'acceptance': 'strict improvement on full mixed calibration PPL', 'sparsity_recipe': str(args.recipe_state), 'recipe_hash': recipe_meta.get('recipe_hash'), 'weighted_sparsity': float(allocation.weighted_sparsity(recipe)), 'selector': f'random-permuted B{args.block_size}', 'k4': 'adaptive DP K4, 5% relative fallback', 'lm_head': 'dense'}, 'alpha': alpha.tolist(), 'current_search_ppl': None, 'sweeps': [], 'transaction': None}

def state_path(args: argparse.Namespace) -> Path:
    return args.output_dir / 'state.json'

def load_or_create_state(args: argparse.Namespace, recipe: np.ndarray, recipe_meta: dict[str, Any], count: int) -> dict[str, Any]:
    path = state_path(args)
    if path.exists():
        return json.loads(path.read_text(encoding='utf-8'))
    result = initial_state(args, recipe, recipe_meta, count)
    atomic_json(path, result)
    return result

def local_cache(path: Path, base_key: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return result
    with path.open('r', encoding='utf-8') as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get('base_key') == base_key:
                result[str(row['alpha_key'])] = row
    return result

@torch.inference_mode()
def scan_layer_pairs(*, args: argparse.Namespace, model_layer: nn.Module, layer_idx: int, rates: np.ndarray, base_alpha: np.ndarray, attention_kwargs: dict[str, Any], adaptive: experiment.AdaptiveAlpha, device: torch.device, sweep: int) -> dict[str, Any]:
    base_alpha = normalized_alpha(base_alpha)
    base_key = alpha_key(base_alpha)
    proposal_path = args.output_dir / 'local_proposals' / f'sweep_{sweep:02d}' / f'layer_{layer_idx:02d}.json'
    if proposal_path.exists():
        row = json.loads(proposal_path.read_text(encoding='utf-8'))
        if row.get('base_key') == base_key and row.get('status') == 'complete':
            print(f'[local] reuse sweep={sweep} layer={layer_idx:02d}', flush=True)
            return row
    components = load_components(args, layer_idx, device)
    fit_hidden, eval_hidden, dense_output = old.load_layer_capture(args.output_dir / 'block_captures', layer_idx, device)
    candidate_state = experiment.CandidateState(layer_idx=layer_idx, layer=model_layer, metrics=components, rates=rates, block_size=int(args.block_size), permutation_seed=int(args.permutation_seed), adaptive=adaptive, device=device)
    candidate = experiment.CandidateLayer(model_layer, candidate_state).to(device)
    log_path = args.output_dir / 'local_trials' / f'sweep_{sweep:02d}' / f'layer_{layer_idx:02d}.jsonl'
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cached = local_cache(log_path, base_key)

    def evaluate(alpha: np.ndarray, pair: tuple[int, int] | None, split: int | None):
        alpha = normalized_alpha(alpha)
        key = alpha_key(alpha)
        if key in cached:
            return (float(cached[key]['objective']), cached[key])
        started = time.perf_counter()
        q = old.build_q(alpha, components)
        candidate_state.set_candidate(q)
        objective, metrics = experiment.score_candidate(candidate, fit_hidden, eval_hidden, dense_output, attention_kwargs)
        row = {'sweep': int(sweep), 'layer': int(layer_idx), 'base_key': base_key, 'alpha_key': key, 'alpha': alpha.tolist(), 'pair': None if pair is None else [int(pair[0]), int(pair[1])], 'pair_components': None if pair is None else [old.COMPONENTS[pair[0]], old.COMPONENTS[pair[1]]], 'split': split, 'objective': float(objective), 'metrics': metrics, 'elapsed_sec': float(time.perf_counter() - started)}
        with log_path.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(row, allow_nan=False) + '\n')
        cached[key] = row
        del q
        return (float(objective), row)
    current_objective, current_row = evaluate(base_alpha, None, None)
    best_objective = float(current_objective)
    best_alpha = base_alpha.copy()
    best_row = current_row
    denominator = int(args.pair_denominator)
    for left, right in itertools.combinations(range(len(old.COMPONENTS)), 2):
        mass = float(base_alpha[left] + base_alpha[right])
        for split in range(denominator + 1):
            alpha = base_alpha.copy()
            alpha[left] = mass * float(split) / float(denominator)
            alpha[right] = mass - alpha[left]
            objective, row = evaluate(alpha, (left, right), split)
            if objective < best_objective:
                best_objective = float(objective)
                best_alpha = normalized_alpha(alpha)
                best_row = row
    q = old.build_q(best_alpha, components)
    q_path = args.output_dir / 'local_proposals' / f'sweep_{sweep:02d}' / f'q_layer_{layer_idx:02d}.pt'
    atomic_tensor(q_path, q.to(dtype=torch.float16))
    result = {'status': 'complete', 'sweep': int(sweep), 'layer': int(layer_idx), 'base_key': base_key, 'base_alpha': base_alpha.tolist(), 'base_objective': float(current_objective), 'candidate_alpha': best_alpha.tolist(), 'candidate_objective': float(best_objective), 'local_improvement': float(current_objective - best_objective), 'candidate_pair': best_row.get('pair_components'), 'candidate_split': best_row.get('split'), 'candidate_q': str(q_path), 'evaluated_unique_candidates': int(len(cached))}
    atomic_json(proposal_path, result)
    candidate.restore()
    del candidate, candidate_state, components
    del fit_hidden, eval_hidden, dense_output, q
    gc.collect()
    torch.cuda.empty_cache()
    print(f"[local] sweep={sweep} layer={layer_idx:02d} objective={current_objective:.8f}->{best_objective:.8f} pair={result['candidate_pair']}", flush=True)
    return result

def load_tensor(path: Path, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.load(path, map_location='cpu', weights_only=True).to(device=device, dtype=dtype).contiguous()

def d_from_absorbed_weight(weight: torch.Tensor) -> torch.Tensor:
    value = torch.linalg.vector_norm(weight.float(), dim=0)
    return (value / value.mean().clamp_min(1e-12)).clamp_min_(1e-06)

def raw_linear(module: nn.Module) -> nn.Module:
    while isinstance(module, allocation.DynamicInputLinear):
        module = module.inner
    return module

class LiveArtifactUpdater:

    def __init__(self, *, args: argparse.Namespace, model: nn.Module, patched: list[allocation.AllocationDecoderLayer], medians: experiment.ExactMedianBank, recipe: np.ndarray, device: torch.device) -> None:
        self.args = args
        self.model = model
        self.patched = patched
        self.medians = medians
        self.recipe = recipe
        self.device = device
        self.base = baseline_paths(args)
        self.selected = selected_paths(args)
        self.hidden = int(model.config.hidden_size)
        self.intermediate = int(model.config.intermediate_size)
        self.hmid_dir = args.root / 'hmid_cache'
        meta = json.loads((self.hmid_dir / 'hmid_meta.json').read_text(encoding='utf-8'))
        self.hmid_rows = int(meta['median_tokens'])

    def q(self, root: Path, layer_idx: int) -> torch.Tensor:
        return load_tensor(old.q_path(root, layer_idx), self.device, torch.float32)

    def base_weight(self, layer_idx: int, name: str) -> torch.Tensor:
        path = lazy_abs.layer_dir(self.base['pack'] / 'common', layer_idx) / f'{name}.pt'
        return load_tensor(path, self.device, torch.float32)

    def save_pack_weight(self, layer_idx: int, name: str, value: torch.Tensor) -> None:
        path = lazy_abs.layer_dir(self.selected['pack'] / 'common', layer_idx) / f'{name}.pt'
        atomic_tensor(path, value.to(dtype=torch.float16))

    def copy_linear(self, layer_idx: int, family: str, name: str, value: torch.Tensor) -> None:
        layer = self.patched[layer_idx].inner
        parent = layer.self_attn if family == 'attn' else layer.mlp
        linear = raw_linear(getattr(parent, name))
        linear.weight.data.copy_(value.to(device=linear.weight.device, dtype=linear.weight.dtype))

    def save_d(self, layer_idx: int, name: str, value: torch.Tensor) -> None:
        atomic_tensor(self.selected['d'] / f'layer_{layer_idx:02d}_{name}_d.pt', value.to(dtype=torch.float16))

    def set_qkv_scale(self, layer_idx: int, scale: torch.Tensor) -> None:
        layer = self.patched[layer_idx].inner
        cpu = scale.detach().cpu().float().contiguous()
        for name in ('q_proj', 'k_proj', 'v_proj'):
            wrapper = getattr(layer.self_attn, name)
            if not isinstance(wrapper, allocation.DynamicInputLinear):
                raise TypeError(f'layer {layer_idx} {name} is not DynamicInputLinear')
            wrapper.scale_cpu = cpu
            wrapper.cache.clear()

    def set_gate_scale(self, layer_idx: int, scale: torch.Tensor) -> None:
        wrapper = self.patched[layer_idx]
        wrapper.gate_scale_cpu = scale.detach().cpu().float().contiguous()
        wrapper.cache.clear()

    def set_base_transition(self, layer_idx: int, value: torch.Tensor) -> None:
        if layer_idx < 0:
            return
        atomic_tensor(lazy_abs.layer_dir(self.selected['pack'] / 'common', layer_idx) / 'base_transition.pt', value.to(dtype=torch.float16))
        wrapper = self.patched[layer_idx]
        wrapper.base_cpu = value.detach().cpu().float().contiguous()
        wrapper.cache.clear()

    @torch.inference_mode()
    def exact_median(self, layer_idx: int, q: torch.Tensor, gate_weight: torch.Tensor, d_gate: torch.Tensor) -> torch.Tensor:
        layer = self.patched[layer_idx].inner
        rate = float(self.recipe[layer_idx, allocation.SITE_INDEX['gate']])
        h_mm = np.memmap(old.fair.hmid_path(self.hmid_dir, layer_idx), dtype=np.float16, mode='r', shape=(self.hmid_rows, self.hidden))
        permutation = old.fixed_permutation(self.hidden, int(self.args.permutation_seed), f'layer{layer_idx}:gate').to(self.device)
        eps = float(layer.post_attention_layernorm.variance_epsilon)
        bins = int(self.args.median_hist_bins)
        q_half = q.to(dtype=torch.float16).contiguous()
        gate_weight_half = gate_weight.to(dtype=torch.float16).contiguous()
        minv = torch.full((self.intermediate,), float('inf'), device=self.device, dtype=torch.float32)
        maxv = torch.full((self.intermediate,), float('-inf'), device=self.device, dtype=torch.float32)

        def chunks():
            for start in range(0, self.hmid_rows, int(self.args.median_chunk)):
                end = min(self.hmid_rows, start + int(self.args.median_chunk))
                h = torch.from_numpy(np.asarray(h_mm[start:end], dtype=np.float16)).to(device=self.device)
                alpha = old.inv_rms(h, eps).to(dtype=torch.float16)
                u = torch.matmul(h, q_half)
                gate_u = allocation.blockwise_topk(u, d_gate, rate, int(self.args.block_size), permutation) * alpha.unsqueeze(-1)
                h1 = layer.mlp.act_fn(F.linear(gate_u, gate_weight_half, bias=None)).float()
                yield h1
                del h, alpha, u, gate_u
        for h1 in chunks():
            minv = torch.minimum(minv, h1.amin(dim=0))
            maxv = torch.maximum(maxv, h1.amax(dim=0))
            del h1
        span = torch.clamp(maxv - minv, min=1e-12)
        histogram = torch.zeros((self.intermediate, bins), device=self.device, dtype=torch.int32)
        flat_histogram = histogram.view(-1)
        offsets = (torch.arange(self.intermediate, device=self.device, dtype=torch.int64) * bins).view(1, -1)
        for h1 in chunks():
            index = ((h1 - minv.view(1, -1)) / span.view(1, -1) * float(bins - 1)).round().to(torch.int64)
            index.clamp_(0, bins - 1)
            flat_index = (index + offsets).reshape(-1)
            flat_histogram.scatter_add_(0, flat_index, torch.ones_like(flat_index, dtype=torch.int32))
            del h1, index, flat_index
        target = int((self.hmid_rows - 1) // 2)
        cumulative = histogram.cumsum(dim=1)
        median_bin = (cumulative > target).to(torch.int64).argmax(dim=1).float()
        median = minv + median_bin / float(bins - 1) * (maxv - minv)
        del h_mm, permutation, q_half, gate_weight_half, minv, maxv, span
        del histogram, flat_histogram, offsets, cumulative, median_bin
        return median

    def install_median(self, layer_idx: int, median: torch.Tensor) -> None:
        atomic_tensor(self.selected['median'] / f'layer_{layer_idx:02d}.pt', median.to(dtype=torch.float16))
        self.medians.cpu[layer_idx] = median.detach().cpu().float().contiguous()
        for key in list(self.medians.device):
            if int(key[0]) == int(layer_idx):
                del self.medians.device[key]

    @torch.inference_mode()
    def apply(self, layer_idx: int, q_new_cpu: torch.Tensor) -> None:
        layer_idx = int(layer_idx)
        q_new = q_new_cpu.to(self.device, dtype=torch.float32).contiguous()
        q_base = self.q(self.base['q'], layer_idx)
        right = torch.matmul(q_base.transpose(0, 1), q_new)
        left = right.transpose(0, 1).contiguous()
        transformed: dict[str, torch.Tensor] = {}
        for name in ('q_proj', 'k_proj', 'v_proj', 'gate_proj', 'up_proj'):
            transformed[name] = torch.matmul(self.base_weight(layer_idx, name), right)
            family = 'attn' if name in ('q_proj', 'k_proj', 'v_proj') else 'mlp'
            self.copy_linear(layer_idx, family, name, transformed[name])
            self.save_pack_weight(layer_idx, name, transformed[name])
        transformed['o_proj'] = torch.matmul(left, self.base_weight(layer_idx, 'o_proj'))
        self.copy_linear(layer_idx, 'attn', 'o_proj', transformed['o_proj'])
        self.save_pack_weight(layer_idx, 'o_proj', transformed['o_proj'])
        if layer_idx == 0:
            base_embed = load_tensor(self.base['pack'] / 'common' / 'embed_tokens.pt', self.device, torch.float32)
            embed = torch.matmul(base_embed, right)
            self.model.model.embed_tokens.weight.data.copy_(embed.to(dtype=self.model.model.embed_tokens.weight.dtype))
            atomic_tensor(self.selected['pack'] / 'common' / 'embed_tokens.pt', embed.to(dtype=torch.float16))
            del base_embed, embed
        if layer_idx > 0:
            previous_down = torch.matmul(left, self.base_weight(layer_idx - 1, 'down_proj'))
            self.copy_linear(layer_idx - 1, 'mlp', 'down_proj', previous_down)
            self.save_pack_weight(layer_idx - 1, 'down_proj', previous_down)
            del previous_down
        q_next = self.q(self.selected['q'], layer_idx + 1) if layer_idx + 1 < len(self.patched) else torch.eye(self.hidden, device=self.device, dtype=torch.float32)
        self.set_base_transition(layer_idx, torch.matmul(q_new.transpose(0, 1), q_next))
        if layer_idx > 0:
            q_previous = self.q(self.selected['q'], layer_idx - 1)
            self.set_base_transition(layer_idx - 1, torch.matmul(q_previous.transpose(0, 1), q_new))
            del q_previous
        d_values = {name: d_from_absorbed_weight(transformed[name]) for name in ('q_proj', 'k_proj', 'v_proj', 'gate_proj', 'up_proj')}
        for name, value in d_values.items():
            self.save_d(layer_idx, name, value)
        shared_qkv = torch.sqrt(sum((d_values[name].square() for name in ('q_proj', 'k_proj', 'v_proj'))).clamp_min_(1e-12))
        self.set_qkv_scale(layer_idx, shared_qkv)
        self.set_gate_scale(layer_idx, d_values['gate_proj'])
        median = self.exact_median(layer_idx, q_new, transformed['gate_proj'], d_values['gate_proj'])
        self.install_median(layer_idx, median)
        atomic_tensor(old.q_path(self.selected['q'], layer_idx), q_new.to(dtype=torch.float16))
        del q_new, q_base, right, left, q_next, shared_qkv, median
        for value in transformed.values():
            del value
        for value in d_values.values():
            del value
        torch.cuda.empty_cache()

def load_eval_model(args: argparse.Namespace, recipe: np.ndarray) -> tuple[nn.Module, torch.device, list[allocation.AllocationDecoderLayer], experiment.ExactMedianBank, allocation.AdaptiveK4]:
    model = old.load_model(args)
    model.config.use_cache = False
    device = input_device(model)
    selected = selected_paths(args)
    old.apply_common_pack(model, selected['pack'])
    controller = allocation.RecipeController(len(list(decoder_layers(model))))
    controller.set_recipe(recipe)
    adaptive = allocation.AdaptiveK4(args.adaptive_json, float(args.fallback_rel))
    medians = experiment.ExactMedianBank(selected['median'])
    patched = allocation.patch_model(model, controller, selected['d'], selected['pack'], medians, adaptive, int(args.block_size), int(args.permutation_seed))
    return (model, device, patched, medians, adaptive)

@torch.inference_mode()
def evaluate_search_ppl(model: nn.Module, ids: np.ndarray, device: torch.device, adaptive: allocation.AdaptiveK4) -> dict[str, Any]:
    adaptive.reset()
    started = time.perf_counter()
    result = allocation.evaluate_ppl(model, ids, device)
    result['elapsed_sec'] = float(time.perf_counter() - started)
    result['adaptive_k4'] = adaptive.stats()
    return result

def restore_incomplete_transaction(args: argparse.Namespace, state: dict[str, Any], updater: LiveArtifactUpdater) -> None:
    transaction = state.get('transaction')
    if not transaction:
        return
    layer_idx = int(transaction['layer'])
    alpha = np.asarray(state['alpha'][layer_idx], dtype=np.float64)
    components = load_components(args, layer_idx, updater.device)
    q = old.build_q(alpha, components)
    print(f'[resume] restoring interrupted layer={layer_idx:02d}', flush=True)
    updater.apply(layer_idx, q)
    state['transaction'] = None
    atomic_json(state_path(args), state)
    del components, q

def append_gate_log(args: argparse.Namespace, row: dict[str, Any]) -> None:
    path = args.output_dir / 'ppl_gate_trials.jsonl'
    with path.open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(row, allow_nan=False) + '\n')

def run_search(args: argparse.Namespace, recipe: np.ndarray, recipe_meta: dict[str, Any]) -> dict[str, Any]:
    count = int(recipe.shape[0])
    state = load_or_create_state(args, recipe, recipe_meta, count)
    search_ids = load_search_ids(args)
    layer_indices = parse_layers(args.layers, count)
    for sweep in range(len(state.get('sweeps', [])), int(args.max_sweeps)):
        active = state.get('active_sweep')
        if active is not None and int(active.get('sweep', -1)) == int(sweep):
            alpha_at_start = np.asarray(active['alpha_at_start'], dtype=np.float64)
        else:
            alpha_at_start = np.asarray(state['alpha'], dtype=np.float64)
        print(f'[sweep] {sweep} local pair grids', flush=True)
        model = old.load_model(args)
        device = input_device(model)
        layers = list(decoder_layers(model))
        adaptive_local = experiment.AdaptiveAlpha(args.adaptive_json, float(args.fallback_rel), device)
        attention_kwargs = old.to_device(torch.load(args.output_dir / 'block_captures' / 'attention_kwargs.pt', map_location='cpu', weights_only=True), device)
        proposals: dict[int, dict[str, Any]] = {}
        for layer_idx in layer_indices:
            proposals[layer_idx] = scan_layer_pairs(args=args, model_layer=layers[layer_idx], layer_idx=layer_idx, rates=recipe[layer_idx], base_alpha=alpha_at_start[layer_idx], attention_kwargs=attention_kwargs, adaptive=adaptive_local, device=device, sweep=sweep)
        del model, layers, adaptive_local, attention_kwargs
        gc.collect()
        torch.cuda.empty_cache()
        print(f'[sweep] {sweep} full mixed-PPL gates', flush=True)
        model, device, patched, medians, adaptive = load_eval_model(args, recipe)
        updater = LiveArtifactUpdater(args=args, model=model, patched=patched, medians=medians, recipe=recipe, device=device)
        restore_incomplete_transaction(args, state, updater)
        active = state.get('active_sweep')
        if active is None or int(active.get('sweep', -1)) != int(sweep):
            active = {'sweep': int(sweep), 'alpha_at_start': alpha_at_start.tolist(), 'completed_layers': [], 'accepted': [], 'evaluated': []}
            state['active_sweep'] = active
            atomic_json(state_path(args), state)
        if state.get('current_search_ppl') is None:
            baseline = evaluate_search_ppl(model, search_ids, device, adaptive)
            state['current_search_ppl'] = float(baseline['ppl'])
            state['initial_search_ppl'] = baseline
            atomic_json(state_path(args), state)
            print(f"[ppl-gate] baseline={baseline['ppl']:.8f}", flush=True)
        accepted: list[dict[str, Any]] = list(active.get('accepted', []))
        evaluated: list[dict[str, Any]] = list(active.get('evaluated', []))
        completed_layers = {int(value) for value in active.get('completed_layers', [])}
        for layer_idx in layer_indices:
            if int(layer_idx) in completed_layers:
                print(f'[ppl-gate] reuse sweep={sweep} layer={layer_idx:02d}', flush=True)
                continue
            proposal = proposals[layer_idx]
            if float(proposal['local_improvement']) <= 0.0:
                completed_layers.add(int(layer_idx))
                active['completed_layers'] = sorted(completed_layers)
                state['active_sweep'] = active
                atomic_json(state_path(args), state)
                continue
            old_alpha = np.asarray(state['alpha'][layer_idx], dtype=np.float64)
            candidate_alpha = np.asarray(proposal['candidate_alpha'], dtype=np.float64)
            if alpha_key(old_alpha) != str(proposal['base_key']):
                raise RuntimeError(f'layer {layer_idx} proposal base changed during sweep')
            current_ppl = float(state['current_search_ppl'])
            state['transaction'] = {'sweep': int(sweep), 'layer': int(layer_idx), 'old_alpha': old_alpha.tolist(), 'candidate_alpha': candidate_alpha.tolist(), 'current_ppl': current_ppl}
            atomic_json(state_path(args), state)
            q_candidate = torch.load(proposal['candidate_q'], map_location='cpu', weights_only=True)
            started = time.perf_counter()
            updater.apply(layer_idx, q_candidate)
            candidate_result = evaluate_search_ppl(model, search_ids, device, adaptive)
            candidate_ppl = float(candidate_result['ppl'])
            accept = candidate_ppl < current_ppl
            gate_row = {'sweep': int(sweep), 'layer': int(layer_idx), 'pair': proposal['candidate_pair'], 'old_alpha': old_alpha.tolist(), 'candidate_alpha': candidate_alpha.tolist(), 'local_objective_before': float(proposal['base_objective']), 'local_objective_candidate': float(proposal['candidate_objective']), 'ppl_before': current_ppl, 'ppl_candidate': candidate_ppl, 'ppl_improvement': current_ppl - candidate_ppl, 'accepted': bool(accept), 'candidate_eval': candidate_result, 'elapsed_sec': float(time.perf_counter() - started)}
            append_gate_log(args, gate_row)
            evaluated.append(gate_row)
            if accept:
                state['alpha'][layer_idx] = candidate_alpha.tolist()
                state['current_search_ppl'] = candidate_ppl
                accepted.append(gate_row)
                print(f'[accept] sweep={sweep} layer={layer_idx:02d} PPL={current_ppl:.8f}->{candidate_ppl:.8f}', flush=True)
            else:
                components = load_components(args, layer_idx, device)
                q_old = old.build_q(old_alpha, components)
                updater.apply(layer_idx, q_old)
                del components, q_old
                print(f'[reject] sweep={sweep} layer={layer_idx:02d} PPL={current_ppl:.8f}->{candidate_ppl:.8f}', flush=True)
            completed_layers.add(int(layer_idx))
            active['completed_layers'] = sorted(completed_layers)
            active['accepted'] = accepted
            active['evaluated'] = evaluated
            state['active_sweep'] = active
            state['transaction'] = None
            atomic_json(state_path(args), state)
            del q_candidate
            torch.cuda.empty_cache()
        sweep_row = {'sweep': int(sweep), 'proposals': int(sum((float(row['local_improvement']) > 0.0 for row in proposals.values()))), 'ppl_evaluations': int(len(evaluated)), 'accepted_updates': int(len(accepted)), 'accepted': accepted, 'ending_search_ppl': float(state['current_search_ppl'])}
        state.setdefault('sweeps', []).append(sweep_row)
        state['active_sweep'] = None
        state['status'] = 'converged' if not accepted else 'searching'
        atomic_json(state_path(args), state)
        del updater, model, patched, medians, adaptive
        gc.collect()
        torch.cuda.empty_cache()
        if not accepted:
            break
    if state.get('status') != 'converged':
        state['status'] = 'max_sweeps_reached'
        atomic_json(state_path(args), state)
    return state

def aggregate(total_nll: float, total_tokens: int) -> dict[str, Any]:
    mean_nll = float(total_nll) / max(1, int(total_tokens))
    return {'ppl': float(math.exp(mean_nll)), 'mean_nll': mean_nll, 'total_nll': float(total_nll), 'total_tokens': int(total_tokens)}

@torch.inference_mode()
def final_ppl(args: argparse.Namespace, recipe: np.ndarray) -> dict[str, Any]:
    output = args.output_dir / 'final_ppl_summary.json'
    model, device, _patched, _medians, adaptive = load_eval_model(args, recipe)
    wikitext_ids, _ = read_evalset(args.evalset_dir)
    wikitext_ids = np.asarray(wikitext_ids[:int(args.wikitext_samples)])
    heldout_ids, domain_ids, domain_names = experiment.load_mixed_evalset(args.mixed_heldout)
    print('[final-ppl] WikiText', flush=True)
    wikitext = evaluate_search_ppl(model, wikitext_ids, device, adaptive)
    print(f"[final-ppl] WikiText={wikitext['ppl']:.8f}", flush=True)
    per_domain: dict[str, Any] = {}
    total_nll = 0.0
    total_tokens = 0
    for domain_idx, domain_name in enumerate(domain_names):
        print(f'[final-ppl] mixed {domain_name}', flush=True)
        result = evaluate_search_ppl(model, heldout_ids[domain_ids == domain_idx], device, adaptive)
        per_domain[domain_name] = result
        total_nll += float(result['total_nll'])
        total_tokens += int(result['total_tokens'])
        print(f"[final-ppl] {domain_name}={result['ppl']:.8f}", flush=True)
    payload = {'status': 'complete', 'method': 'layerwise_q_pair_cd_fullcov', 'wikitext': wikitext, 'mixed': aggregate(total_nll, total_tokens), 'per_domain': per_domain}
    atomic_json(output, payload)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return payload

def write_alpha_outputs(args: argparse.Namespace, state: dict[str, Any]) -> None:
    alpha = np.asarray(state['alpha'], dtype=np.float64)
    csv_path = args.output_dir / 'layerwise_alpha.csv'
    with csv_path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['layer', *old.COMPONENTS])
        for layer_idx, row in enumerate(alpha):
            writer.writerow([layer_idx, *[float(value) for value in row]])
    atomic_json(args.output_dir / 'recipe.json', {'method': 'layerwise_q_pair_cd_fullcov', 'components': list(old.COMPONENTS), 'alpha': alpha.tolist(), 'search_state': str(state_path(args))})

def main() -> None:
    args = parse_args()
    if args.smoke:
        args.cov_rows_per_sequence = min(64, int(args.cov_rows_per_sequence))
        args.layers = '0'
        args.pair_denominator = 2
        args.max_sweeps = 1
        args.search_ppl_rows = 1
        args.skip_final_ppl = True
    if int(args.pair_denominator) <= 0:
        raise ValueError('pair_denominator must be positive')
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    recipe, recipe_meta = load_recipe(args.recipe_state)
    print(f'[load] prepare model={args.model_id}', flush=True)
    model = old.load_model(args)
    model.config.use_cache = False
    device = input_device(model)
    layers = list(decoder_layers(model))
    layer_indices = parse_layers(args.layers, len(layers))
    prepare_sources(args, model, layers, layer_indices, device)
    if len(layer_indices) == len(layers):
        prepare_baseline_artifacts(args, model, recipe, device)
        load_or_create_state(args, recipe, recipe_meta, len(layers))
    del model, layers
    gc.collect()
    torch.cuda.empty_cache()
    if args.smoke:
        proxy_args = component_args(args)
        model = old.load_model(args)
        device = input_device(model)
        layers = list(decoder_layers(model))
        adaptive = experiment.AdaptiveAlpha(args.adaptive_json, float(args.fallback_rel), device)
        attention_kwargs = old.to_device(torch.load(args.output_dir / 'block_captures' / 'attention_kwargs.pt', map_location='cpu', weights_only=True), device)
        scan_layer_pairs(args=args, model_layer=layers[0], layer_idx=0, rates=recipe[0], base_alpha=INITIAL_ALPHA, attention_kwargs=attention_kwargs, adaptive=adaptive, device=device, sweep=0)
        del proxy_args, model
        print('[smoke] complete', flush=True)
        return
    state = load_or_create_state(args, recipe, recipe_meta, len(recipe))
    if not args.skip_search and state.get('status') != 'converged':
        state = run_search(args, recipe, recipe_meta)
    write_alpha_outputs(args, state)
    if not args.skip_final_ppl:
        result = final_ppl(args, recipe)
        print(json.dumps(result, indent=2), flush=True)
    print(f"[done] status={state.get('status')} search_ppl={state.get('current_search_ppl')}", flush=True)
if __name__ == '__main__':
    main()
