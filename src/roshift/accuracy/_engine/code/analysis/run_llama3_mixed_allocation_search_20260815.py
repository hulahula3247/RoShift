"""Mixed-domain TEAL-greedy and LaRoSA-grid allocation for current Ours.

The experiment changes only the 32 x 7 body sparsity recipe.  It keeps the
mixed-domain rotations, shared-QKV scoring, random-permuted B128 selection,
lazy median correction, and adaptive-DP K4 RMS centers fixed.  Every evaluated
token is sparse and the LM head is dense for likelihood measurements.
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
import matplotlib
import numpy as np
import torch
import torch.nn as nn
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from llama3_ours_accuracy_runtime import load_shared_qkv_scale
from rental_validation import rms_epsilon
from run_llama2_ffn_gate_output_threshold_ppl import ce_sum, decoder_layers, input_device, load_model, read_evalset
from run_llama3_larosa_absorbed_lazy_ppl import apply_common_pack, layer_dir
from run_llama3_projection_allocation_search_20260720 import MedianBank
from run_llama3_sparsity_unimodality_full import blockwise_topk, detach_clone, fixed_permutation
ROOT = Path('/repro/data/llama3_1_8b/mixed_calib_c4_alpaca_metamath_code_20260812')
SITES = ('q', 'k', 'v', 'o', 'gate', 'up', 'down')
SITE_INDEX = {site: index for index, site in enumerate(SITES)}
COSTS = np.asarray([1.0, 0.25, 0.25, 1.0, 3.5, 3.5, 3.5], dtype=np.float64)
TOTAL_COST = float(COSTS.sum())

def set_costs(raw: str | list[float] | np.ndarray) -> np.ndarray:
    """Configure per-projection MAC costs while preserving Llama defaults."""
    global COSTS, TOTAL_COST
    if isinstance(raw, str):
        values = [float(value.strip()) for value in raw.split(',') if value.strip()]
    else:
        values = np.asarray(raw, dtype=np.float64).reshape(-1).tolist()
    if len(values) != len(SITES) or any((not math.isfinite(value) or value <= 0.0 for value in values)):
        raise ValueError(f'costs must contain {len(SITES)} finite positive values: {values}')
    COSTS = np.asarray(values, dtype=np.float64)
    TOTAL_COST = float(COSTS.sum())
    return COSTS

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model_id', default='/repro/models/Meta-Llama-3.1-8B')
    parser.add_argument('--device_map', default='cuda:0')
    parser.add_argument('--attn_implementation', default='sdpa')
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--evalset_dir', type=Path, default=Path(_roshift_os.path.join(_roshift_os.environ['ROSHIFT_ENGINE_ROOT'], 'eval_data')))
    parser.add_argument('--output_dir', type=Path, default=Path('/repro/data/llama3_1_8b/mixed_allocation_teal_larosa_20260815'))
    parser.add_argument('--pack_dir', type=Path, default=ROOT / 'ours_absorbed_pack')
    parser.add_argument('--d_dir', type=Path, default=ROOT / 'output_aware_q' / 'd_cache' / 's_plus_lam_0p2')
    parser.add_argument('--median_dir', type=Path, default=ROOT / 'ours_daware_sparse_gate_medians')
    parser.add_argument('--adaptive_json', type=Path, default=ROOT / 'adaptive_dp_mixed_20260814' / 'adaptive_dp_centers.json')
    parser.add_argument('--targets', default='40,50')
    parser.add_argument('--block_size', type=int, default=128)
    parser.add_argument('--permutation_seed', type=int, default=11)
    parser.add_argument('--fallback_rel', type=float, default=0.05)
    parser.add_argument('--costs', default='1,0.25,0.25,1,3.5,3.5,3.5', help='relative MAC costs in q,k,v,o,gate,up,down order')
    parser.add_argument('--greedy_base_step', type=float, default=5.0)
    parser.add_argument('--max_rate', type=float, default=100.0)
    parser.add_argument('--larosa_grid', default='0.70:1.20:0.05')
    parser.add_argument('--proxy_len', type=int, default=128)
    parser.add_argument('--grid_len', type=int, default=512)
    parser.add_argument('--rerank_top', type=int, default=5)
    parser.add_argument('--final_samples', type=int, default=4)
    parser.add_argument('--seed', type=int, default=20260815)
    parser.add_argument('--force_greedy', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    return parser.parse_args()

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def parse_float_grid(raw: str) -> list[float]:
    if ':' not in raw:
        return [float(value.strip()) for value in raw.split(',') if value.strip()]
    lo, hi, step = (float(value) for value in raw.split(':'))
    count = int(round((hi - lo) / step))
    return [round(lo + index * step, 10) for index in range(count + 1)]

def parse_targets(raw: str) -> list[float]:
    values = sorted({float(value.strip()) for value in raw.split(',') if value.strip()})
    if not values or min(values) <= 0.0 or max(values) >= 100.0:
        raise ValueError(f'invalid targets: {values}')
    return values

def recipe_hash(recipe: np.ndarray) -> str:
    rounded = np.round(np.asarray(recipe, dtype=np.float64), 6)
    return hashlib.sha1(rounded.tobytes()).hexdigest()[:16]

def effective_array(recipe: np.ndarray) -> np.ndarray:
    value = np.asarray(recipe, dtype=np.float64).copy()
    value[..., SITE_INDEX['up']] = np.minimum(value[..., SITE_INDEX['up']], value[..., SITE_INDEX['down']])
    return value

def weighted_sparsity(recipe: np.ndarray) -> float:
    value = effective_array(recipe)
    return float((value * COSTS.reshape(*[1] * (value.ndim - 1), -1)).sum() / (value.shape[0] * TOTAL_COST))

def repair_recipe(recipe: np.ndarray, target: float, max_rate: float) -> np.ndarray:
    raw = np.asarray(recipe, dtype=np.float64)
    lo, hi = (-200.0, 200.0)
    for _ in range(90):
        mid = 0.5 * (lo + hi)
        value = np.clip(raw + mid, 0.0, float(max_rate))
        if weighted_sparsity(value) < float(target):
            lo = mid
        else:
            hi = mid
    result = np.clip(raw + 0.5 * (lo + hi), 0.0, float(max_rate))
    if np.any(result[:, SITE_INDEX['up']] > result[:, SITE_INDEX['down']] + 1e-08):
        raise RuntimeError('repair produced an unrealizable Up/Down recipe')
    return result

class RecipeController:

    def __init__(self, num_layers: int, runtime_controller=None) -> None:
        self.recipe = np.zeros((int(num_layers), len(SITES)), dtype=np.float64)
        self.runtime_controller = runtime_controller

    def set_recipe(self, recipe: np.ndarray) -> None:
        value = np.asarray(recipe, dtype=np.float64)
        if value.shape != self.recipe.shape:
            raise ValueError(f'recipe shape {value.shape} != {self.recipe.shape}')
        if np.any(value[:, SITE_INDEX['up']] > value[:, SITE_INDEX['down']] + 1e-08):
            raise ValueError('Up sparsity cannot exceed Down with nested support')
        self.recipe[...] = value

    def set_layer(self, layer_idx: int, rates: np.ndarray) -> None:
        self.recipe[int(layer_idx)] = np.asarray(rates, dtype=np.float64)

    def enabled(self) -> bool:
        return self.runtime_controller is None or bool(self.runtime_controller.sparse_enabled())

    def rate(self, layer_idx: int, site: str) -> float:
        if not self.enabled():
            return 0.0
        return float(self.recipe[int(layer_idx), SITE_INDEX[str(site)]])

class AdaptiveK4:

    def __init__(self, path: Path, fallback_rel: float) -> None:
        payload = json.loads(Path(path).read_text(encoding='utf-8'))
        rows = sorted(payload['layers'], key=lambda row: int(row['layer']))
        self.centers_cpu = [torch.tensor(row['centers'], dtype=torch.float32).contiguous() for row in rows]
        self.fallback_rel = float(fallback_rel)
        self.cache: dict[tuple[int, str], torch.Tensor] = {}
        self.reset()

    def reset(self) -> None:
        self.total = 0
        self.fallback = 0
        self.layer_total = [0 for _ in self.centers_cpu]
        self.layer_fallback = [0 for _ in self.centers_cpu]

    def effective_alpha(self, layer_idx: int, alpha: torch.Tensor) -> torch.Tensor:
        key = (int(layer_idx), str(alpha.device))
        if key not in self.cache:
            self.cache[key] = self.centers_cpu[int(layer_idx)].to(alpha.device)
        centers = self.cache[key]
        alpha_f = alpha.float()
        index = torch.argmin(torch.abs(alpha_f.unsqueeze(-1) - centers.view(*[1] * alpha_f.dim(), -1)), dim=-1)
        nearest = centers[index]
        fallback = torch.abs(nearest - alpha_f) / torch.clamp(torch.abs(alpha_f), min=1e-12) > self.fallback_rel
        count = int(alpha_f.numel())
        fallback_count = int(fallback.sum().item())
        self.total += count
        self.fallback += fallback_count
        self.layer_total[int(layer_idx)] += count
        self.layer_fallback[int(layer_idx)] += fallback_count
        return torch.where(fallback, alpha_f, nearest).to(alpha.dtype)

    def stats(self) -> dict[str, Any]:
        return {'total_tokens': int(self.total), 'fallback_tokens': int(self.fallback), 'fallback_rate': float(self.fallback / max(1, self.total)), 'layer_fallback_rate': [float(fallback / max(1, total)) for fallback, total in zip(self.layer_fallback, self.layer_total)]}

class DynamicInputLinear(nn.Module):

    def __init__(self, inner: nn.Module, controller: RecipeController, layer_idx: int, site: str, scale: torch.Tensor | None, permutation: torch.Tensor, block_size: int) -> None:
        super().__init__()
        self.inner = inner
        self.controller = controller
        self.layer_idx = int(layer_idx)
        self.site = str(site)
        self.weight_score_enabled = True
        self.scale_cpu = None if scale is None else scale.detach().cpu().float().contiguous()
        self.permutation_cpu = permutation.detach().cpu().long().contiguous()
        self.block_size = int(block_size)
        self.cache: dict[tuple[str, str], tuple[torch.Tensor | None, torch.Tensor]] = {}

    def tensors(self, x: torch.Tensor) -> tuple[torch.Tensor | None, torch.Tensor]:
        key = (str(x.device), str(x.dtype))
        if key not in self.cache:
            scale = None if self.scale_cpu is None else self.scale_cpu.to(device=x.device, dtype=torch.float32)
            self.cache[key] = (scale, self.permutation_cpu.to(x.device))
        return self.cache[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rate = self.controller.rate(self.layer_idx, self.site)
        if rate > 0.0:
            scale, permutation = self.tensors(x)
            x = blockwise_topk(x, scale if self.weight_score_enabled else None, rate, self.block_size, permutation)
        return self.inner(x)

class AllocationDecoderLayer(nn.Module):

    def __init__(self, inner: nn.Module, layer_idx: int, controller: RecipeController, d_dir: Path, pack_dir: Path, medians: MedianBank, adaptive: AdaptiveK4, block_size: int, permutation_seed: int) -> None:
        super().__init__()
        self.inner = inner
        self.layer_idx = int(layer_idx)
        self.controller = controller
        self.medians = medians
        self.adaptive = adaptive
        self.block_size = int(block_size)
        self.gate_scale_cpu = torch.load(Path(d_dir) / f'layer_{layer_idx:02d}_gate_proj_d.pt', map_location='cpu', weights_only=True).float().contiguous()
        self.base_cpu = torch.load(layer_dir(Path(pack_dir) / 'common', layer_idx) / 'base_transition.pt', map_location='cpu', weights_only=True).float().contiguous()
        hidden = int(inner.mlp.up_proj.in_features)
        intermediate = int(inner.mlp.down_proj.in_features)
        self.gate_perm_cpu = fixed_permutation(hidden, permutation_seed, f'layer{layer_idx}:gate')
        self.h1_perm_cpu = fixed_permutation(intermediate, permutation_seed, f'layer{layer_idx}:h1')
        self.cache: dict[tuple[str, str], torch.Tensor] = {}
        self.weighted_output_selectors = False
        self.gate_weight_score_enabled = True
        self.output_score_cache: dict[str, torch.Tensor] = {}

    def set_input_selector_weighting(self, *, qkv: bool, gate: bool) -> None:
        """Toggle only ranking weights; preserve masks' budgets and permutations."""
        for name in ('q_proj', 'k_proj', 'v_proj'):
            getattr(self.inner.self_attn, name).weight_score_enabled = bool(qkv)
        self.gate_weight_score_enabled = bool(gate)

    @torch.no_grad()
    def set_weighted_output_selectors(self, enabled: bool) -> None:
        """Backward-compatible switch for the joint O/FFN weighting path."""
        self.set_selector_weighting(o=bool(enabled), ffn='up_down' if enabled else 'none')

    @torch.no_grad()
    def set_selector_weighting(self, *, o: bool, ffn: str) -> None:
        """Split O input scoring from FFN output-channel scoring for ablations.

        FFN modes select the Up-output score factor. Any active FFN mode keeps
        the existing down-column weighting for additional Down pruning.
        Refresh these norms whenever absorbed weights are changed.
        """
        if ffn not in {'none', 'up', 'down', 'up_down'}:
            raise ValueError(f'Unknown FFN score mode: {ffn}')
        self.weighted_output_selectors = ffn != 'none'
        self.output_score_cache.clear()
        o_proj = self.inner.self_attn.o_proj
        if self.weighted_output_selectors:
            down = self.inner.mlp.down_proj.weight.float().norm(dim=0)
            if ffn == 'down':
                score = down
            else:
                up = self.inner.mlp.up_proj.weight.float().norm(dim=1)
                score = up if ffn == 'up' else up * down
            self.output_score_cache.update(down=down, up=score)
        if o:
            o_proj.scale_cpu = o_proj.inner.weight.float().norm(dim=0).cpu()
        else:
            o_proj.scale_cpu = None
        o_proj.cache.clear()

    def tensor(self, name: str, x: torch.Tensor) -> torch.Tensor:
        key = (name, str(x.device))
        if key not in self.cache:
            source = {'gate_scale': self.gate_scale_cpu, 'base': self.base_cpu, 'gate_perm': self.gate_perm_cpu, 'h1_perm': self.h1_perm_cpu}[name]
            dtype = torch.float32 if name == 'gate_scale' else x.dtype
            if name.endswith('perm'):
                self.cache[key] = source.to(device=x.device)
            else:
                self.cache[key] = source.to(device=x.device, dtype=dtype)
        return self.cache[key]

    @staticmethod
    def inv_rms(x: torch.Tensor, norm: nn.Module) -> torch.Tensor:
        eps = rms_epsilon(norm)
        return torch.rsqrt(torch.mean(x.float().square(), dim=-1) + eps).to(x.dtype)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None=None, position_ids: torch.LongTensor | None=None, past_key_values=None, use_cache: bool | None=False, cache_position: torch.LongTensor | None=None, position_embeddings: tuple[torch.Tensor, torch.Tensor] | None=None, **kwargs) -> torch.Tensor:
        u = hidden_states
        attn_alpha = self.inv_rms(u, self.inner.input_layernorm)
        attn_out, _ = self.inner.self_attn(hidden_states=u * attn_alpha.unsqueeze(-1), attention_mask=attention_mask, position_ids=position_ids, past_key_values=past_key_values, use_cache=use_cache, cache_position=cache_position, position_embeddings=position_embeddings, **kwargs)
        u_mid = u + attn_out
        alpha = self.inv_rms(u_mid, self.inner.post_attention_layernorm)
        dense_input = u_mid * alpha.unsqueeze(-1)
        gate_rate = self.controller.rate(self.layer_idx, 'gate')
        if gate_rate > 0.0:
            gate_u = blockwise_topk(u_mid, self.tensor('gate_scale', u_mid) if self.gate_weight_score_enabled else None, gate_rate, self.block_size, self.tensor('gate_perm', u_mid))
            gate = self.inner.mlp.gate_proj(gate_u * alpha.unsqueeze(-1))
        else:
            gate = self.inner.mlp.gate_proj(dense_input)
        up_dense = self.inner.mlp.up_proj(dense_input)
        h1 = self.inner.mlp.act_fn(gate)
        up_rate = self.controller.rate(self.layer_idx, 'up')
        down_rate = self.controller.rate(self.layer_idx, 'down')
        base = torch.matmul(u_mid, self.tensor('base', u_mid))
        if up_rate <= 0.0 and down_rate <= 0.0:
            return base + self.inner.mlp.down_proj(h1 * up_dense)
        median = self.medians.tensor(self.layer_idx, gate_rate, h1.device, h1.dtype).view(*[1] * (h1.dim() - 1), -1)
        centered = h1 - median
        permutation = self.tensor('h1_perm', h1)
        up_compute_rate = min(up_rate, down_rate)
        if up_compute_rate > 0.0:
            up_support = blockwise_topk(centered, self.output_score_cache['up'] if self.weighted_output_selectors else None, up_compute_rate, self.block_size, permutation).ne(0)
            up_sparse = torch.where(up_support, up_dense, torch.zeros_like(up_dense))
        else:
            up_sparse = up_dense
        down_centered = blockwise_topk(centered, None, down_rate, self.block_size, permutation) if down_rate > 0.0 else centered
        if self.weighted_output_selectors:
            sparse_product = centered * up_sparse
            if down_rate > up_compute_rate:
                sparse_product = blockwise_topk(sparse_product, self.output_score_cache['down'], down_rate, self.block_size, permutation)
            sparse_out = self.inner.mlp.down_proj(sparse_product)
        else:
            sparse_out = self.inner.mlp.down_proj(down_centered * up_sparse)
        effective_alpha = self.adaptive.effective_alpha(self.layer_idx, alpha)
        up_raw = up_dense.float() / alpha.float().unsqueeze(-1).clamp_min(1e-12)
        correction_input = median.float() * up_raw * effective_alpha.float().unsqueeze(-1)
        correction = self.inner.mlp.down_proj(correction_input.to(up_dense.dtype))
        return base + correction + sparse_out

def patch_model(model: nn.Module, controller: RecipeController, d_dir: Path, pack_dir: Path, medians: MedianBank, adaptive: AdaptiveK4, block_size: int, permutation_seed: int) -> list[AllocationDecoderLayer]:
    base_layers = list(decoder_layers(model))
    patched: list[AllocationDecoderLayer] = []
    for layer_idx, layer in enumerate(base_layers):
        qkv_scale = load_shared_qkv_scale(d_dir, layer_idx)
        qkv_perm = fixed_permutation(int(qkv_scale.numel()), permutation_seed, f'layer{layer_idx}:qkv')
        for site in ('q', 'k', 'v'):
            name = f'{site}_proj'
            setattr(layer.self_attn, name, DynamicInputLinear(getattr(layer.self_attn, name), controller, layer_idx, site, qkv_scale, qkv_perm, block_size))
        o_dim = int(layer.self_attn.o_proj.in_features)
        layer.self_attn.o_proj = DynamicInputLinear(layer.self_attn.o_proj, controller, layer_idx, 'o', None, fixed_permutation(o_dim, permutation_seed, f'layer{layer_idx}:o'), block_size)
        patched.append(AllocationDecoderLayer(layer, layer_idx, controller, d_dir, pack_dir, medians, adaptive, block_size, permutation_seed))
    model.model.layers = nn.ModuleList(patched)
    return patched

@torch.no_grad()
def evaluate_ppl(model: nn.Module, ids: np.ndarray, device: torch.device) -> dict[str, Any]:
    total_nll = 0.0
    total_tokens = 0
    for row in np.asarray(ids):
        tokens = torch.from_numpy(np.asarray(row, dtype=np.int64)).unsqueeze(0)
        input_ids = tokens[:, :-1].to(device)
        targets = tokens[:, 1:].to(device)
        attention_mask = torch.ones_like(input_ids)
        try:
            output = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, logits_to_keep=int(input_ids.shape[1]))
        except TypeError:
            output = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        logits = output.logits
        if int(logits.shape[1]) != int(targets.shape[1]):
            logits = logits[:, -int(targets.shape[1]):]
        total_nll += ce_sum(logits, targets)
        total_tokens += int(targets.numel())
        del output, logits, targets, input_ids, attention_mask
    mean_nll = total_nll / max(1, total_tokens)
    return {'ppl': float(math.exp(mean_nll)), 'mean_nll': float(mean_nll), 'total_nll': float(total_nll), 'total_tokens': int(total_tokens)}

@torch.no_grad()
def capture_layer_io(model: nn.Module, input_ids: torch.Tensor, device: torch.device) -> dict[int, dict[str, Any]]:
    captures: dict[int, dict[str, Any]] = {}
    handles = []
    for layer_idx, layer in enumerate(decoder_layers(model)):

        def pre_hook(_module, args, kwargs, idx=layer_idx):
            captures[idx] = {'args': detach_clone(args), 'kwargs': detach_clone(kwargs), 'input': args[0].detach().clone()}

        def post_hook(_module, _args, _kwargs, output, idx=layer_idx):
            value = output[0] if isinstance(output, tuple) else output
            captures[idx]['output'] = value.detach().clone()
        handles.append(layer.register_forward_pre_hook(pre_hook, with_kwargs=True))
        handles.append(layer.register_forward_hook(post_hook, with_kwargs=True))
    try:
        mask = torch.ones_like(input_ids, device=device)
        model(input_ids=input_ids.to(device), attention_mask=mask, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    return captures

def relative_block_error(output: torch.Tensor, dense: torch.Tensor, inp: torch.Tensor) -> float:
    error = (output.float() - dense.float()).square().mean()
    update = (dense.float() - inp.float()).square().mean().clamp_min(1e-12)
    return float((error / update).item())

def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)

@torch.no_grad()
def build_teal_paths(layers: list[AllocationDecoderLayer], controller: RecipeController, captures: dict[int, dict[str, Any]], max_target: float, base_step: float, max_rate: float, output: Path, force: bool) -> dict[int, list[dict[str, Any]]]:
    if output.exists() and (not force):
        payload = json.loads(output.read_text(encoding='utf-8'))
        if len(payload.get('paths', {})) == len(layers):
            return {int(key): value for key, value in payload['paths'].items()}
    paths: dict[int, list[dict[str, Any]]] = {}
    started = time.perf_counter()
    controller.set_recipe(np.zeros_like(controller.recipe))
    for layer_idx, layer in enumerate(layers):
        rates = np.zeros(len(SITES), dtype=np.float64)
        entries = [{'effective': 0.0, 'rates': rates.tolist(), 'proxy': 0.0, 'chosen': 'start'}]
        capture = captures[layer_idx]
        while weighted_sparsity(rates.reshape(1, -1)) < float(max_target) - 1e-09:
            best: tuple[float, int, np.ndarray] | None = None
            for site_idx, site in enumerate(SITES):
                candidate = rates.copy()
                candidate[site_idx] += float(base_step) / float(COSTS[site_idx])
                if candidate[site_idx] > float(max_rate) + 1e-09:
                    continue
                if candidate[SITE_INDEX['up']] > candidate[SITE_INDEX['down']] + 1e-09:
                    continue
                if weighted_sparsity(candidate.reshape(1, -1)) <= weighted_sparsity(rates.reshape(1, -1)) + 1e-12:
                    continue
                controller.set_layer(layer_idx, candidate)
                value = layer(*capture['args'], **capture['kwargs'])
                value = value[0] if isinstance(value, tuple) else value
                score = relative_block_error(value, capture['output'], capture['input'])
                if best is None or (score, site_idx) < (best[0], best[1]):
                    best = (score, site_idx, candidate)
            if best is None:
                raise RuntimeError(f'greedy search exhausted candidates at layer {layer_idx}')
            score, chosen_idx, rates = best
            controller.set_layer(layer_idx, rates)
            entries.append({'effective': weighted_sparsity(rates.reshape(1, -1)), 'rates': rates.tolist(), 'proxy': float(score), 'chosen': SITES[chosen_idx]})
        controller.set_layer(layer_idx, np.zeros(len(SITES), dtype=np.float64))
        paths[layer_idx] = entries
        print(f'[teal-greedy] layer={layer_idx:02d} steps={len(entries) - 1} elapsed={time.perf_counter() - started:.1f}s', flush=True)
        save_json(output, {'definition': 'TEAL cost-normalized decoder-block-output greedy on mixed-domain proxy', 'elapsed_s': time.perf_counter() - started, 'paths': {str(key): value for key, value in paths.items()}})
    return paths

def teal_recipe(paths: dict[int, list[dict[str, Any]]], target: float, max_rate: float) -> np.ndarray:
    rows = []
    for layer_idx in sorted(paths):
        entry = min(paths[layer_idx], key=lambda row: abs(float(row['effective']) - float(target)))
        rows.append(np.asarray(entry['rates'], dtype=np.float64))
    return repair_recipe(np.stack(rows), target, max_rate)

def larosa_recipe(target: float, alpha_att: float, alpha_ffn: float, layers: int) -> np.ndarray:
    alpha_o = 2.5 - 1.5 * float(alpha_att)
    alpha_mid = 1.5 - 0.5 * float(alpha_ffn)
    rates = float(target) * np.asarray([alpha_att, alpha_att, alpha_att, alpha_o, alpha_ffn, alpha_mid, alpha_mid], dtype=np.float64)
    return np.repeat(rates.reshape(1, -1), int(layers), axis=0)

class TrialStore:
    FIELDS = ('method', 'target', 'phase', 'alpha_att', 'alpha_ffn', 'recipe_hash', 'weighted_sparsity', 'ppl', 'mean_nll', 'total_nll', 'total_tokens', 'elapsed_s', 'recipe_json')

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.cache: dict[tuple[str, str, str], dict[str, Any]] = {}
        if self.path.exists():
            with self.path.open('r', encoding='utf-8', newline='') as handle:
                for row in csv.DictReader(handle):
                    self.cache[row['method'], row['phase'], row['recipe_hash']] = row

    def score(self, model: nn.Module, controller: RecipeController, adaptive: AdaptiveK4, device: torch.device, method: str, target: float, phase: str, recipe: np.ndarray, ids: np.ndarray, alpha_att: float | str='', alpha_ffn: float | str='') -> dict[str, Any]:
        digest = recipe_hash(recipe)
        key = (str(method), str(phase), digest)
        if key in self.cache:
            cached = self.cache[key]
            return {**cached, 'ppl': float(cached['ppl']), 'mean_nll': float(cached['mean_nll']), 'total_nll': float(cached['total_nll']), 'elapsed_s': float(cached['elapsed_s']), 'recipe': np.asarray(json.loads(cached['recipe_json']), dtype=np.float64)}
        controller.set_recipe(recipe)
        adaptive.reset()
        started = time.perf_counter()
        result = evaluate_ppl(model, ids, device)
        row = {'method': method, 'target': float(target), 'phase': phase, 'alpha_att': alpha_att, 'alpha_ffn': alpha_ffn, 'recipe_hash': digest, 'weighted_sparsity': weighted_sparsity(recipe), 'ppl': result['ppl'], 'mean_nll': result['mean_nll'], 'total_nll': result['total_nll'], 'total_tokens': result['total_tokens'], 'elapsed_s': time.perf_counter() - started, 'recipe_json': json.dumps(recipe.tolist(), separators=(',', ':'))}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        exists = self.path.exists() and self.path.stat().st_size > 0
        with self.path.open('a', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=self.FIELDS)
            if not exists:
                writer.writeheader()
            writer.writerow(row)
        self.cache[key] = {key: str(value) for key, value in row.items()}
        print(f"[ppl] {method} target={target:g} phase={phase} ppl={result['ppl']:.6f} elapsed={row['elapsed_s']:.2f}s", flush=True)
        return {**row, 'recipe': recipe.copy(), 'fallback': adaptive.stats()}

def select_domain_rows(ids: np.ndarray, length: int) -> np.ndarray:
    return np.asarray(ids[[0, 16, 22, 27], :int(length)], dtype=np.int64)

def plot_recipes(path: Path, recipes: dict[str, np.ndarray]) -> None:
    methods = list(recipes)
    fig, axes = plt.subplots(len(methods), 1, figsize=(13.2, 2.45 * len(methods)), dpi=180, sharex=True)
    if len(methods) == 1:
        axes = [axes]
    image = None
    for axis, method in zip(axes, methods):
        matrix = np.asarray(recipes[method]).T
        image = axis.imshow(matrix, aspect='auto', cmap='viridis', vmin=0.0, vmax=100.0)
        axis.set_yticks(range(len(SITES)))
        axis.set_yticklabels(SITES)
        axis.set_title(f'{method}: MAC-weighted sparsity {weighted_sparsity(recipes[method]):.4f}%')
    axes[-1].set_xlabel('layer')
    num_layers = int(np.asarray(next(iter(recipes.values()))).shape[0])
    axes[-1].set_xticks(range(num_layers))
    axes[-1].set_xticklabels(range(num_layers), fontsize=7)
    fig.colorbar(image, ax=axes, fraction=0.018, pad=0.012, label='sparsity (%)')
    fig.subplots_adjust(left=0.08, right=0.91, top=0.97, bottom=0.06, hspace=0.48)
    fig.savefig(path)
    plt.close(fig)

def write_recipe_csv(path: Path, recipes: dict[str, np.ndarray]) -> None:
    with path.open('w', encoding='utf-8', newline='') as handle:
        fields = ['method', 'layer', *SITES, 'layer_weighted_sparsity']
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for method, recipe in recipes.items():
            for layer_idx, rates in enumerate(recipe):
                row = {'method': method, 'layer': layer_idx}
                row.update({site: float(rates[SITE_INDEX[site]]) for site in SITES})
                row['layer_weighted_sparsity'] = weighted_sparsity(rates.reshape(1, -1))
                writer.writerow(row)

def main() -> None:
    args = parse_args()
    set_costs(args.costs)
    if args.smoke:
        args.targets = '50'
        args.proxy_len = 32
        args.grid_len = 64
        args.greedy_base_step = 25.0
        args.larosa_grid = '0.8,1.0,1.2'
        args.rerank_top = 1
        args.final_samples = 1
    targets = parse_targets(args.targets)
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    selection_path = args.root / 'mixed_calibration' / 'selection_input_ids.pt'
    selection_ids = torch.load(selection_path, map_location='cpu', weights_only=True).numpy()
    eval_ids, eval_meta = read_evalset(args.evalset_dir)
    eval_ids = eval_ids[:int(args.final_samples)]
    print(f'[load] model={args.model_id}', flush=True)
    model = load_model(args)
    device = input_device(model)
    original_dense = evaluate_ppl(model, eval_ids, device)
    print(f"[dense-original] ppl={original_dense['ppl']:.6f}", flush=True)
    apply_common_pack(model, args.pack_dir)
    num_layers = len(list(decoder_layers(model)))
    controller = RecipeController(num_layers)
    adaptive = AdaptiveK4(args.adaptive_json, args.fallback_rel)
    medians = MedianBank(args.median_dir, int(model.config.intermediate_size))
    layers = patch_model(model, controller, args.d_dir, args.pack_dir, medians, adaptive, args.block_size, args.permutation_seed)
    zero = np.zeros((num_layers, len(SITES)), dtype=np.float64)
    controller.set_recipe(zero)
    absorbed_dense = evaluate_ppl(model, eval_ids, device)
    dense_rel = abs(absorbed_dense['ppl'] - original_dense['ppl']) / max(original_dense['ppl'], 1e-12)
    print(f"[dense-absorbed] ppl={absorbed_dense['ppl']:.6f} relative_delta={100.0 * dense_rel:.4f}%", flush=True)
    if dense_rel > 0.01:
        raise RuntimeError('absorbed dense path differs from the original model by more than 1%')
    proxy_ids = torch.from_numpy(select_domain_rows(selection_ids, args.proxy_len))
    captures = capture_layer_io(model, proxy_ids, device)
    paths = build_teal_paths(layers, controller, captures, max(targets), args.greedy_base_step, args.max_rate, args.output_dir / 'teal_greedy_paths.json', args.force_greedy)
    del captures
    torch.cuda.empty_cache()
    store = TrialStore(args.output_dir / 'search_trials.csv')
    grid_ids = select_domain_rows(selection_ids, args.grid_len)
    full_mixed_ids = np.asarray(selection_ids, dtype=np.int64)
    alpha_grid = parse_float_grid(args.larosa_grid)
    recipes: dict[str, np.ndarray] = {}
    search_summary: dict[str, Any] = {}
    for target in targets:
        target_label = f's{int(round(target)):02d}'
        uniform = np.full((num_layers, len(SITES)), target, dtype=np.float64)
        teal = teal_recipe(paths, target, args.max_rate)
        recipes[f'uniform_{target_label}'] = uniform
        recipes[f'teal_greedy_{target_label}'] = teal
        store.score(model, controller, adaptive, device, 'uniform', target, f'proxy_{args.grid_len}', uniform, grid_ids)
        store.score(model, controller, adaptive, device, 'teal_greedy', target, f'proxy_{args.grid_len}', teal, grid_ids)
        candidates = []
        for alpha_att in alpha_grid:
            for alpha_ffn in alpha_grid:
                recipe = larosa_recipe(target, alpha_att, alpha_ffn, num_layers)
                row = store.score(model, controller, adaptive, device, 'larosa_grid', target, f'proxy_{args.grid_len}', recipe, grid_ids, alpha_att, alpha_ffn)
                candidates.append(row)
        rerank = sorted(candidates, key=lambda row: float(row['ppl']))[:args.rerank_top]
        reranked = [store.score(model, controller, adaptive, device, 'larosa_grid', target, 'rerank_mixed_32x512', np.asarray(row['recipe']), full_mixed_ids, row.get('alpha_att', ''), row.get('alpha_ffn', '')) for row in rerank]
        larosa_best = min(reranked, key=lambda row: float(row['ppl']))
        larosa = np.asarray(larosa_best['recipe'], dtype=np.float64)
        recipes[f'larosa_grid_{target_label}'] = larosa
        search_summary[target_label] = {'teal_recipe_hash': recipe_hash(teal), 'larosa_recipe_hash': recipe_hash(larosa), 'larosa_alpha_att': float(larosa_best.get('alpha_att') or 0.0), 'larosa_alpha_ffn': float(larosa_best.get('alpha_ffn') or 0.0), 'larosa_rerank_ppl': float(larosa_best['ppl'])}
    final_rows = [{'method': 'dense_original', 'target': 0.0, **original_dense, 'weighted_sparsity': 0.0, 'recipe_hash': 'original'}, {'method': 'dense_absorbed', 'target': 0.0, **absorbed_dense, 'weighted_sparsity': 0.0, 'recipe_hash': 'absorbed_zero'}]
    for method, recipe in recipes.items():
        target = float(method.rsplit('s', 1)[1])
        row = store.score(model, controller, adaptive, device, method.split('_s', 1)[0], target, 'final_wikitext_4x4096_full_sparse', recipe, eval_ids)
        final_rows.append({'method': method, 'target': target, 'ppl': float(row['ppl']), 'mean_nll': float(row['mean_nll']), 'total_nll': float(row['total_nll']), 'total_tokens': int(row['total_tokens']), 'weighted_sparsity': weighted_sparsity(recipe), 'recipe_hash': recipe_hash(recipe), 'fallback': row.get('fallback', adaptive.stats())})
    with (args.output_dir / 'final_ppl.csv').open('w', encoding='utf-8', newline='') as handle:
        fields = ['method', 'target', 'ppl', 'mean_nll', 'total_nll', 'total_tokens', 'weighted_sparsity', 'recipe_hash']
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in final_rows:
            writer.writerow({field: row.get(field, '') for field in fields})
    write_recipe_csv(args.output_dir / 'recipes_by_layer.csv', recipes)
    plot_recipes(args.output_dir / 'recipes_by_layer.png', recipes)
    save_json(args.output_dir / 'recipes.json', {key: value.tolist() for key, value in recipes.items()})
    save_json(args.output_dir / 'summary.json', {'status': 'complete', 'definition': {'calibration': 'mixed-domain disjoint selection split', 'body': 'full sparse, shared-QKV random-permuted B128, seven projection rates', 'budget': 'exact MAC-weighted target with Up realized as min(Up, Down)', 'transition': 'adaptive-DP K4 centers with 5% relative fallback', 'lm_head': 'dense for every likelihood score', 'teal': 'official cost-normalized decoder-block-output greedy adapted to Ours', 'larosa': 'global attention/FFN coefficient grid under exact MAC budget'}, 'args': {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}, 'eval_meta': eval_meta, 'search': search_summary, 'final_ppl': final_rows, 'elapsed_s': time.perf_counter() - started})
    print(json.dumps({'status': 'complete', 'output_dir': str(args.output_dir), 'final_ppl': final_rows}, indent=2), flush=True)
if __name__ == '__main__':
    main()
