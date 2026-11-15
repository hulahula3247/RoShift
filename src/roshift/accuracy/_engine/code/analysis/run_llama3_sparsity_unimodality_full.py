"""Full-sparse coordinate slices for seven Llama-3.1 Ours projections.

This experiment does not optimize a recipe.  It measures one-dimensional
budget-preserving slices for q, k, v, o, gate, up, and down:

* 32 x 7 decoder-branch relative-MSE curves on dense block inputs.
* 7 full-model WikiText PPL curves with the same recipe in every layer.

All model input positions are sparsified with block-local Top-K.  The final LM
head stays dense and the existing output-aware Q/absorbed weights are held
fixed.
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
from collections import defaultdict
from pathlib import Path
from typing import Any
import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import run_llama3_projection_allocation_search_20260720 as alloc
from run_llama2_ffn_gate_output_threshold_ppl import ce_sum, decoder_layers, input_device, load_model, read_evalset
from run_llama3_larosa_absorbed_lazy_ppl import apply_common_pack, layer_dir
ROOT = Path('/repro/data/llama3_1_8b/fair_calib_w103_4096x32_ppl_20260605')
SITES = ('q', 'k', 'v', 'o', 'gate', 'up', 'down')
ATTN_SITES = ('q', 'k', 'v', 'o')
FFN_SITES = ('gate', 'up', 'down')
LOCAL_FIELDS = ('layer', 'site', 'b', 'a', 'group_weighted_sparsity', 'global_weighted_sparsity_if_other_group_50', 'raw_mse', 'dense_energy', 'relative_mse', 'relative_l2', *SITES)
PPL_FIELDS = ('site', 'b', 'a', 'recipe_hash', 'weighted_sparsity', 'ppl', 'mean_nll', 'total_nll', 'total_tokens', 'elapsed_s', 'cached_from', *SITES)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Full-sparse seven-projection unimodality study.')
    parser.add_argument('--model_id', default='/repro/models/Meta-Llama-3.1-8B')
    parser.add_argument('--device_map', default='cuda:0')
    parser.add_argument('--attn_implementation', choices=('auto', 'eager', 'sdpa', 'flash_attention_2'), default='sdpa')
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--evalset_dir', type=Path, default=Path(_roshift_os.path.join(_roshift_os.environ['ROSHIFT_ENGINE_ROOT'], 'eval_data')))
    parser.add_argument('--output_dir', type=Path, default=ROOT / 'sparsity_unimodality_full_7proj_b128_20260811')
    parser.add_argument('--d_dir', type=Path, default=ROOT / 'output_aware_q' / 'd_cache' / 's_plus_lam_0p2')
    parser.add_argument('--median_dir', type=Path, default=ROOT / 'ours_daware_sparse_gate_medians')
    parser.add_argument('--pack_dir', type=Path, default=ROOT / 'ours_absorbed_pack')
    parser.add_argument('--target', type=float, default=50.0)
    parser.add_argument('--min_rate', type=float, default=10.0)
    parser.add_argument('--max_rate', type=float, default=70.0)
    parser.add_argument('--rate_step', type=float, default=5.0)
    parser.add_argument('--proxy_seq_len', type=int, default=4096)
    parser.add_argument('--ppl_seq_len', type=int, default=4096)
    parser.add_argument('--ppl_samples', type=int, default=4)
    parser.add_argument('--layers', default='0:31')
    parser.add_argument('--sites', default=','.join(SITES))
    parser.add_argument('--phases', default='local,ppl')
    parser.add_argument('--seed', type=int, default=20260811)
    parser.add_argument('--block_size', type=int, default=128)
    parser.add_argument('--permutation_seed', type=int, default=11)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--force', action='store_true')
    return parser.parse_args()

def parse_names(raw: str, allowed: tuple[str, ...], label: str) -> list[str]:
    values = [value.strip().lower() for value in str(raw).split(',') if value.strip()]
    bad = sorted(set(values).difference(allowed))
    if bad:
        raise ValueError(f'unsupported {label}: {bad}; allowed={list(allowed)}')
    return list(dict.fromkeys(values))

def parse_layers(raw: str, total: int) -> list[int]:
    text = str(raw).strip()
    if ':' in text and ',' not in text:
        lo, hi = (int(value) for value in text.split(':', 1))
        values = list(range(lo, hi + 1))
    else:
        values = sorted({int(value.strip()) for value in text.split(',') if value.strip()})
    if not values or min(values) < 0 or max(values) >= int(total):
        raise ValueError(f'invalid layers {values} for total={total}')
    return values

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def rate_grid(lo: float, hi: float, step: float) -> list[float]:
    count = int(round((float(hi) - float(lo)) / float(step)))
    return [round(float(lo) + index * float(step), 8) for index in range(count + 1)]

def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    tmp.replace(path)

def load_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open('r', encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle))

def append_csv(path: Path, fields: tuple[str, ...], row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open('a', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        if not exists:
            writer.writeheader()
        writer.writerow({field: row.get(field, '') for field in fields})

def _site_seed(seed: int, site: str) -> int:
    site_code = sum(((index + 1) * ord(char) for index, char in enumerate(site)))
    return int(seed) * 1000003 + site_code

def fixed_permutation(dim: int, seed: int, site: str) -> torch.Tensor:
    generator = torch.Generator(device='cpu')
    generator.manual_seed(_site_seed(int(seed), str(site)))
    return torch.randperm(int(dim), generator=generator, dtype=torch.long)

def blockwise_topk(x: torch.Tensor, scale: torch.Tensor | None, rate: float, block_size: int, permutation: torch.Tensor) -> torch.Tensor:
    """Keep an exact total quota using independent Top-K within B-sized blocks."""
    dim = int(x.shape[-1])
    block_size = int(block_size)
    if block_size <= 0 or dim % block_size:
        raise ValueError(f'dimension {dim} must be divisible by block_size={block_size}')
    keep = alloc.keep_count(dim, float(rate))
    if keep >= dim:
        return x
    if keep <= 0:
        return torch.zeros_like(x)
    perm = permutation.to(device=x.device, dtype=torch.long)
    if int(perm.numel()) != dim:
        raise ValueError(f'permutation length {perm.numel()} != dimension {dim}')
    score = x.float().abs()
    if scale is not None:
        score = score * scale.to(device=x.device, dtype=torch.float32).view(*[1] * (x.dim() - 1), dim)
    score = score.index_select(-1, perm)
    blocks = dim // block_size
    keep_lo, extra = divmod(keep, blocks)
    keep_hi = keep_lo + int(extra > 0)
    block_scores = score.reshape(*score.shape[:-1], blocks, block_size)
    local_indices = torch.topk(block_scores, k=keep_hi, dim=-1, largest=True, sorted=False).indices
    quota = torch.full((blocks,), keep_lo, device=x.device, dtype=torch.long)
    if extra:
        quota[:extra] += 1
    rank_shape = [1] * (local_indices.dim() - 2) + [1, keep_hi]
    ranks = torch.arange(keep_hi, device=x.device).view(*rank_shape)
    quota_shape = [1] * (local_indices.dim() - 2) + [blocks, 1]
    selected = ranks < quota.view(*quota_shape)
    mask_blocks = torch.zeros_like(block_scores, dtype=torch.bool)
    mask_blocks.scatter_(-1, local_indices, selected.expand_as(local_indices))
    mask_permuted = mask_blocks.reshape_as(score)
    mask = torch.zeros_like(x, dtype=torch.bool)
    scatter_index = perm.view(*[1] * (x.dim() - 1), dim).expand_as(x)
    mask.scatter_(-1, scatter_index, mask_permuted)
    return torch.where(mask, x, torch.zeros_like(x))

class SevenSiteController:

    def __init__(self, num_layers: int) -> None:
        self.recipe = np.zeros((int(num_layers), len(SITES)), dtype=np.float64)
        self.index = {site: index for index, site in enumerate(SITES)}

    def zero(self) -> None:
        self.recipe.fill(0.0)

    def set_recipe(self, recipe: np.ndarray) -> None:
        value = np.asarray(recipe, dtype=np.float64)
        if value.shape != self.recipe.shape:
            raise ValueError(f'recipe shape {value.shape} != {self.recipe.shape}')
        self.recipe[...] = value

    def set_layer(self, layer_idx: int, rates: dict[str, float]) -> None:
        self.recipe[int(layer_idx), :] = 0.0
        for site, value in rates.items():
            self.recipe[int(layer_idx), self.index[site]] = float(value)

    def rate(self, layer_idx: int, site: str) -> float:
        return float(self.recipe[int(layer_idx), self.index[str(site)]])

class DynamicInputLinear(nn.Module):

    def __init__(self, inner: nn.Module, controller: SevenSiteController, layer_idx: int, site: str, scale_file: Path | None, block_size: int, permutation: torch.Tensor) -> None:
        super().__init__()
        self.inner = inner
        self.controller = controller
        self.layer_idx = int(layer_idx)
        self.site = str(site)
        self.block_size = int(block_size)
        self.permutation_cpu = permutation.detach().cpu().long().contiguous()
        self.scale_cpu = None if scale_file is None else torch.load(scale_file, map_location='cpu', weights_only=True).float().contiguous()
        self.cache: dict[str, torch.Tensor] = {}
        self.permutation_cache: dict[str, torch.Tensor] = {}

    def scale(self, device: torch.device) -> torch.Tensor | None:
        if self.scale_cpu is None:
            return None
        key = str(device)
        if key not in self.cache:
            self.cache[key] = self.scale_cpu.to(device=device, dtype=torch.float32)
        return self.cache[key]

    def permutation(self, device: torch.device) -> torch.Tensor:
        key = str(device)
        if key not in self.permutation_cache:
            self.permutation_cache[key] = self.permutation_cpu.to(device=device)
        return self.permutation_cache[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rate = self.controller.rate(self.layer_idx, self.site)
        if rate > 0.0:
            x = blockwise_topk(x, self.scale(x.device), rate, self.block_size, self.permutation(x.device))
        return self.inner(x)

class SevenSiteOursLayer(nn.Module):

    def __init__(self, inner: nn.Module, layer_idx: int, controller: SevenSiteController, d_dir: Path, pack_dir: Path, medians: alloc.MedianBank, block_size: int, permutation_seed: int) -> None:
        super().__init__()
        self.inner = inner
        self.layer_idx = int(layer_idx)
        self.controller = controller
        self.medians = medians
        self.block_size = int(block_size)
        self.gate_scale_cpu = torch.load(alloc.d_path(d_dir, layer_idx, 'gate'), map_location='cpu', weights_only=True).float().contiguous()
        self.base_cpu = torch.load(layer_dir(Path(pack_dir) / 'common', layer_idx) / 'base_transition.pt', map_location='cpu', weights_only=True).float().contiguous()
        self.gate_scale_cache: dict[str, torch.Tensor] = {}
        self.base_cache: dict[tuple[str, str], torch.Tensor] = {}
        hidden = int(inner.mlp.up_proj.in_features)
        intermediate = int(inner.mlp.down_proj.in_features)
        self.gate_permutation_cpu = fixed_permutation(hidden, permutation_seed, f'layer{layer_idx}:gate')
        self.h1_permutation_cpu = fixed_permutation(intermediate, permutation_seed, f'layer{layer_idx}:h1')
        self.permutation_cache: dict[tuple[str, str], torch.Tensor] = {}
        self.last_attn: torch.Tensor | None = None
        self.last_ffn: torch.Tensor | None = None

    def gate_scale(self, device: torch.device) -> torch.Tensor:
        key = str(device)
        if key not in self.gate_scale_cache:
            self.gate_scale_cache[key] = self.gate_scale_cpu.to(device=device, dtype=torch.float32)
        return self.gate_scale_cache[key]

    def base_transition(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        key = (str(device), str(dtype))
        if key not in self.base_cache:
            self.base_cache[key] = self.base_cpu.to(device=device, dtype=dtype)
        return self.base_cache[key]

    def permutation(self, site: str, device: torch.device) -> torch.Tensor:
        key = (str(site), str(device))
        if key not in self.permutation_cache:
            source = self.gate_permutation_cpu if site == 'gate' else self.h1_permutation_cpu
            self.permutation_cache[key] = source.to(device=device)
        return self.permutation_cache[key]

    @staticmethod
    def inv_rms(x: torch.Tensor, norm: nn.Module) -> torch.Tensor:
        eps = float(getattr(norm, 'variance_epsilon', 1e-06))
        return torch.rsqrt(torch.mean(x.float().square(), dim=-1) + eps).to(x.dtype)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None=None, position_ids: torch.LongTensor | None=None, past_key_values=None, use_cache: bool | None=False, cache_position: torch.LongTensor | None=None, position_embeddings: tuple[torch.Tensor, torch.Tensor] | None=None, **kwargs) -> torch.Tensor:
        u = hidden_states
        attn_alpha = self.inv_rms(u, self.inner.input_layernorm)
        attn_out, _ = self.inner.self_attn(hidden_states=u * attn_alpha.unsqueeze(-1), attention_mask=attention_mask, position_ids=position_ids, past_key_values=past_key_values, use_cache=use_cache, cache_position=cache_position, position_embeddings=position_embeddings, **kwargs)
        self.last_attn = attn_out.detach()
        u_mid = u + attn_out
        mlp_alpha = self.inv_rms(u_mid, self.inner.post_attention_layernorm)
        dense_input = u_mid * mlp_alpha.unsqueeze(-1)
        gate_rate = self.controller.rate(self.layer_idx, 'gate')
        gate_u = blockwise_topk(u_mid, self.gate_scale(u_mid.device), gate_rate, self.block_size, self.permutation('gate', u_mid.device)) if gate_rate > 0.0 else u_mid
        gate = self.inner.mlp.gate_proj(gate_u * mlp_alpha.unsqueeze(-1))
        up_dense = self.inner.mlp.up_proj(dense_input)
        h1 = self.inner.mlp.act_fn(gate)
        up_rate = self.controller.rate(self.layer_idx, 'up')
        down_rate = self.controller.rate(self.layer_idx, 'down')
        if up_rate <= 0.0 and down_rate <= 0.0:
            mlp_out = self.inner.mlp.down_proj(h1 * up_dense)
        else:
            median = self.medians.tensor(self.layer_idx, gate_rate, h1.device, h1.dtype).view(*[1] * (h1.dim() - 1), -1)
            centered = h1 - median
            up_compute_rate = min(up_rate, down_rate)
            h1_permutation = self.permutation('h1', h1.device)
            up_support = blockwise_topk(centered, None, up_compute_rate, self.block_size, h1_permutation).ne(0)
            up_sparse = torch.where(up_support, up_dense, torch.zeros_like(up_dense))
            down_centered = blockwise_topk(centered, None, down_rate, self.block_size, h1_permutation)
            mlp_input = median * up_dense + down_centered * up_sparse
            mlp_out = self.inner.mlp.down_proj(mlp_input)
        self.last_ffn = mlp_out.detach()
        return torch.matmul(u_mid, self.base_transition(u_mid.device, u_mid.dtype)) + mlp_out

def patch_model(model: nn.Module, controller: SevenSiteController, d_dir: Path, pack_dir: Path, medians: alloc.MedianBank, block_size: int, permutation_seed: int) -> list[SevenSiteOursLayer]:
    base_layers = list(decoder_layers(model))
    patched: list[SevenSiteOursLayer] = []
    for layer_idx, layer in enumerate(base_layers):
        hidden = int(layer.mlp.up_proj.in_features)
        qkv_permutation = fixed_permutation(hidden, permutation_seed, f'layer{layer_idx}:qkv')
        for site in ('q', 'k', 'v'):
            name = f'{site}_proj'
            setattr(layer.self_attn, name, DynamicInputLinear(getattr(layer.self_attn, name), controller, layer_idx, site, alloc.d_path(d_dir, layer_idx, site), block_size, qkv_permutation))
        o_dim = int(layer.self_attn.o_proj.in_features)
        layer.self_attn.o_proj = DynamicInputLinear(layer.self_attn.o_proj, controller, layer_idx, 'o', None, block_size, fixed_permutation(o_dim, permutation_seed, f'layer{layer_idx}:o'))
        patched.append(SevenSiteOursLayer(layer, layer_idx, controller, d_dir, pack_dir, medians, block_size, permutation_seed))
    model.model.layers = nn.ModuleList(patched)
    return patched

def projection_costs(layers: list[SevenSiteOursLayer]) -> dict[str, float]:
    layer = layers[0].inner
    return {'q': float(layer.self_attn.q_proj.inner.weight.numel()), 'k': float(layer.self_attn.k_proj.inner.weight.numel()), 'v': float(layer.self_attn.v_proj.inner.weight.numel()), 'o': float(layer.self_attn.o_proj.inner.weight.numel()), 'gate': float(layer.mlp.gate_proj.weight.numel()), 'up': float(layer.mlp.up_proj.weight.numel()), 'down': float(layer.mlp.down_proj.weight.numel())}

def effective_rate(site: str, rates: dict[str, float]) -> float:
    if site == 'up':
        return min(float(rates['up']), float(rates['down']))
    return float(rates[site])

def weighted_sparsity(rates: dict[str, float], costs: dict[str, float], sites: tuple[str, ...]=SITES) -> float:
    numerator = sum((costs[site] * effective_rate(site, rates) for site in sites))
    denominator = sum((costs[site] for site in sites))
    return float(numerator / denominator)

def solve_slice(site: str, b: float, target: float, costs: dict[str, float], min_rate: float, max_rate: float) -> tuple[dict[str, float], float] | None:
    group = ATTN_SITES if site in ATTN_SITES else FFN_SITES

    def make(a: float) -> dict[str, float]:
        if site == 'up':
            if float(b) <= float(target):
                return {'gate': float(a), 'up': float(b), 'down': float(a)}
            return {'gate': float(a), 'up': float(b), 'down': float(b)}
        if site == 'down':
            if float(b) <= float(target):
                return {'gate': float(a), 'up': float(b), 'down': float(b)}
            return {'gate': float(a), 'up': float(a), 'down': float(b)}
        return {name: float(b) if name == site else float(a) for name in group}
    lo, hi = (0.0, 100.0)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        value = weighted_sparsity(make(mid), costs, group)
        if value < float(target):
            lo = mid
        else:
            hi = mid
    a = 0.5 * (lo + hi)
    rates = make(a)
    achieved = weighted_sparsity(rates, costs, group)
    if abs(achieved - float(target)) > 1e-06:
        return None
    if any((value < float(min_rate) - 1e-07 or value > float(max_rate) + 1e-07 for value in rates.values())):
        return None
    return (rates, a)

def full_recipe(num_layers: int, site: str, group_rates: dict[str, float], target: float) -> np.ndarray:
    recipe = np.full((int(num_layers), len(SITES)), float(target), dtype=np.float64)
    site_index = {name: index for index, name in enumerate(SITES)}
    for name, value in group_rates.items():
        recipe[:, site_index[name]] = float(value)
    return recipe

def rates_from_recipe(recipe: np.ndarray, layer_idx: int=0) -> dict[str, float]:
    return {site: float(recipe[int(layer_idx), index]) for index, site in enumerate(SITES)}

def recipe_hash(recipe: np.ndarray) -> str:
    value = np.round(np.asarray(recipe, dtype=np.float64), 8)
    return hashlib.sha1(value.tobytes()).hexdigest()[:16]

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
def capture_dense_layer_inputs(model: nn.Module, layers: list[SevenSiteOursLayer], input_ids: torch.Tensor, attention_mask: torch.Tensor, device: torch.device) -> dict[int, dict[str, Any]]:
    captures: dict[int, dict[str, Any]] = {}
    handles = []
    for layer_idx, layer in enumerate(layers):

        def pre_hook(_module, args, kwargs, idx=layer_idx):
            captures.setdefault(idx, {})['args'] = detach_clone(args)
            captures.setdefault(idx, {})['kwargs'] = detach_clone(kwargs)

        def post_hook(module, _args, _kwargs, _output, idx=layer_idx):
            captures.setdefault(idx, {})['dense_attn'] = module.last_attn.detach().clone()
            captures.setdefault(idx, {})['dense_ffn'] = module.last_ffn.detach().clone()
        handles.append(layer.register_forward_pre_hook(pre_hook, with_kwargs=True))
        handles.append(layer.register_forward_hook(post_hook, with_kwargs=True))
    try:
        model(input_ids=input_ids.to(device), attention_mask=attention_mask.to(device), use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    return captures

def branch_error(sparse: torch.Tensor, dense: torch.Tensor) -> dict[str, float]:
    diff = sparse.float() - dense.float()
    raw = float(diff.square().mean().item())
    energy = float(dense.float().square().mean().item())
    relative = raw / max(energy, 1e-20)
    return {'raw_mse': raw, 'dense_energy': energy, 'relative_mse': relative, 'relative_l2': math.sqrt(max(relative, 0.0))}

@torch.no_grad()
def evaluate_full_ppl(model: nn.Module, ids: np.ndarray, device: torch.device) -> dict[str, float | int]:
    total_nll = 0.0
    total_tokens = 0
    for row in np.asarray(ids):
        tokens = torch.from_numpy(row.astype(np.int64, copy=False)).unsqueeze(0)
        input_ids = tokens[:, :-1].to(device)
        targets = tokens[:, 1:].to(device)
        attention_mask = torch.ones_like(input_ids, device=device)
        try:
            output = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, logits_to_keep=int(input_ids.shape[1]))
        except TypeError:
            output = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        logits = output.logits
        if int(logits.shape[1]) != int(targets.shape[1]):
            logits = logits[:, -int(targets.shape[1]):, :]
        total_nll += ce_sum(logits, targets)
        total_tokens += int(targets.numel())
        del output, logits, targets, input_ids, attention_mask
    mean_nll = total_nll / max(1, total_tokens)
    return {'ppl': float(math.exp(mean_nll)), 'mean_nll': float(mean_nll), 'total_nll': float(total_nll), 'total_tokens': int(total_tokens)}

def curve_diagnostic(rows: list[dict[str, Any]], value_key: str) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: float(row['b']))
    xs = np.asarray([float(row['b']) for row in ordered], dtype=np.float64)
    ys = np.asarray([float(row[value_key]) for row in ordered], dtype=np.float64)
    if len(ys) < 3:
        return {'classification': 'insufficient', 'points': int(len(ys)), 'best_b': float(xs[int(np.argmin(ys))]) if len(ys) else None}
    span = float(np.max(ys) - np.min(ys))
    tolerance = max(1e-12, 0.01 * span)
    best = None
    for pivot in range(len(ys)):
        violations = []
        for index in range(pivot):
            violations.append(max(0.0, float(ys[index + 1] - ys[index])))
        for index in range(pivot, len(ys) - 1):
            violations.append(max(0.0, float(ys[index] - ys[index + 1])))
        magnitude = float(sum(violations))
        count = int(sum((value > tolerance for value in violations)))
        score = (magnitude, count, pivot)
        if best is None or score < best:
            best = score
    magnitude, count, pivot = best
    fraction = magnitude / max(span, 1e-20)
    if count == 0:
        classification = 'strict_unimodal'
    elif count <= 2 and fraction <= 0.1:
        classification = 'approximately_unimodal'
    else:
        classification = 'non_unimodal'
    return {'classification': classification, 'points': int(len(ys)), 'best_b': float(xs[int(np.argmin(ys))]), 'best_value': float(np.min(ys)), 'best_pivot_b': float(xs[int(pivot)]), 'reversal_count': int(count), 'violation_fraction': float(fraction), 'noise_tolerance': float(tolerance)}

def write_diagnostics(local_rows: list[dict[str, Any]], ppl_rows: list[dict[str, Any]], output_dir: Path) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    local_groups: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in local_rows:
        local_groups[int(row['layer']), str(row['site'])].append(row)
    for (layer, site), rows in sorted(local_groups.items()):
        output.append({'scope': 'local', 'layer': int(layer), 'site': site, **curve_diagnostic(rows, 'relative_mse')})
    ppl_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in ppl_rows:
        if str(row.get('site', '')).startswith('__'):
            continue
        ppl_groups[str(row['site'])].append(row)
    for site, rows in sorted(ppl_groups.items()):
        output.append({'scope': 'ppl', 'layer': 'all', 'site': site, **curve_diagnostic(rows, 'mean_nll')})
    path = output_dir / 'unimodality_summary.csv'
    if output:
        fields = list(dict.fromkeys((key for row in output for key in row.keys())))
        with path.open('w', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(output)
    return output

def plot_local(local_rows: list[dict[str, Any]], output_dir: Path) -> list[str]:
    paths: list[str] = []
    for site in SITES:
        rows = [row for row in local_rows if str(row['site']) == site]
        if not rows:
            continue
        layers = sorted({int(row['layer']) for row in rows})
        cols = 4 if len(layers) <= 16 else 8
        rows_count = int(math.ceil(len(layers) / cols))
        fig, axes = plt.subplots(rows_count, cols, figsize=(2.65 * cols, 2.25 * rows_count), dpi=180, sharex=True, sharey=True, squeeze=False)
        positive = [max(float(row['relative_mse']), 1e-14) for row in rows]
        ymin = max(1e-14, min(positive) * 0.65)
        ymax = max(positive) * 1.55
        for axis, layer_idx in zip(axes.flat, layers):
            sub = sorted([row for row in rows if int(row['layer']) == layer_idx], key=lambda row: float(row['b']))
            axis.plot([float(row['b']) for row in sub], [max(float(row['relative_mse']), 1e-14) for row in sub], color='#f97316', marker='o', markersize=2.8, linewidth=1.15)
            best = min(sub, key=lambda row: float(row['relative_mse']))
            axis.axvline(float(best['b']), color='#111827', linestyle='--', linewidth=0.7)
            axis.set_title(f'Layer {layer_idx}', fontsize=9)
            axis.set_yscale('log')
            axis.set_ylim(ymin, ymax)
            axis.grid(True, alpha=0.25)
        for axis in axes.flat[len(layers):]:
            axis.set_visible(False)
        fig.supxlabel(f'{site} sparsity (%)')
        fig.supylabel('Branch relative MSE')
        fig.suptitle(f'Full-sparse local unimodality sweep: {site}', y=0.995)
        fig.tight_layout(rect=(0.02, 0.03, 1.0, 0.97))
        path = output_dir / f'local_{site}_32_layers.png'
        fig.savefig(path)
        plt.close(fig)
        paths.append(str(path))
    return paths

def plot_ppl(ppl_rows: list[dict[str, Any]], output_dir: Path, block_size: int) -> list[str]:
    rows = [row for row in ppl_rows if str(row.get('site', '')) in SITES]
    if not rows:
        return []
    all_values = [float(row['ppl']) for row in rows]
    ymin = min(all_values)
    ymax = max(all_values)
    log_pad = 0.08
    log_ymin = max(1e-06, ymin / (ymax / ymin) ** log_pad)
    log_ymax = ymax * (ymax / ymin) ** log_pad
    paths: list[str] = []
    for site in SITES:
        sub = sorted([row for row in rows if str(row['site']) == site], key=lambda row: float(row['b']))
        if not sub:
            continue
        fig, axis = plt.subplots(figsize=(7.3, 4.6), dpi=180)
        axis.plot([float(row['b']) for row in sub], [float(row['ppl']) for row in sub], color='#f97316', marker='o', linewidth=1.5)
        axis.axvline(50.0, color='#6b7280', linestyle='--', linewidth=0.9)
        axis.set_yscale('log')
        axis.set_ylim(log_ymin, log_ymax)
        axis.set_xlabel(f'{site} sparsity (%)')
        axis.set_ylabel('Full-sparse WikiText PPL (log scale)')
        axis.set_title(f'Full-model unimodality sweep: {site}')
        axis.grid(True, alpha=0.28)
        fig.tight_layout()
        path = output_dir / f'ppl_{site}.png'
        fig.savefig(path)
        plt.close(fig)
        paths.append(str(path))
    fig, axes = plt.subplots(2, 4, figsize=(14.5, 7.2), dpi=180, sharey=True)
    for axis, site in zip(axes.flat, SITES):
        sub = sorted([row for row in rows if str(row['site']) == site], key=lambda row: float(row['b']))
        axis.plot([float(row['b']) for row in sub], [float(row['ppl']) for row in sub], color='#f97316', marker='o', markersize=3.5, linewidth=1.25)
        axis.axvline(50.0, color='#6b7280', linestyle='--', linewidth=0.8)
        axis.set_title(site)
        axis.grid(True, alpha=0.25)
        axis.set_yscale('log')
        axis.set_ylim(log_ymin, log_ymax)
    axes.flat[-1].set_visible(False)
    fig.supxlabel('Selected projection sparsity (%)')
    fig.supylabel('Full-sparse WikiText PPL (log scale)')
    fig.suptitle('Full-sparse PPL unimodality sweep', y=0.995)
    fig.tight_layout(rect=(0.02, 0.04, 1.0, 0.95))
    combined = output_dir / 'ppl_all_7_sites.png'
    fig.savefig(combined)
    plt.close(fig)
    paths.append(str(combined))
    focus_rows = []
    for row in rows:
        b = float(row['b'])
        ppl = float(row['ppl'])
        if b <= 75.0 and ppl <= 20.0 or 80.0 <= b <= 85.0:
            focus_rows.append(row)
    clip_level = 17.8
    focus_values = [min(float(row['ppl']), clip_level) for row in focus_rows]
    focus_ymin = min(focus_values)
    focus_ymax = max(focus_values)
    focus_pad = max(0.05, 0.05 * (focus_ymax - focus_ymin))
    fig, axes = plt.subplots(2, 4, figsize=(14.5, 7.8), dpi=180, sharey=True)
    for axis, site in zip(axes.flat, SITES):
        sub = sorted([row for row in focus_rows if str(row['site']) == site], key=lambda row: float(row['b']))
        x_values = [float(row['b']) for row in sub]
        actual_values = [float(row['ppl']) for row in sub]
        shown_values = [min(value, clip_level) for value in actual_values]
        axis.plot(x_values, shown_values, color='#f97316', marker='o', markersize=3.5, linewidth=1.25)
        for b, actual, shown in zip(x_values, actual_values, shown_values):
            if actual <= clip_level:
                continue
            axis.scatter([b], [shown], color='#b91c1c', marker='^', s=24, zorder=4)
            label = f'~{actual:.1f}' if actual < 1000.0 else f'~{actual:.1e}'
            axis.annotate(label, (b, shown), xytext=(0, 6), textcoords='offset points', ha='center', va='bottom', rotation=90, fontsize=6.5, color='#991b1b')
        best = min(sub, key=lambda row: float(row['ppl']))
        best_b = float(best['b'])
        axis.axvline(best_b, color='#6b7280', linestyle='--', linewidth=0.8)
        axis.set_xlim(-2.5, 87.5)
        axis.set_xticks([0, 15, 30, 45, 60, 75, 85])
        axis.set_ylim(max(0.0, focus_ymin - focus_pad), clip_level + 1.25)
        axis.set_yticks([9, 10, 12, 14, 16, 18])
        axis.set_title(site)
        axis.grid(True, alpha=0.25)
    axes.flat[-1].set_visible(False)
    curve_handle = axes.flat[0].lines[0]
    optimum_handle = axes.flat[0].lines[1]
    clipped_handle = Line2D([], [], color='#b91c1c', marker='^', linestyle='None', markersize=5)
    fig.legend([curve_handle, optimum_handle, clipped_handle], ['Measured full-sequence PPL', 'Minimum-PPL sparsity', 'Clipped point (~label: actual PPL)'], loc='upper center', bbox_to_anchor=(0.5, 0.94), ncol=3, frameon=False)
    fig.supxlabel('Selected projection sparsity (%)', y=0.105)
    fig.supylabel('Full-sparse WikiText PPL')
    fig.suptitle('Projection-wise Sparsity Allocation Sweep under a Fixed 50% Attention/FFN Budget', y=0.985)
    fig.text(0.5, 0.045, f'Model: Meta-Llama-3.1-8B | Data: WikiText-103 test, 4 x 4096 tokens (16,380 next-token targets) | Full-sequence sparsification\nSame allocation in all 32 layers | Per-token B{int(block_size)} block-local Top-K with exact total quota | Fixed output-aware Q and absorbed weights | Dense LM head\nMain basin: sparsity <= 75% and PPL <= 20 | Additional measured points: 80%, 85% | Values above the y-range are clipped and labeled.', ha='center', va='center', fontsize=8.2, color='#374151')
    fig.tight_layout(rect=(0.02, 0.15, 1.0, 0.89))
    focused = output_dir / 'ppl_all_7_sites_linear_clipped.png'
    fig.savefig(focused)
    plt.close(fig)
    paths.append(str(focused))
    return paths

def main() -> None:
    args = parse_args()
    if int(args.block_size) <= 0:
        raise ValueError('block_size must be positive')
    if args.smoke:
        args.layers = '0,31'
        args.sites = 'q,gate,down'
        args.min_rate = 40.0
        args.max_rate = 60.0
        args.rate_step = 10.0
        args.proxy_seq_len = min(int(args.proxy_seq_len), 64)
        args.ppl_seq_len = min(int(args.ppl_seq_len), 256)
        args.ppl_samples = 1
    phases = parse_names(args.phases, ('local', 'ppl'), 'phases')
    sites = parse_names(args.sites, SITES, 'sites')
    set_seed(int(args.seed))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    calibration_ids, calibration_mask, calibration_meta = alloc.load_calibration(args.root)
    eval_ids, eval_meta = read_evalset(args.evalset_dir)
    eval_ids = eval_ids[:int(args.ppl_samples), :int(args.ppl_seq_len)]
    print(f'[load] model={args.model_id}', flush=True)
    model = load_model(args)
    device = input_device(model)
    original_dense = None
    if 'ppl' in phases:
        original_dense = evaluate_full_ppl(model, eval_ids, device)
        print(f"[sanity] original dense full PPL={original_dense['ppl']:.6f}", flush=True)
    apply_common_pack(model, args.pack_dir)
    num_layers = len(list(decoder_layers(model)))
    intermediate = int(list(decoder_layers(model))[0].mlp.down_proj.in_features)
    controller = SevenSiteController(num_layers)
    medians = alloc.MedianBank(args.median_dir, intermediate)
    layers = patch_model(model, controller, args.d_dir, args.pack_dir, medians, int(args.block_size), int(args.permutation_seed))
    costs = projection_costs(layers)
    selected_layers = parse_layers(args.layers, num_layers)
    candidates: dict[str, list[tuple[float, float, dict[str, float]]]] = {}
    for site in sites:
        values = []
        for b in rate_grid(args.min_rate, args.max_rate, args.rate_step):
            solved = solve_slice(site, b, args.target, costs, args.min_rate, args.max_rate)
            if solved is not None:
                rates, a = solved
                values.append((float(b), float(a), rates))
        minimum_points = 2 if args.smoke else 3
        if len(values) < minimum_points:
            raise RuntimeError(f'only {len(values)} feasible points for {site}: {values}')
        candidates[site] = values
        print(f'[grid] {site}: {len(values)} points b={values[0][0]:g}..{values[-1][0]:g}', flush=True)
    local_path = args.output_dir / 'local_curves.csv'
    ppl_path = args.output_dir / 'ppl_curves.csv'
    if args.force:
        for path in (local_path, ppl_path, args.output_dir / 'unimodality_summary.csv'):
            path.unlink(missing_ok=True)
    local_existing = load_csv(local_path)
    local_done = {(int(row['layer']), str(row['site']), round(float(row['b']), 8)) for row in local_existing}
    if 'local' in phases:
        controller.zero()
        proxy_ids = torch.from_numpy(calibration_ids[:1, :int(args.proxy_seq_len)]).long()
        proxy_mask = torch.from_numpy(calibration_mask[:1, :int(args.proxy_seq_len)]).long()
        captures = capture_dense_layer_inputs(model, layers, proxy_ids, proxy_mask, device)
        for layer_idx in selected_layers:
            capture = captures[layer_idx]
            layer = layers[layer_idx]
            for site in sites:
                dense_branch = capture['dense_attn'] if site in ATTN_SITES else capture['dense_ffn']
                group = ATTN_SITES if site in ATTN_SITES else FFN_SITES
                for b, a, group_rates in candidates[site]:
                    key = (int(layer_idx), site, round(float(b), 8))
                    if key in local_done:
                        continue
                    controller.zero()
                    controller.set_layer(layer_idx, group_rates)
                    _ = layer(*capture['args'], **capture['kwargs'])
                    sparse_branch = layer.last_attn if site in ATTN_SITES else layer.last_ffn
                    error = branch_error(sparse_branch, dense_branch)
                    full_rates = {name: 50.0 for name in SITES}
                    full_rates.update(group_rates)
                    row = {'layer': int(layer_idx), 'site': site, 'b': float(b), 'a': float(a), 'group_weighted_sparsity': weighted_sparsity(group_rates, costs, group), 'global_weighted_sparsity_if_other_group_50': weighted_sparsity(full_rates, costs), **error, **{name: float(group_rates.get(name, 0.0)) for name in SITES}}
                    append_csv(local_path, LOCAL_FIELDS, row)
                    local_done.add(key)
                    print(f"[local] L{layer_idx:02d} {site} b={b:5.1f} a={a:6.2f} rel_mse={error['relative_mse']:.6g}", flush=True)
            controller.zero()
        del captures
        torch.cuda.empty_cache()
    ppl_existing = load_csv(ppl_path)
    ppl_done = {(str(row['site']), round(float(row['b']), 8)) for row in ppl_existing if row.get('b', '') != ''}
    result_by_hash: dict[str, dict[str, Any]] = {}
    for row in ppl_existing:
        digest = str(row.get('recipe_hash', ''))
        if digest:
            result_by_hash[digest] = row
    if 'ppl' in phases:
        zero = np.zeros((num_layers, len(SITES)), dtype=np.float64)
        controller.set_recipe(zero)
        zero_hash = recipe_hash(zero)
        if zero_hash not in result_by_hash:
            start = time.perf_counter()
            absorbed_dense = evaluate_full_ppl(model, eval_ids, device)
            row = {'site': '__dense_absorbed__', 'b': -1.0, 'a': -1.0, 'recipe_hash': zero_hash, 'weighted_sparsity': 0.0, **absorbed_dense, 'elapsed_s': time.perf_counter() - start, 'cached_from': '', **{name: 0.0 for name in SITES}}
            append_csv(ppl_path, PPL_FIELDS, row)
            result_by_hash[zero_hash] = row
            print(f"[sanity] absorbed dense full PPL={absorbed_dense['ppl']:.6f}", flush=True)
        for site in sites:
            for b, a, group_rates in candidates[site]:
                key = (site, round(float(b), 8))
                if key in ppl_done:
                    continue
                recipe = full_recipe(num_layers, site, group_rates, args.target)
                digest = recipe_hash(recipe)
                rates = rates_from_recipe(recipe)
                cached_from = ''
                if digest in result_by_hash:
                    prior = result_by_hash[digest]
                    rec = {'ppl': float(prior['ppl']), 'mean_nll': float(prior['mean_nll']), 'total_nll': float(prior['total_nll']), 'total_tokens': int(float(prior['total_tokens']))}
                    elapsed = 0.0
                    cached_from = str(prior.get('site', ''))
                else:
                    controller.set_recipe(recipe)
                    start = time.perf_counter()
                    rec = evaluate_full_ppl(model, eval_ids, device)
                    elapsed = time.perf_counter() - start
                row = {'site': site, 'b': float(b), 'a': float(a), 'recipe_hash': digest, 'weighted_sparsity': weighted_sparsity(rates, costs), **rec, 'elapsed_s': float(elapsed), 'cached_from': cached_from, **rates}
                append_csv(ppl_path, PPL_FIELDS, row)
                result_by_hash[digest] = row
                ppl_done.add(key)
                print(f"[ppl] {site} b={b:5.1f} a={a:6.2f} ppl={float(rec['ppl']):.6f} time={elapsed:.1f}s", flush=True)
                torch.cuda.empty_cache()
    local_rows = [dict(row) for row in load_csv(local_path)]
    ppl_rows = [dict(row) for row in load_csv(ppl_path)]
    diagnostics = write_diagnostics(local_rows, ppl_rows, args.output_dir)
    graphs = plot_local(local_rows, args.output_dir) + plot_ppl(ppl_rows, args.output_dir, int(args.block_size))
    payload = {'status': 'complete', 'scope': 'unimodality only; no coordinate descent or recipe update', 'sparsity_scope': 'full sparse over every evaluated model-input position', 'local_metric': 'branch relative MSE before residual add', 'attention_branch': 'q/k/v/o recipe, measured after o projection', 'ffn_branch': 'gate/up/down recipe, measured after down projection', 'budget': '50% parameter/MAC-weighted separately within attention and FFN', 'up_down_semantics': 'support-aware piecewise slices; Up computes union support at min(up,down), and Down reranks the same centered h1', 'mask': f'per-token B{int(args.block_size)} block-local top-k with exact total keep quota; fixed random QKV/Gate/O/intermediate partitions; no threshold drift', 'lm_head': 'dense', 'ppl': 'all 4095 next-token targets; body sparsity active from the first input token', 'median': 'calibrated/interpolated median bank with explicit extreme-rate artifacts', 'args': {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}, 'costs': costs, 'calibration_meta': calibration_meta, 'eval_meta': eval_meta, 'original_dense': original_dense, 'local_rows': len(local_rows), 'ppl_rows': len(ppl_rows), 'diagnostics': diagnostics, 'graphs': graphs, 'elapsed_s': time.perf_counter() - started}
    save_json(args.output_dir / 'summary.json', payload)
    print(json.dumps({'status': 'complete', 'local_rows': len(local_rows), 'ppl_rows': len(ppl_rows), 'graphs': len(graphs), 'output_dir': str(args.output_dir), 'elapsed_s': payload['elapsed_s']}, indent=2), flush=True)
if __name__ == '__main__':
    main()
