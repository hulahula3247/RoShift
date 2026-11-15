"""Search a 50%-MAC projection sparsity recipe for the current Llama-3.1 ours path.

The four search procedures are faithful, practical adaptations of the allocation
policies in TEAL, WAS, LaRoSA, and R-Sparse:

* TEAL: decoder-block-output greedy allocation.
* WAS: TEAL intra-block paths plus monotone inter-block TPE search.
* LaRoSA: a global coefficient grid over attention and FFN projection groups.
* R-Sparse: differential-evolution search under an exact MAC constraint.

The residual stream stays in the absorbed Q_l basis.  q/k/v/gate use the
projection-specific output-aware D score, o uses plain top-k, and the lazy h1
mask controls the combined up-output/down-input compute unit.  All search and
final evaluation use tail-only sparsity.  Lazy correction is exact: the
calibrated median is restored inside the down projection, so no K4/fallback
approximation is involved.
"""
from __future__ import annotations
import os as _roshift_os
import argparse
import csv
import hashlib
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
matplotlib.use('Agg')
from matplotlib import pyplot as plt
sys.path.append(str(Path(__file__).resolve().parent))
from run_llama2_ffn_gate_output_threshold_ppl import ce_sum, compute_tail_logits, decoder_layers, input_device, load_model, read_evalset
from run_llama3_larosa_absorbed_lazy_ppl import apply_common_pack, layer_dir, load_tensor
ROOT = Path('/repro/data/llama3_1_8b/fair_calib_w103_4096x32_ppl_20260605')
SITES = ('q', 'k', 'v', 'o', 'gate', 'up_down')
SITE_INDEX = {name: idx for idx, name in enumerate(SITES)}
COSTS = np.asarray([1.0, 0.25, 0.25, 1.0, 3.5, 7.0], dtype=np.float64)
TOTAL_COST = float(COSTS.sum())
SEARCH_METHODS = ('teal', 'larosa', 'was', 'rsparse')

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Projection-level sparsity allocation search for Llama-3.1 ours.')
    p.add_argument('--model_id', default='/repro/data/llama3_1_8b/hf/Meta-Llama-3.1-8B')
    p.add_argument('--device_map', default='cuda:0')
    p.add_argument('--attn_implementation', choices=['auto', 'eager', 'sdpa', 'flash_attention_2'], default='sdpa')
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--evalset_dir', type=Path, default=Path(_roshift_os.path.join(_roshift_os.environ['ROSHIFT_ENGINE_ROOT'], 'eval_data')))
    p.add_argument('--output_dir', type=Path, default=ROOT / 'projection_allocation_search_20260720')
    p.add_argument('--d_dir', type=Path, default=ROOT / 'output_aware_q' / 'd_cache' / 's_plus_lam_0p2')
    p.add_argument('--median_dir', type=Path, default=ROOT / 'ours_daware_sparse_gate_medians')
    p.add_argument('--pack_dir', type=Path, default=ROOT / 'ours_absorbed_pack')
    p.add_argument('--methods', default=','.join(SEARCH_METHODS))
    p.add_argument('--target', type=float, default=50.0)
    p.add_argument('--max_rate', type=float, default=70.0)
    p.add_argument('--final_tail_start', type=int, default=2048)
    p.add_argument('--final_samples', type=int, default=4)
    p.add_argument('--proxy_seq_len', type=int, default=256)
    p.add_argument('--search_seq_len', type=int, default=1024)
    p.add_argument('--rerank_seq_len', type=int, default=4096)
    p.add_argument('--rerank_samples', type=int, default=8)
    p.add_argument('--rerank_top', type=int, default=5)
    p.add_argument('--greedy_base_step', type=float, default=5.0, help='TEAL base step in percentage points.')
    p.add_argument('--greedy_max_target', type=float, default=55.0)
    p.add_argument('--larosa_grid', default='0.70:1.20:0.05')
    p.add_argument('--was_trials', type=int, default=50)
    p.add_argument('--was_knots', type=int, default=8)
    p.add_argument('--de_population', type=int, default=32)
    p.add_argument('--de_generations', type=int, default=5)
    p.add_argument('--de_mutation', type=float, default=0.5)
    p.add_argument('--de_crossover', type=float, default=0.5)
    p.add_argument('--method_budget_seconds', type=float, default=3600.0)
    p.add_argument('--seed', type=int, default=20260720)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--force_greedy', action='store_true')
    return p.parse_args()

def parse_methods(raw: str) -> list[str]:
    methods = [v.strip().lower() for v in raw.split(',') if v.strip()]
    invalid = sorted(set(methods) - set(SEARCH_METHODS))
    if invalid:
        raise ValueError(f'unknown methods {invalid}; allowed={list(SEARCH_METHODS)}')
    return methods

def parse_grid(raw: str) -> list[float]:
    if ':' not in raw:
        return [float(v.strip()) for v in raw.split(',') if v.strip()]
    lo, hi, step = (float(v) for v in raw.split(':'))
    count = int(round((hi - lo) / step))
    return [round(lo + i * step, 10) for i in range(count + 1)]

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def d_path(d_dir: Path, layer_idx: int, name: str) -> Path:
    return Path(d_dir) / f'layer_{int(layer_idx):02d}_{name}_proj_d.pt'

def keep_count(dim: int, rate: float) -> int:
    return max(0, min(int(dim), int(round(int(dim) * (100.0 - float(rate)) / 100.0))))

def exact_topk(x: torch.Tensor, d: torch.Tensor | None, rate: float) -> torch.Tensor:
    keep = keep_count(int(x.shape[-1]), rate)
    if keep >= int(x.shape[-1]):
        return x
    if keep <= 0:
        return torch.zeros_like(x)
    score = x.float().abs()
    if d is not None:
        score = score * d.to(device=x.device, dtype=torch.float32).view(*[1] * (x.dim() - 1), -1)
    idx = torch.topk(score, k=keep, dim=-1, largest=True, sorted=False).indices
    mask = torch.zeros_like(x, dtype=torch.bool)
    mask.scatter_(-1, idx, True)
    return torch.where(mask, x, torch.zeros_like(x))

def tail_topk(x: torch.Tensor, d: torch.Tensor | None, rate: float, sparse_start: int) -> torch.Tensor:
    if float(rate) <= 0.0:
        return x
    if x.dim() < 3 or int(sparse_start) <= 0:
        return exact_topk(x, d, rate)
    start = int(sparse_start)
    if start >= int(x.shape[1]):
        return x
    out = x.clone()
    out[:, start:, :] = exact_topk(x[:, start:, :], d, rate)
    return out

def weighted_sparsity(recipe: np.ndarray) -> float:
    arr = np.asarray(recipe, dtype=np.float64)
    return float((arr * COSTS.reshape(1, -1)).sum() / (arr.shape[0] * TOTAL_COST))

def repair_recipe(recipe: np.ndarray, target: float, max_rate: float) -> np.ndarray:
    """Add one scalar offset with clipping to meet the exact MAC target."""
    raw = np.asarray(recipe, dtype=np.float64)
    lo, hi = (-float(max_rate) * 2.0, float(max_rate) * 2.0)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        val = weighted_sparsity(np.clip(raw + mid, 0.0, float(max_rate)))
        if val < float(target):
            lo = mid
        else:
            hi = mid
    return np.clip(raw + 0.5 * (lo + hi), 0.0, float(max_rate))

def recipe_hash(recipe: np.ndarray) -> str:
    rounded = np.round(np.asarray(recipe, dtype=np.float64), 5)
    return hashlib.sha1(rounded.tobytes()).hexdigest()[:16]

class RecipeController:

    def __init__(self, num_layers: int) -> None:
        self.recipe = np.zeros((int(num_layers), len(SITES)), dtype=np.float64)
        self.sparse_start = 0

    def set_recipe(self, recipe: np.ndarray) -> None:
        arr = np.asarray(recipe, dtype=np.float64)
        if arr.shape != self.recipe.shape:
            raise ValueError(f'recipe shape {arr.shape} != {self.recipe.shape}')
        self.recipe[...] = arr

    def set_layer(self, layer_idx: int, rates: np.ndarray) -> None:
        self.recipe[int(layer_idx), :] = np.asarray(rates, dtype=np.float64)

    def rate(self, layer_idx: int, site: str) -> float:
        return float(self.recipe[int(layer_idx), SITE_INDEX[site]])

class MedianBank:

    def __init__(self, median_dir: Path, intermediate_size: int) -> None:
        self.root = Path(median_dir)
        self.intermediate_size = int(intermediate_size)
        self.available = sorted((int(p.name[1:]) for p in self.root.glob('s*') if p.is_dir()))
        if not self.available:
            raise FileNotFoundError(f'no sXX median directories under {self.root}')
        self.cpu: dict[tuple[int, int], torch.Tensor] = {}
        self.device_cache: dict[tuple[int, int, str, str], torch.Tensor] = {}

    def base(self, layer_idx: int, sparsity: int) -> torch.Tensor:
        key = (int(layer_idx), int(sparsity))
        if key not in self.cpu:
            path = self.root / f's{int(sparsity):02d}' / f'h1_median_layer_{int(layer_idx):02d}.bin'
            arr = np.fromfile(path, dtype=np.float16)
            if int(arr.size) != self.intermediate_size:
                raise RuntimeError(f'bad median size {arr.size} at {path}; expected {self.intermediate_size}')
            self.cpu[key] = torch.from_numpy(arr.copy()).float().contiguous()
        return self.cpu[key]

    def tensor(self, layer_idx: int, gate_rate: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        value = float(np.clip(gate_rate, self.available[0], self.available[-1]))
        lower = max((v for v in self.available if v <= value))
        upper = min((v for v in self.available if v >= value))
        cache_key = (int(layer_idx), int(round(value * 1000)), str(device), str(dtype))
        if cache_key in self.device_cache:
            return self.device_cache[cache_key]
        if lower == upper:
            med = self.base(layer_idx, lower)
        else:
            t = (value - lower) / float(upper - lower)
            med = self.base(layer_idx, lower) * (1.0 - t) + self.base(layer_idx, upper) * t
        out = med.to(device=device, dtype=dtype).contiguous()
        self.device_cache[cache_key] = out
        return out

class DynamicInputLinear(nn.Module):

    def __init__(self, inner: nn.Module, controller: RecipeController, layer_idx: int, site: str, d_file: Path | None) -> None:
        super().__init__()
        self.inner = inner
        self.controller = controller
        self.layer_idx = int(layer_idx)
        self.site = str(site)
        self.d_cpu = None if d_file is None else torch.load(d_file, map_location='cpu').float().contiguous()
        self.d_cache: dict[tuple[str, str], torch.Tensor] = {}

    def d(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor | None:
        if self.d_cpu is None:
            return None
        key = (str(device), str(dtype))
        if key not in self.d_cache:
            self.d_cache[key] = self.d_cpu.to(device=device, dtype=dtype).contiguous()
        return self.d_cache[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rate = self.controller.rate(self.layer_idx, self.site)
        sparse = tail_topk(x, self.d(x.device, x.dtype), rate, self.controller.sparse_start)
        return self.inner(sparse)

class DynamicOursDecoderLayer(nn.Module):

    def __init__(self, inner: nn.Module, layer_idx: int, controller: RecipeController, d_dir: Path, pack_dir: Path, medians: MedianBank) -> None:
        super().__init__()
        self.inner = inner
        self.layer_idx = int(layer_idx)
        self.controller = controller
        self.medians = medians
        self.gate_d_cpu = torch.load(d_path(d_dir, layer_idx, 'gate'), map_location='cpu').float().contiguous()
        self.gate_d_cache: dict[tuple[str, str], torch.Tensor] = {}
        self.base_cpu = torch.load(layer_dir(Path(pack_dir) / 'common', layer_idx) / 'base_transition.pt', map_location='cpu').float().contiguous()
        self.base_cache: dict[tuple[str, str], torch.Tensor] = {}

    def inv_rms(self, x: torch.Tensor, norm: nn.Module) -> torch.Tensor:
        eps = float(getattr(norm, 'variance_epsilon', 1e-06))
        return torch.rsqrt(torch.mean(x.float() * x.float(), dim=-1) + eps).to(dtype=x.dtype)

    def gate_d(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        key = (str(device), str(dtype))
        if key not in self.gate_d_cache:
            self.gate_d_cache[key] = self.gate_d_cpu.to(device=device, dtype=dtype).contiguous()
        return self.gate_d_cache[key]

    def base_transition(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        key = (str(device), str(dtype))
        if key not in self.base_cache:
            self.base_cache[key] = self.base_cpu.to(device=device, dtype=dtype).contiguous()
        return self.base_cache[key]

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None=None, position_ids: torch.LongTensor | None=None, past_key_values=None, use_cache: bool | None=False, cache_position: torch.LongTensor | None=None, position_embeddings: tuple[torch.Tensor, torch.Tensor] | None=None, **kwargs) -> torch.Tensor:
        u = hidden_states
        attn_alpha = self.inv_rms(u, self.inner.input_layernorm)
        attn_out, _ = self.inner.self_attn(hidden_states=u * attn_alpha.unsqueeze(-1), attention_mask=attention_mask, position_ids=position_ids, past_key_values=past_key_values, use_cache=use_cache, cache_position=cache_position, position_embeddings=position_embeddings, **kwargs)
        u_mid = u + attn_out
        mlp_alpha = self.inv_rms(u_mid, self.inner.post_attention_layernorm)
        dense_in = u_mid * mlp_alpha.unsqueeze(-1)
        gate_rate = self.controller.rate(self.layer_idx, 'gate')
        gate_u = tail_topk(u_mid, self.gate_d(u_mid.device, u_mid.dtype), gate_rate, self.controller.sparse_start)
        gate = self.inner.mlp.gate_proj(gate_u * mlp_alpha.unsqueeze(-1))
        up = self.inner.mlp.up_proj(dense_in)
        h1 = self.inner.mlp.act_fn(gate)
        mid_rate = self.controller.rate(self.layer_idx, 'up_down')
        shifted_h1 = h1
        if mid_rate > 0.0:
            med = self.medians.tensor(self.layer_idx, gate_rate, h1.device, h1.dtype)
            med_b = med.view(*[1] * (h1.dim() - 1), -1)
            start = int(self.controller.sparse_start)
            if h1.dim() < 3 or start <= 0:
                shifted_h1 = exact_topk(h1 - med_b, None, mid_rate) + med_b
            elif start < int(h1.shape[1]):
                shifted_h1 = h1.clone()
                shifted_h1[:, start:, :] = exact_topk(h1[:, start:, :] - med_b, None, mid_rate) + med_b
        mlp_out = self.inner.mlp.down_proj(shifted_h1 * up)
        return torch.matmul(u_mid, self.base_transition(u_mid.device, u_mid.dtype)) + mlp_out

def patch_model(model: nn.Module, controller: RecipeController, d_dir: Path, pack_dir: Path, medians: MedianBank) -> list[nn.Module]:
    base_layers = list(decoder_layers(model))
    patched: list[nn.Module] = []
    for layer_idx, layer in enumerate(base_layers):
        for site in ('q', 'k', 'v'):
            name = f'{site}_proj'
            setattr(layer.self_attn, name, DynamicInputLinear(getattr(layer.self_attn, name), controller, layer_idx, site, d_path(d_dir, layer_idx, site)))
        layer.self_attn.o_proj = DynamicInputLinear(layer.self_attn.o_proj, controller, layer_idx, 'o', None)
        patched.append(DynamicOursDecoderLayer(layer, layer_idx, controller, d_dir, pack_dir, medians))
    model.model.layers = nn.ModuleList(patched)
    return patched

def load_calibration(root: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    directory = Path(root) / 'shared_calibration'
    ids = torch.load(directory / 'input_ids.pt', map_location='cpu').numpy().astype(np.int64, copy=False)
    mask = torch.load(directory / 'attention_mask.pt', map_location='cpu').numpy().astype(np.int64, copy=False)
    meta = json.loads((directory / 'calibration_meta.json').read_text(encoding='utf-8'))
    return (ids, mask, meta)

@torch.no_grad()
def evaluate_ids(model: nn.Module, ids: np.ndarray, tail_start: int, device: torch.device) -> dict[str, float | int]:
    total_nll = 0.0
    total_tokens = 0
    for row in np.asarray(ids):
        logits, targets = compute_tail_logits(model, row, int(tail_start), device)
        total_nll += ce_sum(logits, targets)
        total_tokens += int(targets.numel())
        del logits, targets
    return {'total_nll': float(total_nll), 'total_tokens': int(total_tokens), 'ppl': float(math.exp(total_nll / max(1, total_tokens)))}

class SearchEvaluator:
    FIELDS = ('trial', 'method', 'phase', 'recipe_hash', 'weighted_sparsity', 'ppl', 'total_nll', 'total_tokens', 'elapsed_s', 'recipe_json')

    def __init__(self, model: nn.Module, controller: RecipeController, device: torch.device, csv_path: Path) -> None:
        self.model = model
        self.controller = controller
        self.device = device
        self.csv_path = Path(csv_path)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache: dict[tuple[str, str, str], dict[str, Any]] = {}
        self.trial = 0
        if self.csv_path.exists():
            with self.csv_path.open('r', encoding='utf-8', newline='') as handle:
                for row in csv.DictReader(handle):
                    self.cache[row['method'], row['phase'], row['recipe_hash']] = dict(row)
                    self.trial = max(self.trial, int(row['trial']) + 1)

    def score(self, method: str, phase: str, recipe: np.ndarray, ids: np.ndarray, tail_start: int) -> dict[str, Any]:
        digest = recipe_hash(recipe)
        key = (str(method), str(phase), digest)
        if key in self.cache:
            row = self.cache[key]
            return {**row, 'ppl': float(row['ppl']), 'elapsed_s': float(row['elapsed_s']), 'recipe': np.asarray(json.loads(row['recipe_json']), dtype=np.float64)}
        self.controller.set_recipe(recipe)
        self.controller.sparse_start = int(tail_start)
        start = time.perf_counter()
        rec = evaluate_ids(self.model, ids, int(tail_start), self.device)
        elapsed = time.perf_counter() - start
        row = {'trial': int(self.trial), 'method': str(method), 'phase': str(phase), 'recipe_hash': digest, 'weighted_sparsity': weighted_sparsity(recipe), 'ppl': float(rec['ppl']), 'total_nll': float(rec['total_nll']), 'total_tokens': int(rec['total_tokens']), 'elapsed_s': float(elapsed), 'recipe_json': json.dumps(np.asarray(recipe, dtype=np.float64).tolist(), separators=(',', ':'))}
        exists = self.csv_path.exists() and self.csv_path.stat().st_size > 0
        with self.csv_path.open('a', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=self.FIELDS)
            if not exists:
                writer.writeheader()
            writer.writerow(row)
        self.cache[key] = {k: str(v) for k, v in row.items()}
        self.trial += 1
        print(f"[score] {method}/{phase} trial={row['trial']} ppl={row['ppl']:.6f} s={row['weighted_sparsity']:.4f} time={elapsed:.2f}s hash={digest}", flush=True)
        return {**row, 'recipe': np.asarray(recipe, dtype=np.float64)}

def detach_clone(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, tuple):
        return tuple((detach_clone(v) for v in value))
    if isinstance(value, list):
        return [detach_clone(v) for v in value]
    if isinstance(value, dict):
        return {k: detach_clone(v) for k, v in value.items()}
    return value

@torch.no_grad()
def capture_layer_io(model: nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor, device: torch.device) -> dict[int, dict[str, Any]]:
    captures: dict[int, dict[str, Any]] = {}
    handles = []
    layers = list(decoder_layers(model))
    for idx, layer in enumerate(layers):

        def pre_hook(_module, args, kwargs, layer_idx=idx):
            captures.setdefault(layer_idx, {})['args'] = detach_clone(args)
            captures.setdefault(layer_idx, {})['kwargs'] = detach_clone(kwargs)
            captures.setdefault(layer_idx, {})['input_hidden'] = args[0].detach().clone()

        def post_hook(_module, _args, _kwargs, output, layer_idx=idx):
            out = output[0] if isinstance(output, tuple) else output
            captures.setdefault(layer_idx, {})['dense_output'] = out.detach().clone()
        handles.append(layer.register_forward_pre_hook(pre_hook, with_kwargs=True))
        handles.append(layer.register_forward_hook(post_hook, with_kwargs=True))
    try:
        model(input_ids=input_ids.to(device), attention_mask=attention_mask.to(device), use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    return captures

def block_relative_mse(sparse: torch.Tensor, dense: torch.Tensor, inp: torch.Tensor, tail_start: int) -> float:
    start = int(tail_start)
    if sparse.dim() >= 3 and 0 < start < int(sparse.shape[1]):
        sparse = sparse[:, start:, :]
        dense = dense[:, start:, :]
        inp = inp[:, start:, :]
    err = (sparse.float() - dense.float()).pow(2).mean()
    update = (dense.float() - inp.float()).pow(2).mean().clamp_min(1e-12)
    return float((err / update).item())

def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    tmp.replace(path)

@torch.no_grad()
def build_teal_greedy_paths(model: nn.Module, controller: RecipeController, captures: dict[int, dict[str, Any]], args: argparse.Namespace, path: Path) -> tuple[dict[int, list[dict[str, Any]]], float]:
    if path.exists() and (not args.force_greedy):
        raw = json.loads(path.read_text(encoding='utf-8'))
        if len(raw.get('paths', {})) == len(captures):
            return ({int(k): v for k, v in raw['paths'].items()}, float(raw.get('elapsed_s', 0.0)))
    started = time.perf_counter()
    paths: dict[int, list[dict[str, Any]]] = {}
    layers = list(decoder_layers(model))
    controller.sparse_start = int(args.proxy_seq_len // 2)
    for layer_idx, layer in enumerate(layers):
        rates = np.zeros(len(SITES), dtype=np.float64)
        entries: list[dict[str, Any]] = [{'effective': 0.0, 'rates': rates.tolist(), 'proxy': 0.0, 'chosen': 'start'}]
        capture = captures[layer_idx]
        while weighted_sparsity(rates.reshape(1, -1)) < float(args.greedy_max_target) - 1e-08:
            best: tuple[float, int, np.ndarray] | None = None
            for site_idx, site in enumerate(SITES):
                candidate = rates.copy()
                candidate[site_idx] += float(args.greedy_base_step) / float(COSTS[site_idx])
                if candidate[site_idx] > float(args.max_rate) + 1e-09:
                    continue
                controller.set_layer(layer_idx, candidate)
                out = layer(*capture['args'], **capture['kwargs'])
                out0 = out[0] if isinstance(out, tuple) else out
                score = block_relative_mse(out0, capture['dense_output'], capture['input_hidden'], controller.sparse_start)
                if best is None or score < best[0]:
                    best = (score, site_idx, candidate)
            if best is None:
                raise RuntimeError(f'TEAL greedy ran out of valid candidates at layer {layer_idx}')
            score, chosen_idx, rates = best
            controller.set_layer(layer_idx, rates)
            entries.append({'effective': weighted_sparsity(rates.reshape(1, -1)), 'rates': rates.tolist(), 'proxy': float(score), 'chosen': SITES[chosen_idx]})
        controller.set_layer(layer_idx, np.zeros(len(SITES), dtype=np.float64))
        paths[layer_idx] = entries
        elapsed = time.perf_counter() - started
        print(f'[TEAL greedy] layer={layer_idx:02d} steps={len(entries) - 1} elapsed={elapsed:.1f}s', flush=True)
        save_json(path, {'elapsed_s': elapsed, 'paths': {str(k): v for k, v in paths.items()}})
    return (paths, time.perf_counter() - started)

def nearest_path_rates(path: list[dict[str, Any]], target: float) -> np.ndarray:
    entry = min(path, key=lambda item: abs(float(item['effective']) - float(target)))
    return np.asarray(entry['rates'], dtype=np.float64)

def teal_recipe(paths: dict[int, list[dict[str, Any]]], target: float, max_rate: float) -> np.ndarray:
    recipe = np.stack([nearest_path_rates(paths[layer], target) for layer in sorted(paths)], axis=0)
    return repair_recipe(recipe, target, max_rate)

def larosa_recipe(target: float, alpha_att: float, alpha_ffn: float, num_layers: int) -> np.ndarray:
    alpha_o = 2.5 - 1.5 * float(alpha_att)
    alpha_mid = 1.5 - 0.5 * float(alpha_ffn)
    rates = float(target) * np.asarray([alpha_att, alpha_att, alpha_att, alpha_o, alpha_ffn, alpha_mid], dtype=np.float64)
    return np.repeat(rates.reshape(1, -1), int(num_layers), axis=0)

def monotone_schedule(values: Iterable[float], num_layers: int, target: float) -> np.ndarray:
    knots = np.sort(np.asarray(list(values), dtype=np.float64))
    knot_x = np.linspace(0.0, float(num_layers - 1), num=len(knots))
    raw = np.interp(np.arange(num_layers, dtype=np.float64), knot_x, 45.0 + 10.0 * knots)
    lo, hi = (-10.0, 10.0)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        mean = float(np.clip(raw + mid, 45.0, 55.0).mean())
        if mean < target:
            lo = mid
        else:
            hi = mid
    return np.clip(raw + 0.5 * (lo + hi), 45.0, 55.0)

def rerank(evaluator: SearchEvaluator, method: str, candidates: list[dict[str, Any]], ids: np.ndarray, tail_start: int, top_n: int) -> dict[str, Any]:
    unique: dict[str, dict[str, Any]] = {}
    for row in sorted(candidates, key=lambda item: float(item['ppl'])):
        recipe = np.asarray(row['recipe'], dtype=np.float64)
        unique.setdefault(recipe_hash(recipe), row)
    selected = list(unique.values())[:max(1, int(top_n))]
    phase = f'rerank_{int(ids.shape[0])}x{int(ids.shape[1])}_tail{int(tail_start)}'
    rescored = [evaluator.score(method, phase, np.asarray(row['recipe'], dtype=np.float64), ids, tail_start) for row in selected]
    return min(rescored, key=lambda item: float(item['ppl']))

def search_larosa(evaluator: SearchEvaluator, args: argparse.Namespace, search_ids: np.ndarray, rerank_ids: np.ndarray, num_layers: int, rerank_tail_start: int) -> tuple[dict[str, Any], float]:
    started = time.perf_counter()
    grid = parse_grid(args.larosa_grid)
    if args.smoke:
        grid = [0.8, 1.0, 1.2]
    candidates: list[dict[str, Any]] = []
    for alpha_att in grid:
        for alpha_ffn in grid:
            if time.perf_counter() - started > float(args.method_budget_seconds) and candidates:
                break
            recipe = larosa_recipe(args.target, alpha_att, alpha_ffn, num_layers)
            row = evaluator.score('larosa', 'search', recipe, search_ids, args.search_seq_len // 2)
            row['alpha_att'] = alpha_att
            row['alpha_ffn'] = alpha_ffn
            candidates.append(row)
        else:
            continue
        break
    best = rerank(evaluator, 'larosa', candidates, rerank_ids, rerank_tail_start, args.rerank_top)
    return (best, time.perf_counter() - started)

def search_was(evaluator: SearchEvaluator, args: argparse.Namespace, paths: dict[int, list[dict[str, Any]]], search_ids: np.ndarray, rerank_ids: np.ndarray, rerank_tail_start: int) -> tuple[dict[str, Any], float]:
    try:
        import optuna
    except ImportError as exc:
        raise RuntimeError('WAS search requires optuna; install it in the active WSL venv') from exc
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    started = time.perf_counter()
    candidates: list[dict[str, Any]] = []
    trials = 3 if args.smoke else int(args.was_trials)
    knots = max(2, int(args.was_knots))

    def objective(trial) -> float:
        values = [trial.suggest_float(f'u_{idx}', 0.0, 1.0) for idx in range(knots)]
        schedule = monotone_schedule(values, len(paths), args.target)
        recipe = np.stack([nearest_path_rates(paths[layer], schedule[layer]) for layer in sorted(paths)], axis=0)
        recipe = repair_recipe(recipe, args.target, args.max_rate)
        row = evaluator.score('was', 'search', recipe, search_ids, args.search_seq_len // 2)
        candidates.append(row)
        trial.set_user_attr('recipe_hash', recipe_hash(recipe))
        return float(row['ppl'])
    sampler = optuna.samplers.TPESampler(seed=int(args.seed), n_startup_trials=min(10, trials))
    study = optuna.create_study(direction='minimize', sampler=sampler)
    for _ in range(trials):
        if time.perf_counter() - started > float(args.method_budget_seconds) and candidates:
            break
        study.optimize(objective, n_trials=1, catch=(RuntimeError,))
    best = rerank(evaluator, 'was', candidates, rerank_ids, rerank_tail_start, args.rerank_top)
    return (best, time.perf_counter() - started)

def random_recipe(rng: np.random.Generator, shape: tuple[int, int], target: float, max_rate: float) -> np.ndarray:
    layer_trend = np.cumsum(rng.normal(0.0, 1.5, size=(shape[0], 1)), axis=0)
    layer_trend -= layer_trend.mean()
    component = rng.normal(0.0, 10.0, size=(1, shape[1]))
    noise = rng.normal(0.0, 4.0, size=shape)
    return repair_recipe(float(target) + layer_trend + component + noise, target, max_rate)

def search_rsparse(evaluator: SearchEvaluator, args: argparse.Namespace, seeds: list[np.ndarray], search_ids: np.ndarray, rerank_ids: np.ndarray, num_layers: int, rerank_tail_start: int) -> tuple[dict[str, Any], float]:
    started = time.perf_counter()
    rng = np.random.default_rng(int(args.seed) + 91)
    pop_size = 6 if args.smoke else int(args.de_population)
    generations = 1 if args.smoke else int(args.de_generations)
    population = [repair_recipe(seed, args.target, args.max_rate) for seed in seeds[:pop_size]]
    while len(population) < pop_size:
        population.append(random_recipe(rng, (num_layers, len(SITES)), args.target, args.max_rate))
    scored: dict[str, dict[str, Any]] = {}

    def score_recipe(recipe: np.ndarray) -> dict[str, Any]:
        digest = recipe_hash(recipe)
        if digest not in scored:
            scored[digest] = evaluator.score('rsparse', 'search', recipe, search_ids, args.search_seq_len // 2)
        return scored[digest]
    scores = [score_recipe(recipe) for recipe in population]
    for generation in range(generations):
        if time.perf_counter() - started > float(args.method_budget_seconds):
            break
        children: list[np.ndarray] = []
        for idx, parent in enumerate(population):
            pool = [j for j in range(pop_size) if j != idx]
            a, b, c = rng.choice(pool, size=3, replace=False)
            mutant = population[a] + float(args.de_mutation) * (population[b] - population[c])
            cross = rng.random(parent.shape) < float(args.de_crossover)
            cross[rng.integers(0, parent.shape[0]), rng.integers(0, parent.shape[1])] = True
            child = np.where(cross, mutant, parent)
            children.append(repair_recipe(child, args.target, args.max_rate))
        child_scores = [score_recipe(recipe) for recipe in children]
        combined = list(zip(population + children, scores + child_scores))
        combined.sort(key=lambda pair: float(pair[1]['ppl']))
        population = [pair[0] for pair in combined[:pop_size]]
        scores = [pair[1] for pair in combined[:pop_size]]
        print(f"[R-Sparse DE] generation={generation + 1} best={float(scores[0]['ppl']):.6f}", flush=True)
    candidates = list(scored.values())
    best = rerank(evaluator, 'rsparse', candidates, rerank_ids, rerank_tail_start, args.rerank_top)
    return (best, time.perf_counter() - started)

def write_recipe_csv(path: Path, recipes: dict[str, np.ndarray]) -> None:
    fields = ['method', 'layer', *SITES, 'up', 'down', 'layer_effective_sparsity']
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for method, recipe in recipes.items():
            for layer_idx, rates in enumerate(np.asarray(recipe)):
                row = {'method': method, 'layer': layer_idx}
                row.update({site: float(rates[SITE_INDEX[site]]) for site in SITES})
                row['up'] = float(rates[SITE_INDEX['up_down']])
                row['down'] = float(rates[SITE_INDEX['up_down']])
                row['layer_effective_sparsity'] = weighted_sparsity(rates.reshape(1, -1))
                writer.writerow(row)

def plot_final(path: Path, rows: list[dict[str, Any]]) -> None:
    labels = [row['method'] for row in rows]
    values = [float(row['ppl']) for row in rows]
    colors = ['#444444' if name.startswith('dense_') else '#b45309' if name == 'uniform_50' else '#2563eb' for name in labels]
    fig, ax = plt.subplots(figsize=(9.5, 5.2), dpi=180)
    bars = ax.bar(np.arange(len(labels)), values, color=colors, width=0.68)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha='right')
    ax.set_ylabel('WikiText PPL (tail-only sparsity)')
    ax.set_title('Llama-3.1-8B: 50% MAC-weighted projection allocation')
    ax.grid(axis='y', alpha=0.25)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f'{value:.4f}', ha='center', va='bottom', fontsize=8)
    ax.set_ylim(bottom=max(0.0, min(values) - 0.25), top=max(values) + max(0.25, 0.08 * (max(values) - min(values) + 1e-06)))
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)

def plot_recipes(path: Path, recipes: dict[str, np.ndarray], max_rate: float) -> None:
    methods = list(recipes)
    fig, axes = plt.subplots(len(methods), 1, figsize=(13.0, 2.2 * len(methods)), dpi=180, sharex=True)
    if len(methods) == 1:
        axes = [axes]
    image = None
    for ax, method in zip(axes, methods):
        matrix = np.asarray(recipes[method], dtype=np.float64).transpose(1, 0)
        image = ax.imshow(matrix, aspect='auto', cmap='viridis', vmin=0.0, vmax=float(max_rate))
        ax.set_title(f'{method} (global MAC-weighted sparsity {weighted_sparsity(recipes[method]):.2f}%)')
        ax.set_ylabel('projection')
        ax.set_yticks(np.arange(len(SITES)))
        ax.set_yticklabels(SITES)
    axes[-1].set_xlabel('layer')
    axes[-1].set_xticks(np.arange(32))
    axes[-1].set_xticklabels([str(i) for i in range(32)], fontsize=7)
    if image is not None:
        colorbar = fig.colorbar(image, ax=axes, fraction=0.018, pad=0.012)
        colorbar.set_label('sparsity (%)')
    fig.subplots_adjust(left=0.09, right=0.91, top=0.96, bottom=0.07, hspace=0.45)
    fig.savefig(path)
    plt.close(fig)

def main() -> None:
    args = parse_args()
    methods = parse_methods(args.methods)
    if args.smoke:
        args.proxy_seq_len = min(args.proxy_seq_len, 64)
        args.search_seq_len = min(args.search_seq_len, 128)
        args.rerank_seq_len = min(args.rerank_seq_len, 256)
        args.rerank_samples = 1
        args.rerank_top = 1
        args.final_samples = 1
        args.greedy_base_step = max(args.greedy_base_step, 20.0)
        args.greedy_max_target = args.target
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started_all = time.perf_counter()
    calibration_ids, calibration_mask, calibration_meta = load_calibration(args.root)
    eval_ids, eval_meta = read_evalset(args.evalset_dir)
    eval_ids = eval_ids[:int(args.final_samples)]
    model = load_model(args)
    device = input_device(model)
    prior_summary_path = args.output_dir / 'summary.json'
    original_dense = None
    if prior_summary_path.exists() and (not args.smoke):
        prior = json.loads(prior_summary_path.read_text(encoding='utf-8'))
        prior_rows = prior.get('final', [])
        prior_dense = next((row for row in prior_rows if row.get('method') == 'dense_original'), None)
        if prior_dense is not None and int(prior_dense.get('total_tokens', -1)) == int(eval_ids.shape[0]) * (int(eval_ids.shape[1]) - int(args.final_tail_start)):
            original_dense = {'ppl': float(prior_dense['ppl']), 'total_nll': float(prior_dense['total_nll']), 'total_tokens': int(prior_dense['total_tokens'])}
    if original_dense is None:
        original_dense = evaluate_ids(model, eval_ids, int(args.final_tail_start), device)
    print(f"[sanity] original dense ppl={float(original_dense['ppl']):.6f} tokens={int(original_dense['total_tokens'])}", flush=True)
    apply_common_pack(model, args.pack_dir)
    num_layers = len(list(decoder_layers(model)))
    intermediate = int(list(decoder_layers(model))[0].mlp.down_proj.in_features)
    controller = RecipeController(num_layers)
    medians = MedianBank(args.median_dir, intermediate)
    patch_model(model, controller, args.d_dir, args.pack_dir, medians)
    evaluator = SearchEvaluator(model, controller, device, args.output_dir / 'search_trials.csv')
    zero = np.zeros((num_layers, len(SITES)), dtype=np.float64)
    uniform = np.full_like(zero, float(args.target))
    proxy_ids = torch.from_numpy(calibration_ids[:1, :int(args.proxy_seq_len)])
    proxy_mask = torch.from_numpy(calibration_mask[:1, :int(args.proxy_seq_len)])
    controller.set_recipe(zero)
    controller.sparse_start = int(args.proxy_seq_len // 2)
    captures = capture_layer_io(model, proxy_ids, proxy_mask, device)
    greedy_path_file = args.output_dir / ('teal_greedy_paths_smoke.json' if args.smoke else 'teal_greedy_paths.json')
    paths, greedy_elapsed = build_teal_greedy_paths(model, controller, captures, args, greedy_path_file)
    del captures
    torch.cuda.empty_cache()
    search_ids = calibration_ids[:1, :int(args.search_seq_len)]
    rerank_seq_len = min(int(args.rerank_seq_len), int(calibration_ids.shape[1]))
    rerank_count = min(int(args.rerank_samples), int(calibration_ids.shape[0]))
    rerank_ids = calibration_ids[:rerank_count, :rerank_seq_len]
    rerank_tail_start = rerank_seq_len // 2
    best: dict[str, dict[str, Any]] = {}
    timings: dict[str, float] = {'teal_greedy': float(greedy_elapsed)}
    teal = teal_recipe(paths, args.target, args.max_rate)
    if 'teal' in methods:
        start = time.perf_counter()
        short = evaluator.score('teal', 'search', teal, search_ids, args.search_seq_len // 2)
        best['teal'] = rerank(evaluator, 'teal', [short], rerank_ids, rerank_tail_start, 1)
        timings['teal'] = time.perf_counter() - start + greedy_elapsed
    if 'larosa' in methods:
        best['larosa'], timings['larosa'] = search_larosa(evaluator, args, search_ids, rerank_ids, num_layers, rerank_tail_start)
    if 'was' in methods:
        best['was'], timings['was'] = search_was(evaluator, args, paths, search_ids, rerank_ids, rerank_tail_start)
    if 'rsparse' in methods:
        seeds = [uniform, teal]
        for name in ('larosa', 'was'):
            if name in best:
                seeds.append(np.asarray(best[name]['recipe'], dtype=np.float64))
        best['rsparse'], timings['rsparse'] = search_rsparse(evaluator, args, seeds, search_ids, rerank_ids, num_layers, rerank_tail_start)
    recipes: dict[str, np.ndarray] = {'uniform_50': uniform}
    recipes.update({name: np.asarray(row['recipe'], dtype=np.float64) for name, row in best.items()})
    write_recipe_csv(args.output_dir / 'best_recipes_long.csv', recipes)
    final_rows: list[dict[str, Any]] = [{'method': 'dense_original', 'ppl': float(original_dense['ppl']), 'total_nll': float(original_dense['total_nll']), 'total_tokens': int(original_dense['total_tokens']), 'weighted_sparsity': 0.0, 'search_elapsed_s': 0.0, 'recipe_hash': 'original_model'}]
    final_specs = [('dense_absorbed', zero), *recipes.items()]
    for name, recipe in final_specs:
        row = evaluator.score(name, 'full_test', recipe, eval_ids, int(args.final_tail_start))
        final_rows.append({'method': name, 'ppl': float(row['ppl']), 'total_nll': float(row['total_nll']), 'total_tokens': int(row['total_tokens']), 'weighted_sparsity': weighted_sparsity(recipe), 'search_elapsed_s': float(timings.get(name, 0.0)), 'recipe_hash': recipe_hash(recipe)})
    with (args.output_dir / 'final_ppl.csv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(final_rows[0]))
        writer.writeheader()
        writer.writerows(final_rows)
    plot_final(args.output_dir / 'projection_allocation_ppl.png', final_rows)
    plot_recipes(args.output_dir / 'projection_allocation_recipes.png', recipes, args.max_rate)
    sparse_rows = [row for row in final_rows if row['method'] not in {'dense_original', 'dense_absorbed', 'uniform_50'}]
    winner = min(sparse_rows, key=lambda row: float(row['ppl'])) if sparse_rows else None
    summary = {'definition': {'target': 'MAC-weighted 50% over q,k,v,o,gate,shared up-output/down-input', 'sites': list(SITES), 'costs': COSTS.tolist(), 'tail_only': True, 'lazy_correction': 'exact median restoration inside down_proj; no K4/fallback', 'teal': 'decoder-block-output greedy with cost-normalized increments', 'was': 'TEAL intra-block paths plus monotone layer schedule optimized with Optuna TPE', 'larosa': 'global qkv/o and gate/up_down coefficient grid under exact MAC budget', 'rsparse': 'R-Sparse differential-evolution operator adapted to the projection-rate recipe'}, 'args': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, 'calibration_meta': calibration_meta, 'eval_meta': eval_meta, 'timings_s': timings, 'final': final_rows, 'dense_equivalence': {'original_ppl': float(original_dense['ppl']), 'absorbed_ppl': float(next((row['ppl'] for row in final_rows if row['method'] == 'dense_absorbed'))), 'ppl_delta': float(next((row['ppl'] for row in final_rows if row['method'] == 'dense_absorbed'))) - float(original_dense['ppl'])}, 'winner': winner, 'elapsed_s': time.perf_counter() - started_all}
    save_json(args.output_dir / 'summary.json', summary)
    print(json.dumps({'winner': winner, 'final': final_rows, 'output_dir': str(args.output_dir)}, indent=2), flush=True)
if __name__ == '__main__':
    main()
