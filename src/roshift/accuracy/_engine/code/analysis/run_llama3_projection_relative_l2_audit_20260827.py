"""Controlled per-projection relative-L2 audit for TEAL, LaRoSA, and Roshift.

The audit captures the same dense WikiText activations once, then replays each
method locally.  This avoids comparing methods on different token streams or
averaging unstable token-wise relative errors.
"""
from __future__ import annotations
import os as _roshift_os
import argparse
import csv
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
import torch.nn.functional as F
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from run_llama2_ffn_gate_output_threshold_ppl import decoder_layers, input_device, load_model, read_evalset
ROOT = Path('/repro/data/llama3_1_8b/mixed_calib_c4_alpaca_metamath_code_20260812')
PROJECTIONS = ('q', 'k', 'v', 'o', 'gate', 'up', 'down')
METHODS = ('TEAL', 'LaRoSA', 'Ours')
COLORS = {'TEAL': '#e3342f', 'LaRoSA': '#2878b5', 'Ours': '#2ca02c'}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model_id', default='/repro/models/Meta-Llama-3.1-8B')
    parser.add_argument('--device_map', default='cuda:0')
    parser.add_argument('--attn_implementation', default='sdpa')
    parser.add_argument('--evalset_dir', type=Path, default=Path(_roshift_os.path.join(_roshift_os.environ['ROSHIFT_ENGINE_ROOT'], 'eval_data')))
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--output_dir', type=Path, default=Path('/repro/data/llama3_1_8b/projection_relative_l2_audit_20260827'))
    parser.add_argument('--sparsity', type=int, default=50)
    parser.add_argument('--num_sequences', type=int, default=4)
    parser.add_argument('--sample_tokens_per_sequence', type=int, default=128)
    parser.add_argument('--chunk_rows', type=int, default=64)
    parser.add_argument('--block_size', type=int, default=128)
    parser.add_argument('--permutation_seed', type=int, default=11)
    parser.add_argument('--layer_indices', default='')
    parser.add_argument('--force_capture', action='store_true')
    parser.add_argument('--force_recompute', action='store_true')
    parser.add_argument('--seed', type=int, default=20260827)
    return parser.parse_args()

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def parse_layers(raw: str, total: int) -> list[int]:
    if not str(raw).strip():
        return list(range(int(total)))
    result = sorted({int(value.strip()) for value in str(raw).split(',') if value.strip()})
    if not result or min(result) < 0 or max(result) >= int(total):
        raise ValueError(f'invalid layer indices {result} for total={total}')
    return result

def site_seed(seed: int, site: str) -> int:
    code = sum(((index + 1) * ord(char) for index, char in enumerate(site)))
    return int(seed) * 1000003 + code

def fixed_permutation(dim: int, seed: int, site: str) -> torch.Tensor:
    generator = torch.Generator(device='cpu')
    generator.manual_seed(site_seed(seed, site))
    return torch.randperm(int(dim), generator=generator, dtype=torch.long)

def keep_count(dim: int, sparsity: float) -> int:
    keep = round(int(dim) * (100.0 - float(sparsity)) / 100.0)
    return max(0, min(int(dim), int(keep)))

def exact_topk_mask(score: torch.Tensor, sparsity: float) -> torch.Tensor:
    dim = int(score.shape[-1])
    keep = keep_count(dim, sparsity)
    if keep >= dim:
        return torch.ones_like(score, dtype=torch.bool)
    if keep <= 0:
        return torch.zeros_like(score, dtype=torch.bool)
    indices = torch.topk(score.float(), k=keep, dim=-1, largest=True, sorted=False).indices
    mask = torch.zeros_like(score, dtype=torch.bool)
    mask.scatter_(-1, indices, True)
    return mask

def blockwise_topk_mask(x: torch.Tensor, scale: torch.Tensor | None, sparsity: float, block_size: int, permutation: torch.Tensor) -> torch.Tensor:
    dim = int(x.shape[-1])
    block_size = int(block_size)
    if block_size <= 0 or dim % block_size:
        raise ValueError(f'dimension {dim} is not divisible by B={block_size}')
    keep = keep_count(dim, sparsity)
    if keep >= dim:
        return torch.ones_like(x, dtype=torch.bool)
    if keep <= 0:
        return torch.zeros_like(x, dtype=torch.bool)
    perm = permutation.to(device=x.device, dtype=torch.long)
    score = x.float().abs()
    if scale is not None:
        score = score * scale.to(device=x.device, dtype=torch.float32).view(1, dim)
    score = score.index_select(-1, perm)
    blocks = dim // block_size
    keep_lo, extra = divmod(keep, blocks)
    keep_hi = keep_lo + int(extra > 0)
    block_score = score.reshape(-1, blocks, block_size)
    indices = torch.topk(block_score, k=keep_hi, dim=-1, largest=True, sorted=False).indices
    quota = torch.full((blocks,), keep_lo, device=x.device, dtype=torch.long)
    if extra:
        quota[:extra] += 1
    ranks = torch.arange(keep_hi, device=x.device).view(1, 1, keep_hi)
    selected = ranks < quota.view(1, blocks, 1)
    mask_blocks = torch.zeros_like(block_score, dtype=torch.bool)
    mask_blocks.scatter_(-1, indices, selected.expand_as(indices))
    mask_permuted = mask_blocks.reshape_as(score)
    mask = torch.zeros_like(x, dtype=torch.bool)
    mask.scatter_(-1, perm.view(1, dim).expand(int(x.shape[0]), dim), mask_permuted)
    return mask

def q_path(root: Path, layer_idx: int, kind: str) -> Path:
    return root / 'output_aware_q' / 'q_cache' / kind / f'q_layer_{int(layer_idx):02d}.pt'

def d_path(root: Path, layer_idx: int, projection: str) -> Path:
    return root / 'output_aware_q' / 'd_cache' / 's_plus_lam_0p2' / f'layer_{int(layer_idx):02d}_{projection}_proj_d.pt'

def median_path(root: Path, layer_idx: int, sparsity: int) -> Path:
    return root / 'ours_daware_sparse_gate_medians' / f's{int(sparsity):02d}' / f'h1_median_layer_{int(layer_idx):02d}.bin'

def threshold_path(root: Path) -> Path:
    return root / 'teal_thresholds_mixed_20260814' / 'teal_alllinear_thresholds.json'

def threshold_for(payload: dict[str, Any], layer_idx: int, hook: str, sparsity: int) -> float:
    return float(payload['layers'][f'layer_{int(layer_idx):02d}'][hook]['thresholds'][str(int(sparsity))])

def sample_positions(length: int, count: int) -> torch.Tensor:
    count = min(int(length), int(count))
    values = np.linspace(0, int(length) - 1, num=count, dtype=np.int64)
    values = np.unique(values)
    if int(values.size) != count:
        raise RuntimeError(f'expected {count} unique positions, got {values.size}')
    return torch.from_numpy(values)

@torch.inference_mode()
def capture_dense_inputs(*, model: torch.nn.Module, eval_ids: np.ndarray, layer_indices: list[int], tokens_per_sequence: int, output_dir: Path, force: bool) -> dict[str, Any]:
    capture_dir = output_dir / 'dense_input_capture'
    meta_path = capture_dir / 'meta.json'
    expected = [capture_dir / f'layer_{layer_idx:02d}.pt' for layer_idx in layer_indices]
    if not force and meta_path.exists() and all((path.exists() for path in expected)):
        return json.loads(meta_path.read_text(encoding='utf-8'))
    capture_dir.mkdir(parents=True, exist_ok=True)
    layers = list(decoder_layers(model))
    stores: dict[int, dict[str, list[torch.Tensor]]] = {layer_idx: {'attn_input': [], 'o_input': [], 'mlp_input': []} for layer_idx in layer_indices}
    state: dict[str, torch.Tensor] = {}
    handles = []

    def make_hook(layer_idx: int, key: str):

        def hook(_module, inputs, _output):
            x = inputs[0].detach()
            positions = state['positions'].to(device=x.device)
            sampled = x[0].index_select(0, positions).to(device='cpu', dtype=torch.float16)
            stores[layer_idx][key].append(sampled.contiguous())
        return hook
    for layer_idx in layer_indices:
        layer = layers[layer_idx]
        handles.append(layer.self_attn.q_proj.register_forward_hook(make_hook(layer_idx, 'attn_input')))
        handles.append(layer.self_attn.o_proj.register_forward_hook(make_hook(layer_idx, 'o_input')))
        handles.append(layer.mlp.gate_proj.register_forward_hook(make_hook(layer_idx, 'mlp_input')))
    device = input_device(model)
    started = time.perf_counter()
    try:
        for sequence_index, row in enumerate(np.asarray(eval_ids)):
            ids = torch.from_numpy(np.asarray(row, dtype=np.int64)).unsqueeze(0).to(device)
            state['positions'] = sample_positions(int(ids.shape[1]), tokens_per_sequence)
            mask = torch.ones_like(ids)
            model.model(input_ids=ids, attention_mask=mask, use_cache=False)
            print(f"[capture] sequence={sequence_index + 1}/{len(eval_ids)} tokens={int(state['positions'].numel())}", flush=True)
            del ids, mask
    finally:
        for handle in handles:
            handle.remove()
    rows_per_layer = int(len(eval_ids) * tokens_per_sequence)
    for layer_idx in layer_indices:
        payload = {key: torch.cat(parts, dim=0).contiguous() for key, parts in stores[layer_idx].items()}
        for key, value in payload.items():
            if int(value.shape[0]) != rows_per_layer:
                raise RuntimeError(f'layer {layer_idx} {key}: {value.shape[0]} rows != {rows_per_layer}')
        torch.save(payload, capture_dir / f'layer_{layer_idx:02d}.pt')
    meta = {'status': 'complete', 'sequences': int(len(eval_ids)), 'sequence_length': int(eval_ids.shape[1]), 'tokens_per_sequence': int(tokens_per_sequence), 'sampled_tokens_per_layer': rows_per_layer, 'layers': layer_indices, 'elapsed_sec': float(time.perf_counter() - started), 'definition': 'dense post-RMSNorm qkv/gate input and dense o_proj input'}
    meta_path.write_text(json.dumps(meta, indent=2), encoding='utf-8')
    return meta

class Accumulator:

    def __init__(self) -> None:
        self.error_sq = 0.0
        self.dense_sq = 0.0
        self.dropped = 0
        self.elements = 0

    def add(self, dense: torch.Tensor, sparse: torch.Tensor, mask: torch.Tensor) -> None:
        diff = sparse.float() - dense.float()
        self.error_sq += float(diff.square().sum().item())
        self.dense_sq += float(dense.float().square().sum().item())
        self.dropped += int((~mask).sum().item())
        self.elements += int(mask.numel())

    def row(self, layer: int, projection: str, method: str) -> dict[str, Any]:
        relative_l2 = math.sqrt(self.error_sq / max(self.dense_sq, 1e-30))
        return {'layer': int(layer), 'projection': projection, 'method': method, 'relative_l2': float(relative_l2), 'error_l2_sq_sum': float(self.error_sq), 'dense_l2_sq_sum': float(self.dense_sq), 'actual_sparsity': float(self.dropped / max(self.elements, 1)), 'selection_elements': int(self.elements)}

def linear(weight: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return F.linear(x.to(dtype=weight.dtype), weight)

def larosa_input(x: torch.Tensor, q: torch.Tensor, sparsity: float) -> tuple[torch.Tensor, torch.Tensor]:
    rotated = x.float().matmul(q)
    mask = exact_topk_mask(rotated.abs(), sparsity)
    restored = torch.where(mask, rotated, torch.zeros_like(rotated)).matmul(q.t())
    return (restored.to(dtype=x.dtype), mask)

def ours_input(*, x_actual: torch.Tensor, gamma: torch.Tensor, q: torch.Tensor, scale: torch.Tensor, sparsity: float, block_size: int, permutation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    gamma_f = gamma.float().view(1, -1)
    no_gamma = x_actual.float() / gamma_f
    rotated = no_gamma.matmul(q)
    mask = blockwise_topk_mask(rotated, scale, sparsity, block_size, permutation)
    sparse_rotated = torch.where(mask, rotated, torch.zeros_like(rotated))
    restored = sparse_rotated.matmul(q.t()) * gamma_f
    return (restored.to(dtype=x_actual.dtype), mask)

def teal_input(x: torch.Tensor, threshold: float) -> tuple[torch.Tensor, torch.Tensor]:
    mask = x.float().abs().ge(float(threshold))
    return (torch.where(mask, x, torch.zeros_like(x)), mask)

def load_median(root: Path, layer_idx: int, sparsity: int, size: int) -> torch.Tensor:
    values = np.fromfile(median_path(root, layer_idx, sparsity), dtype=np.float16)
    if int(values.size) != int(size):
        raise RuntimeError(f'median layer {layer_idx}: size={values.size}, expected={size}')
    return torch.from_numpy(values.copy())

@torch.inference_mode()
def evaluate_layer(*, model: torch.nn.Module, layer_idx: int, capture_path: Path, root: Path, thresholds: dict[str, Any], sparsity: int, block_size: int, permutation_seed: int, chunk_rows: int) -> list[dict[str, Any]]:
    layer = list(decoder_layers(model))[layer_idx]
    device = input_device(model)
    capture = torch.load(capture_path, map_location='cpu', weights_only=True)
    rows = int(capture['attn_input'].shape[0])
    hidden = int(capture['attn_input'].shape[-1])
    intermediate = int(layer.mlp.up_proj.out_features)
    q_larosa = torch.load(q_path(root, layer_idx, 'pca_mixed'), map_location='cpu', weights_only=True).to(device=device, dtype=torch.float32)
    q_ours = torch.load(q_path(root, layer_idx, 's_plus_lam_0p2'), map_location='cpu', weights_only=True).to(device=device, dtype=torch.float32)
    d_qkv = torch.sqrt(sum((torch.load(d_path(root, layer_idx, projection), map_location='cpu', weights_only=True).float().square() for projection in ('q', 'k', 'v'))).clamp_min_(1e-12)).to(device=device)
    d_gate = torch.load(d_path(root, layer_idx, 'gate'), map_location='cpu', weights_only=True).float().to(device=device)
    median = load_median(root, layer_idx, sparsity, intermediate).to(device=device, dtype=torch.float16)
    qkv_perm = fixed_permutation(hidden, permutation_seed, f'layer{layer_idx}:qkv').to(device)
    o_perm = fixed_permutation(hidden, permutation_seed, f'layer{layer_idx}:o').to(device)
    gate_perm = fixed_permutation(hidden, permutation_seed, f'layer{layer_idx}:gate').to(device)
    h1_perm = fixed_permutation(intermediate, permutation_seed, f'layer{layer_idx}:h1').to(device)
    attn_gamma = layer.input_layernorm.weight.detach().to(device)
    mlp_gamma = layer.post_attention_layernorm.weight.detach().to(device)
    weights = {'q': layer.self_attn.q_proj.weight.detach(), 'k': layer.self_attn.k_proj.weight.detach(), 'v': layer.self_attn.v_proj.weight.detach(), 'o': layer.self_attn.o_proj.weight.detach(), 'gate': layer.mlp.gate_proj.weight.detach(), 'up': layer.mlp.up_proj.weight.detach(), 'down': layer.mlp.down_proj.weight.detach()}
    stats = {(projection, method): Accumulator() for projection in PROJECTIONS for method in METHODS}
    teal_thresholds = {'attn': threshold_for(thresholds, layer_idx, 'attn_input', sparsity), 'o': threshold_for(thresholds, layer_idx, 'attn_o_input', sparsity), 'mlp': threshold_for(thresholds, layer_idx, 'mlp_input', sparsity), 'down': threshold_for(thresholds, layer_idx, 'mlp_down_input', sparsity)}
    for start in range(0, rows, int(chunk_rows)):
        end = min(start + int(chunk_rows), rows)
        attn_x = capture['attn_input'][start:end].to(device)
        o_x = capture['o_input'][start:end].to(device)
        mlp_x = capture['mlp_input'][start:end].to(device)
        teal_attn, teal_attn_mask = teal_input(attn_x, teal_thresholds['attn'])
        larosa_attn, larosa_attn_mask = larosa_input(attn_x, q_larosa, sparsity)
        ours_attn, ours_attn_mask = ours_input(x_actual=attn_x, gamma=attn_gamma, q=q_ours, scale=d_qkv, sparsity=sparsity, block_size=block_size, permutation=qkv_perm)
        for projection in ('q', 'k', 'v'):
            dense = linear(weights[projection], attn_x)
            stats[projection, 'TEAL'].add(dense, linear(weights[projection], teal_attn), teal_attn_mask)
            stats[projection, 'LaRoSA'].add(dense, linear(weights[projection], larosa_attn), larosa_attn_mask)
            stats[projection, 'Ours'].add(dense, linear(weights[projection], ours_attn), ours_attn_mask)
        dense_o = linear(weights['o'], o_x)
        teal_o, teal_o_mask = teal_input(o_x, teal_thresholds['o'])
        larosa_o_mask = exact_topk_mask(o_x.float().abs(), sparsity)
        larosa_o = torch.where(larosa_o_mask, o_x, torch.zeros_like(o_x))
        ours_o_mask = blockwise_topk_mask(o_x, None, sparsity, block_size, o_perm)
        ours_o = torch.where(ours_o_mask, o_x, torch.zeros_like(o_x))
        stats['o', 'TEAL'].add(dense_o, linear(weights['o'], teal_o), teal_o_mask)
        stats['o', 'LaRoSA'].add(dense_o, linear(weights['o'], larosa_o), larosa_o_mask)
        stats['o', 'Ours'].add(dense_o, linear(weights['o'], ours_o), ours_o_mask)
        teal_mlp, teal_mlp_mask = teal_input(mlp_x, teal_thresholds['mlp'])
        larosa_mlp, larosa_mlp_mask = larosa_input(mlp_x, q_larosa, sparsity)
        ours_mlp, ours_mlp_mask = ours_input(x_actual=mlp_x, gamma=mlp_gamma, q=q_ours, scale=d_gate, sparsity=sparsity, block_size=block_size, permutation=gate_perm)
        dense_gate = linear(weights['gate'], mlp_x)
        dense_up = linear(weights['up'], mlp_x)
        teal_gate = linear(weights['gate'], teal_mlp)
        teal_up = linear(weights['up'], teal_mlp)
        larosa_gate = linear(weights['gate'], larosa_mlp)
        larosa_up = linear(weights['up'], larosa_mlp)
        ours_gate = linear(weights['gate'], ours_mlp)
        stats['gate', 'TEAL'].add(dense_gate, teal_gate, teal_mlp_mask)
        stats['gate', 'LaRoSA'].add(dense_gate, larosa_gate, larosa_mlp_mask)
        stats['gate', 'Ours'].add(dense_gate, ours_gate, ours_mlp_mask)
        stats['up', 'TEAL'].add(dense_up, teal_up, teal_mlp_mask)
        stats['up', 'LaRoSA'].add(dense_up, larosa_up, larosa_mlp_mask)
        ours_h1 = F.silu(ours_gate)
        centered = ours_h1 - median.view(1, -1)
        ours_h1_mask = blockwise_topk_mask(centered, None, sparsity, block_size, h1_perm)
        ours_up = torch.where(ours_h1_mask, dense_up, torch.zeros_like(dense_up))
        stats['up', 'Ours'].add(dense_up, ours_up, ours_h1_mask)
        dense_down_input = F.silu(dense_gate) * dense_up
        dense_down = linear(weights['down'], dense_down_input)
        teal_down_input = F.silu(teal_gate) * teal_up
        teal_down_sparse, teal_down_mask = teal_input(teal_down_input, teal_thresholds['down'])
        teal_down = linear(weights['down'], teal_down_sparse)
        larosa_down_input = F.silu(larosa_gate) * larosa_up
        larosa_down_mask = exact_topk_mask(larosa_down_input.float().abs(), sparsity)
        larosa_down_sparse = torch.where(larosa_down_mask, larosa_down_input, torch.zeros_like(larosa_down_input))
        larosa_down = linear(weights['down'], larosa_down_sparse)
        centered_sparse = torch.where(ours_h1_mask, centered, torch.zeros_like(centered))
        ours_down_input = median.view(1, -1) * dense_up + centered_sparse * ours_up
        ours_down = linear(weights['down'], ours_down_input)
        stats['down', 'TEAL'].add(dense_down, teal_down, teal_down_mask)
        stats['down', 'LaRoSA'].add(dense_down, larosa_down, larosa_down_mask)
        stats['down', 'Ours'].add(dense_down, ours_down, ours_h1_mask)
        del attn_x, o_x, mlp_x
    del q_larosa, q_ours, d_qkv, d_gate, median, capture
    torch.cuda.empty_cache()
    return [stats[projection, method].row(layer_idx, projection, method) for projection in PROJECTIONS for method in METHODS]

def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row['projection']), str(row['method'])].append(row)
    output = []
    for projection in PROJECTIONS:
        for method in METHODS:
            values = grouped[projection, method]
            error_sq = sum((float(row['error_l2_sq_sum']) for row in values))
            dense_sq = sum((float(row['dense_l2_sq_sum']) for row in values))
            dropped = sum((float(row['actual_sparsity']) * int(row['selection_elements']) for row in values))
            elements = sum((int(row['selection_elements']) for row in values))
            output.append({'projection': projection, 'method': method, 'pooled_relative_l2': math.sqrt(error_sq / max(dense_sq, 1e-30)), 'mean_layer_relative_l2': float(np.mean([float(row['relative_l2']) for row in values])), 'actual_sparsity': float(dropped / max(elements, 1)), 'layers': len(values)})
    return output

def plot_relative_l2(rows: list[dict[str, Any]], output_dir: Path, args: argparse.Namespace) -> Path:
    fig, axes = plt.subplots(len(PROJECTIONS), 1, figsize=(14.5, 17.0), sharex=True)
    for axis, projection in zip(axes, PROJECTIONS):
        for method in METHODS:
            values = sorted((row for row in rows if row['projection'] == projection and row['method'] == method), key=lambda row: int(row['layer']))
            axis.plot([int(row['layer']) for row in values], [float(row['relative_l2']) for row in values], color=COLORS[method], marker='o', markersize=3.4, linewidth=1.6, label=method)
        axis.set_ylabel(projection)
        axis.grid(True, linestyle='--', linewidth=0.45, alpha=0.35)
        axis.set_ylim(bottom=0.0)
    axes[0].legend(loc='upper right', ncol=3, frameon=True)
    axes[-1].set_xlabel('Layer')
    axes[-1].set_xticks(range(32))
    fig.suptitle(f'Projection Output Relative L2 by Layer (Uniform {args.sparsity}% Sparsity)', y=0.993, fontsize=14)
    fig.text(0.5, 0.006, f'Llama-3.1-8B; WikiText-103 test; pooled L2 over {args.num_sequences} x {args.sample_tokens_per_sequence} sampled tokens; same dense activations and mixed-domain calibration; Up = effective masked output, Down = lazy-corrected FFN output', ha='center', fontsize=9)
    fig.tight_layout(rect=(0.02, 0.02, 0.995, 0.982))
    path = output_dir / 'projection_output_relative_l2_by_layer.png'
    fig.savefig(path, dpi=190)
    plt.close(fig)
    return path

def plot_actual_sparsity(rows: list[dict[str, Any]], output_dir: Path, args: argparse.Namespace) -> Path:
    fig, axes = plt.subplots(len(PROJECTIONS), 1, figsize=(14.5, 16.0), sharex=True, sharey=True)
    target = float(args.sparsity) / 100.0
    for axis, projection in zip(axes, PROJECTIONS):
        for method in METHODS:
            values = sorted((row for row in rows if row['projection'] == projection and row['method'] == method), key=lambda row: int(row['layer']))
            axis.plot([int(row['layer']) for row in values], [float(row['actual_sparsity']) * 100.0 for row in values], color=COLORS[method], marker='o', markersize=2.8, linewidth=1.3, label=method)
        axis.axhline(target * 100.0, color='#333333', linestyle='--', linewidth=0.9)
        axis.set_ylabel(projection)
        axis.grid(True, linestyle='--', linewidth=0.45, alpha=0.3)
    axes[0].legend(loc='upper right', ncol=3)
    axes[-1].set_xlabel('Layer')
    axes[-1].set_xticks(range(32))
    fig.suptitle('Actual Selected Sparsity by Layer', y=0.993)
    fig.tight_layout(rect=(0.02, 0.02, 0.995, 0.982))
    path = output_dir / 'actual_sparsity_by_layer.png'
    fig.savefig(path, dpi=190)
    plt.close(fig)
    return path

def main() -> None:
    args = parse_args()
    if not 0 < int(args.sparsity) < 100:
        raise ValueError('--sparsity must be between 0 and 100')
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    thresholds = json.loads(threshold_path(args.root).read_text(encoding='utf-8'))
    eval_ids, eval_meta = read_evalset(args.evalset_dir)
    eval_ids = np.asarray(eval_ids[:int(args.num_sequences)])
    model = load_model(args)
    layers = list(decoder_layers(model))
    layer_indices = parse_layers(args.layer_indices, len(layers))
    capture_meta = capture_dense_inputs(model=model, eval_ids=eval_ids, layer_indices=layer_indices, tokens_per_sequence=args.sample_tokens_per_sequence, output_dir=args.output_dir, force=args.force_capture)
    partial_path = args.output_dir / 'rows_partial.json'
    rows: list[dict[str, Any]] = []
    completed: set[int] = set()
    if partial_path.exists() and (not args.force_recompute):
        rows = json.loads(partial_path.read_text(encoding='utf-8'))
        completed = {int(row['layer']) for row in rows}
    started = time.perf_counter()
    capture_dir = args.output_dir / 'dense_input_capture'
    for position, layer_idx in enumerate(layer_indices, start=1):
        if layer_idx in completed:
            print(f'[skip] layer={layer_idx:02d} already complete', flush=True)
            continue
        layer_rows = evaluate_layer(model=model, layer_idx=layer_idx, capture_path=capture_dir / f'layer_{layer_idx:02d}.pt', root=args.root, thresholds=thresholds, sparsity=args.sparsity, block_size=args.block_size, permutation_seed=args.permutation_seed, chunk_rows=args.chunk_rows)
        rows.extend(layer_rows)
        partial_path.write_text(json.dumps(rows, indent=2), encoding='utf-8')
        print(f'[eval] layer={layer_idx:02d} ({position}/{len(layer_indices)})', flush=True)
    rows = sorted(rows, key=lambda row: (int(row['layer']), PROJECTIONS.index(row['projection']), METHODS.index(row['method'])))
    aggregate = aggregate_rows(rows)
    rows_csv = args.output_dir / 'projection_relative_l2_by_layer.csv'
    aggregate_csv = args.output_dir / 'projection_relative_l2_aggregate.csv'
    write_csv(rows_csv, rows)
    write_csv(aggregate_csv, aggregate)
    l2_plot = plot_relative_l2(rows, args.output_dir, args)
    sparsity_plot = plot_actual_sparsity(rows, args.output_dir, args)
    summary = {'status': 'complete', 'args': {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}, 'eval_meta': eval_meta, 'capture_meta': capture_meta, 'metric': 'sqrt(sum ||y_method-y_dense||^2 / sum ||y_dense||^2), pooled over sampled tokens', 'scope': {'q_k_v_o_gate': 'local projection replay on the identical dense projection input', 'up': 'effective output consumed by the MLP product; Ours masks dense Up output using centered sparse-gate support', 'down': "method-faithful MLP replay, including each method's Gate/Up approximation before Down", 'transition_and_k4': 'excluded because they are residual-stream operations, not projection outputs'}, 'methods': {'TEAL': 'mixed-domain calibrated absolute thresholds at uniform target sparsity', 'LaRoSA': 'mixed-domain activation-only PCA Q plus exact global Top-K at uniform sparsity', 'Ours': 'mixed output-aware Q/D, shared-QKV random-permuted B128, O B128, and lazy-shift Up/Down'}, 'aggregate': aggregate, 'outputs': [str(rows_csv), str(aggregate_csv), str(l2_plot), str(sparsity_plot)], 'evaluation_elapsed_sec': float(time.perf_counter() - started)}
    (args.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(aggregate, indent=2), flush=True)
    print(f'[done] {l2_plot}', flush=True)
if __name__ == '__main__':
    main()
