"""Layerwise TPE search for the six-component Roshift body rotation.

Each decoder layer independently searches the simplex mixture of

    S_att, S_ffn, M_q, M_k, M_v, M_gate

at fixed 50% random-permuted B128 sparsity.  A candidate is evaluated from the
same dense block input and passes through sparse attention and sparse FFN before
the final decoder-block output error is measured.  The final 32 Q matrices are
then optionally packed and checked with WikiText perplexity.
"""
from __future__ import annotations
import os as _roshift_os
import argparse
import csv
import gc
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any
import matplotlib
import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F
from rental_validation import rms_epsilon
from tqdm import tqdm
matplotlib.use('Agg')
from matplotlib import pyplot as plt
sys.path.append(str(Path(__file__).resolve().parent))
import run_llama3_fair_calib_ppl_20260605 as fair
import run_llama3_mixed_allocation_search_20260815 as allocation
import run_llama3_output_aware_q_ppl as outq
from run_llama2_ffn_gate_output_threshold_ppl import decoder_layers, input_device, load_model, read_evalset
from run_llama3_larosa_absorbed_lazy_ppl import apply_common_pack, build_absorbed_pack
from run_llama3_projection_relative_l2_audit_20260827 import blockwise_topk_mask, fixed_permutation
from run_llama2_larosa_lazy_topk_ppl import RmsAlphaBucketController, q_from_cov
COMPONENTS = ('S_att', 'S_ffn', 'M_q', 'M_k', 'M_v', 'M_gate')
LOGIT_COMPONENTS = COMPONENTS[:-1]
ROOT = Path('/repro/data/llama3_1_8b/mixed_calib_c4_alpaca_metamath_code_20260812')
DEFAULT_OUTPUT = Path('/repro/data/llama3_1_8b/layerwise_q_tpe_block_l2_20260827')
DEFAULT_METRIC_DIR = Path('/repro/data/llama3_1_8b/fair_calib_w103_4096x32_ppl_20260605/output_aware_metric_components_no_up_20260704')
SEED_RECIPES: dict[str, list[float]] = {'uniform_no_up': [0.25, 0.25, 0.125, 0.125, 0.125, 0.125], 'activation_heavy': [0.35, 0.35, 0.075, 0.075, 0.075, 0.075], 'weight_heavy': [0.15, 0.15, 0.175, 0.175, 0.175, 0.175], 'attention_heavy': [0.45, 0.2, 0.1, 0.08, 0.08, 0.09], 'ffn_heavy': [0.2, 0.45, 0.07, 0.07, 0.07, 0.14], 'qkv_heavy': [0.15, 0.15, 0.18, 0.18, 0.18, 0.16], 'gate_heavy': [0.18, 0.25, 0.07, 0.07, 0.07, 0.36], 'activation_only_soft': [0.49, 0.49, 0.005, 0.005, 0.005, 0.005], 'metrics_only_soft': [0.005, 0.005, 0.2475, 0.2475, 0.2475, 0.2475]}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model_id', default='/repro/models/Meta-Llama-3.1-8B')
    parser.add_argument('--device_map', default='cuda:0')
    parser.add_argument('--attn_implementation', default='sdpa')
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--metric_dir', type=Path, default=DEFAULT_METRIC_DIR)
    parser.add_argument('--output_dir', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--evalset_dir', type=Path, default=Path(_roshift_os.path.join(_roshift_os.environ['ROSHIFT_ENGINE_ROOT'], 'eval_data')))
    parser.add_argument('--trials', type=int, default=48)
    parser.add_argument('--sparsity', type=float, default=50.0)
    parser.add_argument('--block_size', type=int, default=128)
    parser.add_argument('--permutation_seed', type=int, default=11)
    parser.add_argument('--context_length', type=int, default=512)
    parser.add_argument('--cov_rows_per_sequence', type=int, default=512)
    parser.add_argument('--cov_sequences', type=int, default=0, help='0 uses all 32 mixed sequences; positive values are smoke-test only')
    parser.add_argument('--fit_rows', default='0,16,22,27')
    parser.add_argument('--eval_rows', default='1,17,23,28')
    parser.add_argument('--layers', default='')
    parser.add_argument('--seed', type=int, default=20260827)
    parser.add_argument('--force_components', action='store_true')
    parser.add_argument('--force_captures', action='store_true')
    parser.add_argument('--force_search', action='store_true')
    parser.add_argument('--search_only', action='store_true')
    parser.add_argument('--prepare_only', action='store_true', help='build split covariance components and block captures, then stop before Q search')
    parser.add_argument('--skip_baseline_ppl', action='store_true')
    parser.add_argument('--final_samples', type=int, default=4)
    parser.add_argument('--median_chunk', type=int, default=512)
    parser.add_argument('--median_hist_bins', type=int, default=2048)
    parser.add_argument('--fallback_rel', type=float, default=0.05)
    return parser.parse_args()

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def parse_ints(raw: str) -> list[int]:
    return [int(value.strip()) for value in str(raw).split(',') if value.strip()]

def parse_layers(raw: str, total: int) -> list[int]:
    if not str(raw).strip():
        return list(range(int(total)))
    values = sorted(set(parse_ints(raw)))
    if not values or values[0] < 0 or values[-1] >= int(total):
        raise ValueError(f'invalid layers {values} for total={total}')
    return values

def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)

def inv_rms(x: torch.Tensor, eps: float) -> torch.Tensor:
    return torch.rsqrt(x.float().square().mean(dim=-1) + float(eps))

def component_path(component_dir: Path, component: str, layer_idx: int) -> Path:
    return Path(component_dir) / f'{component}_layer_{int(layer_idx):02d}.pt'

def metric_path(metric_dir: Path, component: str, layer_idx: int) -> Path:
    return Path(metric_dir) / f'{component}_layer_{int(layer_idx):02d}.pt'

def q_path(q_dir: Path, layer_idx: int) -> Path:
    return Path(q_dir) / f'q_layer_{int(layer_idx):02d}.pt'

def capture_path(capture_dir: Path, layer_idx: int) -> Path:
    return Path(capture_dir) / f'layer_{int(layer_idx):02d}.pt'

def normalized_alpha(values: list[float] | np.ndarray) -> np.ndarray:
    alpha = np.maximum(np.asarray(values, dtype=np.float64), 1e-12)
    return alpha / float(alpha.sum())

def alpha_to_params(alpha: list[float] | np.ndarray) -> dict[str, float]:
    values = normalized_alpha(alpha)
    reference = max(float(values[-1]), 1e-12)
    return {f'logit_{component}': float(np.clip(math.log(max(float(values[index]), 1e-12) / reference), -6.0, 6.0)) for index, component in enumerate(LOGIT_COMPONENTS)}

def params_to_alpha(params: dict[str, float]) -> np.ndarray:
    logits = np.asarray([float(params[f'logit_{component}']) for component in LOGIT_COMPONENTS] + [0.0], dtype=np.float64)
    logits -= float(logits.max())
    values = np.exp(logits)
    return values / float(values.sum())

def trial_alpha(trial: optuna.Trial) -> np.ndarray:
    params = {f'logit_{component}': trial.suggest_float(f'logit_{component}', -6.0, 6.0) for component in LOGIT_COMPONENTS}
    return params_to_alpha(params)

@torch.inference_mode()
def collect_split_components(*, args: argparse.Namespace, model: nn.Module, calibration: torch.Tensor, layers: list[nn.Module], layer_indices: list[int], device: torch.device) -> None:
    component_dir = args.output_dir / 'components'
    meta_path = component_dir / 'component_meta.json'
    expected = [component_path(component_dir, stream, layer_idx) for stream in ('S_att', 'S_ffn') for layer_idx in layer_indices]
    if not args.force_components and meta_path.exists() and all((path.exists() for path in expected)):
        print(f'[components] reuse {component_dir}', flush=True)
        return
    component_dir.mkdir(parents=True, exist_ok=True)
    hidden = int(model.config.hidden_size)
    sample_count = min(int(args.cov_rows_per_sequence), int(calibration.shape[1]))
    positions = torch.linspace(0, int(calibration.shape[1]) - 1, steps=sample_count, device=device).round().long()
    stream_specs = {'S_att': ('input_layernorm',), 'S_ffn': ('post_attention_layernorm',)}
    counts: dict[str, dict[str, int]] = {}
    started = time.perf_counter()
    covariance_sequences = int(calibration.shape[0]) if int(args.cov_sequences) <= 0 else min(int(args.cov_sequences), int(calibration.shape[0]))
    for stream, (norm_name,) in stream_specs.items():
        covariances = {layer_idx: torch.zeros((hidden, hidden), device=device, dtype=torch.float32) for layer_idx in layer_indices}
        stream_counts = {layer_idx: 0 for layer_idx in layer_indices}
        handles = []

        def make_hook(layer_idx: int, eps: float):

            def hook(_module: nn.Module, inputs: tuple[torch.Tensor, ...]):
                values = inputs[0].detach()[0].index_select(0, positions)
                values = values.float()
                values.mul_(inv_rms(values, eps).unsqueeze(-1))
                values.sub_(values.mean(dim=0, keepdim=True))
                covariances[layer_idx].addmm_(values.transpose(0, 1), values)
                stream_counts[layer_idx] += int(values.shape[0])
            return hook
        for layer_idx in layer_indices:
            layer = layers[layer_idx]
            norm = getattr(layer, norm_name)
            handles.append(norm.register_forward_pre_hook(make_hook(layer_idx, rms_epsilon(norm))))
        try:
            for row_idx in tqdm(range(covariance_sequences), desc=f'mixed split covariance {stream}', unit='sequence'):
                ids = calibration[row_idx:row_idx + 1].to(device)
                mask = torch.ones_like(ids)
                model.model(input_ids=ids, attention_mask=mask, use_cache=False)
                del ids, mask
        finally:
            for handle in handles:
                handle.remove()
        for layer_idx in layer_indices:
            covariance = covariances[layer_idx]
            covariance.div_(float(max(stream_counts[layer_idx] - 1, 1)))
            covariance.copy_(0.5 * (covariance + covariance.transpose(0, 1)))
            covariance.div_(torch.trace(covariance).abs().clamp_min(1e-12))
            torch.save(covariance.cpu(), component_path(component_dir, stream, layer_idx))
            covariances[layer_idx] = None
        counts[stream] = {str(layer_idx): int(stream_counts[layer_idx]) for layer_idx in layer_indices}
        del covariances
        gc.collect()
        torch.cuda.empty_cache()
    atomic_json(meta_path, {'definition': 'Separate trace-normalized covariance matrices of no-gamma RMS normalized attention and FFN residual inputs. Each of the 32 mixed-domain calibration sequences contributes deterministic rows.', 'components': list(COMPONENTS), 'calibration_source': str(args.root / 'mixed_calibration' / 'input_ids.pt'), 'calibration_sequences': int(covariance_sequences), 'sequence_length': int(calibration.shape[1]), 'rows_per_sequence': int(sample_count), 'tokens_per_stream': int(covariance_sequences) * int(sample_count), 'counts': counts, 'metric_components': str(args.metric_dir), 'elapsed_sec': float(time.perf_counter() - started)})

def detach_cpu(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().to(device='cpu').contiguous()
    if isinstance(value, tuple):
        return tuple((detach_cpu(item) for item in value))
    if isinstance(value, list):
        return [detach_cpu(item) for item in value]
    if isinstance(value, dict):
        return {key: detach_cpu(item) for key, item in value.items()}
    return value

def selected_attention_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    allowed = {'attention_mask', 'position_ids', 'cache_position', 'position_embeddings'}
    return {key: detach_cpu(value) for key, value in kwargs.items() if key in allowed and value is not None}

@torch.inference_mode()
def collect_block_captures(*, args: argparse.Namespace, model: nn.Module, calibration: torch.Tensor, selection: torch.Tensor, layers: list[nn.Module], layer_indices: list[int], device: torch.device) -> None:
    capture_dir = args.output_dir / 'block_captures'
    meta_path = capture_dir / 'capture_meta.json'
    kwargs_path = capture_dir / 'attention_kwargs.pt'
    expected = [capture_path(capture_dir, i) for i in layer_indices]
    if not args.force_captures and meta_path.exists() and kwargs_path.exists() and all((path.exists() for path in expected)):
        print(f'[captures] reuse {capture_dir}', flush=True)
        return
    fit_rows = parse_ints(args.fit_rows)
    eval_rows = parse_ints(args.eval_rows)
    if max(fit_rows) >= int(calibration.shape[0]) or max(eval_rows) >= int(selection.shape[0]):
        raise ValueError('capture row index exceeds mixed-domain tensors')
    length = int(args.context_length)
    if length > int(calibration.shape[1]) or length > int(selection.shape[1]):
        raise ValueError('context length exceeds source sequence length')
    capture_dir.mkdir(parents=True, exist_ok=True)
    stores: dict[int, dict[str, list[torch.Tensor]]] = {index: {'fit_input': [], 'eval_input': [], 'eval_output': []} for index in layer_indices}
    current_split = ''
    common_kwargs: dict[str, Any] | None = None
    handles = []

    def make_pre_hook(layer_idx: int):

        def hook(_module: nn.Module, call_args, call_kwargs):
            nonlocal common_kwargs
            hidden = call_args[0].detach().to(device='cpu', dtype=torch.float16).squeeze(0).contiguous()
            stores[layer_idx][f'{current_split}_input'].append(hidden)
            if common_kwargs is None:
                common_kwargs = selected_attention_kwargs(call_kwargs)
        return hook

    def make_post_hook(layer_idx: int):

        def hook(_module: nn.Module, _args, _kwargs, output):
            if current_split != 'eval':
                return
            value = output[0] if isinstance(output, tuple) else output
            stores[layer_idx]['eval_output'].append(value.detach().to(device='cpu', dtype=torch.float16).squeeze(0).contiguous())
        return hook
    for layer_idx in layer_indices:
        layer = layers[layer_idx]
        handles.append(layer.register_forward_pre_hook(make_pre_hook(layer_idx), with_kwargs=True))
        handles.append(layer.register_forward_hook(make_post_hook(layer_idx), with_kwargs=True))
    started = time.perf_counter()
    try:
        for split, source, rows in (('fit', calibration, fit_rows), ('eval', selection, eval_rows)):
            current_split = split
            for position, row_idx in enumerate(rows, start=1):
                ids = source[row_idx:row_idx + 1, :length].to(device)
                mask = torch.ones_like(ids)
                model.model(input_ids=ids, attention_mask=mask, use_cache=False)
                print(f'[capture] {split} row={row_idx} ({position}/{len(rows)})', flush=True)
                del ids, mask
    finally:
        for handle in handles:
            handle.remove()
    if common_kwargs is None:
        raise RuntimeError('failed to capture attention kwargs')
    torch.save(common_kwargs, kwargs_path)
    for layer_idx in layer_indices:
        payload = {key: torch.stack(values, dim=0).contiguous() for key, values in stores[layer_idx].items()}
        torch.save(payload, capture_path(capture_dir, layer_idx))
        stores[layer_idx].clear()
    atomic_json(meta_path, {'definition': 'Dense decoder-layer inputs and outputs for disjoint mixed-domain fit/evaluation rows. Fit rows calibrate the candidate gate median; evaluation rows determine final block-output relative L2.', 'fit_source': str(args.root / 'mixed_calibration' / 'input_ids.pt'), 'eval_source': str(args.root / 'mixed_calibration' / 'selection_input_ids.pt'), 'fit_rows': fit_rows, 'eval_rows': eval_rows, 'context_length': length, 'elapsed_sec': float(time.perf_counter() - started)})

def to_device(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device=device)
    if isinstance(value, tuple):
        return tuple((to_device(item, device) for item in value))
    if isinstance(value, list):
        return [to_device(item, device) for item in value]
    if isinstance(value, dict):
        return {key: to_device(item, device) for key, item in value.items()}
    return value

def block_mask(x: torch.Tensor, scale: torch.Tensor | None, sparsity: float, block_size: int, permutation: torch.Tensor) -> torch.Tensor:
    shape = x.shape
    flat = x.reshape(-1, shape[-1])
    mask = blockwise_topk_mask(flat, scale, float(sparsity), int(block_size), permutation)
    return mask.reshape(shape)

class CandidateState:

    def __init__(self, *, layer_idx: int, layer: nn.Module, metrics: list[torch.Tensor], sparsity: float, block_size: int, permutation_seed: int, device: torch.device) -> None:
        self.layer_idx = int(layer_idx)
        self.layer = layer
        self.metrics = metrics
        self.sparsity = float(sparsity)
        self.block_size = int(block_size)
        self.device = device
        hidden = int(layer.self_attn.q_proj.in_features)
        intermediate = int(layer.mlp.down_proj.in_features)
        self.qkv_permutation = fixed_permutation(hidden, permutation_seed, f'layer{layer_idx}:qkv').to(device)
        self.o_permutation = fixed_permutation(hidden, permutation_seed, f'layer{layer_idx}:o').to(device)
        self.gate_permutation = fixed_permutation(hidden, permutation_seed, f'layer{layer_idx}:gate').to(device)
        self.h1_permutation = fixed_permutation(intermediate, permutation_seed, f'layer{layer_idx}:h1').to(device)
        self.gamma_att = layer.input_layernorm.weight.detach()
        self.gamma_ffn = layer.post_attention_layernorm.weight.detach()
        self.q: torch.Tensor | None = None
        self.q_float: torch.Tensor | None = None
        self.qkv_scale: torch.Tensor | None = None
        self.gate_scale: torch.Tensor | None = None
        self.qkv_cache: tuple[int, torch.Tensor] | None = None

    def _d_from_metric(self, metric: torch.Tensor) -> torch.Tensor:
        if self.q_float is None:
            raise RuntimeError('candidate Q is unset')
        projected = torch.matmul(metric, self.q_float)
        diagonal = (self.q_float * projected).sum(dim=0).clamp_min_(1e-20)
        value = diagonal.sqrt_()
        return value / value.mean().clamp_min(1e-12)

    def set_candidate(self, q: torch.Tensor) -> None:
        self.q_float = q.to(device=self.device, dtype=torch.float32).contiguous()
        self.q = self.q_float.to(dtype=self.gamma_att.dtype).contiguous()
        dq = self._d_from_metric(self.metrics[2])
        dk = self._d_from_metric(self.metrics[3])
        dv = self._d_from_metric(self.metrics[4])
        self.qkv_scale = torch.sqrt((dq.square() + dk.square() + dv.square()).clamp_min_(1e-20))
        self.gate_scale = self._d_from_metric(self.metrics[5])
        self.qkv_cache = None

    def reset_attention(self) -> None:
        self.qkv_cache = None

    def reconstruct(self, x: torch.Tensor, gamma: torch.Tensor, scale: torch.Tensor | None, permutation: torch.Tensor) -> torch.Tensor:
        if self.q is None:
            raise RuntimeError('candidate Q is unset')
        rotated = torch.matmul(x, self.q)
        mask = block_mask(rotated, scale, self.sparsity, self.block_size, permutation)
        sparse = torch.where(mask, rotated, torch.zeros_like(rotated))
        restored = torch.matmul(sparse, self.q.transpose(0, 1))
        return restored * gamma.view(*[1] * (x.dim() - 1), -1)

    def qkv_input(self, x: torch.Tensor) -> torch.Tensor:
        key = int(x.data_ptr())
        if self.qkv_cache is None or self.qkv_cache[0] != key:
            if self.qkv_scale is None:
                raise RuntimeError('candidate QKV scale is unset')
            value = self.reconstruct(x, self.gamma_att, self.qkv_scale, self.qkv_permutation)
            self.qkv_cache = (key, value)
        return self.qkv_cache[1]

    def gate_input(self, x: torch.Tensor) -> torch.Tensor:
        if self.gate_scale is None:
            raise RuntimeError('candidate gate scale is unset')
        return self.reconstruct(x, self.gamma_ffn, self.gate_scale, self.gate_permutation)

    def sparse_o(self, x: torch.Tensor) -> torch.Tensor:
        mask = block_mask(x, None, self.sparsity, self.block_size, self.o_permutation)
        return torch.where(mask, x, torch.zeros_like(x))

    def sparse_h1(self, h1: torch.Tensor, median: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        median_view = median.view(*[1] * (h1.dim() - 1), -1)
        centered = h1 - median_view
        mask = block_mask(centered, None, self.sparsity, self.block_size, self.h1_permutation)
        centered_sparse = torch.where(mask, centered, torch.zeros_like(centered))
        return (centered_sparse, mask)

class CandidateQKVLinear(nn.Module):

    def __init__(self, inner: nn.Module, state: CandidateState) -> None:
        super().__init__()
        self.inner = inner
        self.state = state

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.inner(self.state.qkv_input(x))

class CandidateOLinear(nn.Module):

    def __init__(self, inner: nn.Module, state: CandidateState) -> None:
        super().__init__()
        self.inner = inner
        self.state = state

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.inner(self.state.sparse_o(x))

class CandidateLayer(nn.Module):

    def __init__(self, inner: nn.Module, state: CandidateState) -> None:
        super().__init__()
        self.inner = inner
        self.state = state
        self._original_qkv = {projection: getattr(inner.self_attn, projection) for projection in ('q_proj', 'k_proj', 'v_proj')}
        self._original_o = inner.self_attn.o_proj
        for projection in ('q_proj', 'k_proj', 'v_proj'):
            module = getattr(inner.self_attn, projection)
            setattr(inner.self_attn, projection, CandidateQKVLinear(module, state))
        inner.self_attn.o_proj = CandidateOLinear(inner.self_attn.o_proj, state)

    def restore(self) -> None:
        for projection, module in self._original_qkv.items():
            setattr(self.inner.self_attn, projection, module)
        self.inner.self_attn.o_proj = self._original_o

    def attention_and_h1(self, hidden_states: torch.Tensor, attention_kwargs: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        alpha_att = inv_rms(hidden_states, rms_epsilon(self.inner.input_layernorm)).to(hidden_states.dtype)
        no_gamma_att = hidden_states * alpha_att.unsqueeze(-1)
        self.state.reset_attention()
        attention_output, _ = self.inner.self_attn(hidden_states=no_gamma_att, use_cache=False, **attention_kwargs)
        h_mid = hidden_states + attention_output
        alpha_ffn = inv_rms(h_mid, rms_epsilon(self.inner.post_attention_layernorm)).to(h_mid.dtype)
        no_gamma_ffn = h_mid * alpha_ffn.unsqueeze(-1)
        gate = self.inner.mlp.gate_proj(self.state.gate_input(no_gamma_ffn))
        up = self.inner.mlp.up_proj(no_gamma_ffn * self.state.gamma_ffn.view(*[1] * (no_gamma_ffn.dim() - 1), -1))
        h1 = self.inner.mlp.act_fn(gate)
        return (h_mid, h1, up)

    def candidate_median(self, fit_hidden: torch.Tensor, attention_kwargs: dict[str, Any]) -> torch.Tensor:
        _, h1, _ = self.attention_and_h1(fit_hidden, attention_kwargs)
        return h1.reshape(-1, h1.shape[-1]).float().median(dim=0).values.to(h1.dtype)

    def forward(self, hidden_states: torch.Tensor, attention_kwargs: dict[str, Any], median: torch.Tensor) -> torch.Tensor:
        h_mid, h1, up = self.attention_and_h1(hidden_states, attention_kwargs)
        centered_sparse, mask = self.state.sparse_h1(h1, median)
        up_sparse = torch.where(mask, up, torch.zeros_like(up))
        median_view = median.view(*[1] * (up.dim() - 1), -1)
        down_input = centered_sparse * up_sparse + median_view * up
        return h_mid + self.inner.mlp.down_proj(down_input)

def load_layer_components(args: argparse.Namespace, layer_idx: int, device: torch.device) -> list[torch.Tensor]:
    component_dir = args.output_dir / 'components'
    values = []
    for component in COMPONENTS:
        path = component_path(component_dir, component, layer_idx) if component.startswith('S_') else metric_path(args.metric_dir, component, layer_idx)
        values.append(torch.load(path, map_location=device, weights_only=True).float().contiguous())
    return values

def build_q(alpha: np.ndarray, components: list[torch.Tensor]) -> torch.Tensor:
    covariance = torch.zeros_like(components[0])
    for coefficient, component in zip(alpha, components):
        covariance.add_(component, alpha=float(coefficient))
    covariance.copy_(0.5 * (covariance + covariance.transpose(0, 1)))
    q = q_from_cov(covariance, 0).contiguous()
    del covariance
    return q

def load_layer_capture(capture_dir: Path, layer_idx: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    payload = torch.load(capture_path(capture_dir, layer_idx), map_location='cpu', weights_only=True)
    return (payload['fit_input'].to(device=device), payload['eval_input'].to(device=device), payload['eval_output'].to(device=device))

def score_candidate(*, candidate: CandidateLayer, fit_hidden: torch.Tensor, eval_hidden: torch.Tensor, dense_output: torch.Tensor, attention_kwargs: dict[str, Any]) -> tuple[float, dict[str, float]]:
    median = candidate.candidate_median(fit_hidden, attention_kwargs)
    output = candidate(eval_hidden, attention_kwargs, median)
    difference = output.float() - dense_output.float()
    dense_update = dense_output.float() - eval_hidden.float()
    error_sq = float(difference.square().sum().item())
    output_sq = float(dense_output.float().square().sum().item())
    update_sq = float(dense_update.square().sum().item())
    objective = math.sqrt(error_sq / max(update_sq, 1e-30))
    metrics = {'objective_block_update_rel_l2': objective, 'output_relative_l2': math.sqrt(error_sq / max(output_sq, 1e-30)), 'error_sq': error_sq, 'output_sq': output_sq, 'dense_update_sq': update_sq}
    del median, output, difference, dense_update
    return (objective, metrics)

def storage_url(path: Path) -> str:
    return f'sqlite:///{path.resolve()}'

def search_layer(*, args: argparse.Namespace, model_layer: nn.Module, layer_idx: int, attention_kwargs: dict[str, Any], device: torch.device) -> dict[str, Any]:
    components = load_layer_components(args, layer_idx, device)
    fit_hidden, eval_hidden, dense_output = load_layer_capture(args.output_dir / 'block_captures', layer_idx, device)
    state = CandidateState(layer_idx=layer_idx, layer=model_layer, metrics=components, sparsity=args.sparsity, block_size=args.block_size, permutation_seed=args.permutation_seed, device=device)
    candidate = CandidateLayer(model_layer, state).to(device)
    database = args.output_dir / 'optuna.db'
    sampler = optuna.samplers.TPESampler(seed=int(args.seed) + 1009 * int(layer_idx), n_startup_trials=12, multivariate=True)
    study = optuna.create_study(study_name=f'layer_{layer_idx:02d}', direction='minimize', sampler=sampler, storage=storage_url(database), load_if_exists=True)
    if args.force_search and study.trials:
        raise RuntimeError('--force_search cannot erase an existing SQLite study; choose a new output_dir')
    if not study.trials:
        for recipe, alpha in SEED_RECIPES.items():
            study.enqueue_trial(alpha_to_params(alpha), user_attrs={'seed_recipe': recipe})
    trial_log = args.output_dir / 'layer_trials' / f'layer_{layer_idx:02d}.jsonl'
    trial_log.parent.mkdir(parents=True, exist_ok=True)

    def objective(trial: optuna.Trial) -> float:
        started = time.perf_counter()
        alpha = trial_alpha(trial)
        q = build_q(alpha, components)
        state.set_candidate(q)
        value, metrics = score_candidate(candidate=candidate, fit_hidden=fit_hidden, eval_hidden=eval_hidden, dense_output=dense_output, attention_kwargs=attention_kwargs)
        trial.set_user_attr('alpha', alpha.tolist())
        for key, metric in metrics.items():
            trial.set_user_attr(key, float(metric))
        trial.set_user_attr('elapsed_sec', float(time.perf_counter() - started))
        with trial_log.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps({'trial': int(trial.number), 'alpha': alpha.tolist(), 'value': float(value), **metrics, 'elapsed_sec': float(time.perf_counter() - started)}) + '\n')
        del q
        return float(value)
    completed = sum((trial.state == optuna.trial.TrialState.COMPLETE for trial in study.trials))
    remaining = max(0, int(args.trials) - int(completed))
    if remaining:
        study.optimize(objective, n_trials=remaining, gc_after_trial=True, show_progress_bar=False)
    best = study.best_trial
    best_alpha = np.asarray(best.user_attrs['alpha'], dtype=np.float64)
    best_q = build_q(best_alpha, components)
    selected_dir = args.output_dir / 'selected_q'
    selected_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_q.to(device='cpu', dtype=torch.float16).contiguous(), q_path(selected_dir, layer_idx))
    seed_values = {str(trial.user_attrs.get('seed_recipe')): float(trial.value) for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE and trial.user_attrs.get('seed_recipe') is not None}
    if 'uniform_no_up' not in seed_values:
        baseline_q = build_q(np.asarray(SEED_RECIPES['uniform_no_up'], dtype=np.float64), components)
        state.set_candidate(baseline_q)
        baseline_value, _ = score_candidate(candidate=candidate, fit_hidden=fit_hidden, eval_hidden=eval_hidden, dense_output=dense_output, attention_kwargs=attention_kwargs)
        seed_values['uniform_no_up'] = float(baseline_value)
        del baseline_q
    baseline_value = seed_values.get('uniform_no_up', float('nan'))
    row = {'layer': int(layer_idx), 'trials': int(sum((trial.state == optuna.trial.TrialState.COMPLETE for trial in study.trials))), 'best_trial': int(best.number), 'best_objective': float(best.value), 'uniform_no_up_objective': float(baseline_value), 'improvement_percent': float(100.0 * (baseline_value - float(best.value)) / baseline_value) if math.isfinite(baseline_value) and baseline_value > 0.0 else 0.0, 'output_relative_l2': float(best.user_attrs['output_relative_l2']), **{component: float(best_alpha[index]) for index, component in enumerate(COMPONENTS)}, 'seed_values': seed_values}
    atomic_json(args.output_dir / 'layer_best' / f'layer_{layer_idx:02d}.json', row)
    candidate.restore()
    del candidate, state, components, fit_hidden, eval_hidden, dense_output, best_q
    gc.collect()
    torch.cuda.empty_cache()
    return row

def write_search_outputs(args: argparse.Namespace, rows: list[dict[str, Any]]) -> None:
    rows = sorted(rows, key=lambda row: int(row['layer']))
    csv_path = args.output_dir / 'layerwise_best_alpha.csv'
    fields = ['layer', 'trials', 'best_trial', 'best_objective', 'uniform_no_up_objective', 'improvement_percent', 'output_relative_l2', *COMPONENTS]
    with csv_path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    fig, axis = plt.subplots(figsize=(12.5, 5.6), dpi=180)
    layers = [int(row['layer']) for row in rows]
    colors = ('#2e86ab', '#f18f01', '#6a994e', '#9b5de5', '#e76f51', '#3a7d44')
    for component, color in zip(COMPONENTS, colors):
        axis.plot(layers, [100.0 * float(row[component]) for row in rows], marker='o', markersize=3.0, linewidth=1.4, label=component, color=color)
    axis.set_xlabel('Layer')
    axis.set_ylabel('Mixture weight (%)')
    axis.set_xticks(layers)
    axis.set_ylim(0.0, 100.0)
    axis.grid(True, alpha=0.25)
    axis.legend(ncol=3, fontsize=9)
    axis.set_title('Layerwise TPE-optimized Roshift Q mixture')
    fig.tight_layout()
    fig.savefig(args.output_dir / 'layerwise_best_alpha.png')
    plt.close(fig)
    fig, axis = plt.subplots(figsize=(12.5, 4.8), dpi=180)
    baseline = [float(row['uniform_no_up_objective']) for row in rows]
    best = [float(row['best_objective']) for row in rows]
    axis.plot(layers, baseline, marker='o', linewidth=1.4, label='uniform no-up')
    axis.plot(layers, best, marker='o', linewidth=1.4, label='TPE best')
    axis.set_xlabel('Layer')
    axis.set_ylabel('Block-update relative L2')
    axis.set_xticks(layers)
    axis.grid(True, alpha=0.25)
    axis.legend()
    axis.set_title('Dense-block output error at 50% B128 sparsity')
    fig.tight_layout()
    fig.savefig(args.output_dir / 'layerwise_block_l2_comparison.png')
    plt.close(fig)

def prepare_selected_artifacts(args: argparse.Namespace, model: nn.Module, device: torch.device) -> tuple[Path, Path, Path]:
    q_dir = args.output_dir / 'selected_q'
    d_dir = args.output_dir / 'selected_d'
    median_dir = args.output_dir / 'selected_medians'
    pack_dir = args.output_dir / 'selected_pack'
    layers = list(decoder_layers(model))
    outq.build_d_cache(model, q_dir, d_dir, force=False)
    fair.build_daware_medians_stream_hist(model=model, q_dir=q_dir, d_dir=d_dir, hmid_dir=args.root / 'hmid_cache', median_dir=median_dir, k_list=[int(round(args.sparsity))], num_layers=len(layers), hidden_size=int(model.config.hidden_size), intermediate_size=int(model.config.intermediate_size), chunk=int(args.median_chunk), hist_bins=int(args.median_hist_bins), device=device, force=False)
    controller = RmsAlphaBucketController(npz_path=args.root / 'rms_inv_rms_selection.npz', num_layers=len(layers), bucket_k=4, fallback_rel=float(args.fallback_rel), bucket_mode='fixed', bucket_avg_k=4.0)
    build_absorbed_pack(model=model, pack_dir=pack_dir, q_dir=q_dir, median_dir=median_dir, alpha_controller=controller, k_list=[])
    return (d_dir, median_dir, pack_dir)

def evaluate_artifact(*, args: argparse.Namespace, q_label: str, d_dir: Path, median_dir: Path, pack_dir: Path, include_dense: bool) -> dict[str, Any]:
    model = load_model(args)
    device = input_device(model)
    eval_ids, _ = read_evalset(args.evalset_dir)
    eval_ids = np.asarray(eval_ids[:int(args.final_samples)])
    result: dict[str, Any] = {'method': q_label}
    if include_dense:
        result['dense'] = allocation.evaluate_ppl(model, eval_ids, device)
    apply_common_pack(model, pack_dir)
    num_layers = len(list(decoder_layers(model)))
    controller = allocation.RecipeController(num_layers)
    controller.set_recipe(np.full((num_layers, len(allocation.SITES)), float(args.sparsity), dtype=np.float64))
    adaptive = allocation.AdaptiveK4(args.root / 'adaptive_dp_mixed_20260814' / 'adaptive_dp_centers.json', float(args.fallback_rel))
    medians = allocation.MedianBank(median_dir, int(model.config.intermediate_size))
    allocation.patch_model(model, controller, d_dir, pack_dir, medians, adaptive, int(args.block_size), int(args.permutation_seed))
    result['sparse'] = allocation.evaluate_ppl(model, eval_ids, device)
    result['adaptive_k4'] = adaptive.stats()
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result

def final_ppl(args: argparse.Namespace, search_rows: list[dict[str, Any]]) -> dict[str, Any]:
    print('[final] preparing selected Q/D/median/pack', flush=True)
    model = load_model(args)
    device = input_device(model)
    selected_d, selected_medians, selected_pack = prepare_selected_artifacts(args, model, device)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    rows = []
    if not args.skip_baseline_ppl:
        baseline = evaluate_artifact(args=args, q_label='current_mixed_Q', d_dir=args.root / 'output_aware_q' / 'd_cache' / 's_plus_lam_0p2', median_dir=args.root / 'ours_daware_sparse_gate_medians', pack_dir=args.root / 'ours_absorbed_pack', include_dense=True)
        rows.append(baseline)
    selected = evaluate_artifact(args=args, q_label='layerwise_TPE_Q', d_dir=selected_d, median_dir=selected_medians, pack_dir=selected_pack, include_dense=not rows)
    rows.append(selected)
    dense_ppl = float(next((row['dense']['ppl'] for row in rows if 'dense' in row)))
    output_rows = []
    for row in rows:
        sparse_ppl = float(row['sparse']['ppl'])
        output_rows.append({'method': row['method'], 'dense_ppl': dense_ppl, 'ppl': sparse_ppl, 'delta_vs_dense': sparse_ppl - dense_ppl, 'relative_vs_dense_percent': 100.0 * (sparse_ppl / dense_ppl - 1.0), 'fallback_rate': float(row['adaptive_k4']['fallback_rate'])})
    payload = {'protocol': {'model': args.model_id, 'dataset': 'WikiText-103 test', 'sequences': int(args.final_samples), 'sequence_length': 4096, 'tokens': int(args.final_samples) * 4095, 'body_sparsity': float(args.sparsity), 'selector': f'random-permuted B{args.block_size}', 'lm_head': 'dense', 'k4': 'adaptive DP centers, 5% relative fallback'}, 'results': output_rows, 'local_search': {'mean_uniform_objective': float(np.mean([row['uniform_no_up_objective'] for row in search_rows])), 'mean_best_objective': float(np.mean([row['best_objective'] for row in search_rows])), 'mean_improvement_percent': float(np.mean([row['improvement_percent'] for row in search_rows]))}}
    atomic_json(args.output_dir / 'final_ppl_summary.json', payload)
    return payload

def main() -> None:
    args = parse_args()
    if float(args.sparsity) != 50.0:
        raise ValueError('this controlled study is fixed to 50% sparsity')
    set_seed(int(args.seed))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    calibration = torch.load(args.root / 'mixed_calibration' / 'input_ids.pt', map_location='cpu', weights_only=True)
    selection = torch.load(args.root / 'mixed_calibration' / 'selection_input_ids.pt', map_location='cpu', weights_only=True)
    model = load_model(args)
    device = input_device(model)
    layers = list(decoder_layers(model))
    layer_indices = parse_layers(args.layers, len(layers))
    collect_split_components(args=args, model=model, calibration=calibration, layers=layers, layer_indices=layer_indices, device=device)
    collect_block_captures(args=args, model=model, calibration=calibration, selection=selection, layers=layers, layer_indices=layer_indices, device=device)
    del calibration, selection
    if args.prepare_only:
        payload = {'status': 'prepare_complete', 'model': str(args.model_id), 'num_layers': len(layers), 'components': list(COMPONENTS), 'cov_rows_per_sequence': int(args.cov_rows_per_sequence), 'cov_sequences': int(args.cov_sequences), 'elapsed_sec': float(time.perf_counter() - started)}
        atomic_json(args.output_dir / 'prepare_summary.json', payload)
        print(json.dumps(payload, indent=2), flush=True)
        return
    attention_kwargs = to_device(torch.load(args.output_dir / 'block_captures' / 'attention_kwargs.pt', map_location='cpu', weights_only=True), device)
    rows = []
    for position, layer_idx in enumerate(layer_indices, start=1):
        best_path = args.output_dir / 'layer_best' / f'layer_{layer_idx:02d}.json'
        selected_path = q_path(args.output_dir / 'selected_q', layer_idx)
        if best_path.exists() and selected_path.exists() and (not args.force_search):
            with best_path.open('r', encoding='utf-8') as handle:
                resumed = json.load(handle)
            if int(resumed.get('trials', 0)) >= int(args.trials):
                rows.append(resumed)
                print(f"[resume] layer={layer_idx:02d} ({position}/{len(layer_indices)}) best={float(resumed['best_objective']):.6f}", flush=True)
                continue
        row = search_layer(args=args, model_layer=layers[layer_idx], layer_idx=layer_idx, attention_kwargs=attention_kwargs, device=device)
        rows.append(row)
        print(f"[TPE] layer={layer_idx:02d} ({position}/{len(layer_indices)}) best={row['best_objective']:.6f} improvement={row['improvement_percent']:.2f}%", flush=True)
    write_search_outputs(args, rows)
    complete = len(layer_indices) == len(layers)
    search_payload = {'status': 'search_complete' if complete else 'partial_layers', 'args': {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}, 'components': list(COMPONENTS), 'layers': rows, 'elapsed_sec': float(time.perf_counter() - started)}
    atomic_json(args.output_dir / 'search_summary.json', search_payload)
    if complete and (not args.search_only):
        del attention_kwargs, layers, model
        gc.collect()
        torch.cuda.empty_cache()
        payload = final_ppl(args, rows)
        print(json.dumps(payload, indent=2), flush=True)
    else:
        print(json.dumps(search_payload, indent=2), flush=True)
if __name__ == '__main__':
    main()
