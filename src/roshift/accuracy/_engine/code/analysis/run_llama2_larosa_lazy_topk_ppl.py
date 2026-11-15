import argparse
import csv
import json
import math
import os
import time
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from run_llama2_ffn_gate_output_threshold_ppl import (
    GateOutputThresholdController,
    build_calib_loader,
    decoder_layers,
    eval_ppl,
    input_device,
    load_model,
    read_evalset,
)

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


DEFAULT_ROOT = Path("/repro/data/llama2_7b")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "PPL sanity sweep for top-k TEAL, LaRoSA-style rotated input top-k, "
            "and ideal lazy-shift gate-output top-k on Llama-2 FFN."
        )
    )
    p.add_argument("--model_id", default="/mnt/c/AS_data/hf_local/NousResearch-Llama-2-7b-hf")
    p.add_argument("--evalset_dir", type=Path, default=DEFAULT_ROOT / "eval_data_llama2_7b")
    p.add_argument("--threshold_dir", type=Path, default=DEFAULT_ROOT / "ffn_gate_output_threshold_ppl_calib32768" / "calib_h1_32768")
    p.add_argument("--x_cache_dir", type=Path, default=DEFAULT_ROOT / "ffn_lazy_axrrr_stage1_threshold_adaptive_r1024_seq1024_128M_20260602" / "calib_x_corr_32768")
    p.add_argument("--output_dir", type=Path, default=DEFAULT_ROOT / "larosa_lazy_topk_ppl_20260604")
    p.add_argument("--k_list", default="40,50,60")
    p.add_argument("--tail_start", type=int, default=2048)
    p.add_argument("--device_map", default="cuda:0")
    p.add_argument("--attn_implementation", choices=["auto", "eager", "sdpa", "flash_attention_2"], default="sdpa")
    p.add_argument("--limit_eval_samples", type=int, default=None)
    p.add_argument("--q_tokens", type=int, default=32768)
    p.add_argument("--q_chunk", type=int, default=512)
    p.add_argument("--attn_q_tokens", type=int, default=4096)
    p.add_argument("--attn_calib_max_length", type=int, default=512)
    p.add_argument("--attn_calib_batch_size", type=int, default=1)
    p.add_argument(
        "--block_size",
        type=int,
        default=0,
        help="If >0 and < hidden_size, build block-diagonal PCA rotations with this hidden block size.",
    )
    p.add_argument("--force_rebuild_q", action="store_true")
    p.add_argument(
        "--methods",
        default="teal_topk,larosa_gate_up_topk,larosa_lazy_gate_up_topk,lazy_shift_ideal_topk",
        help=(
            "Comma-separated subset of teal_topk,larosa_up_topk,larosa_gate_up_topk,"
            "larosa_lazy_gate_up_topk,lazy_shift_ideal_topk,teal_all_topk,"
            "larosa_all_topk,larosa_lazy_ideal_all_topk,larosa_lazy_bucket_all_topk,"
            "larosa_lazy_bucket_fusedcorr_all_topk."
        ),
    )
    p.add_argument("--larosa_median_tokens", type=int, default=4096)
    p.add_argument(
        "--rms_stats_npz",
        type=Path,
        default=DEFAULT_ROOT / "rmsnorm_rms_gamma_stats_20260604" / "rms_inv_rms_raw_arrays.npz",
        help="NPZ containing inv_rms_post_layer_XX arrays for RMS bucket centers.",
    )
    p.add_argument("--rms_bucket_k", type=int, default=4)
    p.add_argument("--rms_bucket_mode", choices=["fixed", "adaptive_avg"], default="fixed")
    p.add_argument("--rms_bucket_avg_k", type=float, default=4.0)
    p.add_argument("--rms_fallback_rel", type=float, default=0.05)
    return p.parse_args()


def parse_k_list(raw: str) -> List[int]:
    return sorted({int(x.strip()) for x in raw.split(",") if x.strip()})


def parse_methods(raw: str) -> List[str]:
    allowed = {
        "teal_topk",
        "larosa_up_topk",
        "larosa_gate_up_topk",
        "larosa_lazy_gate_up_topk",
        "lazy_shift_ideal_topk",
        "teal_all_topk",
        "larosa_all_topk",
        "larosa_lazy_ideal_all_topk",
        "larosa_lazy_bucket_all_topk",
        "larosa_lazy_bucket_fusedcorr_all_topk",
    }
    out = [x.strip() for x in raw.split(",") if x.strip()]
    bad = [x for x in out if x not in allowed]
    if bad:
        raise ValueError(f"Unknown methods: {bad}; allowed={sorted(allowed)}")
    return out


def topk_abs(x: torch.Tensor, sparsity: int) -> torch.Tensor:
    s = int(sparsity)
    if s <= 0:
        return x
    dim = int(x.shape[-1])
    keep = int(round(dim * (100.0 - float(s)) / 100.0))
    if keep >= dim:
        return x
    if keep <= 0:
        return torch.zeros_like(x)
    idx = torch.topk(x.abs().float(), k=keep, dim=-1, largest=True).indices
    mask = torch.zeros_like(x, dtype=torch.bool)
    mask.scatter_(-1, idx, True)
    return torch.where(mask, x, torch.zeros_like(x))


def load_threshold_payload(threshold_dir: Path) -> dict:
    with (threshold_dir / "thresholds.json").open("r", encoding="utf-8") as f:
        return json.load(f)


def unwrap_mlp(mlp: nn.Module) -> nn.Module:
    while hasattr(mlp, "inner"):
        mlp = mlp.inner
    return mlp


def unwrap_module(module: nn.Module) -> nn.Module:
    while hasattr(module, "inner"):
        module = module.inner
    return module


def restore_dense(model: nn.Module) -> None:
    for layer in decoder_layers(model):
        if hasattr(layer, "self_attn"):
            attn = layer.self_attn
            for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                if hasattr(attn, name):
                    setattr(attn, name, unwrap_module(getattr(attn, name)))
        mlp = unwrap_mlp(layer.mlp)
        for name in ("gate_proj", "up_proj", "down_proj"):
            if hasattr(mlp, name):
                setattr(mlp, name, unwrap_module(getattr(mlp, name)))
        layer.mlp = mlp


class MedianController:
    def __init__(self, threshold_payload: dict, threshold_dir: Path, num_layers: int, intermediate_size: int) -> None:
        self.inner = GateOutputThresholdController(
            thresholds=threshold_payload,
            cache_dir=threshold_dir,
            num_layers=num_layers,
            inter=intermediate_size,
        )

    def median(self, layer_idx: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return self.inner.median(layer_idx, device, dtype)


class LarosaSparseGateMedianController:
    def __init__(self, median_dir: Path, sparsity: int, intermediate_size: int) -> None:
        self.median_dir = Path(median_dir)
        self.sparsity = int(sparsity)
        self.intermediate_size = int(intermediate_size)
        self._cpu: Dict[int, torch.Tensor] = {}
        self._dev: Dict[tuple, torch.Tensor] = {}

    def median(self, layer_idx: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        key = (int(layer_idx), str(device), str(dtype))
        cached = self._dev.get(key)
        if cached is not None:
            return cached
        cpu = self._cpu.get(int(layer_idx))
        if cpu is None:
            path = self.median_dir / f"s{self.sparsity:02d}" / f"h1_median_layer_{int(layer_idx):02d}.bin"
            arr = np.fromfile(path, dtype=np.float16)
            if arr.size != self.intermediate_size:
                raise ValueError(f"Median size mismatch at {path}: {arr.size} != {self.intermediate_size}")
            cpu = torch.from_numpy(arr).to(device="cpu", dtype=torch.float16).contiguous()
            self._cpu[int(layer_idx)] = cpu
        dev = cpu.to(device=device, dtype=dtype).contiguous()
        self._dev[key] = dev
        return dev


def kmeans1d_centers(values: np.ndarray, k: int, iters: int = 100) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("Cannot build RMS buckets from empty values.")
    k = int(k)
    if k <= 0:
        raise ValueError(f"Invalid bucket count: {k}")
    centers = np.percentile(values, np.linspace(0, 100, k + 2)[1:-1])
    for _ in range(int(iters)):
        labels = np.abs(values[:, None] - centers[None, :]).argmin(axis=1)
        new_centers = centers.copy()
        for idx in range(k):
            mask = labels == idx
            if mask.any():
                new_centers[idx] = values[mask].mean()
        if np.allclose(new_centers, centers, rtol=1e-7, atol=1e-9):
            break
        centers = new_centers
    return np.sort(centers).astype(np.float32)


def fallback_rate_for_centers(values: np.ndarray, centers: np.ndarray, fallback_rel: float) -> float:
    values = np.asarray(values, dtype=np.float64)
    centers = np.asarray(centers, dtype=np.float64)
    labels = np.abs(values[:, None] - centers[None, :]).argmin(axis=1)
    nearest = centers[labels]
    rel = np.abs(nearest - values) / np.maximum(np.abs(values), 1e-12)
    return float(np.mean(rel > float(fallback_rel)))


def adaptive_average_k_centers(
    values_by_layer: List[np.ndarray],
    avg_k: float,
    fallback_rel: float,
    candidates: List[int] | None = None,
) -> tuple[List[np.ndarray], dict]:
    candidates = candidates or [2, 3, 4, 6, 8]
    candidates = sorted({int(x) for x in candidates})
    if not candidates:
        raise ValueError("Empty K candidates.")
    num_layers = len(values_by_layer)
    target_budget = int(round(float(avg_k) * num_layers))
    min_budget = candidates[0] * num_layers
    max_budget = candidates[-1] * num_layers
    if target_budget < min_budget or target_budget > max_budget:
        raise ValueError(f"avg_k={avg_k} gives budget={target_budget}, outside [{min_budget}, {max_budget}]")

    center_cache: List[dict[int, np.ndarray]] = []
    fallback_cache: List[dict[int, float]] = []
    for values in values_by_layer:
        c_by_k = {}
        f_by_k = {}
        for k in candidates:
            centers = kmeans1d_centers(values, k)
            c_by_k[k] = centers
            f_by_k[k] = fallback_rate_for_centers(values, centers, fallback_rel)
        center_cache.append(c_by_k)
        fallback_cache.append(f_by_k)

    k_by_layer = [candidates[0] for _ in range(num_layers)]
    budget = sum(k_by_layer)
    while budget < target_budget:
        best = None
        for layer_idx, cur_k in enumerate(k_by_layer):
            pos = candidates.index(cur_k)
            if pos + 1 >= len(candidates):
                continue
            nxt = candidates[pos + 1]
            inc = nxt - cur_k
            if budget + inc > target_budget:
                continue
            gain = fallback_cache[layer_idx][cur_k] - fallback_cache[layer_idx][nxt]
            score = gain / float(inc)
            cand = (score, gain, -inc, layer_idx, nxt)
            if best is None or cand > best:
                best = cand
        if best is None:
            raise RuntimeError(f"Could not exactly allocate adaptive K budget {target_budget}; current={budget}")
        _, _, _, layer_idx, nxt = best
        budget += nxt - k_by_layer[layer_idx]
        k_by_layer[layer_idx] = nxt

    centers = [center_cache[i][k_by_layer[i]] for i in range(num_layers)]
    meta = {
        "mode": "adaptive_avg",
        "avg_k": float(avg_k),
        "target_budget": int(target_budget),
        "actual_budget": int(sum(k_by_layer)),
        "actual_avg_k": float(sum(k_by_layer) / num_layers),
        "candidates": candidates,
        "k_by_layer": {f"layer_{i:02d}": int(k) for i, k in enumerate(k_by_layer)},
        "calib_fallback_rate_by_layer": {
            f"layer_{i:02d}": float(fallback_cache[i][k_by_layer[i]]) for i in range(num_layers)
        },
    }
    return centers, meta


class RmsAlphaBucketController:
    """Capture exact post-RMSNorm alpha=1/rms(x), then bucketize it for correction scaling."""

    def __init__(
        self,
        npz_path: Path,
        num_layers: int,
        bucket_k: int,
        fallback_rel: float,
        bucket_mode: str = "fixed",
        bucket_avg_k: float = 4.0,
    ) -> None:
        self.npz_path = Path(npz_path)
        self.num_layers = int(num_layers)
        self.bucket_k = int(bucket_k)
        self.bucket_mode = str(bucket_mode)
        self.bucket_avg_k = float(bucket_avg_k)
        self.fallback_rel = float(fallback_rel)
        data = np.load(self.npz_path)
        values_by_layer = []
        for layer_idx in range(self.num_layers):
            key = f"inv_rms_post_layer_{layer_idx:02d}"
            if key not in data:
                raise KeyError(f"Missing {key} in {self.npz_path}")
            values_by_layer.append(np.asarray(data[key], dtype=np.float32))
        if self.bucket_mode == "fixed":
            centers = [kmeans1d_centers(values, self.bucket_k) for values in values_by_layer]
            self.allocation_meta = {
                "mode": "fixed",
                "bucket_k": int(self.bucket_k),
                "actual_budget": int(self.bucket_k * self.num_layers),
                "actual_avg_k": float(self.bucket_k),
                "k_by_layer": {f"layer_{i:02d}": int(self.bucket_k) for i in range(self.num_layers)},
            }
        elif self.bucket_mode == "adaptive_avg":
            centers, self.allocation_meta = adaptive_average_k_centers(
                values_by_layer=values_by_layer,
                avg_k=self.bucket_avg_k,
                fallback_rel=self.fallback_rel,
            )
        else:
            raise ValueError(f"Unknown RMS bucket mode: {self.bucket_mode}")
        self.centers_cpu = [torch.from_numpy(c).to(dtype=torch.float32).contiguous() for c in centers]
        self._centers_dev: Dict[tuple, torch.Tensor] = {}
        self._alpha: Dict[int, torch.Tensor] = {}
        self._handles = []
        self.reset_stats()

    def register(self, layers: Iterable[nn.Module]) -> None:
        self.remove()
        for layer_idx, layer in enumerate(layers):
            eps = float(getattr(layer.post_attention_layernorm, "variance_epsilon", 1e-6))

            def _hook(_module, inputs, idx=layer_idx, eps_val=eps):
                x = inputs[0].detach().float()
                alpha = torch.rsqrt(torch.mean(x * x, dim=-1) + eps_val)
                self._alpha[idx] = alpha

            self._handles.append(layer.post_attention_layernorm.register_forward_pre_hook(_hook))

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def reset_stats(self) -> None:
        self.total_tokens = 0
        self.fallback_tokens = 0
        self.layer_total = [0 for _ in range(self.num_layers)]
        self.layer_fallback = [0 for _ in range(self.num_layers)]

    def centers(self, layer_idx: int, device: torch.device) -> torch.Tensor:
        key = (int(layer_idx), str(device))
        cached = self._centers_dev.get(key)
        if cached is not None:
            return cached
        dev = self.centers_cpu[int(layer_idx)].to(device=device)
        self._centers_dev[key] = dev
        return dev

    def ratio(self, layer_idx: int, ref: torch.Tensor) -> torch.Tensor:
        ratio, _fallback = self.ratio_and_fallback(layer_idx, ref)
        return ratio

    def ratio_and_fallback(self, layer_idx: int, ref: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        alpha = self._alpha.get(int(layer_idx))
        if alpha is None:
            raise RuntimeError(f"Missing RMS alpha capture for layer {layer_idx}")
        alpha = alpha.to(device=ref.device, dtype=torch.float32)
        centers = self.centers(layer_idx, ref.device)
        nearest = centers[torch.argmin(torch.abs(alpha.unsqueeze(-1) - centers.view(*([1] * alpha.dim()), -1)), dim=-1)]
        rel = torch.abs(nearest - alpha) / torch.clamp(torch.abs(alpha), min=1e-12)
        fallback = rel > self.fallback_rel
        ratio = torch.where(fallback, torch.ones_like(alpha), nearest / torch.clamp(alpha, min=1e-12))
        tokens = int(alpha.numel())
        fallback_count = int(fallback.sum().item())
        self.total_tokens += tokens
        self.fallback_tokens += fallback_count
        self.layer_total[int(layer_idx)] += tokens
        self.layer_fallback[int(layer_idx)] += fallback_count
        return ratio.to(dtype=ref.dtype), fallback

    def stats(self) -> dict:
        layer_rates = {}
        for idx, total in enumerate(self.layer_total):
            layer_rates[f"layer_{idx:02d}"] = (
                float(self.layer_fallback[idx] / total) if total > 0 else None
            )
        return {
            "bucket_k": int(self.bucket_k),
            "bucket_mode": str(self.bucket_mode),
            "bucket_avg_k": float(self.bucket_avg_k),
            "allocation": self.allocation_meta,
            "fallback_rel": float(self.fallback_rel),
            "total_tokens": int(self.total_tokens),
            "fallback_tokens": int(self.fallback_tokens),
            "fallback_rate": float(self.fallback_tokens / self.total_tokens) if self.total_tokens > 0 else None,
            "layer_fallback_rate": layer_rates,
            "centers": {
                f"layer_{idx:02d}": [float(x) for x in centers.tolist()]
                for idx, centers in enumerate(self.centers_cpu)
            },
        }


class TealTopKMLP(nn.Module):
    """Local TEAL definition: dense gate, top-k up input, top-k down input."""

    def __init__(self, inner: nn.Module, sparsity: int) -> None:
        super().__init__()
        self.inner = inner
        self.sparsity = int(sparsity)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.sparsity <= 0:
            gate = self.inner.gate_proj(x)
            up = self.inner.up_proj(x)
            return self.inner.down_proj(self.inner.act_fn(gate) * up)
        gate = self.inner.gate_proj(x)
        up = self.inner.up_proj(topk_abs(x, self.sparsity))
        down_in = self.inner.act_fn(gate) * up
        down_in = topk_abs(down_in, self.sparsity)
        return self.inner.down_proj(down_in)


class TopKInputLinear(nn.Module):
    def __init__(self, inner: nn.Module, sparsity: int) -> None:
        super().__init__()
        self.inner = inner
        self.sparsity = int(sparsity)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.inner(topk_abs(x, self.sparsity)) if self.sparsity > 0 else self.inner(x)


class RotatedTopKInputLinear(nn.Module):
    def __init__(self, inner: nn.Module, q_path: Path, sparsity: int) -> None:
        super().__init__()
        self.inner = inner
        self.q_path = Path(q_path)
        self.sparsity = int(sparsity)
        self._q_cache: Dict[tuple, torch.Tensor] = {}

    def q(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        key = (str(device), str(dtype))
        cached = self._q_cache.get(key)
        if cached is not None:
            return cached
        q_cpu = torch.load(self.q_path, map_location="cpu")
        q_dev = q_cpu.to(device=device, dtype=dtype).contiguous()
        self._q_cache[key] = q_dev
        return q_dev

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.sparsity <= 0:
            return self.inner(x)
        q = self.q(x.device, x.dtype)
        rotated = torch.matmul(x, q)
        rotated_sparse = topk_abs(rotated, self.sparsity)
        x_eff = torch.matmul(rotated_sparse, q.transpose(-1, -2))
        return self.inner(x_eff)


class LazyShiftIdealTopKMLP(nn.Module):
    """Gate output top-k with exact dense median correction."""

    def __init__(self, inner: nn.Module, layer_idx: int, controller: MedianController, sparsity: int) -> None:
        super().__init__()
        self.inner = inner
        self.layer_idx = int(layer_idx)
        self.controller = controller
        self.sparsity = int(sparsity)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.inner.gate_proj(x)
        up = self.inner.up_proj(x)
        h1 = self.inner.act_fn(gate)
        if self.sparsity <= 0:
            return self.inner.down_proj(h1 * up)
        med = self.controller.median(self.layer_idx, h1.device, h1.dtype)
        med_b = med.view(1, 1, -1) if h1.dim() == 3 else med.view(1, -1)
        centered = h1 - med_b
        centered_sparse = topk_abs(centered, self.sparsity)
        sparse_out = self.inner.down_proj(centered_sparse * up)
        correction = self.inner.down_proj(med_b * up)
        return sparse_out + correction


class LarosaUpTopKMLP(nn.Module):
    """LaRoSA-style rotated top-k for the up-projection input; gate and down follow local TEAL."""

    def __init__(self, inner: nn.Module, q_path: Path, sparsity: int) -> None:
        super().__init__()
        self.inner = inner
        self.q_path = Path(q_path)
        self.sparsity = int(sparsity)
        self._q_cache: Dict[tuple, torch.Tensor] = {}

    def q(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        key = (str(device), str(dtype))
        cached = self._q_cache.get(key)
        if cached is not None:
            return cached
        q_cpu = torch.load(self.q_path, map_location="cpu")
        q_dev = q_cpu.to(device=device, dtype=dtype).contiguous()
        self._q_cache[key] = q_dev
        return q_dev

    def rotated_topk_back(self, x: torch.Tensor) -> torch.Tensor:
        if self.sparsity <= 0:
            return x
        q = self.q(x.device, x.dtype)
        rotated = torch.matmul(x, q)
        rotated_sparse = topk_abs(rotated, self.sparsity)
        return torch.matmul(rotated_sparse, q.transpose(-1, -2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.sparsity <= 0:
            gate = self.inner.gate_proj(x)
            up = self.inner.up_proj(x)
            return self.inner.down_proj(self.inner.act_fn(gate) * up)
        gate = self.inner.gate_proj(x)
        up = self.inner.up_proj(self.rotated_topk_back(x))
        down_in = self.inner.act_fn(gate) * up
        down_in = topk_abs(down_in, self.sparsity)
        return self.inner.down_proj(down_in)


class LarosaGateUpTopKMLP(nn.Module):
    """LaRoSA-style rotated top-k for both gate/up inputs; h4/down input uses Q=I top-k."""

    def __init__(self, inner: nn.Module, q_path: Path, sparsity: int) -> None:
        super().__init__()
        self.inner = inner
        self.q_path = Path(q_path)
        self.sparsity = int(sparsity)
        self._q_cache: Dict[tuple, torch.Tensor] = {}

    def q(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        key = (str(device), str(dtype))
        cached = self._q_cache.get(key)
        if cached is not None:
            return cached
        q_cpu = torch.load(self.q_path, map_location="cpu")
        q_dev = q_cpu.to(device=device, dtype=dtype).contiguous()
        self._q_cache[key] = q_dev
        return q_dev

    def rotated_topk_back(self, x: torch.Tensor) -> torch.Tensor:
        if self.sparsity <= 0:
            return x
        q = self.q(x.device, x.dtype)
        rotated = torch.matmul(x, q)
        rotated_sparse = topk_abs(rotated, self.sparsity)
        return torch.matmul(rotated_sparse, q.transpose(-1, -2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.sparsity <= 0:
            gate = self.inner.gate_proj(x)
            up = self.inner.up_proj(x)
            return self.inner.down_proj(self.inner.act_fn(gate) * up)
        x_sparse = self.rotated_topk_back(x)
        gate = self.inner.gate_proj(x_sparse)
        up = self.inner.up_proj(x_sparse)
        down_in = self.inner.act_fn(gate) * up
        down_in = topk_abs(down_in, self.sparsity)
        return self.inner.down_proj(down_in)


class LarosaLazyGateUpTopKMLP(nn.Module):
    """LaRoSA gate/up input top-k followed by lazy-shift gate-output top-k."""

    def __init__(
        self,
        inner: nn.Module,
        q_path: Path,
        layer_idx: int,
        median_controller: LarosaSparseGateMedianController,
        sparsity: int,
    ) -> None:
        super().__init__()
        self.inner = inner
        self.q_path = Path(q_path)
        self.layer_idx = int(layer_idx)
        self.median_controller = median_controller
        self.sparsity = int(sparsity)
        self._q_cache: Dict[tuple, torch.Tensor] = {}

    def q(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        key = (str(device), str(dtype))
        cached = self._q_cache.get(key)
        if cached is not None:
            return cached
        q_cpu = torch.load(self.q_path, map_location="cpu")
        q_dev = q_cpu.to(device=device, dtype=dtype).contiguous()
        self._q_cache[key] = q_dev
        return q_dev

    def rotated_topk_back(self, x: torch.Tensor) -> torch.Tensor:
        if self.sparsity <= 0:
            return x
        q = self.q(x.device, x.dtype)
        rotated = torch.matmul(x, q)
        rotated_sparse = topk_abs(rotated, self.sparsity)
        return torch.matmul(rotated_sparse, q.transpose(-1, -2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.sparsity <= 0:
            gate = self.inner.gate_proj(x)
            up = self.inner.up_proj(x)
            return self.inner.down_proj(self.inner.act_fn(gate) * up)
        x_sparse = self.rotated_topk_back(x)
        gate = self.inner.gate_proj(x_sparse)
        up = self.inner.up_proj(x_sparse)
        h1 = self.inner.act_fn(gate)
        med = self.median_controller.median(self.layer_idx, h1.device, h1.dtype)
        med_b = med.view(1, 1, -1) if h1.dim() == 3 else med.view(1, -1)
        centered = h1 - med_b
        centered_sparse = topk_abs(centered, self.sparsity)
        sparse_out = self.inner.down_proj(centered_sparse * up)
        correction = self.inner.down_proj(med_b * up)
        return sparse_out + correction


class LarosaLazyIdealAllTopKMLP(nn.Module):
    """All-linear LaRoSA+lazy ideal: gate input top-k; up/down use gate-output top-k."""

    def __init__(
        self,
        inner: nn.Module,
        q_path: Path,
        layer_idx: int,
        median_controller: LarosaSparseGateMedianController,
        sparsity: int,
    ) -> None:
        super().__init__()
        self.inner = inner
        self.q_path = Path(q_path)
        self.layer_idx = int(layer_idx)
        self.median_controller = median_controller
        self.sparsity = int(sparsity)
        self._q_cache: Dict[tuple, torch.Tensor] = {}

    def q(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        key = (str(device), str(dtype))
        cached = self._q_cache.get(key)
        if cached is not None:
            return cached
        q_cpu = torch.load(self.q_path, map_location="cpu")
        q_dev = q_cpu.to(device=device, dtype=dtype).contiguous()
        self._q_cache[key] = q_dev
        return q_dev

    def rotated_topk_back(self, x: torch.Tensor) -> torch.Tensor:
        if self.sparsity <= 0:
            return x
        q = self.q(x.device, x.dtype)
        rotated = torch.matmul(x, q)
        rotated_sparse = topk_abs(rotated, self.sparsity)
        return torch.matmul(rotated_sparse, q.transpose(-1, -2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.sparsity <= 0:
            gate = self.inner.gate_proj(x)
            up = self.inner.up_proj(x)
            return self.inner.down_proj(self.inner.act_fn(gate) * up)

        x_gate = self.rotated_topk_back(x)
        gate = self.inner.gate_proj(x_gate)
        h1 = self.inner.act_fn(gate)
        up = self.inner.up_proj(x)
        med = self.median_controller.median(self.layer_idx, h1.device, h1.dtype)
        med_b = med.view(1, 1, -1) if h1.dim() == 3 else med.view(1, -1)
        centered = h1 - med_b
        centered_sparse = topk_abs(centered, self.sparsity)
        sparse_out = self.inner.down_proj(centered_sparse * up)
        correction = self.inner.down_proj(med_b * up)
        return sparse_out + correction


class LarosaLazyBucketAllTopKMLP(nn.Module):
    """All-linear LaRoSA+lazy with K-bucket RMS correction scaling and exact fallback."""

    def __init__(
        self,
        inner: nn.Module,
        q_path: Path,
        layer_idx: int,
        median_controller: LarosaSparseGateMedianController,
        alpha_controller: RmsAlphaBucketController,
        sparsity: int,
    ) -> None:
        super().__init__()
        self.inner = inner
        self.q_path = Path(q_path)
        self.layer_idx = int(layer_idx)
        self.median_controller = median_controller
        self.alpha_controller = alpha_controller
        self.sparsity = int(sparsity)
        self._q_cache: Dict[tuple, torch.Tensor] = {}

    def q(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        key = (str(device), str(dtype))
        cached = self._q_cache.get(key)
        if cached is not None:
            return cached
        q_cpu = torch.load(self.q_path, map_location="cpu")
        q_dev = q_cpu.to(device=device, dtype=dtype).contiguous()
        self._q_cache[key] = q_dev
        return q_dev

    def rotated_topk_back(self, x: torch.Tensor) -> torch.Tensor:
        if self.sparsity <= 0:
            return x
        q = self.q(x.device, x.dtype)
        rotated = torch.matmul(x, q)
        rotated_sparse = topk_abs(rotated, self.sparsity)
        return torch.matmul(rotated_sparse, q.transpose(-1, -2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.sparsity <= 0:
            gate = self.inner.gate_proj(x)
            up = self.inner.up_proj(x)
            return self.inner.down_proj(self.inner.act_fn(gate) * up)

        x_gate = self.rotated_topk_back(x)
        gate = self.inner.gate_proj(x_gate)
        h1 = self.inner.act_fn(gate)
        up = self.inner.up_proj(x)
        med = self.median_controller.median(self.layer_idx, h1.device, h1.dtype)
        med_b = med.view(1, 1, -1) if h1.dim() == 3 else med.view(1, -1)
        centered = h1 - med_b
        centered_sparse = topk_abs(centered, self.sparsity)
        sparse_out = self.inner.down_proj(centered_sparse * up)
        correction = self.inner.down_proj(med_b * up)
        ratio = self.alpha_controller.ratio(self.layer_idx, correction)
        ratio_b = ratio.unsqueeze(-1)
        return sparse_out + correction * ratio_b


class LarosaLazyBucketFusedCorrectionAllTopKMLP(nn.Module):
    """Same PPL path as K-bucket lazy, but non-fallback correction uses one fused HxH matrix."""

    def __init__(
        self,
        inner: nn.Module,
        q_path: Path,
        layer_idx: int,
        median_controller: LarosaSparseGateMedianController,
        alpha_controller: RmsAlphaBucketController,
        sparsity: int,
    ) -> None:
        super().__init__()
        self.inner = inner
        self.q_path = Path(q_path)
        self.layer_idx = int(layer_idx)
        self.median_controller = median_controller
        self.alpha_controller = alpha_controller
        self.sparsity = int(sparsity)
        self._q_cache: Dict[tuple, torch.Tensor] = {}
        self._corr_cache: Dict[tuple, torch.Tensor] = {}

    def q(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        key = (str(device), str(dtype))
        cached = self._q_cache.get(key)
        if cached is not None:
            return cached
        q_cpu = torch.load(self.q_path, map_location="cpu")
        q_dev = q_cpu.to(device=device, dtype=dtype).contiguous()
        self._q_cache[key] = q_dev
        return q_dev

    def correction_matrix(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        key = (str(device), str(dtype))
        cached = self._corr_cache.get(key)
        if cached is not None:
            return cached
        if getattr(self.inner.up_proj, "bias", None) is not None or getattr(self.inner.down_proj, "bias", None) is not None:
            raise ValueError("Fused correction path assumes bias-free up/down projections.")
        med = self.median_controller.median(self.layer_idx, device, torch.float32).view(1, -1)
        up_w = self.inner.up_proj.weight.detach().to(device=device, dtype=torch.float32)
        down_w = self.inner.down_proj.weight.detach().to(device=device, dtype=torch.float32)
        # For PyTorch Linear with row-vector activations:
        # down(med * up(x)) == x @ (W_up.T @ diag(med) @ W_down.T).
        b = torch.matmul(up_w.transpose(0, 1) * med, down_w.transpose(0, 1))
        b = b.to(dtype=dtype).contiguous()
        self._corr_cache[key] = b
        return b

    def rotated_topk_back(self, x: torch.Tensor) -> torch.Tensor:
        if self.sparsity <= 0:
            return x
        q = self.q(x.device, x.dtype)
        rotated = torch.matmul(x, q)
        rotated_sparse = topk_abs(rotated, self.sparsity)
        return torch.matmul(rotated_sparse, q.transpose(-1, -2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.sparsity <= 0:
            gate = self.inner.gate_proj(x)
            up = self.inner.up_proj(x)
            return self.inner.down_proj(self.inner.act_fn(gate) * up)

        x_gate = self.rotated_topk_back(x)
        gate = self.inner.gate_proj(x_gate)
        h1 = self.inner.act_fn(gate)
        up = self.inner.up_proj(x)
        med = self.median_controller.median(self.layer_idx, h1.device, h1.dtype)
        med_b = med.view(1, 1, -1) if h1.dim() == 3 else med.view(1, -1)
        centered = h1 - med_b
        centered_sparse = topk_abs(centered, self.sparsity)
        sparse_out = self.inner.down_proj(centered_sparse * up)

        b = self.correction_matrix(x.device, x.dtype)
        correction_fused = torch.matmul(x, b)
        ratio, fallback = self.alpha_controller.ratio_and_fallback(self.layer_idx, correction_fused)
        correction = correction_fused * ratio.unsqueeze(-1)

        if bool(fallback.any().item()):
            flat_fallback = fallback.reshape(-1)
            flat_up = up.reshape(-1, up.shape[-1])
            fallback_up = flat_up[flat_fallback]
            fallback_exact = self.inner.down_proj(fallback_up * med.view(1, -1))
            correction_flat = correction.reshape(-1, correction.shape[-1]).clone()
            correction_flat[flat_fallback] = fallback_exact
            correction = correction_flat.view_as(correction)

        return sparse_out + correction


def q_from_cov(cov: torch.Tensor, block_size: int) -> torch.Tensor:
    hidden_size = int(cov.shape[0])
    if int(block_size) <= 0 or int(block_size) >= hidden_size:
        evals, evecs = torch.linalg.eigh(cov)
        order = torch.argsort(evals, descending=True)
        q = evecs[:, order]
        del evals, evecs, order
        return q

    q = torch.zeros_like(cov)
    bs = int(block_size)
    for st in range(0, hidden_size, bs):
        ed = min(hidden_size, st + bs)
        sub = cov[st:ed, st:ed]
        evals, evecs = torch.linalg.eigh(sub)
        order = torch.argsort(evals, descending=True)
        q[st:ed, st:ed] = evecs[:, order]
        del evals, evecs, order, sub
    return q


def make_q_cache(
    x_cache_dir: Path,
    q_dir: Path,
    num_layers: int,
    hidden_size: int,
    q_tokens: int,
    chunk: int,
    force: bool,
    block_size: int = 0,
) -> List[Path]:
    q_dir.mkdir(parents=True, exist_ok=True)
    paths = [q_dir / f"q_layer_{i:02d}.pt" for i in range(num_layers)]
    if all(p.exists() for p in paths) and not force:
        return paths

    print(f"Building LaRoSA PCA Q cache: {q_dir} (block_size={int(block_size)})")
    for layer_idx in tqdm(range(num_layers), desc="PCA Q", unit="layer"):
        out_path = paths[layer_idx]
        if out_path.exists() and not force:
            continue
        x_path = x_cache_dir / f"x_layer_{layer_idx:02d}.dat"
        if not x_path.exists():
            raise FileNotFoundError(f"Missing x cache: {x_path}")
        arr = np.memmap(x_path, mode="r", dtype=np.float16)
        total_rows = int(arr.size // hidden_size)
        rows = min(int(q_tokens), total_rows)
        x_mm = arr[: rows * hidden_size].reshape(rows, hidden_size)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        cov = torch.zeros((hidden_size, hidden_size), device=device, dtype=torch.float32)
        for st in range(0, rows, int(chunk)):
            ed = min(rows, st + int(chunk))
            xb = torch.from_numpy(np.asarray(x_mm[st:ed], dtype=np.float32)).to(device=device)
            xb = xb - xb.mean(dim=0, keepdim=True)
            cov.addmm_(xb.transpose(0, 1), xb)
            del xb
        cov.div_(float(max(rows - 1, 1)))
        q = q_from_cov(cov, int(block_size)).to(device="cpu", dtype=torch.float16).contiguous()
        torch.save(q, out_path)
        del cov, q
        torch.cuda.empty_cache()
    return paths


def make_attention_q_cache(
    model: nn.Module,
    model_id: str,
    q_dir: Path,
    num_layers: int,
    hidden_size: int,
    q_tokens: int,
    max_length: int,
    batch_size: int,
    device: torch.device,
    force: bool,
    block_size: int = 0,
) -> List[Path]:
    q_dir.mkdir(parents=True, exist_ok=True)
    paths = [q_dir / f"attn_q_layer_{i:02d}.pt" for i in range(num_layers)]
    if all(p.exists() for p in paths) and not force:
        return paths

    restore_dense(model)
    covs = [torch.zeros((hidden_size, hidden_size), device=device, dtype=torch.float32) for _ in range(num_layers)]
    counts = [0 for _ in range(num_layers)]

    hooks = []

    def make_hook(layer_idx: int):
        def hook(_module, inputs):
            if counts[layer_idx] >= int(q_tokens):
                return
            x = inputs[0].detach().reshape(-1, hidden_size)
            need = min(int(q_tokens) - counts[layer_idx], int(x.shape[0]))
            if need <= 0:
                return
            x = x[:need].float()
            x = x - x.mean(dim=0, keepdim=True)
            covs[layer_idx].addmm_(x.transpose(0, 1), x)
            counts[layer_idx] += int(need)
        return hook

    for i, layer in enumerate(decoder_layers(model)):
        hooks.append(layer.self_attn.q_proj.register_forward_pre_hook(make_hook(i)))

    loader = build_calib_loader(model_id, batch_size=int(batch_size), max_length=int(max_length))
    print(f"Collecting attention input covariance for LaRoSA Q: tokens={q_tokens}")
    try:
        with torch.no_grad():
            for batch in tqdm(loader, desc="Attention Q calib", unit="batch"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
                if min(counts) >= int(q_tokens):
                    break
    finally:
        for h in hooks:
            h.remove()

    for i, cov in enumerate(tqdm(covs, desc="Attention PCA Q", unit="layer")):
        denom = float(max(counts[i] - 1, 1))
        cov = cov / denom
        q = q_from_cov(cov, int(block_size)).to(device="cpu", dtype=torch.float16).contiguous()
        torch.save(q, paths[i])
        del q
    del covs
    torch.cuda.empty_cache()
    return paths


def make_larosa_sparse_gate_median_cache(
    model: nn.Module,
    x_cache_dir: Path,
    q_paths: Iterable[Path],
    median_dir: Path,
    sparsities: Iterable[int],
    num_layers: int,
    hidden_size: int,
    intermediate_size: int,
    tokens: int,
    chunk: int,
    device: torch.device,
) -> None:
    median_dir.mkdir(parents=True, exist_ok=True)
    q_paths = list(q_paths)
    restore_dense(model)
    layers = list(decoder_layers(model))
    for sparsity in sparsities:
        s_dir = median_dir / f"s{int(sparsity):02d}"
        s_dir.mkdir(parents=True, exist_ok=True)
        meta_path = s_dir / "meta.json"
        for layer_idx in tqdm(range(num_layers), desc=f"LaRoSA sparse-gate medians s={sparsity}", unit="layer"):
            out_path = s_dir / f"h1_median_layer_{layer_idx:02d}.bin"
            if out_path.exists():
                continue
            inner = unwrap_mlp(layers[layer_idx].mlp)
            q = torch.load(q_paths[layer_idx], map_location="cpu").to(device=device, dtype=torch.float16).contiguous()
            x_path = x_cache_dir / f"x_layer_{layer_idx:02d}.dat"
            arr = np.memmap(x_path, mode="r", dtype=np.float16)
            total_rows = int(arr.size // hidden_size)
            rows = min(int(tokens), total_rows)
            x_mm = arr[: rows * hidden_size].reshape(rows, hidden_size)
            h_chunks = []
            for st in range(0, rows, int(chunk)):
                ed = min(rows, st + int(chunk))
                x = torch.from_numpy(np.asarray(x_mm[st:ed], dtype=np.float16)).to(device=device, dtype=torch.float16)
                rotated = torch.matmul(x, q)
                x_sparse = torch.matmul(topk_abs(rotated, int(sparsity)), q.transpose(-1, -2))
                h1 = inner.act_fn(inner.gate_proj(x_sparse))
                h_chunks.append(h1.detach().float().cpu())
                del x, rotated, x_sparse, h1
            h_all = torch.cat(h_chunks, dim=0)
            med = h_all.median(dim=0).values.numpy().astype(np.float16)
            med.tofile(out_path)
            del q, h_chunks, h_all, med
            torch.cuda.empty_cache()
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "definition": "median of SiLU(gate_proj(topk((x Q), sparsity) Q^T)) for LaRoSA sparse gate input",
                    "sparsity": int(sparsity),
                    "tokens": int(tokens),
                    "chunk": int(chunk),
                    "num_layers": int(num_layers),
                    "hidden_size": int(hidden_size),
                    "intermediate_size": int(intermediate_size),
                    "x_cache_dir": str(x_cache_dir),
                },
                f,
                indent=2,
            )


def patch_method(
    model: nn.Module,
    method: str,
    sparsity: int,
    controller: MedianController,
    larosa_median_dir: Path,
    intermediate_size: int,
    q_paths: Iterable[Path],
) -> None:
    restore_dense(model)
    q_paths = list(q_paths)
    for layer_idx, layer in enumerate(decoder_layers(model)):
        inner = unwrap_mlp(layer.mlp)
        if method == "teal_topk":
            layer.mlp = TealTopKMLP(inner, sparsity)
        elif method == "larosa_up_topk":
            layer.mlp = LarosaUpTopKMLP(inner, q_paths[layer_idx], sparsity)
        elif method == "larosa_gate_up_topk":
            layer.mlp = LarosaGateUpTopKMLP(inner, q_paths[layer_idx], sparsity)
        elif method == "larosa_lazy_gate_up_topk":
            sparse_median_controller = LarosaSparseGateMedianController(
                median_dir=larosa_median_dir,
                sparsity=int(sparsity),
                intermediate_size=int(intermediate_size),
            )
            layer.mlp = LarosaLazyGateUpTopKMLP(
                inner=inner,
                q_path=q_paths[layer_idx],
                layer_idx=layer_idx,
                median_controller=sparse_median_controller,
                sparsity=sparsity,
            )
        elif method == "lazy_shift_ideal_topk":
            layer.mlp = LazyShiftIdealTopKMLP(inner, layer_idx, controller, sparsity)
        else:
            raise ValueError(f"Unknown method: {method}")


def patch_attention_all(model: nn.Module, method: str, sparsity: int, attn_q_paths: Iterable[Path]) -> None:
    attn_q_paths = list(attn_q_paths)
    rotated_attn_methods = {
        "larosa_all_topk",
        "larosa_lazy_ideal_all_topk",
        "larosa_lazy_bucket_all_topk",
        "larosa_lazy_bucket_fusedcorr_all_topk",
    }
    for layer_idx, layer in enumerate(decoder_layers(model)):
        attn = layer.self_attn
        for name in ("q_proj", "k_proj", "v_proj"):
            base = unwrap_module(getattr(attn, name))
            if method in rotated_attn_methods:
                setattr(attn, name, RotatedTopKInputLinear(base, attn_q_paths[layer_idx], sparsity))
            else:
                setattr(attn, name, TopKInputLinear(base, sparsity))
        base_o = unwrap_module(attn.o_proj)
        attn.o_proj = TopKInputLinear(base_o, sparsity)


def patch_full_method(
    model: nn.Module,
    method: str,
    sparsity: int,
    larosa_median_dir: Path,
    intermediate_size: int,
    mlp_q_paths: Iterable[Path],
    attn_q_paths: Iterable[Path],
    alpha_controller: RmsAlphaBucketController | None = None,
) -> None:
    restore_dense(model)
    mlp_q_paths = list(mlp_q_paths)
    patch_attention_all(model, method, sparsity, attn_q_paths)
    for layer_idx, layer in enumerate(decoder_layers(model)):
        mlp = unwrap_mlp(layer.mlp)
        if method == "teal_all_topk":
            mlp.gate_proj = TopKInputLinear(unwrap_module(mlp.gate_proj), sparsity)
            mlp.up_proj = TopKInputLinear(unwrap_module(mlp.up_proj), sparsity)
            mlp.down_proj = TopKInputLinear(unwrap_module(mlp.down_proj), sparsity)
            layer.mlp = mlp
        elif method == "larosa_all_topk":
            mlp.gate_proj = RotatedTopKInputLinear(unwrap_module(mlp.gate_proj), mlp_q_paths[layer_idx], sparsity)
            mlp.up_proj = RotatedTopKInputLinear(unwrap_module(mlp.up_proj), mlp_q_paths[layer_idx], sparsity)
            mlp.down_proj = TopKInputLinear(unwrap_module(mlp.down_proj), sparsity)
            layer.mlp = mlp
        elif method == "larosa_lazy_ideal_all_topk":
            sparse_median_controller = LarosaSparseGateMedianController(
                median_dir=larosa_median_dir,
                sparsity=int(sparsity),
                intermediate_size=int(intermediate_size),
            )
            layer.mlp = LarosaLazyIdealAllTopKMLP(
                inner=mlp,
                q_path=mlp_q_paths[layer_idx],
                layer_idx=layer_idx,
                median_controller=sparse_median_controller,
                sparsity=sparsity,
            )
        elif method == "larosa_lazy_bucket_all_topk":
            if alpha_controller is None:
                raise ValueError("alpha_controller is required for larosa_lazy_bucket_all_topk")
            sparse_median_controller = LarosaSparseGateMedianController(
                median_dir=larosa_median_dir,
                sparsity=int(sparsity),
                intermediate_size=int(intermediate_size),
            )
            layer.mlp = LarosaLazyBucketAllTopKMLP(
                inner=mlp,
                q_path=mlp_q_paths[layer_idx],
                layer_idx=layer_idx,
                median_controller=sparse_median_controller,
                alpha_controller=alpha_controller,
                sparsity=sparsity,
            )
        elif method == "larosa_lazy_bucket_fusedcorr_all_topk":
            if alpha_controller is None:
                raise ValueError("alpha_controller is required for larosa_lazy_bucket_fusedcorr_all_topk")
            sparse_median_controller = LarosaSparseGateMedianController(
                median_dir=larosa_median_dir,
                sparsity=int(sparsity),
                intermediate_size=int(intermediate_size),
            )
            layer.mlp = LarosaLazyBucketFusedCorrectionAllTopKMLP(
                inner=mlp,
                q_path=mlp_q_paths[layer_idx],
                layer_idx=layer_idx,
                median_controller=sparse_median_controller,
                alpha_controller=alpha_controller,
                sparsity=sparsity,
            )
        else:
            raise ValueError(f"Unknown full method: {method}")


def write_outputs(output_dir: Path, result: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    rows = []
    dense_ppl = float(result["dense"]["ppl"])
    rows.append({"method": "dense", "sparsity": 0, "ppl": dense_ppl, "delta_vs_dense": 0.0})
    for method, vals in result["curves"].items():
        for k, rec in vals.items():
            ppl = float(rec["ppl"])
            rows.append({"method": method, "sparsity": int(k), "ppl": ppl, "delta_vs_dense": ppl - dense_ppl})
    with (output_dir / "ppl_table.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "sparsity", "ppl", "delta_vs_dense"])
        writer.writeheader()
        writer.writerows(rows)

    labels = {
        "teal_topk": "TEAL top-k (up/down input)",
        "larosa_up_topk": "LaRoSA top-k (up input)",
        "larosa_gate_up_topk": "LaRoSA top-k (gate/up input)",
        "larosa_lazy_gate_up_topk": "LaRoSA + lazy shift",
        "lazy_shift_ideal_topk": "lazy shift ideal (dense gate/up)",
        "teal_all_topk": "TEAL all-linear top-k",
        "larosa_all_topk": "LaRoSA all-linear top-k",
        "larosa_lazy_ideal_all_topk": "LaRoSA + lazy ideal all-linear",
        "larosa_lazy_bucket_all_topk": "LaRoSA + lazy bucket+fallback",
        "larosa_lazy_bucket_fusedcorr_all_topk": "LaRoSA + lazy fused-correction bucket+fallback",
    }
    colors = {
        "teal_topk": "tab:blue",
        "larosa_up_topk": "tab:purple",
        "larosa_gate_up_topk": "tab:red",
        "larosa_lazy_gate_up_topk": "tab:green",
        "lazy_shift_ideal_topk": "tab:olive",
        "teal_all_topk": "tab:blue",
        "larosa_all_topk": "tab:red",
        "larosa_lazy_ideal_all_topk": "tab:green",
        "larosa_lazy_bucket_all_topk": "tab:orange",
        "larosa_lazy_bucket_fusedcorr_all_topk": "tab:brown",
    }
    fig, ax = plt.subplots(figsize=(7.4, 4.8), dpi=180)
    all_ks = sorted({int(k) for vals in result["curves"].values() for k in vals.keys()})
    ax.axhline(dense_ppl, color="black", linestyle="--", linewidth=1.0, label=f"dense {dense_ppl:.4f}")
    for method, vals in result["curves"].items():
        xs = [k for k in all_ks if str(k) in vals]
        ys = [float(vals[str(k)]["ppl"]) for k in xs]
        ax.plot(xs, ys, marker="o", linewidth=1.8, color=colors.get(method), label=labels.get(method, method))
    ax.set_xlabel("Target sparsity (%)")
    ax.set_ylabel("PPL")
    ax.set_title("Top-k FFN sparsity PPL")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "ppl_topk_larosa_lazy.png")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    started = time.time()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")
    os.environ.setdefault("HF_HOME", str(DEFAULT_ROOT.parent / "hf_cache"))

    k_list = parse_k_list(args.k_list)
    methods = parse_methods(args.methods)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    with (args.x_cache_dir / "io_meta.json").open("r", encoding="utf-8") as f:
        x_meta = json.load(f)
    num_layers = int(x_meta["num_layers"])
    hidden_size = int(x_meta["hidden_size"])
    intermediate_size = int(x_meta["intermediate_size"])

    q_paths = make_q_cache(
        x_cache_dir=args.x_cache_dir,
        q_dir=output_dir / "q_cache_hidden_pca",
        num_layers=num_layers,
        hidden_size=hidden_size,
        q_tokens=int(args.q_tokens),
        chunk=int(args.q_chunk),
        force=bool(args.force_rebuild_q),
        block_size=int(args.block_size),
    )

    eval_ids, eval_meta = read_evalset(args.evalset_dir)
    if args.limit_eval_samples is not None:
        eval_ids = eval_ids[: int(args.limit_eval_samples)]

    print(f"Loading model: {args.model_id}")
    model = load_model(args)
    dev = input_device(model)
    print(f"Using input device: {dev}")

    full_methods = {
        "teal_all_topk",
        "larosa_all_topk",
        "larosa_lazy_ideal_all_topk",
        "larosa_lazy_bucket_all_topk",
        "larosa_lazy_bucket_fusedcorr_all_topk",
    }
    needs_attn_q = any(
        m
        in {
            "larosa_all_topk",
            "larosa_lazy_ideal_all_topk",
            "larosa_lazy_bucket_all_topk",
            "larosa_lazy_bucket_fusedcorr_all_topk",
        }
        for m in methods
    )
    attn_q_paths = [Path("") for _ in range(num_layers)]
    if needs_attn_q:
        attn_q_paths = make_attention_q_cache(
            model=model,
            model_id=args.model_id,
            q_dir=output_dir / "q_cache_attention_pca",
            num_layers=num_layers,
            hidden_size=hidden_size,
            q_tokens=int(args.attn_q_tokens),
            max_length=int(args.attn_calib_max_length),
            batch_size=int(args.attn_calib_batch_size),
            device=dev,
            force=bool(args.force_rebuild_q),
            block_size=int(args.block_size),
        )

    controller = None
    if any(m not in full_methods for m in methods):
        threshold_payload = load_threshold_payload(args.threshold_dir)
        controller = MedianController(threshold_payload, args.threshold_dir, num_layers, intermediate_size)

    if any(
        m
        in {
            "larosa_lazy_gate_up_topk",
            "larosa_lazy_ideal_all_topk",
            "larosa_lazy_bucket_all_topk",
            "larosa_lazy_bucket_fusedcorr_all_topk",
        }
        for m in methods
    ):
        make_larosa_sparse_gate_median_cache(
            model=model,
            x_cache_dir=args.x_cache_dir,
            q_paths=q_paths,
            median_dir=output_dir / "larosa_sparse_gate_medians",
            sparsities=[k for k in k_list if int(k) > 0],
            num_layers=num_layers,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            tokens=int(args.larosa_median_tokens),
            chunk=int(args.q_chunk),
            device=dev,
        )

    alpha_controller = None
    if "larosa_lazy_bucket_all_topk" in methods or "larosa_lazy_bucket_fusedcorr_all_topk" in methods:
        alpha_controller = RmsAlphaBucketController(
            npz_path=args.rms_stats_npz,
            num_layers=num_layers,
            bucket_k=int(args.rms_bucket_k),
            fallback_rel=float(args.rms_fallback_rel),
            bucket_mode=str(args.rms_bucket_mode),
            bucket_avg_k=float(args.rms_bucket_avg_k),
        )
        alpha_controller.register(decoder_layers(model))

    restore_dense(model)
    dense = eval_ppl(model, eval_ids, int(args.tail_start), dev, "dense")
    curves = {m: {} for m in methods}
    for method in methods:
        for k in k_list:
            print(f"Evaluating {method} sparsity={k}")
            if alpha_controller is not None:
                alpha_controller.reset_stats()
            if method in full_methods:
                patch_full_method(
                    model=model,
                    method=method,
                    sparsity=k,
                    larosa_median_dir=output_dir / "larosa_sparse_gate_medians",
                    intermediate_size=intermediate_size,
                    mlp_q_paths=q_paths,
                    attn_q_paths=attn_q_paths,
                    alpha_controller=alpha_controller,
                )
            else:
                patch_method(
                    model=model,
                    method=method,
                    sparsity=k,
                    controller=controller,
                    larosa_median_dir=output_dir / "larosa_sparse_gate_medians",
                    intermediate_size=intermediate_size,
                    q_paths=q_paths,
                )
            rec = eval_ppl(model, eval_ids, int(args.tail_start), dev, f"{method} k={k}")
            if method in {"larosa_lazy_bucket_all_topk", "larosa_lazy_bucket_fusedcorr_all_topk"} and alpha_controller is not None:
                rec["rms_bucket_stats"] = alpha_controller.stats()
            curves[method][str(k)] = rec
            torch.cuda.empty_cache()

    restore_dense(model)
    if alpha_controller is not None:
        alpha_controller.remove()
    result = {
        "model_id": str(args.model_id),
        "evalset": {
            "path": str(args.evalset_dir),
            "metadata": eval_meta,
            "num_samples_used": int(eval_ids.shape[0]),
            "tail_start": int(args.tail_start),
        },
        "sparsity": {
            "type": "per-token top-k by absolute value",
            "k_list": k_list,
            "block_size": int(args.block_size),
        },
        "definitions": {
            "dense": "unmodified FFN",
            "teal_topk": "local TEAL baseline: gate dense, up input top-k, down input top-k",
            "larosa_up_topk": "LaRoSA-style hidden PCA rotation only for up input top-k; gate dense, down input top-k unrotated",
            "larosa_gate_up_topk": "LaRoSA-style hidden PCA rotation for the shared MLP input; both gate and up use the same rotated top-k input; down input uses Q=I top-k",
            "larosa_lazy_gate_up_topk": "LaRoSA gate/up input top-k, then top-k on centered sparse-gate SiLU(gate)-median plus exact sparse-gate median correction",
            "lazy_shift_ideal_topk": "dense gate/up, top-k on centered SiLU(gate)-median, plus exact dense median correction; Q cancels if gate/up are dense",
            "teal_all_topk": "input top-k on q/k/v/o/gate/up/down",
            "larosa_all_topk": "q/k/v/gate/up use PCA-rotated input top-k; o/down use Q=I input top-k",
            "larosa_lazy_ideal_all_topk": "q/k/v use PCA-rotated input top-k, o uses input top-k, gate uses PCA-rotated input top-k; up/down are selected by top-k on centered gate output and exact median correction is added",
            "larosa_lazy_bucket_all_topk": "same sparse path as larosa_lazy_ideal_all_topk, but median correction is scaled by nearest K-bucket 1/rms(x); tokens with relative alpha error above fallback_rel use exact correction",
            "larosa_lazy_bucket_fusedcorr_all_topk": "same sparse path as larosa_lazy_bucket_all_topk, but non-fallback median correction is computed as one fused HxH matrix x @ (W_up.T diag(median) W_down.T); fallback tokens use exact down(median * up)",
        },
        "calibration": {
            "x_cache_dir": str(args.x_cache_dir),
            "q_cache_dir": str(output_dir / "q_cache_hidden_pca"),
            "attn_q_cache_dir": str(output_dir / "q_cache_attention_pca") if needs_attn_q else None,
            "attn_q_tokens": int(args.attn_q_tokens),
            "q_tokens": int(args.q_tokens),
            "block_size": int(args.block_size),
            "threshold_dir": str(args.threshold_dir),
            "larosa_sparse_gate_median_dir": str(output_dir / "larosa_sparse_gate_medians"),
            "larosa_median_tokens": int(args.larosa_median_tokens),
            "rms_stats_npz": str(args.rms_stats_npz),
            "rms_bucket_k": int(args.rms_bucket_k),
            "rms_bucket_mode": str(args.rms_bucket_mode),
            "rms_bucket_avg_k": float(args.rms_bucket_avg_k),
            "rms_fallback_rel": float(args.rms_fallback_rel),
        },
        "dense": dense,
        "curves": curves,
        "elapsed_minutes": (time.time() - started) / 60.0,
    }
    write_outputs(output_dir, result)
    print(json.dumps({"dense": dense["ppl"], **{m: {k: v["ppl"] for k, v in vals.items()} for m, vals in curves.items()}}, indent=2))
    print(f"Saved: {output_dir / 'summary.json'}")
    print(f"Saved: {output_dir / 'ppl_table.csv'}")
    print(f"Saved: {output_dir / 'ppl_topk_larosa_lazy.png'}")


if __name__ == "__main__":
    main()
