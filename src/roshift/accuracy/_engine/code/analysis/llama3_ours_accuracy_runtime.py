#!/usr/bin/env python3
"""PyTorch reference implementation of the released Ours accuracy path.

This module intentionally uses regular PyTorch operators.  It mirrors the
mathematics of the CUDA runtime for PPL and lm-eval accuracy checks; it is not
intended to measure latency.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import torch
import torch.nn as nn
import torch.nn.functional as F

import run_llama3_output_aware_absorbed_lazy_ppl as lazy_abs
from run_llama2_ffn_gate_output_threshold_ppl import decoder_layers


class SparseController(Protocol):
    def sparse_enabled(self) -> bool: ...


@dataclass(frozen=True)
class LMHeadPolicy:
    name: str
    first_dim: int
    second_dim: int
    z1: float
    z2: float


LMHEAD_POLICIES = {
    "aggressive": LMHeadPolicy("aggressive", 512, 1536, 6.6, 6.4),
    "conservative": LMHeadPolicy("conservative", 512, 2048, 11.2, 5.75),
    "mixed_strict": LMHeadPolicy("mixed_strict", 512, 1536, 6.975, 3.85),
}
LMHEAD_POLICY_NAMES = ("dense", "qhead_dense", *LMHEAD_POLICIES)


def parse_lmhead_policies(raw: str) -> list[str]:
    values = [value.strip() for value in str(raw).split(",") if value.strip()]
    invalid = sorted(set(values).difference(LMHEAD_POLICY_NAMES))
    if invalid:
        raise ValueError(
            f"unsupported LM-head policies {invalid}; expected {LMHEAD_POLICY_NAMES}"
        )
    if not values:
        raise ValueError("at least one LM-head policy is required")
    return list(dict.fromkeys(values))


def _site_seed(seed: int, site: str) -> int:
    site_code = sum((index + 1) * ord(char) for index, char in enumerate(site))
    return int(seed) * 1_000_003 + site_code


def fixed_permutation(dim: int, seed: int, site: str) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(_site_seed(int(seed), str(site)))
    return torch.randperm(int(dim), generator=generator, dtype=torch.long)


def keep_count(dim: int, sparsity: int) -> int:
    keep = round(int(dim) * (100.0 - float(sparsity)) / 100.0)
    return max(0, min(int(dim), int(keep)))


def blockwise_topk_mask(
    x: torch.Tensor,
    score_scale: torch.Tensor | None,
    sparsity: int,
    block_size: int,
    permutation: torch.Tensor,
) -> torch.Tensor:
    """Exact local top-k with a fixed total keep count across equal blocks."""

    dim = int(x.shape[-1])
    block_size = int(block_size)
    if block_size == 0:
        block_size = dim
    if block_size < 0 or dim % block_size:
        raise ValueError(f"dimension {dim} is not divisible by block size {block_size}")
    keep = keep_count(dim, int(sparsity))
    if keep >= dim:
        return torch.ones_like(x, dtype=torch.bool)
    if keep <= 0:
        return torch.zeros_like(x, dtype=torch.bool)

    perm = permutation.to(device=x.device, dtype=torch.long)
    score = x.float().abs()
    if score_scale is not None:
        scale = score_scale.to(device=x.device, dtype=torch.float32)
        score = score * scale.view(*([1] * (x.dim() - 1)), dim)
    score = score.index_select(-1, perm)

    blocks = dim // block_size
    keep_lo, extra = divmod(keep, blocks)
    keep_hi = keep_lo + int(extra > 0)
    block_scores = score.reshape(*score.shape[:-1], blocks, block_size)
    top_indices = torch.topk(
        block_scores,
        k=keep_hi,
        dim=-1,
        largest=True,
        sorted=False,
    ).indices

    quota = torch.full((blocks,), keep_lo, device=x.device, dtype=torch.long)
    if extra:
        quota[:extra] += 1
    rank_shape = [1] * (top_indices.dim() - 2) + [1, keep_hi]
    ranks = torch.arange(keep_hi, device=x.device).view(*rank_shape)
    quota_shape = [1] * (top_indices.dim() - 2) + [blocks, 1]
    selected = ranks < quota.view(*quota_shape)

    mask_blocks = torch.zeros_like(block_scores, dtype=torch.bool)
    mask_blocks.scatter_(-1, top_indices, selected.expand_as(top_indices))
    mask_permuted = mask_blocks.reshape_as(score)
    mask = torch.zeros_like(x, dtype=torch.bool)
    scatter_index = perm.view(*([1] * (x.dim() - 1)), dim).expand_as(x)
    mask.scatter_(-1, scatter_index, mask_permuted)
    return mask


def apply_blockwise_topk(
    x: torch.Tensor,
    score_scale: torch.Tensor | None,
    sparsity: int,
    block_size: int,
    permutation: torch.Tensor,
    sparse_start: int,
) -> torch.Tensor:
    if int(sparsity) <= 0:
        return x

    def sparse(part: torch.Tensor) -> torch.Tensor:
        mask = blockwise_topk_mask(
            part, score_scale, int(sparsity), int(block_size), permutation
        )
        return torch.where(mask, part, torch.zeros_like(part))

    if x.dim() < 3 or int(sparse_start) <= 0:
        return sparse(x)
    start = int(sparse_start)
    if start >= int(x.shape[-2]):
        return x
    out = x.clone()
    out[..., start:, :] = sparse(x[..., start:, :])
    return out


def _enabled(controller: SparseController | None) -> bool:
    return controller is None or bool(controller.sparse_enabled())


class BlockwiseInputLinear(nn.Module):
    def __init__(
        self,
        inner: nn.Module,
        *,
        score_scale: torch.Tensor | None,
        permutation: torch.Tensor,
        sparsity: int,
        block_size: int,
        sparse_start: int,
        controller: SparseController | None,
    ) -> None:
        super().__init__()
        self.inner = inner
        self.score_scale_cpu = (
            None
            if score_scale is None
            else score_scale.detach().cpu().float().contiguous()
        )
        self.permutation_cpu = permutation.detach().cpu().long().contiguous()
        self.sparsity = int(sparsity)
        self.block_size = int(block_size)
        self.sparse_start = int(sparse_start)
        self.controller = controller
        self._cache: dict[tuple[str, str], tuple[torch.Tensor | None, torch.Tensor]] = {}

    def tensors(
        self, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        key = (str(device), str(dtype))
        if key not in self._cache:
            scale = (
                None
                if self.score_scale_cpu is None
                else self.score_scale_cpu.to(device=device, dtype=dtype)
            )
            self._cache[key] = (scale, self.permutation_cpu.to(device=device))
        return self._cache[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.sparsity > 0 and _enabled(self.controller):
            scale, permutation = self.tensors(x.device, x.dtype)
            x = apply_blockwise_topk(
                x,
                scale,
                self.sparsity,
                self.block_size,
                permutation,
                self.sparse_start,
            )
        return self.inner(x)


def unwrap_blockwise_linear(module: nn.Module) -> nn.Module:
    while isinstance(module, BlockwiseInputLinear):
        module = module.inner
    return lazy_abs.unwrap_sparse_linear(module)


def restore_body(model: nn.Module, base_layers: list[nn.Module]) -> None:
    model.model.layers = nn.ModuleList(base_layers)
    for layer in base_layers:
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            setattr(
                layer.self_attn,
                name,
                unwrap_blockwise_linear(getattr(layer.self_attn, name)),
            )


def load_shared_qkv_scale(
    d_dir: Path, layer_idx: int
) -> torch.Tensor:
    values = []
    for name in ("q_proj", "k_proj", "v_proj"):
        values.append(
            torch.load(
                lazy_abs.d_path(Path(d_dir), int(layer_idx), name),
                map_location="cpu",
                weights_only=True,
            ).float()
        )
    return torch.sqrt(torch.clamp(sum(value.square() for value in values), min=1e-12))


class BlockwiseOursDecoderLayer(lazy_abs.OutputAwareAbsorbedLazyDecoderLayer):
    def __init__(
        self,
        *,
        inner: nn.Module,
        layer_idx: int,
        pack_dir: Path,
        d_dir: Path,
        median_dir: Path,
        sparsity: int,
        intermediate_size: int,
        alpha_controller,
        transition_mode: str,
        block_size: int,
        h1_block_size: int,
        sparse_start: int,
        permutation_seed: int,
        permute_h1: bool,
        controller: SparseController | None,
        bucket_dir: Path | None = None,
    ) -> None:
        super().__init__(
            inner=inner,
            layer_idx=layer_idx,
            pack_dir=pack_dir,
            d_dir=d_dir,
            median_dir=median_dir,
            sparsity=sparsity,
            intermediate_size=intermediate_size,
            alpha_controller=alpha_controller,
            transition_mode=transition_mode,
        )
        self.block_size = int(block_size)
        self.h1_block_size = int(h1_block_size)
        self.sparse_start = int(sparse_start)
        self.controller = controller
        self.bucket_dir = None if bucket_dir is None else Path(bucket_dir)
        hidden = int(inner.mlp.up_proj.in_features)
        intermediate = int(inner.mlp.down_proj.in_features)
        self.gate_permutation_cpu = fixed_permutation(
            hidden, permutation_seed, f"layer{layer_idx}:gate"
        )
        self.h1_permutation_cpu = (
            fixed_permutation(
                intermediate, permutation_seed, f"layer{layer_idx}:h1"
            )
            if permute_h1
            else torch.arange(intermediate, dtype=torch.long)
        )
        self._permutation_cache: dict[tuple[str, str], torch.Tensor] = {}

    def path_sparse(
        self, name: str, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        if name != "bucket_transition.pt" or self.bucket_dir is None:
            return super().path_sparse(name, device, dtype)
        key = (f"adaptive:s{self.sparsity}:{name}", str(device), str(dtype))
        if key not in self._transition_cache:
            path = (
                self.bucket_dir
                / f"s{self.sparsity:02d}"
                / f"layer_{self.layer_idx:02d}"
                / name
            )
            self._transition_cache[key] = lazy_abs.load_tensor(path, device, dtype)
        return self._transition_cache[key]

    def permutation(self, site: str, device: torch.device) -> torch.Tensor:
        key = (site, str(device))
        if key not in self._permutation_cache:
            source = (
                self.gate_permutation_cpu
                if site == "gate"
                else self.h1_permutation_cpu
            )
            self._permutation_cache[key] = source.to(device=device)
        return self._permutation_cache[key]

    def has_sparse_rows(self, x: torch.Tensor) -> bool:
        if self.sparsity <= 0 or not _enabled(self.controller):
            return False
        return x.dim() < 3 or self.sparse_start <= 0 or self.sparse_start < int(x.shape[-2])

    def transition_scoped(
        self, u_mid: torch.Tensor, alpha: torch.Tensor, sparse_rows: bool
    ) -> torch.Tensor:
        base = self.path_common("base_transition.pt", u_mid.device, u_mid.dtype)
        if not sparse_rows:
            return torch.matmul(u_mid, base)
        if u_mid.dim() < 3 or self.sparse_start <= 0:
            return self.transition(u_mid, alpha)
        start = int(self.sparse_start)
        out = torch.matmul(u_mid, base)
        out[..., start:, :] = self.transition(
            u_mid[..., start:, :], alpha[..., start:]
        )
        return out

    def sparse_h1(self, h1: torch.Tensor, median: torch.Tensor) -> torch.Tensor:
        permutation = self.permutation("h1", h1.device)
        if h1.dim() < 3 or self.sparse_start <= 0:
            centered = h1 - median
            return apply_blockwise_topk(
                centered,
                None,
                self.sparsity,
                self.h1_block_size,
                permutation,
                0,
            )
        start = int(self.sparse_start)
        if start >= int(h1.shape[-2]):
            return h1
        out = h1.clone()
        centered = h1[..., start:, :] - median
        out[..., start:, :] = apply_blockwise_topk(
            centered,
            None,
            self.sparsity,
            self.h1_block_size,
            permutation,
            0,
        )
        return out

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values=None,
        use_cache: bool | None = False,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        u = hidden_states
        attn_alpha = self.inv_rms(u, self.inner.input_layernorm)
        attn_out, _ = self.inner.self_attn(
            hidden_states=u * attn_alpha.unsqueeze(-1),
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        u_mid = u + attn_out

        mlp_alpha = self.inv_rms(u_mid, self.inner.post_attention_layernorm)
        sparse_rows = self.has_sparse_rows(u_mid)
        dense_input = u_mid * mlp_alpha.unsqueeze(-1)
        if sparse_rows:
            gate_u = apply_blockwise_topk(
                u_mid,
                self.d("gate_proj", u_mid.device, u_mid.dtype),
                self.sparsity,
                self.block_size,
                self.permutation("gate", u_mid.device),
                self.sparse_start,
            )
            gate = self.inner.mlp.gate_proj(
                gate_u * mlp_alpha.unsqueeze(-1)
            )
        else:
            gate = self.inner.mlp.gate_proj(dense_input)
        up = self.inner.mlp.up_proj(dense_input)
        h1 = self.inner.mlp.act_fn(gate)
        if sparse_rows:
            median = self.median_controller.median(
                self.layer_idx, h1.device, h1.dtype
            )
            median = median.view(*([1] * (h1.dim() - 1)), -1)
            h1 = self.sparse_h1(h1, median)
        mlp_out = self.inner.mlp.down_proj(h1 * up)
        return self.transition_scoped(u_mid, mlp_alpha, sparse_rows) + mlp_out


def patch_body(
    *,
    model: nn.Module,
    base_layers: list[nn.Module],
    pack_dir: Path,
    d_dir: Path,
    median_dir: Path,
    sparsity: int,
    intermediate_size: int,
    alpha_controller,
    transition_mode: str,
    block_size: int = 128,
    h1_block_size: int | None = None,
    sparse_start: int = 0,
    permutation_seed: int = 11,
    permute_o_h1: bool = True,
    controller: SparseController | None = None,
    bucket_dir: Path | None = None,
) -> None:
    resolved_h1_block_size = (
        int(block_size) if h1_block_size is None else int(h1_block_size)
    )
    restore_body(model, base_layers)
    patched: list[nn.Module] = []
    for layer_idx, layer in enumerate(base_layers):
        qkv_scale = load_shared_qkv_scale(d_dir, layer_idx)
        qkv_permutation = fixed_permutation(
            int(qkv_scale.numel()),
            permutation_seed,
            f"layer{layer_idx}:qkv",
        )
        for name in ("q_proj", "k_proj", "v_proj"):
            setattr(
                layer.self_attn,
                name,
                BlockwiseInputLinear(
                    unwrap_blockwise_linear(getattr(layer.self_attn, name)),
                    score_scale=qkv_scale,
                    permutation=qkv_permutation,
                    sparsity=sparsity,
                    block_size=block_size,
                    sparse_start=sparse_start,
                    controller=controller,
                ),
            )

        o_proj = unwrap_blockwise_linear(layer.self_attn.o_proj)
        o_dim = int(o_proj.in_features)
        o_permutation = (
            fixed_permutation(
                o_dim, permutation_seed, f"layer{layer_idx}:o"
            )
            if permute_o_h1
            else torch.arange(o_dim, dtype=torch.long)
        )
        layer.self_attn.o_proj = BlockwiseInputLinear(
            o_proj,
            score_scale=None,
            permutation=o_permutation,
            sparsity=sparsity,
            block_size=block_size,
            sparse_start=sparse_start,
            controller=controller,
        )
        patched.append(
            BlockwiseOursDecoderLayer(
                inner=layer,
                layer_idx=layer_idx,
                pack_dir=pack_dir,
                d_dir=d_dir,
                median_dir=median_dir,
                sparsity=sparsity,
                intermediate_size=intermediate_size,
                alpha_controller=alpha_controller,
                transition_mode=transition_mode,
                block_size=block_size,
                h1_block_size=resolved_h1_block_size,
                sparse_start=sparse_start,
                permutation_seed=permutation_seed,
                permute_h1=permute_o_h1,
                controller=controller,
                bucket_dir=bucket_dir,
            )
        )
    model.model.layers = nn.ModuleList(patched)


def _row_std(values: torch.Tensor) -> torch.Tensor:
    values = values.float()
    mean = values.mean(dim=-1)
    mean_square = values.square().mean(dim=-1)
    return (mean_square - mean.square()).clamp_min_(0.0).sqrt_().clamp_min_(1e-8)


def _masked_stats(
    values: torch.Tensor, mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    count = mask.sum(dim=-1).clamp_min_(1)
    selected = values.masked_fill(~mask, 0.0)
    mean = selected.sum(dim=-1) / count
    mean_square = selected.square().sum(dim=-1) / count
    sigma = (mean_square - mean.square()).clamp_min_(0.0).sqrt_().clamp_min_(1e-8)
    maximum = values.masked_fill(~mask, float("-inf")).amax(dim=-1)
    return maximum, sigma


@torch.no_grad()
def fold_lmhead_weight(
    raw_weight: torch.Tensor,
    gamma: torch.Tensor,
    qhead: torch.Tensor,
    chunk_rows: int = 4096,
) -> torch.Tensor:
    folded = torch.empty_like(raw_weight)
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        q_float = qhead.float()
        gamma_float = gamma.float()
        for start in range(0, int(raw_weight.shape[0]), int(chunk_rows)):
            end = min(start + int(chunk_rows), int(raw_weight.shape[0]))
            part = raw_weight[start:end].float() * gamma_float.view(1, -1)
            folded[start:end].copy_(torch.mm(part, q_float))
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32
    return folded.contiguous()


class StagedSparseQHeadLMHead(nn.Module):
    """Full-logits PyTorch reference for the staged sparse LM-head policy."""

    def __init__(
        self,
        inner: nn.Module,
        *,
        final_gamma: torch.Tensor,
        qhead: torch.Tensor,
        policy: str,
        controller: SparseController | None = None,
        chunk_rows: int = 32,
        fold_chunk_rows: int = 4096,
    ) -> None:
        super().__init__()
        if policy not in LMHEAD_POLICIES:
            raise ValueError(f"unsupported LM-head policy: {policy}")
        self.inner = inner
        self.controller = controller
        self.policy_name = str(policy)
        self.chunk_rows = int(chunk_rows)
        if self.chunk_rows <= 0:
            raise ValueError("chunk_rows must be positive")
        device = inner.weight.device
        dtype = inner.weight.dtype
        gamma = final_gamma.detach().to(device=device, dtype=dtype).contiguous()
        q = qhead.detach().to(device=device, dtype=dtype).contiguous()
        self.register_buffer("final_gamma", gamma, persistent=False)
        self.register_buffer("qhead", q, persistent=False)
        self.register_buffer(
            "folded_weight",
            fold_lmhead_weight(inner.weight.detach(), gamma, q, fold_chunk_rows),
            persistent=False,
        )
        self.reset_stats()

    @property
    def weight(self) -> torch.Tensor:
        return self.inner.weight

    def set_policy(self, policy: str) -> None:
        if policy not in LMHEAD_POLICY_NAMES:
            raise ValueError(f"unsupported LM-head policy: {policy}")
        self.policy_name = str(policy)

    def reset_stats(self) -> None:
        self.rows = 0
        self.c1_sum = 0
        self.c2_sum = 0

    def stats(self) -> dict:
        policy = LMHEAD_POLICIES.get(self.policy_name)
        if policy is None or self.rows <= 0:
            return {
                "policy": self.policy_name,
                "rows": int(self.rows),
                "effective_sparsity": 0.0,
                "effective_sparsity_pct": 0.0,
            }
        vocab = int(self.folded_weight.shape[0])
        c1_ratio = self.c1_sum / float(self.rows * vocab)
        c2_ratio = self.c2_sum / float(self.rows * vocab)
        equivalent = policy.first_dim
        equivalent += (policy.second_dim - policy.first_dim) * c1_ratio
        equivalent += (int(self.folded_weight.shape[1]) - policy.second_dim) * c2_ratio
        effective = 1.0 - equivalent / float(self.folded_weight.shape[1])
        return {
            "policy": self.policy_name,
            "rows": int(self.rows),
            "c1_ratio": float(c1_ratio),
            "c2_ratio": float(c2_ratio),
            "equivalent_dims": float(equivalent),
            "effective_sparsity": float(effective),
            "effective_sparsity_pct": float(100.0 * effective),
        }

    def _qhead_dense(self, hidden: torch.Tensor) -> torch.Tensor:
        gamma = self.final_gamma.view(*([1] * (hidden.dim() - 1)), -1)
        safe_gamma = torch.where(
            gamma.abs() < 1e-8, torch.ones_like(gamma), gamma
        )
        rotated = torch.matmul(hidden / safe_gamma, self.qhead)
        return F.linear(rotated, self.folded_weight)

    def _staged_chunk(
        self, hidden: torch.Tensor, policy: LMHeadPolicy
    ) -> torch.Tensor:
        safe_gamma = torch.where(
            self.final_gamma.abs() < 1e-8,
            torch.ones_like(self.final_gamma),
            self.final_gamma,
        )
        rotated = torch.mm(hidden / safe_gamma.view(1, -1), self.qhead)
        first = F.linear(
            rotated[:, : policy.first_dim],
            self.folded_weight[:, : policy.first_dim],
        ).float()
        sigma1 = _row_std(first)
        maximum1 = first.amax(dim=-1)
        c1 = first >= (maximum1 - policy.z1 * sigma1).unsqueeze(-1)

        middle = F.linear(
            rotated[:, policy.first_dim : policy.second_dim],
            self.folded_weight[:, policy.first_dim : policy.second_dim],
        ).float()
        cumulative = first + middle
        maximum2, sigma2 = _masked_stats(cumulative, c1)
        c2 = c1 & (
            cumulative >= (maximum2 - policy.z2 * sigma2).unsqueeze(-1)
        )
        tail = F.linear(
            rotated[:, policy.second_dim :],
            self.folded_weight[:, policy.second_dim :],
        ).float()
        logits = (cumulative + tail).masked_fill_(~c2, float("-inf"))
        if getattr(self.inner, "bias", None) is not None:
            logits.add_(self.inner.bias.float())
        self.rows += int(hidden.shape[0])
        self.c1_sum += int(c1.sum().item())
        self.c2_sum += int(c2.sum().item())
        return logits.to(dtype=hidden.dtype)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if self.policy_name == "dense" or not _enabled(self.controller):
            return self.inner(hidden)
        if self.policy_name == "qhead_dense":
            return self._qhead_dense(hidden)
        policy = LMHEAD_POLICIES[self.policy_name]
        shape = hidden.shape[:-1]
        flat = hidden.reshape(-1, hidden.shape[-1])
        chunks = []
        for start in range(0, int(flat.shape[0]), self.chunk_rows):
            chunks.append(
                self._staged_chunk(flat[start : start + self.chunk_rows], policy)
            )
        logits = torch.cat(chunks, dim=0)
        return logits.view(*shape, logits.shape[-1])


def install_lmhead(
    model: nn.Module,
    *,
    qhead_path: Path,
    policy: str,
    controller: SparseController | None = None,
    chunk_rows: int = 32,
    fold_chunk_rows: int = 4096,
) -> StagedSparseQHeadLMHead:
    if isinstance(model.lm_head, StagedSparseQHeadLMHead):
        model.lm_head.set_policy(policy)
        return model.lm_head
    qhead = torch.load(qhead_path, map_location="cpu", weights_only=True)
    wrapper = StagedSparseQHeadLMHead(
        model.lm_head,
        final_gamma=model.model.norm.weight.detach(),
        qhead=qhead,
        policy=policy,
        controller=controller,
        chunk_rows=chunk_rows,
        fold_chunk_rows=fold_chunk_rows,
    )
    model.lm_head = wrapper
    return wrapper


def restore_lmhead(model: nn.Module) -> None:
    if isinstance(model.lm_head, StagedSparseQHeadLMHead):
        model.lm_head = model.lm_head.inner


def validate_artifacts(
    *,
    pack_dir: Path,
    d_dir: Path,
    median_dir: Path,
    qhead_path: Path,
    sparsities: list[int],
    num_layers: int,
    require_qhead: bool = True,
    bucket_dir: Path | None = None,
) -> None:
    required = [Path(pack_dir) / "common" / "embed_tokens.pt"]
    if require_qhead:
        required.append(Path(qhead_path))
    for layer_idx in range(int(num_layers)):
        common = Path(pack_dir) / "common" / f"layer_{layer_idx:02d}"
        required.extend(
            common / name
            for name in (
                "q_proj.pt",
                "k_proj.pt",
                "v_proj.pt",
                "o_proj.pt",
                "gate_proj.pt",
                "up_proj.pt",
                "down_proj.pt",
                "base_transition.pt",
            )
        )
        for projection in ("q_proj", "k_proj", "v_proj", "gate_proj"):
            required.append(lazy_abs.d_path(Path(d_dir), layer_idx, projection))
        for sparsity in sparsities:
            if int(sparsity) <= 0:
                continue
            sparse = Path(pack_dir) / f"s{int(sparsity):02d}" / f"layer_{layer_idx:02d}"
            required.append(sparse / "corr_transition.pt")
            bucket_root = Path(pack_dir) if bucket_dir is None else Path(bucket_dir)
            required.append(
                bucket_root
                / f"s{int(sparsity):02d}"
                / f"layer_{layer_idx:02d}"
                / "bucket_transition.pt"
            )
            required.append(
                lazy_abs.median_path(Path(median_dir), int(sparsity), layer_idx)
            )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing accuracy artifacts: {missing[:12]}")
