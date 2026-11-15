"""Unmeasured portability repair, NOT part of the historical timing record.

Preserve the row-owned GEMV and stream/event path, but honor actual K_l.
The archived minimal kernel unconditionally read four centers.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def transition_kernel(output, hidden, alpha_ptr, base, correction, buckets,
                      centers, FALLBACK_REL: tl.constexpr, D: tl.constexpr,
                      NUM_CENTERS: tl.constexpr, BLOCK_ROWS: tl.constexpr,
                      BLOCK_K: tl.constexpr):
    rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    alpha = tl.load(alpha_ptr).to(tl.float32)
    best = tl.abs(alpha - tl.load(centers))
    bucket = tl.full((), 0, tl.int64)
    for index in tl.static_range(1, NUM_CENTERS):
        distance = tl.abs(alpha - tl.load(centers + index))
        choose = distance < best
        best = tl.where(choose, distance, best)
        bucket = tl.where(choose, index, bucket)
    fallback = best / tl.maximum(tl.abs(alpha), 1.0e-12) > FALLBACK_REL
    acc = tl.zeros((BLOCK_ROWS,), dtype=tl.float32)
    for start in range(0, D, BLOCK_K):
        cols = start + tl.arange(0, BLOCK_K)
        x = tl.load(hidden + cols, mask=cols < D, other=0.0,
                    eviction_policy="evict_last")
        offsets = rows[:, None] * D + cols[None, :]
        valid = (rows[:, None] < D) & (cols[None, :] < D)
        bucket_offsets = bucket * (D * D) + offsets
        ptr = tl.where(fallback, base + offsets, buckets + bucket_offsets)
        matrix = tl.load(ptr, mask=valid, other=0.0,
                         eviction_policy="evict_first").to(tl.float32)
        corr = tl.load(correction + offsets, mask=valid & fallback, other=0.0,
                       eviction_policy="evict_first").to(tl.float32)
        acc += tl.sum((matrix + alpha * corr) * x.to(tl.float32)[None, :], axis=1)
    tl.store(output + rows, acc, mask=rows < D)


def launch_transition(hidden, alpha, base, correction, buckets, centers, fallback_rel):
    dim = hidden.shape[-1]
    count = centers.numel()
    if hidden.numel() != dim or alpha.numel() != 1:
        raise ValueError("Transition is single-token/batch-1 only")
    if not 1 <= count <= 32 or buckets.shape != (count, dim, dim):
        raise ValueError("Invalid variable-K bucket shape")
    if base.shape != (dim, dim) or correction.shape != (dim, dim):
        raise ValueError("Invalid dense transition shape")
    if not all(t.is_contiguous() for t in (base, correction, buckets, centers)):
        raise ValueError("Runtime transposed matrices must be contiguous")
    output = torch.empty_like(hidden)
    transition_kernel[(triton.cdiv(dim, 4),)](
        output, hidden.contiguous(), alpha.contiguous(), base, correction,
        buckets, centers, FALLBACK_REL=float(fallback_rel), D=dim,
        NUM_CENTERS=count, BLOCK_ROWS=4, BLOCK_K=512, num_warps=1, num_stages=1)
    return output


def install():
    import ours.kernels.transition as transition
    transition._launch_transition = launch_transition

