from __future__ import annotations

from typing import Tuple

import torch
import triton
import triton.language as tl

from kernels.compile_wrapper import BaseKernel

from .join_rms import join_rms


_STREAMS: dict[tuple[int, int], torch.cuda.Stream] = {}
_READY: dict[tuple[int, int, int], torch.cuda.Event] = {}
_DONE: dict[tuple[int, int, int], torch.cuda.Event] = {}


def _state(reference: torch.Tensor, layer: int, priority: int):
    device = reference.device.index
    if device is None:
        device = torch.cuda.current_device()
    stream_key = (int(device), int(priority))
    event_key = (*stream_key, int(layer))
    with torch.cuda.device(device):
        if stream_key not in _STREAMS:
            _STREAMS[stream_key] = torch.cuda.Stream(
                device=device, priority=int(priority)
            )
        if event_key not in _READY:
            _READY[event_key] = torch.cuda.Event(
                blocking=False, enable_timing=False
            )
            _DONE[event_key] = torch.cuda.Event(
                blocking=False, enable_timing=False
            )
    return _STREAMS[stream_key], _READY[event_key], _DONE[event_key]


@triton.jit
def _transition_kernel(
    output,
    hidden,
    alpha_ptr,
    base,
    correction,
    buckets,
    centers,
    FALLBACK_REL: tl.constexpr,
    D: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    alpha = tl.load(alpha_ptr).to(tl.float32)

    c0 = tl.load(centers)
    c1 = tl.load(centers + 1)
    c2 = tl.load(centers + 2)
    c3 = tl.load(centers + 3)
    best = tl.abs(alpha - c0)
    bucket = tl.full((), 0, tl.int64)
    choose = tl.abs(alpha - c1) < best
    best = tl.where(choose, tl.abs(alpha - c1), best)
    bucket = tl.where(choose, 1, bucket)
    choose = tl.abs(alpha - c2) < best
    best = tl.where(choose, tl.abs(alpha - c2), best)
    bucket = tl.where(choose, 2, bucket)
    choose = tl.abs(alpha - c3) < best
    best = tl.where(choose, tl.abs(alpha - c3), best)
    bucket = tl.where(choose, 3, bucket)
    fallback = best / tl.maximum(tl.abs(alpha), 1.0e-12) > FALLBACK_REL

    acc = tl.zeros((BLOCK_ROWS,), dtype=tl.float32)
    for start in range(0, D, BLOCK_K):
        cols = start + tl.arange(0, BLOCK_K)
        x = tl.load(
            hidden + cols,
            mask=cols < D,
            other=0.0,
            eviction_policy="evict_last",
        )
        offsets = rows[:, None] * D + cols[None, :]
        valid = (rows[:, None] < D) & (cols[None, :] < D)
        bucket_offsets = bucket * (D * D) + offsets
        matrix_ptr = tl.where(fallback, base + offsets, buckets + bucket_offsets)
        matrix = tl.load(
            matrix_ptr,
            mask=valid,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        corr = tl.load(
            correction + offsets,
            mask=valid & fallback,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        acc += tl.sum(
            (matrix + alpha * corr) * x.to(tl.float32)[None, :],
            axis=1,
        )
    tl.store(output + rows, acc, mask=rows < D)


def _launch_transition(
    hidden: torch.Tensor,
    alpha: torch.Tensor,
    base: torch.Tensor,
    correction: torch.Tensor,
    buckets: torch.Tensor,
    centers: torch.Tensor,
    fallback_rel: float,
) -> torch.Tensor:
    dimension = hidden.shape[-1]
    output = torch.empty_like(hidden)
    _transition_kernel[(triton.cdiv(dimension, 4),)](
        output,
        hidden.contiguous(),
        alpha.contiguous(),
        base,
        correction,
        buckets,
        centers,
        FALLBACK_REL=float(fallback_rel),
        D=dimension,
        BLOCK_ROWS=4,
        BLOCK_K=512,
        num_warps=1,
        num_stages=1,
    )
    return output


def _async_launch(
    hidden: torch.Tensor,
    alpha: torch.Tensor,
    base: torch.Tensor,
    correction: torch.Tensor,
    buckets: torch.Tensor,
    centers: torch.Tensor,
    fallback_rel: float,
    layer: int,
    priority: int,
) -> torch.Tensor:
    stream, ready, done = _state(hidden, layer, priority)
    current = torch.cuda.current_stream(hidden.device)
    ready.record(current)
    stream.wait_event(ready)
    with torch.cuda.stream(stream):
        output = _launch_transition(
            hidden,
            alpha,
            base,
            correction,
            buckets,
            centers,
            fallback_rel,
        )
        done.record(stream)
    return output


class _TransitionOp(BaseKernel):
    def meta(
        self,
        hidden: torch.Tensor,
        alpha: torch.Tensor,
        base: torch.Tensor,
        correction: torch.Tensor,
        buckets: torch.Tensor,
        centers: torch.Tensor,
        fallback_rel: float,
        layer: int,
        priority: int,
    ) -> torch.Tensor:
        return hidden.new_empty(hidden.shape)

    def forward(
        self,
        hidden: torch.Tensor,
        alpha: torch.Tensor,
        base: torch.Tensor,
        correction: torch.Tensor,
        buckets: torch.Tensor,
        centers: torch.Tensor,
        fallback_rel: float,
        layer: int,
        priority: int,
    ) -> torch.Tensor:
        return _async_launch(
            hidden,
            alpha,
            base,
            correction,
            buckets,
            centers,
            fallback_rel,
            layer,
            priority,
        )


class _JoinOp(BaseKernel):
    def meta(
        self,
        transition: torch.Tensor,
        mlp: torch.Tensor,
        layer: int,
        priority: int,
    ) -> torch.Tensor:
        return transition.new_empty(transition.shape)

    def forward(
        self,
        transition: torch.Tensor,
        mlp: torch.Tensor,
        layer: int,
        priority: int,
    ) -> torch.Tensor:
        _, _, done = _state(transition, layer, priority)
        torch.cuda.current_stream(transition.device).wait_event(done)
        return transition + mlp


class _JoinRMSOp(BaseKernel):
    def meta(
        self,
        transition: torch.Tensor,
        mlp: torch.Tensor,
        eps: float,
        layer: int,
        priority: int,
        threads: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            transition.new_empty(transition.shape),
            transition.new_empty(transition.shape),
        )

    def forward(
        self,
        transition: torch.Tensor,
        mlp: torch.Tensor,
        eps: float,
        layer: int,
        priority: int,
        threads: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        _, _, done = _state(transition, layer, priority)
        torch.cuda.current_stream(transition.device).wait_event(done)
        return join_rms(transition, mlp, eps, threads=threads)


class AsyncTransition:
    """Persistent auxiliary-stream K4 transition and its two join operations."""

    def __init__(self) -> None:
        self._launch = _TransitionOp.initialize(
            "ours_minimal_transition", "cuda"
        ).operator(True)
        self._join = _JoinOp.initialize(
            "ours_minimal_transition_join", "cuda"
        ).operator(True)
        self._join_rms = _JoinRMSOp.initialize(
            "ours_minimal_transition_join_rms", "cuda"
        ).operator(True)

    def launch(
        self,
        hidden: torch.Tensor,
        alpha: torch.Tensor,
        runtime: dict,
        layer: int,
        priority: int,
    ) -> torch.Tensor:
        return self._launch(
            hidden,
            alpha,
            runtime["base_transition"],
            runtime["corr_transition"],
            runtime["bucket_transition"],
            runtime["alpha_centers"],
            float(runtime["fallback_rel"]),
            int(layer),
            int(priority),
        )

    def join(
        self,
        transition: torch.Tensor,
        mlp: torch.Tensor,
        layer: int,
        priority: int,
    ) -> torch.Tensor:
        return self._join(
            transition, mlp, int(layer), int(priority)
        )

    def join_next_rms(
        self,
        transition: torch.Tensor,
        mlp: torch.Tensor,
        eps: float,
        layer: int,
        priority: int,
        threads: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._join_rms(
            transition,
            mlp,
            float(eps),
            int(layer),
            int(priority),
            int(threads),
        )
