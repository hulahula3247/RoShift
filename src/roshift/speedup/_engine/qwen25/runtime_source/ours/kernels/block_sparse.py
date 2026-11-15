from __future__ import annotations

import torch

from .extensions import block_sparse_ops

_INPUT_OP = None
_PERMUTED_INPUT_OP = None
_OUTPUT_OP = None
_DOWN_OP = None


def preload() -> None:
    global _INPUT_OP, _PERMUTED_INPUT_OP, _OUTPUT_OP, _DOWN_OP
    if _INPUT_OP is None:
        (
            _INPUT_OP,
            _PERMUTED_INPUT_OP,
            _OUTPUT_OP,
            _DOWN_OP,
        ) = block_sparse_ops()


def pairwise_input(
    value: torch.Tensor,
    weight: torch.Tensor,
    score_input: torch.Tensor,
    score_scale: torch.Tensor,
) -> torch.Tensor:
    """B64 pairwise-rank selector fused with input-sparse GEMV."""
    if _INPUT_OP is None:
        preload()
    return _INPUT_OP(value, weight, score_input, score_scale, 64, 16)


def permuted_pairwise_input(
    value: torch.Tensor,
    weight: torch.Tensor,
    score_input: torch.Tensor,
    score_scale: torch.Tensor,
    permutation: torch.Tensor,
) -> torch.Tensor:
    """Random-permuted B64/K32 selector plus input-sparse GEMV."""
    if _PERMUTED_INPUT_OP is None:
        preload()
    return _PERMUTED_INPUT_OP(
        value,
        weight,
        score_input,
        score_scale,
        permutation,
    )


def output_sparse_up(
    value: torch.Tensor,
    weight: torch.Tensor,
    centered_gate: torch.Tensor,
    score_scale: torch.Tensor,
    *,
    threads: int = 256,
    load_policy: int = 2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """B64 output selector fused with Up; returns support reused by Down."""
    if _OUTPUT_OP is None:
        preload()
    return _OUTPUT_OP(
        value,
        weight,
        centered_gate,
        score_scale,
        64,
        16,
        int(threads),
        int(load_policy),
    )


def input_sparse(
    value: torch.Tensor,
    weight: torch.Tensor,
    indices: torch.Tensor,
    *,
    load_policy: int = 2,
) -> torch.Tensor:
    """Sparse Down GEMV using the B64 support produced by Up."""
    if _DOWN_OP is None:
        preload()
    return _DOWN_OP(value, weight, indices, 64, int(load_policy))
