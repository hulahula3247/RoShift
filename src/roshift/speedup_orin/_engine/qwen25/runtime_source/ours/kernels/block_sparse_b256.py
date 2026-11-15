from __future__ import annotations

import torch

from .extensions_b256 import block_sparse_b256_ops


_PAIRWISE_INPUT = None
_PERMUTED_INPUT = None
_OUTPUT_UP = None
_DOWN = None


def preload() -> None:
    global _PAIRWISE_INPUT, _PERMUTED_INPUT, _OUTPUT_UP, _DOWN
    if _PAIRWISE_INPUT is None:
        (
            _PAIRWISE_INPUT,
            _PERMUTED_INPUT,
            _OUTPUT_UP,
            _DOWN,
        ) = block_sparse_b256_ops()


def pairwise_input_b256(
    value: torch.Tensor,
    weight: torch.Tensor,
    score_input: torch.Tensor,
    score_scale: torch.Tensor,
) -> torch.Tensor:
    """B256/K128 pairwise selector plus input-sparse O GEMV."""
    if _PAIRWISE_INPUT is None:
        preload()
    return _PAIRWISE_INPUT(value, weight, score_input, score_scale, 256, 16)


def permuted_pairwise_input_b256(
    value: torch.Tensor,
    weight: torch.Tensor,
    score_input: torch.Tensor,
    score_scale: torch.Tensor,
    permutation: torch.Tensor,
) -> torch.Tensor:
    """Random-permuted B256/K128 selector plus QKV/Gate GEMV."""
    if _PERMUTED_INPUT is None:
        preload()
    return _PERMUTED_INPUT(
        value,
        weight,
        score_input,
        score_scale,
        permutation,
    )


def output_sparse_up_b256(
    value: torch.Tensor,
    weight: torch.Tensor,
    centered_gate: torch.Tensor,
    score_scale: torch.Tensor,
    *,
    threads: int = 256,
    load_policy: int = 2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """B256/K128 output selector fused with Up."""
    if _OUTPUT_UP is None:
        preload()
    return _OUTPUT_UP(
        value,
        weight,
        centered_gate,
        score_scale,
        256,
        16,
        int(threads),
        int(load_policy),
    )


def input_sparse_b256(
    value: torch.Tensor,
    weight: torch.Tensor,
    indices: torch.Tensor,
    *,
    load_policy: int = 2,
) -> torch.Tensor:
    """Sparse Down GEMV using B256/K128 support from Up."""
    if _DOWN is None:
        preload()
    return _DOWN(value, weight, indices, 256, int(load_policy))
