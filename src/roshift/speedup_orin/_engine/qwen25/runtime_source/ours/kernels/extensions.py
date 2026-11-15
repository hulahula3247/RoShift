from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import torch
from torch.utils.cpp_extension import load


CSRC = Path(__file__).with_name("csrc")


def _register_fake(name: str, function) -> None:
    try:
        torch.library.register_fake(name, function)
    except RuntimeError as error:
        if "already has" not in str(error):
            raise


@lru_cache(maxsize=1)
def dynamic_input_op():
    load(
        name="ours_minimal_dynamic_cutoff_v2",
        sources=[str(CSRC / "dynamic_cutoff_gemv.cu")],
        extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"],
        with_cuda=True,
        verbose=False,
    )

    def fake(
        value: torch.Tensor,
        weight: torch.Tensor,
        score_input: torch.Tensor,
        score_scale: torch.Tensor,
        method: int,
        sample_size: int,
        rms_ratio: float,
        max_keep_ratio: float,
        selected_chunk: int,
        output_threads: int,
        load_policy: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del (
            score_input,
            score_scale,
            method,
            sample_size,
            rms_ratio,
            max_keep_ratio,
            selected_chunk,
            output_threads,
            load_policy,
        )
        return (
            value.new_empty((*value.shape[:-1], weight.shape[0])),
            value.new_empty((1,), dtype=torch.float32),
            value.new_empty((32,), dtype=torch.int32),
        )

    _register_fake(
        "ours_minimal_dynamic::dynamic_input",
        fake,
    )
    return torch.ops.ours_minimal_dynamic.dynamic_input.default


@lru_cache(maxsize=1)
def block_sparse_ops():
    load(
        name="ours_minimal_block_sparse_v5",
        sources=[str(CSRC / "block_sparse_gemv.cu")],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )

    def fake_input(
        value: torch.Tensor,
        weight: torch.Tensor,
        score_input: torch.Tensor,
        score_scale: torch.Tensor,
        block_size: int,
        radix_bits: int,
    ) -> torch.Tensor:
        del score_input, score_scale, block_size, radix_bits
        return value.new_empty((*value.shape[:-1], weight.shape[0]))

    def fake_output(
        value: torch.Tensor,
        weight: torch.Tensor,
        output_score: torch.Tensor,
        score_scale: torch.Tensor,
        block_size: int,
        radix_bits: int,
        threads: int,
        load_policy: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del output_score, score_scale, radix_bits, threads, load_policy
        return (
            value.new_empty((*value.shape[:-1], weight.shape[0])),
            value.new_empty(
                (weight.shape[0] // block_size, block_size // 2),
                dtype=torch.uint8,
            ),
        )

    def fake_permuted_input(
        value: torch.Tensor,
        weight: torch.Tensor,
        score_input: torch.Tensor,
        score_scale: torch.Tensor,
        permutation: torch.Tensor,
    ) -> torch.Tensor:
        del score_input, score_scale, permutation
        return value.new_empty((*value.shape[:-1], weight.shape[0]))

    def fake_from_indices(
        value: torch.Tensor,
        weight: torch.Tensor,
        local_indices: torch.Tensor,
        block_size: int,
        load_policy: int,
    ) -> torch.Tensor:
        del local_indices, block_size, load_policy
        return value.new_empty((*value.shape[:-1], weight.shape[0]))

    _register_fake(
        "ours_minimal_block_sparse::pairwise_input",
        fake_input,
    )
    _register_fake(
        "ours_minimal_block_sparse::permuted_pairwise_input",
        fake_permuted_input,
    )
    _register_fake(
        "ours_minimal_block_sparse::output_sparse_up",
        fake_output,
    )
    _register_fake(
        "ours_minimal_block_sparse::input_sparse_down",
        fake_from_indices,
    )
    return (
        torch.ops.ours_minimal_block_sparse.pairwise_input.default,
        torch.ops.ours_minimal_block_sparse.permuted_pairwise_input.default,
        torch.ops.ours_minimal_block_sparse.output_sparse_up.default,
        torch.ops.ours_minimal_block_sparse.input_sparse_down.default,
    )


@lru_cache(maxsize=1)
def join_rms_op():
    load(
        name="qwen_minimal_join_rms_v2",
        sources=[str(CSRC / "join_rms.cu")],
        extra_cuda_cflags=["-O3", "-lineinfo"],
        with_cuda=True,
        verbose=False,
    )

    def fake(
        transition: torch.Tensor,
        mlp: torch.Tensor,
        eps: float,
        block_threads: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del mlp, eps, block_threads
        return (
            transition.new_empty(transition.shape),
            transition.new_empty(transition.shape),
        )

    _register_fake(
        "ours_minimal_join_rms::join_rms",
        fake,
    )
    return torch.ops.ours_minimal_join_rms.join_rms.default


@lru_cache(maxsize=1)
def lm_head_extension():
    return load(
        name="ours_minimal_lm_head_v3",
        sources=[str(CSRC / "lm_head.cu")],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )
