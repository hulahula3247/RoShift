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
def block_sparse_b256_ops():
    load(
        name="ours_minimal_block_sparse_b256_v1",
        sources=[str(CSRC / "block_sparse_b128_gemv.cu")],
        extra_cuda_cflags=[
            "-O3",
            "--use_fast_math",
            "-DOURS_BLOCK_SIZE=256",
        ],
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
        "ours_minimal_block_sparse_b256::pairwise_input",
        fake_input,
    )
    _register_fake(
        "ours_minimal_block_sparse_b256::permuted_pairwise_input",
        fake_permuted_input,
    )
    _register_fake(
        "ours_minimal_block_sparse_b256::output_sparse_up",
        fake_output,
    )
    _register_fake(
        "ours_minimal_block_sparse_b256::input_sparse_down",
        fake_from_indices,
    )
    return (
        torch.ops.ours_minimal_block_sparse_b256.pairwise_input.default,
        torch.ops.ours_minimal_block_sparse_b256.permuted_pairwise_input.default,
        torch.ops.ours_minimal_block_sparse_b256.output_sparse_up.default,
        torch.ops.ours_minimal_block_sparse_b256.input_sparse_down.default,
    )
