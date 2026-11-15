from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from kernels.compile_wrapper import BaseKernel

from .extensions import lm_head_extension


class _LocalExactOp(BaseKernel):
    def meta(
        self,
        hidden: torch.Tensor,
        weight: torch.Tensor,
        prefix_dim: int,
        scaled_margin: float,
        prefix_warps: int,
        tail_warps: int,
        rows_per_block: int,
        prefix_rows_per_warp: int,
        count_candidates: bool,
        packed_weight: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            hidden.new_empty((*hidden.shape[:-1], weight.shape[0])),
            hidden.new_empty((1,), dtype=torch.int32),
            hidden.new_empty((5,), dtype=torch.float32),
        )

    def forward(
        self,
        hidden: torch.Tensor,
        weight: torch.Tensor,
        prefix_dim: int,
        scaled_margin: float,
        prefix_warps: int,
        tail_warps: int,
        rows_per_block: int,
        prefix_rows_per_warp: int,
        count_candidates: bool,
        packed_weight: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return tuple(
            lm_head_extension().early_exit_lmhead_local(
                hidden,
                weight,
                prefix_dim,
                scaled_margin,
                prefix_warps,
                tail_warps,
                rows_per_block,
                prefix_rows_per_warp,
                count_candidates,
                packed_weight,
            )
        )


class _ThreeGroupOp(BaseKernel):
    def meta(
        self,
        hidden: torch.Tensor,
        weight: torch.Tensor,
        prefix_dim: int,
        middle_dim: int,
        first_scaled_margin: float,
        second_scaled_margin: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del prefix_dim, middle_dim, first_scaled_margin, second_scaled_margin
        return (
            hidden.new_empty((*hidden.shape[:-1], weight.shape[0])),
            hidden.new_empty((2,), dtype=torch.int32),
            hidden.new_empty((10,), dtype=torch.float32),
        )

    def forward(
        self,
        hidden: torch.Tensor,
        weight: torch.Tensor,
        prefix_dim: int,
        middle_dim: int,
        first_scaled_margin: float,
        second_scaled_margin: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return tuple(
            lm_head_extension().early_exit_lmhead_three_group(
                hidden,
                weight,
                prefix_dim,
                middle_dim,
                first_scaled_margin,
                second_scaled_margin,
            )
        )


class EarlyExitLMHead(nn.Module):
    """Dense prefill head and conservative local-exact decode head."""

    def __init__(
        self,
        weight: torch.Tensor,
        *,
        prefix_dim: int,
        scaled_margin: float,
        prefix_warps: int,
        tail_warps: int,
        rows_per_block: int,
        prefix_rows_per_warp: int,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(weight.contiguous(), requires_grad=False)
        self.prefix_dim = int(prefix_dim)
        self.scaled_margin = float(scaled_margin)
        self.prefix_warps = int(prefix_warps)
        self.tail_warps = int(tail_warps)
        self.rows_per_block = int(rows_per_block)
        self.prefix_rows_per_warp = int(prefix_rows_per_warp)
        lm_head_extension()
        self._decode = _LocalExactOp.initialize(
            "ours_minimal_lm_head_local", "cuda"
        ).operator(True)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.shape[1] != 1:
            return F.linear(hidden, self.weight)
        logits, _, _ = self._decode(
            hidden,
            self.weight,
            self.prefix_dim,
            self.scaled_margin,
            self.prefix_warps,
            self.tail_warps,
            self.rows_per_block,
            self.prefix_rows_per_warp,
            False,
            False,
        )
        return logits


class ThreeGroupEarlyExitLMHead(nn.Module):
    """512/1024/2560 staged decode head; dense prefill is unchanged."""

    def __init__(
        self,
        weight: torch.Tensor,
        *,
        prefix_dim: int,
        middle_dim: int,
        first_scaled_margin: float,
        second_scaled_margin: float,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(weight.contiguous(), requires_grad=False)
        self.prefix_dim = int(prefix_dim)
        self.middle_dim = int(middle_dim)
        self.first_scaled_margin = float(first_scaled_margin)
        self.second_scaled_margin = float(second_scaled_margin)
        lm_head_extension()
        self._decode = _ThreeGroupOp.initialize(
            "ours_minimal_lm_head_three_group", "cuda"
        ).operator(True)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.shape[1] != 1:
            return F.linear(hidden, self.weight)
        logits, _, _ = self._decode(
            hidden,
            self.weight,
            self.prefix_dim,
            self.middle_dim,
            self.first_scaled_margin,
            self.second_scaled_margin,
        )
        return logits


class DenseFoldedLMHead(nn.Module):
    """Dense decode baseline using the same final-Q-folded weight."""

    def __init__(self, weight: torch.Tensor) -> None:
        super().__init__()
        self.weight = nn.Parameter(weight.contiguous(), requires_grad=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return F.linear(hidden, self.weight)
