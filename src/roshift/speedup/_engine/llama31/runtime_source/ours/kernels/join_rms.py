from __future__ import annotations

import torch

from .extensions import join_rms_op

_OP = None


def preload() -> None:
    global _OP
    if _OP is None:
        _OP = join_rms_op()


def join_rms(
    transition: torch.Tensor,
    mlp: torch.Tensor,
    eps: float,
    *,
    threads: int = 1024,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Join both residual branches and prepare the next attention RMS input."""
    if _OP is None:
        preload()
    return _OP(transition, mlp, float(eps), int(threads))
