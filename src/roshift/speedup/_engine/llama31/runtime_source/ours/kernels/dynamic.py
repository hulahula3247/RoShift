from __future__ import annotations

import torch

from .extensions import dynamic_input_op


METHOD_SEGMENT128 = 18
_OP = None


def preload() -> None:
    global _OP
    if _OP is None:
        _OP = dynamic_input_op()


def dynamic_input(
    value: torch.Tensor,
    weight: torch.Tensor,
    score_input: torch.Tensor,
    score_scale: torch.Tensor,
    rms_ratio: float,
    *,
    output_threads: int,
    load_policy: int,
) -> torch.Tensor:
    """Dynamic segment-128 cutoff followed by compact sparse GEMV."""
    if _OP is None:
        preload()
    output, _, _ = _OP(
        value,
        weight,
        score_input,
        score_scale,
        METHOD_SEGMENT128,
        256,
        float(rms_ratio),
        0.5,
        128,
        int(output_threads),
        int(load_policy),
    )
    return output
