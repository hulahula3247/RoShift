from __future__ import annotations

import sys
from pathlib import Path

import torch

from .config import OursConfig


ROOT = Path(__file__).resolve().parents[1]
GPT_FAST = ROOT / "gpt-fast"


def load_ours_model(
    checkpoint: str | Path,
    config: OursConfig | None = None,
    *,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
):
    """Create the TEAL model skeleton and fill it directly from the sidecar."""
    if str(GPT_FAST) not in sys.path:
        sys.path.insert(0, str(GPT_FAST))
    from model import Transformer

    from .runtime import install_ours

    checkpoint = Path(checkpoint)
    model_name = checkpoint.name if checkpoint.is_dir() else checkpoint.parent.name
    with torch.device("meta"):
        model = Transformer.from_name(model_name)
    return install_ours(
        model,
        config or OursConfig(),
        device=device,
        dtype=dtype,
    )
