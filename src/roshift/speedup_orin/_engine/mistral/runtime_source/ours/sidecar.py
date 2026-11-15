from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open


class Sidecar:
    """Load the folded weights and K4/lazy-shift tensors from one manifest."""

    def __init__(
        self,
        manifest: str | os.PathLike[str],
        *,
        device: str | torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.path = Path(manifest)
        self.root = self.path.parent
        self.device = torch.device(device)
        self.dtype = dtype
        self.payload = json.loads(self.path.read_text(encoding="utf-8"))
        self._handles: dict[Path, Any] = {}

    def _resolve(self, value: str | os.PathLike[str]) -> Path:
        text = str(value)
        if os.name != "nt" and len(text) >= 3 and text[1] == ":":
            return Path(f"/mnt/{text[0].lower()}/{text[3:].replace(chr(92), '/')}")
        path = Path(text)
        return path if path.is_absolute() else self.root / path

    def _tensor(
        self,
        spec: Any,
        *,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        if isinstance(spec, torch.Tensor):
            tensor = spec.detach()
        elif isinstance(spec, dict) and {"safetensors", "key"} <= spec.keys():
            path = self._resolve(spec["safetensors"])
            handle = self._handles.get(path)
            if handle is None:
                handle = safe_open(str(path), framework="pt", device="cpu")
                self._handles[path] = handle
            tensor = handle.get_tensor(str(spec["key"]))
        elif isinstance(spec, (str, os.PathLike)):
            tensor = torch.load(
                self._resolve(spec), map_location="cpu", weights_only=True
            )
        else:
            tensor = torch.as_tensor(spec)
        if dtype is not None:
            tensor = tensor.to(dtype=dtype)
        return tensor

    def _to_device(
        self,
        spec: Any,
        *,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        return self._tensor(
            spec, dtype=self.dtype if dtype is None else dtype
        ).to(device=self.device)

    def embedding(self) -> torch.Tensor:
        return self._to_device(self.payload["embed_tokens"])

    def projection(self, layer: int, name: str) -> dict[str, torch.Tensor]:
        entry = self.payload["layers"][layer][name]
        result = {"weight": self._to_device(entry["weight"])}
        if "score_scale" in entry:
            result["score_scale"] = self._to_device(
                entry["score_scale"]
            ).flatten().contiguous()
        return result

    def layer_runtime(self, layer: int) -> dict[str, Any]:
        entry = self.payload["layers"][layer]["runtime"]
        result = {
            "base_transition": self._to_device(entry["base_transition"])
            .transpose(-2, -1)
            .contiguous(),
            "corr_transition": self._to_device(entry["corr_transition"])
            .transpose(-2, -1)
            .contiguous(),
            "bucket_transition": self._to_device(entry["bucket_transition"])
            .transpose(-2, -1)
            .contiguous(),
            "alpha_centers": self._to_device(
                entry["alpha_centers"]
            ).flatten().contiguous(),
            "h1_median": self._to_device(entry["h1_median"])
            .flatten()
            .contiguous(),
            "fallback_rel": float(entry.get("fallback_rel", 0.05)),
        }
        return result

    def final_lm_head(self) -> torch.Tensor:
        fusion = self.payload.get("final_output_fusion", {})
        if not fusion.get("enabled", False):
            raise ValueError("The manifest does not contain final-Q fusion.")
        return self._to_device(fusion["lm_head"]).contiguous()

    @property
    def qk_output_layout(self) -> str:
        return str(self.payload.get("qk_output_layout", "runtime"))
