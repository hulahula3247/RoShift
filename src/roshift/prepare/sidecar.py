"""Portable, sharded tensor container without pickle or original machine paths.

This is an output container, not the completed model-specific folding pipeline.
One shard is written at a time to avoid a second full model copy in RAM.
"""

from pathlib import Path
import re

from ..io import atomic_json, digest, within


def write_sidecar(directory, tensors, metadata):
    from safetensors.torch import save_file
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if any(directory.iterdir()):
        raise FileExistsError("Sidecar output must be empty; never overwrite measured artifacts")
    index = {}
    for name, tensor in tensors:
        if not re.fullmatch(r"[a-zA-Z0-9_.-]+", name) or name in index:
            raise ValueError("Invalid or duplicate tensor name")
        relative = f"weights/{name}.safetensors"
        path = within(directory, relative)
        path.parent.mkdir(exist_ok=True)
        save_file({"value": tensor.detach().cpu().contiguous()}, str(path))
        index[name] = {"file": relative, "sha256": digest(path), "shape": list(tensor.shape), "dtype": str(tensor.dtype)}
    if not index:
        raise ValueError("No sidecar tensors supplied")
    manifest = {"schema": 1, "complete": True, "metadata": metadata, "tensors": index}
    atomic_json(directory / "sidecar.json", manifest)
    return manifest


def verify_sidecar(directory):
    import json
    directory = Path(directory)
    payload = json.loads((directory / "sidecar.json").read_text(encoding="utf-8"))
    if payload.get("schema") != 1 or not payload.get("complete") or not payload.get("tensors"):
        return False
    return all(within(directory, row["file"]).is_file() and digest(within(directory, row["file"])) == row["sha256"]
               for row in payload["tensors"].values())
