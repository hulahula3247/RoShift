"""Bounded paths, atomic metadata, and content-addressed stage receipts."""

import hashlib
import json
import os
import tempfile
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def within(root, relative):
    root = Path(root).resolve()
    relative = Path(relative)
    if relative.is_absolute() or relative.drive or ".." in relative.parts:
        raise ValueError("Expected a bounded relative path")
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Path escapes its root")
    return path


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def completed_stage(root, key, inputs):
    path = within(root, key + ".json")
    if not path.is_file():
        return False
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if receipt.get("status") != "complete" or receipt.get("inputs") != fingerprint(inputs):
        return False
    outputs = receipt.get("outputs", {})
    if not outputs:
        return False
    return all(within(root, p).is_file() and digest(within(root, p)) == h for p, h in outputs.items())


def finish_stage(root, key, inputs, outputs):
    if not outputs:
        raise ValueError("A complete stage needs real output files")
    hashes = {p: digest(within(root, p)) for p in outputs}
    atomic_json(within(root, key + ".json"), {"status": "complete", "inputs": fingerprint(inputs), "outputs": hashes})
