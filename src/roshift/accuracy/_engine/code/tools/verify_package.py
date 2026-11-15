"""Verify every immutable payload hash without loading models or initializing CUDA."""
import hashlib
import json
from pathlib import Path


def verify(root):
    root = Path(root).resolve()
    manifest = json.loads((root/'MANIFEST.json').read_text())
    for relative, expected in manifest['files'].items():
        path = (root/relative).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise RuntimeError(f'Missing/unsafe payload: {relative}')
        h = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(8 << 20), b''):
                h.update(block)
        if path.stat().st_size != expected['bytes'] or h.hexdigest() != expected['sha256']:
            raise RuntimeError(f'Payload checksum mismatch: {relative}')
    print(f"Verified {len(manifest['files'])} payload files; no GPU work", flush=True)


if __name__ == '__main__':
    verify(Path(__file__).resolve().parents[1])
