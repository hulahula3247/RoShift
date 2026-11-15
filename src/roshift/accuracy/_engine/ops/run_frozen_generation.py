"""Validate a generation job against its source manifest."""
import os as _roshift_os
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
ROOT = Path(_roshift_os.path.join(_roshift_os.environ['ROSHIFT_ENGINE_ROOT'], ''))

def sha(path):
    out = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            out.update(block)
    return out.hexdigest()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=ROOT / 'ops/generation_source_manifest.json')
    args, remainder = parser.parse_known_args()
    if remainder and remainder[0] == '--':
        remainder = remainder[1:]
    manifest = json.loads(args.manifest.read_text())
    if manifest.get('status') != 'verified':
        raise RuntimeError('Generation source manifest is not verified')
    if sha(ROOT / 'code/MANIFEST.json') != manifest['core_manifest_sha256']:
        raise RuntimeError('Core source manifest does not match the generation manifest')
    for name, expected in manifest['files'].items():
        if sha(name) != expected:
            raise RuntimeError(f'Generation source hash mismatch: {name}')
    observed = hashlib.sha256(json.dumps(manifest['files'], sort_keys=True).encode()).hexdigest()
    if observed != manifest['source_bundle_sha256']:
        raise RuntimeError('Generation source bundle identity mismatch')
    os.execv(sys.executable, [sys.executable, str(ROOT / 'ops/run_generation.py'), *remainder])
if __name__ == '__main__':
    main()
