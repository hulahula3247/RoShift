"""Integrity of the facade and supplemental files outside the frozen core."""
import json
from pathlib import Path
from ..io import digest


def verify_release():
    package=Path(__file__).resolve().parents[1]
    manifest=Path(__file__).with_name('release_sources.json')
    files=json.loads(manifest.read_text())
    for relative,expected in files.items():
        path=(package/relative).resolve()
        if not path.is_relative_to(package) or not path.is_file() or digest(path)!=expected:
            raise RuntimeError('Release file differs: '+relative)
    return digest(manifest)


def profile_identity(request,work,root):
    model=request['model']
    if request['method']=='roshift':
        directory=work/model/'rebuilt'/f"{model}_{request['allocation']}_s{request['sparsity']}"
        names=('profile.json','materialized.json')
        return {name:digest(directory/name) for name in names}
    if request['method']=='larosa':
        directory=root/'baselines'/request['method']/model
        files={str(p.relative_to(directory)):digest(p) for p in directory.rglob('*.json')}
        if not files:
            raise RuntimeError('Baseline preparation receipts are missing')
        return files
    return {'checkpoint':'verified_against_bundled_weight_and_tokenizer_hashes'}
