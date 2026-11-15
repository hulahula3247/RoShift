"""Create a source-only ZIP and a per-file SHA-256 manifest."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    output=args.output.resolve()
    if output.is_relative_to(root):
        raise ValueError('Archive output must be outside the source tree')
    external=json.loads((root/'src/roshift/assets/external_data.json').read_text())
    included_tokens=[name for name in external['files'] if (root/'src/roshift'/name).exists()]
    if included_tokens:
        raise RuntimeError('Raw token payloads must not be distributed: '+', '.join(included_tokens))
    files=[]
    for folder in ('src/roshift','docs','environments','tests','tools','third_party'):
        files.extend(path for path in (root/folder).rglob('*') if path.is_file()
                     and '__pycache__' not in path.parts and path.suffix not in ('.pyc','.pyo'))
    files.extend(root/name for name in ('README.md','REPRODUCIBILITY.md','pyproject.toml','.gitignore','LICENSE'))
    files=sorted(set(files))
    manifest={path.relative_to(root).as_posix():hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
    manifest_path=root/'SHA256SUMS.json'
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for path in [*files,manifest_path]:
            archive.write(path,arcname='roshift_reproduction/'+path.relative_to(root).as_posix())
    with zipfile.ZipFile(output) as archive:
        failure=archive.testzip()
        if failure is not None:
            raise RuntimeError('ZIP CRC failure: '+failure)
        for path,expected in manifest.items():
            if hashlib.sha256(archive.read('roshift_reproduction/'+path)).hexdigest()!=expected:
                raise RuntimeError('ZIP SHA-256 failure: '+path)
    receipt=dict(files=len(manifest),bytes=output.stat().st_size,
                 sha256=hashlib.sha256(output.read_bytes()).hexdigest(),crc='passed',file_hashes='passed')
    output.with_suffix('.verification.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':
    main()
