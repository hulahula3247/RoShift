"""Refresh the source integrity manifest."""
import hashlib
import json
from pathlib import Path


root=Path(__file__).resolve().parents[1]/'src/roshift'
target=root/'accuracy/release_sources.json'
core=root/'accuracy/_engine/code'
covered=json.loads((core/'MANIFEST.json').read_text())['files']
files={}
for path in sorted(root.rglob('*')):
    if not path.is_file() or path==target or '__pycache__' in path.parts or path.suffix=='.pyc':
        continue
    if path.is_relative_to(core) and str(path.relative_to(core)).replace('\\','/') in covered:
        continue
    files[path.relative_to(root).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
target.write_text(json.dumps(files,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'supplemental_files':len(files)}))
