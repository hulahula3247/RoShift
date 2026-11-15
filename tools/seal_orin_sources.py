"""Seal only Orin sources; never rewrite the original accuracy manifest."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]/'src/roshift'
target = root/'speedup_orin/release_sources.json'
inputs = list((root/'speedup_orin').rglob('*')) + list((root/'assets').rglob('*'))
inputs += [root/name for name in ('__init__.py', 'catalog.py', 'data.py', 'io.py', 'dispatch.py')]
files = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
         for p in sorted(inputs) if p.is_file() and p != target
         and '__pycache__' not in p.parts and p.suffix not in ('.pyc', '.pyo')}
target.write_text(json.dumps(files, indent=2)+'\n', encoding='utf-8')
print(json.dumps({'orin_files': len(files)}))
