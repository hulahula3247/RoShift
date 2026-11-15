"""Import the packaged research entrypoints without loading a model or CUDA."""
import os as _roshift_os
import importlib
import json
import sys
from pathlib import Path
root = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(root / 'vendor'), str(root / 'analysis'), str(root)]
manifest = json.loads((root / 'source_closure.json').read_text())
for name in manifest['entrypoints']:
    importlib.import_module(name)
    print('import OK', name, flush=True)
import torch
assert not torch.cuda.is_initialized(), 'Import unexpectedly initialized CUDA'
outside = []
for name, module in list(sys.modules.items()):
    file = vars(module).get('__file__') if module is not None else None
    if file and _roshift_os.path.join(_roshift_os.environ['ROSHIFT_ENGINE_ROOT'], '') in str(file) and ('/.venv/' not in str(file)) and (not Path(file).resolve().is_relative_to(root)):
        outside.append((name, file))
assert not outside, f'Imports leaked into sender workspace: {outside}'
print('Import closure passed; CUDA not initialized', flush=True)
