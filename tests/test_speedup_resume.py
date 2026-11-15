"""Exercise the actual runner's resume guard without loading a GPU model."""
import ast
import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from roshift.speedup_orin.runtime import ENGINE


class DenseResume(unittest.TestCase):
    def test_dense_output_does_not_change_input_protocol(self):
        source = ENGINE/'run_method.py'
        tree = ast.parse(source.read_text())
        main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
        cutoff = next(i for i, n in enumerate(main.body) if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == 'started' for t in n.targets))
        main.body = main.body[:cutoff] + [ast.Return(ast.Name(id='state', ctx=ast.Load()))]
        tree.body = [n for n in tree.body if not isinstance(n, ast.If)]
        ast.fix_missing_locations(tree)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            prompts = root/'prompts'
            prompts.mkdir()
            (prompts/'prompts.pt').write_bytes(b'fixed prompt identity')
            common = types.SimpleNamespace(bench=types.SimpleNamespace(DATASETS=('wikitext2',)),
                PROMPTS=prompts, RUNTIME=ENGINE/'llama31/runtime_source', threshold_path=lambda *a: None)
            support = types.ModuleType('support')
            for k, v in dict(DATASETS=('wikitext2',), MODELS=('llama31',), PERCENTS=(50,),
                ResidentModel=lambda *a: types.SimpleNamespace(), dump=lambda *a: None,
                open_runtime=lambda *a: (common, ENGINE/'llama31', root), preflight=lambda *a: [],
                sha=lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()).items():
                setattr(support, k, v)
            torch = types.ModuleType('torch')
            torch.__version__ = 'test'
            torch.version = types.SimpleNamespace(cuda='test')
            torch.cuda = types.SimpleNamespace(get_device_name=lambda: 'test',
                                               get_device_capability=lambda: (8, 7))
            argv = ['run_method', '--model', 'llama31', '--method', 'dense',
                    '--output', str(root/'output'), '--percents', '50']
            namespace = {'__file__': str(source), '__name__': 'resume_test'}
            with patch.dict(sys.modules, {'support': support, 'torch': torch,
                    'worker': types.ModuleType('worker')}), patch.object(sys, 'argv', argv), \
                    contextlib.redirect_stdout(io.StringIO()):
                exec(compile(tree, str(source), 'exec'), namespace)
                first = namespace['main']()
                output = root/'output/dense'
                output.mkdir(parents=True)
                (output/'dense_prefill_logits.pt').write_bytes(b'Dense creates its reference')
                for status, completed in [('interrupted_or_failed', []), ('complete', [0])]:
                    first.update(status=status, completed=completed)
                    (output/'state.json').write_text(json.dumps(first))
                    resumed = namespace['main']()
                    self.assertEqual(first['protocol'], resumed['protocol'])
                    self.assertEqual(completed, resumed['completed'])
                # A genuinely different measurement setting must still fail.
                argv.extend(['--repeats', '1'])
                with self.assertRaisesRegex(ValueError, 'different protocol/source'):
                    namespace['main']()
