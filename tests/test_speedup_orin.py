"""Boundaries that keep the optional Orin runner independent of the release."""
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock


class OrinEnvironmentTests(unittest.TestCase):
    def test_child_environment_does_not_reuse_original_runtime_or_caches(self):
        from roshift.speedup_orin import environment

        inherited = {
            'PYTHONPATH': '/unrelated/accuracy/vendor:/unrelated/site-packages',
            'ROSHIFT_ENGINE_ROOT': '/unrelated/accuracy/_engine',
            'ROSHIFT_INPUT_ROOT': '/unrelated/accuracy/inputs',
            'ROSHIFT_SPEED_REQUEST': '/unrelated/speedup/request.json',
            'TORCH_EXTENSIONS_DIR': '/unrelated/cache/extensions',
            'TORCHINDUCTOR_CACHE_DIR': '/unrelated/cache/inductor',
            'TRITON_CACHE_DIR': '/unrelated/cache/triton',
            'CUDA_CACHE_PATH': '/unrelated/cache/driver',
        }
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.dict(os.environ, inherited):
                before = dict(os.environ)
                # Exercise namespace isolation independently of the test host's disk size.
                with mock.patch.object(environment.shutil, 'disk_usage',
                                       return_value=mock.Mock(free=20 * 2**30)):
                    root, child = environment.work_environment(directory)
                self.assertEqual(dict(os.environ), before)
            self.assertEqual(root, Path(directory).resolve())
            cache_root = root / 'cache' / 'speedup_orin'
            for key in ('TORCH_EXTENSIONS_DIR', 'TORCHINDUCTOR_CACHE_DIR',
                        'TRITON_CACHE_DIR', 'CUDA_CACHE_PATH', 'HF_HOME',
                        'XDG_CACHE_HOME', 'PIP_CACHE_DIR'):
                with self.subTest(key=key):
                    self.assertTrue(Path(child[key]).is_relative_to(cache_root))
                    self.assertTrue(Path(child[key]).is_dir())
            for key in ('TMPDIR', 'TEMP', 'TMP'):
                self.assertEqual(Path(child[key]), root / 'tmp' / 'speedup_orin')
            self.assertEqual(child['PYTHONPATH'],
                             str(Path(environment.__file__).resolve().parents[2]))
            for key in ('ROSHIFT_ENGINE_ROOT', 'ROSHIFT_INPUT_ROOT',
                        'ROSHIFT_SPEED_REQUEST'):
                self.assertNotIn(key, child)


class OrinCommandTests(unittest.TestCase):
    def test_orin_default_outputs_cannot_overlap_original_defaults(self):
        from roshift.speedup.execution import parser as original_parser
        from roshift.speedup.execution import make_request as original_request
        from roshift.speedup_orin.execution import parser, make_request

        args = ['run', '--model', 'llama31', '--data-dir', 'artifacts']
        original = original_request(original_parser().parse_args(args))
        orin = make_request(parser().parse_args(args))
        root = Path('artifacts').resolve()
        self.assertEqual(Path(orin['output']), root / 'speedup_orin/results/llama31')
        self.assertNotEqual(orin['output'], original['output'])
        self.assertNotEqual(orin['artifact_root'], original['artifact_root'])

    def test_orin_supports_measurement_but_does_not_dispatch_preparation(self):
        from roshift.speedup_orin.execution import parser

        suffix = ['--model', 'llama31', '--data-dir', 'artifacts']
        for action in ('run', 'validate', 'summarize'):
            self.assertEqual(parser().parse_args([action, *suffix]).action, action)
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as rejected:
                parser().parse_args(['prepare', *suffix])
        self.assertEqual(rejected.exception.code, 2)

    def test_console_dispatch_routes_only_the_explicit_orin_command(self):
        from roshift import cli, dispatch
        from roshift.speedup_orin import execution

        with mock.patch.object(cli, 'main', return_value='original') as original:
            with mock.patch.object(execution, 'main', return_value='orin') as orin:
                for args in (['speedup', 'run', '--model', 'llama31'],
                             ['accuracy', 'evaluate', '--model', 'llama31'],
                             ['--help'], []):
                    with self.subTest(args=args):
                        self.assertEqual(dispatch.main(args), 'original')
                        original.assert_called_with(args)
                        orin.assert_not_called()
                tail = ['run', '--model', 'llama31', '--data-dir', 'artifacts']
                self.assertEqual(dispatch.main(['speedup-orin', *tail]), 'orin')
                orin.assert_called_once_with(tail)


class OrinIntegrityTests(unittest.TestCase):
    def test_independent_manifest_detects_tampering_without_accuracy_manifest(self):
        from roshift.speedup_orin import integrity

        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory) / 'roshift'
            backend = package / 'speedup_orin'
            backend.mkdir(parents=True)
            payload = backend / 'payload.py'
            payload.write_text('value = 1\n')
            manifest = backend / 'release_sources.json'
            manifest.write_text(json.dumps({
                'speedup_orin/payload.py': hashlib.sha256(payload.read_bytes()).hexdigest(),
            }))
            with mock.patch.object(integrity, '__file__', str(backend / 'integrity.py')):
                self.assertEqual(integrity.verify_release(),
                                 hashlib.sha256(manifest.read_bytes()).hexdigest())
                payload.write_text('value = 2\n')
                with self.assertRaises(RuntimeError):
                    integrity.verify_release()


class OrinReportTests(unittest.TestCase):
    def write_pair(self, root, *, backend='orin', sources='a' * 64):
        for method, ms in (('dense', 10), ('ours', 5)):
            runtime = dict(backend='orin' if method == 'dense' else backend,
                           backend_sources_sha256='a' * 64 if method == 'dense' else sources,
                           id='method_once_orin_v1', device='Orin',
                           capability=[8, 7], torch='test', cuda='12.2')
            protocol = dict(prompt_cache_sha256={'x': 'abc'},
                            model_config_sha256='cfg', datasets=['x'],
                            prefill=1024, decode=128, repeats=1,
                            backend=runtime['backend'],
                            backend_sources_sha256=runtime['backend_sources_sha256'])
            path = root / method / ('s0' if method == 'dense' else 's50') / 'result.json'
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(dict(
                status='complete', mean_ms_per_token=ms, protocol=protocol,
                args=dict(method=method, sparsity=.5),
                method_once=dict(protocol=runtime))))

    def test_same_orin_sources_can_pair_raw_measurements(self):
        from roshift.speedup_orin.report import summarize

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_pair(root)
            report = summarize(root)
            ours = next(row for row in report['rows'] if row['method'] == 'roshift')
            self.assertEqual(ours['speedup'], 2)

    def test_other_backend_cannot_supply_a_baseline_pair(self):
        from roshift.speedup_orin.report import summarize

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_pair(root, backend='rtx5090')
            with self.assertRaises(ValueError):
                summarize(root)

    def test_different_orin_sources_cannot_supply_a_baseline_pair(self):
        from roshift.speedup_orin.report import summarize

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_pair(root, sources='b' * 64)
            with self.assertRaises(ValueError):
                summarize(root)


if __name__ == '__main__':
    unittest.main()
