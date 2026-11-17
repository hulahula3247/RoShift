"""Regression tests for the reported first-run and reuse failures."""
import ast
from contextlib import redirect_stdout
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from roshift.accuracy.execution import ENGINE, work_environment


ROOT = Path(__file__).resolve().parents[1]
CORE = ENGINE/'code'
spec = importlib.util.spec_from_file_location('repair_prepare_remote', CORE/'tools/prepare_remote.py')
prepare = importlib.util.module_from_spec(spec)
with patch.dict(sys.modules, {'fcntl': SimpleNamespace()} if os.name == 'nt' else {}):
    spec.loader.exec_module(prepare)


class ReproductionRepairs(unittest.TestCase):
    def test_matplotlib_is_explicitly_pinned(self):
        self.assertIn('matplotlib==3.10.8', (ROOT/'environments/accuracy.txt').read_text())

    def test_failed_venv_cannot_run_activation(self):
        text = (ROOT/'docs/ACCURACY.md').read_text(encoding='utf-8')
        self.assertIn('python3.12-venv', text)
        self.assertIn('python3.12 -m venv "$ROSHIFT_DATA/env" &&\nsource ', text)

    def test_loading_guard_default_and_explicit_override(self):
        with tempfile.TemporaryDirectory() as folder:
            for explicit in (None, '0', '1'):
                env = {'HF_HOME': folder}
                if explicit is not None:
                    env['ROSHIFT_SKIP_ALLOCATOR_WARMUP'] = explicit
                with patch.dict(os.environ, env, clear=True), patch(
                        'roshift.accuracy.execution.shutil.disk_usage',
                        return_value=SimpleNamespace(free=20 << 30)):
                    _, actual = work_environment(folder)
                self.assertEqual(actual['ROSHIFT_SKIP_ALLOCATOR_WARMUP'], explicit or '1')
                self.assertNotIn('PYTORCH_CUDA_ALLOC_CONF', actual)
                self.assertNotIn('PYTORCH_ALLOC_CONF', actual)

    def test_guard_skips_reservation_before_accessing_cuda(self):
        tree = ast.parse((CORE/'vendor/transformers/modeling_utils.py').read_text(encoding='utf-8'))
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                        and n.name == 'caching_allocator_warmup')
        module = ast.Module(body=[ast.ImportFrom(module='__future__',
            names=[ast.alias(name='annotations')], level=0), function], type_ignores=[])
        namespace = {'os': os}
        exec(compile(ast.fix_missing_locations(module), '<loader guard>', 'exec'), namespace)
        # The rest of the real function requires globals deliberately absent here.
        with patch.dict(os.environ, {'ROSHIFT_SKIP_ALLOCATOR_WARMUP': '1'}):
            namespace['caching_allocator_warmup'](None, {'weight': 'cuda:0'}, None)
        with patch.dict(os.environ, {'ROSHIFT_SKIP_ALLOCATOR_WARMUP': '0'}):
            with self.assertRaises(NameError):
                namespace['caching_allocator_warmup'](None, {'weight': 'cuda:0'}, None)

    def test_stage_guards_distinguish_reuse_and_pending_work(self):
        with tempfile.TemporaryDirectory() as folder:
            work = Path(folder)
            identity = {'model': 'llama31'}
            self.assertEqual(prepare.scratch_requirement(work, 'dense_statistics', identity)[0], 80)
            for label, name in (('dense_statistics', 'dense_preparation_state.json'),
                                ('common_pack', 'common_ready.json')):
                prepare.dump(work/'llama31/common'/name, {'status': 'complete'})
                self.assertEqual(prepare.scratch_requirement(work, label, identity)[0], 10)
                with patch.object(prepare.shutil, 'disk_usage',
                                  return_value=SimpleNamespace(free=20 << 30)), redirect_stdout(io.StringIO()):
                    prepare.check_scratch(work, label, identity)
            profile = 'llama31_uniform_s50'
            self.assertTrue(25 <= prepare.scratch_requirement(work, profile, identity)[0] < 80)
            prepare.dump(work/'llama31/rebuilt'/profile/'materialized.json', {'status': 'complete'})
            self.assertEqual(prepare.scratch_requirement(work, profile, identity)[0], 10)
            self.assertEqual(prepare.scratch_requirement(work, 'head_basis', identity)[0], 10)

    def test_low_space_still_rejected_with_stage_and_requirement(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(prepare.shutil, 'disk_usage', return_value=SimpleNamespace(free=20 << 30)):
                with self.assertRaisesRegex(RuntimeError, 'dense_statistics: 20.0 GiB free, 80 GiB required'):
                    prepare.check_scratch(Path(folder), 'dense_statistics', {'model': 'llama31'})
            with patch.object(prepare.shutil, 'disk_usage', return_value=SimpleNamespace(free=9 << 30)):
                with self.assertRaises(RuntimeError):
                    prepare.check_scratch(Path(folder), 'head_basis', {'model': 'llama31'})

    def test_complete_markers_do_not_skip_subprocess_verification(self):
        with tempfile.TemporaryDirectory() as folder:
            work = Path(folder)
            prepare.dump(work/'regeneration_selection.json', {'model': 'llama31', 'selection': {},
                         'run_sha256': 'fixture', 'q_head_deferred': False})
            prepare.dump(work/'llama31/common/common_ready.json', {'status': 'complete'})
            child = SimpleNamespace(wait=lambda: 0)
            with patch.object(prepare.shutil, 'disk_usage', return_value=SimpleNamespace(free=20 << 30)), \
                 patch.object(prepare.subprocess, 'Popen', return_value=child) as start, \
                 redirect_stdout(io.StringIO()):
                prepare._run_script(work, 'common_pack', 'fixture.py', [])
            start.assert_called_once()

    def test_identity_matches_rc4_rc5_but_integrity_manifest_differs(self):
        expected = 'c91494751ba332da04db3caa81d6dea4f1fb02eeb2d1c97e5a36bf11a5d695f9'
        self.assertEqual(prepare.artifact_manifest_sha(), expected)
        self.assertNotEqual(prepare.sha(CORE/'MANIFEST.json'), expected)
        legacy = json.loads((CORE/'ARTIFACT_MANIFEST_RC5.json').read_text())['files']
        current = json.loads((CORE/'MANIFEST.json').read_text())['files']
        changed = {name for name in legacy if legacy[name] != current[name]}
        self.assertEqual(changed, {
            'tools/prepare_remote.py', 'tools/materialize_profile.py',
            'vendor/transformers/modeling_utils.py',
            'analysis/run_llama3_larosa_teal_lmeval_suite_20260604.py',
            'context/source_reference/official_method_eval_20260827/official_common.py',
            'context/source_reference/analysis/run_teal_resume_20260912.py',
            'context/source_reference/analysis/run_teal_official_suite_20260910.py',
            'context/source_reference/larosa/README.md',
            'vendor/lm_eval/tasks/README.md',
            'tools/build_package.py', 'tools/run_likelihood.py',
        })

    def test_modified_reference_cannot_reuse_old_artifact_identity(self):
        read = prepare.read
        def changed(path):
            result = read(path)
            if Path(path).name == 'MANIFEST.json':
                name = 'context/source_reference/analysis/run_teal_resume_20260912.py'
                result['files'][name]['sha256'] = 'changed'
            return result
        with patch.object(prepare, 'read', side_effect=changed):
            with self.assertRaisesRegex(RuntimeError, 'compatibility check failed'):
                prepare.artifact_manifest_sha()

    def test_new_numerical_change_cannot_claim_old_artifact_identity(self):
        read = prepare.read
        def changed(path):
            result = read(path)
            if Path(path).name == 'MANIFEST.json':
                name = 'analysis/rental_prepare_dense_20260916.py'
                result['files'][name]['sha256'] = 'changed'
            return result
        with patch.object(prepare, 'read', side_effect=changed):
            with self.assertRaisesRegex(RuntimeError, 'compatibility check failed'):
                prepare.artifact_manifest_sha()

    def test_profile_fingerprint_is_unchanged(self):
        data = {'id': 'fixture', 'alpha': [[.25, .25, .125, .125, .125, .125]]}
        prefix = json.dumps(data, sort_keys=True)
        old = hashlib.sha256((prefix + prepare.sha(CORE/'ARTIFACT_MANIFEST_RC5.json') + 'dense').encode()).hexdigest()
        new = hashlib.sha256((prefix + prepare.artifact_manifest_sha() + 'dense').encode()).hexdigest()
        self.assertEqual(old, new)


if __name__ == '__main__':
    unittest.main()
