"""CPU/stdlib-only package tests. Never download models or initialize CUDA."""
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import prepare_remote as tool


class RegenerationTests(unittest.TestCase):
    @unittest.skipUnless(os.name == 'posix', 'Legacy queue uses Linux file locks')
    def test_actual_search_queue_counts_and_no_historical_tensor_dependency(self):
        tool.configure_environment()
        data = tool.read(tool.KIT/'regeneration.json')
        with tempfile.TemporaryDirectory() as tmp:
            total_alloc = total_q = 0
            for key in data['models']:
                work = Path(tmp)/key
                for profile in data['profiles']:
                    if profile['model'] == key:
                        tool.dump(work/key/'rebuilt'/profile['id']/'state.json',
                                  {'status':'complete', 'test_fixture_only':True})
                suite = tool.configure_search(key, work, Path('/models')/key)
                suite.prepare_plan()
                plan = tool.read(work/'plan.json')
                expected = tool.plan(key, work)
                self.assertEqual(plan['allocation_missing'], expected['missing_allocation_searches'])
                self.assertEqual(plan['q_missing'], expected['missing_q_searches'])
                total_alloc += plan['allocation_missing']
                total_q += plan['q_missing']
                suite.publish_status(dict(status='running', errors=[]))
                rows = tool.read(work/'progress.json')['rotation']
                self.assertFalse(any(r['status']=='historical_elsewhere_tensor_not_bundled' for r in rows))
                self.assertEqual({r['allocation'] for r in rows}, {'nonuniform'})
                self.assertFalse(list(work.glob('*/q/uniform_*')))
            self.assertEqual((total_alloc, total_q), (10,13))

    def test_snapshot_and_condition_coverage(self):
        data = tool.read(tool.KIT/'regeneration.json')
        self.assertEqual(len(data['allocations']), 8)
        self.assertEqual(len(data['profiles']), 8)
        self.assertEqual(len({p['id'] for p in data['profiles']}), 8)
        self.assertEqual(sum(tool.plan(k, Path('/work'))['missing_allocation_searches'] for k in data['models']), 10)
        self.assertEqual(sum(tool.plan(k, Path('/work'))['missing_q_searches'] for k in data['models']), 13)
        self.assertEqual(sum(tool.plan(k, Path('/work'), 'both')['missing_q_searches'] for k in data['models']), 28)

    def test_target_subset_and_uniform_are_explicit(self):
        plan = tool.plan('llama31', Path('/work'), targets='50', skip_head=True)
        self.assertEqual(plan['q_profiles_rebuilt_from_coefficients'], ['llama31_nonuniform_s50'])
        self.assertEqual((plan['missing_allocation_searches'], plan['missing_q_searches']), (0, 0))
        self.assertNotIn('beta0_Q_head', plan['stages'])
        uniform = tool.plan('llama31', Path('/work'), 'uniform', '50')
        self.assertEqual(uniform['q_profiles_rebuilt_from_coefficients'], ['llama31_uniform_s50'])
        self.assertEqual(uniform['allocation_recipes_reused'], 0)
        self.assertEqual(uniform['missing_allocation_searches'], 0)
        for targets in ('', '40,40', '25', '40,nope', '40,'):
            with self.assertRaises(ValueError):
                tool.plan('llama31', Path('/work'), targets=targets)

    def test_scope_expansion_fingerprints_preserve_common_identity(self):
        common = {'package_sha256': 'package', 'model': 'llama31', 'model_path': '/model'}
        one = tool.execution_identity(common, *tool.selection('nonuniform', '50'), skip_head=True)
        all_targets = tool.execution_identity(common, *tool.selection(), skip_head=True)
        both = tool.execution_identity(common, *tool.selection('both', '50'), skip_head=True)
        self.assertEqual(common, {k: one[k] for k in common})
        self.assertEqual(common, {k: all_targets[k] for k in common})
        self.assertEqual(len({r['run_sha256'] for r in (one, all_targets, both)}), 3)
        self.assertEqual(tool.selection_record(*tool.selection('nonuniform', '60,40')),
                         tool.selection_record(*tool.selection('nonuniform', '40,60')))

    def test_multi_gpu_experiments_are_rejected(self):
        for count in (0, 2, 8):
            fake = SimpleNamespace(cuda=SimpleNamespace(device_count=lambda: count))
            with self.assertRaises(RuntimeError):
                tool.require_single_gpu(fake)
        tool.require_single_gpu(SimpleNamespace(cuda=SimpleNamespace(device_count=lambda: 1)))

    def test_all_stage_filters_rebuild_search_and_export_and_allows_scope_expansion(self):
        fake_torch = SimpleNamespace(cuda=SimpleNamespace(
            device_count=lambda: 1, is_available=lambda: True,
            get_device_properties=lambda index: SimpleNamespace(total_memory=32 << 30)))
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)/'work'
            args = SimpleNamespace(model='llama31', work=work, model_path=Path(tmp)/'model',
                allocation='nonuniform', targets='50', skip_head=True, download=False, stage='all')
            calls = []
            configurations = []
            def configure(key, directory, model, allocation, targets):
                configurations.append((allocation, targets))
                def run():
                    tool.dump(work/'progress.json', {'status': 'complete'})
                    for mode in ('uniform', 'nonuniform'):
                        for target in (40, 50, 60):
                            tool.dump(work/key/'q'/f'{mode}_s{target}'/'final_recipe.json', {})
                return SimpleNamespace(run=run)
            with patch.dict(sys.modules, {'torch': fake_torch}), \
                 patch('verify_package.verify'), patch.object(tool, 'verify_model'), \
                 patch.object(tool.signal, 'signal'), patch.object(tool.shutil, 'copytree'), \
                 patch.object(tool.shutil, 'disk_usage', return_value=SimpleNamespace(free=1 << 40)), \
                 patch.object(tool, 'run_script', side_effect=lambda work, label, script, argv: calls.append(label)), \
                 patch.object(tool, 'configure_search', side_effect=configure):
                tool.execute(args)
                self.assertEqual(calls, ['dense_statistics', 'common_pack',
                    'llama31_nonuniform_s50', 'nonuniform_s50_export'])
                self.assertEqual(configurations, [('nonuniform', (50,))])
                common_identity = tool.read(work/'regeneration_identity.json')
                first_scope = tool.read(work/'regeneration_selection.json')
                args.targets = '40,50,60'
                calls.clear()
                tool.execute(args)
                self.assertEqual(common_identity, tool.read(work/'regeneration_identity.json'))
                self.assertNotEqual(first_scope['run_sha256'], tool.read(work/'regeneration_selection.json')['run_sha256'])
                self.assertFalse(any('uniform' in x and 'nonuniform' not in x for x in calls))
                self.assertEqual(len(list((work/'selection_history').glob('*.json'))), 2)
                self.assertEqual(tool.read(work/'regeneration_progress.json')['selection']['targets'], [40, 50, 60])

    def test_recipe_shapes_and_normalization(self):
        configs = tool.read(tool.KIT/'models.json')
        for p in tool.read(tool.KIT/'regeneration.json')['profiles']:
            layers = configs[p['model']]['layers']
            tool.validate_alpha(p['alpha'], layers)
            self.assertEqual(len(p['recipe']), layers)
            self.assertTrue(all(len(r) == 7 and all(0 <= v <= 100 for v in r) for r in p['recipe']))
            self.assertEqual(p['replay_layers'], sorted(set(p['replay_layers'])))
            self.assertTrue(set(p['accepted_layers']).issubset(p['replay_layers']))
            self.assertTrue(all(0 <= x < layers for x in p['replay_layers']))
            self.assertFalse(p['bitwise_equivalence_claim'])

    def test_invalid_coefficients_fail(self):
        for alpha in ([[.25]*6], [[float('nan'), 0, 0, 0, 0, 1]], [[-1, 2, 0, 0, 0, 0]], [[1]]):
            with self.assertRaises(ValueError):
                tool.validate_alpha(alpha, 1)

    def test_unsafe_paths_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(ValueError):
                tool.safe_child(root, '../secret')
            with self.assertRaises(ValueError):
                tool.safe_child(root, '.')
            self.assertEqual(tool.safe_child(root, 'layer/q.pt'), root/'layer/q.pt')

    def test_plan_does_not_load_torch(self):
        output = subprocess.run([sys.executable, '-c',
            'import sys; sys.path.insert(0, sys.argv[1]); import prepare_remote as p; '
            'p.plan("llama31", p.Path("/tmp/not-written")); '
            'assert "torch" not in sys.modules; assert "huggingface_hub" not in sys.modules',
            str(tool.KIT/'tools')], capture_output=True, text=True)
        self.assertEqual(output.returncode, 0, output.stderr)

    def test_dry_run_does_not_create_work_or_download(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)/'absent_work'
            model = Path(tmp)/'absent_model'
            output = subprocess.run([sys.executable, str(tool.KIT/'tools/prepare_remote.py'),
                '--model', 'llama31', '--model-path', str(model), '--work', str(work),
                '--download', '--stage', 'all'], capture_output=True, text=True)
            self.assertEqual(output.returncode, 0, output.stderr)
            self.assertFalse(work.exists())
            self.assertFalse(model.exists())

    def test_package_cannot_be_work_directory(self):
        output = subprocess.run([sys.executable, str(tool.KIT/'tools/prepare_remote.py'),
            '--model', 'llama31', '--model-path', '/not-a-real-model',
            '--work', str(tool.KIT/'would_corrupt')], capture_output=True, text=True)
        self.assertNotEqual(output.returncode, 0)
        self.assertFalse((tool.KIT/'would_corrupt').exists())

    def test_model_and_tokenizer_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            kit, model = root/'kit', root/'model'
            model.mkdir()
            (model/'model.safetensors').write_bytes(b'test-only-not-real-weights')
            (model/'tokenizer.json').write_text('{}')
            cfg = {'model_type': 'llama', 'hidden_size': 8}
            tool.dump(model/'config.json', cfg)
            tool.dump(kit/'inputs/toy/config.json', cfg)
            tool.dump(kit/'inputs/toy/base_model_identity.json', {'weights':[
                {'path':'model.safetensors','bytes':(model/'model.safetensors').stat().st_size,
                 'sha256':tool.sha(model/'model.safetensors')}]})
            tool.dump(kit/'regeneration.json', {'models':{'toy':{'tokenizer_sha256':
                {'tokenizer.json':tool.sha(model/'tokenizer.json')}}}})
            with patch.object(tool, 'KIT', kit):
                tool.verify_model('toy', model)
                (model/'tokenizer.json').write_text('{"changed":true}')
                with self.assertRaises(RuntimeError):
                    tool.verify_model('toy', model)
                (model/'model.safetensors').write_bytes(b'tampered')
                with self.assertRaises(RuntimeError):
                    tool.verify_model('toy', model)

    def test_required_source_closure(self):
        for name in tool.read(tool.KIT/'source_closure.json')['files']:
            self.assertTrue((tool.KIT/name).is_file(), name)
        for script in ('prepare_remote.py', 'materialize_profile.py'):
            ast.parse((tool.KIT/'tools'/script).read_text())

    def test_no_large_tensors_or_credentials(self):
        files = tool.read(tool.KIT/'MANIFEST.json')['files']
        for path, entry in files.items():
            self.assertNotIn(Path(path).name, ('.env', 'token', 'id_rsa', 'id_ed25519'))
            self.assertFalse(path.endswith('.safetensors'))
            if path.endswith('.pt'):
                self.assertTrue(path.startswith('inputs/'))
                self.assertLess(entry['bytes'], 2 << 20)

    def test_export_hash_validation(self):
        import materialize_profile as stage
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'q.pt').write_bytes(b'cpu-test-not-real-tensor')
            manifest = stage.exported_files(root)
            stage.verify_export(root, manifest)
            (root/'q.pt').write_bytes(b'broken')
            with self.assertRaises(RuntimeError):
                stage.verify_export(root, manifest)


if __name__ == '__main__':
    unittest.main(verbosity=2)
