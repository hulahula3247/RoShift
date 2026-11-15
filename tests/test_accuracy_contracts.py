import ast
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from roshift.accuracy.execution import ENGINE,parser,make_request,work_environment
from roshift.catalog import asset


class AccuracyContracts(unittest.TestCase):
    def args(self,*extra):
        return parser().parse_args(['evaluate','--model','llama31','--data-dir',tempfile.gettempdir(),*extra])

    def test_default_is_frozen_and_no_execution(self):
        a=self.args()
        self.assertFalse(a.execute)
        self.assertFalse(a.use_search)
        self.assertEqual(a.batch_size,1)

    def test_invalid_policies_rejected(self):
        for tail in [('--method','dense'),
                     ('--limit','0'),('--tasks','BoolQ','BoolQ')]:
            with self.assertRaises(ValueError):
                make_request(self.args(*tail))

    def test_benchmark_set(self):
        tasks=asset('benchmarks.json')
        self.assertEqual(len(tasks),11)
        self.assertEqual(set(tasks), {'MMLU', 'BoolQ', 'PIQA', 'ARC-E', 'ARC-C',
                                     'WinoGrande', 'CommonsenseQA', 'MBPP', 'GSM8K',
                                     'BBH', 'MATH-500'})
        self.assertEqual(sum(x['kind']=='likelihood' for x in tasks.values()),7)
        self.assertEqual(tasks['GSM8K']['shots'],5)
        self.assertEqual(tasks['BBH']['count'],6511)
        self.assertEqual(tasks['MBPP']['count'],500)

    def test_manifest_has_required_tasks(self):
        paths=json.loads((ENGINE/'code/MANIFEST.json').read_text())['files']
        for root in ('arc','bbh','commonsense_qa','gsm8k','mbpp','minerva_math','mmlu','piqa','winogrande'):
            self.assertTrue(any(p.startswith('vendor/lm_eval/tasks/'+root+'/') for p in paths),root)
        self.assertTrue((ENGINE.parent/'tasks/boolq.yaml').is_file())

    def test_all_baselines_pinned(self):
        hashes=asset('upstream_sources.json')
        for method in ('teal','larosa'):
            self.assertTrue(any(k.startswith('baselines/'+method+'/') for k in hashes))

    def test_python_sources_parse(self):
        for folder in (ENGINE/'code/analysis',ENGINE/'code/tools'):
            for path in folder.glob('*.py'):
                ast.parse(path.read_text(encoding='utf-8'),filename=str(path))

    def test_profile_metadata_overrides_historical_cohort(self):
        source=ast.parse((ENGINE/'code/tools/materialize_profile.py').read_text())
        records=[node for node in ast.walk(source) if isinstance(node,ast.Call)
                 and isinstance(node.func,ast.Name) and node.func.id=='dict'
                 and any(k.arg=='cohort' for k in node.keywords)]
        record=next(node for node in records if any(k.arg=='selected' for k in node.keywords))
        spec={'cohort':'historical','alpha':[[.25,.25,.125,.125,.125,.125]]}
        value=eval(compile(ast.Expression(record),'<profile-metadata>','eval'),
                   {'spec':spec,'out':Path('/profile'),'common':Path('/common')})
        self.assertEqual(value['cohort'],'remote_regenerated_shared_dense_median')
        self.assertEqual(value['alpha'],spec['alpha'])
        self.assertEqual(spec['cohort'],'historical')

    def test_generation_settings_match_benchmarks(self):
        source = ast.parse((ENGINE/'ops/generation_tasks.py').read_text(encoding='utf-8'))
        build = next(node for node in source.body if isinstance(node, ast.FunctionDef) and node.name == 'build')
        result = next(node.value for node in build.body if isinstance(node, ast.Assign)
                      and any(isinstance(target, ast.Name) and target.id == 'result' for target in node.targets))
        settings = eval(compile(ast.Expression(result), '<generation-settings>', 'eval'),
                        {'CODE': Path('/source'), 'sha': lambda path: 'fixture', 'tasks': {}, 'sources': {}})
        caps = {row['task']: row['max_new_tokens'] for row in asset('benchmarks.json').values()
                if row['kind'] == 'generation'}
        self.assertEqual(settings['generation_cap_confirmation']['caps'], caps)
        self.assertEqual(settings['approved_seeds'], dict(random_seed=0, numpy_random_seed=1234,
                                                        torch_random_seed=1234, fewshot_random_seed=1234))
        self.assertFalse(settings['historical_exact_protocol_identity_verified'])
        self.assertEqual(settings['head_scope'], dict(roshift='validated_sparse_head_on_decode_only',
                                                     teal='dense', larosa='dense'))
        consumer = (ENGINE/'ops/run_generation.py').read_text(encoding='utf-8')
        self.assertIn("protocol['protocol_note']", consumer)

    def test_windows_volume_guard(self):
        with self.assertRaises(ValueError):
            work_environment('C:/should_not_be_created' if os.name=='nt' else '/mnt/c/should_not_be_created')

    def test_writable_paths_outside_source(self):
        with tempfile.TemporaryDirectory() as folder, \
             patch('roshift.accuracy.execution.shutil.disk_usage',return_value=SimpleNamespace(free=20*2**30)):
            root,env=work_environment(folder)
            for key in ('HF_HOME','HF_DATASETS_CACHE','TMPDIR','TORCH_EXTENSIONS_DIR'):
                self.assertTrue(Path(env[key]).is_relative_to(root))

    def test_low_free_space_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder, \
             patch('roshift.accuracy.execution.shutil.disk_usage',return_value=SimpleNamespace(free=2**30)):
            with self.assertRaises(RuntimeError):
                work_environment(folder)

    def test_missing_auxiliary_stderr_is_null(self):
        from roshift.accuracy.evaluate import save
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'smoke.json'
            save(path,{'metric_value':.5,'raw':{'stderr':float('nan')}})
            self.assertEqual(json.loads(path.read_text()),{'metric_value':.5,'raw':{'stderr':None}})

    def test_model_cycles_released_before_allocator_cache(self):
        import types
        from roshift.accuracy.evaluate import release_cuda
        events=[]
        fake=types.SimpleNamespace(cuda=types.SimpleNamespace(empty_cache=lambda:events.append('cache')))
        with patch.dict('sys.modules',{'torch':fake}), \
             patch('roshift.accuracy.evaluate.gc.collect',side_effect=lambda:events.append('gc')):
            release_cuda()
        self.assertEqual(events,['gc','cache'])

    def test_ppl_cleanup_precedes_head_model_load(self):
        import types
        from contextlib import nullcontext
        from roshift.accuracy.evaluate import ensure_head
        events=[]
        worker=types.SimpleNamespace(run=lambda args:events.append('load_head'))
        fake_torch=types.SimpleNamespace(inference_mode=nullcontext)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            (root/'sparse_head_validation_v2.json').write_text(json.dumps(
                {'passed':True,'generation_probe':{'passed':True}}))
            def ppl(*args):
                events.append('ppl')
                return types.SimpleNamespace(),root
            with patch.dict('sys.modules',{'torch':fake_torch,'roshift_head_v2':types.SimpleNamespace(worker=worker)}), \
                 patch('roshift.accuracy.evaluate.ensure_ppl',side_effect=ppl), \
                 patch('roshift.accuracy.evaluate.release_cuda',side_effect=lambda:events.append('cleanup')):
                self.assertTrue(ensure_head({},root,root)['passed'])
        self.assertEqual(events,['ppl','cleanup','load_head'])

    def test_profile_fingerprint_changes_with_recipe(self):
        from roshift.accuracy.integrity import profile_identity
        with tempfile.TemporaryDirectory() as folder:
            work=Path(folder)
            profile=work/'llama31/rebuilt/llama31_uniform_s50'
            profile.mkdir(parents=True)
            for name in ('profile.json','materialized.json'):
                (profile/name).write_text('{}')
            request=dict(model='llama31',method='roshift',allocation='uniform',sparsity=50)
            before=profile_identity(request,work,work)
            (profile/'profile.json').write_text('{"changed":true}')
            self.assertNotEqual(before,profile_identity(request,work,work))

    def test_existing_login_not_copied(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            login=root/'existing_login'
            login.mkdir()
            token=login/'token'
            token.write_text('not-a-token')
            with patch.dict(os.environ,{'HF_HOME':str(login)},clear=True), \
                 patch('roshift.accuracy.execution.shutil.disk_usage',return_value=SimpleNamespace(free=20*2**30)):
                _,env=work_environment(root/'data')
            self.assertEqual(env['HF_TOKEN_PATH'],str(token.resolve()))
            self.assertFalse((Path(env['HF_HOME'])/'token').exists())

    def test_download_mapping_keeps_original_tokenizer_hash(self):
        import hashlib
        from roshift.data import download_model
        import types
        text=b'tokenizer fixture'
        config=dict(hf_id='fixture/repo',download_revision='revision',weights=[],
                    tokenizer_sha256={'tokenizer.model':hashlib.sha256(text).hexdigest()},
                    download_file_map={'tokenizer.model':'original/tokenizer.model'})
        def download(**kwargs):
            self.assertIn('original/tokenizer.model',kwargs['allow_patterns'])
            folder=kwargs['local_dir']/'original'
            folder.mkdir(parents=True)
            (folder/'tokenizer.model').write_bytes(text)
        with tempfile.TemporaryDirectory() as folder:
            hub=types.SimpleNamespace(snapshot_download=download)
            with patch.dict('sys.modules',{'huggingface_hub':hub}), \
                 patch('roshift.data.model_config',return_value=config), \
                 patch('roshift.data.verify_model',return_value={'ok':True}):
                download_model('fixture',folder)
            self.assertEqual((Path(folder)/'models/fixture/tokenizer.model').read_bytes(),text)


if __name__=='__main__':
    unittest.main()
