import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from roshift.speedup.execution import parser, make_request
from roshift.speedup.runtime import spec_for, ENGINE, PERCENTS
from roshift.speedup.report import summarize
from roshift.speedup.data import text_rows


class SpeedupContracts(unittest.TestCase):
    def request(self,*extra):
        return make_request(parser().parse_args(['run','--model','llama31','--data-dir','artifacts',*extra]))

    def test_defaults(self):
        r=self.request()
        self.assertEqual(r['percents'],[25,40,50,60,75])
        self.assertEqual(r['repeats'],4)
        self.assertFalse(r['smoke_subset'])

    def test_subset_is_not_full(self):
        self.assertTrue(self.request('--datasets','wikitext2','--repeats','1')['smoke_subset'])

    def test_duplicate_rejected(self):
        with self.assertRaises(ValueError):
            self.request('--percents','50','50')

    def test_repeats_rejected(self):
        with self.assertRaises(ValueError):
            self.request('--repeats','0')

    def test_five_models_real_sources(self):
        for model in ('llama31','llama2','mistral','qwen25','qwen3'):
            spec=spec_for(model)
            self.assertGreater(spec['layers'],0)
            self.assertEqual(spec['dim']%128,0)
            self.assertTrue((ENGINE/model/'block_sparse_b128_gemv.cu').is_file())
            self.assertLess(spec['head_policy']['second_dim'],spec['dim'])

    def test_geometry_frozen_at_50(self):
        for model in ('mistral','llama2','qwen25','qwen3'):
            record=spec_for(model)['geometry']
            self.assertEqual(record['selected_at_sparsity'],50)
            self.assertTrue(record['never_retune_other_sparsities'])

    def test_paper_block_quotas(self):
        self.assertEqual([round(128*(100-p)/100) for p in PERCENTS],[96,77,64,51,32])

    def test_all_model_kernel_variants_generate(self):
        spec=importlib.util.spec_from_file_location('speedup_support_test',ENGINE/'support.py')
        module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for model in ('llama31','llama2','mistral','qwen25','qwen3'):
            source=(ENGINE/model/'block_sparse_b128_gemv.cu').read_text()
            for percent in PERCENTS:
                generated=module.build_source(source,module.keep_for(percent),'test_variant')
                self.assertIn('TORCH_LIBRARY(test_variant,',generated)

    def test_prompt_format(self):
        rows=list(text_rows('gsm8k',[{'question':' q ','answer':' a '}]))
        self.assertEqual(rows,['Question: q\nAnswer: a'])

    def test_no_raw_timing_tokens(self):
        from roshift.speedup.runtime import ROOT
        self.assertFalse(list(ROOT.rglob('*.pt')))

    def test_original_measured_decode_loop(self):
        for model in ('llama31','llama2','mistral','qwen25','qwen3'):
            worker=(ENGINE/model/'worker.py').read_text()
            self.assertIn('discard_full_replay_before_each_record=True',worker)
            self.assertIn('compiled_validity',worker)
            self.assertIn('prefill=1024, decode=128',worker)
            self.assertIn('sanity_dataset = bench.DATASETS[0]',worker)

    def test_cache_budget_uses_model_layer_count(self):
        for model in ('llama31','llama2','mistral','qwen25','qwen3'):
            calibration=(ENGINE/model/'calibrate_centers.py').read_text()
            self.assertIn('solve_budget_dp(costs,4*LAYERS,1,32,4)',calibration)

    def test_report_pairs_raw_times(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            protocol=dict(prompt_cache_sha256={'x':'abc'},model_config_sha256='cfg',
                          datasets=['x'],prefill=1024,decode=128,repeats=1)
            for method,ms in (('dense',10),('ours',5)):
                path=root/method/('s0' if method=='dense' else 's50')/'result.json'
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps(dict(status='complete',mean_ms_per_token=ms,protocol=protocol,
                                               args=dict(method=method,sparsity=.5))))
            result=summarize(root)
            self.assertEqual(result['rows'][1]['speedup'],2)
            self.assertEqual(result['rows'][1]['tokens_per_second'],200)
            self.assertFalse(result['paper_protocol'])

    def test_report_refuses_mixed_prompts(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for method,sha in (('dense','a'),('teal','b')):
                path=root/method/('s0' if method=='dense' else 's50')/'result.json'
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps(dict(status='complete',mean_ms_per_token=1,
                    protocol={'prompt_cache_sha256':sha},args={'method':method,'sparsity':.5})))
            with self.assertRaises(ValueError):
                summarize(root)

    def test_report_refuses_mixed_devices(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for method,device in (('dense','GPU A'),('teal','GPU B')):
                path=root/method/('s0' if method=='dense' else 's50')/'result.json'
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps(dict(status='complete',mean_ms_per_token=1,
                    protocol={},args={'method':method,'sparsity':.5},
                    method_once={'protocol':{'device':device}})))
            with self.assertRaises(ValueError):
                summarize(root)


if __name__=='__main__':
    unittest.main()
