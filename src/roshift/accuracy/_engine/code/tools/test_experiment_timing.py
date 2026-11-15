"""Stdlib accounting tests; no CUDA, models, datasets or numerical search."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import prepare_remote
from experiment_timing import Attempt, FILENAME, classify, job, reused, summarize
import rental_local_suite_20260916 as suite


def clock(*values):
    iterator=iter(values)
    return lambda: next(iterator)


class TimingTests(unittest.TestCase):
    def test_complete_failed_interrupted_and_reused_accounting(self):
        with tempfile.TemporaryDirectory() as temporary, patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'3'},clear=False):
            path=Path(temporary)/FILENAME
            metadata=classify('qwen3:was50')
            with Attempt(path,clock=clock(10,3610),**metadata):
                pass
            with self.assertRaises(RuntimeError):
                with Attempt(path,clock=clock(5000,6800),**metadata):
                    raise RuntimeError('actual failed attempt')
            with self.assertRaises(KeyboardInterrupt):
                with Attempt(path,clock=clock(7000,7900),**metadata):
                    raise KeyboardInterrupt()
            reused(path,**metadata,origin='historical_choice_not_measured_now')
            result=summarize([path])
            self.assertEqual(result['allocation_gpu_hours'],1.75)
            self.assertEqual(result['reused_stages'],1)
            self.assertEqual(result['open_attempts'],[])
            row=result['leaf_stages'][0]
            self.assertEqual((row['complete'],row['failed'],row['interrupted']),(1,1,1))
            events=[json.loads(line) for line in path.read_text().splitlines()]
            self.assertTrue(all(event['cuda_visible_devices']=='3' for event in events))
            self.assertEqual(events[-1]['gpu_hours'],0.)

    def test_parent_container_is_not_double_counted_with_leaf_or_total(self):
        with tempfile.TemporaryDirectory() as temporary, patch.dict(os.environ,{},clear=True):
            path=Path(temporary)/FILENAME
            with job(path,clock=clock(0,7200),model='qwen3',category='job',substage='prepare',targets=[50]):
                with job(path,clock=clock(100,7000),model='qwen3',category='job',substage='search',targets=[50]):
                    with Attempt(path,clock=clock(1000,4600),**classify('qwen3:grid50')):
                        pass
            report=summarize([path])
            self.assertEqual(report['total_job_gpu_hours'],2.)
            self.assertEqual(report['allocation_gpu_hours'],1.)
            self.assertNotIn('ROSHIFT_TIMING_TOTAL_JOB_ID',os.environ)
            self.assertNotIn('ROSHIFT_TIMING_PARENT_ID',os.environ)

    def test_cpu_dataset_and_shared_target_classification(self):
        self.assertEqual(classify('gemma1:heldout')['category'],'dataset_cpu')
        self.assertFalse(classify('gemma1:wikitext')['gpu_reserved'])
        shared=classify('gemma1:teal',['--targets','40,60'])
        self.assertEqual(shared['targets'],[40,60])
        self.assertIsNone(shared['target'])
        self.assertEqual(shared['attribution'],'shared_targets')
        self.assertEqual(classify('gemma1:teal',['--targets','50'])['target'],50)
        self.assertEqual(classify('gemma1:nonuniform50:pairwise')['category'],'q_search')
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/FILENAME
            with Attempt(path,clock=clock(0,3600),**classify('gemma1:wikitext')):
                pass
            self.assertEqual(summarize([path])['leaf_stages'][0]['gpu_hours'],0.)

    def test_unterminated_attempt_remains_visible(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/FILENAME
            attempt=Attempt(path,clock=clock(10,30),**classify('mistral:was60'))
            attempt.__enter__()
            self.assertEqual(len(summarize([path])['open_attempts']),1)
            attempt.__exit__(RuntimeError,RuntimeError('reconciled failure'),None)
            self.assertEqual(summarize([path])['open_attempts'],[])

    def test_prepare_subprocess_reuse_and_failure_recorded_without_recharging_stage(self):
        with tempfile.TemporaryDirectory() as temporary:
            work=Path(temporary)
            identity={'model':'llama31','run_sha256':'fixture'}
            prepare_remote.dump(work/'regeneration_selection.json',identity)
            with patch.object(prepare_remote,'_run_script',side_effect=RuntimeError('subprocess failed')):
                with self.assertRaises(RuntimeError):
                    prepare_remote.run_script(work,'dense_statistics','stage.py',[])
            marker=work/'llama31/common/dense_preparation_state.json'
            prepare_remote.dump(marker,{'status':'complete'})
            with patch.object(prepare_remote,'_run_script') as call:
                prepare_remote.run_script(work,'dense_statistics','stage.py',[])
                call.assert_called_once()
            events=[json.loads(line) for line in (work/FILENAME).read_text().splitlines()]
            self.assertEqual([(e['scope'],e['status']) for e in events if e['event']=='end'],
                             [('leaf','failed'),('container','complete')])
            self.assertEqual(events[-1]['event'],'reused')
            self.assertEqual(events[-1]['gpu_hours'],0.)

    def test_real_target_scoped_teal_marker_cannot_reuse_other_targets(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)/'allocation'
            fifty=suite.teal_directory(root,[50])
            suite.dump(fifty/'summary.json',{'status':'complete','args':{'targets':'50'}})
            suite.dump(fifty/'recipes.json',{'teal_paper_s50':[[50.]*7]})
            self.assertTrue(suite.complete(fifty/'summary.json'))
            for targets in ([40],[60],[40,50,60]):
                path=suite.teal_directory(root,targets)
                self.assertNotEqual(path,fifty)
                self.assertFalse(suite.complete(path/'summary.json'))
                self.assertFalse((path/'recipes.json').exists())
            self.assertEqual(suite.read(fifty/'recipes.json'),{'teal_paper_s50':[[50.]*7]})
            self.assertEqual(suite.teal_directory(root,[60,40]),suite.teal_directory(root,[40,60]))


if __name__=='__main__':
    unittest.main(verbosity=2)
