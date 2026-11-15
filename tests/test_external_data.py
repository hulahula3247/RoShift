import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from roshift.accuracy.datasets import fill, fingerprint, require_match, format_record, ensure_roshift
from roshift.catalog import asset, MODELS


class ExternalDataTests(unittest.TestCase):
    def test_no_dataset_payloads_shipped(self):
        root = Path(__file__).resolve().parents[1]/'src/roshift'
        references = asset('external_data.json')
        self.assertEqual(len(references['files']),36)
        for relative in references['files']:
            self.assertFalse((root/relative).exists(),relative)
        for revision in references['revisions'].values():
            self.assertRegex(revision,r'^[a-f0-9]{40}$')

    def test_fail_closed_content_and_dtype(self):
        value = torch.tensor([[1,2]],dtype=torch.int64)
        reference = fingerprint(value)
        require_match(value,reference,'good')
        for changed in (value+1,value.int(),value.flatten()):
            with self.assertRaisesRegex(RuntimeError,'mismatch'):
                require_match(changed,reference,'changed')

    def test_crossing_record_is_discarded(self):
        class Tokenizer:
            eos_token_id=99
            def __call__(self,text,**kwargs):
                return {'input_ids':[int(x) for x in text.split()]}
        iterator=iter([{'text':'1 2 3 4'},{'text':'5'},{'text':'6 7'}])
        self.assertEqual(fill(iterator,Tokenizer(),0,3),[1,2,3])
        self.assertEqual(fill(iterator,Tokenizer(),0,3),[5,99,6])
        with self.assertRaisesRegex(RuntimeError,'exhausted'):
            fill(iterator,Tokenizer(),0,1)

    def test_formatting(self):
        self.assertEqual(format_record({'instruction':' q ','input':' c ','output':' a '},1),
                         'Instruction:\nq\n\nInput:\nc\n\nResponse:\na')
        self.assertEqual(format_record({'query':'q','response':'a'},2),'Question:\nq\n\nAnswer:\na')
        self.assertEqual(format_record({'text':'q','test_list':['t'],'code':'a'},4),
                         'Programming problem:\nq\n\nTests:\nt\n\nSolution:\na')

    def test_cache_identity_rejected_before_network(self):
        with tempfile.TemporaryDirectory() as directory:
            target=Path(directory)/'external_data/inputs/llama31'
            target.mkdir(parents=True)
            (target/'data_ready.json').write_text(json.dumps({'identity':{}}))
            with patch('roshift.accuracy.datasets.load',side_effect=AssertionError('network')):
                with self.assertRaisesRegex(RuntimeError,'another source'):
                    ensure_roshift('llama31',Path(directory),Path(directory))

    def test_mit_is_scoped(self):
        root=Path(__file__).resolve().parents[1]
        self.assertIn('MIT License',(root/'LICENSE').read_text())
        self.assertIn('RoShift contributors',(root/'LICENSE').read_text())
        self.assertIn('not to third-party',(root/'third_party/README.md').read_text())
        self.assertTrue((root/'third_party/lm_eval_LICENSE.txt').is_file())


if __name__=='__main__':
    unittest.main()
