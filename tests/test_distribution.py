import contextlib
import io
from pathlib import Path
import unittest
from unittest.mock import patch

from roshift.accuracy import baselines, worker
from roshift.accuracy.execution import ENGINE, make_request, parser
from roshift.catalog import METHODS, asset
from roshift.plan import make_plan


class DistributionContracts(unittest.TestCase):
    def test_supported_methods(self):
        self.assertEqual(METHODS, ('dense', 'teal', 'larosa', 'roshift'))
        self.assertNotIn('wina', asset('baselines.json'))
        self.assertFalse(any('wina' in key.lower() for key in asset('upstream_sources.json')))
        self.assertFalse(any(row.get('method', '').lower() == 'wina'
                             for row in asset('run_settings.json')))

    def test_excluded_source_and_environment(self):
        self.assertFalse(any('wina' in str(path.relative_to(ENGINE)).lower()
                             for path in ENGINE.rglob('*')))
        root = Path(__file__).resolve().parents[1]
        self.assertFalse((root/'environments/wina-hf450.txt').exists())
        self.assertFalse((root/'tools/validate_wina_scope.py').exists())
        self.assertTrue((root/'third_party/Apache-2.0.txt').is_file())

    def test_cli_and_plan_reject_excluded_method(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser().parse_args(['evaluate', '--model', 'llama31',
                                 '--method', 'wina', '--data-dir', '/data/unused'])
        args = parser().parse_args(['evaluate', '--model', 'llama31', '--data-dir', '/data/unused'])
        args.method = 'wina'
        with self.assertRaisesRegex(ValueError, 'Unsupported method'):
            make_request(args)
        with self.assertRaises(ValueError):
            make_plan('accuracy', 'llama31', 'wina', 50)

    def test_no_direct_worker_or_baseline_fallback(self):
        with patch.object(worker, 'work_environment', side_effect=AssertionError('side effect')):
            with self.assertRaisesRegex(ValueError, 'Unsupported method'):
                worker.setup({'method': 'wina'})
        for function in (baselines.prepare_baseline, baselines.load_baseline):
            with self.assertRaisesRegex(ValueError, 'Unsupported baseline'):
                function('wina', 'llama31', None, None, 'nonuniform', 50)


if __name__ == '__main__':
    unittest.main()
