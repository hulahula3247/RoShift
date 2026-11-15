"""Keep the public first-run instructions connected to executable interfaces."""
from pathlib import Path
import re
import unittest

from roshift.accuracy.execution import parser as accuracy_parser
from roshift.speedup.execution import parser as speedup_parser


ROOT = Path(__file__).resolve().parents[1]


class ReadmeContracts(unittest.TestCase):
    def test_root_help_lists_execution_paths(self):
        import contextlib
        import io
        from roshift.cli import main
        output = io.StringIO()
        with contextlib.redirect_stdout(output), self.assertRaises(SystemExit) as result:
            main(['--help'])
        self.assertEqual(result.exception.code, 0)
        self.assertIn('accuracy', output.getvalue())
        self.assertIn('speedup', output.getvalue())

    def test_relative_links_exist(self):
        paths = [ROOT/'README.md', ROOT/'REPRODUCIBILITY.md',
                 ROOT/'third_party/README.md', *sorted((ROOT/'docs').glob('*.md'))]
        for path in paths:
            text = path.read_text(encoding='utf-8')
            for target in re.findall(r'\]\(([^)]+)\)', text):
                if target.startswith(('http:', 'https:', '#')):
                    continue
                self.assertTrue((path.parent / target.split('#')[0]).is_file(), f'{path}: {target}')

    def test_docs_are_user_guides_only(self):
        docs = list((ROOT/'docs').iterdir())
        required = {'ACCURACY.md', 'SPEEDUP.md', 'VALIDATION.md',
                    'DATA_LICENSES.md', 'ORIN.md'}
        self.assertTrue(required.issubset({p.name for p in docs}))
        self.assertTrue(all(p.is_file() and p.suffix == '.md' for p in docs))
        for name in ('ACCURACY', 'SPEEDUP', 'ORIN'):
            english = (ROOT/'docs'/f'{name}.md').read_text(encoding='utf-8')
            korean = (ROOT/'docs'/f'{name}.ko.md').read_text(encoding='utf-8')
            pattern = r'```[^\n]*\n(.*?)```'
            self.assertEqual(re.findall(pattern, english, re.S),
                             re.findall(pattern, korean, re.S), name)
        for path in [ROOT/'README.md', *sorted((ROOT/'docs').glob('*.md'))]:
            self.assertNotRegex(path.read_text(encoding='utf-8'), r'\bRC\d', str(path))

    def test_shell_examples_parse(self):
        import shlex
        for name in ('README.md', 'README.ko.md'):
            text = (ROOT / name).read_text(encoding='utf-8').replace('\\\n', '')
            checked = 0
            for line in text.splitlines():
                if not line.startswith('roshift '):
                    continue
                args = shlex.split(line)
                if args[1] == 'check':
                    continue
                parser = accuracy_parser if args[1] == 'accuracy' else speedup_parser
                parser().parse_args(args[2:])
                checked += 1
            self.assertGreaterEqual(checked, 1, name)


if __name__ == '__main__':
    unittest.main()
