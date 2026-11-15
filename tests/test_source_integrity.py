import ast
import unittest

from roshift.accuracy.execution import ENGINE


class SourceIntegrityContracts(unittest.TestCase):
    def test_original_matrix_operations_restored(self):
        source = (ENGINE/'code/analysis/test_rental_preparation_20260916.py').read_text(encoding='utf-8')
        self.assertNotIn('contact-redacted', source)
        tree = ast.parse(source)
        projections = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.MatMult):
                value = ast.unparse(node)
                if 'layer.self_attn.o_proj.weight.float()' in value:
                    projections.add('o')
                if 'layer.mlp.down_proj.weight.float()' in value:
                    projections.add('down')
        self.assertEqual(projections, {'o', 'down'})

    def test_upstream_attribution_and_ssh_examples_preserved(self):
        root = ENGINE/'code/vendor/transformers/models'
        bertweet = (root/'bertweet/tokenization_bertweet.py').read_text(encoding='utf-8')
        self.assertIn('cgpotts@stanford.edu', bertweet)
        self.assertIn('Author: Christopher Potts', bertweet)
        self.assertNotIn('contact-redacted', bertweet)
        for relative in ('flaubert/tokenization_flaubert.py', 'xlm/tokenization_xlm.py',
                         'marian/convert_marian_to_pytorch.py', 'marian/convert_marian_tatoeba_to_pytorch.py'):
            source = (root/relative).read_text(encoding='utf-8')
            self.assertIn('git@github.com:', source)
            self.assertNotIn('contact-redacted', source)


if __name__ == '__main__':
    unittest.main()
