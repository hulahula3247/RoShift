import unittest

from roshift.accuracy.accounting import document_count


class SampleAccountingTests(unittest.TestCase):
    def test_filter_records_are_not_extra_documents(self):
        rows = [('gsm8k', dict(doc_id=i, filter=f, doc_hash=str(i), resps=[['42']]))
                for f in ('strict-match', 'flexible-extract') for i in range(2)]
        self.assertEqual(document_count(rows), 2)

    def test_leaf_tasks_have_distinct_document_namespaces(self):
        self.assertEqual(document_count([(name, dict(doc_id=0))
            for name in ('mmlu_a', 'mmlu_b')]), 2)

    def test_conflicting_duplicate_fails(self):
        with self.assertRaises(ValueError):
            document_count([('x', dict(doc_id=0, prompt_hash='a')),
                            ('x', dict(doc_id=0, prompt_hash='b'))])

    def test_missing_document_identity_fails(self):
        with self.assertRaises(ValueError):
            document_count([('x', {})])


if __name__ == '__main__':
    unittest.main()
