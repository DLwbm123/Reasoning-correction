import os
import unittest

os.environ.setdefault('RUN_ROOT', '/tmp/test-only')
from run_e0 import validate


class CandidateFormat(unittest.TestCase):
    def test_valid_answer_without_explanation_is_not_fabricated_or_discarded(self):
        result = {'answer':'C', 'claim':'unstructured explanation'}
        validate('repair', result)
        self.assertEqual(result['claims'], [])
        self.assertIn('_schema_warning', result)
        self.assertEqual(result['answer'],'C')

    def test_missing_answer_remains_a_failure(self):
        with self.assertRaises(ValueError):
            validate('repair', {'claims':[]})


if __name__ == '__main__':
    unittest.main()
