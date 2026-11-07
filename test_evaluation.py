import json
from pathlib import Path
import tempfile
import unittest
from evaluate_e0 import BRANCHES, evaluate


class OutcomeAccounting(unittest.TestCase):
    def test_repair_and_invalid_harm_are_counted_without_dropping(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'inputs').mkdir()
            (root/'outputs/cases').mkdir(parents=True)
            labels = [{'case_id':'a','group_id':'g1','reference_letter':'A'},
                      {'case_id':'b','group_id':'g2','reference_letter':'B'}]
            (root/'inputs/labels.private.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in labels))
            for cid, initial, answer in [('a', 'D', {'answer':'A'}), ('b','B',{'_parse_failed':True})]:
                (root/f'outputs/cases/{cid}.json').write_text(json.dumps({'initial':{'answer':initial},
                    'answers':{b:answer for b in BRANCHES}}))
            (root/'outputs/calls.jsonl').write_text('')
            metrics = evaluate(root)
            for b in BRANCHES:
                r = metrics['branches'][b]
                self.assertEqual((r['WC'],r['CW'],r['invalid']), (1,1,1))
                self.assertEqual(r['delta_accuracy'],0)
                self.assertEqual(r['harm_per_edit'],.5)
            self.assertTrue(metrics['complete'])

    def test_unrun_cases_remain_explicit_and_run_marked_incomplete(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); (root/'inputs').mkdir(); (root/'outputs/cases').mkdir(parents=True)
            labels=[{'case_id':c,'group_id':c,'reference_letter':'A'} for c in ['a','b']]
            (root/'inputs/labels.private.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in labels))
            (root/'outputs/cases/a.json').write_text(json.dumps({'pipeline_error':'recorded failure'}))
            (root/'outputs/calls.jsonl').write_text('')
            metrics=evaluate(root)
            self.assertFalse(metrics['complete'])
            self.assertEqual(metrics['unrun_cases'],1)
            self.assertEqual(metrics['mechanism']['pipeline_error'],1)
            self.assertEqual(metrics['branches']['raw']['invalid'],1)


if __name__ == '__main__':
    unittest.main()
