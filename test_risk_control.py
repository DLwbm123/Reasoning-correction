import math
import unittest
import tempfile
import json
from pathlib import Path
from unittest.mock import patch
from risk_control import upper_bound, select_policy, apply_policy


class RiskChecks(unittest.TestCase):
    def test_one_sided_zero_event_and_multiplicity(self):
        self.assertAlmostEqual(upper_bound(0, 93), 1-.05**(1/93))
        self.assertGreater(upper_bound(0, 305, .05/24), .02)
        self.assertLessEqual(upper_bound(0, 306, .05/24), .02)
        self.assertEqual(upper_bound(1, 1), 1.)
        self.assertEqual(upper_bound(0, 0), 1.)
        with self.assertRaises(ValueError): upper_bound(3, 2)

    def test_zero_calibration_edits_not_zero_population_bound(self):
        rows = [{"case_id": str(i), "c0": 1, "ca": 1, "valid": True,
                 "candidate_answer": "A", "initial_answer": "A"} for i in range(93)]
        p = [[0., 0., 0., 1.]]*93
        empirical, _ = select_policy(rows, p)
        exact, table = select_policy(rows, p, exact=True)
        self.assertFalse(empirical["keep_only"])
        self.assertTrue(exact["keep_only"])
        self.assertTrue(all(x["simultaneous_upper"] > .02 for x in table))
        self.assertTrue(all(not x["accepted"] for x in apply_policy(rows, p, exact)))

    def test_evaluation_guard_precedes_labels(self):
        from risk_pipeline import evaluate
        with tempfile.TemporaryDirectory() as d:
            with patch("risk_pipeline.partition") as read_labels:
                with self.assertRaises(FileNotFoundError): evaluate(Path(d))
                read_labels.assert_not_called()

    def test_calibration_works_without_evaluation_label_file(self):
        from risk_pipeline import partition
        with tempfile.TemporaryDirectory() as d:
            root = Path(d);(root/"inputs").mkdir()
            labels = [{"case_id":str(i),"group_id":str(i),"reference_letter":"A"} for i in range(800)]
            (root/"inputs/labels.calibration.private.jsonl").write_text("\n".join(map(json.dumps,labels)))
            (root/"inputs/splits.private.json").write_text(json.dumps({str(i):"calibration" for i in range(800)}))
            data = {"rows":[{"case_id":str(i),"group_id":str(i),"initial_answer":"A","candidate_answer":"B"}
                            for i in range(800)],"probabilities":{"ce":[[0,0,1,0]]*800}}
            rows,p = partition(root,data,"calibration")
            self.assertEqual(len(rows),800);self.assertEqual(sum(r["c0"] for r in rows),800)
            self.assertEqual(len(p["ce"]),800)


if __name__ == "__main__": unittest.main()
