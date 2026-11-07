import math
import unittest
from evidence_score import auxiliary, score_payload, SCORE_KEYS, choose_by_support


class ScoreChecks(unittest.TestCase):
    def test_no_answers_or_reference_in_dto(self):
        row = {"question": "q", "choices": dict(zip("ABCD", "abcd")), "reference": "B", "answer": "C"}
        self.assertEqual(set(score_payload(row, [{"observation": "x"}], True)), SCORE_KEYS)
        self.assertEqual(score_payload(row, [{"observation": "x"}], False)["evidence"], [])

    def test_identical_answer_and_invalid_flags(self):
        s = {"without_evidence": {"logp": [-1., -2., -3., -4.], "entropy": 1.},
             "with_evidence": {"logp": [-2., -1., -3., -4.], "entropy": .9}}
        v = auxiliary(s, "B", "B")
        self.assertEqual(v[6:], [0., 0., 1., 1.])
        v = auxiliary(s, None, "A")
        self.assertEqual(v[:4], [0., -1., 0., -2.])
        self.assertEqual(v[-2:], [0., 1.])
        self.assertTrue(all(math.isfinite(x) for x in v))

    def test_fixed_policy_ignores_labels_ties_and_invalid(self):
        pair = [{"initial_answer": "A", "candidate_answer": "B", "valid": True, "ca": 0},
                {"initial_answer": "A", "candidate_answer": "C", "valid": True, "ca": 1}]
        self.assertEqual(choose_by_support(pair, [-2., -1., -1., -3.]), 0)
        pair[0]["ca"], pair[1]["ca"] = 1, 0
        self.assertEqual(choose_by_support(pair, [-2., -1., -1., -3.]), 0)
        self.assertIsNone(choose_by_support(pair, [-1., -1., -2., -3.]))
        pair[0]["valid"] = False
        self.assertEqual(choose_by_support(pair, [-2., -1., -1., -3.]), 1)


if __name__ == "__main__": unittest.main()
