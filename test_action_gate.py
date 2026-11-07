import unittest
import numpy as np
import torch

from action_gate import actionable, eligible_rows, project_probabilities, case_rank_loss
from train_gate_baseline import decisions


class ActionGateChecks(unittest.TestCase):
    def row(self, original="A", candidate="B", valid=True):
        return {"case_id": "x", "initial_answer": original, "candidate_answer": candidate,
                "valid": valid, "c0": 1, "ca": 0}

    def test_no_ground_truth_needed_to_identify_action(self):
        row = self.row(); self.assertTrue(actionable(row))
        row.update(c0=0, ca=1, **{"class": 1})
        self.assertTrue(actionable(row))
        self.assertFalse(actionable(self.row(candidate="A")))
        self.assertFalse(actionable(self.row(candidate=None, valid=False)))

    def test_projection_respects_impossible_closed_answer_outcomes(self):
        rows = [self.row(), self.row(original=None), self.row(candidate="A")]
        p = project_probabilities(np.full((3, 4), .25), rows)
        np.testing.assert_allclose(p[0], [1/3, 1/3, 1/3, 0])
        np.testing.assert_allclose(p[1], [.5, .5, 0, 0])
        np.testing.assert_allclose(p[2], [0, 0, 0, 0])  # explicitly unscored, not a posterior

    def test_same_answer_never_accepted_even_with_spurious_wc_score(self):
        row = self.row(candidate="A")
        result = decisions(eligible_rows([row]), [[0, 1, 0, 0]], 1, 0)
        self.assertFalse(result[0]["accepted"])

    def test_rank_case_weight_and_no_duplicate_keep(self):
        logits = torch.zeros((2, 2, 3), requires_grad=True)
        gains = torch.tensor([[1., 0.], [0., 0.]])
        active = torch.tensor([[True, False], [False, False]])
        loss = case_rank_loss(logits, gains, active)
        self.assertAlmostEqual(float(loss), .05, places=6)  # first case .1, second 0
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertEqual(float(logits.grad[1].abs().sum()), 0.)


if __name__ == "__main__":
    unittest.main()
