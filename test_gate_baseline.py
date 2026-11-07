import unittest

from prepare_gate import transition, split_groups, DTO_KEYS
from train_gate_baseline import calibrate, decisions, metrics


class GateBaselineChecks(unittest.TestCase):
    def test_class_order(self):
        self.assertEqual([transition(a, b) for a, b in [(0, 0), (0, 1), (1, 0), (1, 1)]], [0, 1, 2, 3])

    def test_group_assignment_and_encoder_label_boundary(self):
        s = split_groups([f"g{i}" for i in range(200)])
        self.assertEqual([list(s.values()).count(x) for x in ["train", "dev", "calibration", "diagnostic_eval"]],
                         [120, 20, 20, 40])
        self.assertFalse(DTO_KEYS & {"class", "c0", "ca", "reference", "branch", "split", "group_id"})
        with self.assertRaises(ValueError):
            split_groups(["duplicate"]*200)

    def test_invalid_candidate_and_nonpositive_utility_keep(self):
        row = {"case_id": "c", "c0": 1, "ca": 0, "valid": False,
               "candidate_answer": None, "initial_answer": "A"}
        result = decisions([row], [[0, 1, 0, 0]], 1, 0)
        self.assertEqual(result[0]["ca"], 1)
        row["valid"] = True
        self.assertFalse(decisions([row], [[0, .1, .1, .8]], 1, 0)[0]["accepted"])

    def test_calibration_rejects_harm_and_net_identity(self):
        rows = [{"case_id": "c", "c0": 1, "ca": 0, "valid": True,
                 "candidate_answer": "B", "initial_answer": "A"}]
        policy = calibrate(rows, [[0, .9, .1, 0]])
        self.assertTrue(policy["keep_only"])
        result = metrics(decisions(rows, [[0, .9, .1, 0]], policy["lambda"], policy["delta"]))
        self.assertEqual(result["CW"], 0)
        self.assertEqual(result["delta_accuracy"], 0)


if __name__ == "__main__":
    unittest.main()
