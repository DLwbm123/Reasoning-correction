"""Check the generic offline preparation adapter without images/model access."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from prepare_gate import prepare_pool


class PoolPreparation(unittest.TestCase):
    def test_same_root_no_copy_preserves_failure_and_group_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/"inputs").mkdir(); (root/"outputs/cases").mkdir(parents=True)
            manifest = [{"case_id": "c1", "group_id": "g1", "image": "images/1.jpg",
                         "question": "q", "choices": {"A": "x", "B": "y"}},
                        {"case_id": "c2", "group_id": "g2", "image": "images/2.jpg",
                         "question": "q", "choices": {"A": "x", "B": "y"}}]
            labels = [{"case_id": r["case_id"], "group_id": r["group_id"], "reference_letter": "A"}
                      for r in manifest]
            for name, rows in [("manifest.jsonl", manifest), ("labels.private.jsonl", labels)]:
                (root/"inputs"/name).write_text("\n".join(map(json.dumps, rows)))
            (root/"outputs/cases/c1.json").write_text(json.dumps({"case_id": "c1", "group_id": "g1",
                "initial": {"answer": "A"}, "answers": {"selective": {"answer": "B"}, "reset": {"answer": "A"}}}))
            (root/"outputs/cases/c2.json").write_text(json.dumps({"case_id": "c2", "group_id": "g2",
                "pipeline_error": "synthetic failure"}))
            with contextlib.redirect_stdout(io.StringIO()):
                prepare_pool(root, root, {"g1": "train", "g2": "dev"}, copy_images=False,
                             split_seed=9, retrospective=False)
            targets = [json.loads(x) for x in (root/"inputs/supervision.private.jsonl").read_text().splitlines()]
            self.assertEqual([r["class"] for r in targets], [2, 3, 0, 0])
            self.assertEqual([r["valid"] for r in targets], [True, True, False, False])
            dto = [json.loads(x) for x in (root/"inputs/features.jsonl").read_text().splitlines()]
            self.assertEqual(len(dto), 4)
            self.assertNotIn("reference_letter", json.dumps(dto))
            self.assertNotIn("synthetic failure", json.dumps(dto))
            self.assertFalse(json.loads((root/"preparation.json").read_text())["retrospective_development_only"])


if __name__ == "__main__":
    unittest.main()
