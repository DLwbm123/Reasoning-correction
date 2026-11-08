"""Archive/identity adapter checks; no clinical text or patient files."""
import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from prepare_grounded_sources import SplitArchive, build_task_records, positive_pairs, selected_report, training_ontology


class GroundedSourceChecks(unittest.TestCase):
    def test_zip_reads_across_tiny_volumes_without_assembled_copy(self):
        raw = io.BytesIO()
        with zipfile.ZipFile(raw, "w", compression=zipfile.ZIP_DEFLATED) as z:
            z.writestr("images/x.png", b"registered bytes"*200)
        archive = raw.getvalue()
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for i, start in enumerate(range(0, len(archive), 17)):
                path = Path(tmp)/str(i)
                path.write_bytes(archive[start:start+17]); paths.append(path)
            with SplitArchive(paths) as reader, zipfile.ZipFile(reader) as z:
                self.assertEqual(z.read("images/x.png"), b"registered bytes"*200)
                reader.seek(-10, 2)
                self.assertEqual(reader.read(), archive[-10:])

    def test_only_selected_identity_is_decoded_and_temporal_or_same_labels_excluded(self):
        # An unselected malformed object must not be decoded as an evaluation item.
        text = '[{"StudyID":"heldout",BROKEN},' + json.dumps({"StudyID": "train", "findings": []}) + ']'
        self.assertEqual(selected_report(text, "train")["StudyID"], "train")
        with self.assertRaises(ValueError):
            selected_report(text, "absent")
        common = {"abnormal": True, "boxes": [[0, 0, 1, 1]], "progression": None}
        a = dict(common, labels=["category-a"])
        b = dict(common, labels=["category-b"])
        self.assertEqual(positive_pairs({"findings": [a, b]}), [a, b])
        self.assertIsNone(positive_pairs({"findings": [a, dict(b, progression="new")]}))
        self.assertIsNone(positive_pairs({"findings": [a, dict(b, labels=["category-a"])]}))

    def test_initial_manifest_has_no_finding_text_labels_patient_or_correction_identity(self):
        row = {"case_id": "g0", "patient_id": "PRIVATE_PATIENT", "image_id": "PRIVATE_NAME",
               "correction_target_index": 1, "findings": [
                   {"boxes": [[.1, .1, .2, .2]], "labels": ["PRIVATE_LABEL1"],
                    "locations": [], "sentence_en": "PRIVATE_SENTENCE1"},
                   {"boxes": [[.4, .4, .6, .6]], "labels": ["PRIVATE_LABEL2"],
                    "locations": ["PRIVATE_LOCATION"], "sentence_en": "PRIVATE_SENTENCE2"}]}
        manifests, updates, labels = build_task_records([row])
        for sentinel in ["PRIVATE_PATIENT", "PRIVATE_NAME", "PRIVATE_LABEL1", "PRIVATE_LABEL2",
                         "PRIVATE_SENTENCE1", "PRIVATE_SENTENCE2", "PRIVATE_LOCATION", "correction_target_index"]:
            self.assertNotIn(sentinel, json.dumps(manifests))
        self.assertEqual(updates[0]["region_id"], "r2")
        self.assertFalse(labels[0]["regions"][0]["location_scored"])
        self.assertTrue(labels[0]["regions"][1]["location_scored"])

    def test_ontology_uses_current_training_positive_rows_and_validates_locations(self):
        rows = [{"boxes_count": "1", "label": "finding-a", "locations": "['place-a']"},
                {"boxes_count": "0", "label": "unlocalized", "locations": "[]"}]
        value = training_ontology(rows)
        self.assertEqual(value["finding_tags"], ["finding-a"])
        self.assertEqual(value["location_tags"], ["place-a"])
        with self.assertRaises(ValueError):
            training_ontology([dict(rows[0], locations="{'not': 'a list'}")])


if __name__ == "__main__":
    unittest.main()
