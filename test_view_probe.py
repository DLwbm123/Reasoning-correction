"""G10 source/view binding checks, fabricated fields and images only."""
import json
import os
from pathlib import Path
import tempfile
import unittest

from view_probe import crop_rectangle,probe_dto,extract_requested,evaluate


class ViewBinding(unittest.TestCase):
    def test_crop_is_registered_union_in_native_frame(self):
        self.assertEqual(crop_rectangle([[.1,.2,.4,.5],[.6,.3,.8,.9]],(100,200)),(10,40,80,180))
        with self.assertRaises(ValueError):crop_rectangle([[.8,0,.2,1]],(100,100))

    def test_same_metadata_and_ontology_without_labels_or_outcome_fields(self):
        row={"case_id":"v0","view_id":"image_0","image":"images/v0.png","regions":[{"region_id":"r1","boxes":[[0,0,1,1]]}]}
        vocab={"finding_tags":["tag"],"location_tags":["place"],"source":"TRAIN_ONLY"}
        provenance={"union_box":[0,0,100,100],"native_size":[100,100],"full_size":[100,100]}
        a=probe_dto(row,"r1","duplicate_full",vocab,provenance);b=probe_dto(row,"r1","full_plus_crop",vocab,provenance)
        self.assertEqual(a["ontology"],b["ontology"])
        self.assertEqual(a["regions"],b["regions"])
        self.assertEqual(a["image_manifest"][1]["parent_source_id"],"image_0")
        self.assertNotIn("source",a["ontology"])
        with self.assertRaises(ValueError):probe_dto(dict(row,reference="SECRET"),"r1","duplicate_full",vocab,provenance)

    def test_only_requested_region_is_taken_and_unknown_values_are_not_filtered(self):
        raw='```json\n{"values":{"r1.finding":["unverified out-of-vocabulary"],"r1.location":[],"r2.finding":["unsolicited"]}}\n```'
        result=extract_requested(raw,"v0","r1","call-1")
        self.assertEqual(set(result["values"]),{"r1.finding","r1.location"})
        self.assertEqual(result["values"]["r1.finding"],["unverified out-of-vocabulary"])
        self.assertFalse(result["semantic_eligible"])
        self.assertEqual(result["transport"]["view_id"],"image_0")
        self.assertEqual(result["transport"]["facts"]["r2.finding"]["value"],["unsolicited"])

    def test_eval_cannot_open_labels_without_all_predictions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/"outputs").mkdir();(root/"inputs").mkdir()
            (root/"inputs/labels.private.jsonl").write_text('NOT JSON: PRIVATE_REFERENCE')
            with self.assertRaises(FileNotFoundError):evaluate(root)

    def test_actual_cached_processor_accepts_two_images_without_loading_a_model(self):
        from PIL import Image
        from transformers import AutoProcessor
        path=Path(os.environ["RUN_ROOT"],"model_path.txt").read_text().strip()
        processor=AutoProcessor.from_pretrained(path,local_files_only=True,min_pixels=128*28*28,max_pixels=384*28*28)
        images=[Image.new("RGB",(64,64),"gray"),Image.new("RGB",(32,64),"gray")]
        messages=[{"role":"user","content":[{"type":"image","image":im} for im in images]+[{"type":"text","text":"Describe the supplied views."}]}]
        text=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        batch=processor(text=[text],images=images,return_tensors="pt")
        self.assertEqual(batch["image_grid_thw"].shape[0],2)
        self.assertGreater(batch["pixel_values"].numel(),0)


if __name__=="__main__":unittest.main()
