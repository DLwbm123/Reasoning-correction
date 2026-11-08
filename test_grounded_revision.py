"""Necessary G9 image/DTO/patch binding tests; no dataset annotations."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from grounded_revision import (SLOTS, initial_dto, full_result, revision_dto, patch_result,
                               copy_control, render_image, parse_object)


class GroundedBinding(unittest.TestCase):
    def setUp(self):
        self.row = {"case_id":"g0","view_id":"image_0","image":"images/g0.png","regions":[
            {"region_id":"r1","boxes":[[.1,.1,.2,.2]]},{"region_id":"r2","boxes":[[.4,.4,.8,.8]]}]}
        self.vocab = {"finding_tags":["finding-a","finding-b"],"location_tags":["place-a"],"source":"TRAIN_ONLY"}
        self.report = {"values":{s:[] for s in SLOTS},"parents":{s:[] for s in SLOTS}}
        self.report["values"].update({"r1.finding":["finding-a"],"r2.finding":["finding-b"],"answer":"Unreviewed short summary"})
        self.report["parents"]["answer"] = ["r1.finding","r2.finding"]
        self.original = full_result(json.dumps(self.report),self.row,self.vocab)
        self.update = {"case_id":"g0","region_id":"r1","view_id":"image_0","source_id":"image_0",
            "source_type":"released_dataset_annotation","sentence_en":"a released source phrase",
            "finding_tags":["finding-b"],"location_tags":[],"location_scored":False}

    def test_initial_dto_excludes_update_original_patient_and_ontology_provenance(self):
        data = initial_dto(self.row,self.vocab)
        self.assertEqual(set(data),{"case_id","view_id","regions","ontology"})
        self.assertNotIn("source",data["ontology"])
        bad = dict(self.row,reference="PRIVATE_REFERENCE")
        with self.assertRaises(ValueError): initial_dto(bad,self.vocab)

    def test_release_identity_and_no_unreleased_region_reference(self):
        data = revision_dto(self.row,self.vocab,self.original,self.update,"full_recompute")
        self.assertEqual(data["source_update"]["region_id"],"r1")
        self.assertEqual(data["original_assertions"],self.original["assertions"])
        for field,value in [("case_id","another"),("view_id","another"),("source_id","another"),("reference","PRIVATE_OTHER_REGION")]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                revision_dto(self.row,self.vocab,self.original,dict(self.update,**{field:value}),"full_recompute")

    def test_span_selection_binds_exact_source_and_never_admits_semantics(self):
        data = revision_dto(self.row,self.vocab,self.original,self.update,"patch_graph")
        ops = [{"op":"KEEP","target_id":s} for s in SLOTS]
        new = copy.deepcopy(self.original["assertions"][0]);new.update(id="new-r1",text='["finding-b"]')
        ops[0] = {"op":"REPLACE","target_id":"r1.finding","assertion":new}
        raw = {"case_id":"g0","view_id":"image_0","operations":ops,"supports":[
            {"assertion_id":"new-r1","evidence_id":"review","span_id":"whole","relation":"SUPPORTED"}]}
        result = patch_result(json.dumps(raw),data,self.vocab)
        self.assertEqual(result["values"]["r1.finding"],["finding-b"])
        self.assertFalse(result["semantic_eligible"])
        self.assertEqual(result["transaction"]["candidate"]["support_proposals"][0]["quote"],data["evidence"][0]["text"])
        raw["supports"][0]["span_id"] = "invented"
        with self.assertRaises(ValueError): patch_result(json.dumps(raw),data,self.vocab)
        raw["supports"][0].update(span_id="whole",evidence_id="image_registry")
        with self.assertRaises(ValueError): patch_result(json.dumps(raw),data,self.vocab)

    def test_graph_ablation_and_quarantine_are_actual_representation_changes(self):
        no_graph=revision_dto(self.row,self.vocab,self.original,self.update,"patch_no_graph")
        graph=revision_dto(self.row,self.vocab,self.original,self.update,"patch_graph")
        self.assertTrue(graph["original_assertions"][-1]["proposed_parents"])
        self.assertFalse(any(a["proposed_parents"] for a in no_graph["original_assertions"]))
        old=revision_dto(self.row,self.vocab,self.original,self.update,"old_quarantine")
        self.assertFalse(old["original_assertions"])
        self.assertIn("answer",old["isolated_ids"])
        self.assertTrue(self.original["assertions"][-1]["proposed_parents"])

    def test_full_control_can_run_without_initial_ledger_and_copy_never_sees_other_truth(self):
        data=revision_dto(self.row,self.vocab,None,self.update,"full_recompute")
        self.assertEqual(data["original_status"],"unavailable_not_reconstructed")
        control=copy_control(None,self.update)
        self.assertEqual(control["values"]["r1.finding"],["finding-b"])
        self.assertIsNone(control["values"]["r2.finding"])
        control=copy_control(self.original,self.update)
        self.assertEqual(control["values"]["r2.finding"],self.original["values"]["r2.finding"])

    def test_strict_tags_and_complete_slots_reject_unknown_duplicate_or_missing_fields(self):
        for value in [["not-in-vocab"],["finding-a","finding-a"],"finding-a"]:
            report=copy.deepcopy(self.report);report["values"]["r1.finding"]=value
            with self.assertRaises(ValueError): full_result(json.dumps(report),self.row,self.vocab)
        with self.assertRaises(ValueError): parse_object('{"x":1,"x":2}')

    def test_real_png_16bit_mapping_preserves_dynamic_range_instead_of_rgb_clipping(self):
        import numpy as np
        from PIL import Image
        with tempfile.TemporaryDirectory() as tmp:
            source,target=Path(tmp)/"in.png",Path(tmp)/"out.png"
            Image.fromarray(np.array([[1000,2000],[3000,4000]],dtype=np.uint16)).save(source)
            record=render_image(source,target,[])
            with Image.open(target) as im: values=np.asarray(im)
            self.assertEqual(values[:,:,0].tolist(),[[0,85],[170,255]])
            self.assertEqual(record["native_min"],1000)
            self.assertEqual(record["native_max"],4000)

    def test_missing_phase_locks_prevent_opening_update_or_eval_labels(self):
        from grounded_worker import deferred_updates
        from evaluate_grounded_revision import evaluate
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/"inputs").mkdir();(root/"outputs").mkdir()
            (root/"inputs/updates.private.jsonl").write_text('NOT JSON: PRIVATE_UPDATE')
            (root/"inputs/labels.private.jsonl").write_text('NOT JSON: PRIVATE_EVALUATION')
            with self.assertRaises(FileNotFoundError): deferred_updates(root,[str(i) for i in range(32)])
            with self.assertRaises(RuntimeError): evaluate(root)

    def test_metrics_distinguish_wrong_assertion_loss_and_failed_candidate_fallback(self):
        from grounded_revision import METHODS
        from evaluate_grounded_revision import evaluate
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/"inputs").mkdir();(root/"outputs/initial").mkdir(parents=True);(root/"outputs/revisions").mkdir()
            (root/"outputs/initial_done.json").write_text(json.dumps({"cases":32,"valid":32,"locked_epoch":1}))
            (root/"outputs/updates_opened.json").write_text(json.dumps({"epoch":2}))
            (root/"outputs/collection_done.json").write_text('{}')
            (root/"outputs/calls.private.jsonl").write_text('')
            labels,updates=[],[]
            for i in range(32):
                cid=f"t{i}";old=copy.deepcopy(self.original["values"]);old["r2.finding"]=["finding-a"];old["r2.location"]=["place-a"]
                labels.append({"case_id":cid,"regions":[{"region_id":"r1","finding_tags":["finding-b"],"location_tags":[],"location_scored":False},
                    {"region_id":"r2","finding_tags":["finding-a"],"location_tags":["place-a"],"location_scored":True}]})
                updates.append(dict(self.update,case_id=cid))
                (root/"outputs/initial"/(cid+".json")).write_text(json.dumps({"valid":True,"result":{"values":old}}))
                copied=copy.deepcopy(old);copied["r1.finding"]=["finding-b"]
                bundle={"copy_control":{"values":copied},"methods":{m:{"valid":True,"result":{"values":old}} for m in METHODS}}
                wrong=copy.deepcopy(copied);wrong["r2.finding"]=["finding-b"];wrong["r2.location"]=[]
                bundle["methods"]["full_recompute"]={"valid":True,"result":{"values":wrong}}
                bundle["methods"]["local_edit"]={"valid":False,"result":None}
                (root/"outputs/revisions"/(cid+"_bundle.json")).write_text(json.dumps(bundle))
            for name,records in [("labels.private.jsonl",labels),("updates.private.jsonl",updates)]:
                (root/"inputs"/name).write_text(''.join(json.dumps(r)+'\n' for r in records))
            evaluate(root)
            result=json.loads((root/"outputs/metrics.json").read_text())["source_metrics"]
            self.assertEqual(result["copy_control"]["target_wrong_repaired"],32)
            self.assertEqual(result["full_recompute"]["other_correct_changed_to_nonempty_wrong"],32)
            self.assertEqual(result["full_recompute"]["other_correct_lost_or_unresolved"],32)
            self.assertEqual(result["local_edit"]["candidate_raw_exact_slots"],0)
            self.assertEqual(result["local_edit"]["source_exact_slots"],64)
            self.assertEqual(result["local_edit"]["fallback_cases"],32)
            with self.assertRaises(RuntimeError): evaluate(root)


if __name__ == "__main__": unittest.main()
