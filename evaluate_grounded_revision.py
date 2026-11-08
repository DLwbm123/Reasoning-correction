"""Only after all G9 outputs are frozen, read private source labels once."""
import json
import os
from pathlib import Path
import random
import time

from grounded_revision import METHODS, SLOTS


def equal(value, truth):
    return isinstance(value, list) and set(value) == set(truth)


def evaluate(root):
    if not (root/"outputs/collection_done.json").exists() or (root/"outputs/metrics.json").exists():
        raise RuntimeError("All outputs required; no repeated evaluation")
    initial_lock = json.loads((root/"outputs/initial_done.json").read_text())
    opened = json.loads((root/"outputs/updates_opened.json").read_text())
    if initial_lock["cases"] != 32 or initial_lock["locked_epoch"] > opened["epoch"]:
        raise ValueError("Deferred-update phase isolation failed")
    labels_opened = time.time()
    labels = [json.loads(s) for s in (root/"inputs/labels.private.jsonl").read_text().splitlines()]
    updates = {r["case_id"]:r for r in [json.loads(s) for s in (root/"inputs/updates.private.jsonl").read_text().splitlines()]}
    if len(labels) != 32:
        raise ValueError("Original 32-case denominator required")
    metrics, vectors, private = {}, {}, []
    methods = ["initial", "copy_control", *METHODS]
    for method in methods:
        metrics[method] = {"cases":32,"source_exact_slots":0,"source_scored_slots":0,"target_initial_wrong":0,
            "target_wrong_repaired":0,"other_initial_correct":0,"other_correct_lost_or_wrong":0,
            "other_correct_changed_to_nonempty_wrong":0,"other_correct_lost_or_unresolved":0,
            "other_initial_wrong":0,"other_wrong_repaired":0,"schema_valid_cases":0,
            "candidate_raw_exact_slots":0,"fallback_cases":0,"unresolved_slots":0,"changed_scored_slots":0,
            "serialized_value_changes":0,"target_initial_nonempty_wrong":0,"target_nonempty_wrong_repaired":0,
            "target_initial_unresolved":0}
        vectors[method] = []
    for label in labels:
        cid = label["case_id"]
        initial = json.loads((root/"outputs/initial"/(cid+".json")).read_text())
        bundle = json.loads((root/"outputs/revisions"/(cid+"_bundle.json")).read_text())
        original = initial["result"]["values"] if initial["valid"] else {slot:None for slot in SLOTS}
        truth = {}
        for r in label["regions"]:
            truth[r["region_id"]+".finding"] = r["finding_tags"]
            if r["location_scored"]:
                truth[r["region_id"]+".location"] = r["location_tags"]
        target = updates[cid]["region_id"]
        detail = {"case_id":cid,"initial_valid":initial["valid"],"methods":{}}
        for method in methods:
            if method == "initial":
                valid, raw_values = initial["valid"], original
            elif method == "copy_control":
                valid, raw_values = True, bundle["copy_control"]["values"]
            else:
                result = bundle["methods"][method]
                valid = result["valid"]
                raw_values = result["result"]["values"] if valid else {slot:None for slot in SLOTS}
            # Schema-failure fallback is explicit; source agreement is not deployment admission.
            values = raw_values if valid or method in {"initial","copy_control"} else original
            m = metrics[method]
            m["schema_valid_cases"] += valid
            m["fallback_cases"] += not valid and method not in {"initial","copy_control"}
            exact, raw_exact = [], []
            for slot, reference in truth.items():
                old, value = original[slot], values[slot]
                before, after = equal(old, reference), equal(value, reference)
                exact.append(int(after)); raw_exact.append(int(equal(raw_values[slot],reference)))
                m["source_scored_slots"] += 1
                m["source_exact_slots"] += after
                m["candidate_raw_exact_slots"] += raw_exact[-1]
                m["unresolved_slots"] += value is None or value == []
                m["serialized_value_changes"] += old != value
                m["changed_scored_slots"] += (set(old) if isinstance(old,list) else None) != (set(value) if isinstance(value,list) else None)
                if slot.startswith(target+"."):
                    m["target_initial_wrong"] += not before
                    m["target_wrong_repaired"] += not before and after
                    m["target_initial_nonempty_wrong"] += not before and old is not None and old != []
                    m["target_nonempty_wrong_repaired"] += not before and old is not None and old != [] and after
                    m["target_initial_unresolved"] += old is None or old == []
                else:
                    m["other_initial_correct"] += before
                    m["other_correct_lost_or_wrong"] += before and not after
                    m["other_correct_changed_to_nonempty_wrong"] += before and not after and value is not None and value != []
                    m["other_correct_lost_or_unresolved"] += before and (value is None or value == [])
                    m["other_initial_wrong"] += not before
                    m["other_wrong_repaired"] += not before and after
            case_mean = sum(exact)/len(exact)
            vectors[method].append(case_mean)
            detail["methods"][method] = {"source_exact":sum(exact),"source_denominator":len(exact),"raw_exact":sum(raw_exact),"schema_valid":valid}
        private.append(detail)
    for method, m in metrics.items():
        m["source_slot_accuracy"] = m["source_exact_slots"]/m["source_scored_slots"]
        m["patient_mean_source_accuracy"] = sum(vectors[method])/32
    metrics["copy_control"]["schema_valid_cases"] = None
    comparisons = {}
    rng = random.Random(2026110817)
    samples = [[rng.randrange(32) for _ in range(32)] for _ in range(2000)]
    for method in METHODS:
        for control in ["copy_control", "full_recompute", "local_edit"]:
            if method == control:
                continue
            diff = [a-b for a,b in zip(vectors[method], vectors[control])]
            draws = sorted(sum(diff[i] for i in ids)/32 for ids in samples)
            comparisons[method+"_minus_"+control] = {"patient_mean_difference":sum(diff)/32,
                "percentile95":[draws[49],draws[1949]],"patients":32,"bootstrap_draws":2000,
                "descriptive_no_multiplicity_correction":True}
    calls = [json.loads(s) for s in (root/"outputs/calls.private.jsonl").read_text().splitlines()]
    result = {"source_metrics":metrics,"paired_comparisons":comparisons,"initial_valid_cases":initial_lock["valid"],
        "cost":{"calls":len(calls),"forwards":sum(r["forwards"] for r in calls),
                "input_tokens":sum(r["input_tokens"] for r in calls),"output_tokens":sum(r.get("output_tokens",0) for r in calls),
                "visual_tokens":sum(r["visual_tokens"] for r in calls),"generation_seconds":sum(r["seconds"] for r in calls)},
        "initial_lock_epoch":initial_lock["locked_epoch"],"updates_opened_epoch":opened["epoch"],
        "labels_opened_epoch":labels_opened,"finished_epoch":time.time(),"clinical_expert_adjudications_added":0,
        "summary_semantic_accuracy":"not_adjudicated", "causal_dependency_accuracy":"not_adjudicated"}
    with (root/"outputs/metrics.json").open("x") as f:
        json.dump(result,f,indent=2)
    with (root/"outputs/evaluation_rows.private.json").open("x") as f:
        json.dump(private,f,indent=2)
    print("EVALUATION_COMPLETE",len(labels),"CASES",flush=True)


if __name__ == "__main__":
    evaluate(Path(os.environ["RUN_ROOT"]))
