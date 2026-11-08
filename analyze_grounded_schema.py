"""Read completed G9 outputs once; diagnostic extraction never admits candidates."""
from collections import Counter
import json
import os
from pathlib import Path


def diagnostic_object(raw):
    text = raw.strip()
    fence = text.startswith("```") and text.endswith("```")
    if fence:
        text = "\n".join(text.splitlines()[1:-1])
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("Not an object")
    return value, fence


def main():
    root = Path(os.environ["JOB_ROOT"])
    result = root/"results"
    result.mkdir(exist_ok=False)
    state = json.loads((root/"outputs/supervisor.json").read_text())
    if state["status"] != "completed":
        raise RuntimeError("Completed outputs required")
    metrics = json.loads((root/"outputs/metrics.json").read_text())
    calls = [json.loads(s) for s in (root/"outputs/calls.private.jsonl").read_text().splitlines()]
    vocab = json.loads((root/"inputs/ontology.private.json").read_text())
    stages = {}
    initial_values = {}
    for stage in ["initial","full_recompute","evidence_summary","local_edit","old_quarantine"]:
        counter = Counter()
        for call in calls:
            if call["stage"] != stage:
                continue
            counter["calls"] += 1
            counter["at_output_cap"] += call["output_tokens"] == call["max_output_tokens"]
            try:
                value, fence = diagnostic_object(call["raw_output"])
                counter["outer_fence_removed_for_diagnosis_only"] += fence
            except (ValueError,TypeError):
                counter["unparsed_for_diagnosis"] += 1
                continue
            values = value.get("values")
            if not isinstance(values,dict):
                counter["no_values_object"] += 1
                continue
            if stage == "initial":
                initial_values[call["case_id"]] = values
            for slot in ["r1.finding","r1.location","r2.finding","r2.location"]:
                tags = values.get(slot)
                counter["tag_cells"] += 1
                if isinstance(tags,list):
                    counter["empty_list_cells"] += not tags
                    if tags:
                        counter["nonempty_list_cells"] += 1
                        allowed = vocab["finding_tags" if slot.endswith("finding") else "location_tags"]
                        counter["nonempty_in_vocab_cells"] += all(isinstance(s,str) and s in allowed for s in tags)
                else:
                    counter["missing_or_nonlist_cells"] += 1
        stages[stage] = dict(counter)
    # Development-only, post-hoc LEAF diagnostic. No answer/parents are filled in,
    # no frozen failure/eligibility/primary outcome is changed, no method compared.
    labels = [json.loads(s) for s in (root/"inputs/labels.private.jsonl").read_text().splitlines()]
    diag = Counter()
    for label in labels:
        values = initial_values.get(label["case_id"],{})
        statuses=[]
        for region in label["regions"]:
            for suffix,key in [("finding","finding_tags"),("location","location_tags")]:
                if suffix == "location" and not region["location_scored"]:
                    continue
                tags = values.get(region["region_id"]+"."+suffix)
                matched = isinstance(tags,list) and all(isinstance(s,str) for s in tags) and set(tags)==set(region[key])
                statuses.append(matched)
                diag["source_scored_slots"] += 1
                diag["source_exact_slots"] += matched
                diag["source_mismatch_slots"] += not matched
        diag["patients"] += 1
        diag["patients_with_mixed_source_agreement"] += any(statuses) and not all(statuses)
    diagnosis = {"frozen_run_status":"completed_with_output_protocol_failure",
        "frozen_initial_schema_valid":metrics["initial_valid_cases"],"original_cases":32,
        "stage_structure_diagnostic":stages,"initial_leaf_source_diagnostic":dict(diag),
        "diagnostic_changes_frozen_admission":False,"diagnostic_model_calls":0,
        "answer_or_parents_imputed":False,"clinical_truth_of_open_text_established":False,
        "primary_preservation_denominator":0,"paired_effect_inference":"unavailable due to no valid initial ledgers"}
    (result/"schema_diagnosis.json").write_text(json.dumps(diagnosis,indent=2))
    (result/"cost.json").write_text(json.dumps(metrics["cost"],indent=2))
    (result/"frozen_accounting.json").write_text(json.dumps({"interpretation":"Null-based accounting under all-schema failure; NOT zero diagnostic accuracy or a valid clinical method comparison. Paired CIs remain private and uninterpreted.",
        "source_metrics":metrics["source_metrics"],"initial_lock_epoch":metrics["initial_lock_epoch"],
        "updates_opened_epoch":metrics["updates_opened_epoch"],"labels_opened_epoch":metrics["labels_opened_epoch"]},indent=2))
    print(json.dumps(diagnosis,indent=2))


if __name__ == "__main__":
    main()
