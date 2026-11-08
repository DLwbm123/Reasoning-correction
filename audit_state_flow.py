"""One read-only G8 calibration audit; never selects on diagnostic-evaluation cases."""
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import random
import time


def main():
    start = time.time()
    base = Path(__file__).resolve().parent
    source = base/"runs/risk_calibration_20261108_v1"
    target = base/"runs/state_flow_audit_20261108_v1"
    private = base/"private_analysis/state_flow_audit_20261108_v1"
    if private.exists(): raise RuntimeError("Audit packet already created; do not repeat")
    assert json.loads((source/"outputs/supervisor.json").read_text())["status"] == "completed"
    split = json.loads((source/"inputs/splits.private.json").read_text())
    rows = [r for r in map(json.loads,(source/"inputs/manifest.jsonl").read_text().splitlines())
            if split[r["group_id"]] == "calibration"]
    labels = {r["case_id"]:r for r in map(json.loads,(source/"inputs/labels.calibration.private.jsonl").read_text().splitlines())}
    assert len(rows) == len(labels) == 800
    counts, kinds, statuses, reasons = Counter(),Counter(),Counter(),Counter()
    groups = defaultdict(list)
    for row in rows:
        if time.time()-start>3600: raise RuntimeError("One-hour read-only audit budget")
        case = json.loads((source/"outputs/cases"/(row["case_id"]+".json")).read_text())
        claims, state, evidence = case.get("claims",[]),case.get("state",{}),case.get("evidence",[])
        verdicts = state.get("verdicts",{})
        supported = [c for c in claims if verdicts.get(c["claim_id"],{}).get("status") == "SUPPORTED"]
        facts = [c for c in supported if c["kind"] in {"visual","context"}]
        counts["cases"]+=1;counts["claims"]+=len(claims)
        counts["claims_with_parents"]+=sum(bool(c["parents"]) for c in claims)
        counts["parent_edges"]+=sum(len(c["parents"]) for c in claims)
        counts["supported_non_fact_kind_claims"]+=sum(c["kind"] not in {"visual","context"} for c in supported)
        counts["cases_with_parent_edges"]+=any(c["parents"] for c in claims)
        counts["graph_changes_k_cases"]+=state.get("flat",[]) != state.get("retained",[])
        kinds.update(c["kind"] for c in claims);statuses.update(v["status"] for v in verdicts.values())
        if not state.get("retained"):
            if case.get("pipeline_error"): reason="pipeline_failure"
            elif not claims: reason="no_normalized_claims"
            elif not any(e["assessability"]=="adequate" for e in evidence): reason="no_adequate_evidence"
            elif not supported: reason="no_program_supported_claims"
            elif not facts: reason="supported_only_excluded_kinds"
            elif not state.get("flat"): reason="no_grounded_text_for_supported_fact"
            else: reason="graph_closure_removed_flat_facts"
            reasons[reason]+=1
        label = labels[row["case_id"]]
        assert label["group_id"] == row["group_id"]
        c0 = case.get("initial",{}).get("answer") == label["reference_letter"]
        ca = case.get("answers",{}).get("selective",{}).get("answer") == label["reference_letter"]
        groups[("C" if c0 else "W")+("C" if ca else "W")].append(row)
    selected=[];rng=random.Random(2026110813)
    for name in ["WW","WC","CW","CC"]:
        bucket=sorted(groups[name],key=lambda r:r["group_id"]);rng.shuffle(bucket)
        selected.extend((name,r) for r in bucket[:2])
    private.mkdir(parents=True)
    total=0;index=[]
    for i,(name,row) in enumerate(selected):
        assert split[row["group_id"]] == "calibration"
        case=json.loads((source/"outputs/cases"/(row["case_id"]+".json")).read_text())
        data={"stratum":name,"manifest":row,"case":case,"closed_answer_reference":labels[row["case_id"]],
              "source_image":str((source/"inputs"/row["image"]).resolve()),"medical_fact_adjudication":"not_available"}
        text=json.dumps(data,ensure_ascii=False,indent=2);total+=len(text.encode())
        if total>50000000: raise RuntimeError("Private audit storage budget")
        filename=str(i+1).zfill(2)+"_"+name+".json";(private/filename).write_text(text)
        index.append({"file":filename,"stratum":name})
    (private/"INDEX.json").write_text(json.dumps(index,indent=2))
    summary={"scope":"G8 calibration only, post-completion, no new inference or expert labels",
        "source_articles":800,"counts":dict(counts),"claim_kind_counts":dict(kinds),
        "program_verdict_counts":dict(statuses),"empty_k_reasons_ordered_mutually_exclusive":dict(reasons),
        "selective_outcome_strata_counts":{k:len(v) for k,v in groups.items()},
        "private_packet_cases":len(selected),"private_packet_strata":dict(Counter(k for k,_ in selected)),
        "private_packet_json_bytes":total,"selection_seed":2026110813,"new_expert_adjudications":0,
        "new_model_calls":0,"new_training":False,"seconds":time.time()-start,
        "limitation":"Bindings/adequacy/support graph are model/program records, not independent medical truth"}
    (target/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2))


if __name__ == "__main__": main()
