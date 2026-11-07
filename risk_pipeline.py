"""Frozen predictors, physically separated calibration/evaluation labels, fixed risk rules."""
from collections import defaultdict
import json
import os
from pathlib import Path
import random
import time

from evidence_score import auxiliary, choose_by_support
from prepare_gate import transition
from risk_control import apply_policy, select_policy, upper_bound
from train_gate_baseline import metrics, calibrate


def read_lines(path):
    return [json.loads(s) for s in path.read_text().splitlines()]


def save(root, name, value):
    path = root/"outputs"/name
    if path.exists(): raise RuntimeError("Frozen artifact already exists: "+name)
    path.write_text(json.dumps(value, indent=2))


def probabilities(root, rows, scores):
    import torch
    torch.set_num_threads(4)
    from support_gate import score_net
    prior = Path((root/"predictor_root.txt").read_text().strip())
    scored_indices = [i for i,r in enumerate(rows) if scores[r["case_id"]]["status"] == "completed"]
    for i,r in enumerate(rows):
        if i not in scored_indices:
            assert scores[r["case_id"]]["status"] == "unavailable_non_actionable" and not r["valid"]
    if not scored_indices: return {config:[None]*len(rows) for config in ["ce","ce_rank"]}
    x = torch.tensor([auxiliary(scores[rows[i]["case_id"]], rows[i]["initial_answer"], rows[i]["candidate_answer"]) for i in scored_indices], dtype=torch.float32)
    result = {}
    for config in ["ce", "ce_rank"]:
        values = []
        for seed in [2026110711, 2026110712, 2026110713]:
            ck = torch.load(prior/"outputs"/("score_"+config+"_"+str(seed)+".pt"), map_location="cpu", weights_only=True)
            assert ck["dimension"] == 10
            net = score_net(10).eval(); net.load_state_dict(ck["state_dict"]); net.requires_grad_(False)
            with torch.inference_mode():
                values.append((net((x-ck["mean"])/ck["std"])/ck["temperature"]).softmax(-1))
        result[config] = [None]*len(rows)
        for i,value in zip(scored_indices,torch.stack(values).mean(0).tolist()): result[config][i] = value
    return result


def boot(root):
    shared = Path((root/"shared_root.txt").read_text().strip())
    old_scores = Path((root/"old_scores_root.txt").read_text().strip())
    rows = [r for r in read_lines(shared/"inputs/supervision.private.jsonl") if r["split"] == "calibration"]
    assert len(rows) == 400 and len({r["case_id"] for r in rows}) == 200
    sc = {r["case_id"]: r for r in read_lines(old_scores/"outputs/scores_old.jsonl")}
    p = probabilities(root, rows, sc)
    save(root, "old_policies_locked.json", {"locked_epoch": time.time(),
        "policies": {c: calibrate(rows, p[c]) for c in p},
        "ensemble_seeds": [2026110711,2026110712,2026110713],
        "new_calibration_or_evaluation_labels_read": False})
    print("OLD_EMPIRICAL_POLICIES_LOCKED", 2, flush=True)


def label_free_rows(root):
    done = json.loads((root/"outputs/collection_done.json").read_text())
    assert done["status"] == "completed" and done["cases"] == 1000
    manifest = read_lines(root/"inputs/manifest.jsonl")
    assert len(manifest) == 1000
    rows = []
    for r in manifest:
        case = json.loads((root/"outputs/cases"/(r["case_id"]+".json")).read_text())
        assert case["case_id"] == r["case_id"] and case["group_id"] == r["group_id"]
        initial = case.get("initial", {}).get("answer")
        for branch in ["selective", "reset"]:
            candidate = case.get("answers", {}).get(branch, {})
            rows.append({"case_id": r["case_id"], "group_id": r["group_id"], "branch": branch,
                "initial_answer": initial, "candidate_answer": candidate.get("answer"),
                "valid": candidate.get("answer") in list("ABCD"),
                "explanation_missing": not bool(candidate.get("claims"))})
    return rows


def predict(root):
    assert (root/"outputs/old_policies_locked.json").exists()
    done = json.loads((root/"outputs/scores_new_done.json").read_text())
    sc = {r["case_id"]: r for r in read_lines(root/"outputs/scores_new.jsonl")}
    assert done["cases"] == len(sc) == 1000
    rows = label_free_rows(root)
    p = probabilities(root, rows, sc)
    save(root, "predictions.private.json", {"rows": rows, "probabilities": p,
        "unscored_invalid_candidate_rows":sum(v is None for v in p["ce"]),
        "finished_epoch": time.time(), "new_labels_read": False})
    print("LABEL_FREE_PREDICTIONS_COMPLETE", len(rows), flush=True)


def partition(root, data, split_name):
    split = json.loads((root/"inputs/splits.private.json").read_text())
    indices = [i for i, r in enumerate(data["rows"]) if split[r["group_id"]] == split_name]
    # Only the selected physical label file is ever opened by this call.
    labels = {r["case_id"]: r for r in read_lines(root/"inputs"/("labels."+split_name+".private.jsonl"))}
    assert len(labels) == (800 if split_name == "calibration" else 200)
    assert {data["rows"][i]["case_id"] for i in indices} == set(labels)
    rows = []
    for i in indices:
        r = data["rows"][i]; label = labels[r["case_id"]]
        assert label["group_id"] == r["group_id"]
        c0 = int(r["initial_answer"] == label["reference_letter"])
        ca = int(r["candidate_answer"] == label["reference_letter"])
        rows.append(dict(r, c0=c0, ca=ca, **{"class": transition(c0,ca)}))
    return rows, {c: [p[i] for i in indices] for c, p in data["probabilities"].items()}


def calibrate_new(root):
    data = json.loads((root/"outputs/predictions.private.json").read_text())
    rows, p = partition(root, data, "calibration")
    old = json.loads((root/"outputs/old_policies_locked.json").read_text())
    policies, tables = {}, {}
    for config in ["ce", "ce_rank"]:
        policies[config+"_old_empirical"] = {"config": config, "policy": old["policies"][config]}
        for name, exact in [("new_empirical",False),("new_cp",True)]:
            policy, table = select_policy(rows, p[config], exact=exact)
            policies[config+"_"+name] = {"config": config, "policy": policy}
            tables[config+"_"+name] = table
    save(root, "risk_policies_locked.json", {"locked_epoch": time.time(), "policies": policies,
        "calibration_articles": 800, "simultaneous_rules": 24, "per_rule_alpha": .05/24,
        "family_error_budget": .05, "evaluation_labels_read": False})
    save(root, "calibration_tables.json", tables)
    print("NEW_RISK_POLICIES_LOCKED", len(policies), flush=True)


def evaluate(root):
    # This guard precedes access to the physically isolated evaluation labels.
    if (root/"outputs/metrics.json").exists(): raise RuntimeError("Completed evaluation cannot be repeated")
    locked = json.loads((root/"outputs/risk_policies_locked.json").read_text())
    assert locked["evaluation_labels_read"] is False
    data = json.loads((root/"outputs/predictions.private.json").read_text())
    rows, p = partition(root, data, "diagnostic_eval")
    assert locked["locked_epoch"] > data["finished_epoch"]
    outcomes, results = {}, {}
    for name, spec in locked["policies"].items():
        cases = apply_policy(rows, p[spec["config"]], spec["policy"])
        m = metrics(cases)
        m["descriptive_single_policy_one_sided_95_upper"] = 0. if spec["policy"].get("keep_only") else upper_bound(m["CW"],m["initial_correct"])
        results[name], outcomes[name] = m, cases
    grouped = defaultdict(list)
    for r in rows: grouped[r["case_id"]].append(r)
    sc = {r["case_id"]: r for r in read_lines(root/"outputs/scores_new.jsonl")}
    for name in ["keep","selective","reset","pool_oracle_upper_bound","fixed_without_evidence","fixed_with_evidence"]:
        cases = []
        for cid, pair in grouped.items():
            base = pair[0]; chosen = None
            if name in ["selective","reset"]:
                chosen = next(r for r in pair if r["branch"] == name)
            elif name.startswith("fixed_"):
                if sc[cid]["status"] == "unavailable_non_actionable":
                    assert all(not r["valid"] for r in pair)
                    index = None
                else: index = choose_by_support(pair,sc[cid][name.removeprefix("fixed_")]["logp"])
                chosen = pair[index] if index is not None else None
            ca = max([base["c0"]]+[r["ca"] for r in pair]) if name == "pool_oracle_upper_bound" else chosen["ca"] if chosen else base["c0"]
            cases.append({"c0":base["c0"],"ca":ca,"accepted":chosen is not None,
                          "edited":bool(chosen and chosen["candidate_answer"] != base["initial_answer"])})
        results[name],outcomes[name] = metrics(cases),cases
        if name == "pool_oracle_upper_bound":
            results[name].pop("acceptance_coverage");results[name].pop("edit_coverage")
    paired = []
    for config in ["ce","ce_rank"]:
        for a,b in [(config+"_new_cp",config+"_new_empirical"),(config+"_new_cp",config+"_old_empirical"),
                    (config+"_new_empirical",config+"_old_empirical"),(config+"_new_cp","keep")]:
            delta = [x["ca"]-y["ca"] for x,y in zip(outcomes[a],outcomes[b])]
            rng = random.Random(2026110812)
            samples = sorted(sum(rng.choices(delta,k=200))/200 for _ in range(2000))
            paired.append({"method":a,"control":b,"delta_accuracy":sum(delta)/200,"paired_95ci":[samples[49],samples[1949]]})
    save(root,"metrics.json",{"status":"completed","n":200,"results":results,"paired_comparisons":paired,
        "finished_epoch":time.time(),"new_models_trained":0,"clinical_safety_claimed":False})
    save(root,"paired_outcomes.private.json",{"case_order":list(grouped),"outcomes":outcomes})
    print("RISK_EVALUATION_COMPLETE",flush=True)


if __name__ == "__main__":
    root = Path(os.environ["RUN_ROOT"])
    stage = os.environ["RISK_STAGE"]
    artifact = {"boot":"old_policies_locked.json","predict":"predictions.private.json",
                "calibrate":"risk_policies_locked.json","evaluate":"metrics.json"}[stage]
    if (root/"outputs"/artifact).exists(): raise RuntimeError("Stage already completed; no repeat")
    {"boot":boot,"predict":predict,"calibrate":calibrate_new,"evaluate":evaluate}[stage](root)
