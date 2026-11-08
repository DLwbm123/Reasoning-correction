"""Post-completion G8 evidence, cost, discrete paired sensitivity and recovery accounting."""
from collections import Counter
import json
import math
import os
from pathlib import Path
import shutil
import random


def main():
    root = Path(os.environ["RUN_ROOT"])
    out, public = root/"outputs", root/"results"
    done = json.loads((out/"supervisor.json").read_text())
    assert done["status"] == "completed"
    public.mkdir(exist_ok=True)
    m = json.loads((out/"metrics.json").read_text())
    policies = json.loads((out/"risk_policies_locked.json").read_text())
    old = json.loads((out/"old_policies_locked.json").read_text())
    predictions = json.loads((out/"predictions.private.json").read_text())
    paired = json.loads((out/"paired_outcomes.private.json").read_text())
    assert len(paired["case_order"]) == len(set(paired["case_order"])) == 200
    assert old["locked_epoch"] < predictions["finished_epoch"] < policies["locked_epoch"] < m["finished_epoch"]
    cases = paired["outcomes"]
    sensitivity = []
    for name, values in cases.items():
        assert len(values) == 200 and sum(r["ca"] for r in values) == m["results"][name]["final_correct"]
        if name == "pool_oracle_upper_bound": continue
        delta = [r["ca"]-r["c0"] for r in values]
        wc,cw = sum(v>0 for v in delta),sum(v<0 for v in delta)
        discordant = wc+cw
        exact = min(1.,2*sum(math.comb(discordant,i) for i in range(min(wc,cw)+1))/2**discordant) if discordant else 1.
        rng = random.Random(2026110812)
        b = sorted(sum(rng.choices(delta,k=200))/200 for _ in range(2000))
        sensitivity.append({"method":name,"delta_vs_keep":sum(delta)/200,"paired_percentile_95ci":[b[49],b[1949]],
            "discordant_articles":discordant,"exact_two_sided_paired_sign_p":exact,
            "interpretation":"Descriptive, unadjusted; percentile bootstrap can be unstable with few discordant pairs"})
    tables = json.loads((out/"calibration_tables.json").read_text())
    scores = [json.loads(s) for s in (out/"scores_new.jsonl").read_text().splitlines()]
    assert len(scores) == len({r["case_id"] for r in scores}) == 1000
    score_counts = Counter(r["status"] for r in scores)
    assert sum(c["forwards"] for r in scores for c in r["calls"]) == 1998
    decisions = {}
    for name,spec in policies["policies"].items():
        p = spec["policy"]
        decisions[name] = {"lambda":p["lambda"],"delta":p["delta"],"explicit_constant_keep":p["keep_only"],
            "calibration_n":p.get("calibration_metrics",{}).get("n"),
            "calibration_initial_correct":p.get("calibration_metrics",{}).get("initial_correct"),
            "calibration_upper":p.get("calibration_upper"),
            "diagnostic_zero_edits":m["results"][name]["edit_coverage"]==0}
    summary = {"stage_order_valid":True,"calibration_grid_admissible_counts":{k:sum(x["admissible"] for x in v) for k,v in tables.items()},
        "policy_classification":decisions,"diagnostic_sensitivity":sensitivity,
        "score_status_counts":dict(score_counts),"actual_score_forwards":1998,"completed_prefix_not_replayed":272,
        "unscored_invalid_rows":predictions["unscored_invalid_candidate_rows"],
        "postfreeze_recovery_disclosed":True,"failed_score_process_allocator_peak_available":False,
        "new_model_training":False,"scope":"Fixed G8 only; no refitting, threshold changes or clinical risk certification"}
    (public/"risk_analysis.json").write_text(json.dumps(summary,indent=2)+"\n")
    for name in ["metrics.json","risk_policies_locked.json","old_policies_locked.json","calibration_tables.json","scores_new_done.json","collection_done.json"]:
        shutil.copy2(out/name,public/name)
    print(json.dumps(summary,indent=2))


if __name__ == "__main__": main()
