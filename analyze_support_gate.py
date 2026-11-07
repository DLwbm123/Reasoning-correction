"""Fixed G6 paired results plus descriptive old train/dev/cal support-score diagnosis."""
from collections import defaultdict
import json
import os
from pathlib import Path
import statistics

from analyze_action_gate import interval
from diagnose_gate_scores import auc


def main():
    root = Path(os.environ["RUN_ROOT"])
    out, public = root/"outputs", root/"results"
    assert json.loads((out/"supervisor.json").read_text())["status"] == "completed"
    public.mkdir(exist_ok=True)
    result = json.loads((out/"metrics.json").read_text())
    paired = json.loads((out/"paired_outcomes.private.json").read_text())
    assert len(paired["case_order"]) == len(set(paired["case_order"])) == 200
    cases = paired["outcomes"]
    families = defaultdict(list)
    for item in result["models"]:
        values = cases[item["name"]]
        assert len(values) == 200 and sum(v["ca"] for v in values) == item["metrics"]["final_correct"]
        families[item["family"]].append(item)
    summary = {"families": {}, "paired_comparisons": [],
               "scope": "Full 200 articles, all seeds; no refitting/selection on new diagnostic labels"}
    for family, items in families.items():
        summary["families"][family] = {
            "accuracy_each_seed": [x["metrics"]["accuracy"] for x in items],
            "mean_accuracy": statistics.mean(x["metrics"]["accuracy"] for x in items),
            "WC_each_seed": [x["metrics"]["WC"] for x in items],
            "CW_each_seed": [x["metrics"]["CW"] for x in items],
            "same_200_articles_shared_across_seeds": True}
    for config in ["ce", "ce_rank"]:
        deltas, keep_deltas = [], []
        for seed in [2026110711, 2026110712, 2026110713]:
            a, b = cases["hybrid_"+config+"_"+str(seed)], cases["frozen_"+config+"_"+str(seed)]
            assert all(x["c0"] == y["c0"] for x, y in zip(a, b))
            delta = [x["ca"]-y["ca"] for x, y in zip(a, b)]
            deltas.append(delta); keep_deltas.append([x["ca"]-x["c0"] for x in a])
            summary["paired_comparisons"].append({"config": config, "seed": seed,
                "delta_vs_frozen": sum(delta)/200, "paired_95ci": interval(delta),
                "same_correctness_outcome_cases": sum(x["ca"] == y["ca"] for x, y in zip(a, b))})
        for control, values in [("frozen", deltas), ("keep", keep_deltas)]:
            delta = [sum(v)/3 for v in zip(*values)]
            summary["paired_comparisons"].append({"config": config, "seed": "mean_of_three",
                "control": control, "delta": sum(delta)/200, "paired_95ci": interval(delta),
                "resampling_unit": "article with all seeds paired; not 600 independent cases"})
    manifest = [json.loads(s) for s in (root/"inputs/manifest.jsonl").read_text().splitlines()]
    summary["selective_reset_same_answer_cases"] = sum(
        (v := json.loads((out/"cases"/(r["case_id"]+".json")).read_text()))
        .get("answers", {}).get("selective", {}).get("answer") ==
        v.get("answers", {}).get("reset", {}).get("answer") for r in manifest)
    (public/"paired_comparisons.json").write_text(json.dumps(summary, indent=2)+"\n")

    # Old training-source labels only; this diagnosis does not consult new diagnostic outcomes.
    sc = {x["case_id"]: x for x in map(json.loads, (out/"scores_old.jsonl").read_text().splitlines())}
    supervision = [x for x in map(json.loads, (root/"cache/shared/inputs/supervision.private.jsonl").read_text().splitlines())
                   if x["split"] in {"train", "dev", "calibration"}]
    diagnostic = {"scope": "Post-completion descriptive old train/dev/cal only; no new fit or policy selection", "splits": {}}
    for split in ["train", "dev", "calibration"]:
        rows = [x for x in supervision if x["split"] == split and x["class"] in [1, 2]
                and x["initial_answer"] in list("ABCD") and x["candidate_answer"] in list("ABCD")]
        target = [x["class"] == 1 for x in rows]
        values = {k: [] for k in ["without_evidence", "with_evidence", "margin_shift"]}
        for row in rows:
            a, b = [sc[row["case_id"]][k]["logp"] for k in ["without_evidence", "with_evidence"]]
            i, j = "ABCD".index(row["initial_answer"]), "ABCD".index(row["candidate_answer"])
            ma, mb = a[j]-a[i], b[j]-b[i]
            for key, value in zip(values, [ma, mb, mb-ma]): values[key].append(value)
        diagnostic["splits"][split] = {"transition_rows_with_two_valid_answers": len(rows),
            "WC_rows": sum(target), "CW_rows": len(rows)-sum(target),
            "WC_vs_CW_auc": {k: auc(v, target) for k, v in values.items()},
            "two_candidates_per_article_are_correlated": True}
    (public/"old_support_score_diagnostics.json").write_text(json.dumps(diagnostic, indent=2)+"\n")
    print(json.dumps({"paired": summary, "old_scores": diagnostic}, indent=2))


if __name__ == "__main__": main()
