"""Post-completion paired G3 comparisons, without refitting or changing policies."""
from collections import defaultdict
import json
import os
from pathlib import Path
import random
import statistics


def interval(delta):
    rng = random.Random(2026110718)
    samples = sorted(sum(rng.choices(delta, k=len(delta)))/len(delta) for _ in range(2000))
    return [samples[49], samples[1949]]


def main():
    root = Path(os.environ["RUN_ROOT"])
    out, public = root/"outputs", root/"results"
    public.mkdir(exist_ok=True)
    m = json.loads((out/"metrics.json").read_text())
    paired = json.loads((out/"paired_outcomes.private.json").read_text())
    assert len(paired["case_order"]) == 200 and len(set(paired["case_order"])) == 200
    outcomes = paired["outcomes"]
    families = defaultdict(list)
    for x in m["models"]:
        cases = outcomes[x["name"]]
        assert len(cases) == 200 and sum(r["ca"] for r in cases) == x["metrics"]["final_correct"]
        families[x["family"]].append(x)
    summary = {"family_summaries": {}, "paired_comparisons": [],
        "scope": "fixed policies, full 200-case denominator, no retraining or threshold selection"}
    for family, items in families.items():
        values = [i["metrics"]["accuracy"] for i in items]
        summary["family_summaries"][family] = {"accuracy_each_seed": values,
            "mean_accuracy": statistics.mean(values), "sample_sd_across_seeds": statistics.stdev(values),
            "WC_each_seed": [i["metrics"]["WC"] for i in items],
            "CW_each_seed": [i["metrics"]["CW"] for i in items],
            "three_seeds_share_same_200_cases": True}
    for config in ["ce", "ce_rank"]:
        for control in ["frozen_", "projected_"]:
            mean_deltas = []
            for seed in [2026110711, 2026110712, 2026110713]:
                a = outcomes["action_"+config+"_"+str(seed)]
                b = outcomes[control+config+"_"+str(seed)]
                assert all(x["c0"] == y["c0"] for x, y in zip(a, b))
                delta = [x["ca"]-y["ca"] for x, y in zip(a, b)]
                mean_deltas.append(delta)
                summary["paired_comparisons"].append({"method": "action_"+config,
                    "control": control+config, "seed": seed,
                    "delta_accuracy": sum(delta)/200, "paired_95ci": interval(delta)})
            # Resample articles once, keeping all seeds paired within the same article.
            delta = [sum(values)/3 for values in zip(*mean_deltas)]
            summary["paired_comparisons"].append({"method": "action_"+config,
                "control": control+config, "seed": "mean_of_predeclared_three",
                "delta_accuracy": sum(delta)/200, "paired_95ci": interval(delta)})
    locked = json.loads((out/"policies_locked.json").read_text())
    collection = json.loads((out/"collection_done.json").read_text())
    features = json.loads((out/"feature_summary.json").read_text())
    summary["lock_before_collection_finished"] = locked["locked_epoch"] < collection["finished_epoch"]
    summary["model_count"] = len(locked["models"])
    summary["trained_new_heads"] = sum("checkpoint" in x for x in locked["models"])
    summary["features_rows"] = features["rows"]
    (public/"paired_comparisons.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
