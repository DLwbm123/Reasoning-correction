"""Read-only aggregate analysis after the frozen pool/training have completed."""
from collections import Counter
import json
import math
import os
from pathlib import Path
import random
import statistics


def main():
    root = Path(os.environ["RUN_ROOT"])
    out, public = root/"outputs", root/"results"
    public.mkdir(exist_ok=True)
    done = json.loads((out/"supervisor.json").read_text())
    if done["status"] != "completed":
        raise ValueError("No final outcome analysis while collecting")
    manifest = [json.loads(x) for x in (root/"inputs/manifest.jsonl").read_text().splitlines()]
    labels = {x["case_id"]: x for x in map(json.loads, (root/"inputs/labels.private.jsonl").read_text().splitlines())}
    split = json.loads((root/"inputs/splits.private.json").read_text())
    counters, statuses, strata = Counter(), Counter(), {}
    for row in manifest:
        cid = row["case_id"]
        x = json.loads((out/"cases"/(cid+".json")).read_text())
        assert x["case_id"] == cid and x["group_id"] == row["group_id"]
        state, initial = x.get("state", {}), x.get("initial", {})
        counters["cases"] += 1
        counters["pipeline_error_cases"] += bool(x.get("pipeline_error"))
        counters["initial_invalid_cases"] += initial.get("answer") not in list("ABCD")
        counters["k_empty_cases"] += not state.get("retained")
        counters["graph_changes_k_cases"] += state.get("flat", []) != state.get("retained", [])
        counters["trigger_cases"] += bool(state.get("deployment_trigger"))
        counters["requested_observations"] += len(x.get("public_tests", []))
        counters["registered_observations"] += len(x.get("evidence", []))
        counters["cases_with_adequate_observation"] += any(e["assessability"] == "adequate" for e in x.get("evidence", []))
        statuses.update(e["assessability"] for e in x.get("evidence", []))
        for answer in x.get("answers", {}).values():
            counters["candidate_schema_warning_rows"] += bool(answer.get("_schema_warning"))
        if split[row["group_id"]] == "diagnostic_eval":
            c0 = initial.get("answer") == labels[cid]["reference_letter"]
            for branch in ["selective", "reset"]:
                ca = x.get("answers", {}).get(branch, {}).get("answer") == labels[cid]["reference_letter"]
                strata.setdefault(branch, []).append(int(ca)-int(c0))
    ci = {}
    for branch, delta in strata.items():
        rng = random.Random(2026110714)
        samples = sorted(sum(rng.choices(delta, k=len(delta)))/len(delta) for _ in range(2000))
        ci[branch] = {"delta_accuracy": sum(delta)/len(delta), "paired_95ci": [samples[49], samples[1949]]}
    costs, times = Counter(), []
    with (out/"calls.jsonl").open() as f:
        for line in f:
            call = json.loads(line)
            costs["generation_calls"] += 1
            costs["unparsed_generation_calls"] += not call["parsed"]
            costs["retry_calls"] += bool(call["attempt"])
            for key in ["input_tokens", "output_tokens", "visual_tokens", "forwards"]:
                costs[key] += call[key]
            for key in ["peak_memory_allocated", "peak_memory_reserved"]:
                costs[key] = max(costs[key], call[key])
            times.append(call["seconds"])
    times.sort()
    costs.update({"generation_seconds_sum": sum(times), "call_latency_p50": statistics.median(times),
                  "call_latency_p95": times[int(.95*(len(times)-1))]})
    start = json.loads((root/"started.json").read_text())
    summary = {"collection": dict(counters), "observation_status_counts": dict(statuses),
        "paired_branch_vs_keep": ci, "generation_cost": dict(costs),
        "original_start_to_completed_seconds": done["finished_epoch"]-start["started_epoch"],
        "analysis_scope": "post-completion aggregate counts; no policy/model changes"}
    assert summary["collection"]["cases"] == int(os.environ.get("EXPECTED_CASES", "1600"))
    (public/"collection_diagnostics.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
