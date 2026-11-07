"""Read-only G4 analysis; wrapper normalization never changes the frozen gate."""
from collections import Counter
import json
import os
from pathlib import Path

from evidence_audit import bind_result, summarize


def main():
    root = Path(os.environ["RUN_ROOT"])
    records = [json.loads(s) for s in (root/"outputs/preflight_records.jsonl").read_text().splitlines()]
    inputs = {x["case_id"]: x for x in
              map(json.loads, (root/"inputs/preflight.jsonl").read_text().splitlines())}
    official = json.loads((root/"outputs/preflight_summary.json").read_text())
    assert len(records) == 64 and len(inputs) == 32
    assert summarize(records)["passed"] == official["passed"]
    errors = Counter(e for r in records for e in set(r["bound"]["errors"]))
    roots, normalized = Counter(), []
    for r in records:
        raw = r["raw_output"].strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        value = json.loads(raw)
        roots[type(value).__name__] += 1
        # Descriptive only: preserve every element, no fabricated fields or citations.
        if isinstance(value, list):
            value = {"audits": value}
        request = inputs[r["case_id"]]["requests"][r["permutation"]]
        bound = bind_result(value, r["payload"], request["bindings"], request["expected"])
        normalized.append(dict(r, bound=bound))
    original_pairs = []
    by_case = {}
    for r in records:
        by_case.setdefault(r["case_id"], {})[r["permutation"]] = r
    for pair in by_case.values():
        if all(pair[i]["bound"]["schema_ok"] for i in [0, 1]):
            original_pairs.extend(pair[0]["bound"]["by_branch"][b]["eligible"] ==
                                  pair[1]["bound"]["by_branch"][b]["eligible"]
                                  for b in ["selective", "reset"])
    result = {
        "frozen_preflight": official,
        "request_error_counts_nonexclusive": dict(errors),
        "complete_json_root_types_after_code_fence_removal": dict(roots),
        "complete_json_readable_requests": sum(roots.values()),
        "both_permutations_valid_candidate_pairs": len(original_pairs),
        "agreements_within_valid_pairs": sum(original_pairs),
        "actual_forwards": sum(r["forwards"] for r in records),
        "visual_tokens": sum(r["visual_tokens"] for r in records),
        "output_token_cap_hits": sum(r["output_tokens"] >= 1024 for r in records),
        "maximum_output_tokens": max(r["output_tokens"] for r in records),
        "peak_allocated_bytes": max(r["peak_allocated_bytes"] for r in records),
        "peak_reserved_bytes": max(r["peak_reserved_bytes"] for r in records),
        "posthoc_wrapper_only_diagnostic": summarize(normalized),
        "posthoc_request_error_counts_nonexclusive": dict(Counter(
            e for r in normalized for e in set(r["bound"]["errors"]))),
        "posthoc_scope": "No model calls, refitting, gate admission or prompt change. "
                         "Diagnostic only; frozen failed result and full-extension stop remain binding.",
        "new_expert_adjudications": 0,
    }
    target = root/"results"
    target.mkdir(exist_ok=True)
    (target/"preflight_analysis.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
