"""Sample old training articles without reading any correctness supervision."""
import json
from pathlib import Path
import random
import time
from collections import Counter

from evidence_audit import build_request


def main():
    code = Path(__file__).resolve().parent
    root = code/"runs/evidence_audit_20261108_v1"
    if (root/"inputs/preflight.jsonl").exists():
        raise RuntimeError("A completed input selection cannot be replaced")
    (root/"inputs").mkdir(exist_ok=True)
    if not (root/"started.json").exists():
        now = time.time()
        (root/"started.json").write_text(json.dumps({"started_epoch": now, "deadline_epoch": now+16*3600,
            "preflight_deadline_epoch": now+3600, "total_budget_seconds": 57600}, indent=2))
    old = code/"runs/gate_pool_20261107_v1"
    splits = json.loads((old/"inputs/splits.private.json").read_text())
    rows = [r for r in map(json.loads, (old/"inputs/manifest.jsonl").read_text().splitlines())
            if splits[r["group_id"]] == "train"]
    assert len(rows) == 1000
    rows.sort(key=lambda r: r["group_id"])
    random.Random(2026110803).shuffle(rows)
    chosen = rows[:32]
    inputs = []
    for index, row in enumerate(chosen):
        case = json.loads((old/"outputs/cases"/(row["case_id"]+".json")).read_text())
        requests = []
        for reverse in [False, True]:
            payload, bindings, expected = build_request(row, case, 2026110803+index, reverse)
            requests.append({"payload": payload, "bindings": bindings, "expected": expected})
        inputs.append({"case_id": row["case_id"], "requests": requests})
    assert len({r["group_id"] for r in chosen}) == 32
    (root/"inputs/preflight.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in inputs)+"\n")
    formats = Counter(u["source_format"] for r in inputs for c in r["requests"][0]["payload"]["anonymous_candidates"]
                      for u in c["units"] if u["claim_index"] >= 0)
    summary = {"train_article_candidates": 1000, "selected_articles": 32, "requests": 64,
        "seed": 2026110803, "correctness_labels_read": False, "image_inputs": False,
        "original_answer_in_audit_payload": False, "format_retry_limit_selected": 0,
        "clinical_expert_adjudications": 0, "canonical_claim_source_formats": dict(formats)}
    (root/"selection.json").write_text(json.dumps(summary, indent=2)); print(json.dumps(summary))


if __name__ == "__main__":
    main()
