"""Offline supervision preparation; only label-free DTOs reach the encoder."""
import json
from pathlib import Path
import random
import shutil
import time


DTO_KEYS = {"question", "choices", "evidence", "initial_output", "candidate_output",
            "context_facts", "format_flags"}


def transition(initial_correct, candidate_correct):
    return 2 * int(initial_correct) + int(candidate_correct)


def split_groups(groups):
    groups = sorted(set(groups))
    if len(groups) != 200:
        raise ValueError("Expected exactly 200 independent article groups")
    random.Random(2026110707).shuffle(groups)
    return {g: split for split, subset in [("train", groups[:120]), ("dev", groups[120:140]),
            ("calibration", groups[140:160]), ("diagnostic_eval", groups[160:])]
            for g in subset}


def prepare_pool(old, new, split, *, copy_images=True, split_seed=2026110707, retrospective=True):
    inputs = new / "inputs"
    labels = {x["case_id"]: x for x in map(json.loads,
        (old / "inputs/labels.private.jsonl").read_text().splitlines())}
    manifest = [json.loads(x) for x in (old / "inputs/manifest.jsonl").read_text().splitlines()]
    if len(manifest) != len(split) or len({r["case_id"] for r in manifest}) != len(split):
        raise ValueError("Unexpected source denominator")
    feature_rows, targets = [], []
    if copy_images:
        (inputs / "images").mkdir()
    for row in manifest:
        cid = row["case_id"]
        case = json.loads((old / "outputs/cases" / (cid + ".json")).read_text())
        if case["group_id"] != row["group_id"] or labels[cid]["group_id"] != row["group_id"]:
            raise ValueError("Group registration mismatch")
        # Copy source image once; preserving E0 source files and labels.
        image = Path(row["image"])
        if image.is_absolute() or ".." in image.parts:
            raise ValueError("Unsafe image path")
        if copy_images:
            shutil.copyfile(old / "inputs" / image, inputs / "images" / image.name)
        evidence = case.get("evidence", [])
        refs = {e["evidence_id"]: "E" + str(i) for i, e in enumerate(evidence)}
        def local_ids(value):
            if isinstance(value, dict):
                return {k: local_ids(v) for k, v in value.items()}
            if isinstance(value, list):
                return [local_ids(v) for v in value]
            return refs.get(value, value) if isinstance(value, str) else value
        initial = case.get("initial", {})
        c0 = initial.get("answer") == labels[cid]["reference_letter"]
        for branch in ["selective", "reset"]:
            candidate = case.get("answers", {}).get(branch, {})
            payload = local_ids({"question": row["question"], "choices": row["choices"],
                "evidence": evidence, "initial_output": initial, "candidate_output": candidate,
                "context_facts": case.get("contexts", {}).get(branch, []),
                "format_flags": {"initial_valid_letter": initial.get("answer") in ["A", "B", "C", "D"],
                    "candidate_valid_letter": candidate.get("answer") in ["A", "B", "C", "D"],
                    "candidate_explanation_missing": not bool(candidate.get("claims")),
                    "independent_evidence_audit": "not_implemented"}})
            assert set(payload) == DTO_KEYS
            fid = cid + ":" + str(len(feature_rows) % 2)
            feature_rows.append({"feature_id": fid, "case_id": cid,
                "image": "images/" + image.name, "payload": payload})
            ca = candidate.get("answer") == labels[cid]["reference_letter"]
            targets.append({"feature_id": fid, "case_id": cid, "group_id": row["group_id"],
                "split": split[row["group_id"]], "branch": branch, "class": transition(c0, ca),
                "c0": int(c0), "ca": int(ca), "initial_answer": initial.get("answer"),
                "candidate_answer": candidate.get("answer"),
                "valid": candidate.get("answer") in ["A", "B", "C", "D"],
                "explanation_missing": not bool(candidate.get("claims"))})
    for name, values in [("features.jsonl", feature_rows), ("supervision.private.jsonl", targets)]:
        (inputs / name).write_text("\n".join(json.dumps(v, ensure_ascii=False) for v in values) + "\n")
    counts = {s: len({r["case_id"] for r in targets if r["split"] == s})
              for s in ["train", "dev", "calibration", "diagnostic_eval"]}
    (new / "preparation.json").write_text(json.dumps({"source_cases": len(manifest),
        "candidate_rows": len(targets), "case_counts": counts,
        "split_seed": split_seed, "official_test_used": False,
        "retrospective_development_only": retrospective}, indent=2))
    print(json.dumps(counts))


def main():
    root = Path(__file__).resolve().parent
    old = root / "runs/e0_20261107_v2"
    new = root / "runs/gate_baseline_20261107_v1"
    (new / "inputs").mkdir(exist_ok=False)
    started = time.time()
    (new / "started.json").write_text(json.dumps({"started_epoch": started,
        "deadline_epoch": started + 3 * 3600, "total_budget_seconds": 10800}, indent=2))
    manifest = [json.loads(x) for x in (old / "inputs/manifest.jsonl").read_text().splitlines()]
    prepare_pool(old, new, split_groups(r["group_id"] for r in manifest))


if __name__ == "__main__":
    main()
