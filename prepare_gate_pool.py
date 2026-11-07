"""Freeze a fresh article-disjoint training-source pool; no model outcomes consulted."""
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import time
import zipfile

from prepare_data import SOURCE, VISUAL, EXCLUDE, article


def main():
    code = Path(__file__).resolve().parent
    action = os.environ.get("PREPARE_ACTION_PROBE") == "1"
    evidence_score = os.environ.get("PREPARE_EVIDENCE_SCORE") == "1"
    score_policy = os.environ.get("PREPARE_SCORE_POLICY") == "1"
    risk_calibration = os.environ.get("PREPARE_RISK_CALIBRATION") == "1"
    score_policy = score_policy or risk_calibration
    evidence_score = evidence_score or score_policy
    root = code/("runs/risk_calibration_20261108_v1" if risk_calibration else
                 "runs/score_policy_20261108_v1" if score_policy else
                 "runs/evidence_score_20261108_v1" if evidence_score else
                 "runs/action_gate_20261107_v1" if action else "runs/gate_pool_20261107_v1")
    action = action or evidence_score
    n = 200 if action else 1600
    seed = 2026110715 if action else 2026110708
    execution_seed = 2026110716 if action else 2026110709
    if evidence_score:
        seed, execution_seed = 2026110804, 2026110805
    if score_policy:
        seed, execution_seed = 2026110807, 2026110808
    if risk_calibration:
        seed, execution_seed, n = 2026110810, 2026110811, 1000
    budget = 4*3600 if action else 20*3600
    if score_policy: budget = 3*3600
    if risk_calibration: budget = 6*3600
    (root/"inputs/images").mkdir(parents=True, exist_ok=False)
    start = time.time()
    (root/"started.json").write_text(json.dumps({"started_epoch": start,
        "deadline_epoch": start+budget, "total_budget_seconds": budget}, indent=2))
    test_groups = set()
    for name in ["test.csv", "test_clean.csv", "test_2.csv"]:
        with (SOURCE/name).open(newline="") as f:
            test_groups.update(article(r["Figure_path"]) for r in csv.DictReader(f))
    old_groups, old_pixels = set(), set()
    old_runs = ["e0_20261107_v2", "e0_interface_20261107_v1"]
    if action: old_runs.append("gate_pool_20261107_v1")
    if evidence_score: old_runs.append("action_gate_20261107_v1")
    if score_policy: old_runs.append("evidence_score_20261108_v1")
    if risk_calibration: old_runs.append("score_policy_20261108_v1")
    for name in old_runs:
        rows = [json.loads(x) for x in (code/"runs"/name/"inputs/manifest.jsonl").read_text().splitlines()]
        old_groups.update(r["group_id"] for r in rows)
        old_pixels.update(r["image_sha256"] for r in rows)
    candidates = []
    with (SOURCE/"train.csv").open(newline="") as f:
        for i, row in enumerate(csv.DictReader(f)):
            group, q = article(row["Figure_path"]), row["Question"].strip()
            if group and group not in test_groups|old_groups and VISUAL.search(q) and not EXCLUDE.search(q):
                # Source-label syntax, not correctness/model outcomes, is checked before sampling.
                if row["Answer_label"].strip() in ["A", "B", "C", "D"]:
                    candidates.append((i, row))
    random.Random(seed).shuffle(candidates)
    infer, labels, groups, pixels = [], [], set(), set(old_pixels)
    with zipfile.ZipFile(SOURCE/"images.zip") as archive:
        for i, row in candidates:
            group = article(row["Figure_path"]); member = "images/"+row["Figure_path"]
            if group in groups or member not in archive.NameToInfo:
                continue
            data = archive.read(member)
            # Scientific exact-image grouping only; not a model/file-transfer integrity hash.
            fingerprint = hashlib.sha256(data).hexdigest()
            if fingerprint in pixels:
                continue
            pixels.add(fingerprint); groups.add(group)
            cid = ("m" if risk_calibration else "k" if score_policy else "j" if evidence_score else "h" if action else "g")+str(len(infer)).zfill(4); image = "images/"+cid+".jpg"
            (root/"inputs"/image).write_bytes(data)
            infer.append({"case_id": cid, "group_id": group, "question": row["Question"].strip(),
                "choices": {x: row["Choice "+x].strip() for x in "ABCD"},
                "image": image, "image_sha256": fingerprint})
            labels.append({"case_id": cid, "group_id": group, "reference_letter": row["Answer_label"].strip(),
                "source_row": i, "source_image": row["Figure_path"]})
            if len(infer) == n:
                break
    if len(infer) != n:
        raise RuntimeError("Insufficient eligible groups; no label-based replenishment")
    # Independently shuffle selected groups, without answer/outcome stratification.
    ordered = sorted(groups); random.Random(seed).shuffle(ordered)
    splits = {g: "diagnostic_eval" for g in ordered} if action else {g: s for s, subset in [("train", ordered[:1000]), ("dev", ordered[1000:1200]),
             ("calibration", ordered[1200:1400]), ("diagnostic_eval", ordered[1400:])]
              for g in subset}
    if risk_calibration:
        splits = {g: "calibration" if i < 800 else "diagnostic_eval" for i, g in enumerate(ordered)}
    # Random case interleaving avoids a hard stop preferentially retaining train cases.
    random.Random(execution_seed).shuffle(infer)
    files = [("manifest.jsonl", infer)]
    if risk_calibration:
        files += [("labels."+s+".private.jsonl", [r for r in labels if splits[r["group_id"]] == s])
                  for s in ["calibration", "diagnostic_eval"]]
    else: files.append(("labels.private.jsonl", labels))
    for name, rows in files:
        (root/"inputs"/name).write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows)+"\n")
    (root/"inputs/splits.private.json").write_text(json.dumps(splits))
    summary = {"n": n, "unique_articles": len(groups), "official_test_article_overlap": len(groups&test_groups),
        "old_article_overlap": len(groups&old_groups), "old_exact_encoded_image_overlap": 0,
        "split_cases": {s: list(splits.values()).count(s) for s in ["train", "dev", "calibration", "diagnostic_eval"]},
        "sample_split_seed": seed, "execution_seed": execution_seed,
        "selection_uses_model_outcomes": False, "near_duplicate_or_patient_independence_claimed": False,
        "eligible_rows_before_group_sampling": len(candidates)}
    (root/"selection.json").write_text(json.dumps(summary, indent=2)); print(json.dumps(summary))


if __name__ == "__main__":
    main()
