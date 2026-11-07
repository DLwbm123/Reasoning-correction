"""Fresh article pool using unchanged D1 role prompts and the two gate candidates."""
import json
from pathlib import Path
import random
import time
import traceback

import run_e0 as base
from interface_run import InterfaceBackend
from mechanism import payload, claims_from, public_tests, rebuild


def run_case(backend, row, index):
    from PIL import Image
    if set(row) != {"case_id", "group_id", "question", "choices", "image", "image_sha256"}:
        raise ValueError("Unexpected inference manifest fields")
    cid = row["case_id"]
    with Image.open(base.ROOT / "inputs" / row["image"]) as src:
        image = src.convert("RGB")
    initial = backend.generate("initial", payload("initial", question=row["question"], choices=row["choices"]), image, cid)
    claims = claims_from(initial)
    plan = backend.generate("plan", payload("plan", question=row["question"], claims=claims, tools=[]), None, cid)
    tests = public_tests(plan)
    observed = backend.generate("observe", payload("observe", public_tests=tests,
        image_manifest=[{"source_id": "image_0", "transform": "full image, original orientation"}]), image, cid) if tests else {"observations": []}
    # The D1 one-test executor assigns provenance; preserve its original adequacy.
    evidence = []
    for obs in observed["observations"]:
        evidence.append({"evidence_id": cid+":E"+str(len(evidence)), "source_id": obs["source_id"],
            "test_id": obs["test_id"], "assessability": obs["assessability"],
            "observation": obs["observation"], "limitations": obs["limitations"]})
    bindings = backend.generate("bind", payload("bind", question=row["question"], claims=claims, evidence=evidence), None, cid) if claims else {"verdicts": []}
    state = rebuild(claims, bindings, evidence)
    contexts = {"selective": state["retained"], "reset": []}
    order = ["selective", "reset"]; random.Random(4711+index).shuffle(order)
    answers, cache, aliases = {}, {}, {}
    for branch in order:
        data = payload("repair", question=row["question"], choices=row["choices"], evidence=evidence,
                       historical_facts=contexts[branch])
        key = json.dumps(data, sort_keys=True)
        if key in cache:
            answers[branch], aliases[branch] = cache[key]
        else:
            answers[branch] = backend.generate("repair", data, image, cid, branch)
            cache[key] = answers[branch], branch
    return {"case_id": cid, "group_id": row["group_id"], "initial": initial, "claims": claims,
        "plan": plan, "public_tests": tests, "observed": observed, "evidence": evidence,
        "bindings": bindings, "state": state, "contexts": contexts, "answers": answers,
        "deterministic_aliases": aliases}


def main():
    root = base.ROOT
    started = json.loads((root/"started.json").read_text())
    deadline = started["deadline_epoch"] - 7200
    rows = [json.loads(x) for x in (root/"inputs/manifest.jsonl").read_text().splitlines()]
    (root/"outputs/cases").mkdir(exist_ok=False)
    backend = InterfaceBackend()
    for i, row in enumerate(rows):
        if time.time() >= deadline:
            raise RuntimeError("Pool deadline reserve; incomplete pool cannot train")
        try:
            result = run_case(backend, row, i)
        except Exception as exc:
            result = {"case_id": row["case_id"], "group_id": row["group_id"], "pipeline_error": traceback.format_exc()}
            base.write_json(root/"outputs/cases"/(row["case_id"]+".json"), result)
            if any(word in str(exc).lower() for word in ["cuda", "out of memory"]):
                raise
        base.write_json(root/"outputs/cases"/(row["case_id"]+".json"), result)
        base.write_json(root/"outputs/progress.json", {"stage": "collection", "completed_cases": i+1,
            "planned_cases": len(rows), "calls": backend.call_count, "updated_epoch": time.time()})
    base.write_json(root/"outputs/collection_done.json", {"status": "completed", "cases": len(rows), "finished_epoch": time.time()})


if __name__ == "__main__":
    main()
