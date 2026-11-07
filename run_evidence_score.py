"""Two image-prefill calls per case; request DTO never contains answer identities/labels."""
import json
import os
from pathlib import Path
import time

from evidence_score import SCORE_KEYS, score_payload, check_letter_tokens


def main():
    import torch
    from PIL import Image
    import run_e0 as base
    root = Path(os.environ["RUN_ROOT"])
    old = os.environ.get("SCORE_PHASE") == "old"
    phase = "old" if old else "new"
    source = Path((root/"shared_root.txt").read_text().strip()) if old else root
    target = root/"outputs"/("scores_"+phase+".jsonl")
    if target.exists():
        raise RuntimeError("No automatic repeat of score calls")
    rows = [json.loads(s) for s in (source/"inputs/manifest.jsonl").read_text().splitlines()]
    if old:
        split = json.loads((source/"inputs/splits.private.json").read_text())
        rows = [r for r in rows if split[r["group_id"]] in {"train", "dev", "calibration"}]
    assert len(rows) == (1400 if old else int(os.environ.get("SCORE_NEW_CASES", "200")))
    backend = base.Backend()
    records = []
    start = time.time()
    for row in rows:
        cid = row["case_id"]
        case = json.loads((source/"outputs/cases"/(cid+".json")).read_text())
        assert case["case_id"] == cid and case["group_id"] == row["group_id"]
        path = Path(row["image"])
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("Unsafe registered image path")
        with Image.open(source/"inputs"/path) as src:
            image = src.convert("RGB")
        result = {"case_id": cid, "status": "completed", "calls": []}
        for use_e in [False, True]:
            data = score_payload(row, case.get("evidence", []), use_e)
            assert set(data) == SCORE_KEYS
            messages = [{"role": "system", "content": "Answer the offline medical image multiple-choice question using only the supplied image and information. Supplied observations may be incomplete or mistaken. Reply with a single letter A, B, C, or D."},
                        {"role": "user", "content": [{"type": "image", "image": image},
                         {"type": "text", "text": json.dumps(data, ensure_ascii=False)}]}]
            prompt = backend.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)+"Answer:"
            tokens = check_letter_tokens(backend.processor.tokenizer, prompt)
            batch = backend.processor(text=[prompt], images=[image], return_tensors="pt").to("cuda")
            if batch["input_ids"].shape[1] > 8192:
                raise RuntimeError("Fixed sequence budget exceeded; no truncation")
            begin = time.time()
            with torch.inference_mode():
                output = backend.model(**batch, use_cache=False, return_dict=True)
                logp = output.logits[0, -1, tokens].float().log_softmax(-1)
                entropy = float(-(logp.exp()*logp).sum())
            torch.cuda.synchronize()
            key = "with_evidence" if use_e else "without_evidence"
            result[key] = {"logp": logp.cpu().tolist(), "entropy": entropy}
            result["calls"].append({"context": key, "payload": data, "executed_source_id": "image_0",
                "registered_image": str(path), "choice_token_ids": tokens, "forwards": 1,
                "input_tokens": int(batch["input_ids"].shape[1]),
                "visual_tokens": int((batch["input_ids"] == backend.model.config.image_token_id).sum()),
                "seconds": time.time()-begin, "output_tokens": 0})
            del output, batch, logp
        with target.open("a") as f:
            f.write(json.dumps(result, ensure_ascii=False)+"\n")
        records.append(result)
        (root/"outputs/progress.json").write_text(json.dumps({"stage": "scores_"+phase,
            "completed_cases": len(records), "planned_cases": len(rows), "forward_calls": 2*len(records),
            "updated_epoch": time.time()}))
        if len(records) % 20 == 0:
            print("SCORED_CASES", phase, len(records), flush=True)
    summary = {"phase": phase, "cases": len(records), "forward_calls": 2*len(records),
        "seconds": time.time()-start, "generation_calls": 0, "input_tokens": sum(c["input_tokens"] for r in records for c in r["calls"]),
        "visual_tokens": sum(c["visual_tokens"] for r in records for c in r["calls"]),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "frozen_parameters": all(not p.requires_grad for p in backend.model.parameters())}
    (root/"outputs"/("scores_"+phase+"_done.json")).write_text(json.dumps(summary, indent=2))
    print("SCORING_COMPLETE", phase, flush=True)


if __name__ == "__main__":
    main()
