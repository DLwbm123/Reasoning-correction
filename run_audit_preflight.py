"""Frozen G4 preflight; 64 actual text-only requests, no labels and no reruns."""
import json
import os
from pathlib import Path
import time

from evidence_audit import PAYLOAD_KEYS, bind_result, summarize


def main():
    import torch
    import run_e0 as base
    root = Path(os.environ["RUN_ROOT"])
    out = root/"outputs"
    if (out/"preflight_records.jsonl").exists():
        raise RuntimeError("Preflight records already exist; no automatic repeat")
    backend = base.Backend()
    prompts = json.loads((Path(__file__).resolve().parent/"prompts_audit.json").read_text())
    started = json.loads((root/"started.json").read_text())
    inputs = [json.loads(x) for x in (root/"inputs/preflight.jsonl").read_text().splitlines()]
    assert len(inputs) == 32
    records = []
    for row in inputs:
        for permutation in [0, 1]:
            if time.time() >= started["preflight_deadline_epoch"]-30:
                raise RuntimeError("Original one-hour preflight deadline")
            payload = row["requests"][permutation]["payload"]
            if set(payload) != PAYLOAD_KEYS:
                raise ValueError("Unexpected auditor fields")
            request = row["requests"][permutation]
            messages = [{"role": "system", "content": prompts["system"]}, {"role": "user",
                "content": prompts["audit"]+"\nINPUT:\n"+json.dumps(payload, ensure_ascii=False)}]
            text = backend.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            batch = backend.processor(text=[text], return_tensors="pt").to("cuda")
            n = batch["input_ids"].shape[1]
            begin = time.time()
            forwards = [0]
            handle = backend.model.register_forward_pre_hook(lambda *args: forwards.__setitem__(0, forwards[0]+1))
            try:
                with torch.inference_mode():
                    ids = backend.model.generate(**batch, do_sample=False, num_beams=1, max_new_tokens=1024, use_cache=True)
            finally:
                handle.remove()
            torch.cuda.synchronize()
            raw = backend.processor.tokenizer.decode(ids[0, n:], skip_special_tokens=True)
            try:
                parsed = base.parse_json(raw)
                bound = bind_result(parsed, payload, request["bindings"], request["expected"])
            except (ValueError, TypeError, KeyError) as exc:
                parsed = None
                bound = {"schema_ok": False, "errors": ["parse_or_schema_error: "+str(exc)], "by_branch": {}}
            record = {"case_id": row["case_id"], "permutation": permutation, "payload": payload,
                "raw_output": raw, "parsed_output": parsed, "bound": bound, "attempt": 0,
                "input_tokens": n, "output_tokens": ids.shape[1]-n, "visual_tokens": 0, "forwards": forwards[0],
                "seconds": time.time()-begin, "epoch": time.time(),
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved()}
            with (out/"preflight_records.jsonl").open("a") as f:
                f.write(json.dumps(record, ensure_ascii=False)+"\n")
            records.append(record)
            (out/"progress.json").write_text(json.dumps({"stage": "preflight", "completed_requests": len(records),
                "planned_requests": 64, "schema_source_valid_requests": sum(r["bound"]["schema_ok"] for r in records),
                "updated_epoch": time.time()}))
            print("AUDIT_REQUEST", len(records), "SCHEMA_OK", bound["schema_ok"], flush=True)
            del batch, ids
    summary = summarize(records)
    summary.update(finished_epoch=time.time(), actual_generation_calls=len(records), format_retries=0,
        generation_seconds=sum(r["seconds"] for r in records), input_tokens=sum(r["input_tokens"] for r in records),
        output_tokens=sum(r["output_tokens"] for r in records))
    (out/"preflight_summary.json").write_text(json.dumps(summary, indent=2))
    print("PREFLIGHT_COMPLETE", summary["status"], flush=True)


if __name__ == "__main__":
    main()
