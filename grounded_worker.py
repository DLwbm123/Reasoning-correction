"""G9 render/collection stages; never opens the offline reference-label file."""
import json
import os
from pathlib import Path
import random
import subprocess
import time
import traceback

from grounded_revision import (METHODS, SYSTEM, FULL_SCHEMA, INITIAL_PROMPT, PATCH_PROMPT,
    INSTRUCTIONS, initial_dto, full_result, revision_dto, patch_result, copy_control, render_image, parse_object)


def write(path, obj):
    with path.open("x") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def deferred_updates(root, expected_ids):
    lock = json.loads((root/"outputs/initial_done.json").read_text())
    if lock["cases"] != 32 or len(expected_ids) != 32:
        raise ValueError("All-initials lock required before released review")
    records = rows(root/"inputs/updates.private.jsonl")
    if len(records) != 32 or {r["case_id"] for r in records} != set(expected_ids):
        raise ValueError("Released annotation identities differ from frozen pool")
    return {r["case_id"]:r for r in records}


def guard(root, started):
    if time.time() >= started["collection_deadline_epoch"]-60:
        raise RuntimeError("Original collection cutoff; no clock reset")
    paths = [root]
    if os.environ.get("NEUTRAL_ROOT"):
        paths.append(Path(os.environ["NEUTRAL_ROOT"]))
    total = sum(os.lstat(Path(parent)/name).st_blocks*512 for path in paths for parent,_,files in os.walk(path) for name in files)
    if total >= 1_000_000_000:
        raise RuntimeError("Original conservative 1GB write guard")


def render(root):
    import PIL
    import numpy as np
    manifest = rows(root/"inputs/manifest.jsonl")
    ontology = json.loads((root/"inputs/ontology.private.json").read_text())
    started = json.loads((root/"started.json").read_text())
    target = root/"outputs/rendered"
    target.mkdir(exist_ok=False)
    records = []
    for row in manifest:
        guard(root, started)
        initial_dto(row, ontology)
        relative = Path(row["image"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe registered image path")
        record = {"case_id": row["case_id"], "native_relative_path": row["image"],
                  "rendered_relative_path": "outputs/rendered/"+row["case_id"]+".png"}
        try:
            record.update(render_image(root/"inputs"/relative, root/record["rendered_relative_path"], row["regions"]), status="rendered")
        except Exception as exc:
            record.update(status="unavailable", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        records.append(record)
    if len(manifest) != 32 or len({r["case_id"] for r in manifest}) != 32:
        raise ValueError("Frozen 32 identities required")
    write(root/"outputs/render_records.private.json", records)
    write(root/"outputs/render_done.json", {"cases": len(records), "rendered": sum(r["status"]=="rendered" for r in records),
        "unavailable": sum(r["status"]!="rendered" for r in records), "pillow": PIL.__version__, "numpy": np.__version__,
        "finished_epoch": time.time(), "ps": subprocess.check_output(["ps", "-p", str(os.getpid())+","+str(os.getppid()), "-o", "pid,ppid,args="], text=True)})


class Generator:
    def __init__(self, root):
        import run_e0
        self.base = run_e0.Backend()
        self.root = root
        self.count = 0

    def generate(self, cid, stage, prompt, data, image, budget):
        import torch
        if self.count >= 224:
            raise RuntimeError("Frozen 224-call cap")
        messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt+"\nINPUT:\n"+json.dumps(data, ensure_ascii=False)}]}]
        processor, model = self.base.processor, self.base.model
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        batch = processor(text=[text], images=[image], return_tensors="pt").to("cuda")
        n = batch["input_ids"].shape[1]
        if n > 8192:
            raise RuntimeError("8192-token input limit; no truncation")
        self.count += 1
        forwards = [0]
        hook = model.register_forward_pre_hook(lambda *args: forwards.__setitem__(0, forwards[0]+1))
        start = time.time()
        record = {"case_id": cid, "stage": stage, "payload": data, "executed_source_id": "image_0",
            "rendered_relative_path": "outputs/rendered/"+cid+".png", "actual_image_size": list(image.size),
            "input_tokens": n, "visual_tokens": int((batch["input_ids"]==model.config.image_token_id).sum()),
            "max_output_tokens": budget, "attempt": 0, "call_index": self.count}
        try:
            with torch.inference_mode():
                ids = model.generate(**batch, do_sample=False, num_beams=1, max_new_tokens=budget, use_cache=True)
            torch.cuda.synchronize()
            raw = processor.tokenizer.decode(ids[0,n:], skip_special_tokens=True)
            record.update(raw_output=raw, output_tokens=int(ids.shape[1]-n), generation_status="completed")
        except Exception:
            record.update(generation_status="failed", traceback=traceback.format_exc())
            raise
        finally:
            hook.remove()
            record.update(forwards=forwards[0], seconds=time.time()-start, epoch=time.time(),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved())
            with (self.root/"outputs/calls.private.jsonl").open("a") as f:
                f.write(json.dumps(record, ensure_ascii=False)+"\n")
        return raw


def collect(root):
    from PIL import Image
    import torch
    manifest = rows(root/"inputs/manifest.jsonl")
    ontology = json.loads((root/"inputs/ontology.private.json").read_text())
    started = json.loads((root/"started.json").read_text())
    if (root/"outputs/initial").exists() or (root/"outputs/collection_done.json").exists():
        raise RuntimeError("No automatic replay of model stages")
    renders = {r["case_id"]:r for r in json.loads((root/"outputs/render_records.private.json").read_text())}
    (root/"outputs/initial").mkdir()
    (root/"outputs/revisions").mkdir()
    gen = Generator(root)
    initial = {}
    for index, row in enumerate(manifest):
        guard(root, started)
        cid = row["case_id"]
        record = {"case_id": cid, "valid": False, "result": None, "raw_output": None, "parsed_output": None}
        if renders[cid]["status"] == "rendered":
            with Image.open(root/renders[cid]["rendered_relative_path"]) as src:
                image = src.convert("RGB")
            raw = gen.generate(cid, "initial", INITIAL_PROMPT, initial_dto(row, ontology), image, 1024)
            record["raw_output"] = raw
            try:
                record["parsed_output"] = parse_object(raw)
                record.update(result=full_result(raw, row, ontology), valid=True)
            except (ValueError, TypeError, KeyError) as exc:
                record["error"] = str(exc)
        else:
            record["error"] = "Preserved image-render failure; not generated"
        initial[cid] = record
        write(root/"outputs/initial"/(cid+".json"), record)
        (root/"outputs/progress.json").write_text(json.dumps({"stage":"initial", "completed_cases":index+1,"planned_cases":32,"actual_calls":gen.count,"epoch":time.time()}))
        print("INITIAL", index+1, "VALID", record["valid"], flush=True)
    write(root/"outputs/initial_done.json", {"cases":32,"valid":sum(r["valid"] for r in initial.values()),"locked_epoch":time.time()})
    # No deferred annotation file is opened before the all-initials lock above.
    updates = deferred_updates(root, initial)
    write(root/"outputs/updates_opened.json", {"epoch":time.time(),"cases":len(updates)})
    if set(updates) != set(initial):
        raise ValueError("Released annotation identities differ from frozen pool")
    for index, row in enumerate(manifest):
        cid = row["case_id"]
        order = list(METHODS)
        random.Random(2026110816+index).shuffle(order)
        original = initial[cid]
        bundle = {"case_id":cid,"order":order,"methods":{}}
        bundle["copy_control"] = copy_control(original["result"], updates[cid])
        if renders[cid]["status"] == "rendered":
            with Image.open(root/renders[cid]["rendered_relative_path"]) as src:
                image = src.convert("RGB")
        for method in order:
            guard(root, started)
            record = {"valid":False,"result":None,"raw_output":None,"parsed_output":None}
            if renders[cid]["status"] == "rendered" and (original["valid"] or not method.startswith("patch_")):
                data = revision_dto(row, ontology, original["result"], updates[cid], method)
                prompt = PATCH_PROMPT if method.startswith("patch_") else INSTRUCTIONS[method]+FULL_SCHEMA
                raw = gen.generate(cid, method, prompt, data, image, 1536)
                record["raw_output"] = raw
                try:
                    record["parsed_output"] = parse_object(raw)
                    result = patch_result(raw, data, ontology) if method.startswith("patch_") else full_result(raw, row, ontology)
                    record.update(valid=True,result=result)
                except (ValueError, TypeError, KeyError) as exc:
                    record["error"] = str(exc)
            else:
                record["error"] = "Unavailable: image or immutable initial ledger invalid; no fabricated ledger"
            bundle["methods"][method] = record
            write(root/"outputs/revisions"/(cid+"_"+method+".json"), record)
        write(root/"outputs/revisions"/(cid+"_bundle.json"), bundle)
        (root/"outputs/progress.json").write_text(json.dumps({"stage":"revision","completed_cases":index+1,"planned_cases":32,"actual_calls":gen.count,"epoch":time.time()}))
        print("REVISED",index+1,"CALLS",gen.count,flush=True)
    write(root/"outputs/collection_done.json", {"cases":32,"method_case_records":192,"actual_calls":gen.count,
        "max_calls":224,"format_retries":0,"finished_epoch":time.time(),"torch":torch.__version__,
        "all_parameters_frozen":all(not p.requires_grad for p in gen.base.model.parameters())})


if __name__ == "__main__":
    root = Path(os.environ["RUN_ROOT"])
    if os.environ["GROUNDED_STAGE"] == "render":
        render(root)
    elif os.environ["GROUNDED_STAGE"] == "collect":
        collect(root)
    else:
        raise ValueError("Unknown frozen stage")
