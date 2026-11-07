"""Frozen image/text features. This process never opens supervision files."""
import json
import os
from pathlib import Path
import time

from prepare_gate import DTO_KEYS


def main():
    import numpy as np
    import torch
    from PIL import Image
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    root = Path(os.environ["RUN_ROOT"])
    outdir = root / "outputs"
    if (outdir / "features.npz").exists():
        raise RuntimeError("Refusing to overwrite completed features")
    torch.set_num_threads(4)
    if torch.cuda.mem_get_info()[0] < 24 * 1024**3:
        raise RuntimeError("Less than 24 GiB free before loading")
    torch.cuda.set_per_process_memory_fraction(.60)
    path = (root / "model_path.txt").read_text().strip()
    processor = AutoProcessor.from_pretrained(path, min_pixels=256*28*28,
        max_pixels=768*28*28, local_files_only=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(path, torch_dtype=torch.bfloat16,
        device_map={"": 0}, attn_implementation="sdpa", local_files_only=True).eval()
    model.requires_grad_(False)
    print("MODEL_LOADED", flush=True)
    rows = [json.loads(x) for x in (root / "inputs/features.jsonl").read_text().splitlines()]
    vectors, ids, cache, timings = [], [], {}, []
    started = time.time()
    for i, row in enumerate(rows):
        if set(row) != {"feature_id", "case_id", "image", "payload"} or set(row["payload"]) != DTO_KEYS:
            raise ValueError("Unexpected encoder fields")
        text = ("Encode the evidence and the proposed answer revision for a downstream outcome classifier. "
                "No correctness label is supplied. Do not generate an answer.\n" +
                json.dumps(row["payload"], ensure_ascii=False, sort_keys=True))
        key = (row["image"], text)
        if key not in cache:
            with Image.open(root / "inputs" / row["image"]) as src:
                image = src.convert("RGB")
            messages = [{"role": "user", "content": [{"type": "image", "image": image},
                        {"type": "text", "text": text}]}]
            # No generation prefix: final language token is the fixed scoring marker.
            prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            # Template includes a trailing turn terminator; append marker so its last token is scored.
            prompt += "\nSCORE_STATE"
            batch = processor(text=[prompt], images=[image], return_tensors="pt").to("cuda")
            n = batch["input_ids"].shape[1]
            if n > 8192:
                raise RuntimeError("Frozen sequence budget exceeded; no automatic truncation")
            begin = time.time()
            with torch.inference_mode():
                output = model(**batch, output_hidden_states=True, use_cache=False, return_dict=True)
                vector = output.hidden_states[-1][0, -1].float().cpu().numpy()
            torch.cuda.synchronize()
            cache[key] = vector
            timings.append({"tokens": n, "visual_tokens": int((batch["input_ids"] == model.config.image_token_id).sum()),
                            "seconds": time.time() - begin})
            del output, batch, image
        vectors.append(cache[key]); ids.append(row["feature_id"])
        (outdir / "progress.json").write_text(json.dumps({"stage": "features", "completed_rows": i+1,
            "planned_rows": len(rows), "unique_forwards": len(cache), "updated_epoch": time.time()}))
        if (i+1) % 10 == 0:
            print("FEATURE_ROWS", i+1, "FORWARDS", len(cache), flush=True)
    np.savez_compressed(outdir / "features.npz", x=np.stack(vectors), ids=np.array(ids))
    (outdir / "feature_summary.json").write_text(json.dumps({"rows": len(rows), "dimension": len(vectors[0]),
        "unique_forwards": len(cache), "seconds": time.time()-started,
        "input_tokens": sum(t["tokens"] for t in timings), "visual_tokens": sum(t["visual_tokens"] for t in timings),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "frozen_parameters": all(not p.requires_grad for p in model.parameters())}, indent=2))
    print("FEATURES_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
