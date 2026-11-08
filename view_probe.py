"""G10 fixed view-factor diagnosis. No review labels are opened by generation."""
import json
import math
import os
from pathlib import Path
import random
import time

from fact_transport import extract_fact_transport

CONDITIONS=("duplicate_full","full_plus_crop")


def crop_rectangle(boxes,size):
    if not boxes or any(len(b)!=4 or not(0<=b[0]<b[2]<=1 and 0<=b[1]<b[3]<=1) for b in boxes):raise ValueError("Bad registered box")
    return (max(0,math.floor(min(b[0] for b in boxes)*size[0])),
            max(0,math.floor(min(b[1] for b in boxes)*size[1])),
            min(size[0],math.ceil(max(b[2] for b in boxes)*size[0])),
            min(size[1],math.ceil(max(b[3] for b in boxes)*size[1])))


def probe_dto(row,region,condition,ontology,provenance):
    if set(row)!={"case_id","view_id","image","regions"} or condition not in CONDITIONS:raise ValueError("Probe fields")
    if region not in [r["region_id"] for r in row["regions"]]:raise ValueError("Unknown requested region")
    return {"case_id":row["case_id"],"requested_region":region,"regions":row["regions"],
        "ontology":{k:ontology[k] for k in ["finding_tags","location_tags"]},
        "image_manifest":[{"source_id":"image_0","role":"full_current"},
            {"source_id":"image_1","role":"duplicate_current" if condition=="duplicate_full" else "registered_crop",
             "parent_source_id":"image_0","registered_union_box_native_pixels":provenance["union_box"],
             "native_size":provenance["native_size"],"full_render_size":provenance["full_size"]}]}


def extract_requested(raw,cid,region,response_id):
    receipt=extract_fact_transport(raw,case_id=cid,view_id="image_0",response_id=response_id)
    slots=[region+".finding",region+".location"]
    values={s:None for s in slots}
    if receipt["facts"]:
        for slot in slots:
            cell=receipt["facts"][slot]
            if cell["status"] in {"present_tag_list","present_empty_list"}:values[slot]=cell["value"]
    return {"values":values,"transport":receipt,"semantic_eligible":False}


def rows(path):return [json.loads(s) for s in path.read_text().splitlines()]
def write(path,value):
    with path.open("x") as f:json.dump(value,f,ensure_ascii=False,indent=2)


def guard(root):
    start=json.loads((root/"started.json").read_text())
    if time.time()>=start["collection_deadline_epoch"]-60:raise RuntimeError("Original collection deadline")
    paths=[root,Path(os.environ["NEUTRAL_ROOT"])]
    used=sum(os.lstat(Path(parent)/n).st_blocks*512 for p in paths for parent,_,files in os.walk(p) for n in files)
    if used>=1_000_000_000:raise RuntimeError("Conservative 1GB guard")


def render(root):
    import numpy as np
    from PIL import Image,ImageDraw
    manifest=rows(root/"inputs/manifest.jsonl");target=root/"outputs/views";target.mkdir()
    records=[]
    for row in manifest:
        guard(root)
        relative=Path(row["image"])
        if relative.is_absolute() or ".." in relative.parts:raise ValueError("Unsafe image path")
        with Image.open(root/"inputs"/relative) as im:im.load();arr=np.asarray(im);native_size=im.size
        if arr.ndim!=2 or not np.issubdtype(arr.dtype,np.integer) or arr.min()<0 or arr.max()>65535 or arr.max()<=arr.min():raise ValueError("Native gray image invalid; preserve frozen identities")
        low,high=int(arr.min()),int(arr.max());rgb=Image.fromarray(np.rint((arr.astype(np.float32)-low)*255/(high-low)).clip(0,255).astype(np.uint8)).convert("RGB")
        def resized(image):
            ratio=min(1,1024/max(image.size));return image.resize(tuple(max(1,round(v*ratio)) for v in image.size),Image.Resampling.LANCZOS)
        full=resized(rgb);draw=ImageDraw.Draw(full)
        for r,color in zip(row["regions"],["red","blue"]):
            for b in r["boxes"]:
                xy=[b[0]*full.width,b[1]*full.height,b[2]*full.width,b[3]*full.height]
                draw.rectangle(xy,outline=color,width=3);draw.text((xy[0]+2,xy[1]+2),r["region_id"],fill=color)
        full.save(target/(row["case_id"]+"_full.png"))
        for region in row["regions"]:
            xy=crop_rectangle(region["boxes"],native_size)
            if xy[2]-xy[0]<2 or xy[3]-xy[1]<2:raise ValueError("Registered crop unavailable; no invented pixels")
            crop=resized(rgb.crop(xy));crop.save(target/(row["case_id"]+"_"+region["region_id"]+".png"))
            records.append({"case_id":row["case_id"],"region_id":region["region_id"],"source_id":"image_1","parent_source_id":"image_0",
                "union_box":list(xy),"native_size":list(native_size),"full_size":list(full.size),"crop_size":list(crop.size),
                "normalization_min":low,"normalization_max":high,"crop_before_full_downsampling":True})
    write(root/"outputs/view_registry.private.json",records)
    write(root/"outputs/render_done.json",{"cases":len(manifest),"registered_regions":len(records),"finished_epoch":time.time()})


def collect(root):
    import torch
    from PIL import Image
    from transformers import AutoProcessor
    from run_e0 import Backend
    if any((root/"outputs"/name).exists() for name in ["predictions.private.jsonl","requests.private.jsonl","generation_failure.private.json"]):raise RuntimeError("No model replay")
    manifest=rows(root/"inputs/manifest.jsonl");ontology=json.loads((root/"inputs/ontology.private.json").read_text())
    registry={(x["case_id"],x["region_id"]):x for x in json.loads((root/"outputs/view_registry.private.json").read_text())}
    backend=Backend();processor=AutoProcessor.from_pretrained((root/"model_path.txt").read_text().strip(),local_files_only=True,min_pixels=128*28*28,max_pixels=384*28*28)
    count=0
    for i,row in enumerate(manifest):
        jobs=[(r["region_id"],c) for r in row["regions"] for c in CONDITIONS];random.Random(2026110819+i).shuffle(jobs)
        for region,condition in jobs:
            guard(root)
            if count>=64:raise RuntimeError("64-call hard cap")
            data=probe_dto(row,region,condition,ontology,registry[(row["case_id"],region)])
            fullpath=root/"outputs/views"/(row["case_id"]+"_full.png")
            other=fullpath if condition=="duplicate_full" else root/"outputs/views"/(row["case_id"]+"_"+region+".png")
            views=[Image.open(p).convert("RGB") for p in [fullpath,other]]
            prompt='Inspect ONLY requested_region (r1 red / r2 blue in the full image). The two views are from the same current image. Return one JSON object {"values":{"'+region+'.finding":["exact finding tag"],"'+region+'.location":["exact location tag"]}}. Use distinct exact strings from the supplied ontology. Empty list means unresolved, not absence. Do not infer patient side from screen side. No summary, parents, diagnosis beyond visible findings, or tool claims are requested.'
            messages=[{"role":"system","content":"Offline grounded-image readout diagnosis. No reference findings are supplied. Return the requested JSON."},{"role":"user","content":[{"type":"image","image":x} for x in views]+[{"type":"text","text":prompt+"\nINPUT:\n"+json.dumps(data,ensure_ascii=False)}]}]
            text=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
            batch=processor(text=[text],images=views,return_tensors="pt").to("cuda")
            if batch["input_ids"].shape[1]>8192 or batch["image_grid_thw"].shape[0]!=2:raise RuntimeError("Image count/input cap binding")
            n=batch["input_ids"].shape[1];forwards=[0];hook=backend.model.register_forward_pre_hook(lambda *a:forwards.__setitem__(0,forwards[0]+1));start=time.time()
            request={"case_id":row["case_id"],"region_id":region,"condition":condition,"attempt_index":count+1,
                "payload":data,"image_sources":[str(p.relative_to(root)) for p in [fullpath,other]],
                "actual_image_sizes":[list(x.size) for x in views],"input_tokens":n}
            with (root/"outputs/requests.private.jsonl").open("a") as f:f.write(json.dumps(request,ensure_ascii=False)+"\n")
            try:
                with torch.inference_mode():ids=backend.model.generate(**batch,do_sample=False,num_beams=1,max_new_tokens=768,use_cache=True)
                torch.cuda.synchronize();raw=processor.tokenizer.decode(ids[0,n:],skip_special_tokens=True)
            except Exception as exc:
                write(root/"outputs/generation_failure.private.json",dict(request,error=f"{type(exc).__name__}: {exc}",forwards=forwards[0],seconds=time.time()-start))
                raise
            finally:hook.remove()
            count+=1;response_id=f"call-{count}"
            record={"case_id":row["case_id"],"region_id":region,"condition":condition,"response_id":response_id,"payload":data,"raw_output":raw,
                "image_sources":[str(p.relative_to(root)) for p in [fullpath,other]],"actual_image_sizes":[list(x.size) for x in views],
                "input_tokens":n,"output_tokens":int(ids.shape[1]-n),"visual_tokens":int((batch["input_ids"]==backend.model.config.image_token_id).sum()),
                "forwards":forwards[0],"seconds":time.time()-start,"max_output_tokens":768,"epoch":time.time(),
                "peak_allocated_bytes":torch.cuda.max_memory_allocated(),"peak_reserved_bytes":torch.cuda.max_memory_reserved(),
                "extraction":extract_requested(raw,row["case_id"],region,response_id)}
            with (root/"outputs/predictions.private.jsonl").open("a") as f:f.write(json.dumps(record,ensure_ascii=False)+"\n")
            for x in views:x.close()
            del batch,ids
            (root/"outputs/progress.json").write_text(json.dumps({"calls":count,"planned_calls":64,"epoch":time.time()}));print("READOUT",count,flush=True)
    write(root/"outputs/collection_done.json",{"cases":16,"calls":count,"locked_epoch":time.time(),"torch":torch.__version__,"all_parameters_frozen":all(not p.requires_grad for p in backend.model.parameters())})


def evaluate(root):
    lock=json.loads((root/"outputs/collection_done.json").read_text())
    if lock["calls"]!=64 or (root/"outputs/metrics.json").exists():raise RuntimeError("Complete frozen predictions required")
    records=rows(root/"outputs/predictions.private.jsonl")
    manifest=rows(root/"inputs/manifest.jsonl")
    expected={(r["case_id"],region["region_id"],c) for r in manifest for region in r["regions"] for c in CONDITIONS}
    if len(records)!=64 or len(expected)!=64 or {(x["case_id"],x["region_id"],x["condition"]) for x in records}!=expected:raise ValueError("Duplicate/missing readout identity")
    label_epoch=time.time();labels=rows(root/"inputs/labels.private.jsonl")
    if len(labels)!=16 or {x["case_id"] for x in labels}!={x["case_id"] for x in manifest}:raise ValueError("Reference identity mismatch")
    truth={};metrics={c:{"source_scored_slots":0,"source_exact_slots":0,"requested_slots":64,"extracted_tag_slots":0,"source_matched_patients_with_mixture":0} for c in CONDITIONS}
    for row in labels:
        for region in row["regions"]:
            truth[(row["case_id"],region["region_id"]+".finding")]=region["finding_tags"]
            if region["location_scored"]:truth[(row["case_id"],region["region_id"]+".location")]=region["location_tags"]
    case_correct={c:{} for c in CONDITIONS}
    for r in records:
        c=r["condition"];cid=r["case_id"]
        for slot,value in r["extraction"]["values"].items():
            metrics[c]["extracted_tag_slots"]+=isinstance(value,list)
            if (cid,slot) not in truth:continue
            matched=isinstance(value,list) and set(value)==set(truth[(cid,slot)])
            metrics[c]["source_scored_slots"]+=1;metrics[c]["source_exact_slots"]+=matched
            case_correct[c].setdefault(cid,[]).append(int(matched))
    for c,m in metrics.items():
        m["source_exact_fraction"]=m["source_exact_slots"]/m["source_scored_slots"]
        m["source_matched_patients_with_mixture"]=sum(any(v) and not all(v) for v in case_correct[c].values())
    ids=sorted(case_correct[CONDITIONS[0]]);diff=[sum(case_correct[CONDITIONS[1]][i])/len(case_correct[CONDITIONS[1]][i])-sum(case_correct[CONDITIONS[0]][i])/len(case_correct[CONDITIONS[0]][i]) for i in ids]
    rng=random.Random(2026110820);boot=sorted(sum(diff[rng.randrange(16)] for _ in range(16))/16 for _ in range(2000))
    write(root/"outputs/metrics.json",{"conditions":metrics,"paired_patient_mean_difference":sum(diff)/16,"descriptive_percentile95":[boot[49],boot[1949]],
        "clinical_accuracy_established":False,"revision_effect_tested":False,"labels_opened_epoch":label_epoch,"predictions_locked_epoch":lock["locked_epoch"],
        "cost":{"calls":len(records),"input_tokens":sum(x["input_tokens"] for x in records),"output_tokens":sum(x["output_tokens"] for x in records),"visual_tokens":sum(x["visual_tokens"] for x in records),"forwards":sum(x["forwards"] for x in records),"generation_seconds":sum(x["seconds"] for x in records)},"finished_epoch":time.time()})


if __name__=="__main__":
    root=Path(os.environ["RUN_ROOT"])
    {"render":render,"collect":collect,"evaluate":evaluate}[os.environ["VIEW_STAGE"]](root)
