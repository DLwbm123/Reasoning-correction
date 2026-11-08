"""G10 new-patient source preparation; excludes G9, never selects by model events."""
import csv
import io
import json
import os
from pathlib import Path
import random
import shutil
import time
import zipfile

from prepare_grounded_sources import SplitArchive, positive_pairs, selected_report


def main():
    root=Path(os.environ["JOB_ROOT"]);source=Path(os.environ["SOURCE_ROOT"])
    previous=Path(os.environ["PREVIOUS_ROOT"])
    started=json.loads((root/"started.json").read_text())
    inputs=root/"inputs";inputs.mkdir(exist_ok=False);(inputs/"images").mkdir()
    # IDs only are used from the previous developer pool, not its event/metric table.
    excluded={json.loads(s)["patient_id"] for s in (previous/"inputs/source_pool.private.jsonl").read_text().splitlines()}
    groups={};patient_splits={}
    with zipfile.ZipFile(source/"metadata/master_table.csv.zip") as z:
        with z.open("master_table.csv") as f:
            for row in csv.DictReader(io.TextIOWrapper(f)):
                patient_splits.setdefault(row["PatientID"],set()).add(row["split"])
                if row["split"]=="train" and row["prior_study"]=="False" and row["PatientID"] not in excluded and float(row["boxes_count"] or 0)>0 and not row["progression_status"]:
                    groups.setdefault(row["StudyID"],[]).append(row)
    if any(len(v)!=1 for v in patient_splits.values()):raise ValueError("Patient split overlap")
    candidates=sorted(k for k,v in groups.items() if len({r["label"] for r in v})>=2)
    random.Random(2026110818).shuffle(candidates)
    text=(source/"metadata/grounded_reports_20240819.json").read_text()
    chosen=[];seen=set()
    for sid in candidates:
        if time.time()>=started["collection_deadline_epoch"]-60:raise RuntimeError("Original collection clock")
        patient=groups[sid][0]["PatientID"]
        if patient in seen:continue
        report=selected_report(text,sid);pair=positive_pairs(report)
        if pair is None:continue
        if any(r["ImageID"]!=report["ImageID"] for r in groups[sid]):raise ValueError("Current image mismatch")
        chosen.append({"case_id":f"v{len(chosen):04d}","patient_id":patient,"study_id":sid,
            "image_id":report["ImageID"],"findings":pair});seen.add(patient)
        if len(chosen)==16:break
    if len(chosen)!=16:raise RuntimeError("Insufficient fixed eligibility; no resampling protocol")
    with (inputs/"source_pool.private.jsonl").open("x") as f:
        for row in chosen:f.write(json.dumps(row,ensure_ascii=False)+"\n")
    manifests=[];labels=[]
    for row in chosen:
        manifests.append({"case_id":row["case_id"],"view_id":"image_0","image":"images/"+row["case_id"]+".png",
            "regions":[{"region_id":f"r{i+1}","boxes":f["boxes"]} for i,f in enumerate(row["findings"])]})
        labels.append({"case_id":row["case_id"],"regions":[{"region_id":f"r{i+1}","finding_tags":f["labels"],
            "location_tags":f["locations"],"location_scored":bool(f["locations"])} for i,f in enumerate(row["findings"])]})
    for name,records in [("manifest.jsonl",manifests),("labels.private.jsonl",labels)]:
        with (inputs/name).open("x") as f:
            for row in records:f.write(json.dumps(row,ensure_ascii=False)+"\n")
    shutil.copy2(previous/"inputs/ontology.private.json",inputs/"ontology.private.json")
    parts=sorted((source/"Padchest_GR_files").glob("PadChest_GR.zip.[0-9][0-9][0-9]"))
    total=0
    with SplitArchive(parts) as reader,zipfile.ZipFile(reader) as z:
        index={}
        for info in z.infolist():index.setdefault(Path(info.filename).name,[]).append(info)
        for row in chosen:
            if time.time()>=started["collection_deadline_epoch"]-60:raise RuntimeError("Original collection clock")
            info,=index[row["image_id"]]
            if info.file_size>100_000_000 or total+info.file_size>500_000_000:raise RuntimeError("Storage guard")
            with z.open(info) as src,(inputs/"images"/(row["case_id"]+".png")).open("xb") as out:
                while block:=src.read(1024*1024):out.write(block)
            total+=info.file_size
    summary={"cases":16,"patients":len(seen),"overlap_g9_patients":len(seen & excluded),
        "source_finding_entries":32,"source_location_entries_scored":sum(r["location_scored"] for x in labels for r in x["regions"]),
        "selection_seed":2026110818,"new_native_image_bytes":total,"vocabulary_reused_unchanged":True,
        "model_outcomes_used_to_select":False,"model_calls":0,"finished_epoch":time.time()}
    (root/"source_receipt.json").write_text(json.dumps(summary,indent=2));print(json.dumps(summary))


if __name__=="__main__":main()
