"""Select a small train-only PadChest-GR source pool, without model outputs.

Reads existing split volumes with virtual seeks; never assembles/extracts the full
archive. Patient IDs, finding text and source filenames stay in ignored inputs.
"""
import bisect
import ast
import csv
import io
import json
import os
from pathlib import Path
import random
import re
import struct
import time
import zipfile


class SplitArchive(io.RawIOBase):
    def __init__(self, paths):
        self.paths = list(paths)
        self.ends = [0]
        for path in self.paths:
            self.ends.append(self.ends[-1] + path.stat().st_size)
        self.position = 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        position = offset + (0 if whence == 0 else self.position if whence == 1 else self.ends[-1])
        if position < 0:
            raise ValueError("Negative archive seek")
        self.position = position
        return position

    def read(self, size=-1):
        remaining = min(self.ends[-1] - self.position,
                        self.ends[-1] if size < 0 else size)
        chunks = []
        while remaining > 0:
            index = bisect.bisect_right(self.ends, self.position) - 1
            local = self.position - self.ends[index]
            count = min(remaining, self.ends[index+1] - self.position)
            with self.paths[index].open("rb") as f:
                f.seek(local)
                block = f.read(count)
            if len(block) != count:
                raise IOError("Short read in existing split archive")
            chunks.append(block)
            self.position += count
            remaining -= count
        return b"".join(chunks)


def selected_report(text, study_id):
    # Decode only this training study's object, not held-out finding objects.
    match = re.search(r'"StudyID"\s*:\s*' + re.escape(json.dumps(study_id)), text)
    if match is None:
        raise ValueError("Selected training study absent from reports")
    start = text.rfind("{", 0, match.start())
    value, _ = json.JSONDecoder().raw_decode(text[start:])
    if value["StudyID"] != study_id:
        raise ValueError("Report identity binding failed")
    return value


def positive_pairs(report):
    findings = [f for f in report["findings"] if f.get("abnormal") is True
                and f.get("boxes") and f.get("labels") and f.get("progression") is None]
    for i, first in enumerate(findings):
        for second in findings[i+1:]:
            # Separate category labels are not asserted to be causally independent.
            if set(first["labels"]).isdisjoint(second["labels"]):
                return [first, second]
    return None


def build_task_records(chosen):
    """Separate initial-image DTOs, deferred reviewer updates and evaluation labels."""
    manifests, updates, labels = [], [], []
    for row in chosen:
        cid, pair = row["case_id"], row["findings"]
        regions = [{"region_id": f"r{i+1}", "boxes": f["boxes"]} for i, f in enumerate(pair)]
        manifests.append({"case_id": cid, "view_id": "image_0",
                          "image": f"images/{cid}.png", "regions": regions})
        target = row["correction_target_index"]
        finding = pair[target]
        updates.append({"case_id": cid, "region_id": f"r{target+1}",
            "view_id": "image_0", "source_id": "image_0", "source_type": "released_dataset_annotation",
            "sentence_en": finding["sentence_en"], "finding_tags": finding["labels"],
            "location_tags": finding["locations"],
            "location_scored": bool(finding["locations"])})
        labels.append({"case_id": cid, "regions": [{"region_id": f"r{i+1}",
            "finding_tags": f["labels"], "location_tags": f["locations"],
            "location_scored": bool(f["locations"])} for i, f in enumerate(pair)]})
    return manifests, updates, labels


def training_ontology(rows):
    labels, locations = set(), set()
    for row in rows:
        if float(row["boxes_count"] or 0) > 0:
            labels.add(row["label"])
            locs = ast.literal_eval(row["locations"])
            if not isinstance(locs, list) or any(not isinstance(s, str) for s in locs):
                raise ValueError("Location ontology format")
            locations.update(locs)
    return {"finding_tags": sorted(labels), "location_tags": sorted(locations),
            "source": "official train, current-image positive rows; not selected-model output"}


def main():
    source = Path(os.environ["SOURCE_ROOT"])
    run = Path(os.environ["JOB_ROOT"])
    started = json.loads((run/"started.json").read_text())
    if (run/"inputs/source_pool.private.jsonl").exists():
        raise RuntimeError("Existing source pool cannot be regenerated")
    inputs = run/"inputs"
    (inputs/"images").mkdir(parents=True, exist_ok=True)
    rows, metadata = [], {}
    with zipfile.ZipFile(source/"metadata/master_table.csv.zip") as z:
        with z.open("master_table.csv") as f:
            for row in csv.DictReader(io.TextIOWrapper(f)):
                # Held-out finding columns are not used in selection or inference.
                key = row["StudyID"]
                identity = {k: row[k] for k in ["StudyID", "PatientID", "split"]}
                if key in metadata and metadata[key] != identity:
                    raise ValueError("Inconsistent study split identity")
                metadata[key] = identity
                if row["split"] == "train" and row["prior_study"] == "False":
                    rows.append(row)
    patients = {}
    for m in metadata.values():
        patients.setdefault(m["PatientID"], set()).add(m["split"])
    if any(len(v) != 1 for v in patients.values()):
        raise ValueError("Official patient splits overlap")
    grouped = {}
    for row in rows:
        if float(row["boxes_count"] or 0) > 0 and not row["progression_status"]:
            grouped.setdefault(row["StudyID"], []).append(row)
    candidates = sorted(sid for sid, rr in grouped.items() if len({r["label"] for r in rr}) >= 2)
    random.Random(2026110814).shuffle(candidates)
    report_text = (source/"metadata/grounded_reports_20240819.json").read_text()
    chosen, seen_patients = [], set()
    inspected = 0
    for study in candidates:
        if time.time() >= started["deadline_epoch"] - 7200:
            raise RuntimeError("Original preparation/collection cutoff")
        patient = metadata[study]["PatientID"]
        if patient in seen_patients:
            continue
        report = selected_report(report_text, study)
        pair = positive_pairs(report)
        inspected += 1
        if pair is None:
            continue
        if any(r["ImageID"] != report["ImageID"] for r in grouped[study]):
            raise ValueError("Current-image metadata and report mismatch")
        chosen.append({"case_id": f"g{len(chosen):04d}", "study_id": study,
            "patient_id": patient, "split": "official_train", "image_id": report["ImageID"],
            "findings": pair, "all_current_findings": report["findings"],
            "correction_target_index": random.Random(2026110815+len(chosen)).randrange(2)})
        seen_patients.add(patient)
        if len(chosen) == 32:
            break
    if len(chosen) != 32:
        raise RuntimeError("Insufficient predeclared source eligibility; no replacement protocol")
    ontology = training_ontology(rows)
    for row in chosen:
        for finding in row["findings"]:
            if not set(finding["labels"]) <= set(ontology["finding_tags"]) or not set(finding["locations"]) <= set(ontology["location_tags"]):
                raise ValueError("Source JSON categories do not match training CSV ontology")
    with (inputs/"ontology.private.json").open("x") as f:
        json.dump(ontology, f, indent=2)
    # Freeze identities BEFORE reading any image. Extraction failures do not resample.
    with (inputs/"source_pool.private.jsonl").open("x") as f:
        for row in chosen:
            f.write(json.dumps(row, ensure_ascii=False)+"\n")
    extract_images(source, run, chosen, started, inspected)


def extract_images(source, run, chosen, started, inspected=None):
    inputs = run/"inputs"
    if any((inputs/"images").iterdir()) or (run/"source_readiness.json").exists():
        raise RuntimeError("Image extraction already began; no automatic repeat")
    parts = sorted((source/"Padchest_GR_files").glob("PadChest_GR.zip.[0-9][0-9][0-9]"))
    if [int(p.suffix[1:]) for p in parts] != list(range(1, len(parts)+1)):
        raise ValueError("Missing archive volume")
    copied = 0
    image_status = []
    with SplitArchive(parts) as reader, zipfile.ZipFile(reader) as z:
        by_basename = {}
        for info in z.infolist():
            by_basename.setdefault(Path(info.filename).name, []).append(info)
        for row in chosen:
            if time.time() >= started["deadline_epoch"] - 7200:
                raise RuntimeError("Original preparation/collection cutoff")
            info, = by_basename[row["image_id"]]
            if info.file_size > 100_000_000 or copied + info.file_size > 1_000_000_000:
                raise RuntimeError("Conservative extraction storage guard")
            path = inputs/"images"/(row["case_id"]+".png")
            with path.open("xb") as out, z.open(info) as src:
                while block := src.read(1024*1024):
                    out.write(block)
            copied += info.file_size
            with path.open("rb") as f:
                header = f.read(33)
            if len(header) != 33 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[8:16] != b"\x00\x00\x00\rIHDR":
                raise ValueError("Extracted source is not a registered PNG")
            width, height, depth, color = struct.unpack(">IIBB", header[16:26])
            if not width or not height or depth not in {1, 2, 4, 8, 16} or color not in {0, 2, 3, 4, 6}:
                raise ValueError("Invalid PNG header")
            image_status.append({"case_id": row["case_id"], "size": [width, height],
                "bit_depth": depth, "color_type": color, "pixel_decode_status": "not_executed",
                "original_encoded_bytes": path.stat().st_size})
    (inputs/"image_status.private.json").write_text(json.dumps(image_status, indent=2))
    for name, records in zip(["manifest.jsonl", "updates.private.jsonl", "labels.private.jsonl"],
                             build_task_records(chosen)):
        with (inputs/name).open("x") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False)+"\n")
    summary = {"source": "existing local PadChest-GR", "selected_current_training_studies": len(chosen),
        "selected_distinct_patients": len({r["patient_id"] for r in chosen}), "selected_finding_entries": 2*len(chosen),
        "inspected_training_report_objects": inspected, "selection_seed": 2026110814,
        "heldout_finding_objects_decoded": 0, "selected_patient_overlap_official_heldout": 0,
        "images_extracted": len(image_status), "new_original_image_bytes": copied,
        "original_archive_parts": len(parts), "existing_archive_bytes_readonly": sum(p.stat().st_size for p in parts),
        "assembled_archive_bytes": 0, "model_calls": 0, "new_expert_adjudications": 0,
        "png_bit_depths": sorted({r["bit_depth"] for r in image_status}),
        "png_color_types": sorted({r["color_type"] for r in image_status}),
        "full_pixel_decodes": 0,
        "source_temporal_findings_selected": 0,
        "status": "source_pool_ready_not_inference_started", "finished_epoch": time.time()}
    (run/"source_readiness.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    if os.environ.get("EXTRACT_FROZEN_POOL") == "1":
        root = Path(os.environ["JOB_ROOT"])
        chosen = [json.loads(s) for s in (root/"inputs/source_pool.private.jsonl").read_text().splitlines()]
        if len(chosen) != 32 or len({r["case_id"] for r in chosen}) != 32:
            raise RuntimeError("Frozen source pool count/identity invalid")
        extract_images(Path(os.environ["SOURCE_ROOT"]), root, chosen,
                       json.loads((root/"started.json").read_text()))
    else:
        main()
