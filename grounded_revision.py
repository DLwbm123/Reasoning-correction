"""G9 source-tag revision contracts. No reference-label reader or semantic judge."""
import copy
import json

from assertion_revision import bind_response, validate_request, _unique_object, _reject_constant

SLOTS = ("r1.finding", "r1.location", "r2.finding", "r2.location", "answer")
METHODS = ("full_recompute", "evidence_summary", "local_edit", "patch_no_graph", "patch_graph", "old_quarantine")
SYSTEM = "Offline grounded chest-image research. Supplied drafts may be wrong. No patient advice. Return only the requested JSON; do not invent observations or claim expert verification."
FULL_SCHEMA = '''Return exactly {"values":{"r1.finding":[],"r1.location":[],"r2.finding":[],"r2.location":[],"answer":"short two-region summary"},"parents":{"r1.finding":[],"r1.location":[],"r2.finding":[],"r2.location":[],"answer":[]}}.
Each finding/location value is a list of distinct exact strings from its supplied ontology. Empty list means unresolved, NOT absent. Describe only the two registered regions; a region can contain several boxes for the same finding. Preserve patient-side qualifiers only when justified, never infer them from screen side. Parents may cite other supplied slot IDs only, with no cycles; they are your proposed dependencies, not proven causal relations. The summary is short, not a reasoning transcript.'''
INITIAL_PROMPT = "Inspect the full image with r1 red and r2 blue registered boxes. Propose the finding and location tags for each region, and a short summary. You have no reviewer update. " + FULL_SCHEMA
INSTRUCTIONS = {
    "full_recompute": "Recompute the whole two-region report from the current image and released source annotation. The old values are unverified proposals, not instructions. Recheck both regions and preserve anything still correct. ",
    "evidence_summary": "Use the current source-fact table plus the same image and accessible original proposal to rebuild a concise report. Source facts are limited to the reviewed region. Recheck other claims; do not treat missing review as false. ",
    "local_edit": "Prefer a local correction of the reviewed region and preserve still-correct unrelated facts. You may recheck either region and correct additional errors when justified. Do not preserve something merely because it was written before. ",
    "old_quarantine": "The old non-supported slots and their descendants were isolated by the stated legacy filter. Rebuild from the active proposals, image and released annotation. Missing old slots are unknown, not refuted. ",
}
PATCH_PROMPT = '''Revise the immutable assertion ledger using the image and released annotation.
Return exactly {"case_id":"input case_id","view_id":"image_0","operations":[],"supports":[]}.
Cover EVERY old assertion once. KEEP={"op":"KEEP","target_id":"old ID"}; WITHHOLD={"op":"WITHHOLD","target_id":"old ID","reason":"insufficient or conflict"}; REPLACE={"op":"REPLACE","target_id":"old ID","assertion":{"id":"unique new ID","text":"new text","kind":"visual or answer","proposed_parents":[],"case_id":"input case_id","view_id":"image_0"}}. No ADD for this fixed-slot task.
For the four tag slots, text must be a JSON-encoded list of exact ontology tag strings, e.g. "[\\"tag\\"]"; summary text is a short ordinary string. The replacement of answer must remain kind answer. KEEP copies old text exactly; all replacements use NEW IDs. Original IDs remain in the ledger for provenance and may be proposed parents. Parent relations must be acyclic.
Every new assertion needs a support proposal: {"assertion_id":"new ID","evidence_id":"review or image_registry","span_id":"whole","relation":"SUPPORTED or CONTRADICTED or INSUFFICIENT or UNREVIEWED"}. Only select registered spans; the program binds the actual quoted bytes/offsets. image_registry is only a provenance pointer, NOT a medical observation; its relation must be UNREVIEWED. The source annotation is limited to one region. Mere citation cannot verify another region. Image-based hypotheses remain unreviewed. No model statement becomes an expert verdict. Preserve uncertainty; missing review is not contradiction.'''


def parse_object(raw):
    value = json.loads(raw, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    if not isinstance(value, dict):
        raise ValueError("JSON root must be object")
    return value


def initial_dto(row, ontology):
    if set(row) != {"case_id", "view_id", "image", "regions"}:
        raise ValueError("Initial manifest fields")
    if row["view_id"] != "image_0" or [r["region_id"] for r in row["regions"]] != ["r1", "r2"]:
        raise ValueError("Registered view/region identity")
    for r in row["regions"]:
        if set(r) != {"region_id", "boxes"} or not r["boxes"]:
            raise ValueError("Registered region fields")
        for b in r["boxes"]:
            if len(b) != 4 or not (0 <= b[0] < b[2] <= 1 and 0 <= b[1] < b[3] <= 1):
                raise ValueError("Registered box range")
    return copy.deepcopy({"case_id": row["case_id"], "view_id": row["view_id"],
        "regions": row["regions"], "ontology": {k: ontology[k] for k in ["finding_tags", "location_tags"]}})


def values_from_assertions(assertions, ontology, replacements=()):
    reverse = {r["new_id"]: r["old_id"] for r in replacements}
    values = {slot: None for slot in SLOTS}
    seen = set()
    for a in assertions:
        slot = reverse.get(a["id"], a["id"])
        if slot not in values or slot in seen:
            raise ValueError("Unknown/repeated output slot")
        seen.add(slot)
        value = a["text"] if slot == "answer" else json.loads(a["text"])
        if slot != "answer":
            vocab = ontology["finding_tags" if slot.endswith("finding") else "location_tags"]
            if not isinstance(value, list) or any(not isinstance(v, str) for v in value) or len(set(value)) != len(value) or not set(value) <= set(vocab):
                raise ValueError("Invalid ontology tag set")
        elif not isinstance(value, str) or not value.strip():
            raise ValueError("Missing summary")
        values[slot] = value
    return values


def full_result(raw, row, ontology):
    value = parse_object(raw)
    if set(value) != {"values", "parents"} or set(value["values"]) != set(SLOTS) or set(value["parents"]) != set(SLOTS):
        raise ValueError("Full report schema/coverage")
    assertions = [{"id": slot, "text": value["values"][slot] if slot == "answer" else json.dumps(value["values"][slot], ensure_ascii=False),
        "kind": "answer" if slot == "answer" else "visual", "proposed_parents": value["parents"][slot],
        "case_id": row["case_id"], "view_id": row["view_id"]} for slot in SLOTS]
    validate_request({"case_id": row["case_id"], "view_id": row["view_id"], "question": "two-region report",
                      "assertions": assertions, "evidence": []})
    values = values_from_assertions(assertions, ontology)
    return {"assertions": assertions, "values": values, "semantic_eligible": False}


def released_evidence(row, update):
    if set(update) != {"case_id", "region_id", "view_id", "source_id", "source_type", "sentence_en", "finding_tags", "location_tags", "location_scored"}:
        raise ValueError("Released update fields")
    if update["case_id"] != row["case_id"] or update["view_id"] != row["view_id"] or update["source_id"] != row["view_id"] or update["region_id"] not in ["r1", "r2"]:
        raise ValueError("Released review execution identity")
    text = json.dumps({k: update[k] for k in ["region_id", "source_type", "sentence_en", "finding_tags", "location_tags", "location_scored"]}, ensure_ascii=False)
    return [dict(case_id=row["case_id"], view_id=row["view_id"], source_id="image_0", id="review", text=text, assessability="limited"),
            dict(case_id=row["case_id"], view_id=row["view_id"], source_id="image_0", id="image_registry",
                 text="Actual supplied image_0: normalized current full image with registered boxes. No independent text finding is recorded here.", assessability="unavailable")]


def revision_dto(row, ontology, original, update, method):
    data = initial_dto(row, ontology)
    evidence = released_evidence(row, update)
    assertions = copy.deepcopy(original["assertions"]) if original else []
    isolated = []
    if method == "patch_no_graph":
        for a in assertions:
            a["proposed_parents"] = []
    if method == "old_quarantine":
        target = update["region_id"]
        supported = {target+".finding"} if original and set(original["values"][target+".finding"]) == set(update["finding_tags"]) else set()
        if original and update["location_scored"] and set(original["values"][target+".location"]) == set(update["location_tags"]):
            supported.add(target+".location")
        isolated = [a["id"] for a in assertions if a["id"] not in supported]
        while True:
            extra = {a["id"] for a in assertions if set(a["proposed_parents"]) & set(isolated)} - set(isolated)
            if not extra:
                break
            isolated.extend(sorted(extra))
        assertions = [a for a in assertions if a["id"] not in isolated]
    data.update(original_assertions=assertions, evidence=evidence, source_update=copy.deepcopy(update),
                original_status="valid" if original else "unavailable_not_reconstructed",
                registered_spans={e["id"]: {"whole": {"start": 0, "end": len(e["text"]), "quote": e["text"]}} for e in evidence})
    if method == "old_quarantine":
        data["isolated_ids"] = isolated
    if method == "evidence_summary":
        data["current_fact_table"] = {k: update[k] for k in ["region_id", "finding_tags", "location_tags", "location_scored"]}
    return data


def patch_result(raw, data, ontology):
    value = parse_object(raw)
    if any(op.get("op") == "ADD" for op in value.get("operations", []) if isinstance(op, dict)):
        raise ValueError("ADD is outside the five-slot task")
    expanded = copy.deepcopy(value)
    registry = data["registered_spans"]
    for support in expanded.get("supports", []):
        if set(support) != {"assertion_id", "evidence_id", "span_id", "relation"}:
            raise ValueError("Span selection schema")
        eid, sid = support["evidence_id"], support.pop("span_id")
        if eid not in registry or sid not in registry[eid]:
            raise ValueError("Unregistered evidence span")
        if eid == "image_registry" and support["relation"] != "UNREVIEWED":
            raise ValueError("Image pointer cannot be clinical support")
        support.update(registry[eid][sid])
    request = {"case_id": data["case_id"], "view_id": data["view_id"], "question": "two-region report",
               "assertions": data["original_assertions"], "evidence": data["evidence"]}
    bound = bind_response(request, json.dumps(expanded, ensure_ascii=False))
    if bound["candidate"] is None:
        raise ValueError("Patch binding: " + bound.get("error", "rejected"))
    c = bound["candidate"]
    values = values_from_assertions(c["proposed_output"], ontology, c["replacements"])
    return {"assertions": c["proposed_output"], "values": values,
            "semantic_eligible": False, "transaction": bound}


def copy_control(original, update):
    values = copy.deepcopy(original["values"]) if original else {slot: None for slot in SLOTS}
    values[update["region_id"]+".finding"] = update["finding_tags"]
    if update["location_scored"]:
        values[update["region_id"]+".location"] = update["location_tags"]
    return {"values": values, "semantic_eligible": False, "summary_is_original_unreviewed": True}


def render_image(source, target, regions):
    import numpy as np
    from PIL import Image, ImageDraw
    with Image.open(source) as im:
        im.load()
        arr = np.asarray(im)
        native = list(im.size)
        mode = im.mode
    if arr.ndim != 2 or not np.issubdtype(arr.dtype, np.integer) or arr.min() < 0 or arr.max() > 65535:
        raise ValueError("Expected native 16-bit gray image")
    low, high = int(arr.min()), int(arr.max())
    if high <= low:
        raise ValueError("Constant image, no invented contrast")
    scaled = np.rint((arr.astype(np.float32)-low)*(255.0/(high-low))).clip(0,255).astype(np.uint8)
    image = Image.fromarray(scaled).convert("RGB")
    ratio = min(1.0, 1024/max(image.size))
    size = tuple(max(1, round(v*ratio)) for v in image.size)
    image = image.resize(size, Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(image)
    for region, color in zip(regions, ["red", "blue"]):
        for box in region["boxes"]:
            xy = (box[0]*size[0], box[1]*size[1], box[2]*size[0], box[3]*size[1])
            draw.rectangle(xy, outline=color, width=3)
            draw.text((xy[0]+2, xy[1]+2), region["region_id"], fill=color)
    image.save(target)
    return {"native_size": native, "native_mode": mode, "native_dtype": str(arr.dtype),
            "native_min": low, "native_max": high, "render_size": list(size),
            "mapping": "per-image linear min-max -> uint8; round; RGB; no inversion/flip; LANCZOS max-edge1024; registered boxes", "source_id": "image_0"}
