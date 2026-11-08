"""Lossless field transport for a future protocol, NOT a G9 admission repair.

No ontology, labels, graph inference, clinical verification or model invocation.
Each extracted value retains its exact source characters and dispatch identity.
"""
import json
import math
import re


FACT_SLOTS = ("r1.finding", "r1.location", "r2.finding", "r2.location")


def _object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate_json_key")
        value[key] = item
    return value


def _finite_float(text):
    value = float(text)
    if not math.isfinite(value):
        raise ValueError("nonfinite_json_number")
    return value


def _constant(text):
    raise ValueError("nonfinite_json_constant")


DECODER = json.JSONDecoder(object_pairs_hook=_object, parse_float=_finite_float, parse_constant=_constant)


def _space(text, pos):
    while pos < len(text) and text[pos] in " \r\n\t":
        pos += 1
    return pos


def _members(text, start):
    """Index values in one already-validated JSON object; preserve original spans."""
    pos = _space(text, start+1)
    result = {}
    while text[pos] != "}":
        key, end = DECODER.raw_decode(text, pos)
        pos = _space(text, end)
        if text[pos] != ":":
            raise ValueError("object_colon_missing")
        value_start = _space(text, pos+1)
        _, value_end = DECODER.raw_decode(text, value_start)
        result[key] = (value_start, value_end)
        pos = _space(text, value_end)
        if text[pos] == "}":
            break
        if text[pos] != ",":
            raise ValueError("object_separator_missing")
        pos = _space(text, pos+1)
    return result


def _cell(text, span, offset, path, value):
    if span is None:
        return {"status":"missing", "source_path":None, "source_format":"missing",
                "raw_span":None, "raw_text":None, "value":None}
    start, end = span
    return {"status":"present", "source_path":path, "source_format":type(value).__name__,
            "raw_span":[offset+start, offset+end], "raw_text":text[start:end], "value":value}


def extract_fact_transport(raw, *, case_id, view_id, response_id):
    """Accept only documented envelopes/locations, without changing model values.

    Dispatch IDs are assigned by the caller, never inferred from model text.
    Missing metadata does not erase facts; ambiguous fields reject the envelope.
    Status `extracted` means transport only, never full-case/semantic eligibility.
    """
    receipt = {"case_id":case_id,"view_id":view_id,"response_id":response_id,
        "raw_output":raw if isinstance(raw,str) else repr(raw), "parsed_output":None,
        "status":"rejected", "facts":None, "metadata":None,
        "complete_case_admitted":False, "semantic_eligible":False}
    try:
        if any(not isinstance(v,str) or not v.strip() for v in [case_id,view_id,response_id]):
            raise ValueError("missing_dispatch_identity")
        if not isinstance(raw,str):
            raise ValueError("raw_output_must_be_text")
        fence = re.fullmatch(r'\s*```(?:json)?[ \t]*\r?\n(?P<body>.*?)\r?\n```[ \t]*\s*',raw,re.I|re.S)
        text, offset = (fence.group("body"),fence.start("body")) if fence else (raw,0)
        receipt["source_format"] = "fenced_json" if fence else "plain_json"
        start = _space(text,0)
        value, end = DECODER.raw_decode(text,start)
        if _space(text,end) != len(text) or not isinstance(value,dict):
            raise ValueError("expected_one_complete_object")
        receipt["parsed_output"] = value
        if "values" not in value or set(value)-{"values","answer","parents"}:
            raise ValueError("unexpected_root_fields")
        values = value["values"]
        if not isinstance(values,dict) or set(values)-set(FACT_SLOTS)-{"answer"}:
            raise ValueError("unexpected_fact_fields")
        if "answer" in values and "answer" in value:
            raise ValueError("ambiguous_answer_locations")
        root_spans = _members(text,start)
        fact_spans = _members(text,root_spans["values"][0])
        facts = {}
        for slot in FACT_SLOTS:
            cell = _cell(text,fact_spans.get(slot),offset,["values",slot],values.get(slot))
            if slot in values:
                item = values[slot]
                if not isinstance(item,list) or any(not isinstance(s,str) for s in item):
                    cell["status"] = "invalid_tag_list"
                elif len(set(item)) != len(item):
                    cell["status"] = "duplicate_tags"
                else:
                    cell["status"] = "present_tag_list" if item else "present_empty_list"
            facts[slot] = cell
        if "answer" in values:
            answer = _cell(text,fact_spans["answer"],offset,["values","answer"],values["answer"])
        else:
            answer = _cell(text,root_spans.get("answer"),offset,["answer"],value.get("answer"))
        if answer["status"] != "missing":
            answer["status"] = "present_text" if isinstance(answer["value"],str) and answer["value"].strip() else "present_nontext_or_empty"
        answer["semantics_verified"] = False
        parents = _cell(text,root_spans.get("parents"),offset,["parents"],value.get("parents"))
        if parents["status"] != "missing":
            parents["status"] = "present_unvalidated_mapping" if isinstance(parents["value"],dict) else "invalid_mapping_type"
        parents["graph_usable"] = False
        parents["clinical_causality_verified"] = False
        receipt.update(facts=facts,metadata={"answer":answer,"parents":parents},
            status="extracted" if all(c["status"] in {"present_tag_list","present_empty_list"} for c in facts.values()) else "partial")
    except (ValueError,TypeError,IndexError) as exc:
        receipt["error"] = str(exc)
    return receipt
