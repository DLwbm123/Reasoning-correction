"""Identity-preserving proposal adapter; structural validity is not semantic admission.

This is separate from the frozen E0/G1--G8 code. All relations returned by a
generator remain model proposals, including those labelled SUPPORTED.
"""
import copy
import json
from pathlib import Path


KINDS = {"visual", "context", "knowledge", "reasoning", "answer"}
RELATIONS = {"SUPPORTED", "CONTRADICTED", "INSUFFICIENT", "UNREVIEWED"}
SYSTEM = """Propose an offline research revision, not a medical adjudication.
Return one JSON object with exactly case_id, view_id, operations, supports.
Cover every original assertion exactly once with KEEP, WITHHOLD or REPLACE.
KEEP: {op,target_id}. WITHHOLD: {op,target_id,reason}, reason insufficient or conflict.
REPLACE: {op,target_id,assertion}. ADD: {op,assertion}.
A new assertion has exactly id,text,kind,proposed_parents,case_id,view_id.
Never overwrite an old ID or change KEEP text. Use a new ID for every replacement.
kind is visual/context/knowledge/reasoning/answer; final answer is an assertion too.
Each support has exactly assertion_id,evidence_id,start,end,quote,relation.
Use zero-based Python Unicode character offsets [start,end) in the cited evidence
text, with quote exactly equal to that span. relation is SUPPORTED, CONTRADICTED,
INSUFFICIENT or UNREVIEWED. New assertions need at least one registered citation.
References do not establish truth. Preserve uncertainty, negation, object and scope.
Missing evidence is not contradiction. Parent edges are proposals, not proven rules.
Do not add observations, outside facts, clinical thresholds or an untracked answer.
"""


def _keys(value, expected):
    if not isinstance(value, dict) or set(value) != set(expected.split()):
        raise ValueError("unexpected_or_missing_fields")


def _text(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("empty_or_nonstring_field")


def _assertion(value, scope):
    _keys(value, "id text kind proposed_parents case_id view_id")
    for key in ("id", "text"):
        _text(value[key])
    if value["kind"] not in KINDS:
        raise ValueError("unknown_assertion_kind")
    if (value["case_id"], value["view_id"]) != scope:
        raise ValueError("assertion_scope_mismatch")
    parents = value["proposed_parents"]
    if not isinstance(parents, list) or any(not isinstance(p, str) for p in parents):
        raise ValueError("invalid_parent_ids")
    if len(set(parents)) != len(parents):
        raise ValueError("duplicate_parent")


def _parents(assertions):
    pending = {a["id"]: set(a["proposed_parents"]) for a in assertions}
    if any(not ps <= pending.keys() for ps in pending.values()):
        raise ValueError("unknown_parent")
    while pending:
        ready = {aid for aid, ps in pending.items() if not ps & pending.keys()}
        if not ready:
            raise ValueError("cyclic_proposed_parents")
        for aid in ready:
            del pending[aid]


def validate_request(request):
    _keys(request, "case_id view_id question assertions evidence")
    for key in ("case_id", "view_id", "question"):
        _text(request[key])
    scope = request["case_id"], request["view_id"]
    if not isinstance(request["assertions"], list) or not request["assertions"]:
        raise ValueError("missing_original_assertions")
    ids = set()
    for a in request["assertions"]:
        _assertion(a, scope)
        if a["id"] in ids:
            raise ValueError("duplicate_original_id")
        ids.add(a["id"])
    if sum(a["kind"] == "answer" for a in request["assertions"]) != 1:
        raise ValueError("original_answer_slot_required")
    _parents(request["assertions"])
    if not isinstance(request["evidence"], list):
        raise ValueError("evidence_not_list")
    evidence_ids = set()
    for e in request["evidence"]:
        _keys(e, "id text case_id view_id source_id assessability")
        for key in ("id", "text", "source_id"):
            _text(e[key])
        if (e["case_id"], e["view_id"]) != scope:
            raise ValueError("evidence_scope_mismatch")
        if e["source_id"] != request["view_id"]:
            raise ValueError("evidence_source_view_mismatch")
        if e["assessability"] not in {"adequate", "limited", "unavailable"}:
            raise ValueError("unknown_assessability")
        if e["id"] in evidence_ids:
            raise ValueError("duplicate_evidence_id")
        evidence_ids.add(e["id"])


def messages_for(request):
    """Only the closed inference schema reaches a new message list."""
    validate_request(request)
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": json.dumps(request, ensure_ascii=False, allow_nan=False)}]


def _unique_object(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError("duplicate_json_key")
        obj[key] = value
    return obj


def _reject_constant(value):
    raise ValueError("nonfinite_json")


def _apply(request, response):
    _keys(response, "case_id view_id operations supports")
    scope = request["case_id"], request["view_id"]
    if (response["case_id"], response["view_id"]) != scope:
        raise ValueError("response_scope_mismatch")
    if not isinstance(response["operations"], list) or not isinstance(response["supports"], list):
        raise ValueError("operations_and_supports_must_be_lists")
    old = {a["id"]: a for a in request["assertions"]}
    added, decisions, output, withheld, replaced = {}, {}, [], [], []
    for operation in response["operations"]:
        if not isinstance(operation, dict):
            raise ValueError("operation_not_object")
        op = operation.get("op")
        fields = {"KEEP": "op target_id", "WITHHOLD": "op target_id reason",
                  "REPLACE": "op target_id assertion", "ADD": "op assertion"}
        if op not in fields:
            raise ValueError("unknown_operation")
        _keys(operation, fields[op])
        if op != "ADD":
            target = operation["target_id"]
            if not isinstance(target, str) or target not in old or target in decisions:
                raise ValueError("unknown_or_repeated_target")
            decisions[target] = op
        if op == "KEEP":
            output.append(old[target])
        if op == "WITHHOLD":
            if operation["reason"] not in {"insufficient", "conflict"}:
                raise ValueError("unknown_withholding_reason")
            withheld.append({"id": target, "reason": operation["reason"]})
        if op in {"ADD", "REPLACE"}:
            a = operation["assertion"]
            _assertion(a, scope)
            if a["id"] in old or a["id"] in added:
                raise ValueError("new_id_overwrites_or_repeats")
            if op == "REPLACE" and old[target]["kind"] == "answer" and a["kind"] != "answer":
                raise ValueError("answer_replacement_must_remain_answer_slot")
            added[a["id"]] = a
            output.append(a)
            if op == "REPLACE":
                replaced.append({"old_id": target, "new_id": a["id"]})
    if decisions.keys() != old.keys():
        raise ValueError("incomplete_original_coverage")
    _parents([*old.values(), *added.values()])
    if sum(a["kind"] == "answer" for a in output) > 1:
        raise ValueError("multiple_output_answers")
    evidence = {e["id"]: e for e in request["evidence"]}
    citations, seen = [], set()
    for s in response["supports"]:
        _keys(s, "assertion_id evidence_id start end quote relation")
        if not isinstance(s["assertion_id"], str) or s["assertion_id"] not in old.keys() | added.keys():
            raise ValueError("unknown_supported_assertion")
        if not isinstance(s["evidence_id"], str) or s["evidence_id"] not in evidence:
            raise ValueError("unknown_evidence")
        if s["relation"] not in RELATIONS:
            raise ValueError("unknown_relation")
        e, start, end = evidence[s["evidence_id"]], s["start"], s["end"]
        if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(e["text"]):
            raise ValueError("invalid_evidence_span")
        if s["quote"] != e["text"][start:end]:
            raise ValueError("quote_does_not_match_span")
        key = s["assertion_id"], s["evidence_id"], start, end, s["relation"]
        if key in seen:
            raise ValueError("duplicate_support")
        seen.add(key)
        citations.append(dict(s, assessor="model_proposal", semantic_admission="unreviewed"))
    if not added.keys() <= {s["assertion_id"] for s in citations}:
        raise ValueError("new_assertion_without_citation")
    # Copy the whole result: callers cannot mutate the immutable ledger indirectly.
    return copy.deepcopy({"original_assertions": request["assertions"],
        "new_assertions": list(added.values()), "proposed_output": output,
        "withheld": withheld, "replacements": replaced, "support_proposals": citations,
        "output_semantic_status": "unreviewed", "semantic_eligible": False,
        "answer_withheld": not any(a["kind"] == "answer" for a in output)})


def bind_response(request, raw):
    """A failure has no applied candidate, and preserves raw/parsed proposal data."""
    receipt = {"request": copy.deepcopy(request), "raw_output": raw if isinstance(raw, str) else repr(raw),
               "parsed_output": None, "status": "rejected", "candidate": None}
    try:
        validate_request(request)
        if not isinstance(raw, str):
            raise ValueError("response_not_serialized_text")
        response = json.loads(raw, object_pairs_hook=_unique_object,
                              parse_constant=_reject_constant)
        receipt["parsed_output"] = response
        candidate = _apply(request, response)
        receipt.update(status="structurally_valid_proposal", candidate=candidate)
    except (ValueError, TypeError, KeyError) as exc:
        receipt["error"] = str(exc)
    return receipt


def execute_revision(request, complete, receipt_path):
    """One completion, no retry. `complete` consumes fresh messages and returns raw JSON.

    The caller owns its model, image binding, budgets and independent supervision.
    This text adapter neither opens images nor claims a real-model run occurred.
    The receipt is private and opened exclusively before making any model call.
    """
    with Path(receipt_path).open("x", encoding="utf-8") as f:
        try:
            messages = messages_for(request)
            raw = complete(messages)
            receipt = bind_response(request, raw)
        except Exception as exc:
            receipt = {"request": copy.deepcopy(request), "raw_output": None,
                       "parsed_output": None, "status": "execution_failed",
                       "candidate": None, "error": f"{type(exc).__name__}: {exc}"}
        f.write(json.dumps(receipt, ensure_ascii=False, allow_nan=False) + "\n")
    return receipt
