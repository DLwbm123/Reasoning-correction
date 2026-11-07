"""Anonymous, evidence-only candidate audit. No reference or original answer access."""
import json
import random

PAYLOAD_KEYS = {"question", "choices", "evidence", "required_facts", "anonymous_candidates"}
STATUSES = {"SUPPORTED", "CONTRADICTED", "UNRESOLVED"}


def build_request(manifest, case, seed, reverse=False):
    order = ["selective", "reset"]
    random.Random(seed).shuffle(order)
    if reverse:
        order.reverse()
    refs = {e["evidence_id"]: "E"+str(i) for i, e in enumerate(case.get("evidence", []))}
    def remap(value):
        if isinstance(value, dict): return {k: remap(v) for k, v in value.items()}
        if isinstance(value, list): return [remap(v) for v in value]
        return refs.get(value, value) if isinstance(value, str) else value
    evidence = remap(case.get("evidence", []))
    candidates, bindings, expected = [], {}, {}
    for i, branch in enumerate(order):
        aid = "u"+str(i)
        output = case.get("answers", {}).get(branch, {})
        letter = output.get("answer")
        units = [{"claim_index": -1, "kind": "reasoning",
                  "text": manifest["choices"].get(letter, "") if isinstance(letter, str) else ""}]
        for index, claim in enumerate(output.get("claims", [])):
            if isinstance(claim, dict):
                unit = {"claim_index": index, "kind": claim.get("kind", "untyped"),
                        "text": claim.get("text", ""), "source_format": "object"}
            else:
                unit = {"claim_index": index, "kind": "untyped",
                        "text": claim if isinstance(claim, str) else json.dumps(claim, ensure_ascii=False),
                        "source_format": "string" if isinstance(claim, str) else "other_json"}
            units.append(unit)
        candidates.append({"candidate_id": aid, "answer_letter": letter,
            "answerability": output.get("answerability"), "units": units})
        bindings[aid] = branch
        expected[aid] = {"valid_answer": letter in list("ABCD"),
                         "units": {u["claim_index"]: u["kind"] for u in units}}
    data = {"question": manifest["question"], "choices": manifest["choices"], "evidence": evidence,
        "required_facts": remap(case.get("state", {}).get("retained", [])),
        "anonymous_candidates": candidates}
    assert set(data) == PAYLOAD_KEYS
    return data, bindings, expected


def bind_result(result, payload, bindings, expected):
    """Fail closed on malformed coverage or references, without inventing observations."""
    ids = {e["evidence_id"] for e in payload["evidence"]}
    adequate = {e["evidence_id"] for e in payload["evidence"] if e["assessability"] == "adequate"}
    errors, mapped = [], {}
    audits = result.get("audits") if isinstance(result, dict) else None
    if not isinstance(audits, list) or len(audits) != len(bindings):
        return {"schema_ok": False, "errors": ["missing_audits"], "by_branch": {}}
    seen = set()
    for audit in audits:
        if not isinstance(audit, dict):
            errors.append("audit_not_object"); continue
        aid = audit.get("candidate_id")
        if aid not in bindings or aid in seen:
            errors.append("candidate_id_mismatch"); continue
        seen.add(aid)
        local = []
        if type(audit.get("eligible")) is not bool or type(audit.get("preserves_required_facts")) is not bool:
            local.append("invalid_boolean")
        checks = audit.get("checks")
        required = {int(index): kind for index, kind in expected[aid]["units"].items()}
        observed = set()
        all_supported = True
        if not isinstance(checks, list):
            checks = []; local.append("missing_checks")
        for check in checks:
            if not isinstance(check, dict):
                local.append("check_not_object"); continue
            index = check.get("claim_index")
            if type(index) is not int or index not in required or index in observed:
                local.append("claim_index_mismatch"); continue
            observed.add(index)
            status, cited = check.get("status"), check.get("evidence_ids")
            if status not in STATUSES:
                local.append("invalid_status")
            if not isinstance(cited, list) or any(not isinstance(v, str) for v in cited):
                local.append("invalid_citations"); cited = []
            if not set(cited) <= ids:
                local.append("invented_citation")
            if status == "SUPPORTED" and (index == -1 or required[index] != "knowledge"):
                if not cited or not set(cited) <= adequate:
                    local.append("support_without_adequate_source")
            all_supported = all_supported and status == "SUPPORTED"
        if observed != set(required):
            local.append("incomplete_claim_coverage")
        if audit.get("eligible") is True and (not all_supported or not audit.get("preserves_required_facts")):
            local.append("inconsistent_eligibility")
        eligible = (not local and audit.get("eligible") is True and all_supported
                    and audit.get("preserves_required_facts") is True and expected[aid]["valid_answer"])
        mapped[bindings[aid]] = {"eligible": bool(eligible), "anonymous_id": aid,
            "original_audit": audit, "validation_errors": local}
        errors.extend(local)
    if seen != set(bindings):
        errors.append("incomplete_candidate_coverage")
    return {"schema_ok": not errors, "errors": errors, "by_branch": mapped}


def summarize(records):
    planned = 64
    valid = sum(r["bound"]["schema_ok"] for r in records)
    by_case = {}
    for r in records:
        by_case.setdefault(r["case_id"], {})[r["permutation"]] = r
    comparisons = agreements = accepted = rejected = 0
    for pair in by_case.values():
        first, second = pair.get(0), pair.get(1)
        for branch in ["selective", "reset"]:
            comparisons += 1
            good = first and second and first["bound"]["schema_ok"] and second["bound"]["schema_ok"]
            if good:
                agreements += first["bound"]["by_branch"][branch]["eligible"] == second["bound"]["by_branch"][branch]["eligible"]
            if first and first["bound"]["schema_ok"]:
                yes = first["bound"]["by_branch"][branch]["eligible"]
                accepted += yes; rejected += not yes
    # Missing records/pairs cannot inflate coverage or agreement denominators.
    consistency = agreements/64
    passed = len(records) == planned and valid/planned >= .90 and consistency >= .80 and accepted > 0 and rejected > 0
    return {"status": "passed" if passed else "gate_failed", "passed": passed,
        "planned_cases": 32, "planned_requests": planned, "completed_requests": len(records),
        "schema_source_valid_requests": valid, "schema_source_valid_rate": valid/planned,
        "eligible_agreements": agreements, "planned_candidate_pairs": 64,
        "eligible_agreement_rate": consistency, "canonical_accepted_candidates": accepted,
        "canonical_rejected_candidates": rejected, "expert_adjudications": 0,
        "interpretation": "engineering/nondegeneracy only; not semantic audit accuracy or clinical efficacy"}
