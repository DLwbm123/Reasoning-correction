import json
import unittest

from evidence_audit import build_request, bind_result, summarize


class EvidenceAuditChecks(unittest.TestCase):
    def setUp(self):
        self.row = {"question": "q", "choices": {"A": "feature a", "B": "feature b"}}
        self.case = {"initial": {"answer": "PRIVATE_ORIGINAL_SENTINEL"}, "reference_letter": "GT_SENTINEL",
            "evidence": [{"evidence_id": "private:E0", "source_id": "image_0", "assessability": "adequate", "observation": "a visible fact"}],
            "state": {"retained": []}, "answers": {"selective": {"answer": "A", "claims": []},
                                                  "reset": {"answer": "B", "claims": []}}}

    def result(self, payload, status="SUPPORTED", citation="E0"):
        return {"audits": [{"candidate_id": c["candidate_id"], "eligible": status == "SUPPORTED",
            "preserves_required_facts": True, "checks": [{"claim_index": u["claim_index"],
            "status": status, "evidence_ids": [citation] if citation else []} for u in c["units"]]}
            for c in payload["anonymous_candidates"]]}

    def test_payload_privacy_and_json_roundtrip_binding(self):
        payload, mapping, expected = build_request(self.row, self.case, 7)
        text = json.dumps(payload)
        for forbidden in ["PRIVATE_ORIGINAL_SENTINEL", "GT_SENTINEL", "selective", "reset", "private:E0"]:
            self.assertNotIn(forbidden, text)
        expected = json.loads(json.dumps(expected))
        bound = bind_result(self.result(payload), payload, mapping, expected)
        self.assertTrue(bound["schema_ok"])
        self.assertTrue(all(v["eligible"] for v in bound["by_branch"].values()))

    def test_final_answer_cannot_be_omitted_for_empty_explanation(self):
        p, m, e = build_request(self.row, self.case, 7)
        result = self.result(p); result["audits"][0]["checks"] = []
        b = bind_result(result, p, m, e)
        self.assertFalse(b["schema_ok"])
        self.assertFalse(b["by_branch"][m[result["audits"][0]["candidate_id"]]]["eligible"])

    def test_string_claim_is_preserved_without_invented_type_or_citation(self):
        self.case["answers"]["selective"]["claims"] = ["Original plain statement"]
        p, m, e = build_request(self.row, self.case, 7)
        aid = next(a for a, branch in m.items() if branch == "selective")
        candidate = next(c for c in p["anonymous_candidates"] if c["candidate_id"] == aid)
        unit = candidate["units"][1]
        self.assertEqual(unit["text"], "Original plain statement")
        self.assertEqual(unit["kind"], "untyped")
        self.assertNotIn("evidence_ids", unit)
        self.assertTrue(bind_result(self.result(p), p, m, json.loads(json.dumps(e)))["schema_ok"])

    def test_invented_or_inadequate_support_is_rejected(self):
        p, m, e = build_request(self.row, self.case, 7)
        self.assertFalse(bind_result(self.result(p, citation="FAKE"), p, m, e)["schema_ok"])
        p["evidence"][0]["assessability"] = "limited"
        self.assertFalse(bind_result(self.result(p), p, m, e)["schema_ok"])

    def test_duplicate_identity_rejected_and_unresolved_never_eligible(self):
        p, m, e = build_request(self.row, self.case, 7)
        result = self.result(p); result["audits"][1]["candidate_id"] = result["audits"][0]["candidate_id"]
        self.assertFalse(bind_result(result, p, m, e)["schema_ok"])
        b = bind_result(self.result(p, "UNRESOLVED", ""), p, m, e)
        self.assertTrue(b["schema_ok"])
        self.assertFalse(any(v["eligible"] for v in b["by_branch"].values()))

    def test_actual_permutation_mapping_and_preflight_denominators(self):
        p, m, e = build_request(self.row, self.case, 7)
        pp, mm, ee = build_request(self.row, self.case, 7, True)
        self.assertEqual(m["u0"], mm["u1"])
        records = []
        for i in range(32):
            for permutation in [0, 1]:
                records.append({"case_id": str(i), "permutation": permutation,
                    "bound": {"schema_ok": True, "by_branch": {"selective": {"eligible": True}, "reset": {"eligible": False}}}})
        self.assertTrue(summarize(records)["passed"])
        missing = summarize(records[:-1])
        self.assertFalse(missing["passed"])
        self.assertEqual(missing["planned_requests"], 64)
        for r in records:
            r["bound"]["by_branch"]["selective"]["eligible"] = False
        self.assertFalse(summarize(records)["passed"])  # all-reject is not admitted


if __name__ == "__main__":
    unittest.main()
