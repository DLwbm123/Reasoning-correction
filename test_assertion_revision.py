"""Transaction-boundary checks with fabricated text, not medical evidence."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from assertion_revision import bind_response, execute_revision, messages_for


class AssertionRevisionChecks(unittest.TestCase):
    def setUp(self):
        self.scope = {"case_id": "case-1", "view_id": "image_0"}
        self.request = dict(self.scope, question="Describe the supplied facts.", assertions=[
            self.a("a", "Possibly a round region."), self.a("b", "A separate marker is visible."),
            self.a("answer", "uncertain", "answer")], evidence=[dict(self.scope,
                id="e1", text="圆形 region; marker visible.", source_id="image_0", assessability="limited")])
        self.response = dict(self.scope, operations=[
            {"op": "KEEP", "target_id": x} for x in ["a", "b", "answer"]], supports=[])

    def a(self, aid, text, kind="visual", parents=()):
        return dict(self.scope, id=aid, text=text, kind=kind, proposed_parents=list(parents))

    def cite(self, aid="new", **updates):
        value = dict(assertion_id=aid, evidence_id="e1", start=0, end=2,
                     quote="圆形", relation="SUPPORTED")
        value.update(updates)
        return value

    def bind(self, response=None, request=None):
        return bind_response(request or self.request, json.dumps(response or self.response, ensure_ascii=False))

    def replacement(self):
        response = copy.deepcopy(self.response)
        response["operations"][0] = {"op": "REPLACE", "target_id": "a",
            "assertion": self.a("new", "A round region.")}
        response["supports"] = [self.cite()]
        return response

    def test_keep_is_exact_and_never_semantically_admitted_by_model_status(self):
        self.response["supports"] = [self.cite("a")]
        before = copy.deepcopy(self.request)
        result = self.bind()
        self.assertEqual(result["status"], "structurally_valid_proposal")
        c = result["candidate"]
        self.assertEqual(c["proposed_output"], before["assertions"])
        self.assertFalse(c["semantic_eligible"])
        self.assertEqual(c["support_proposals"][0]["assessor"], "model_proposal")
        self.assertEqual(c["support_proposals"][0]["semantic_admission"], "unreviewed")
        self.request["assertions"][0]["text"] = "Changed later by caller"
        self.assertEqual(c["original_assertions"], before["assertions"])

    def test_replace_and_add_create_new_ids_and_preserve_old_ledger(self):
        response = self.replacement()
        response["operations"].append({"op": "ADD", "assertion": self.a("extra", "Another proposal.")})
        response["supports"].append(self.cite("extra", relation="INSUFFICIENT"))
        c = self.bind(response)["candidate"]
        self.assertEqual(c["replacements"], [{"old_id": "a", "new_id": "new"}])
        self.assertEqual(c["original_assertions"], self.request["assertions"])
        self.assertEqual(c["proposed_output"][1], self.request["assertions"][1])
        self.assertEqual(len(c["new_assertions"]), 2)
        self.assertFalse(c["semantic_eligible"])

    def test_withholding_preserves_unknown_history_and_does_not_refute_descendants(self):
        self.request["assertions"][1]["proposed_parents"] = ["a"]
        self.response["operations"][0] = {"op": "WITHHOLD", "target_id": "a", "reason": "insufficient"}
        c = self.bind()["candidate"]
        self.assertEqual(c["withheld"], [{"id": "a", "reason": "insufficient"}])
        self.assertIn(self.request["assertions"][1], c["proposed_output"])
        self.assertEqual(c["output_semantic_status"], "unreviewed")
        self.assertEqual(c["original_assertions"][0]["text"], "Possibly a round region.")

    def test_missing_repeated_and_mutating_operations_reject_atomically(self):
        variants = []
        missing = copy.deepcopy(self.response); missing["operations"].pop(); variants.append(missing)
        repeated = copy.deepcopy(self.response); repeated["operations"].append({"op": "WITHHOLD", "target_id": "a", "reason": "conflict"}); variants.append(repeated)
        overwrite = self.replacement(); overwrite["operations"][0]["assertion"]["id"] = "a"; variants.append(overwrite)
        mutate = copy.deepcopy(self.response); mutate["operations"][0]["text"] = "Certainly a round region."; variants.append(mutate)
        duplicate = self.replacement(); duplicate["operations"].append(copy.deepcopy(duplicate["operations"][0])); variants.append(duplicate)
        original = copy.deepcopy(self.request)
        for response in variants:
            with self.subTest(response=response):
                result = self.bind(response)
                self.assertEqual(result["status"], "rejected")
                self.assertIsNone(result["candidate"])
                self.assertIsNotNone(result["parsed_output"])
                self.assertEqual(self.request, original)

    def test_case_view_and_label_contamination_are_rejected_before_dispatch(self):
        variants = []
        for obj in [self.request, self.request["assertions"][0], self.request["evidence"][0]]:
            bad = copy.deepcopy(self.request)
            dest = bad if obj is self.request else bad["assertions"][0] if obj is self.request["assertions"][0] else bad["evidence"][0]
            dest["reference"] = "PRIVATE_LABEL_SENTINEL"; variants.append(bad)
        for key in ["case_id", "view_id"]:
            for field in ["assertions", "evidence"]:
                bad = copy.deepcopy(self.request); bad[field][0][key] = "another"; variants.append(bad)
        bad = copy.deepcopy(self.request); bad["evidence"][0]["source_id"] = "another_image"; variants.append(bad)
        for request in variants:
            with self.subTest(request=request), tempfile.TemporaryDirectory() as tmp:
                calls = []
                result = execute_revision(request, lambda m: calls.append(m), Path(tmp)/"receipt.json")
                self.assertFalse(calls)
                self.assertEqual(result["status"], "execution_failed")
                self.assertIsNone(result["candidate"])
        for key in ["case_id", "view_id"]:
            response = copy.deepcopy(self.response); response[key] = "another"
            self.assertIsNone(self.bind(response)["candidate"])

    def test_span_binding_unicode_unknown_ids_and_forged_assessor(self):
        good = self.replacement()
        self.assertIsNotNone(self.bind(good)["candidate"])
        for updates in [dict(start=True), dict(start=-1), dict(end=1000), dict(end=0),
                        dict(quote="different"), dict(evidence_id="invented"),
                        dict(assertion_id="invented"), dict(assessor="expert")]:
            response = copy.deepcopy(good); response["supports"][0].update(updates)
            with self.subTest(updates=updates):
                self.assertIsNone(self.bind(response)["candidate"])
        response = copy.deepcopy(good); response["supports"] = []
        self.assertIsNone(self.bind(response)["candidate"])

    def test_parent_cycles_unknown_parents_and_untracked_answers(self):
        for parents in [["missing"], ["new"]]:
            response = self.replacement(); response["operations"][0]["assertion"]["proposed_parents"] = parents
            self.assertIsNone(self.bind(response)["candidate"])
        response = self.replacement(); response["answer"] = "untracked final answer"
        self.assertIsNone(self.bind(response)["candidate"])
        response = self.replacement(); response["operations"][0]["assertion"]["kind"] = "answer"
        self.assertIsNone(self.bind(response)["candidate"])
        response = copy.deepcopy(self.response)
        response["operations"][2] = {"op": "REPLACE", "target_id": "answer",
                                     "assertion": self.a("new", "Untracked answer as a visual fact")}
        response["supports"] = [self.cite()]
        self.assertIsNone(self.bind(response)["candidate"])
        self.response["operations"][2] = {"op": "WITHHOLD", "target_id": "answer", "reason": "conflict"}
        self.assertTrue(self.bind()["candidate"]["answer_withheld"])

    def test_strict_serialization_and_raw_failures_are_retained(self):
        raw = json.dumps(self.response)
        variants = [raw.replace('"case_id": "case-1"', '"case_id":"other","case_id":"case-1"', 1),
                    "[" + raw + "]", raw + " trailing", "```json\n" + raw + "\n```", "{", "NaN"]
        for value in variants:
            result = bind_response(self.request, value)
            self.assertEqual(result["status"], "rejected")
            self.assertEqual(result["raw_output"], value)
            self.assertIsNone(result["candidate"])

    def test_actual_serialized_callback_binding_and_no_repeat_receipt(self):
        calls = []
        def complete(messages):
            calls.append(copy.deepcopy(messages))
            dto = json.loads(messages[1]["content"])
            self.assertEqual(dto, self.request)
            return json.dumps(dict(self.scope, operations=[{"op": "KEEP", "target_id": a["id"]}
                for a in dto["assertions"]], supports=[]))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"receipt.json"
            result = execute_revision(self.request, complete, path)
            self.assertEqual(json.loads(path.read_text()), result)
            self.assertEqual(result["status"], "structurally_valid_proposal")
            with self.assertRaises(FileExistsError):
                execute_revision(self.request, complete, path)
            self.assertEqual(len(calls), 1)
        messages = messages_for(self.request)
        messages.append({"role": "assistant", "content": "old cache"})
        self.assertEqual(len(messages_for(self.request)), 2)

    def test_backend_and_parse_failures_persist_without_retry_or_fabricated_candidate(self):
        for response in ["{", RuntimeError("backend stopped")]:
            calls = []
            def complete(messages):
                calls.append(messages)
                if isinstance(response, Exception):
                    raise response
                return response
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp)/"receipt.json"
                record = execute_revision(self.request, complete, path)
                self.assertEqual(json.loads(path.read_text()), record)
                self.assertIsNone(record["candidate"])
                self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
