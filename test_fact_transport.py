"""Synthetic structure fixtures only; never opens G9 data or generates a case."""
import json
import unittest

from fact_transport import extract_fact_transport, FACT_SLOTS


class FactTransportChecks(unittest.TestCase):
    def extract(self,raw):
        return extract_fact_transport(raw,case_id="case-1",view_id="image_0",response_id="call-1")

    def values(self):
        return {slot:["Unverified tag "+str(i)] for i,slot in enumerate(FACT_SLOTS)}

    def test_three_documented_root_shapes_preserve_facts_without_imputing_metadata(self):
        values=self.values()
        variants=[{"values":values},{"values":values,"answer":["original list summary"]},
                  {"values":values,"answer":"original string summary","parents":{"answer":["r1.finding"]}}]
        for value in variants:
            with self.subTest(keys=list(value)):
                result=self.extract(json.dumps(value))
                self.assertEqual(result["status"],"extracted")
                self.assertEqual({s:c["value"] for s,c in result["facts"].items()},values)
                self.assertFalse(result["semantic_eligible"])
                self.assertFalse(result["complete_case_admitted"])
        missing=self.extract(json.dumps(variants[0]))["metadata"]
        self.assertEqual(missing["parents"]["status"],"missing")
        self.assertIsNone(missing["parents"]["value"])
        self.assertEqual(missing["answer"]["status"],"missing")

    def test_fenced_unicode_and_escaped_strings_keep_exact_raw_source_spans(self):
        raw='  ```json\r\n {"values": {"r1.finding": [ "may be present", "不确定", "quote \\\"x\\\"" ], "r1.location": [], "r2.finding": ["tag"], "r2.location": ["place"], "answer":"an original summary"}}\r\n```  '
        result=self.extract(raw)
        self.assertEqual(result["status"],"extracted")
        self.assertEqual(result["source_format"],"fenced_json")
        for cell in [*result["facts"].values(),result["metadata"]["answer"]]:
            start,end=cell["raw_span"]
            self.assertEqual(raw[start:end],cell["raw_text"])
            self.assertEqual(json.loads(cell["raw_text"]),cell["value"])
        self.assertEqual(result["facts"]["r1.finding"]["raw_text"],'[ "may be present", "不确定", "quote \\\"x\\\"" ]')

    def test_missing_null_empty_and_invalid_fact_cells_stay_distinct(self):
        result=self.extract('{"values":{"r1.finding":null,"r1.location":[],"r2.finding":"not a list"}}')
        self.assertEqual(result["status"],"partial")
        self.assertEqual(result["facts"]["r1.finding"]["source_format"],"NoneType")
        self.assertEqual(result["facts"]["r1.location"]["status"],"present_empty_list")
        self.assertEqual(result["facts"]["r2.finding"]["value"],"not a list")
        self.assertEqual(result["facts"]["r2.location"]["status"],"missing")
        self.assertIsNone(result["facts"]["r2.location"]["raw_span"])

    def test_summary_locations_and_wrong_metadata_types_do_not_erase_facts(self):
        values=self.values();values["answer"]=["do not join or rewrite this list"]
        result=self.extract(json.dumps({"values":values,"parents":[]}))
        self.assertEqual(result["status"],"extracted")
        self.assertEqual(result["metadata"]["answer"]["source_path"],["values","answer"])
        self.assertEqual(result["metadata"]["answer"]["source_format"],"list")
        self.assertEqual(result["metadata"]["answer"]["value"],values["answer"])
        self.assertEqual(result["metadata"]["parents"]["status"],"invalid_mapping_type")
        self.assertFalse(result["metadata"]["parents"]["graph_usable"])

    def test_no_ontology_filter_casefold_sort_or_dedup(self):
        values=self.values();values["r1.finding"]=[" Z ","a","NOT_IN_ANY_VOCAB"]
        result=self.extract(json.dumps({"values":values}))
        self.assertEqual(result["facts"]["r1.finding"]["value"],values["r1.finding"])
        values["r1.finding"]=["duplicate","duplicate"]
        result=self.extract(json.dumps({"values":values}))
        self.assertEqual(result["status"],"partial")
        self.assertEqual(result["facts"]["r1.finding"]["status"],"duplicate_tags")
        self.assertEqual(result["facts"]["r2.finding"]["status"],"present_tag_list")

    def test_duplicate_keys_alias_conflicts_and_extra_reference_fields_reject_whole_envelope(self):
        raw='{"values":{"r1.finding":[],"r1.\\u0066inding":["other"]}}'
        variants=[raw,'{"values":{},"values":{}}',
                  '{"values":{"answer":"same"},"answer":"same"}',
                  '{"values":{"answer":"first"},"answer":"second"}',
                  '{"values":{},"reference":"PRIVATE_REFERENCE"}',
                  '{"values":{"reference":"PRIVATE_REFERENCE"}}']
        for raw in variants:
            with self.subTest(raw=raw):
                result=self.extract(raw)
                self.assertEqual(result["status"],"rejected")
                self.assertIsNone(result["facts"])
                self.assertEqual(result["raw_output"],raw)

    def test_only_one_documented_json_envelope_and_finite_numbers(self):
        for raw in ['[]','{"values":{}} trailing','prefix {"values":{}}',
                    '```python\n{"values":{}}\n```','```json\n{"values":{}}\n``` trailing',
                    '{"values":{},"answer":NaN}','{"values":{},"answer":1e999}','{"values":{},"answer":Infinity}']:
            with self.subTest(raw=raw):
                self.assertEqual(self.extract(raw)["status"],"rejected")

    def test_dispatch_identity_and_unknown_graph_are_never_inferred(self):
        result=self.extract(json.dumps({"values":self.values(),"parents":{}}))
        self.assertEqual((result["case_id"],result["view_id"],result["response_id"]),("case-1","image_0","call-1"))
        self.assertEqual(result["metadata"]["parents"]["value"],{})
        self.assertFalse(result["metadata"]["parents"]["graph_usable"])
        bad=extract_fact_transport('{"values":{}}',case_id="",view_id="image_0",response_id="call-1")
        self.assertEqual(bad["status"],"rejected")


if __name__ == "__main__": unittest.main()
