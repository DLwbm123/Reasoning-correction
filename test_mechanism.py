import json
import unittest
from mechanism import payload, public_tests, rebuild, random_facts, summary_facts


class PermissionsAndState(unittest.TestCase):
    def test_observer_rejects_private_fields(self):
        for field in ['claims', 'answer', 'reference', 'claim_ids', 'raw_trace']:
            with self.assertRaises(ValueError):
                payload('observe', public_tests=[], image_manifest=[], **{field: 'SECRET'})

    def test_private_planner_mapping_not_forwarded(self):
        plan = {'tests': [{'test_id': 't', 'claim_ids': ['SECRET'], 'public_question': 'Describe both sides',
                           'scope': 'image_verifiable', 'answer': 'SECRET'}]}
        self.assertNotIn('SECRET', json.dumps(public_tests(plan)))

    def test_repair_rejects_old_answer_and_trace(self):
        with self.assertRaises(ValueError):
            payload('repair', question='q', choices={}, evidence=[], historical_facts=[], raw_trace='SECRET')

    def test_descendants_isolated_independent_later_fact_retained(self):
        claims = [{'claim_id': 'a', 'text': 'SECRET', 'kind': 'visual', 'parents': []},
                  {'claim_id': 'b', 'text': 'derived', 'kind': 'visual', 'parents': ['a']},
                  {'claim_id': 'c', 'text': 'independent', 'kind': 'visual', 'parents': []}]
        evidence = [{'evidence_id': 'e', 'assessability': 'adequate'}]
        bindings = {'verdicts': [{'claim_id': x, 'status': 'CONTRADICTED' if x == 'a' else 'SUPPORTED',
                                 'evidence_ids': ['e'], 'grounded_text': 'clean '+x} for x in 'abc']}
        result = rebuild(claims, bindings, evidence)
        self.assertEqual(result['isolated_ids'], ['a', 'b'])
        self.assertEqual([x['text'] for x in result['retained']], ['clean c'])
        self.assertNotIn('SECRET', json.dumps(payload('repair', question='q', choices={}, evidence=evidence,
                                                     historical_facts=result['retained'])))

    def test_missing_and_wrong_source_never_contradiction(self):
        c = [{'claim_id': 'a', 'kind': 'visual', 'text': 'x', 'parents': []}]
        for eid in [[], ['wrong']]:
            r = rebuild(c, {'verdicts': [{'claim_id': 'a', 'status': 'CONTRADICTED', 'evidence_ids': eid}]}, [])
            self.assertFalse(r['deployment_trigger'])
            self.assertEqual(r['verdicts']['a']['status'], 'UNRESOLVED')

    def test_random_drop_uses_count_without_error_identity(self):
        c = [{'claim_id': str(i), 'kind': 'visual', 'text': str(i), 'parents': []} for i in range(6)]
        self.assertEqual(len(random_facts(c, [{'text': 'one'}], 4, len)), 1)

    def test_invalid_summary_reference_excluded(self):
        self.assertEqual(summary_facts({'facts': [{'text': 'x', 'evidence_ids': ['wrong']}]}, [], 2), [])


if __name__ == '__main__':
    unittest.main()
