import os
import unittest
os.environ.setdefault('RUN_ROOT','/tmp/test-only')
from interface_run import bind_single_observation


class ExecutionBinding(unittest.TestCase):
    def test_only_execution_provenance_is_assigned(self):
        raw={'observations':[{'test_id':'hallucinated','source_id':'hallucinated',
            'observation':'unclear boundary','assessability':'limited'}]}
        result=bind_single_observation(raw,{'test_id':'t2'})
        self.assertEqual(result['test_id'],'t2')
        self.assertEqual(result['source_id'],'image_0')
        self.assertEqual(result['observation'],'unclear boundary')
        self.assertEqual(result['assessability'],'limited')

    def test_missing_or_multiple_observations_not_fabricated(self):
        self.assertIsNone(bind_single_observation({'observations':[]},{'test_id':'t1'}))
        self.assertIsNone(bind_single_observation({'observations':[{},{}]},{'test_id':'t1'}))

    def test_no_observation_from_invented_status(self):
        self.assertIsNone(bind_single_observation({'observations':[{'observation':'x','assessability':'certain'}]},
                                                 {'test_id':'t1'}))
