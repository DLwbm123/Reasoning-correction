import os
import unittest
os.environ.setdefault('RUN_ROOT','/tmp/test-only')
from replay_oracle import request_text


class ReplayPermissions(unittest.TestCase):
    def setUp(self):
        self.row={'question':'What modality is used?',
            'premises':{'supported':'CORRECT_PREMISE','contradicted':'WRONG_PREMISE'},
            'oracle_evidence':[{'observation':'ORACLE_VALUE'}],'reference':'SECRET_REFERENCE'}

    def test_no_oracle_or_correct_pair_in_false_no_evidence(self):
        prefix,text=request_text(self.row,'assistant','contradicted',False)
        self.assertEqual(prefix[-1]['role'],'assistant')
        alltext=str(prefix)+text
        self.assertIn('WRONG_PREMISE',alltext)
        for forbidden in ['ORACLE_VALUE','CORRECT_PREMISE','SECRET_REFERENCE']:self.assertNotIn(forbidden,alltext)

    def test_fresh_query_has_no_history(self):
        prefix,text=request_text(self.row,'none',None,True)
        self.assertEqual(prefix,[])
        self.assertIn('ORACLE_VALUE',text)
        self.assertNotIn('WRONG_PREMISE',text)

    def test_quoted_history_does_not_create_an_assistant_turn(self):
        prefix,text=request_text(self.row,'quoted','contradicted',True)
        self.assertEqual(prefix,[])
        self.assertIn('WRONG_PREMISE',text)
        self.assertIn('ORACLE_VALUE',text)
