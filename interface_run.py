"""Small successor interface diagnosis; the completed E0 code/outputs stay frozen."""
import json
from pathlib import Path
import run_e0 as base
from mechanism import payload


def bind_single_observation(result, test):
    observations=result.get('observations',[])
    if not isinstance(observations,list) or len(observations)!=1:
        return None
    obs=observations[0]
    if not isinstance(obs,dict) or not isinstance(obs.get('observation'),str) or not obs['observation'].strip():
        return None
    if obs.get('assessability') not in {'adequate','limited','unavailable'}:
        return None
    # Provenance is the actual execution context of this one-question/one-image call.
    # No semantic observation value, quality or correctness is invented or repaired.
    return {'test_id':test['test_id'],'source_id':'image_0',
            'assessability':obs['assessability'],'observation':obs['observation'],
            'limitations':obs.get('limitations',[])}


class InterfaceBackend(base.Backend):
    def __init__(self):
        super().__init__()
        self.prompts=json.loads((base.CODE/'prompts_interface.json').read_text())

    def generate(self,role,data,image,case_id,branch=None):
        if role!='observe':
            return super().generate(role,data,image,case_id,branch)
        observations=[]
        for test in data['public_tests']:
            one=payload('observe',public_tests=[test],image_manifest=data['image_manifest'])
            result=super().generate('observe',one,image,case_id,branch='single_test_'+test['test_id'])
            bound=bind_single_observation(result,test)
            with (base.ROOT/'outputs/execution_bindings.jsonl').open('a') as f:
                f.write(json.dumps({'case_id':case_id,'requested_test_id':test['test_id'],
                    'single_actual_source_id':'image_0','original_output':result,
                    'execution_bound':bound is not None})+'\n')
            if bound is not None:observations.append(bound)
        return {'observations':observations}


if __name__=='__main__':
    base.Backend=InterfaceBackend
    base.main()
