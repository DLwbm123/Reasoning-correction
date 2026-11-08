"""Aggregate completed G10 readout and costs; never re-admit a G9 output."""
import json
from pathlib import Path
import os


def main():
    root=Path(os.environ['ANALYSIS_ROOT'])
    supervisor=json.loads((root/'outputs/supervisor.json').read_text())
    if supervisor['status']!='completed':raise RuntimeError('Wait for all stages; no survivor analysis')
    metrics=json.loads((root/'outputs/metrics.json').read_text())
    records=[json.loads(s) for s in (root/'outputs/predictions.private.jsonl').read_text().splitlines()]
    labels=[json.loads(s) for s in (root/'inputs/labels.private.jsonl').read_text().splitlines()]
    ontology=json.loads((root/'inputs/ontology.private.json').read_text())
    truth={(x['case_id'],r['region_id']+'.'+kind):r[kind+'_tags'] for x in labels for r in x['regions']
           for kind in ['finding','location'] if kind=='finding' or r['location_scored']}
    result={}
    for condition in ['duplicate_full','full_plus_crop']:
        subset=[r for r in records if r['condition']==condition]
        if len(subset)!=32:raise ValueError('Complete 32 calls per condition required')
        table={k:{'requested':32,'extracted_tag_lists':0,'source_scored':0,'source_exact':0,'proposed_strings':0,'out_of_vocabulary_strings':0} for k in ['finding','location']}
        for r in subset:
            for slot,value in r['extraction']['values'].items():
                kind=slot.split('.')[1];m=table[kind]
                if isinstance(value,list):
                    m['extracted_tag_lists']+=1;m['proposed_strings']+=len(value)
                    m['out_of_vocabulary_strings']+=sum(s not in ontology[kind+'_tags'] for s in value)
                if (r['case_id'],slot) in truth:
                    m['source_scored']+=1
                    m['source_exact']+=isinstance(value,list) and set(value)==set(truth[(r['case_id'],slot)])
        result[condition]={'fields':table,'calls':len(subset),'response_envelopes_rejected':sum(r['extraction']['transport']['status']=='rejected' for r in subset),
            'output_cap_hits':sum(r['output_tokens']>=r['max_output_tokens'] for r in subset),
            'cost':{k:sum(r[k] for r in subset) for k in ['input_tokens','output_tokens','visual_tokens','forwards','seconds']},
            'peak_allocated_bytes':max(r['peak_allocated_bytes'] for r in subset),'peak_reserved_bytes':max(r['peak_reserved_bytes'] for r in subset)}
        if sum(x['source_exact'] for x in table.values())!=metrics['conditions'][condition]['source_exact_slots']:raise ValueError('Frozen metric mismatch')
    target=root/'results';target.mkdir(exist_ok=True)
    (target/'readout_and_cost.json').write_text(json.dumps(result,indent=2))
    (target/'metrics.json').write_text(json.dumps(metrics,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
