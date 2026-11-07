"""Separate, exact closed-label evaluator for the synthetic diagnostic."""
import json
import os
from pathlib import Path
import random
from replay_oracle import CONDITIONS


def main():
    root=Path(os.environ['RUN_ROOT'])
    labels=[json.loads(x) for x in (root/'inputs/labels.private.jsonl').read_text().splitlines()]
    cases={p.stem:json.loads(p.read_text()) for p in (root/'outputs/cases').glob('*.json')}
    scores=[]
    for label in labels:
        if label['case_id'] not in cases:continue
        c=cases[label['case_id']]
        row={'case_id':label['case_id']}
        for condition,*_ in CONDITIONS:
            row[condition]=int(c['answers'].get(condition,{}).get('modality')==label['reference_modality'])
        scores.append(row)
    n=len(scores)
    if not n:raise RuntimeError('no collected cases')
    metrics={'n':n,'n_planned':len(labels),'complete':n==len(labels),
             'scope':'synthetic, answer-revealing modality annotation diagnostic; not natural errors or medical performance',
             'conditions':{name:{'correct':sum(x[name] for x in scores),'accuracy':sum(x[name] for x in scores)/n}
                           for name,*_ in CONDITIONS},'paired_differences':{}}
    rng=random.Random(2026110704)
    pairs=[('none_noE','assistant_contradicted_noE'),('none_E','assistant_contradicted_E'),
           ('assistant_contradicted_E','assistant_contradicted_noE'),
           ('assistant_supported_E','assistant_contradicted_E'),
           ('quoted_contradicted_E','assistant_contradicted_E')]
    for a,b in pairs:
        samples=sorted(sum(x[a]-x[b] for x in rng.choices(scores,k=n))/n for _ in range(2000))
        metrics['paired_differences'][a+' minus '+b]={'difference':sum(x[a]-x[b] for x in scores)/n,
           '95ci':[samples[49],samples[1949]]}
    calls=[json.loads(x) for x in (root/'outputs/calls.jsonl').read_text().splitlines()]
    metrics['cost']={'generation_calls':len(calls),'failed_calls':sum(not c['parsed'] for c in calls),
        'output_tokens':sum(c['output_tokens'] for c in calls),'input_tokens':sum(c['input_tokens'] for c in calls),
        'visual_tokens':sum(c['visual_tokens'] for c in calls),'forward_calls':sum(c['forwards'] for c in calls),
        'sum_generation_seconds':sum(c['seconds'] for c in calls)}
    (root/'outputs/metrics.json').write_text(json.dumps(metrics,indent=2))
    print(json.dumps(metrics,indent=2))


if __name__=='__main__':main()
