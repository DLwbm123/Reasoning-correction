"""Separate offline evaluation process with reference permissions."""
import csv
import json
import os
from pathlib import Path
import random
import statistics

BRANCHES = ['raw', 'selective', 'reset', 'random_drop', 'evidence_summary']


def score(record, reference):
    # No semantic judge, fuzzy answer matching, or changes to references.
    return isinstance(record, dict) and record.get('answer') == reference


def evaluate(root):
    labels = {r['case_id']: r for r in map(json.loads, (root / 'inputs/labels.private.jsonl').read_text().splitlines())}
    cases = {p.stem: json.loads(p.read_text()) for p in (root / 'outputs/cases').glob('*.json')}
    rows = []
    for cid, label in labels.items():
        r = cases.get(cid, {})
        state = r.get('state', {})
        row = {'case_id': cid, 'group_id': label['group_id'], 'collected': cid in cases,
               'initial_correct': int(score(r.get('initial'), label['reference_letter'])),
               'trigger': bool(state.get('deployment_trigger')), 'k_empty': not state.get('retained'),
               'graph_changes_k': state.get('flat', []) != state.get('retained', []),
               'adequate_evidence': any(e['assessability'] == 'adequate' for e in r.get('evidence', [])),
               'pipeline_error': bool(r.get('pipeline_error'))}
        for b in BRANCHES:
            answer = r.get('answers', {}).get(b, {})
            row[b] = int(score(answer, label['reference_letter']))
            # Forced replacement: an invalid candidate also destroys a previously valid answer.
            row[b + '_edited'] = answer.get('answer') != r.get('initial', {}).get('answer')
            row[b + '_invalid'] = answer.get('answer') not in ['A', 'B', 'C', 'D']
        rows.append(row)
    with (root / 'outputs/per_case_scores.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    rows = [r for r in rows if r['collected']]
    n = len(rows)
    if not n:
        raise RuntimeError('No collected cases to evaluate')
    initial_c = sum(x['initial_correct'] for x in rows)
    metrics = {'n_planned': len(labels), 'n_collected': len(cases), 'initial_accuracy': initial_c / n,
               'complete': len(cases) == len(labels), 'unrun_cases': len(labels) - len(cases),
               'denominator': 'collected cases including all parsing and pipeline failures; incomplete runs are descriptive only',
               'branches': {}, 'mechanism': {
                   k: sum(bool(x[k]) for x in rows) for k in ['trigger', 'k_empty', 'graph_changes_k', 'adequate_evidence', 'pipeline_error']}}
    rng = random.Random(20261107)
    for b in BRANCHES:
        wc = sum(not x['initial_correct'] and x[b] for x in rows)
        cw = sum(x['initial_correct'] and not x[b] for x in rows)
        final = sum(x[b] for x in rows)
        assert final - initial_c == wc - cw
        edits = sum(x[b + '_edited'] for x in rows)
        differences = []
        # Sampling is exactly one case per article: article-cluster bootstrap reduces to case resampling.
        for _ in range(2000):
            sample = rng.choices(rows, k=n)
            differences.append(sum(x[b] - x['initial_correct'] for x in sample) / n)
        differences.sort()
        metrics['branches'][b] = {'accuracy': final/n, 'delta_accuracy': (wc-cw)/n,
            'WC': wc, 'CW': cw, 'repair_rate': wc/(n-initial_c) if n != initial_c else None,
            'harm_given_initial_correct': cw/initial_c if initial_c else None,
            'population_harm': cw/n, 'edit_coverage': edits/n, 'harm_per_edit': cw/edits if edits else None,
            'invalid': sum(x[b+'_invalid'] for x in rows),
            'delta_accuracy_95ci': [differences[49], differences[1949]]}
    metrics['paired_branch_comparisons'] = {}
    for a,b in [('selective','raw'),('selective','reset'),('selective','random_drop'),('selective','evidence_summary')]:
        differences = sorted(sum(x[a]-x[b] for x in rng.choices(rows,k=n))/n for _ in range(2000))
        metrics['paired_branch_comparisons'][a+' minus '+b] = {
            'delta':sum(x[a]-x[b] for x in rows)/n, '95ci':[differences[49],differences[1949]]}
    calls = [json.loads(x) for x in (root/'outputs/calls.jsonl').read_text().splitlines()]
    times = sorted(x['seconds'] for x in calls)
    metrics['cost'] = {'generation_calls':len(calls), 'forward_calls':sum(x['forwards'] for x in calls),
        'input_tokens':sum(x['input_tokens'] for x in calls), 'visual_tokens':sum(x['visual_tokens'] for x in calls),
        'output_tokens':sum(x['output_tokens'] for x in calls), 'sum_generation_seconds':sum(times),
        'per_call_latency_p50':statistics.median(times) if times else None,
        'per_call_latency_p95':times[int(.95*(len(times)-1))] if times else None,
        'peak_gpu_allocated_bytes':max((x['peak_memory_allocated'] for x in calls),default=0),
        'peak_gpu_reserved_bytes':max((x['peak_memory_reserved'] for x in calls),default=0)}
    (root/'outputs/metrics.json').write_text(json.dumps(metrics,indent=2))
    return metrics


if __name__ == '__main__':
    print(json.dumps(evaluate(Path(os.environ['RUN_ROOT'])), indent=2))
