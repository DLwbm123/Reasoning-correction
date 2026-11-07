"""State operations and role permissions; no reference-answer access."""
import itertools
import json
import random

ROLE_KEYS = {
    'initial': {'question', 'choices'},
    'plan': {'question', 'claims', 'tools'},
    'observe': {'public_tests', 'image_manifest'},
    'bind': {'question', 'claims', 'evidence'},
    'summarize': {'question', 'evidence', 'max_facts'},
    'repair': {'question', 'choices', 'evidence', 'historical_facts'},
}


def payload(role, **values):
    if set(values) != ROLE_KEYS[role]:
        raise ValueError(f'Unexpected fields for {role}: {set(values)}')
    # Detach caller state; the backend always creates fresh messages.
    return json.loads(json.dumps(values))


def claims_from(result):
    out, known = [], set()
    for raw in result.get('claims', [])[:6]:
        if not isinstance(raw, dict):
            continue
        cid = str(raw.get('claim_id', ''))
        if not cid or cid in known or not isinstance(raw.get('text'), str):
            continue
        parents = raw.get('parents', [])
        if not isinstance(parents, list) or any(p not in known for p in parents):
            continue
        if raw.get('kind') not in {'visual', 'context', 'knowledge', 'reasoning'}:
            continue
        out.append({k: raw[k] for k in ('claim_id', 'text', 'kind', 'parents')})
        known.add(cid)
    return out


def public_tests(plan):
    # Private claim mappings, scope flags, old texts and answer letters cannot be serialized.
    return [{k: t.get(k, [] if k in {'required_views', 'prerequisites'} else '')
             for k in ('test_id', 'public_question', 'required_views', 'prerequisites')}
            for t in plan.get('tests', [])[:2]
            if isinstance(t, dict) and t.get('scope') == 'image_verifiable'
            and isinstance(t.get('public_question'), str)]


def register_evidence(observed, tests, case_id):
    ids = {t['test_id'] for t in tests}
    evidence = []
    for obs in observed.get('observations', [])[:2]:
        if not isinstance(obs, dict) or obs.get('test_id') not in ids:
            continue
        good_source = obs.get('source_id') == 'image_0'
        adequate = obs.get('assessability') == 'adequate' and good_source
        evidence.append({'evidence_id': f'{case_id}:E{len(evidence)+1}',
                         'source_id': 'image_0', 'test_id': obs['test_id'],
                         'assessability': 'adequate' if adequate else 'unavailable',
                         'observation': str(obs.get('observation', '')) if good_source else '',
                         'limitations': obs.get('limitations', []) if good_source else ['source mismatch']})
    return evidence


def rebuild(claims, bindings, evidence):
    usable = {e['evidence_id'] for e in evidence if e['assessability'] == 'adequate'}
    verdicts = {}
    for v in bindings.get('verdicts', []):
        if not isinstance(v, dict):
            continue
        cid = v.get('claim_id')
        refs = v.get('evidence_ids', [])
        status = v.get('status')
        if status not in {'SUPPORTED', 'CONTRADICTED', 'UNRESOLVED'}:
            status = 'UNRESOLVED'
        if not isinstance(refs, list) or not refs or not set(refs) <= usable:
            status, refs = 'UNRESOLVED', []
        verdicts[cid] = {'status': status, 'evidence_ids': refs,
                         'grounded_text': str(v.get('grounded_text', ''))}
    bad = {c['claim_id'] for c in claims if verdicts.get(c['claim_id'], {}).get('status') != 'SUPPORTED'}
    isolated = set(bad)
    while True:
        extra = {c['claim_id'] for c in claims if set(c['parents']) & isolated}
        if extra <= isolated:
            break
        isolated |= extra
    flat, retained = [], []
    for c in claims:
        v = verdicts.get(c['claim_id'], {})
        if c['kind'] in {'visual', 'context'} and v.get('status') == 'SUPPORTED' and v.get('grounded_text'):
            fact = {'text': v['grounded_text'], 'kind': c['kind'], 'evidence_ids': v['evidence_ids']}
            flat.append(fact)
            if c['claim_id'] not in isolated:
                retained.append(fact)
    return {'retained': retained, 'flat': flat, 'isolated_ids': sorted(isolated),
            'verdicts': verdicts,
            'deployment_trigger': any(v['status'] == 'CONTRADICTED' for v in verdicts.values())}


def raw_facts(claims):
    return [{'text': c['text'], 'kind': c['kind'], 'evidence_ids': []} for c in claims]


def random_facts(claims, retained, seed, token_length):
    facts = raw_facts(claims)
    count = min(len(retained), len(facts))
    if not count:
        return []
    target = token_length(json.dumps(retained))
    options = list(itertools.combinations(range(len(facts)), count))
    rng = random.Random(seed)
    rng.shuffle(options)
    # No verdict, error identity or reference is consulted. Match count, then token budget.
    best = min(options, key=lambda ix: abs(token_length(json.dumps([facts[i] for i in ix])) - target))
    return [facts[i] for i in best]


def summary_facts(result, evidence, maximum):
    ids = {e['evidence_id'] for e in evidence if e['assessability'] == 'adequate'}
    out = []
    for f in result.get('facts', [])[:maximum]:
        if not isinstance(f, dict):
            continue
        refs = f.get('evidence_ids', [])
        if isinstance(f.get('text'), str) and isinstance(refs, list) and refs and set(refs) <= ids:
            out.append({'text': f['text'], 'kind': 'visual', 'evidence_ids': refs})
    return out
