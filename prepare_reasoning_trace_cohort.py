"""One fixed training-source trace cohort; nominations are not error labels."""
import json
import os
from pathlib import Path
import random
import re
import time


def read_jsonl(path):
    with path.open() as f:
        return [json.loads(line) for line in f]


def main():
    root=Path(os.environ['TRACE_ROOT']);source=Path(os.environ['TRACE_SOURCE'])
    clock=json.loads((root/'started.json').read_text())
    def guard():
        if time.time()>clock['deadline_epoch']-60:raise RuntimeError('Original preparation deadline')
        used=sum(os.lstat(Path(p)/n).st_blocks*512 for p,_,files in os.walk(root) for n in files)
        if used>=20_000_000:raise RuntimeError('20MB conservative write guard')
    def write(path,value):
        guard()
        with path.open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2)
    splits=json.loads((source/'inputs/splits.private.json').read_text())
    manifest=read_jsonl(source/'inputs/manifest.jsonl')
    candidates=sorted((x for x in manifest if splits[x['group_id']]=='train'),key=lambda x:(x['group_id'],x['case_id']))
    if len({x['group_id'] for x in candidates})!=len(candidates):raise ValueError('Expected one article per source case')
    random.Random(2026110821).shuffle(candidates)
    if len(candidates)<100:raise ValueError('Insufficient frozen source; no replacement')
    selected=candidates[:100];out=root/'outputs';resume=os.environ.get('TRACE_RESUME')=='1'
    identities=[{'case_id':x['case_id'],'group_id':x['group_id']} for x in selected]
    if resume:
        if (root/'summary_after_recovery.json').exists():raise RuntimeError('One bounded CPU recovery only')
        if json.loads((out/'selected.private.json').read_text())!=identities:raise ValueError('Frozen identity/order changed')
        previous={x['case_id']:x for x in json.loads((out/'references.private.json').read_text())}
    else:
        out.mkdir(exist_ok=False);write(out/'selected.private.json',identities);previous={}
    # Correctness metadata are accessed only after cohort identities are written.
    selected_ids={r['case_id'] for r in selected}
    def selected_lines(path):
        # Existing source files are mixed; decode only the frozen training identities.
        with path.open() as f:
            for line in f:
                identity=re.search(r'"case_id"\s*:\s*"([^"\\]+)"',line)
                if identity is None:raise ValueError('Source line identity unavailable')
                if identity.group(1) in selected_ids:yield json.loads(line)
    labels={x['case_id']:x for x in selected_lines(source/'inputs/labels.private.jsonl')}
    supervision={}
    for row in selected_lines(source/'inputs/supervision.private.jsonl'):
        if row['case_id'] in labels:supervision.setdefault(row['case_id'],[]).append(row)
    packet=out/'blind'
    if not resume:packet.mkdir()
    references=[];nominated=[];failures=[];tagged_cases=0;tagged_items=0;aliases=0
    allowed={'case_id','group_id','initial','claims','plan','public_tests','observed','evidence','bindings','state','contexts','answers','deterministic_aliases','pipeline_error'}
    for row in selected:
        guard();cid=row['case_id']
        try:
            if resume and (packet/(cid+'.json')).exists():
                blind=json.loads((packet/(cid+'.json')).read_text())
                if blind['case_id']!=cid or blind['group_id']!=row['group_id']:raise ValueError('Existing packet identity mismatch')
                tagged=blind['model_tag_nominations'];tagged_cases+=bool(tagged);tagged_items+=len(tagged)
                aliases+=bool(blind['trace'].get('deterministic_aliases'))
                if tagged and len(nominated)<12:nominated.append(cid)
                references.append(previous[cid]);continue
            case=json.loads((source/'outputs/cases'/(cid+'.json')).read_text())
            if set(case)-allowed:raise ValueError('Unexpected trace fields; do not copy unchecked supervision')
            if case['case_id']!=cid or case['group_id']!=row['group_id']:raise ValueError('Trace identity mismatch')
            label=labels[cid];sup=supervision[cid]
            if label['group_id']!=row['group_id'] or len(sup)!=2 or any(x['split']!='train' or x['group_id']!=row['group_id'] for x in sup):raise ValueError('Reference/split identity mismatch')
            for s in sup:
                if case['initial']['answer']!=s['initial_answer'] or case['answers'][s['branch']]['answer']!=s['candidate_answer']:raise ValueError('Cached answer binding mismatch')
            containers=[('initial',case.get('claims',[]))]+[(b,a.get('claims',[])) for b,a in case.get('answers',{}).items()]
            entries=[(b,x) for b,claims in containers if isinstance(claims,list) for x in claims]
            tagged=[{'stage':b,'index':i,'model_kind':x.get('kind')} for i,(b,x) in enumerate(entries) if isinstance(x,dict) and x.get('kind') in {'reasoning','inference'}]
            tagged_cases+=bool(tagged);tagged_items+=len(tagged);aliases+=bool(case.get('deterministic_aliases'))
            blind={'case_id':cid,'group_id':row['group_id'],'question':row['question'],'choices':row['choices'],
                'existing_image_reference':str(source/'inputs'/row['image']),'trace':case,'model_tag_nominations':tagged,
                'review_status':'unreviewed','error_origin':'undetermined','medical_fact_adjudication':'not_available'}
            write(packet/(cid+'.json'),blind)
            if tagged and len(nominated)<12:nominated.append(cid)
            references.append({'case_id':cid,'group_id':row['group_id'],'closed_answer_reference':label,'cached_answer_outcomes':sup})
        except Exception as exc:
            failures.append({'case_id':cid,'error':type(exc).__name__+': '+str(exc)})
    suffix='_after_recovery' if resume else ''
    write(out/('references'+suffix+'.private.json'),references)
    write(out/('review_queue'+suffix+'.private.json'),{'case_ids':nominated,'selection':'first model-kind-tagged cases in frozen cohort order, never outcome-selected','all_unreviewed':True})
    write(out/('failures'+suffix+'.private.json'),failures)
    summary={'status':'completed' if not failures else 'completed_with_failures','selected_articles':100,'source_split':'G2 train',
        'blind_trace_packets':len(references),'failed_selected_cases':len(failures),'model_kind_tagged_cases':tagged_cases,
        'model_kind_tagged_items_including_related_branches':tagged_items,'cases_with_deterministic_aliases':aliases,
        'first_review_queue_cases':len(nominated),'new_error_adjudications':0,'new_expert_adjudications':0,'new_model_calls':0,
        'new_image_copies':0,'new_training':False,'seed':2026110821,'unseen_evaluation':False,
        'bounded_cpu_recovery':resume,'finished_epoch':time.time()}
    write(root/('summary'+suffix+'.json'),summary);print(json.dumps(summary))


if __name__=='__main__':main()
