"""Controlled history/oracle headroom diagnosis, not a medical performance benchmark."""
import json
import os
from pathlib import Path
import random
import time
import run_e0 as base

CONDITIONS = [('none_noE','none',None,False),('none_E','none',None,True)] + [
    (f'{presentation}_{value}_{"E" if evidence else "noE"}',presentation,value,evidence)
    for presentation in ['quoted','assistant'] for value in ['supported','contradicted']
    for evidence in [False,True]]
SYSTEM = ('This is an offline image-modality research task. Answer from the current image and any '
          'registered evidence. Earlier statements may be useful or inaccurate. Return only one JSON '
          'object: {"modality":"CT"}. Allowed values are CT, MRI, XRAY. Do not output explanations.')


def request_text(row,presentation,value,evidence):
    history=row['premises'][value] if value else None
    observation=row['oracle_evidence'] if evidence else []
    final={'question':row['question'],'registered_evidence':observation,
           'earlier_quoted_statement':history if presentation=='quoted' else None,
           'answer_format':{'modality':'CT|MRI|XRAY'}}
    prefix=[]
    if presentation=='assistant':
        prefix=[{'role':'user','content':'Earlier question: '+row['question']},
                {'role':'assistant','content':history}]
    return prefix,json.dumps(final,ensure_ascii=False)


def ask(backend,row,image,condition):
    import torch
    name,presentation,value,evidence=condition
    prefix,text=request_text(row,presentation,value,evidence)
    for attempt in range(2):
        system=SYSTEM+(' Strictly return the JSON object and one allowed value.' if attempt else '')
        messages=[{'role':'system','content':system}]+prefix+[
            {'role':'user','content':[{'type':'image','image':image},{'type':'text','text':text}]}]
        rendered=backend.processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        inputs=backend.processor(text=[rendered],images=[image],padding=True,return_tensors='pt').to('cuda')
        n=inputs['input_ids'].shape[-1]
        forwards=[0]
        hook=backend.model.register_forward_pre_hook(lambda *a:forwards.__setitem__(0,forwards[0]+1))
        t=time.time()
        with torch.inference_mode():
            ids=backend.model.generate(**inputs,do_sample=False,num_beams=1,max_new_tokens=48,use_cache=True)
        torch.cuda.synchronize();elapsed=time.time()-t;hook.remove()
        raw=backend.processor.tokenizer.decode(ids[0,n:],skip_special_tokens=True)
        result=None
        try:
            result=base.parse_json(raw)
            answer=str(result.get('modality','')).upper().replace('-','').replace(' ','')
            if answer not in ['CT','MRI','XRAY']:raise ValueError('invalid modality')
            result['modality']=answer
        except (ValueError,TypeError):result=None
        record={'case_id':row['case_id'],'condition':name,'attempt':attempt,
                'historical_prefix':prefix,'current_user_text':text,'output_text':raw,
                'input_tokens':n,'visual_tokens':int((inputs['input_ids']==backend.model.config.image_token_id).sum().item()),
                'output_tokens':ids.shape[-1]-n,'forwards':forwards[0],'seconds':elapsed,'parsed':result is not None}
        with (base.ROOT/'outputs/calls.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        del inputs,ids,messages
        print(json.dumps({k:record[k] for k in ['case_id','condition','attempt','parsed','seconds']}),flush=True)
        if result is not None:return result
    return {'_parse_failed':True}


def main():
    from PIL import Image
    root=base.ROOT;deadline=json.loads((root/'started.json').read_text())['deadline_epoch']-120
    rows=[json.loads(x) for x in (root/'inputs/manifest.jsonl').read_text().splitlines()]
    backend=base.Backend();(root/'outputs/cases').mkdir(exist_ok=True);start=time.time()
    status='completed'
    for i,row in enumerate(rows):
        if time.time()>deadline or base.disk_bytes(root.parent)>=40_000_000_000:
            status='budget_stop';break
        image=Image.open(root/'inputs'/row['image']).convert('RGB')
        conditions=list(CONDITIONS);random.Random(2026110704+i).shuffle(conditions)
        result={'case_id':row['case_id'],'group_id':row['group_id'],'answers':{},'condition_order':[x[0] for x in conditions]}
        for condition in conditions:result['answers'][condition[0]]=ask(backend,row,image,condition)
        image.close();base.write_json(root/f'outputs/cases/{row["case_id"]}.json',result)
        base.write_json(root/'outputs/progress.json',{'status':'running','completed_cases':i+1,'planned_cases':len(rows),
             'inference_seconds':time.time()-start,'updated_epoch':time.time()})
    base.write_json(root/'outputs/collection_done.json',{'status':status,'completed_cases':len(list((root/'outputs/cases').glob('*.json'))),
         'planned_cases':len(rows),'inference_seconds':time.time()-start,'finished_epoch':time.time()})


if __name__=='__main__':main()
