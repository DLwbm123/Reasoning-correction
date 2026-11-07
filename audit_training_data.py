"""Read-only local source audit; raw audit materials stay in private_analysis."""
from collections import Counter
from pathlib import Path
import hashlib
import io
import json
import os
import random
import re

from PIL import Image, ImageOps
import pyarrow.parquet as pq

DATA=Path(os.environ.get('DATA_AUDIT_ROOT','/Volumes/DataP'))
BENCH=DATA/'Medical_MLLM_Hallucination_Benchmarks/datasets'
ROOT=Path(__file__).resolve().parent
PRIVATE=ROOT/'private_analysis/data_audit_20261107_v1'
PUBLIC=ROOT/'runs/data_audit_20261107_v1'
RNG=random.Random(2026110705)
PIXEL_CACHE={}


def fingerprint(blob):
    # Needed to reproduce image grouping and exclude shared split images. Not a copy-integrity check.
    encoded=hashlib.sha256(blob).hexdigest()
    if encoded not in PIXEL_CACHE:
        with Image.open(io.BytesIO(blob)) as image:
            image=ImageOps.exif_transpose(image)
            pixels=image.tobytes()
            head=json.dumps([image.mode,list(image.size)]).encode()
            PIXEL_CACHE[encoded]=(hashlib.sha256(head+pixels).hexdigest(),image.size,image.mode)
    return PIXEL_CACHE[encoded]


def rows_for(files,columns):
    for path in sorted(files):
        for batch in pq.ParquetFile(path).iter_batches(batch_size=64,columns=columns):
            yield from batch.to_pylist()


def blob_from(row):
    v=row['image']
    return v.get('bytes') if isinstance(v,dict) else v


def preview(blob,audit_id):
    path=PRIVATE/'images'/f'{audit_id}.png'
    with Image.open(io.BytesIO(blob)) as im:
        ImageOps.exif_transpose(im).convert('RGB').save(path)
    return str(path)


def audit_clinhallu(source,folder,prefix):
    cache=DATA/'Agent_memory_datasets'/folder/'data'
    files=list(cache.glob('*.parquet'))
    heldout=[p for p in files if not p.name.startswith('train')]
    forbidden=set();heldout_rows=0
    for row in rows_for(heldout,['image']):
        blob=blob_from(row)
        if blob:forbidden.add(fingerprint(blob)[0])
        heldout_rows+=1
    train=[];first={};images={}
    for index,row in enumerate(rows_for([p for p in files if p.name.startswith('train')],['image','question','answer'])):
        blob=blob_from(row)
        key,size,mode=fingerprint(blob)
        first.setdefault(key,index)
        images.setdefault(key,blob)
        train.append({'question':row['question'].strip(),'answer':row['answer'].strip(),
                      'group':key,'image_name':f'{prefix}_{first[key]:05d}.jpg','size':size,'mode':mode})
    gold_path=BENCH/f'ClinHallu/hf/data/{source}-train/gold_cot.jsonl'
    gold=[json.loads(x) for x in gold_path.read_text().splitlines()]
    stats=Counter();eligible=[];mapped=[]
    for g in gold:
        idx=g.get('original_index')
        if not isinstance(idx,int) or idx<0 or idx>=len(train):stats['out_of_bounds_index']+=1;continue
        row=train[idx]
        checks={'question':g['question'].strip()==row['question'],
                'answer':g['answer'].strip()==row['answer'],
                'image_name':len(g['images'])==1 and Path(g['images'][0]).name==row['image_name']}
        for k,v in checks.items():stats[k+'_match']+=int(v)
        if not all(checks.values()):
            stats['not_fully_linked']+=1;continue
        stats['fully_linked']+=1;mapped.append(g)
        if row['group'] in forbidden:stats['excluded_shared_heldout_image']+=1;continue
        stats['eligible_train_reference_rows']+=1;eligible.append((g,row))
    chosen=[];used=set()
    for typ in ['yes_no','open_ended']:
        candidates=[x for x in eligible if x[0]['question_type']==typ];RNG.shuffle(candidates)
        count=0
        for g,row in candidates:
            if row['group'] in used:continue
            used.add(row['group']);audit_id=f'{prefix}_{typ}_{count+1}'
            vr=g['gold_cot'].get('visual_recognition',{}).get('text','')
            a=g['answer'].strip().casefold()
            item={'audit_id':audit_id,'source':source,'question_type':typ,'source_index':g['original_index'],
                  'image_preview':preview(images[row['group']],audit_id),'source_image_name':g['images'][0],
                  'question':g['question'],'answer':g['answer'],'gold_cot':g['gold_cot'],
                  'image_group':row['group'],'shared_heldout_image':False,
                  'answer_lexically_present_in_visual_stage':bool(a and re.search(r'(?<!\w)'+re.escape(a)+r'(?!\w)',vr.casefold())),
                  'visual_stage_written_sentence_count':len([s for s in re.split(r'[.!?]+',vr) if s.strip()]),
                  'semantic_nonanswer_status':'unreviewed','expert_fact_status':'not established'}
            chosen.append(item);count+=1
            if count==2:break
    train_groups=set(first)
    summary={'gold_train_rows':len(gold),'source_train_rows':len(train),'unique_train_image_groups':len(train_groups),
             'heldout_rows_images_only':heldout_rows,'unique_heldout_image_groups':len(forbidden),
             'shared_image_groups':len(train_groups&forbidden),'exact_decoded_pixel_identity_only':True,
             **dict(stats),'selected_private_cases':len(chosen),'selected_strata':dict(Counter(x['question_type'] for x in chosen)),
             'reference_type':'model-generated, format/answer-consistency filtered; not per-case expert oracle'}
    print(source,json.dumps(summary),flush=True)
    return summary,chosen


def audit_vgmed():
    root=BENCH/'VGMED/hf'
    slake=DATA/'knowledge_editing/data/m3bench/SLAKE'
    train=json.loads((slake/'train.json').read_text())
    allowed={r['img_name'] for r in train if r.get('q_lang')=='en'}
    forbidden={r['img_name'] for split in ['validation','test'] for r in json.loads((slake/(split+'.json')).read_text()) if r.get('q_lang')=='en'}
    allowed-=forbidden
    exists={};candidates=[];summaries={}
    def available(name):
        if not isinstance(name,str) or Path(name).is_absolute() or '..' in Path(name).parts:return False
        if name not in exists:exists[name]=(root/name).is_file()
        return exists[name]
    for kind in ['attribute','localization']:
        stats=Counter();prefixes=Counter()
        with (root/f'VGMED_{kind}.jsonl').open() as f:
            for index,line in enumerate(f):
                stats['rows']+=1
                match=re.search(r'"image"\s*:\s*"([^"\\]+)"',line)
                if not match:stats['unrecognized_image_field']+=1;continue
                name=match.group(1);parts=Path(name).parts
                prefixes[parts[1] if len(parts)>1 and parts[0]=='image_folder' else parts[0]]+=1
                stats['rows_with_available_image']+=int(available(name))
                if 'Slake1.0/imgs/' not in name:continue
                rel=name.split('Slake1.0/imgs/',1)[1]
                if rel not in allowed:
                    stats['slake_rows_outside_english_train']+=1;continue
                # Only training-linked records have question/label/box fields inspected.
                row=json.loads(line);stats['english_train_slake_rows']+=1
                has_box=isinstance(row.get('original_bboxes'),list) and bool(row['original_bboxes'])
                has_mask=available(row.get('mask'))
                stats['english_train_rows_with_box']+=int(has_box)
                stats['english_train_rows_with_mask']+=int(has_mask)
                if kind=='attribute' and available(name) and (has_box or has_mask):candidates.append((index,row,rel))
        summaries[kind]={**dict(stats),'source_prefix_rows':dict(prefixes)}
    RNG.shuffle(candidates);chosen=[];used=set()
    for index,row,rel in candidates:
        if rel in used:continue
        used.add(rel);audit_id=f'vgmed_{len(chosen)+1:02d}'
        blob=(root/row['image']).read_bytes()
        with Image.open(io.BytesIO(blob)) as im:size=list(im.size)
        entry={'audit_id':audit_id,'source':'VGMED/SLAKE-English-train','source_index':index,'source_image_name':row['image'],
               'image_preview':preview(blob,audit_id),'question':row['question'],'target_label':row.get('label'),
               'original_bboxes':row.get('original_bboxes'),'modality':row.get('modality'),
               'source_mask_path':str(root/row['mask']) if row.get('mask') else None,'image_size':size,
               'label_is_question_answer':'not established; ROI target is not automatically an abnormality answer',
               'bbox_format':'needs source/geometry review; local consumer branches on startswith(Slake)',
               'shared_heldout_image':False,'expert_fact_status':'not established'}
        if available(row.get('mask')):
            with Image.open(root/row['mask']) as mask:
                entry['mask_size']=list(mask.size);entry['mask_mode']=mask.mode
                entry['mask_unique_values_count']=len(set(mask.getdata()))
                entry['mask_size_matches_image']=list(mask.size)==size
        chosen.append(entry)
        if len(chosen)==8:break
    print('VGMED',json.dumps(summaries),flush=True)
    return {'tables':summaries,'english_train_allowed_images':len(allowed),'selected_private_cases':len(chosen),
            'note':'Path availability is not annotation truth; heldout questions/labels not inspected.'},chosen


def main():
    PRIVATE.mkdir(parents=True,exist_ok=False);(PRIVATE/'images').mkdir()
    summary={};cases=[]
    for source,folder,prefix in [('vqa-rad','vqa_rad','vqarad'),('pathvqa','pathvqa','pathvqa')]:
        s,c=audit_clinhallu(source,folder,prefix);summary[source]=s;cases+=c
    s,c=audit_vgmed();summary['vgmed']=s;cases+=c
    assert len(cases)<=16 and all(not c['shared_heldout_image'] for c in cases)
    (PRIVATE/'cases.json').write_text(json.dumps(cases,ensure_ascii=False,indent=2))
    (PRIVATE/'machine_checks.json').write_text(json.dumps(summary,indent=2))
    lines=['# Private training audit packet','',
           'Not expert adjudication. References are not automatically true. Never publish this packet.','']
    for case in cases:
        lines += ['## '+case['audit_id'],'',f"![preview]({case['image_preview']})",'',
                  '```json',json.dumps({k:v for k,v in case.items() if k!='image_preview'},ensure_ascii=False,indent=2),'```','']
    (PRIVATE/'PACKET.md').write_text('\n'.join(lines))
    PUBLIC.mkdir(exist_ok=True)
    (PUBLIC/'machine_checks.json').write_text(json.dumps(summary,indent=2))
    print('PACKET_READY',len(cases),'private cases',flush=True)


if __name__=='__main__':main()
