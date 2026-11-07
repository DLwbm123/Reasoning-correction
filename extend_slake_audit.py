"""Add explicitly separate original-SLAKE annotations after missing VGMED ROI findings."""
from collections import Counter
from pathlib import Path
import json
import random
import numpy as np
from PIL import Image,ImageOps,ImageDraw

ROOT=Path(__file__).resolve().parent
PRIVATE=ROOT/'private_analysis/data_audit_20261107_v1'
SOURCE=Path('/Volumes/DataP/knowledge_editing/data/m3bench/SLAKE')


def label_plane(arr):
    if arr.ndim==2:return arr
    if arr.ndim!=3 or arr.shape[2] not in [3,4]:return None
    if not (np.array_equal(arr[...,0],arr[...,1]) and np.array_equal(arr[...,0],arr[...,2])):return None
    if arr.shape[2]==4 and not np.all(arr[...,3]==255):return None
    return arr[...,0]


def main():
    mapfile=Path('/Volumes/DataP/Agent_memory_datasets/slake/mask.txt')
    labels={int(line.split(':',1)[0]):line.split(':',1)[1].strip() for line in mapfile.read_text().splitlines() if ':' in line}
    train=json.loads((SOURCE/'train.json').read_text())
    forbidden={x['img_name'] for split in ['validation','test'] for x in json.loads((SOURCE/(split+'.json')).read_text()) if x.get('q_lang')=='en'}
    images={}
    for row in train:
        if row.get('q_lang')=='en' and row['img_name'] not in forbidden:images.setdefault(row['img_name'],[]).append(row)
    eligible=[];stats=Counter()
    for rel,qa in images.items():
        image=SOURCE/'imgs'/rel;folder=image.parent;maskfile=folder/'mask.png'
        stats['english_train_images']+=1
        if not maskfile.is_file():stats['no_mask']+=1;continue
        with Image.open(maskfile) as mask:raw=np.asarray(mask);size=list(mask.size)
        arr=label_plane(raw)
        with Image.open(image) as im:imsize=list(im.size)
        if arr is None or size!=imsize:stats['unsupported_mask_or_size_mismatch']+=1;continue
        if raw.ndim==3:stats['opaque_repeated_channel_masks_decoded']+=1
        values=[int(v) for v in np.unique(arr) if int(v)!=0]
        known=[v for v in values if v in labels]
        if len(known)<2:stats['fewer_than_two_registered_labels']+=1;continue
        regions=[]
        for v in known:
            yy,xx=np.where(arr==v)
            regions.append({'mask_value':v,'source_label':labels[v],'pixel_count':int(len(xx)),
                            'bbox_xywh_pixels':[int(xx.min()),int(yy.min()),int(xx.max()-xx.min()+1),int(yy.max()-yy.min()+1)]})
        detfile=folder/'detection.json'
        detection=json.loads(detfile.read_text()) if detfile.is_file() else None
        eligible.append({'image_name':rel,'image_path':image,'modality':qa[0].get('modality'),'mask_file':str(maskfile),
                         'regions':regions,'unknown_nonzero_mask_values':[x for x in values if x not in labels],
                         'detection':detection,'questions':[{k:r[k] for k in ['question','answer','qid']} for r in qa[:3]],
                         'image_size':imsize})
    rng=random.Random(2026110706);rng.shuffle(eligible);chosen=[];used=set()
    for modality in ['CT','MRI','X-Ray']:
        choices=[x for x in eligible if x['modality']==modality][:2]
        chosen+=choices;used.update(x['image_name'] for x in choices)
    for item in eligible:
        if len(chosen)>=8:break
        if item['image_name'] not in used:chosen.append(item);used.add(item['image_name'])
    prior=json.loads((PRIVATE/'cases.json').read_text());assert len(prior)==8
    appended=[]
    for i,item in enumerate(chosen):
        audit_id=f'slake_mask_{i+1:02d}';preview=PRIVATE/'images'/f'{audit_id}.png'
        with Image.open(item['image_path']) as im:ImageOps.exif_transpose(im).convert('RGB').save(preview)
        record={k:v for k,v in item.items() if k!='image_path'}
        record.update(audit_id=audit_id,source='original SLAKE English train mask/detection; NOT VGMED ROI',
             image_preview=str(preview),shared_heldout_image=False,
             annotation_interpretation='registered segmentation labels and pixel geometry only; not independent expert adjudication',
             clinical_diagnosis_status='not established',patient_side_inferred=False,physical_measurements_inferred=False)
        appended.append(record)
    allcases=prior+appended;assert len(allcases)<=16
    (PRIVATE/'cases_with_slake_extension.json').write_text(json.dumps(allcases,ensure_ascii=False,indent=2))
    summary={**dict(stats),'eligible_multi_label_mask_images':len(eligible),
        'eligible_modality_counts':dict(Counter(x['modality'] for x in eligible)),
        'selected_original_slake_cases':len(appended),'selected_modality_counts':dict(Counter(x['modality'] for x in appended)),
        'total_private_audit_cases':len(allcases),'extension_seed':2026110706,'not_a_VGMED_restoration':True}
    (ROOT/'runs/data_audit_20261107_v1/slake_extension_checks.json').write_text(json.dumps(summary,indent=2))
    (PRIVATE/'extension_cases.json').write_text(json.dumps(appended,ensure_ascii=False,indent=2))
    contact=Image.new('RGB',(1600,1600),'white');draw=ImageDraw.Draw(contact)
    for i,c in enumerate(allcases):
        with Image.open(c['image_preview']) as im:
            im=im.convert('RGB');im.thumbnail((390,365));x=(i%4)*400+(400-im.width)//2;y=(i//4)*400+30
            contact.paste(im,(x,y));draw.text(((i%4)*400+5,(i//4)*400+5),c['audit_id'],fill='black')
    contact.save(PRIVATE/'contact_sheet.png')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
