"""Freeze a development-only, article-disjoint PMC-VQA mechanism sample."""
import csv
import hashlib
import json
import random
import re
import zipfile
from pathlib import Path

SOURCE = Path('/Volumes/DataP/knowledge_editing/data/medmkeb/sources/PMC-VQA')
DEST = Path(__file__).parent / 'runs/e0_20261107_v1/inputs'
VISUAL = re.compile(r'\b(visible|shown|seen|image|imaging|scan|radiograph|radiographic|ct|mri|ultrasound|echogenic|opacity|opacities|density|intensity|located|location|left|right|bilateral|unilateral|lobe|organ|anatomical|anatomic|shape|size|number|how many|color|stain)\b', re.I)
EXCLUDE = re.compile(r'\b(treatment|therapy|prognosis|survival|risk factor|management|medication|dose|gene|mutation|according to the (text|article)|follow.up|previous|prior examination)\b', re.I)


def article(name):
    match = re.search(r'PMC\d+', name)
    return match.group() if match else None


def main():
    DEST.mkdir(parents=True, exist_ok=False)
    forbidden = set()
    for name in ('test.csv', 'test_clean.csv', 'test_2.csv'):
        with (SOURCE / name).open(newline='') as f:
            forbidden.update(article(r['Figure_path']) for r in csv.DictReader(f))
    candidates = []
    with (SOURCE / 'train.csv').open(newline='') as f:
        for row_index, row in enumerate(csv.DictReader(f)):
            q = row['Question'].strip()
            group = article(row['Figure_path'])
            if group and group not in forbidden and VISUAL.search(q) and not EXCLUDE.search(q):
                if row['Answer_label'].strip() in 'ABCD' and len(row['Answer_label'].strip()) == 1:
                    candidates.append((row_index, row))
    random.Random(20261107).shuffle(candidates)
    chosen = []
    groups = set()
    with zipfile.ZipFile(SOURCE / 'images.zip') as archive:
        for idx, row in candidates:
            group = article(row['Figure_path'])
            member = 'images/' + row['Figure_path']
            if group in groups or member not in archive.NameToInfo:
                continue
            case_id = f'c{len(chosen):04d}'
            image = archive.read(member)
            image_path = f'images/{case_id}.jpg'
            (DEST / 'images').mkdir(exist_ok=True)
            (DEST / image_path).write_bytes(image)
            infer = {'case_id': case_id, 'group_id': group, 'question': row['Question'].strip(),
                     'choices': {letter: row['Choice ' + letter].strip() for letter in 'ABCD'},
                     'image': image_path, 'image_sha256': hashlib.sha256(image).hexdigest()}
            label = {'case_id': case_id, 'group_id': group, 'reference_letter': row['Answer_label'].strip(),
                     'reference_text': row['Answer'], 'source_row': idx, 'source_image': row['Figure_path']}
            chosen.append((infer, label))
            groups.add(group)
            if len(chosen) == 200:
                break
    assert len(chosen) == 200, f'Only {len(chosen)} eligible articles available'
    for name, records in [('manifest.jsonl', [x[0] for x in chosen]), ('labels.private.jsonl', [x[1] for x in chosen])]:
        (DEST / name).write_text(''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in records))
    (DEST / 'selection.json').write_text(json.dumps({
        'seed': 20261107, 'n': 200, 'source': str(SOURCE / 'train.csv'),
        'excluded_article_count': len(forbidden), 'eligible_rows_before_article_sampling': len(candidates),
        'unique_articles': len(groups), 'test_article_overlap': len(groups & forbidden),
        'visual_question_regex': VISUAL.pattern, 'exclusion_regex': EXCLUDE.pattern,
        'selection_uses_answers': False, 'label_validity_check_only': True,
        'purpose': 'development mechanism pilot; never a final held-out benchmark',
    }, indent=2))
    print(json.loads((DEST / 'selection.json').read_text()))


if __name__ == '__main__':
    main()
