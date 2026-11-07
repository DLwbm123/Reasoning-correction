"""Frozen-generator mechanism experiment. This process never reads reference labels."""
import hashlib
import json
import os
from pathlib import Path
import random
import time
import traceback

from mechanism import (payload, claims_from, public_tests, register_evidence,
                       rebuild, raw_facts, random_facts, summary_facts)

ROOT = Path(os.environ['RUN_ROOT'])
CODE = Path(__file__).resolve().parent
BRANCHES = ['raw', 'selective', 'reset', 'random_drop', 'evidence_summary']
BUDGET = {'initial': 512, 'plan': 320, 'observe': 384, 'bind': 512, 'summarize': 256, 'repair': 384}


def write_json(path, obj):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2))
    tmp.replace(path)


def parse_json(text):
    start = text.find('{')
    if start < 0:
        raise ValueError('no JSON object')
    value, _ = json.JSONDecoder().raw_decode(text[start:])
    if not isinstance(value, dict):
        raise ValueError('not a JSON object')
    return value


def validate(role, result):
    field = {'initial': 'claims', 'plan': 'tests', 'observe': 'observations', 'bind': 'verdicts',
             'summarize': 'facts', 'repair': 'claims'}[role]
    if role == 'repair' and 'claims' not in result:
        # Answer accuracy and explanation schema compliance are distinct outcomes.
        # No missing explanation or its kind/evidence is fabricated.
        result['claims'] = []
        result['_schema_warning'] = 'missing claims list; explanation unavailable'
    if not isinstance(result.get(field), list):
        raise ValueError('missing list field ' + field)
    if role in {'initial', 'repair'} and result.get('answer') not in ['A', 'B', 'C', 'D']:
        raise ValueError('answer must be one letter A/B/C/D')


class Backend:
    def __init__(self):
        import torch
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        self.torch = torch
        self.prompts = json.loads((CODE / 'prompts_e0.json').read_text())
        self.calls_path = ROOT / 'outputs/calls.jsonl'
        torch.set_num_threads(4)
        free, total = torch.cuda.mem_get_info()
        if free < 19500 * 1024**2:
            raise RuntimeError(f'Insufficient GPU memory before loading: {free / 1024**2:.0f} MiB')
        torch.cuda.set_per_process_memory_fraction(.78)
        path = (ROOT / 'model_path.txt').read_text().strip()
        self.processor = AutoProcessor.from_pretrained(path, min_pixels=256 * 28 * 28,
                                                       max_pixels=768 * 28 * 28, local_files_only=True)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            path, torch_dtype=torch.bfloat16, device_map={'': 0},
            attn_implementation='sdpa', local_files_only=True).eval()
        self.model.requires_grad_(False)
        self.model.generation_config.temperature = None
        self.model.generation_config.top_p = None
        self.model.generation_config.top_k = None
        self.model.generation_config.do_sample = False
        self.call_count = 0
        print('MODEL_LOADED', flush=True)

    def token_length(self, value):
        return len(self.processor.tokenizer.encode(value, add_special_tokens=False))

    def generate(self, role, data, image, case_id, branch=None):
        import torch
        # This function owns each fresh message list and never returns caches.
        for attempt in range(2):
            system = self.prompts['system']
            if attempt:
                system += ' Strictly follow the requested JSON schema; output no other text.'
            content = []
            if image is not None:
                content.append({'type': 'image', 'image': image})
            content.append({'type': 'text', 'text': self.prompts[role] + '\nINPUT:\n' + json.dumps(data, ensure_ascii=False)})
            messages = [{'role': 'system', 'content': system}, {'role': 'user', 'content': content}]
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(text=[text], images=[image] if image is not None else None,
                                    padding=True, return_tensors='pt').to('cuda')
            n_input = inputs['input_ids'].shape[-1]
            visual_tokens = int((inputs['input_ids'] == self.model.config.image_token_id).sum().item())
            forwards = [0]
            handle = self.model.register_forward_pre_hook(lambda *a: forwards.__setitem__(0, forwards[0] + 1))
            start = time.time()
            with torch.inference_mode():
                ids = self.model.generate(**inputs, do_sample=False, num_beams=1,
                                          max_new_tokens=BUDGET[role], use_cache=True)
            torch.cuda.synchronize()
            elapsed = time.time() - start
            handle.remove()
            raw = self.processor.tokenizer.decode(ids[0, n_input:], skip_special_tokens=True)
            n_out = ids.shape[-1] - n_input
            self.call_count += 1
            record = {'case_id': case_id, 'role': role, 'branch': branch, 'attempt': attempt,
                      'payload': data, 'output_text': raw, 'input_tokens': n_input,
                      'output_tokens': n_out, 'visual_tokens': visual_tokens, 'forwards': forwards[0],
                      'seconds': elapsed, 'peak_memory_allocated': torch.cuda.max_memory_allocated(),
                      'peak_memory_reserved': torch.cuda.max_memory_reserved()}
            try:
                result = parse_json(raw)
                validate(role, result)
                record['parsed'] = True
                if result.get('_schema_warning'):
                    record['schema_warning'] = result['_schema_warning']
            except (ValueError, TypeError, KeyError) as exc:
                result = None
                record.update(parsed=False, parse_error=str(exc))
            with self.calls_path.open('a') as f:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
            del inputs, ids, messages, content
            print(json.dumps({k: record[k] for k in ['case_id', 'role', 'branch', 'attempt', 'parsed', 'seconds', 'output_tokens']}), flush=True)
            if result is not None:
                return result
        return {'_parse_failed': True}


def disk_bytes(root):
    # Count actual blocks, including model cache blobs; symlinks are not followed.
    total = 0
    for parent, _, files in os.walk(root):
        for name in files:
            try:
                total += os.lstat(os.path.join(parent, name)).st_blocks * 512
            except FileNotFoundError:
                pass
    return total


def run_case(backend, row, index):
    from PIL import Image
    # Label/reference fields cannot enter the inference DTO.
    expected = {'case_id', 'group_id', 'question', 'choices', 'image', 'image_sha256'}
    if set(row) != expected:
        raise ValueError('Unexpected manifest fields')
    case_id = row['case_id']
    image = Image.open(ROOT / 'inputs' / row['image']).convert('RGB')
    q, choices = row['question'], row['choices']
    initial = backend.generate('initial', payload('initial', question=q, choices=choices), image, case_id)
    claims = claims_from(initial)
    plan = backend.generate('plan', payload('plan', question=q, claims=claims, tools=[]), None, case_id)
    tests = public_tests(plan)
    if tests:
        observed = backend.generate('observe', payload('observe', public_tests=tests,
            image_manifest=[{'source_id': 'image_0', 'transform': 'full image, original orientation'}]), image, case_id)
    else:
        observed = {'observations': []}
    evidence = register_evidence(observed, tests, case_id)
    if claims:
        bindings = backend.generate('bind', payload('bind', question=q, claims=claims, evidence=evidence), None, case_id)
    else:
        bindings = {'verdicts': []}
    state = rebuild(claims, bindings, evidence)
    retained = state['retained']
    summary = {'facts': []}
    if retained:
        summary = backend.generate('summarize', payload('summarize', question=q, evidence=evidence,
                                                       max_facts=len(retained)), None, case_id)
    contexts = {'raw': raw_facts(claims), 'selective': retained, 'reset': [],
                'random_drop': random_facts(claims, retained, 20261107 + index, backend.token_length),
                'evidence_summary': summary_facts(summary, evidence, len(retained))}
    order = list(BRANCHES)
    random.Random(4711 + index).shuffle(order)
    answers, cache, aliases = {}, {}, {}
    for branch in order:
        data = payload('repair', question=q, choices=choices, evidence=evidence,
                       historical_facts=contexts[branch])
        key = json.dumps(data, sort_keys=True)
        # Identical deterministic inputs produce one shared candidate, not extra sampling.
        if key in cache:
            result, source = cache[key]
            aliases[branch] = source
        else:
            result = backend.generate('repair', data, image, case_id, branch)
            cache[key] = result, branch
        answers[branch] = result
    image.close()
    return {'case_id': case_id, 'group_id': row['group_id'], 'initial': initial,
            'claims': claims, 'plan': plan, 'public_tests': tests, 'observed': observed,
            'evidence': evidence, 'bindings': bindings, 'state': state, 'summary': summary,
            'contexts': contexts, 'answers': answers, 'branch_order': order,
            'deterministic_aliases': aliases,
            'context_tokens': {k: backend.token_length(json.dumps(v)) for k, v in contexts.items()}}


def main():
    started = json.loads((ROOT / 'started.json').read_text())
    deadline = started['deadline_epoch'] - 180
    manifest = [json.loads(x) for x in (ROOT / 'inputs/manifest.jsonl').read_text().splitlines()]
    backend = Backend()
    output = ROOT / 'outputs/cases'
    output.mkdir(exist_ok=True)
    start = time.time()
    status = 'completed'
    for index, row in enumerate(manifest):
        if time.time() > deadline:
            status = 'time_budget_stop'
            break
        if (output / (row['case_id'] + '.json')).exists():
            raise RuntimeError('Unexpected existing case: new experiments must not silently overwrite/re-run')
        if disk_bytes(ROOT.parent) >= 40_000_000_000:
            status = 'storage_budget_stop'
            break
        try:
            result = run_case(backend, row, index)
        except Exception as exc:
            result = {'case_id': row['case_id'], 'group_id': row['group_id'],
                      'pipeline_error': traceback.format_exc()}
            write_json(output / (row['case_id'] + '.json'), result)
            if 'out of memory' in str(exc).lower() or 'CUDA' in str(exc):
                raise
        write_json(output / (row['case_id'] + '.json'), result)
        write_json(ROOT / 'outputs/progress.json', {'status': 'running', 'completed_cases': index + 1,
            'planned_cases': len(manifest), 'calls': backend.call_count,
            'inference_seconds': time.time() - start, 'updated_epoch': time.time(),
            'last_case': row['case_id']})
        print('CASE_COMPLETE', row['case_id'], index + 1, flush=True)
    write_json(ROOT / 'outputs/collection_done.json', {'status': status, 'finished_epoch': time.time(),
               'completed_cases': len(list(output.glob('*.json'))), 'planned_cases': len(manifest),
               'calls': backend.call_count, 'inference_seconds': time.time() - start})


if __name__ == '__main__':
    main()
