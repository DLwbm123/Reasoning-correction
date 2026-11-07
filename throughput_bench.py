"""Bounded, reference-free hardware timing; never a medical accuracy experiment."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT = Path(os.environ['RUN_ROOT'])


def main():
    import torch
    from PIL import Image
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    started = json.loads((ROOT/'started.json').read_text())
    if time.time() >= started['deadline_epoch'] - 120:
        raise RuntimeError('setup consumed the timing budget')
    deadline = min(600, max(1,int(started['deadline_epoch']-time.time()-60)))
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('benchmark deadline')))
    signal.alarm(deadline)
    torch.set_num_threads(4)
    free,total=torch.cuda.mem_get_info()
    if free < 19500*1024**2:raise RuntimeError('not enough free GPU memory')
    torch.cuda.set_per_process_memory_fraction(min(.78,20*1024**3/total))
    p=(ROOT/'model_path.txt').read_text().strip()
    processor=AutoProcessor.from_pretrained(p,min_pixels=256*28*28,max_pixels=768*28*28,
                                           use_fast=False,local_files_only=True)
    t=time.perf_counter()
    model=Qwen2_5_VLForConditionalGeneration.from_pretrained(p,torch_dtype=torch.bfloat16,
            device_map={'':0},attn_implementation='sdpa',local_files_only=True).eval()
    model.requires_grad_(False)
    model.generation_config.do_sample=False
    model.generation_config.temperature=None
    model.generation_config.top_p=None
    model.generation_config.top_k=None
    torch.cuda.synchronize()
    load=time.perf_counter()-t
    prompts=json.loads((ROOT/'code/prompts_e0.json').read_text())
    rows=json.loads((ROOT/'inputs/replay.json').read_text())
    tensors=[]
    for row in rows:
        im=Image.open(ROOT/'inputs'/row['image']).convert('RGB')
        messages=[{'role':'system','content':prompts['system']},{'role':'user','content':[
            {'type':'image','image':im},{'type':'text','text':prompts[row['role']]+'\nINPUT:\n'+json.dumps(row['payload'],ensure_ascii=False)}]}]
        text=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        data=processor(text=[text],images=[im],padding=True,return_tensors='pt')
        tensors.append((row,data)); im.close()
    results=[]
    for repeat in [-1,0,1,2]:
        for index,(row,cpu) in enumerate(tensors):
            if repeat==-1 and index:continue
            inputs=cpu.to('cuda')
            torch.cuda.synchronize(); t=time.perf_counter()
            with torch.inference_mode():
                ids=model.generate(**inputs,do_sample=False,num_beams=1,use_cache=True,
                    min_new_tokens=128,max_new_tokens=128)
            torch.cuda.synchronize(); elapsed=time.perf_counter()-t
            item={'repeat':repeat,'request_index':index,'role':row['role'],
                  'input_tokens':inputs['input_ids'].shape[-1],
                  'visual_tokens':int((inputs['input_ids']==model.config.image_token_id).sum().item()),
                  'output_tokens':ids.shape[-1]-inputs['input_ids'].shape[-1],
                  'seconds':elapsed}
            if repeat>=0:results.append(item)
            print(json.dumps(item),flush=True)
            del inputs,ids
    data={'purpose':'fixed-token throughput only; no correctness labels or clinical performance',
          'torch':torch.__version__,'transformers':__import__('transformers').__version__,
          'gpu':torch.cuda.get_device_name(),'load_seconds':load,'measurements':results,
          'total_seconds':sum(x['seconds'] for x in results),
          'tokens_per_second':sum(x['output_tokens'] for x in results)/sum(x['seconds'] for x in results),
          'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
          'hardware_snapshot':subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.used,utilization.gpu,power.draw',
                                  '--format=csv,noheader','-i',os.environ['CUDA_VISIBLE_DEVICES']],text=True)}
    (ROOT/'outputs/timing.json').write_text(json.dumps(data,indent=2))
    signal.alarm(0)


if __name__=='__main__':main()
