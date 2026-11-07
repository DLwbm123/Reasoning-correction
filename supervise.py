"""Neutral-entry supervisor: deadline, storage limit, process groups, offline evaluation."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

NEUTRAL = Path(os.environ.get('NEUTRAL_ROOT', '/tmp/r7x4'))
ROOT = (NEUTRAL/'root.txt').read_text().strip()
R = Path(ROOT)
ENV = dict(os.environ, RUN_ROOT=ROOT, CUDA_VISIBLE_DEVICES='6',
           HF_HOME=str(R/'cache'), TRANSFORMERS_OFFLINE='1', HF_HUB_OFFLINE='1',
           HF_HUB_DISABLE_TELEMETRY='1', TOKENIZERS_PARALLELISM='false',
           OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', PYTHONUNBUFFERED='1',
           PYTHONHOME=os.environ.get('PYTHONHOME','/home/jiangsuiyang/anaconda3/envs/m3d'),
           PYTHONPATH=str(R/'code')+os.pathsep+os.environ.get('PYTHONPATH',''), TMPDIR=str(R/'cache'))
DEADLINE = json.loads((R/'started.json').read_text())['deadline_epoch']
child = None


def stop_child():
    if child is not None and child.poll() is None:
        os.killpg(child.pid, signal.SIGTERM)
        try:
            child.wait(timeout=45)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()


def handler(sig, frame):
    stop_child()
    sys.exit(128 + sig)


for s in [signal.SIGTERM, signal.SIGINT]:
    signal.signal(s, handler)


def used_bytes():
    total = 0
    # Include the preserved first interface-failure attempt and its shared model cache.
    for parent, _, files in os.walk(R.parent):
        for name in files:
            try:
                total += os.lstat(os.path.join(parent, name)).st_blocks * 512
            except FileNotFoundError:
                pass
    # Small neutral interpreter and launcher files outside the experiment root also count.
    for p in NEUTRAL.glob('*'):
        if p.is_file() and not p.is_symlink():
            total += p.stat().st_blocks * 512
    return total


def main():
    global child
    (R/'outputs/supervisor.json').write_text(json.dumps({'status':'waiting_for_weights','pid':os.getpid(),'deadline_epoch':DEADLINE}))
    while not (R/'environment.json').exists():
        if (R/'setup_failure.txt').exists():
            raise RuntimeError('Model setup failed; see setup_failure.txt')
        if time.time() > DEADLINE - 600:
            raise RuntimeError('Deadline reached before inference')
        time.sleep(15)
    with (R/'outputs/inference.log').open('a') as log:
        child = subprocess.Popen([str(NEUTRAL/'p'),'-u',str(NEUTRAL/'r.py')],env=ENV,
                                 stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        (R/'outputs/inference_pid.txt').write_text(str(child.pid))
        (R/'outputs/supervisor.json').write_text(json.dumps({'status':'inference_running','pid':os.getpid(),'child_pid':child.pid,'deadline_epoch':DEADLINE}))
        last_disk_check = 0
        while child.poll() is None:
            if time.time() > DEADLINE - 120:
                stop_child()
                raise RuntimeError('Hard experiment deadline')
            if time.time() - last_disk_check > 60:
                used = used_bytes()
                last_disk_check = time.time()
                (R/'outputs/storage.json').write_text(json.dumps({'allocated_bytes':used,'stop_at_bytes':40000000000,'hard_limit_bytes':50000000000,'epoch':time.time()}))
                if used >= 40_000_000_000:
                    stop_child()
                    raise RuntimeError('Storage stop threshold reached')
            time.sleep(5)
        code = child.returncode
    if code:
        raise RuntimeError(f'Inference process exit {code}; no automatic costly rerun')
    with (R/'outputs/evaluation.log').open('w') as log:
        child = subprocess.Popen([str(NEUTRAL/'p'),'-u',str(NEUTRAL/'e.py')],env=ENV,
                                 stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            code = child.wait(timeout=min(120,max(1,DEADLINE-time.time()-30)))
        except subprocess.TimeoutExpired:
            stop_child(); raise
    if code:
        raise RuntimeError(f'Evaluation process exit {code}')
    (R/'outputs/supervisor.json').write_text(json.dumps({'status':'completed','finished_epoch':time.time(),'exit_code':0}))


if __name__ == '__main__':
    try:
        main()
    except Exception:
        stop_child()
        (R/'outputs/supervisor_failure.txt').write_text(traceback.format_exc())
        (R/'outputs/supervisor.json').write_text(json.dumps({'status':'failed','finished_epoch':time.time()}))
        raise
