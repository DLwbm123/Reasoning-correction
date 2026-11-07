"""Two-stage neutral supervisor with an independent deadline and storage guard."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback

N = Path(os.environ["NEUTRAL_ROOT"])
R = Path((N / "root.txt").read_text().strip())
DEADLINE = json.loads((R / "started.json").read_text())["deadline_epoch"]
child = None


def write(name, value):
    p = R / "outputs" / name
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2)); tmp.replace(p)


def stop():
    if child is not None and child.poll() is None:
        os.killpg(child.pid, signal.SIGTERM)
        try:
            child.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL); child.wait()


def handle(sig, frame):
    stop()
    write("supervisor.json", {"status": "terminated", "signal": sig, "epoch": time.time()})
    raise SystemExit(128+sig)


def allocated(path):
    total = 0
    for parent, _, files in os.walk(path):
        for name in files:
            try:
                total += os.lstat(os.path.join(parent, name)).st_blocks*512
            except FileNotFoundError:
                pass
    return total


def capture():
    ids = ",".join(map(str, [child.pid, os.getpid(), os.getppid()]))
    ps = subprocess.check_output(["ps", "-p", ids, "-o", "pid,ppid,args="], text=True)
    gpu = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory",
                                   "--format=csv,noheader"], text=True)
    own = [line for line in gpu.splitlines() if line.split(",")[0].strip() == str(child.pid)]
    if any(word in (ps+"\n"+"\n".join(own)).lower()
           for word in ["wangbomin", "reasoning-correction", "clear-med", "gate_baseline"]):
        raise RuntimeError("Non-neutral process display")
    if own:
        write("process_" + str(child.pid) + ".json", {"ps": ps, "own_gpu": own, "epoch": time.time()})
    return bool(own)


def main():
    global child
    env = dict(os.environ, RUN_ROOT=str(R), CUDA_VISIBLE_DEVICES="6", PYTHONHOME="/usr",
        PYTHONPATH=str(R/"code")+":/data/bmw/envs/s0/lib/python3.12/site-packages:/data/bmw/envs/v0/lib/python3.12/site-packages",
        HF_HOME=str(R/"cache"), TMPDIR=str(R/"cache"), TRANSFORMERS_OFFLINE="1", HF_HUB_OFFLINE="1",
        HF_HUB_DISABLE_TELEMETRY="1", TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="4",
        MKL_NUM_THREADS="4", PYTHONUNBUFFERED="1")
    for sig in [signal.SIGTERM, signal.SIGINT]:
        signal.signal(sig, handle)
    for stage, entry in [("features", "r.py"), ("training", "t.py")]:
        if time.time() > DEADLINE-120:
            raise RuntimeError("Original experiment deadline")
        with (R/"outputs"/(stage+".log")).open("x") as log:
            child = subprocess.Popen([str(N/"p"), "-u", str(N/entry)], env=env,
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            write("supervisor.json", {"status": "running", "stage": stage, "supervisor_pid": os.getpid(),
                "child_pid": child.pid, "deadline_epoch": DEADLINE, "updated_epoch": time.time()})
            saved, disk_time = False, 0
            while child.poll() is None:
                if not saved:
                    saved = capture()
                if time.time() > DEADLINE-60:
                    raise RuntimeError("Hard deadline guard")
                if time.time()-disk_time > 60:
                    used = allocated(R)+allocated(N)
                    write("storage.json", {"allocated_bytes": used, "stop_at_bytes": 6000000000,
                          "hard_limit_bytes": 50000000000, "epoch": time.time()})
                    if used >= 6_000_000_000:
                        raise RuntimeError("Conservative storage guard")
                    disk_time = time.time()
                time.sleep(2)
            if child.returncode:
                raise RuntimeError(stage + " exited " + str(child.returncode) + "; no automatic expensive rerun")
    write("supervisor.json", {"status": "completed", "finished_epoch": time.time(), "exit_code": 0})


if __name__ == "__main__":
    try:
        main()
    except Exception:
        stop()
        (R/"outputs/supervisor_failure.txt").write_text(traceback.format_exc())
        write("supervisor.json", {"status": "failed", "epoch": time.time()})
        raise
