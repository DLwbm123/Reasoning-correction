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
    cpu_only = os.environ.get("CPU_ONLY") == "1"
    gpu = "" if cpu_only else subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory",
        "--format=csv,noheader"], text=True)
    own = [line for line in gpu.splitlines() if line.split(",")[0].strip() == str(child.pid)]
    if any(word in (ps+"\n"+"\n".join(own)).lower()
           for word in ["wangbomin", "reasoning-correction", "clear-med", "gate_baseline"]):
        raise RuntimeError("Non-neutral process display")
    if own or cpu_only:
        write("process_" + str(child.pid) + ".json", {"ps": ps, "own_gpu": own, "epoch": time.time()})
    return bool(own) or cpu_only


def main():
    global child
    config = json.loads((R/"pipeline.json").read_text()) if (R/"pipeline.json").exists() else {}
    env = dict(os.environ, RUN_ROOT=str(R), CUDA_VISIBLE_DEVICES="6", PYTHONHOME="/usr",
        PYTHONPATH=str(R/"code")+":/data/bmw/envs/s0/lib/python3.12/site-packages:/data/bmw/envs/v0/lib/python3.12/site-packages",
        HF_HOME=str(R/"cache"), TMPDIR=str(R/"cache"), TRANSFORMERS_OFFLINE="1", HF_HUB_OFFLINE="1",
        HF_HUB_DISABLE_TELEMETRY="1", TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="4",
        MKL_NUM_THREADS="4", PYTHONUNBUFFERED="1")
    env.update(config.get("environment", {}))
    # This launcher map and environment are frozen program configuration, never model output.
    for sig in [signal.SIGTERM, signal.SIGINT]:
        signal.signal(sig, handle)
    stages = list(config.get("stages", [("features", "r.py"), ("training", "t.py")]))
    for stage, entry in stages:
        if time.time() > DEADLINE-120:
            raise RuntimeError("Original experiment deadline")
        with (R/"outputs"/(stage+".log")).open("x") as log:
            stage_deadline = config.get("absolute_stage_deadlines", {}).get(stage, DEADLINE)
            child = subprocess.Popen([str(N/"p"), "-u", str(N/entry)], env=env,
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            write("supervisor.json", {"status": "running", "stage": stage, "supervisor_pid": os.getpid(),
                "child_pid": child.pid, "deadline_epoch": DEADLINE, "updated_epoch": time.time()})
            saved, disk_time = False, 0
            while child.poll() is None:
                if not saved:
                    saved = capture()
                if time.time() > min(DEADLINE-60, stage_deadline-30):
                    raise RuntimeError("Hard deadline guard")
                if time.time()-disk_time > 60:
                    used = allocated(R)+allocated(N)
                    threshold = int(config.get("storage_stop_bytes", 6000000000))
                    write("storage.json", {"allocated_bytes": used, "stop_at_bytes": threshold,
                          "hard_limit_bytes": 50000000000, "epoch": time.time()})
                    if used >= threshold:
                        raise RuntimeError("Conservative storage guard")
                    disk_time = time.time()
                time.sleep(2)
            if child.returncode:
                raise RuntimeError(stage + " exited " + str(child.returncode) + "; no automatic expensive rerun")
        if stage == config.get("admission_stage"):
            gate = json.loads((R/"outputs/preflight_summary.json").read_text())
            if not gate["passed"]:
                write("supervisor.json", {"status": "completed_preflight_failed", "finished_epoch": time.time(),
                    "exit_code": 0, "full_extension_run": False, "reason": "Frozen engineering gate not met"})
                return
            write("supervisor.json", {"status": "preflight_passed_waiting_extension", "supervisor_pid": os.getpid(),
                "deadline_epoch": DEADLINE, "updated_epoch": time.time()})
            ready = R/"extension.json"
            while not ready.exists():
                if time.time() > DEADLINE-7200:
                    raise RuntimeError("Extension not ready before reserved final-stage window")
                time.sleep(10)
            # Agent-written frozen configuration, never a model/tool-generated command.
            extension = json.loads(ready.read_text())
            for label, path in extension["stages"]:
                if Path(path).name != path or not (N/path).is_file():
                    raise RuntimeError("Invalid neutral extension entry")
                stages.append((label, path))
    write("supervisor.json", {"status": "completed", "finished_epoch": time.time(), "exit_code": 0})


if __name__ == "__main__":
    try:
        main()
    except Exception:
        stop()
        (R/"outputs/supervisor_failure.txt").write_text(traceback.format_exc())
        write("supervisor.json", {"status": "failed", "epoch": time.time()})
        raise
