"""Offline post-collection label stage; never imported by inference."""
import json
import os
from pathlib import Path
from prepare_gate import prepare_pool


if __name__ == "__main__":
    root = Path(os.environ["RUN_ROOT"])
    done = json.loads((root/"outputs/collection_done.json").read_text())
    action = os.environ.get("ACTION_PROBE") == "1"
    if action and not (root/"outputs/policies_locked.json").exists():
        raise RuntimeError("Action policies must be locked before opening new labels")
    if done["status"] != "completed" or done["cases"] != (200 if action else 1600):
        raise RuntimeError("Do not train on an incomplete pool")
    split = json.loads((root/"inputs/splits.private.json").read_text())
    if (root/"inputs/features.jsonl").exists():
        raise RuntimeError("Prepared pool already exists")
    prepare_pool(root, root, split, copy_images=False, split_seed=2026110715 if action else 2026110708, retrospective=False)
