"""Neutral-entry fit stage; reads only old train/dev/cal resources."""
import os
from pathlib import Path
from action_gate import fit

if __name__ == "__main__":
    fit(Path(os.environ["RUN_ROOT"]))
