"""Kaggle driver (pc): dedicated curriculum parity sweep slice.

The curriculum flag is part of the experiment identity. The resulting
``*_parity_sparse_curriculum_seed*.json`` files are collected separately from
the fixed-length release sweep.
"""
import os
import shutil
import subprocess
import sys
from glob import glob

REPO = "https://github.com/arjunkshah12345-hash/scar.git"
WORK = "/kaggle/working/scar"

if not os.path.exists(WORK):
    subprocess.run(["git", "clone", REPO, WORK], check=True)
os.chdir(WORK)

subprocess.run([
    sys.executable,
    "run_all.py",
    "--configs",
    "parity_sparse_curriculum",
    "--workers",
    "4",
], check=True)

os.makedirs("/kaggle/working/results_curriculum", exist_ok=True)
for path in glob("results/*_parity_sparse_curriculum_seed*.json"):
    shutil.copy(path, "/kaggle/working/results_curriculum")
shutil.copytree("logs", "/kaggle/working/logs_curriculum", dirs_exist_ok=True)
print("curriculum sweep complete; results copied to /kaggle/working/results_curriculum")
