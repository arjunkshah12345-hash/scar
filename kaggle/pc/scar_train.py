"""Kaggle driver (pc): parity sparse with length curriculum (4->8->16->32)."""
import os, shutil, subprocess, sys

REPO = "https://github.com/arjunkshah12345-hash/scar.git"
WORK = "/kaggle/working/scar"

if not os.path.exists(WORK):
    subprocess.run(["git", "clone", REPO, WORK], check=True)
os.chdir(WORK)

configs = "parity_sparse_curriculum"
workers = "4"
subprocess.run([sys.executable, "run_all.py", "--configs", configs, "--workers", workers], check=True)

shutil.copytree("results", "/kaggle/working/results", dirs_exist_ok=True)
shutil.copytree("logs", "/kaggle/working/logs", dirs_exist_ok=True)
print("sweep complete; results copied to /kaggle/working/results")
