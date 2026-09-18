"""Kaggle driver: clone the repo, run the full 84-run sweep, save results to output."""
import os, shutil, subprocess, sys

REPO = "https://github.com/arjunkshah12345-hash/scar.git"
WORK = "/kaggle/working/scar"

if not os.path.exists(WORK):
    subprocess.run(["git", "clone", REPO, WORK], check=True)
os.chdir(WORK)

# resume-safe: run_all.py skips result files that already exist
subprocess.run([sys.executable, "run_all.py"], check=True)

shutil.copytree("results", "/kaggle/working/results", dirs_exist_ok=True)
shutil.copytree("logs", "/kaggle/working/logs", dirs_exist_ok=True)
print("sweep complete; results copied to /kaggle/working/results")
