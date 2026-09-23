"""Kaggle driver (cc): dedicated serial cost benchmark only."""
import os, shutil, subprocess, sys

REPO = "https://github.com/arjunkshah12345-hash/scar.git"
WORK = "/kaggle/working/scar"

if not os.path.exists(WORK):
    subprocess.run(["git", "clone", REPO, WORK], check=True)
os.chdir(WORK)

subprocess.run([sys.executable, "benchmark_cost.py", "--out", "data/cost_bench.json"], check=True)
shutil.copy("data/cost_bench.json", "/kaggle/working/cost_bench.json")
print("cost benchmark complete")
