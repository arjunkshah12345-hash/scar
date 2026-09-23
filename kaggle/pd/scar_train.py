"""Kaggle driver (pd): clone the repo, run the parity dense sweep slice, save results."""
import os, shutil, subprocess, sys

REPO = "https://github.com/arjunkshah12345-hash/scar.git"
WORK = "/kaggle/working/scar"

if not os.path.exists(WORK):
    subprocess.run(["git", "clone", REPO, WORK], check=True)
os.chdir(WORK)

configs = "parity_dense"
workers = "4"  # 4 vCPU on Kaggle CPU sessions
# resume-safe: run_all.py skips result files that already exist
subprocess.run([sys.executable, "run_all.py", "--configs", configs, "--workers", workers], check=True)

# save sweep results FIRST; the optional cost benchmark must never jeopardize them
shutil.copytree("results", "/kaggle/working/results", dirs_exist_ok=True)
shutil.copytree("logs", "/kaggle/working/logs", dirs_exist_ok=True)

DO_COST = False  # cost benchmark runs in its own kernel (scar-sweep-cc)
if DO_COST:
    # dedicated serial cost benchmark; non-fatal — results are already saved
    rc = subprocess.run([sys.executable, "benchmark_cost.py",
                         "--out", "data/cost_bench.json"]).returncode
    if rc == 0:
        shutil.copy("data/cost_bench.json", "/kaggle/working/cost_bench.json")
print("sweep complete; results copied to /kaggle/working/results")
