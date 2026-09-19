"""Parallel driver: 7 models x task-density configs x 3 seeds.

Usage: python3 run_all.py [--configs parity_dense,five_sparse] [--workers 7]
Designed for Kaggle CPU sessions (4 cores): tiny sequential models run ~13x
faster on CPU than GPU -- kernel-launch overhead dominates on GPU.
"""
import argparse, itertools, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor

MODELS = ["elman", "lstm", "gru", "transformer", "token_merge", "rlt", "scar"]
CONFIGS = {
    "parity_dense": ("parity", "dense"),
    "parity_sparse": ("parity", "sparse"),
    "five_dense": ("five", "dense"),
    "five_sparse": ("five", "sparse"),
}
SEEDS = [0, 1, 2]
STEPS = 2500

ap = argparse.ArgumentParser()
ap.add_argument("--configs", default=",".join(CONFIGS))
ap.add_argument("--workers", type=int, default=7)
args = ap.parse_args()

selected = args.configs.split(",")
jobs = [(CONFIGS[c], m, s) for c in selected for m in MODELS for s in SEEDS]

def run(job):
    (task, density), model, seed = job
    tag = f"{model}_{task}_{density}_seed{seed}"
    out = os.path.join("results", f"{tag}.json")
    if os.path.exists(out):
        print(f"skip {out}", flush=True)
        return
    log = open(f"logs/{tag}.log", "w")
    t0 = time.time()
    subprocess.run(
        [sys.executable, "bench.py", "--model", model, "--task", task,
         "--density", density, "--seed", str(seed), "--steps", str(STEPS),
         "--device", "cpu", "--out", "results"],
        stdout=log, stderr=subprocess.STDOUT,
    )
    print(f"done {tag} in {time.time()-t0:.0f}s", flush=True)

os.makedirs("logs", exist_ok=True)
os.makedirs("results", exist_ok=True)
with ThreadPoolExecutor(max_workers=args.workers) as ex:
    ex.map(run, jobs)
print("ALL DONE", flush=True)
