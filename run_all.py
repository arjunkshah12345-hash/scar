"""Parallel driver: 7 models x 4 task-density configs x 3 seeds = 84 runs."""
import itertools, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor

MODELS = ["elman", "lstm", "gru", "transformer", "token_merge", "rlt", "scar"]
CONFIGS = [("parity", "dense"), ("parity", "sparse"), ("five", "dense"), ("five", "sparse")]
SEEDS = [0, 1, 2]
STEPS = 2500

jobs = list(itertools.product(CONFIGS, MODELS, SEEDS))

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
         "--out", "results"],
        stdout=log, stderr=subprocess.STDOUT,
    )
    print(f"done {tag} in {time.time()-t0:.0f}s", flush=True)

os.makedirs("logs", exist_ok=True)
os.makedirs("results", exist_ok=True)
with ThreadPoolExecutor(max_workers=7) as ex:
    ex.map(run, jobs)
print("ALL DONE", flush=True)
