"""Parallel driver: 9 models x task-density configs x 3 seeds.

Usage inside a Kaggle kernel: python3 run_all.py [--configs parity_dense,recall_sparse] [--workers 4]
Designed for Kaggle CPU sessions (4 cores): tiny sequential models run ~13x
faster on CPU than GPU -- kernel-launch overhead dominates on GPU.
Resume-safe: existing result files are skipped, so a kernel can be re-run.
"""
import argparse, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor

# Tuple fields: task, supervision density, length curriculum.
CONFIGS = {
    "parity_dense": ("parity", "dense", False),
    "parity_sparse": ("parity", "sparse", False),
    "parity_sparse_curriculum": ("parity", "sparse", True),
    "five_dense": ("five", "dense", False),
    "five_sparse": ("five", "sparse", False),
    "recall_sparse": ("recall", "sparse", False),
}
MODELS = ["elman", "lstm", "gru", "transformer", "token_merge", "rlt",
          "scar", "scar_carrier", "scar_norecall"]
RELEASE_CONFIGS = ("parity_dense", "parity_sparse", "five_dense", "five_sparse", "recall_sparse")
SEEDS = [0, 1, 2]
STEPS = 2500


def result_tag(config, model, seed):
    task, density, curriculum = CONFIGS[config]
    suffix = "_curriculum" if curriculum else ""
    return f"{model}_{task}_{density}{suffix}_seed{seed}"


def output_dir(config):
    return "results_curriculum" if CONFIGS[config][2] else "results"


def run(job):
    config, (task, density, curriculum), model, seed = job
    tag = result_tag(config, model, seed)
    out_dir = output_dir(config)
    out = os.path.join(out_dir, f"{tag}.json")
    if os.path.exists(out):
        print(f"skip {out}", flush=True)
        return
    log = open(f"logs/{tag}.log", "w")
    t0 = time.time()
    cmd = [sys.executable, "bench.py", "--model", model, "--task", task,
           "--density", density, "--seed", str(seed), "--steps", str(STEPS),
           "--device", "cpu", "--out", out_dir]
    if curriculum:
        cmd.append("--curriculum")
    rc = subprocess.run(
        cmd,
        stdout=log, stderr=subprocess.STDOUT,
    ).returncode
    status = "done" if rc == 0 else f"FAILED rc={rc}"
    print(f"{status} {tag} in {time.time()-t0:.0f}s", flush=True)

def main():
    if not os.path.isdir("/kaggle/working"):
        raise SystemExit(
            "Refusing local training. Run one of the kaggle/* drivers on Kaggle CPU."
        )
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default=",".join(RELEASE_CONFIGS))
    ap.add_argument("--workers", type=int, default=7)
    args = ap.parse_args()

    selected = [c.strip() for c in args.configs.split(",") if c.strip()]
    unknown = sorted(set(selected) - set(CONFIGS))
    if unknown:
        raise SystemExit(f"unknown config(s): {', '.join(unknown)}; choose from {', '.join(CONFIGS)}")
    os.makedirs("logs", exist_ok=True)
    for out_dir in {output_dir(config) for config in selected}:
        os.makedirs(out_dir, exist_ok=True)
    jobs = [(c, CONFIGS[c], m, s) for c in selected for m in MODELS for s in SEEDS]
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(run, jobs))
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
