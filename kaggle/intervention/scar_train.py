"""Kaggle CPU driver for frozen-SCAR memory interventions."""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = "https://github.com/arjunkshah12345-hash/scar.git"
REF = "research/v3"
WORK = Path("/kaggle/working/scar")
OUT = WORK / "study2_results" / "intervention"
SEEDS = range(5)
EVAL_LENGTHS = "64,512,2048,4096"

if not WORK.exists():
    subprocess.run(["git", "clone", "--branch", REF, "--single-branch", REPO, str(WORK)], check=True)
os.chdir(WORK)
commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
OUT.mkdir(parents=True, exist_ok=True)
logs = WORK / "study2_logs" / "intervention"
logs.mkdir(parents=True, exist_ok=True)
(OUT / "manifest.json").write_text(json.dumps({
    "study": "study2", "protocol_version": "v3.0",
    "experiment_family": "v3E_intervention", "git_commit": commit,
    "seeds": list(SEEDS), "eval_lengths": [int(x) for x in EVAL_LENGTHS.split(",")],
}, indent=2))

for seed in SEEDS:
    experiment_id = f"v3E_intervention_scar_seed{seed}"
    artifact = OUT / f"{experiment_id}.json"
    if artifact.exists():
        continue
    cmd = [sys.executable, "bench.py", "--model", "scar", "--task", "recall",
           "--density", "sparse", "--seed", str(seed), "--steps", "2500",
           "--train_ops", "64", "--eval_lengths", EVAL_LENGTHS,
           "--eval_examples", "4096", "--eval_batch", "256", "--device", "cpu",
           "--interventions", "fastest,slowest,equalize,shuffle,noise",
           "--study", "study2", "--protocol_version", "v3.0",
           "--experiment_id", experiment_id, "--out", str(OUT)]
    log_path = logs / f"{experiment_id}.log"
    with log_path.open("w") as log:
        result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise SystemExit(f"failed: {experiment_id}; inspect {log_path}")

subprocess.run([sys.executable, "-m", "study2.validate", str(OUT), "--expected-count", "5"], check=True)
shutil.copytree(OUT, "/kaggle/working/study2-results/intervention", dirs_exist_ok=True)
shutil.copytree(logs, "/kaggle/working/study2-logs/intervention", dirs_exist_ok=True)
