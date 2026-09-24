"""Kaggle CPU driver for Study 2B, training context 128."""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = "https://github.com/arjunkshah12345-hash/scar.git"
REF = "research/v3"
WORK = Path("/kaggle/working/scar")
OUT = WORK / "study2_results" / "ratio128"
MODELS = ["gru", "transformer", "rlt", "scar", "scar_carrier", "scar_norecall"]
SEEDS = range(3)
EVAL_LENGTHS = "128,256,512,1024,2048"
EXPECTED = len(MODELS) * len(list(SEEDS))

if not WORK.exists():
    subprocess.run(["git", "clone", "--depth", "1", "--branch", REF, "--single-branch", REPO, str(WORK)], check=True)
os.chdir(WORK)
commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
OUT.mkdir(parents=True, exist_ok=True)
logs = WORK / "study2_logs" / "ratio128"
logs.mkdir(parents=True, exist_ok=True)
(OUT / "manifest.json").write_text(json.dumps({
    "study": "study2", "protocol_version": "v3.0",
    "experiment_family": "v3B_recall_train128", "git_commit": commit,
    "models": MODELS, "seeds": list(SEEDS), "train_ops": 128,
    "eval_lengths": [int(x) for x in EVAL_LENGTHS.split(",")],
}, indent=2))

for model in MODELS:
    for seed in SEEDS:
        experiment_id = f"v3B_recall_train128_{model}_seed{seed}"
        artifact = OUT / f"{experiment_id}.json"
        if artifact.exists():
            continue
        log_path = logs / f"{experiment_id}.log"
        cmd = [sys.executable, "bench.py", "--model", model, "--task", "recall",
               "--density", "sparse", "--seed", str(seed), "--steps", "2500",
               "--train_ops", "128", "--eval_lengths", EVAL_LENGTHS,
               "--eval_examples", "2048", "--eval_batch", "1" if model in {"transformer", "rlt"} else "256",
               "--device", "cpu", "--study", "study2", "--protocol_version", "v3.0",
               "--experiment_id", experiment_id, "--out", str(OUT)]
        with log_path.open("w") as log:
            result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise SystemExit(f"failed: {experiment_id}; inspect {log_path}")

subprocess.run([sys.executable, "-m", "study2.validate", str(OUT), "--expected-count", str(EXPECTED)], check=True)
shutil.copytree(OUT, "/kaggle/working/study2-results/ratio128", dirs_exist_ok=True)
shutil.copytree(logs, "/kaggle/working/study2-logs/ratio128", dirs_exist_ok=True)
