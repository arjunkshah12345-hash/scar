"""Kaggle CPU driver for Study 2C: associative key/value recall."""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = "https://github.com/arjunkshah12345-hash/scar.git"
REF = "research/v3"
WORK = Path("/kaggle/working/scar")
OUT = WORK / "study2_results" / "associative_recall"
MODELS = ["gru", "lstm", "transformer", "rlt", "scar", "scar_carrier", "scar_norecall"]
SEEDS = range(3)
EVAL_PAIRS = "1,2,4,8,16,32"
EXPECTED = len(MODELS) * len(list(SEEDS))


def run(cmd, **kwargs):
    return subprocess.run(cmd, check=True, **kwargs)


if not WORK.exists():
    run(["git", "clone", "--depth", "1", "--branch", REF, "--single-branch", REPO, str(WORK)])
os.chdir(WORK)
commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
OUT.mkdir(parents=True, exist_ok=True)
log_dir = WORK / "study2_logs" / "associative_recall"
log_dir.mkdir(parents=True, exist_ok=True)

manifest = {
    "study": "study2",
    "protocol_version": "v3.0",
    "experiment_family": "v3C_associative_recall",
    "git_commit": commit,
    "models": MODELS,
    "seeds": list(SEEDS),
    "train_pairs": 4,
    "eval_pairs": [int(x) for x in EVAL_PAIRS.split(",")],
    "eval_examples": 4096,
    "expected_count": len(MODELS) * len(list(SEEDS)),
}
(OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))

for model in MODELS:
    for seed in SEEDS:
        experiment_id = f"v3C_assoc_train4_{model}_seed{seed}"
        artifact = OUT / f"{experiment_id}.json"
        if artifact.exists():
            print(f"skip {artifact}", flush=True)
            continue
        log_path = log_dir / f"{experiment_id}.log"
        cmd = [
            sys.executable, "bench.py",
            "--model", model, "--task", "assoc", "--density", "sparse",
            "--seed", str(seed), "--steps", "2500", "--train_ops", "4",
            "--eval_lengths", EVAL_PAIRS, "--eval_examples", "4096",
            "--eval_batch", "64", "--device", "cpu",
            "--study", "study2", "--protocol_version", "v3.0",
            "--experiment_id", experiment_id, "--out", str(OUT),
        ]
        print("run " + " ".join(cmd), flush=True)
        with log_path.open("w") as log:
            result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode != 0:
            raise SystemExit(f"failed: {experiment_id}; inspect {log_path}")

run([sys.executable, "-m", "study2.validate", str(OUT), "--expected-count", str(EXPECTED)])
shutil.copytree(OUT, "/kaggle/working/study2-results/associative_recall", dirs_exist_ok=True)
shutil.copytree(log_dir, "/kaggle/working/study2-logs/associative_recall", dirs_exist_ok=True)
print(f"Study 2C complete: {EXPECTED} artifacts at /kaggle/working/study2-results/associative_recall")
