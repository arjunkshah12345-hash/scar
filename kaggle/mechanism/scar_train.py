"""Kaggle CPU driver for Study 2E: slot count and decay mechanisms."""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = "https://github.com/arjunkshah12345-hash/scar.git"
REF = "research/v3"
WORK = Path("/kaggle/working/scar")
MODELS = [("slot", str(k), f"v3E_slot{k}") for k in (1, 2, 4, 8, 16, 32, 64)]
MODELS += [("decay", mode, f"v3E_decay_{mode}") for mode in
           ("learned_multi", "learned_single", "fixed_multi", "fixed_single")]
SEEDS = range(3)
EVAL_LENGTHS = "64,512,2048"

if not WORK.exists():
    subprocess.run(["git", "clone", "--branch", REF, "--single-branch", REPO, str(WORK)], check=True)
os.chdir(WORK)
commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
root = WORK / "study2_results"
logs = WORK / "study2_logs" / "mechanism"
logs.mkdir(parents=True, exist_ok=True)
expected = 0

for family, value, prefix in MODELS:
    out = root / ("slot_sweep" if family == "slot" else "decay_sweep")
    out.mkdir(parents=True, exist_ok=True)
    for seed in SEEDS:
        expected += 1
        experiment_id = f"{prefix}_seed{seed}"
        artifact = out / f"{experiment_id}.json"
        if artifact.exists():
            continue
        k = value if family == "slot" else "16"
        mode = value if family == "decay" else "learned_multi"
        cmd = [sys.executable, "bench.py", "--model", "scar", "--task", "recall",
               "--density", "sparse", "--seed", str(seed), "--steps", "2500",
               "--train_ops", "64", "--eval_lengths", EVAL_LENGTHS,
               "--eval_examples", "2048", "--eval_batch", "256", "--device", "cpu",
               "--scar_k", k, "--scar_decay_mode", mode,
               "--study", "study2", "--protocol_version", "v3.0",
               "--experiment_id", experiment_id, "--out", str(out)]
        log_path = logs / f"{experiment_id}.log"
        with log_path.open("w") as log:
            result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise SystemExit(f"failed: {experiment_id}; inspect {log_path}")

for directory, count in ((root / "slot_sweep", 21), (root / "decay_sweep", 12)):
    subprocess.run([sys.executable, "-m", "study2.validate", str(directory), "--expected-count", str(count)], check=True)
    (directory / "manifest.json").write_text(json.dumps({
        "study": "study2", "protocol_version": "v3.0",
        "experiment_family": directory.name, "git_commit": commit,
        "eval_lengths": [int(x) for x in EVAL_LENGTHS.split(",")],
        "seeds": list(SEEDS),
    }, indent=2))
    shutil.copytree(directory, f"/kaggle/working/study2-results/{directory.name}", dirs_exist_ok=True)
shutil.copytree(logs, "/kaggle/working/study2-logs/mechanism", dirs_exist_ok=True)
print(f"Study 2E complete: {expected} artifacts")
