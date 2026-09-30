"""Kaggle CPU driver for Study 2D: selective copy/capacity/interference."""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


REPO = "https://github.com/arjunkshah12345-hash/scar.git"
REF = "research/v3"
WORK = Path("/kaggle/working/scar")
OUT = WORK / "study2_results" / "selective_copy"
MODELS = ["gru", "lstm", "transformer", "rlt", "scar", "scar_carrier", "scar_norecall"]
SEEDS = range(3)
EVAL_MARKED = "1,2,4,8,16,32"
CONDITIONS = [("high", 16), ("low", 2)]
EXPECTED = len(MODELS) * len(list(SEEDS)) * len(CONDITIONS)


def run(cmd, **kwargs):
    return subprocess.run(cmd, check=True, **kwargs)


if not WORK.exists():
    run(["git", "clone", "--depth", "1", "--branch", REF, "--single-branch", REPO, str(WORK)])
os.chdir(WORK)
commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
OUT.mkdir(parents=True, exist_ok=True)
log_dir = WORK / "study2_logs" / "selective_copy"
log_dir.mkdir(parents=True, exist_ok=True)

manifest = {
    "study": "study2",
    "protocol_version": "v3.0",
    "experiment_family": "v3D_selective_copy",
    "git_commit": commit,
    "models": MODELS,
    "seeds": list(SEEDS),
    "train_items": 16,
    "train_marked": 4,
    "eval_items": 64,
    "eval_marked": [int(x) for x in EVAL_MARKED.split(",")],
    "eval_examples": 4096,
    "distractor_conditions": {name: vocab for name, vocab in CONDITIONS},
    "expected_count": EXPECTED,
}
(OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))

for condition, distractor_vocab in CONDITIONS:
    for model in MODELS:
        for seed in SEEDS:
            experiment_id = f"v3D_copy_items64_entropy{distractor_vocab}_{model}_seed{seed}"
            artifact = OUT / f"{experiment_id}.json"
            if artifact.exists():
                print(f"skip {artifact}", flush=True)
                continue
            log_path = log_dir / f"{experiment_id}.log"
            cmd = [
                sys.executable,
                "-m",
                "study2.selective_copy",
                "--model",
                model,
                "--seed",
                str(seed),
                "--steps",
                "2500",
                "--train_items",
                "16",
                "--train_marked",
                "4",
                "--eval_items",
                "64",
                "--eval_marked",
                EVAL_MARKED,
                "--eval_examples",
                "4096",
                "--eval_batch",
                "64",
                "--distractor_vocab",
                str(distractor_vocab),
                "--device",
                "cpu",
                "--study",
                "study2",
                "--protocol_version",
                "v3.0",
                "--experiment_id",
                experiment_id,
                "--out",
                str(OUT),
            ]
            print("run " + " ".join(cmd), flush=True)
            with log_path.open("w") as log:
                result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode != 0:
                raise SystemExit(f"failed: {experiment_id}; inspect {log_path}")

run([sys.executable, "-m", "study2.validate", str(OUT), "--expected-count", str(EXPECTED)])
shutil.copytree(OUT, "/kaggle/working/study2-results/selective_copy", dirs_exist_ok=True)
shutil.copytree(log_dir, "/kaggle/working/study2-logs/selective_copy", dirs_exist_ok=True)
print(f"Study 2D complete: {EXPECTED} artifacts at /kaggle/working/study2-results/selective_copy")
