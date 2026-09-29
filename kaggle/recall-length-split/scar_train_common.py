"""Kaggle CPU driver shared by the split Study 2A recall kernels."""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = "https://github.com/arjunkshah12345-hash/scar.git"
REF = "research/v3"
WORK = Path("/kaggle/working/scar")
SEEDS = range(5)
EVAL_LENGTHS = "64,128,256,512,1024,2048,4096"
EXPECTED = len(list(SEEDS))


def run(cmd, **kwargs):
    return subprocess.run(cmd, check=True, **kwargs)


def run_model(model: str, selected_seeds=None, device: str = "cpu") -> None:
    seeds = list(SEEDS if selected_seeds is None else selected_seeds)
    if not WORK.exists():
        run(["git", "clone", "--depth", "1", "--branch", REF, "--single-branch", REPO, str(WORK)])
    os.chdir(WORK)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    out = WORK / "study2_results" / "recall_length"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    log_dir = WORK / "study2_logs" / "recall_length"
    if log_dir.exists():
        shutil.rmtree(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "study": "study2",
        "protocol_version": "v3.0",
        "experiment_family": "v3A_recall_length",
        "git_commit": commit,
        "models": [model],
        "seeds": seeds,
        "train_ops": 64,
        "eval_lengths": [int(x) for x in EVAL_LENGTHS.split(",")],
        "eval_examples": 4096,
        "device": device,
        "expected_count": len(seeds),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))

    for seed in seeds:
        experiment_id = f"v3A_recall_train64_{model}_seed{seed}"
        artifact = out / f"{experiment_id}.json"
        log_path = log_dir / f"{experiment_id}.log"
        cmd = [
            sys.executable, "bench.py",
            "--model", model, "--task", "recall", "--density", "sparse",
            "--seed", str(seed), "--steps", "2500", "--train_ops", "64",
            "--eval_lengths", EVAL_LENGTHS, "--eval_examples", "4096",
            # Full-attention Transformer evaluation remains memory-bound at the
            # 4,096-operation endpoint, while RLT's recurrent evaluator is
            # memory-light but compute-heavy.  Keep the Transformer batch
            # conservative and use a larger RLT batch to finish the same
            # 4,096 fresh examples within Kaggle's CPU session limit.
            "--eval_batch", (
                "32" if model == "rlt" else
                ("4" if device == "cuda" and model == "transformer" else
                 ("8" if model == "transformer" else "256"))
            ),
            "--device", device, "--study", "study2", "--protocol_version", "v3.0",
            "--experiment_id", experiment_id, "--out", str(out),
        ]
        print("run " + " ".join(cmd), flush=True)
        with log_path.open("w") as log:
            result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode != 0:
            raise SystemExit(f"failed: {experiment_id}; inspect {log_path}")
        if not artifact.exists():
            raise SystemExit(f"missing artifact: {artifact}")

    run([sys.executable, "-m", "study2.validate", str(out), "--expected-count", str(len(seeds))])
    shutil.copytree(out, "/kaggle/working/study2-results/recall_length", dirs_exist_ok=True)
    shutil.copytree(log_dir, "/kaggle/working/study2-logs/recall_length", dirs_exist_ok=True)
    print(f"Study 2A split complete: {EXPECTED} {model} artifacts", flush=True)
