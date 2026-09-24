"""Validate Study 2 JSON artifacts without training.

Usage:
    python3 -m study2.validate study2_results/recall_length
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


REQUIRED = {
    "study", "protocol_version", "experiment_id", "git_commit", "model",
    "variant", "seed", "task", "task_parameters", "train_context",
    "eval_contexts", "eval_examples", "params", "optimizer", "lr",
    "weight_decay", "warmup_steps", "batch_size", "steps",
    "training_examples", "training_tokens", "supervision", "curriculum",
    "metrics", "raw_metrics", "train_seconds", "train_ms_per_step",
    "inference_ms_per_example", "env",
}
ENV_REQUIRED = {
    "torch", "numpy", "python", "platform", "device", "cpu", "gpu",
    "torch_threads", "git_commit", "timestamp_utc",
}


def validate_artifact(row: dict, path: Path | None = None, expected_study="study2") -> None:
    label = str(path) if path else "artifact"
    missing = sorted(REQUIRED - set(row))
    if missing:
        raise ValueError(f"{label}: missing fields: {', '.join(missing)}")
    if row["study"] != expected_study:
        raise ValueError(f"{label}: study={row['study']!r}, expected {expected_study!r}")
    if not str(row["protocol_version"]).startswith("v3"):
        raise ValueError(f"{label}: unexpected protocol version {row['protocol_version']!r}")
    if not row["experiment_id"].startswith("v3"):
        raise ValueError(f"{label}: experiment_id must start with v3")
    if not isinstance(row["seed"], int) or row["seed"] < 0:
        raise ValueError(f"{label}: invalid seed")
    if not isinstance(row["eval_contexts"], list) or not row["eval_contexts"]:
        raise ValueError(f"{label}: eval_contexts must be a non-empty list")
    if sorted(row["eval_contexts"]) != row["eval_contexts"]:
        raise ValueError(f"{label}: eval_contexts must be sorted")
    if set(map(str, row["eval_contexts"])) != set(row["metrics"]["accuracy_pct"]):
        raise ValueError(f"{label}: metric lengths do not match eval_contexts")
    if not isinstance(row["params"], int) or row["params"] <= 0:
        raise ValueError(f"{label}: invalid parameter count")
    if not isinstance(row["optimizer"], dict) or row["optimizer"].get("name") != "AdamW":
        raise ValueError(f"{label}: optimizer provenance is incomplete")
    env_missing = sorted(ENV_REQUIRED - set(row["env"]))
    if env_missing:
        raise ValueError(f"{label}: missing env fields: {', '.join(env_missing)}")
    if row["env"].get("git_commit") != row["git_commit"]:
        raise ValueError(f"{label}: top-level and env git commits differ")


def validate_directory(directory: str | Path, expected_count: int | None = None) -> list[Path]:
    root = Path(directory)
    paths = sorted(root.glob("*.json"))
    if expected_count is not None and len(paths) != expected_count:
        raise ValueError(f"{root}: found {len(paths)} JSON files, expected {expected_count}")
    seen = set()
    for path in paths:
        with path.open() as f:
            row = json.load(f)
        validate_artifact(row, path)
        key = (row["experiment_id"], row["model"], row["seed"])
        if key in seen:
            raise ValueError(f"{path}: duplicate artifact key {key}")
        seen.add(key)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory")
    parser.add_argument("--expected-count", type=int)
    args = parser.parse_args()
    paths = validate_directory(args.directory, args.expected_count)
    print(f"validated {len(paths)} Study 2 artifacts in {args.directory}")


if __name__ == "__main__":
    main()
