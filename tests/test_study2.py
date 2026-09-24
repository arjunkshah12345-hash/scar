"""Static checks for the cloud-only Study 2 infrastructure."""
from pathlib import Path

from study2.validate import validate_artifact
from study2.analyze import bootstrap_ci
from study2.state_memory import persistent_state_bytes


ROOT = Path(__file__).resolve().parents[1]


def test_protocol_is_frozen_before_results():
    text = (ROOT / "EXPERIMENT_PROTOCOL_V3.md").read_text()
    assert "Status: frozen before Study 2 result collection" in text
    assert "Study 1: immutable corrected baseline" in text
    assert "Study 2A" in text and "Study 2E" in text
    assert "All optimizer steps and timing involving training run in Kaggle kernels." in text


def test_recall_length_kernel_is_cloud_only_and_provenance_aware():
    driver = (ROOT / "kaggle" / "recall-length" / "scar_train.py").read_text()
    assert "--branch" in driver and "research/v3" in driver
    assert "--protocol_version" in driver
    assert "--experiment_id" in driver
    assert "study2.validate" in driver
    assert "/kaggle/working" in driver


def test_associative_recall_kernel_is_separate_from_study1():
    metadata = ROOT / "kaggle" / "associative-recall" / "kernel-metadata.json"
    driver = ROOT / "kaggle" / "associative-recall" / "scar_train.py"
    assert metadata.exists() and driver.exists()
    source = driver.read_text()
    assert "--task" in source and '"assoc"' in source
    assert "v3C_assoc_train4" in source
    assert "study2_results" in source


def test_ratio_kernels_have_distinct_experiment_families():
    for name, train_ops in (("ratio32", "train32"), ("ratio128", "train128")):
        driver = (ROOT / "kaggle" / name / "scar_train.py").read_text()
        assert "study2.validate" in driver
        assert f"v3B_recall_{train_ops}" in driver


def test_mechanism_kernel_covers_slots_and_decay_modes():
    driver = (ROOT / "kaggle" / "mechanism" / "scar_train.py").read_text()
    assert "--scar_k" in driver and "--scar_decay_mode" in driver
    assert "slot_sweep" in driver and "decay_sweep" in driver
    assert "study2.validate" in driver


def test_intervention_kernel_requests_frozen_memory_perturbations():
    driver = (ROOT / "kaggle" / "intervention" / "scar_train.py").read_text()
    assert "--interventions" in driver
    for name in ("fastest", "slowest", "equalize", "shuffle", "noise"):
        assert name in driver


def test_study2_validator_accepts_complete_shape():
    row = {
        "study": "study2", "protocol_version": "v3.0",
        "experiment_id": "v3A_demo_gru_seed0", "git_commit": "abc",
        "model": "gru", "variant": "gru", "seed": 0, "task": "recall",
        "task_parameters": {}, "train_context": 64, "eval_contexts": [64],
        "eval_examples": 4, "params": 100, "optimizer": {"name": "AdamW"},
        "lr": 0.003, "weight_decay": 0.01, "warmup_steps": 200,
        "batch_size": 64, "steps": 10, "training_examples": 640,
        "training_tokens": 40960, "supervision": "recall_sparse",
        "curriculum": False, "metrics": {"accuracy_pct": {"64": 50.0}},
        "raw_metrics": {}, "train_seconds": 1.0, "train_ms_per_step": 1.0,
        "inference_ms_per_example": {"64": 1.0},
        "env": {
            "torch": "x", "numpy": "x", "python": "x", "platform": "x",
            "device": "cpu", "cpu": "x", "gpu": None, "torch_threads": 2,
            "git_commit": "abc", "timestamp_utc": "x",
        },
    }
    validate_artifact(row)


def test_bootstrap_summary_is_deterministic_and_bounded():
    import numpy as np

    first = bootstrap_ci([100.0, 50.0, 0.0], np.random.default_rng(4), draws=1000)
    second = bootstrap_ci([100.0, 50.0, 0.0], np.random.default_rng(4), draws=1000)
    assert first == second
    assert 0.0 <= first[0] <= first[1] <= 100.0


def test_persistent_state_scaling_matches_architecture_claim():
    assert persistent_state_bytes("scar", 64) == persistent_state_bytes("scar", 4096)
    assert persistent_state_bytes("gru", 64) == persistent_state_bytes("gru", 4096)
    assert persistent_state_bytes("transformer", 4096) > persistent_state_bytes("transformer", 64)
    assert persistent_state_bytes("rlt", 4096) > persistent_state_bytes("rlt", 64)
