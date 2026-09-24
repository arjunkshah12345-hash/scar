"""Static checks for the cloud-only Study 2 infrastructure."""
from pathlib import Path

from study2.validate import validate_artifact
from study2.analyze import bootstrap_ci, family_name
from study2.state_memory import persistent_state_bytes
from study2.selective_copy import BOS, MARK, SLOT, make_batch


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


def test_study2_validator_excludes_provenance_manifest(tmp_path):
    import json

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
    (tmp_path / "v3A_demo_gru_seed0.json").write_text(json.dumps(row))
    (tmp_path / "manifest.json").write_text(json.dumps({"study": "study2", "git_commit": "abc"}))
    from study2.validate import validate_directory

    assert len(validate_directory(tmp_path, expected_count=1)) == 1

    (tmp_path / "manifest.json").write_text(json.dumps({"study": "study2", "git_commit": "different"}))
    try:
        validate_directory(tmp_path, expected_count=1)
    except ValueError as exc:
        assert "does not match manifest" in str(exc)
    else:
        raise AssertionError("manifest commit mismatch was accepted")


def test_bootstrap_summary_is_deterministic_and_bounded():
    import numpy as np

    first = bootstrap_ci([100.0, 50.0, 0.0], np.random.default_rng(4), draws=1000)
    second = bootstrap_ci([100.0, 50.0, 0.0], np.random.default_rng(4), draws=1000)
    assert first == second
    assert 0.0 <= first[0] <= first[1] <= 100.0


def test_analysis_keeps_mechanism_and_entropy_conditions_separate(tmp_path):
    selective = {
        "task": "selective_copy",
        "task_parameters": {"distractor_vocab": 2},
    }
    slot = {
        "task": "recall",
        "experiment_id": "v3E_slot16_seed0",
        "task_parameters": {"scar_k": 16, "scar_decay_mode": "learned_multi"},
    }
    decay = {
        "task": "recall",
        "experiment_id": "v3E_decay_learned_multi_seed0",
        "task_parameters": {"scar_k": 16, "scar_decay_mode": "learned_multi"},
    }
    assert family_name(selective, tmp_path / "selective_copy" / "x.json") == "selective_copy_entropy2"
    assert family_name(slot, tmp_path / "mechanism" / "x.json") == "mechanism_slots16"
    assert family_name(decay, tmp_path / "mechanism" / "x.json") == "mechanism_decay_learned_multi"


def test_persistent_state_scaling_matches_architecture_claim():
    assert persistent_state_bytes("scar", 64) == persistent_state_bytes("scar", 4096)
    assert persistent_state_bytes("gru", 64) == persistent_state_bytes("gru", 4096)
    assert persistent_state_bytes("transformer", 4096) > persistent_state_bytes("transformer", 64)
    assert persistent_state_bytes("rlt", 4096) > persistent_state_bytes("rlt", 64)


def test_selective_copy_batch_marks_ordered_targets_and_slots():
    import numpy as np

    seq, targets, slots = make_batch(
        total_items=12,
        marked_items=4,
        batch_size=5,
        rng=np.random.default_rng(7),
    )
    assert seq.shape == (5, 12 * 2 + 1 + 2 * 4)
    assert targets.shape == (5, 4)
    assert slots.tolist() == [26, 28, 30, 32]
    assert BOS != SLOT
    for row, expected in zip(seq, targets):
        marked_positions = (row[:24:2] == MARK).nonzero().flatten()
        assert len(marked_positions) == 4
        assert row[24] != MARK
        assert row[slots - 1].tolist() == [SLOT] * 4
        assert row[2 * marked_positions + 1].tolist() == expected.tolist()


def test_selective_copy_kernel_is_cloud_only_and_multi_output():
    metadata = ROOT / "kaggle" / "selective-copy" / "kernel-metadata.json"
    driver = ROOT / "kaggle" / "selective-copy" / "scar_train.py"
    assert metadata.exists() and driver.exists()
    source = driver.read_text()
    assert "v3D_copy_items64_entropy" in source
    assert "CONDITIONS = [(\"high\", 16), (\"low\", 2)]" in source
    assert "study2.selective_copy" in source
    module = (ROOT / "study2" / "selective_copy.py").read_text()
    assert "free_running_exact_sequence_pct" in module
    assert "Refusing local training" in module
