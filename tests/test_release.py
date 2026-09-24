"""Release-contract checks for the checked-in SCAR artifact bundle."""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import run_all


RELEASE_CONFIGS = {
    "parity_dense": "parity_dense",
    "parity_sparse": "parity_sparse",
    "five_dense": "five_dense",
    "five_sparse": "five_sparse",
    "recall_sparse": "recall_sparse",
}


def test_sweep_has_exactly_five_release_configs():
    assert set(run_all.RELEASE_CONFIGS) == set(RELEASE_CONFIGS)
    assert set(run_all.CONFIGS) == set(RELEASE_CONFIGS) | {"parity_sparse_curriculum"}
    assert run_all.result_tag("parity_sparse", "scar", 0) == "scar_parity_sparse_seed0"
    assert run_all.result_tag("parity_sparse_curriculum", "scar", 0) == "scar_parity_sparse_curriculum_seed0"
    assert run_all.output_dir("parity_sparse") == "results"
    assert run_all.output_dir("parity_sparse_curriculum") == "results_curriculum"


def test_checked_in_results_are_complete_and_match_protocol():
    paths = glob.glob("results/*.json")
    assert len(paths) == 9 * 5 * 3
    seen = set()
    for path in paths:
        with open(path) as f:
            row = json.load(f)
        config = row["supervision"]
        assert config in RELEASE_CONFIGS
        key = (row["model"], config, row["seed"])
        assert key not in seen
        seen.add(key)
        assert row["steps"] == 2500
        assert row.get("curriculum") is False
    assert len(seen) == len(paths)


def test_aggregated_summary_covers_every_model_config_pair():
    with open("data/summary.json") as f:
        summary = json.load(f)
    assert len(summary) == 9 * 5
    assert all(row["n_seeds"] == 3 for row in summary.values())
    assert summary["scar|parity_sparse"]["curriculum"] is False
    assert summary["scar|recall_sparse"]["curriculum"] is False


def test_curriculum_has_a_dedicated_kaggle_driver():
    metadata = os.path.join(os.path.dirname(__file__), "..", "kaggle", "pc", "kernel-metadata.json")
    driver = os.path.join(os.path.dirname(__file__), "..", "kaggle", "pc", "scar_train.py")
    assert os.path.exists(metadata)
    assert os.path.exists(driver)
    source = open(driver).read()
    assert "parity_sparse_curriculum" in source
    assert "results_curriculum" in source
