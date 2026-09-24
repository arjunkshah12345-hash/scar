# SCAR reproducibility guide

SCAR separates the immutable corrected Study 1 release from the preregistered
Study 2 follow-up. Every substantive optimizer step runs in Kaggle CPU
kernels; local commands below validate, aggregate, visualize, and compile only.

## Environment

Install the release-build dependencies with:

```sh
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install -r requirements.txt
```

The paper compiler is [Tectonic](https://tectonic-typesetting.github.io/).

## Study 1

The checked-in `results/` directory contains the 135 fixed-length release
artifacts. `results_curriculum/` contains the separate 27-run curriculum
condition, and `results_v1/` is quarantined historical output. Do not merge
these directories. Validate and regenerate the corrected release locally with:

```sh
python3 -m pytest tests/ -q
python3 aggregate.py
python3 make_charts.py
python3 make_paper.py
(cd paper && tectonic paper.tex)
```

The Study 1 training drivers live under `kaggle/` and refuse to train outside
`/kaggle/working`.

## Study 2

The frozen protocol is [`EXPERIMENT_PROTOCOL_V3.md`](EXPERIMENT_PROTOCOL_V3.md).
Push the relevant Kaggle kernels under `kaggle/`, then collect their compact
JSON artifacts without downloading checkpoints or logs:

```sh
bash kaggle/collect_study2.sh
python3 -m study2.analyze study2_results --out analysis/study2
python3 study2/state_memory.py --out analysis/study2/state_memory.json
```

The collector checks exact family counts and manifest commit provenance. The
analysis emits per-seed means, sample standard deviations, bootstrap intervals,
and free-running exact-sequence summaries where a task provides them.

## Publication gate

After all seven Study 2 families are committed on `research/v3`, run:

```sh
./scripts/publish_study2_pr.sh
```

The helper reruns validation, tests, analysis, paper generation, and Tectonic;
it refuses to publish if generated files differ or if any family is incomplete.
It pushes the branch and queues a PR to `main` through the repository’s GitHub
safety queue. Set `RUN_GITHUB_WORKER=1` to process that queue immediately.

No checkpoints, local training directories, temporary Kaggle logs, or browser
profiles are part of the release or arXiv source bundle.
