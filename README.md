# SCAR: State-Carrier with Attentive Recall

Can a 1990s recurrence core + a tiny O(1) memory read path replace the looped
transformer's machinery? SCAR is a minimal architecture: a gated GRU state-carrier
writing into a **multi-timescale compressed memory** (k learned-decay slots,
O(1) update, never grows) read by a small attentive-recall head. No KV cache,
no looped decoder, fixed per-token cost.

Tested inside a **supervision-density-controlled harness** — the follow-up to
[rlt-research](https://github.com/arjunkshah12345-hash/rlt-research), where task
construction (dense vs sparse targets) was shown to reorder architecture rankings.

> **Methodology note (v2).** While preparing this revision we audited the harness
> and fixed three bugs that invalidated all v1 numbers: (1) the memory decay
> initialization collapsed every slot to λ≈0.5 (parameters were stored as
> log λ but consumed through a sigmoid — now stored in logit space, restoring
> half-lives ≈6.6→693 tokens); (2) training prepended a BOS token but evaluation
> did not (now one convention everywhere, enforced by a regression test);
> (3) the transformer baseline was misdescribed as sliding-window (it is full
> causal attention). v1 outputs are quarantined in `results_v1/` and
> `data/summary_v1.json`; nothing in v1 is mixed into v2 results.

## The hypotheses
1. **Architecture**: RLT's benefit comes from state-carriage + a cheap read path,
   not O(T) encoder memory or looping. The long-horizon ablation tests this.
2. **Efficiency**: the original ~1×-GRU cost target is an explicit hypothesis;
   the measured result is 3.12× GRU and 0.40× RLT, so that target is falsified
   while the relative reduction versus RLT survives.
3. **Method**: any architecture claim that doesn't survive a clean density
   sweep is a claim about the task generator, not the model.

## The sweep
9 models (elman, lstm, gru, transformer, token_merge, rlt, scar,
scar_carrier, scar_norecall) × 5 configs (parity/five × dense/sparse +
recall sparse) × 3 seeds = **135 matched runs** (~75–92K params each;
train 32 ops — recall 64; eval 16–256 ops — recall 64–512; 2048 programs/length).

The five release conditions use fixed training lengths. The earlier
`parity_sparse` + 4→8→16→32 curriculum runs are retained separately in
`results_curriculum/` and are not mixed into the density sweep.

- **scar_carrier**: memory removed, carrier widened to d=112 (param-matched).
- **scar_norecall**: recall head removed, carrier widened to d=98 (param-matched).
- **recall**: delayed first-token retrieval over a 10-symbol vocab (chance 12.5%).

## Reproduce
```
python3 -m pytest tests/ -v  # regression + release-contract tests
```

Training is cloud-only. Do not run the sweep on the local machine: the driver
refuses to train outside `/kaggle/working`. Push the appropriate Kaggle kernel
from `kaggle/*/` and use `kaggle/collect.sh` to pull the JSON artifacts. After
the cloud runs are complete, the local, non-training release build is:

```
python3 aggregate.py         # build data/summary.json (mean/std/min/max + per-seed)
python3 make_charts.py       # charts/
python3 build_site.py        # site/index.html (accuracy at trained length + extrapolation)
python3 make_paper.py        # paper/paper.tex (data-derived result numbers)
cd paper && tectonic paper.tex
```
The dedicated cost benchmark is also cloud-only; `data/cost_bench.json` is
pulled from the `cc` Kaggle kernel by `kaggle/collect.sh`.

Training runs on Kaggle CPU (`kaggle/`, one kernel per task-density config plus
a cost-benchmark kernel) — tiny sequential models are ~13x faster on CPU than
GPU. Code + results live here.

## Study 2 protocol (research/v3)

The follow-up study is preregistered in
[`EXPERIMENT_PROTOCOL_V3.md`](EXPERIMENT_PROTOCOL_V3.md). It keeps the v2
release immutable and adds cloud-only probes for delayed recall, train/test
length ratios, associative key/value recall, selective copy with capacity and
distractor-entropy conditions, memory-slot mechanisms, and frozen-memory
interventions. Every Study 2 driver clones the exact `research/v3` commit and
writes provenance-rich JSON under `study2_results/`; no optimizer step is
allowed on the local machine.

The primary Study 2 release gate is the seven-family, 172-run matrix: recall
length (35), ratio32 (18), ratio128 (18), associative recall (21), selective
copy (42), mechanism (33), and intervention (5). The supervision/curriculum
comparison and language-modeling extension described as exploratory follow-ups
in the protocol are deliberately outside this primary release: no language
model result is claimed, and the existing curriculum archive remains separate
from the fixed-density Study 1 tables.

After the Kaggle kernels finish, collect and validate them locally with:

```
bash kaggle/collect_study2.sh
python3 -m study2.analyze study2_results --out analysis/study2
```

Incomplete or failed kernel outputs must not be included in paper tables. The
paper is regenerated only after the complete artifact families pass validation.

When the complete Study 2 release is committed on `research/v3`, the
publication helper performs the final non-training checks, pushes the branch,
and queues the PR to `main`:

```
./scripts/publish_study2_pr.sh
```

Set `RUN_GITHUB_WORKER=1` if the safety worker should submit the queued PR in
the same invocation.

## Results
`results/` holds the v2 run JSONs; `data/summary.json` the aggregation;
`charts/` the figures; `site/` the generated result site; `paper/` the paper
(`paper.tex` generated by `make_paper.py`, compiled with tectonic).

## Project hygiene
`requirements.txt` records the Python environment, `.github/workflows/ci.yml`
runs the non-training regression/release tests, `LICENSE` grants MIT use, and
`CITATION.cff` provides the software citation metadata. GitHub repository
description, homepage, and topics should be set to the SCAR project page when
publishing the release.
