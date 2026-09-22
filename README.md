# SCAR: State-Carrier with Attentive Recall

Can a 1990s recurrence core + a tiny O(1) memory read path replace the looped
transformer's machinery? SCAR is a minimal architecture: a gated GRU state-carrier
writing into a **multi-timescale compressed memory** (k learned-decay slots,
O(1) update, never grows) read by a small attentive-recall head. No KV cache,
no looped decoder, fixed per-token cost.

Tested inside a **supervision-density-controlled harness** — the follow-up to
[rlt-research](https://github.com/arjunkshah12345-hash/rlt-research), where task
construction (dense vs sparse targets) was shown to reorder architecture rankings.

## The two hypotheses
1. **Architecture**: RLT's benefit comes from state-carriage + a cheap read path,
   not O(T) encoder memory or looping. SCAR should match RLT at ~1x GRU cost,
   not 4-8x.
2. **Method**: any architecture claim that doesn't survive a density sweep is a
   claim about the task generator, not the model.

## The sweep
7 models (elman, lstm, gru, transformer, token_merge, rlt, scar) x
4 configs (parity/five x dense/sparse supervision) x 3 seeds = **84 matched runs**
(~75-91K params each, train 32 ops, eval 16/32/64/128 ops, 2048 programs/length).

## Reproduce
```
python3 run_all.py          # 84 runs; designed for 4 parallel Kaggle CPU sessions
python3 aggregate.py        # build data/summary.json from results/
```
Single run: `python3 bench.py --model scar --task five --density sparse --seed 0`

Training runs on Kaggle CPU (`kaggle/`, one kernel per task-density config) —
tiny sequential models are ~13x faster on CPU than GPU. Code + results live here.

## The paper
The full scientific paper is in [`paper.txt`](paper.txt): abstract, method,
all result tables, harness findings (chance floor, density swap), the GPU
infrastructure footnote, limitations, and appendices. Headline: SCAR matches
or beats RLT in every regime at 0.64-0.87x its cost — flat 97.8% vs RLT's
67.0% at length 128 on parity-dense — but costs 3-5x a GRU, so the cost bet
is half-won.

## Results
`results/` holds all 84 run JSONs; `data/summary.json` the aggregation;
`charts/` the figures; `site/` the generated result site.
