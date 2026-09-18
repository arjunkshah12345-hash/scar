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
python3 run_all.py          # 84 runs; designed for a Kaggle GPU session
python3 aggregate.py        # build data/summary.json from results/
```
Single run: `python3 bench.py --model scar --task five --density sparse --seed 0`

Training runs on Kaggle (see `kaggle/`); this repo holds code + results only.
