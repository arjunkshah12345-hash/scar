"""Dedicated serial cost benchmark for all models.

Measures wall-clock cost in a controlled single-process setting (no parallel
sweeps competing for threads): per-step TRAIN cost at the training length and
per-program INFERENCE cost at an evaluation length, after warmup, with
mean/median/std/min/max over repeated iterations. Everything (threads, batch,
lengths) is recorded in the output JSON so cost claims are reproducible.

Usage: python3 benchmark_cost.py --out data/cost_bench.json [--threads 2]
"""
import argparse, json, math, os, platform, statistics, time
from datetime import datetime, timezone
import numpy as np
import torch
import torch.nn.functional as F

import bench
from bench import MODELS, add_bos, make_batch, gen_automaton, count_params

CONFIGS = {
    "elman": {"d": 192},
    "lstm": {"d": 96},
    "gru": {"d": 112},
    "transformer": {"d": 56, "L": 3, "h": 4, "ffn": 112},
    "token_merge": {"d": 56, "L": 3, "h": 4, "ffn": 112, "window": 8},
    "rlt": {"d": 40, "LE": 2, "LD": 2, "h": 4, "ffn": 80, "window": 8},
    "scar": {"d": 88, "k": 16, "r": 40},
    "scar_carrier": {"d": 112, "k": 16, "r": 40, "use_memory": False, "use_recall": False},
    "scar_norecall": {"d": 98, "k": 16, "r": 40, "use_memory": True, "use_recall": False},
}
VOCAB = 5          # parity-sized input for all models (architecture cost is vocab-insensitive)
TRAIN_OPS = 32     # training length (BOS + 32)
INFER_OPS = 128    # inference length (BOS + 128)
BATCH = 64
WARMUP = 20
ITERS = 100


def timeit(fn, warmup=WARMUP, iters=ITERS):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1e3)
    return {
        "mean_ms": statistics.mean(times),
        "median_ms": statistics.median(times),
        "std_ms": statistics.stdev(times) if len(times) > 1 else 0.0,
        "min_ms": min(times),
        "max_ms": max(times),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/cost_bench.json")
    ap.add_argument("--threads", type=int, default=2, help="torch CPU threads (match sweep config)")
    ap.add_argument("--iters", type=int, default=ITERS)
    ap.add_argument("--warmup", type=int, default=WARMUP)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)

    rng = np.random.default_rng(0)
    auto = gen_automaton(rng)
    BOS = bench.TASK_VOCAB["parity"] - 2

    results = {}
    for name in MODELS:
        model = MODELS[name](VOCAB, CONFIGS)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
        model.train()

        # -- train step (batch of programs, full loss.backward + opt.step)
        seq, ans, run = make_batch("parity", TRAIN_OPS, BATCH, rng, auto)
        x = add_bos(seq, BOS)
        def train_step():
            logits = model(x)
            loss = F.cross_entropy(logits[:, 1:].reshape(-1, VOCAB), run.reshape(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        train_t = timeit(train_step, warmup=args.warmup, iters=args.iters)

        # -- inference (eval protocol: BOS + seq, score final position)
        seq_i, ans_i, _ = make_batch("parity", INFER_OPS, BATCH, rng, auto)
        x_i = add_bos(seq_i, BOS)
        model.eval()
        def infer_step():
            with torch.no_grad():
                model(x_i)[:, -1].argmax(-1)
        infer_t = timeit(infer_step, warmup=args.warmup, iters=args.iters)

        results[name] = {
            "params": count_params(model),
            "train": dict(train_t, ops=TRAIN_OPS, batch=BATCH),
            "inference": dict(infer_t, ops=INFER_OPS, batch=BATCH),
        }
        print(f"{name:14s} train {train_t['median_ms']:8.1f} ms/step "
              f"(std {train_t['std_ms']:6.1f})  infer {infer_t['median_ms']:8.1f} ms/batch "
              f"(std {infer_t['std_ms']:6.1f})", flush=True)

    out = {
        "config": {"vocab": VOCAB, "train_ops": TRAIN_OPS, "infer_ops": INFER_OPS,
                    "batch": BATCH, "warmup": WARMUP, "iters": ITERS,
                    "torch_threads": args.threads},
        "env": {"torch": torch.__version__, "numpy": np.__version__,
                 "python": platform.python_version(), "platform": platform.platform(),
                 "timestamp_utc": datetime.now(timezone.utc).isoformat()},
        "models": results,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()