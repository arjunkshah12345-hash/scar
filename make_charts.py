"""Charts + report site for the SCAR sweep, built from data/summary.json.

Every number shown is machine-derived from results/*.json via aggregate.py.
Run after the sweep: python3 make_charts.py && python3 build_site.py
"""
import json, os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODELS = ["elman", "lstm", "gru", "transformer", "token_merge", "rlt", "scar"]
LEN_KEYS = ["16", "32", "64", "128"]


def load():
    with open("data/summary.json") as f:
        return json.load(f)


def acc(s, model, config, length):
    key = f"{model}|{config}"
    if key not in s:
        return 0.0
    return s[key]["acc_by_len"][str(length)]["mean"]


def bar_chart(s, ax, config, title):
    width = 0.13
    xs = range(len(MODELS))
    for i, L in enumerate(LEN_KEYS):
        vals = [acc(s, m, config, L) for m in MODELS]
        ax.bar([x + i * width for x in xs], vals, width, label=f"len {L}")
    ax.set_xticks([x + 1.5 * width for x in xs])
    ax.set_xticklabels(MODELS, rotation=20, fontsize=8)
    ax.axhline(50 if config.startswith("parity") else 20, color="gray", lw=0.6, ls="--")
    ax.set_title(title, fontsize=9)
    ax.set_ylim(0, 105)


def cost_chart(s, ax):
    base = s.get("gru|parity_dense", {}).get("train_ms_per_step")
    if base is None:  # fallback: cheapest present config as baseline
        base = min(v["train_ms_per_step"] for v in s.values())
    ms = [s.get(f"{m}|parity_dense", {}).get("train_ms_per_step") or base for m in MODELS]
    rel = [m / base for m in ms]
    ax.bar(MODELS, rel, color=["#888"] * 6 + ["#c2410c"])
    ax.axhline(1.0, color="gray", lw=0.6, ls="--")
    ax.set_title("Training cost per step (x GRU)", fontsize=10)
    for i, r in enumerate(rel):
        ax.text(i, r + 0.05, f"{r:.1f}x", ha="center", fontsize=7)


def density_chart(s, ax, task, chance):
    """Final-length accuracy: dense vs sparse supervision per model."""
    xs, ds, sp = [], [], []
    for i, m in enumerate(MODELS):
        xs.append(i)
        ds.append(acc(s, m, f"{task}_dense", 32))
        sp.append(acc(s, m, f"{task}_sparse", 32))
    w = 0.35
    ax.bar([i - w / 2 for i in range(len(MODELS))], ds, w, label="dense")
    ax.bar([i + w / 2 for i in range(len(MODELS))], sp, w, label="sparse")
    ax.axhline(chance, color="gray", lw=0.6, ls="--")
    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels(MODELS, rotation=30, fontsize=7)
    ax.set_title(f"{task} @ train length: dense vs sparse", fontsize=10)
    ax.legend(fontsize=7)


def main():
    s = json.load(open("data/summary.json"))
    os.makedirs("charts", exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    bar_chart(s, axes[0], "parity_dense", "parity, dense (running) targets")
    bar_chart(s, axes[1], "five_sparse", "five-state, sparse (final) targets")
    fig.suptitle("Accuracy by eval length (mean of 3 seeds, matched params)")
    fig.tight_layout()
    fig.savefig("charts/accuracy_by_config.png", dpi=150)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    cost_chart(s, ax)
    fig.tight_layout()
    fig.savefig("charts/cost.png", dpi=150)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    density_chart(s, axes[0], "parity", 50)
    density_chart(s, axes[1], "five", 20)
    fig.suptitle("Does the ranking survive the supervision-density swap?")
    fig.tight_layout()
    fig.savefig("charts/density_swap.png", dpi=150)

    print("wrote charts/accuracy_by_config.png, charts/cost.png, charts/density_swap.png")


if __name__ == "__main__":
    main()
