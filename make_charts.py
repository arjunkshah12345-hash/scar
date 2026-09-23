"""Charts for the SCAR sweep, built from data/summary.json + data/cost_bench.json.

Every number shown is machine-derived from results/*.json via aggregate.py.
Run after the sweep: python3 aggregate.py && python3 make_charts.py && python3 build_site.py
Tolerant of partial sweeps: missing (model, config) rows are simply skipped.
"""
import json, os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODELS = ["elman", "lstm", "gru", "transformer", "token_merge", "rlt",
          "scar", "scar_carrier", "scar_norecall"]
PALETTE = ["#6b7280", "#9ca3af", "#374151", "#7c3aed", "#0891b2", "#dc2626",
           "#c2410c", "#f59e0b", "#fbbf24"]
COLORS = dict(zip(MODELS, PALETTE))
TASK_LENS = {"parity": [16, 32, 64, 128, 256],
             "five": [16, 32, 64, 128, 256],
             "recall": [64, 128, 256, 512]}
CHANCE = {"parity": 50, "five": 20, "recall": 12.5}


def acc(s, model, config, length, stat="mean"):
    e = s.get(f"{model}|{config}")
    if e is None or str(length) not in e["acc_by_len"]:
        return None
    return e["acc_by_len"][str(length)][stat]


def length_curves(s, configs, fname, suptitle):
    """Accuracy vs eval length per model; band = min-max across seeds."""
    fig, axes = plt.subplots(1, len(configs), figsize=(4.1 * len(configs), 3.8), squeeze=False)
    for j, cfg in enumerate(configs):
        ax = axes[0][j]
        task = cfg.split("_")[0]
        lens = TASK_LENS[task]
        for m in MODELS:
            means = [acc(s, m, cfg, l) for l in lens]
            if means[0] is None:
                continue
            mins = [acc(s, m, cfg, l, "min") for l in lens]
            maxs = [acc(s, m, cfg, l, "max") for l in lens]
            ax.plot(lens, means, marker="o", ms=3, lw=1.4, color=COLORS[m],
                    label=m if j == 0 else None)
            ax.fill_between(lens, mins, maxs, color=COLORS[m], alpha=0.12)
        ax.axhline(CHANCE[task], color="gray", lw=0.7, ls="--")
        ax.text(lens[0], CHANCE[task] + 1.5, "chance", fontsize=7, color="gray")
        ax.set_xscale("log", base=2)
        ax.set_xticks(lens)
        ax.set_xticklabels(lens, fontsize=7)
        ax.set_title(cfg, fontsize=9)
        ax.set_xlabel("eval length (ops)", fontsize=8)
        ax.set_ylabel("accuracy (%)", fontsize=8)
        ax.set_ylim(0, 103)
    fig.suptitle(suptitle, fontsize=10)
    fig.legend(loc="center right", fontsize=7, frameon=False)
    fig.tight_layout(rect=(0, 0, 0.88, 0.94))
    fig.savefig(fname, dpi=150)
    plt.close(fig)


def cost_chart(fname):
    """Cost from the DEDICATED serial benchmark (data/cost_bench.json), not
    from sweep timings. Median over repeated iterations; whiskers = min/max."""
    if not os.path.exists("data/cost_bench.json"):
        print("skip cost chart: data/cost_bench.json not found")
        return
    cb = json.load(open("data/cost_bench.json"))
    models = [m for m in MODELS if m in cb["models"]]
    base = cb["models"]["gru"]["train"]["median_ms"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, phase, title in ((axes[0], "train", "train ms/step (x GRU)"),
                              (axes[1], "inference", "inference ms/batch (x GRU)")):
        med = [cb["models"][m][phase]["median_ms"] / base for m in models]
        lo = [cb["models"][m][phase]["min_ms"] / base for m in models]
        hi = [cb["models"][m][phase]["max_ms"] / base for m in models]
        colors = [COLORS[m] for m in models]
        ax.bar(models, med, color=colors)
        ax.errorbar(models, med, yerr=[ [med[i]-lo[i] for i in range(len(med))],
                                        [hi[i]-med[i] for i in range(len(med))] ],
                    fmt="none", ecolor="#333", lw=0.8, capsize=2)
        ax.set_yscale("log")
        ax.set_title(title, fontsize=9)
        ax.tick_params(axis="x", rotation=40, labelsize=7)
        for i, v in enumerate(med):
            ax.text(i, v * 1.06, f"{v:.2f}x", ha="center", fontsize=6.5)
    fig.suptitle("Dedicated serial cost benchmark (median + min/max over "
                 f"{cb['config']['iters']} iters, {cb['config']['torch_threads']} threads)", fontsize=9)
    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    plt.close(fig)


def density_chart(s, ax, task, train_len):
    """Accuracy AT THE TRAINED LENGTH: dense vs sparse supervision per model."""
    configs = [f"{task}_dense", f"{task}_sparse"] if task != "recall" else [f"{task}_sparse"]
    width = 0.8 / len(configs)
    for i, cfg in enumerate(configs):
        vals = [acc(s, m, cfg, train_len) or 0.0 for m in MODELS]
        ax.bar([x + i * width - 0.4 + width / 2 for x in range(len(MODELS))],
               vals, width, label=cfg.split("_")[1])
    ax.axhline(CHANCE[task], color="gray", lw=0.6, ls="--")
    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels(MODELS, rotation=40, fontsize=7)
    ax.set_title(f"{task} @{train_len} ops (trained length): dense vs sparse", fontsize=9)
    ax.legend(fontsize=7)


def main():
    s = json.load(open("data/summary.json"))
    os.makedirs("charts", exist_ok=True)

    length_curves(s, ["parity_dense", "parity_sparse", "five_dense", "five_sparse"],
                  "charts/accuracy_by_config.png",
                  "Accuracy vs eval length (line = seed mean, band = min-max of 3 seeds)")
    length_curves(s, ["recall_sparse"], "charts/recall_curves.png",
                  "Delayed first-token recall: accuracy vs eval length")

    cost_chart("charts/cost.png")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    density_chart(s, axes[0], "parity", 32)
    density_chart(s, axes[1], "five", 32)
    fig.suptitle("Does the ranking survive the supervision-density swap?", fontsize=10)
    fig.tight_layout()
    fig.savefig("charts/density_swap.png", dpi=150)
    plt.close(fig)

    print("wrote charts/accuracy_by_config.png, charts/recall_curves.png, "
          "charts/cost.png, charts/density_swap.png")


if __name__ == "__main__":
    main()
