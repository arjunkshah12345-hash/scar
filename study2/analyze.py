"""Aggregate and plot Study 2 artifacts without training.

Usage:
    python3 -m study2.analyze study2_results --out analysis/study2
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .validate import validate_artifact


def bootstrap_ci(values, rng, draws=20_000):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return [None, None]
    if len(values) == 1:
        return [float(values[0]), float(values[0])]
    sample = rng.choice(values, size=(draws, len(values)), replace=True).mean(axis=1)
    return [float(np.percentile(sample, 2.5)), float(np.percentile(sample, 97.5))]


def summarize(values, seeds, rng):
    values = np.asarray(values, dtype=float)
    return {
        "n_seeds": int(len(values)),
        "seeds": [int(x) for x in seeds],
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "median": float(np.median(values)),
        "min": float(values.min()),
        "max": float(values.max()),
        "bootstrap95_mean": bootstrap_ci(values, rng),
    }


def load(root):
    rows = []
    for path in sorted(Path(root).rglob("v3*.json")):
        with path.open() as f:
            row = json.load(f)
        validate_artifact(row, path)
        row["_family"] = family_name(row, path)
        rows.append(row)
    return rows


def family_name(row, path):
    family = path.parent.name
    params = row.get("task_parameters", {})
    if row.get("task") == "selective_copy":
        return f"{family}_entropy{params['distractor_vocab']}"
    if family == "mechanism":
        if row["experiment_id"].startswith("v3E_slot"):
            return f"{family}_slots{params['scar_k']}"
        return f"{family}_decay_{params['scar_decay_mode']}"
    return family


def aggregate(rows, metric_key="accuracy_pct"):
    rng = np.random.default_rng(20260924)
    grouped = defaultdict(list)
    for row in rows:
        family = row["_family"]
        metrics = row["metrics"] if metric_key == "accuracy_pct" else row["raw_metrics"]
        for context, value in metrics[metric_key].items():
            grouped[(family, row["model"], int(context))].append(
                (row["seed"], float(value))
            )
    summary = {}
    for (family, model, context), pairs in sorted(grouped.items()):
        pairs.sort()
        summary.setdefault(family, {}).setdefault(model, {})[str(context)] = summarize(
            [value for _seed, value in pairs], [seed for seed, _value in pairs], rng
        )
    return summary


def aggregate_interventions(rows):
    """Aggregate frozen-memory intervention accuracies separately from baseline."""
    rng = np.random.default_rng(20260924)
    grouped = defaultdict(list)
    for row in rows:
        for intervention, metrics in row.get("interventions", {}).items():
            for context, value in metrics.items():
                grouped[(row["_family"], row["model"], intervention, int(context))].append(
                    (row["seed"], float(value))
                )
    summary = {}
    for (family, model, intervention, context), pairs in sorted(grouped.items()):
        pairs.sort()
        summary.setdefault(family, {}).setdefault(model, {}).setdefault(
            intervention, {}
        )[str(context)] = summarize(
            [value for _seed, value in pairs],
            [seed for seed, _value in pairs],
            rng,
        )
    return summary


def plot_family(family, models, out, suffix="accuracy", ylabel="Accuracy (%)"):
    fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    for model, points in sorted(models.items()):
        x = np.array([int(k) for k in points])
        x_order = np.argsort(x)
        x = x[x_order]
        y = np.array([points[str(v)]["mean"] for v in x])
        lo = np.array([points[str(v)]["bootstrap95_mean"][0] for v in x])
        hi = np.array([points[str(v)]["bootstrap95_mean"][1] for v in x])
        ax.plot(x, y, marker="o", linewidth=1.8, label=model)
        ax.fill_between(x, lo, hi, alpha=0.12)
    ax.set_xlabel("Evaluation units (operations or associative pairs)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Study 2: {family.replace('_', ' ')} ({suffix.replace('_', ' ')})")
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, fontsize=8, frameon=False)
    fig.savefig(out / f"{family}_{suffix}.png", dpi=180)
    plt.close(fig)


def plot_ratio_comparison(summary, out):
    """Plot Study 2B against eval/train ratio across both training contexts."""
    if "ratio32" not in summary or "ratio128" not in summary:
        return
    fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    styles = [("ratio32", 32, "-"), ("ratio128", 128, "--")]
    models = sorted(set(summary["ratio32"]) | set(summary["ratio128"]))
    for model in models:
        for family, train_ops, linestyle in styles:
            points = summary.get(family, {}).get(model, {})
            if not points:
                continue
            contexts = sorted(int(k) for k in points)
            x = np.asarray([context / train_ops for context in contexts])
            y = np.asarray([points[str(context)]["mean"] for context in contexts])
            lo = np.asarray([points[str(context)]["bootstrap95_mean"][0] for context in contexts])
            hi = np.asarray([points[str(context)]["bootstrap95_mean"][1] for context in contexts])
            label = f"{model} (train {train_ops})"
            ax.plot(x, y, marker="o", linewidth=1.6, linestyle=linestyle, label=label)
            ax.fill_between(x, lo, hi, alpha=0.08)
    ax.set_xlabel("Evaluation length / training length")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Study 2B: length extrapolation by train/test ratio")
    ax.set_xscale("log", base=2)
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, fontsize=7, frameon=False)
    fig.savefig(out / "study2_ratio_comparison.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", help="directory containing collected Study 2 JSON directories")
    parser.add_argument("--out", default="analysis/study2")
    args = parser.parse_args()
    rows = load(args.root)
    if not rows:
        raise SystemExit("no Study 2 artifacts found")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = aggregate(rows)
    with (out / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    for family, models in summary.items():
        plot_family(family, models, out)
    plot_ratio_comparison(summary, out)

    intervention_rows = [row for row in rows if row.get("interventions")]
    if intervention_rows:
        intervention_summary = aggregate_interventions(intervention_rows)
        with (out / "intervention_summary.json").open("w") as f:
            json.dump(intervention_summary, f, indent=2)

    exact_rows = [
        row for row in rows
        if "free_running_exact_sequence_pct" in row.get("raw_metrics", {})
    ]
    if exact_rows:
        exact_summary = aggregate(exact_rows, "free_running_exact_sequence_pct")
        with (out / "exact_sequence_summary.json").open("w") as f:
            json.dump(exact_summary, f, indent=2)
        for family, models in exact_summary.items():
            plot_family(
                family,
                models,
                out,
                suffix="exact_sequence",
                ylabel="Exact-sequence accuracy (%)",
            )
    print(f"analyzed {len(rows)} Study 2 artifacts across {len(summary)} families")


if __name__ == "__main__":
    main()
