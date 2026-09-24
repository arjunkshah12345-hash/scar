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
        row["_family"] = path.parent.name
        rows.append(row)
    return rows


def aggregate(rows):
    rng = np.random.default_rng(20260924)
    grouped = defaultdict(list)
    for row in rows:
        family = row["_family"]
        for context, value in row["metrics"]["accuracy_pct"].items():
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


def plot_family(family, models, out):
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
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"Study 2: {family.replace('_', ' ')}")
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, fontsize=8, frameon=False)
    fig.savefig(out / f"{family}_accuracy.png", dpi=180)
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
    print(f"analyzed {len(rows)} Study 2 artifacts across {len(summary)} families")


if __name__ == "__main__":
    main()
