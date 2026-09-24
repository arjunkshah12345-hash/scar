"""Aggregate results/*.json into data/summary.json.

Reports mean, std, min, max AND the individual per-seed accuracies at every
evaluated length, plus the chance level, so no statistic in the paper/site can
hide seed variation. Tolerant of partial sweeps: whatever runs exist are
aggregated, and n_seeds is reported per row.
"""
import glob, json, os, statistics
from collections import defaultdict

rows = defaultdict(list)
for path in sorted(glob.glob("results/*.json")):
    with open(path) as f:
        r = json.load(f)
    rows[(r["model"], r["supervision"])].append(r)

summary = {}
for (model, supervision), rs in sorted(rows.items()):
    lengths = sorted(int(l) for l in rs[0]["acc"])
    train_ops = rs[0]["train_ops"]
    curriculum_values = sorted({bool(r.get("curriculum", False)) for r in rs})
    curriculum = curriculum_values[0] if len(curriculum_values) == 1 else curriculum_values
    summary[f"{model}|{supervision}"] = {
        "n_seeds": len(rs),
        "params": sorted({r["params"] for r in rs}),
        "steps": rs[0]["steps"], "batch": rs[0]["batch"], "lr": rs[0].get("lr"),
        "train_ops": train_ops,
        "curriculum": curriculum,
        "chance_pct": rs[0].get("chance_pct"),
        "trained_length_in_eval": train_ops in lengths,
        "acc_by_len": {str(l): {
            "mean": round(statistics.mean(a["acc"][str(l)] for a in rs), 2),
            "std": round(statistics.stdev([a["acc"][str(l)] for a in rs]), 2) if len(rs) > 1 else 0.0,
            "min": round(min(a["acc"][str(l)] for a in rs), 2),
            "max": round(max(a["acc"][str(l)] for a in rs), 2),
            "seeds": [round(a["acc"][str(l)], 2) for a in rs],
        } for l in lengths},
        "train_ms_per_step": round(statistics.mean(r["train_ms_per_step"] for r in rs), 1),
        "train_seconds": round(statistics.mean(r["train_seconds"] for r in rs), 1),
    }

os.makedirs("data", exist_ok=True)
with open("data/summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"wrote data/summary.json ({len(summary)} model|config rows, "
      f"{sum(r['n_seeds'] for r in summary.values())} runs)")
