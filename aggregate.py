"""Aggregate results/*.json into data/summary.json (mean/min/max across seeds)."""
import glob, json, os
from collections import defaultdict

rows = defaultdict(list)
for path in glob.glob("results/*.json"):
    with open(path) as f:
        r = json.load(f)
    key = (r["model"], r["task"], r["supervision"].split("_")[-1])
    rows[key].append(r)

summary = {}
for (model, task, density), rs in sorted(rows.items()):
    accs = [r["acc"] for r in rs]
    summary[f"{model}|{task}_{density}"] = {
        "n_seeds": len(rs),
        "params": rs[0]["params"],
        "train_ms_per_step": round(sum(r["train_ms_per_step"] for r in rs) / len(rs), 2),
        "acc_by_len": {l: {
            "mean": round(sum(a[l] for a in accs) / len(accs), 1),
            "min": round(min(a[l] for a in accs), 1),
            "max": round(max(a[l] for a in accs), 1),
        } for l in accs[0]},
    }

os.makedirs("data", exist_ok=True)
with open("data/summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"wrote data/summary.json ({len(summary)} model|config rows)")
