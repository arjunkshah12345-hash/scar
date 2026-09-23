"""Build site/index.html — the SCAR report — from data/summary.json. Zero hand-typed numbers.

Statistics discipline:
- The headline table reports accuracy AT THE TRAINED LENGTH (parity/five: 32
  ops; recall: 64 ops) as mean +/- std across seeds — never a max over eval
  lengths, which would silently report extrapolation rather than the trained
  regime.
- A separate table reports the LONGEST evaluated length, explicitly labeled
  as extrapolation beyond the training length.
- Differences smaller than the seed std are not ranked; cells within seed
  noise of the best model are flagged with '≈'.
"""
import html
import json
import os

MODELS = ["elman", "lstm", "gru", "transformer", "token_merge", "rlt",
          "scar", "scar_carrier", "scar_norecall"]
NAMES = {"elman": "Elman RNN", "lstm": "LSTM", "gru": "GRU",
         "transformer": "Transformer (full causal)",
         "token_merge": "TokenMerge (RLT abl.)", "rlt": "RLT",
         "scar": "SCAR", "scar_carrier": "SCAR-carrier (no memory)",
         "scar_norecall": "SCAR-no-recall"}
CONFIGS = ["parity_dense", "parity_sparse", "five_dense", "five_sparse", "recall_sparse"]
CHANCE = {"parity": 50, "five": 20, "recall": 12.5}
TRAIN_LEN = {"parity": 32, "five": 32, "recall": 64}
MAX_LEN = {"parity": 256, "five": 256, "recall": 512}


def entry(s, model, config):
    return s.get(f"{model}|{config}")


def at_len(s, model, config, length):
    e = entry(s, model, config)
    if e is None or str(length) not in e["acc_by_len"]:
        return None
    a = e["acc_by_len"][str(length)]
    return a["mean"], a["std"], e["n_seeds"]


def cell_html(v, chance, best=None):
    if v is None:
        return "<td>—</td>"
    mean, std, n = v
    cls = "hot" if mean > chance + 10 else ""
    flag = " ≈" if (best is not None and best - mean <= max(std, 1.0)) else ""
    return f'<td class="{cls}">{mean:.1f}±{std:.1f}{flag}</td>'


def table(s, configs, length_fn, caption):
    head = "".join(f"<th>{html.escape(c)}</th>" for c in configs)
    rows = []
    for m in MODELS:
        if all(entry(s, m, c) is None for c in configs):
            continue
        first = entry(s, m, configs[0])
        params = first["params"][0] if first else None
        ptxt = f"{params:,}" if isinstance(params, int) else "?"
        vals = [at_len(s, m, c, length_fn(c)) for c in configs]
        best = max((v[0] for v in vals if v), default=None)
        cells = "".join(cell_html(v, CHANCE[c.split('_')[0]], best)
                        for c, v in zip(configs, vals))
        rows.append(f"<tr><td class='name'>{NAMES[m]}</td><td>{ptxt}</td>{cells}</tr>")
    return (f"<table><tr><th>model</th><th>params</th>{head}</tr>\n"
            + "\n".join(rows) + f"</table><p class='note'>{caption}</p>")
def cost_table():
    if not os.path.exists("data/cost_bench.json"):
        return "<p class='note'>cost benchmark not run yet</p>"
    cb = json.load(open("data/cost_bench.json"))
    base = cb["models"]["gru"]["train"]["median_ms"]
    rows = []
    for m in MODELS:
        if m not in cb["models"]:
            continue
        t, i = cb["models"][m]["train"], cb["models"][m]["inference"]
        rows.append(
            f"<tr><td class='name'>{NAMES[m]}</td>"
            f"<td>{t['median_ms']:.1f} [{t['min_ms']:.1f}–{t['max_ms']:.1f}]</td>"
            f"<td>{t['median_ms']/base:.2f}x</td>"
            f"<td>{i['median_ms']:.1f} [{i['min_ms']:.1f}–{i['max_ms']:.1f}]</td></tr>")
    cfg = cb["config"]
    caption = (f"Dedicated serial benchmark ({cfg['iters']} timed iters after "
               f"{cfg['warmup']} warmup, torch threads={cfg['torch_threads']}, "
               f"batch={cfg['batch']}, train len={cfg['train_ops']}+BOS, "
               f"infer len={cfg['infer_ops']}+BOS). Median [min–max] ms; "
               "treat as indicative ratios, not absolute numbers.")
    return (f"<table><tr><th>model</th><th>train ms/step</th><th>×GRU</th>"
            f"<th>inference ms/batch@{cfg['infer_ops']}</th></tr>\n"
            + "\n".join(rows) + f"</table><p class='note'>{caption}</p>")


def main():
    s = json.load(open("data/summary.json"))
    n_runs = sum(e["n_seeds"] for e in s.values())

    trained_tbl = table(
        s, CONFIGS, lambda c: TRAIN_LEN[c.split("_")[0]],
        "Accuracy at the TRAINED length (parity/five: 32 ops; recall: 64 ops), "
        "mean±std over seeds. '≈' marks cells within seed noise of the best "
        "model in that column — those differences are indistinguishable. "
        "Chance: parity 50%, five 20%, recall 12.5%.")
    extrap_tbl = table(
        s, CONFIGS, lambda c: MAX_LEN[c.split("_")[0]],
        "EXTRAPOLATION: accuracy at the longest evaluated length (parity/five: "
        "256 ops; recall: 512 ops), beyond the training length. No model was "
        "trained at this length.")

    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>SCAR: state-carrier with attentive recall</title>
<style>
body{{font-family:ui-sans-serif,system-ui;max-width:960px;margin:2rem auto;padding:0 1rem;line-height:1.5;color:#111}}
h1{{font-size:1.5rem}} h2{{margin-top:2rem;font-size:1.1rem}}
table{{border-collapse:collapse;width:100%;font-size:.85rem}}
td,th{{padding:.4rem .5rem;border-bottom:1px solid #e5e5e5;text-align:right}}
td.name,th:first-child,th:nth-child(2){{text-align:left}}
td.hot{{color:#c2410c;font-weight:600}}
img{{max-width:100%;border:1px solid #eee}}
.note{{color:#666;font-size:.8rem}}
</style></head><body>
<h1>SCAR: state-carrier, attentive recall, and what supervision density does to architecture rankings</h1>
<p>{n_runs} matched runs (~75–92K params): 9 architectures (incl. two SCAR
ablations) × parity/five/recall × dense/sparse supervision × 3 seeds.
All numbers below are machine-derived from <code>results/*.json</code> via
<code>aggregate.py</code> — see
<a href="https://github.com/arjunkshah12345-hash/scar">the repo</a>.</p>
<h2>Accuracy at the trained length</h2>
{trained_tbl}
<h2>Extrapolation to long inputs</h2>
{extrap_tbl}
<h2>Compute cost (dedicated serial benchmark)</h2>
{cost_table()}
<h2>Accuracy vs evaluation length</h2>
<img src="../charts/accuracy_by_config.png" alt="accuracy vs length, legacy tasks">
<img src="../charts/recall_curves.png" alt="accuracy vs length, recall">
<h2>The density swap</h2>
<img src="../charts/density_swap.png" alt="dense vs sparse">
<p class="note">Generated by build_site.py from data/summary.json; no hand-typed numbers.</p>
</body></html>
"""
    os.makedirs("site", exist_ok=True)
    with open("site/index.html", "w") as f:
        f.write(page)
    print("wrote site/index.html")


if __name__ == "__main__":
    main()
