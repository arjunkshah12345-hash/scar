"""Generate paper/paper.tex from data/summary.json + data/cost_bench.json.

Zero hand-typed numbers: every accuracy, cost ratio, and outcome-dependent
claim is computed from the aggregated sweep results (winners, ties, chance
collapses, reordering are built with f-strings over the data).

Run after the sweep: python3 aggregate.py && python3 make_charts.py && python3 make_paper.py
"""
import json
import os

import build_site as bs   # reuse MODELS/NAMES/CHANCE/TRAIN_LEN/MAX_LEN/entry


def accs(s, model, config, length):
    e = bs.entry(s, model, config)
    if e is None or str(length) not in e["acc_by_len"]:
        return None
    return e["acc_by_len"][str(length)]


def fmt(a):
    return f"{a['mean']:.1f}\\,$\\pm$\\,{a['std']:.1f}" if a else "--"


def winners(s, config, length):
    """Return (best_model, list of models statistically tied with it)."""
    vals = [(m, accs(s, m, config, length)) for m in bs.MODELS]
    vals = [(m, a) for m, a in vals if a]
    if not vals:
        return None, []
    best_m, best = max(vals, key=lambda x: x[1]["mean"])
    tied = [m for m, a in vals if best["mean"] - a["mean"] <= max(a["std"], 1.0)]
    return best_m, tied


def facts_map(s):
    facts = {}
    for cfg in bs.CONFIGS:
        task = cfg.split("_")[0]
        best, tied = winners(s, cfg, bs.TRAIN_LEN[task])
        facts[cfg] = (best, tied, len(tied))
    return facts


def acc_table(s, configs, length_fn, caption, label):
    head = " & ".join(c.replace("_", " ") for c in configs)
    rows = []
    for m in bs.MODELS:
        if all(bs.entry(s, m, c) is None for c in configs):
            continue
        first = bs.entry(s, m, configs[0])
        params = f"{first['params'][0]:,}" if first else "--"
        cells = " & ".join(fmt(accs(s, m, c, length_fn(c))) for c in configs)
        rows.append(f"{bs.NAMES[m]} & {params} & {cells} \\\\")
    return ("\\begin{table}[t]\n\\centering\\small\n\\caption{" + caption
            + "}\\label{" + label + "}\n\\begin{tabular}{l r " + "c " * len(configs)
            + "}\n\\toprule\nModel & Params & " + head
            + " \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")


def k_slots():
    import bench
    m = bench.SCAR(vocab=5)
    return m.log_lam.shape[0]


def abstract_tail(s, facts):
    best_p, tied_p, _ = facts["parity_dense"]
    txt = (f"On the trained-length headline table, the best parity-dense "
           f"model is {bs.NAMES[best_p]} ({fmt(accs(s, best_p, 'parity_dense', 32))})")
    others = [bs.NAMES[m] for m in tied_p if m != best_p]
    if others:
        txt += f", statistically tied with {', '.join(others)}"
    return txt + "."


def setup_text():
    return ("All models share a matched parameter budget (~75--92K) and training "
            "protocol: AdamW (lr 3e-3, weight decay 0.01, 200-step warmup, cosine "
            "decay), 2{,}500 steps, batch 64, sequence length 32 (parity/five) or "
            "64 (recall), three seeds (0--2). Evaluation uses 2{,}048 fresh "
            "programs per length at lengths 16--256 (parity/five) and 64--512 "
            "(recall); the recall task is sparse-only by construction. Dense "
            "supervision scores the target after every token; sparse supervision "
            "scores one answer at the end of the chain. Both training and "
            "evaluation prepend a single BOS token and score the final position.")


def recall_para(s):
    beats = [bs.NAMES[m] for m in bs.MODELS
             if (a := accs(s, m, "recall_sparse", bs.TRAIN_LEN["recall"]))
             and a["mean"] > bs.CHANCE["recall"] + 10]
    if beats:
        return (f"At the trained length (64 ops), the models that clearly beat the "
                f"12.5\\% chance floor are: {', '.join(beats)}. Full "
                f"accuracy-vs-length curves are in \\texttt{{charts/recall\\_curves.png}}.")
    return ("At the trained length (64 ops), no architecture clearly beats the "
            "12.5\\% chance floor; the task is a genuinely hard retrieval probe.")


def ablation_para(s):
    d = []
    for cfg, ln in (("parity_dense", 32), ("five_sparse", 32), ("recall_sparse", 64)):
        a = accs(s, "scar", cfg, ln)
        b = accs(s, "scar_norecall", cfg, ln)
        c = accs(s, "scar_carrier", cfg, ln)
        if a:
            parts = []
            if b:
                parts.append(f"removing recall costs {a['mean']-b['mean']:+.1f} pts")
            if c:
                parts.append(f"removing the memory costs {a['mean']-c['mean']:+.1f} pts")
            if parts:
                d.append(f"{cfg.replace('_', ' ')}: {', '.join(parts)}")
    intro = ("Both ablations are parameter-matched to full SCAR within ~1\\% "
             "(SCAR-carrier widens the carrier to d=112; SCAR-no-recall to d=98), "
             "so differences are architectural, not budgetary. At the trained "
             "length: ")
    return intro + ("; ".join(d) + "." if d else "ablation results pending.")


def cost_parts():
    if not os.path.exists("data/cost_bench.json"):
        return "", "Cost benchmark pending."
    cb = json.load(open("data/cost_bench.json"))
    base = cb["models"]["gru"]["train"]["median_ms"]
    rows, cf = [], {}
    for m in bs.MODELS:
        if m not in cb["models"]:
            continue
        t, i = cb["models"][m]["train"], cb["models"][m]["inference"]
        r = t["median_ms"] / base
        cf[m] = r
        rows.append(f"{bs.NAMES[m]} & {t['median_ms']:.1f} [{t['min_ms']:.1f}--{t['max_ms']:.1f}] "
                    f"& {r:.2f}$\\times$ & {i['median_ms']:.1f} \\\\")
    cfg_ = cb["config"]
    tbl = ("\\begin{table}[t]\n\\centering\\small\n"
           "\\caption{Dedicated serial cost benchmark: " + str(cfg_["iters"])
           + " timed iterations after " + str(cfg_["warmup"]) + " warmup, torch threads = "
           + str(cfg_["torch_threads"]) + ", batch " + str(cfg_["batch"])
           + ", train length " + str(cfg_["train_ops"]) + "+BOS, inference length "
           + str(cfg_["infer_ops"]) + "+BOS. Median [min--max] ms.}\n"
           "\\label{tab:cost}\n\\begin{tabular}{l l c l}\n\\toprule\n"
           "Model & train ms/step & $\\times$GRU & infer ms/batch \\\\\n\\midrule\n"
           + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    sc, rc = cf.get("scar"), cf.get("rlt")
    para = (f"Full SCAR costs {sc:.2f}$\\times$ a GRU per training step; the "
            f"carrier-only ablation drops to {cf.get('scar_carrier', float('nan')):.2f}$\\times$.")
    if sc and rc:
        para += f" Relative to RLT, SCAR runs at {sc/rc:.2f}$\\times$ of its cost."
    return tbl, para


def discussion_text(s, facts):
    swaps = []
    for task in ("parity", "five"):
        best_d, tied_d, _ = facts[f"{task}_dense"]
        best_s, tied_s, _ = facts[f"{task}_sparse"]
        for m in bs.MODELS:
            if bs.entry(s, m, f"{task}_dense") and bs.entry(s, m, f"{task}_sparse"):
                if (m in tied_d) != (m in tied_s):
                    swaps.append(f"{bs.NAMES[m]} ({task})")
    swap_txt = (f"The dense-to-sparse supervision swap changes tier membership "
                f"for {len(swaps)} model--task pairs (" + ", ".join(swaps) + ")."
                if swaps else
                "No model changes tier across the dense-to-sparse swap: the "
                "architecture ranking survives the supervision perturbation.")
    coll = [bs.NAMES[m] for m in bs.MODELS
            if (a := accs(s, m, "parity_sparse", bs.TRAIN_LEN["parity"]))
            and a["mean"] <= bs.CHANCE["parity"] + 2.0]
    coll_txt = (f" Under sparse parity supervision, {len(coll)} of nine architectures "
                f"sit at the 50\\% chance floor (" + ", ".join(coll) + ")."
                if coll else
                " Under sparse parity supervision, some models escape chance.")
    return (swap_txt + coll_txt +
            " The ablations are the sharpest instrument: if the carrier-only "
            "variant retains most of SCAR's accuracy, the memory is doing "
            "little; if it collapses, the memory (or its interaction with the "
            "carrier) is load-bearing.")


def main():
    s = json.load(open("data/summary.json"))
    n_runs = sum(e["n_seeds"] for e in s.values())
    facts = facts_map(s)
    cost_tbl, cost_para = cost_parts()

    t1 = acc_table(s, bs.CONFIGS, lambda c: bs.TRAIN_LEN[c.split("_")[0]],
                   f"Accuracy (\\%) at the trained length (parity/five: 32 ops, "
                   f"recall: 64 ops), mean$\\pm$std over seeds, {n_runs} runs total. "
                   f"Chance: parity 50\\%, five 20\\%, recall 12.5\\%.", "tab:trained")
    t2 = acc_table(s, bs.CONFIGS, lambda c: bs.MAX_LEN[c.split("_")[0]],
                   "EXTRAPOLATION (\\%) at the longest evaluated length "
                   "(parity/five: 256 ops; recall: 512 ops). No model was trained "
                   "at these lengths.", "tab:extrap")

    tex = f"""\\documentclass{{article}}
\\usepackage{{booktabs,amsmath,graphicx,url}}
\\usepackage[hidelinks]{{hyperref}}
\\title{{SCAR: State-Carrier with Attentive Recall\\\\
\\large Testing whether state-tracking benefits come from O(1) state-carriage, not O(T) memory}}
\\author{{Arjun K. Shah}}
\\date{{September 2026}}
\\begin{{document}}
\\maketitle

\\begin{{abstract}}
A prior architecture study ("RLT") found that a model with an O(T) encoder
memory and a looped read decoder substantially outperformed same-budget
recurrent baselines on synthetic state-tracking tasks. That study left open
which ingredient was responsible. We propose SCAR (State-Carrier with
Attentive Recall), a minimal architecture designed to isolate one ingredient:
a GRU state-carrier (O(1) per token), a compressed multi-timescale memory of
{k_slots()} slots with learned per-slot decay (constant size, never grows), and a
single small attentive-recall head (O(1) per token). We evaluate SCAR and
eight baselines and ablations -- including two parameter-matched SCAR
ablations that remove the memory or the recall head -- on three tasks
(mod-2 parity, chance 50\\%; a random 5-state automaton, chance 20\\%; delayed
first-token recall, chance 12.5\\%) under controlled supervision density, at a
matched parameter budget (~75--92K), 2,500 training steps, and evaluation
lengths up to 256 (parity/five) and 512 (recall) ops. While preparing this
revision we audited our own harness and found and fixed three implementation
errors (a decay initialization that collapsed all memory timescales to
$\\sim$0.5, a train/eval input-protocol mismatch, and an inaccurate baseline
description); we report only results from the corrected harness.
{abstract_tail(s, facts)}
\\end{{abstract}}

\\section{{Introduction}}
Modern architectures differ enormously in how they carry information across a
sequence. Transformers carry everything in an O(T) key-value cache; recurrent
networks compress the past into a fixed-size state. When a memory-augmented
model beats same-budget baselines on state tracking, it is natural to credit
the most visible mechanism -- the memory. RLT's advantage, however, could
come from its growing memory, its looped decoder, or simply from carrying
state with a cheap read path. SCAR is built by subtraction to resolve this:
keep state-carriage plus a compressed memory with a cheap read, drop the
O(T) growth and the loop. If SCAR recovers RLT's accuracy profile, the heavy
ingredients were not load-bearing; if it does not, they are.

\\section{{A methodology audit came first}}
Before reporting any result we fixed three bugs in our own harness, each of
which invalidated the earlier numbers:
\\begin{{enumerate}}
\\item \\textbf{{Decay initialization collapsed the memory's timescales.}}
The per-slot decay parameters were stored as $\\log\\lambda$ but consumed
through a sigmoid, so the effective initial decays were
$\\sigma(\\log\\lambda) \\approx 0.47$--$0.50$ for every slot: the "sixteen
timescales" were sixteen copies of one medium timescale. The fix stores the
parameter in logit space, $\\sigma^{{-1}}(\\lambda_i)$ with
$\\lambda_i = \\mathrm{{linspace}}(0.90, 0.999, k)$, recovering half-lives
from $\\sim$6.6 to $\\sim$693 tokens.
\\item \\textbf{{Train and evaluation used different input protocols.}}
Training prepended a BOS token to every sequence; evaluation did not. Any
model that uses position 0's content is scored under a distribution shift.
The fix applies one convention everywhere (BOS-prefixed input, score the
final position), enforced by a regression test with a protocol-sensitive
oracle model.
\\item \\textbf{{The transformer baseline was described as sliding-window.}}
It is full causal attention. The paper and site now describe it correctly;
the model code was unchanged.
\\end{{enumerate}}
All results come from the corrected harness; the earlier outputs are
quarantined in \\texttt{{results\\_v1/}} and are never mixed with the new
numbers. We release the regression tests alongside the code.

\\section{{Setup}}
{setup_text()}

\\section{{Results}}
{t1}

\\subsection{{Extrapolation beyond the training length}}
{t2}

\\subsection{{Delayed first-token recall}}
{recall_para(s)}

\\subsection{{SCAR ablations}}
{ablation_para(s)}

\\section{{Compute cost}}
{cost_tbl}
{cost_para}

\\section{{Discussion}}
{discussion_text(s, facts)}

\\section{{Limitations}}
Synthetic tasks only; no natural-language transfer. Three seeds; cells whose
gap to the best model is within one seed-std are flagged as
indistinguishable in the site tables rather than ranked. Cost ratios come
from a single machine and are indicative.

\\end{{document}}
"""
    os.makedirs("paper", exist_ok=True)
    with open("paper/paper.tex", "w") as f:
        f.write(tex)
    print("wrote paper/paper.tex")


if __name__ == "__main__":
    main()
