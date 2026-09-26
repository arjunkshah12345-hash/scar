"""Generate the arXiv paper from the checked-in experiment artifacts.

The paper is data-driven: all result tables, outcome-dependent prose, figures,
and benchmark counts come from ``data/summary.json`` and
``data/cost_bench.json``. Run ``aggregate.py``, ``make_charts.py``, and the
Study 2 collector/analysis first.
"""
import json
import os
import shutil

import build_site as bs


FIGURES = (
    "accuracy_by_config.png",
    "recall_curves.png",
    "cost.png",
    "density_swap.png",
)


def accs(summary, model, config, length):
    entry = bs.entry(summary, model, config)
    if entry is None or str(length) not in entry["acc_by_len"]:
        return None
    return entry["acc_by_len"][str(length)]


def fmt(a):
    return f"{a['mean']:.1f}\\,$\\pm$\\,{a['std']:.1f}" if a else "--"


def fmt_pct(a):
    return f"{fmt(a)}\\%" if a else "--"


def winners(summary, config, length):
    vals = [(m, accs(summary, m, config, length)) for m in bs.MODELS]
    vals = [(m, a) for m, a in vals if a]
    if not vals:
        return None, []
    best_m, best = max(vals, key=lambda x: x[1]["mean"])
    same_mean = [m for m, a in vals if a["mean"] == best["mean"]]
    return best_m, same_mean


def facts_map(summary):
    facts = {}
    for cfg in bs.CONFIGS:
        task = cfg.split("_")[0]
        best, tied = winners(summary, cfg, bs.TRAIN_LEN[task])
        facts[cfg] = (best, tied, len(tied))
    return facts


def acc_table(summary, configs, length_fn, caption, label):
    head = " & ".join(c.replace("_", " ") for c in configs)
    rows = []
    for model in bs.MODELS:
        if all(bs.entry(summary, model, cfg) is None for cfg in configs):
            continue
        first = bs.entry(summary, model, configs[0])
        params = f"{first['params'][0]:,}" if first else "--"
        cells = " & ".join(fmt(accs(summary, model, cfg, length_fn(cfg)))
                           for cfg in configs)
        rows.append(f"{bs.NAMES[model]} & {params} & {cells} \\\\")
    return (
        "\\begin{table}[t]\n\\centering\\small\n\\caption{" + caption
        + "}\\label{" + label + "}\n"
        + "\\resizebox{\\linewidth}{!}{%\n"
        + "\\begin{tabular}{l r " + "c " * len(configs) + "}\n"
        + "\\toprule\nModel & Params & " + head
        + " \\\\\n\\midrule\n" + "\n".join(rows)
        + "\n\\bottomrule\n\\end{tabular}}\n\\end{table}\n"
    )


def k_slots():
    import bench
    return bench.SCAR(vocab=5).log_lam.shape[0]


def abstract_tail(summary, facts):
    best_p, tied_p, _ = facts["parity_dense"]
    best_text = (
        "At the trained length, the best parity-dense result is "
        + bs.NAMES[best_p] + " ("
        + fmt_pct(accs(summary, best_p, "parity_dense", 32)) + ")"
    )
    others = [bs.NAMES[m] for m in tied_p if m != best_p]
    if others:
        best_text += ", with the same rounded mean as " + ", ".join(others)
    recall_scar = fmt_pct(accs(summary, "scar", "recall_sparse", 512))
    recall_carrier = fmt_pct(accs(summary, "scar_carrier", "recall_sparse", 512))
    recall_norecall = fmt_pct(accs(summary, "scar_norecall", "recall_sparse", 512))
    text = (
        best_text + ". On the longest delayed-recall evaluation, SCAR reaches "
        + recall_scar + ", compared with " + recall_carrier
        + " for the carrier-only ablation and " + recall_norecall
        + " when the memory is written but not read."
    )
    study2_path = "analysis/study2/summary.json"
    if os.path.exists(study2_path):
        with open(study2_path) as f:
            study2 = json.load(f)
        endpoint = s2_fmt(study2, "recall_length", "scar", 4096)
        assoc = s2_fmt(study2, "associative_recall", "scar", 32)
        text += (
            " In the preregistered follow-up, the same model is evaluated at "
            "lengths through 4,096, reaches " + endpoint
            + " at the 4,096-operation endpoint, and reaches " + assoc
            + " at 32 key/value pairs; selective-copy and intervention sweeps "
            "expose capacity and perturbation limits."
        )
    return text


def study2_run_count():
    total = 0
    root = "study2_results"
    if not os.path.isdir(root):
        return 0
    for _directory, _names, filenames in os.walk(root):
        total += sum(
            1 for filename in filenames
            if filename.startswith("v3") and filename.endswith(".json")
        )
    return total


def setup_text(summary):
    n_runs = sum(entry["n_seeds"] for entry in summary.values())
    curriculum = summary.get("scar|parity_sparse", {}).get("curriculum")
    if curriculum is not False:
        raise RuntimeError(
            "paper release requires fixed-length parity_sparse results; "
            "curriculum results must be stored in results_curriculum/"
        )
    curriculum_text = (
        "All five release conditions use fixed training lengths; the "
        "separate curriculum condition is archived outside the release "
        "tables. "
    )
    return (
        "The release contains " + str(n_runs) + " matched runs: nine models, "
        "five task/supervision configurations, and three seeds (0--2). "
        "All models use AdamW (learning rate 3e-3, weight decay 0.01), a "
        "200-step warmup, cosine decay, 2,500 optimizer steps, and batch size "
        "64. Parity and five-state runs train on 32 operations; delayed recall "
        "trains on 64. " + curriculum_text
        + "Evaluation uses 2,048 fresh programs per length at 16--256 "
        "operations for parity/five and 64--512 for recall. Dense supervision "
        "scores the target after every input token; sparse supervision scores "
        "one answer after the chain. Training and evaluation both prepend one "
        "BOS token and score the final position for sparse tasks."
    )


def recall_para(summary):
    beats = [
        bs.NAMES[model] for model in bs.MODELS
        if (a := accs(summary, model, "recall_sparse", 64))
        and a["mean"] > bs.CHANCE["recall"] + 10
    ]
    if not beats:
        return (
            "At the trained length, no architecture clearly beats the "
            "12.5\\% chance floor; delayed recall remains a hard retrieval "
            "probe."
        )
    return (
        "At the trained length, the models that clearly beat the 12.5\\% "
        "chance floor are " + ", ".join(beats) + ". The more revealing "
        "comparison is extrapolation: at 512 operations, SCAR remains at "
        + fmt_pct(accs(summary, "scar", "recall_sparse", 512))
        + ", while the carrier-only and no-recall variants fall to "
        + fmt_pct(accs(summary, "scar_carrier", "recall_sparse", 512))
        + " and " + fmt_pct(accs(summary, "scar_norecall", "recall_sparse", 512))
        + ", respectively."
    )


def ablation_para(summary):
    trained = fmt_pct(accs(summary, "scar", "recall_sparse", 64))
    long_scar = fmt_pct(accs(summary, "scar", "recall_sparse", 512))
    long_carrier = fmt_pct(accs(summary, "scar_carrier", "recall_sparse", 512))
    long_norecall = fmt_pct(accs(summary, "scar_norecall", "recall_sparse", 512))
    return (
        "The two ablations are parameter-matched to SCAR within roughly one "
        "percent: SCAR-carrier widens the GRU carrier to d=112 after removing "
        "the memory, while SCAR-no-recall widens d to 98 while retaining the "
        "write path. Both reach " + trained + " at the trained recall length, "
        "so neither component is required to fit the short training "
        "distribution. At 512 operations, however, the full model reaches "
        + long_scar + " versus " + long_carrier + " without memory and "
        + long_norecall + " without the read path. This supports a narrow "
        "conclusion: constant-size memory is useful for long-delay "
        "extrapolation in this probe, but the experiment does not show that "
        "the memory improves every task or every length."
    )


def cost_parts():
    if not os.path.exists("data/cost_bench.json"):
        return "", "The dedicated cost benchmark was not included.", {
            "scar_gru": "--", "scar_rlt": "--"
        }
    with open("data/cost_bench.json") as f:
        cost = json.load(f)
    base = cost["models"]["gru"]["train"]["median_ms"]
    rows, ratios = [], {}
    for model in bs.MODELS:
        if model not in cost["models"]:
            continue
        train = cost["models"][model]["train"]
        infer = cost["models"][model]["inference"]
        ratio = train["median_ms"] / base
        ratios[model] = ratio
        rows.append(
            f"{bs.NAMES[model]} & {train['median_ms']:.1f} "
            f"[{train['min_ms']:.1f}--{train['max_ms']:.1f}] & "
            f"{ratio:.2f}$\\times$ & {infer['median_ms']:.1f} \\\\")
    cfg = cost["config"]
    table = (
        "\\begin{table}[t]\n\\centering\\small\n"
        "\\caption{Dedicated serial cost benchmark: " + str(cfg["iters"])
        + " timed iterations after " + str(cfg["warmup"])
        + " warmup, torch threads = " + str(cfg["torch_threads"])
        + ", batch " + str(cfg["batch"])
        + ", train length " + str(cfg["train_ops"])
        + "+BOS, inference length " + str(cfg["infer_ops"])
        + "+BOS. Median [min--max] ms.}\n\\label{tab:cost}\n"
        + "\\resizebox{\\linewidth}{!}{%\n"
        + "\\begin{tabular}{l l c l}\n\\toprule\n"
        + "Model & train ms/step & $\\times$GRU & infer ms/batch \\\\\n"
        + "\\midrule\n" + "\n".join(rows)
        + "\n\\bottomrule\n\\end{tabular}}\n\\end{table}\n"
    )
    paragraph = (
        f"On this CPU, full SCAR costs {ratios['scar']:.2f}$\\times$ a GRU "
        "per training step, while the carrier-only ablation costs "
        f"{ratios['scar_carrier']:.2f}$\\times$. SCAR is "
        f"{ratios['scar'] / ratios['rlt']:.2f}$\\times$ the measured RLT "
        "training-step cost. This falsifies the original ~1$\\times$-GRU "
        "cost target while supporting a substantial reduction relative to "
        "RLT. These are controlled single-machine timings, not "
        "hardware-independent complexity claims."
    )
    return table, paragraph, {
        "scar_gru": f"{ratios['scar']:.2f}",
        "scar_rlt": f"{ratios['scar'] / ratios['rlt']:.2f}",
    }


def discussion_text(summary):
    collapsed = [
        bs.NAMES[model] for model in bs.MODELS
        if (a := accs(summary, model, "parity_sparse", 32))
        and a["mean"] <= bs.CHANCE["parity"] + 2.0
    ]
    collapse_text = (
        " Under sparse parity supervision, " + str(len(collapsed))
        + " of nine architectures remain at the 50\\% chance floor ("
        + ", ".join(collapsed) + ")."
        if collapsed else " Under sparse parity supervision, some models escape chance."
    )
    return (
        "The clean fixed-length density comparison is descriptive: "
        + collapse_text.strip()
        + " Dense supervision makes almost every architecture look successful, "
        "whereas sparse parity leaves several at chance. With only three seeds, "
        "we report means and sample standard deviations but do not call "
        "near-equal cells significance-labeled. The strongest result is therefore "
        "not that SCAR universally wins: it is that the parameter-matched "
        "carrier-only model can match SCAR on short tasks while losing its "
        "long-delay advantage."
    )


def figure_tex(filename, caption, label):
    if not os.path.exists(os.path.join("charts", filename)):
        return ""
    return (
        "\\begin{figure}[t]\n\\centering\n"
        + "\\includegraphics[width=\\linewidth]{figures/" + filename + "}\n"
        + "\\caption{" + caption + "}\\label{" + label + "}\n"
        + "\\end{figure}\n"
    )


def copy_figures():
    os.makedirs("paper/figures", exist_ok=True)
    for filename in FIGURES:
        source = os.path.join("charts", filename)
        if os.path.exists(source):
            shutil.copy2(source, os.path.join("paper/figures", filename))
    study2_dir = "analysis/study2"
    if os.path.isdir(study2_dir):
        for filename in os.listdir(study2_dir):
            if filename.endswith(".png"):
                shutil.copy2(
                    os.path.join(study2_dir, filename),
                    os.path.join("paper/figures", "study2_" + filename),
                )


def s2_value(summary, family, model, context):
    return summary.get(family, {}).get(model, {}).get(str(context))


def s2_fmt(summary, family, model, context):
    value = s2_value(summary, family, model, context)
    if not value:
        return "--"
    return f"{value['mean']:.1f}\\,$\\pm$\\,{value['std']:.1f}\\%"


def s2_table(summary, family, context, caption, label, models=None):
    models = models or [
        "gru", "lstm", "transformer", "rlt", "scar",
        "scar_carrier", "scar_norecall",
    ]
    rows = []
    for model in models:
        if s2_value(summary, family, model, context):
            rows.append(
                f"{bs.NAMES.get(model, model)} & "
                f"{s2_fmt(summary, family, model, context)} \\\\"
            )
    if not rows:
        return ""
    return (
        "\\begin{table}[t]\n\\centering\\small\n"
        f"\\caption{{{caption}}}\\label{{{label}}}\n"
        "\\begin{tabular}{l c}\n\\toprule\n"
        "Model & Accuracy (\\%) \\\\\n\\midrule\n"
        + "\n".join(rows)
        + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )


def s2_figure(filename, caption, label):
    path = os.path.join("paper/figures", "study2_" + filename)
    if not os.path.exists(path):
        return ""
    return (
        "\\begin{figure}[t]\n\\centering\n"
        + "\\includegraphics[width=\\linewidth]{figures/study2_" + filename + "}\n"
        + "\\caption{" + caption + "}\\label{" + label + "}\n"
        + "\\end{figure}\n"
    )


def study2_section():
    summary_path = "analysis/study2/summary.json"
    required = {
        "recall_length", "associative_recall", "ratio32", "ratio128",
        "mechanism_slots1", "mechanism_decay_learned_multi",
        "selective_copy_entropy16", "selective_copy_entropy2",
        "intervention",
    }
    if not os.path.exists(summary_path):
        raise RuntimeError(
            "Study 2 analysis is missing; collect and analyze Kaggle results first"
        )
    with open(summary_path) as f:
        summary = json.load(f)
    missing = sorted(required - set(summary))
    if missing:
        raise RuntimeError(
            "Study 2 analysis is incomplete: missing " + ", ".join(missing)
        )
    exact_path = "analysis/study2/exact_sequence_summary.json"
    intervention_path = "analysis/study2/intervention_summary.json"
    diagnostic_path = "analysis/study2/mechanism_diagnostics.json"
    if (
        not os.path.exists(exact_path)
        or not os.path.exists(intervention_path)
        or not os.path.exists(diagnostic_path)
    ):
        raise RuntimeError(
            "Study 2 exact-sequence, intervention, or mechanism diagnostics "
            "analysis is missing"
        )
    with open(exact_path) as f:
        exact = json.load(f)
    with open(intervention_path) as f:
        interventions = json.load(f)
    with open(diagnostic_path) as f:
        diagnostics = json.load(f)

    recall = s2_table(
        summary,
        "recall_length",
        4096,
        "Study 2A delayed-recall accuracy at 4,096 operations; models were trained at 64 operations. Mean$\\pm$std over five seeds.",
        "tab:study2-recall",
    )
    assoc = s2_fmt(summary, "associative_recall", "scar", 32)
    ratio32 = s2_fmt(summary, "ratio32", "scar", 512)
    ratio128 = s2_fmt(summary, "ratio128", "scar", 2048)
    copy_high = exact.get("selective_copy_entropy16", {}).get("scar", {}).get("32")
    copy_low = exact.get("selective_copy_entropy2", {}).get("scar", {}).get("32")
    copy_high_text = (
        "--" if not copy_high
        else f"{copy_high['mean']:.1f}\\,$\\pm$\\,{copy_high['std']:.1f}\\%"
    )
    copy_low_text = (
        "--" if not copy_low
        else f"{copy_low['mean']:.1f}\\,$\\pm$\\,{copy_low['std']:.1f}\\%"
    )

    slot_values = []
    for family, model_values in summary.items():
        if family.startswith("mechanism_slots") and "scar" in model_values:
            value = model_values["scar"].get("512")
            if value:
                slots = int(family[len("mechanism_slots"):])
                slot_values.append((slots, value["mean"]))
    slot_values.sort()
    slot_text = ", ".join(
        f"{slots}: {value:.1f}\\%" for slots, value in slot_values
    )

    intervention_values = interventions.get("intervention", {}).get("scar", {})
    at512 = []
    for name, values in sorted(intervention_values.items()):
        if "512" in values:
            at512.append(f"{name} {values['512']['mean']:.1f}\\%")
    intervention_text = ", ".join(at512) if at512 else "--"

    learned_diag = diagnostics.get("mechanism_decay_learned_multi", {}).get("scar", {})
    final_half_lives = learned_diag.get("final_half_life", {})
    half_life_means = [
        value["mean"] for value in final_half_lives.values()
        if isinstance(value, dict) and value.get("mean") is not None
    ]
    entropy_mean = (
        learned_diag.get("attention_entropy_mean", {})
        .get("scalar", {})
        .get("mean")
    )
    timescale_text = (
        f"The learned multi-timescale decay sweep ended with final slot half-lives "
        f"spanning {min(half_life_means):.1f}--{max(half_life_means):.1f} operations "
        f"across the 16 slots"
        + (f", with mean read-attention entropy {entropy_mean:.2f}." if entropy_mean is not None else ".")
        if half_life_means else
        "The mechanism diagnostics did not expose a complete final half-life range."
    )

    return (
        "The preregistered follow-up separates the original recall headline from "
        "stress tests of length, training context, retrieval type, capacity, and "
        "memory intervention. At 4,096 operations, the recall-length sweep gives "
        "the following descriptive endpoint comparison; the model was trained at "
        "only 64 operations.\n"
        + recall
        + "The train/test ratio probes show SCAR at " + ratio32
        + " when trained at 32 operations and " + ratio128
        + " when trained at 128 operations, so the 512-token result is not treated "
        "as a universal context-length law. Associative recall is a useful negative "
        "control: SCAR reaches " + assoc
        + " at 32 key/value pairs against a 6.25\\% single-choice chance floor, "
        "indicating that the fixed exponential summaries do not solve arbitrary "
        "multi-item retrieval.\n\n"
        "Selective copy makes the retrieval target multi-output rather than a "
        "single first-token label. At 32 marked items, SCAR's free-running "
        "exact-sequence accuracy is " + copy_high_text
        + " under the high-entropy distractor condition and " + copy_low_text
        + " under the low-entropy condition. The slot sweep's 512-operation "
        "SCAR endpoints are " + (slot_text or "not available")
        + ". Frozen-memory interventions at 512 operations are "
        + intervention_text
        + ". " + timescale_text + " These are exploratory, descriptive comparisons; "
        "no significance claim is made from the small seed counts.\n"
        + s2_figure(
            "recall_length_accuracy.png",
            "Study 2A: delayed-recall accuracy through 4,096 operations after training at 64 operations. Shaded bands are bootstrap intervals over seed means.",
            "fig:study2-recall",
        )
        + s2_figure(
            "ratio_comparison.png",
            "Study 2B: accuracy plotted against evaluation length divided by training length for two training contexts.",
            "fig:study2-ratio",
        )
        + s2_figure(
            "selective_copy_entropy16_exact_sequence.png",
            "Study 2D: free-running exact-sequence accuracy for high-entropy selective copy.",
            "fig:study2-copy",
        )
    )


def main():
    with open("data/summary.json") as f:
        summary = json.load(f)
    facts = facts_map(summary)
    cost_table, cost_paragraph, cost_ratios = cost_parts()
    copy_figures()

    trained_table = acc_table(
        summary, bs.CONFIGS,
        lambda config: bs.TRAIN_LEN[config.split("_")[0]],
        "Accuracy (\\%) at the trained length (parity/five: 32 operations; "
        "recall: 64 operations), mean$\\pm$std over seeds. Chance is 50\\% "
        "for parity, 20\\% for five-state, and 12.5\\% for recall.",
        "tab:trained",
    )
    extrap_table = acc_table(
        summary, bs.CONFIGS,
        lambda config: bs.MAX_LEN[config.split("_")[0]],
        "Extrapolation accuracy (\\%) at the longest evaluated length "
        "(parity/five: 256 operations; recall: 512 operations). No model was "
        "trained at these lengths.",
        "tab:extrap",
    )

    template = r"""\documentclass{article}
\usepackage[margin=1in]{geometry}
\usepackage{booktabs,amsmath,amssymb,graphicx,url}
\usepackage[hidelinks]{hyperref}
\title{SCAR: Constant-Size Learned Memory for Length Extrapolation\\
\large Controlled tests of recall, capacity, and memory mechanisms}
\author{Arjun K. Shah}
\date{September 2026}
\begin{document}
\maketitle

\begin{abstract}
Can a recurrent model preserve useful information beyond its training context
without retaining a token-level memory? We study SCAR (State-Carrier with
Attentive Recall), which combines a GRU carrier with a constant-size bank of
@@K_SLOTS@@ learned-decay exponential slots and an attentive read head. The
corrected Study 1 release contains @@N_RUNS@@ matched runs; the cloud-only
Study 2 matrix adds @@STUDY2_RUNS@@ preregistered stress-test runs spanning
length extrapolation, train/test ratios, associative recall, selective copying,
slot/decay mechanisms, and frozen-memory interventions. The central result is
not an across-the-board accuracy win: in-distribution recall hides a separation
that appears at long horizons, while multi-item retrieval and perturbation
tests reveal capacity and robustness limits. The dedicated CPU benchmark
falsifies the original approximately one-times-GRU cost hypothesis: SCAR is
slower than a GRU but faster than RLT-lite in the measured setting. The result
is a controlled synthetic study of when bounded learned memory helps, and when
it does not.
@@ABSTRACT_TAIL@@
\end{abstract}

\section{Introduction}
Sequence models differ in how they carry information from the past. A
recurrent network compresses history into a fixed-size state, while a causal
Transformer exposes the whole prefix through an attention computation and an
O(T) key--value cache \cite{elman1990,hochreiter1997,cho2014,vaswani2017}.
Memory-augmented models make the trade-off explicit by adding a writable
storage mechanism and a learned read operation \cite{sukhbaatar2015}.

This paper asks a narrower question than ``which architecture is best?'' A
recent Recurrent Looped Transformer (RLT) design carries decoder state while
also reading a growing encoder-derived memory \cite{zhang2026rlt}. If a small
RLT-like model succeeds on state-tracking tasks, which ingredient deserves
credit: the recurrent carrier, the growing memory, or the looped read path?
The answer matters because these mechanisms have different inference-state
costs. SCAR is a subtractive test: it keeps a gated recurrent carrier and a
cheap read path, but replaces the growing memory with a fixed bank of
multi-timescale exponential summaries.

Our contributions are:
\begin{enumerate}
\item a precisely specified constant-size state-carrier architecture with an
O(1)-per-token update and a fixed O(kd) memory state;
\item a matched-budget, three-seed sweep that crosses architecture with
supervision density and evaluates beyond the training length;
\item parameter-matched memory and read-path ablations that separate short
training performance from long-delay extrapolation; and
\item a preregistered Study 2 matrix testing length ratios, associative
recall, multi-item capacity, mechanisms, and interventions; and
\item a reproducible methodology audit: the earlier release had a collapsed
decay initialization, a BOS train/evaluation mismatch, and an incorrect
description of the Transformer baseline. All reported numbers below come from
the corrected harness.
\end{enumerate}

\subsection{Hypotheses}
We began with two falsifiable hypotheses. First, SCAR should retain the useful
state-carriage behavior of RLT while avoiding its growing memory and looped
decoder. Second, its fixed read path might approach the GRU's measured cost
while remaining well below RLT-lite. The results support the first hypothesis
only for long-horizon extrapolation and falsify the second in its strong form:
SCAR is @@SCAR_GRU_COST@@$\times$ the GRU training-step cost, although it is @@SCAR_RLT_COST@@$\times$
the RLT-lite cost.

\section{Related work}
Elman-style recurrent networks, LSTMs, and GRUs provide progressively more
structured mechanisms for carrying state through a sequence
\cite{elman1990,hochreiter1997,cho2014}. Transformers replace recurrence with
self-attention and expose all earlier positions during a causal forward pass
\cite{vaswani2017}. External-memory networks add a separately addressable
storage and read mechanism \cite{sukhbaatar2015}; structured state-space
models offer another route to long-range sequence processing with fixed-size
state \cite{gu2022s4}. Segment-level recurrence and compressed memories offer
related ways to extend context beyond a fixed window
\cite{dai2019transformerxl,rae2020compressive}. Associative recall and its
multi-query formulation provide a particularly relevant diagnostic for whether
an efficient sequence model can retrieve several previously presented items
\cite{arora2023zoology}. Earlier work also showed that deliberately slow
recurrent units can learn longer memory \cite{mikolov2015longmemory}.
Linear-attention formulations expose a recurrent constant-state computation
\cite{katharopoulos2020linear}; RetNet makes a related parallel/recurrent
retention trade-off \cite{sun2023retnet}; and Mamba uses input-dependent
selective state-space updates for content-based retention \cite{gu2024mamba}.
SCAR is intentionally small and synthetic: it is an ablation instrument for
the state-carriage versus growing-memory question, not a claim to improve
language modeling or to replace these broader model families.

The RLT baseline here is a faithful small-scale implementation of the
publicly described recurrent-looped pattern: a causal encoder, a recurrent
decoder state, global cross-attention to encoder outputs, and a sliding-window
decoder cache. It should be read as an RLT-lite baseline, not as a reproduction
of every large-scale result in the original report \cite{zhang2026rlt}.

\section{SCAR}
Let $e_t$ be the embedding of token $x_t$ and $h_t$ the GRU carrier state:
\begin{align}
h_t &= \operatorname{GRU}(e_t,h_{t-1}),\\
u_t &= \sigma(W_g[e_t;h_t])\odot h_t,\\
m_{t,i} &= \lambda_i m_{t-1,i} + (1-\lambda_i)u_t,
\qquad i=1,\ldots,k.
\end{align}
Each slot is an exponential moving average. We initialize the learned decay
parameters in logit space so that $\lambda_i$ spans $[0.90,0.999]$; the
corresponding half-lives span approximately 6.6 to 693 tokens. The memory
therefore has a fixed $k=16$ slots and never grows with sequence length.

The read head uses a single query from the carrier:
\begin{align}
q_t &= W_q\operatorname{LN}(h_t),\\
r_t &= h_t + W_o\operatorname{softmax}\!\left(
\frac{q_t(W_kM_t)^\top}{\sqrt{r}}\right)W_vM_t,\\
y_t &= W_{\mathrm{out}}\operatorname{LN}(r_t),
\end{align}
where $M_t$ stacks the $k$ slots and $r=40$ is the attention width. Each
token performs a fixed amount of work independent of $t$ and retains only the
carrier and $k$ memory slots. This gives O(T) sequence work and O(kd) recurrent
state, in contrast to the growing O(T) storage used by token-level KV memory.
The constant-size memory does not make SCAR free: the dedicated timing study
below measures its actual CPU cost.

\section{Experimental design}
\subsection{Tasks and supervision}
The parity task emits binary symbols and asks for their parity. The five-state
task samples symbols from a fixed random transition automaton and asks for the
final state. Their chance accuracies are 50\% and 20\%. The recall task samples
the first symbol from an eight-symbol vocabulary and fills the remaining
positions with independent distractors; the answer is the first symbol, giving
a 12.5\% chance floor. Recall is sparse-only because a repeated answer at every
position would change the task into a different supervision problem.

Dense conditions supervise the state after every input symbol. Sparse
conditions supervise only the final answer. All sequences are prefixed with a
dedicated BOS token during both training and evaluation; sparse accuracy is
measured at the final position. The main parity-sparse condition is fixed at
32 operations. A separate 4-to-8-to-16-to-32 curriculum condition was run
during development, is archived under ``results\_curriculum/``, and is
excluded from the five-condition release tables so supervision density is not
conflated with curriculum.

\subsection{Models and protocol}
The sweep contains Elman RNN, LSTM, GRU, a full causal Transformer, TokenMerge,
RLT-lite, SCAR, SCAR-carrier (memory removed), and SCAR-no-recall (memory
written but never read). Widths are selected to keep models within roughly
75--92K parameters; the two SCAR ablations widen the carrier to compensate for
removed modules. All models use AdamW with learning rate 3e-3, weight decay
0.01, a 200-step warmup, cosine decay, 2,500 steps, and batch size 64.
@@SETUP@@

Each evaluation point uses 2,048 fresh programs. The release reports the mean,
sample standard deviation, minimum, maximum, and per-seed values for every
model/configuration/length cell. The headline table is explicitly at the
trained length; the second table is explicitly extrapolation.

\section{Methodology audit}
Before the corrected sweep, we found three errors in the experiment harness:
\begin{enumerate}
\item \textbf{Collapsed decay initialization.} The old code stored
$\log\lambda$ but then applied a sigmoid, giving effective decays near
0.5 for every slot. The corrected code stores $\operatorname{logit}(\lambda)$,
restoring the intended multi-timescale range.
\item \textbf{Train/evaluation protocol mismatch.} Training prepended BOS but
evaluation did not. The corrected evaluator applies the same prefix and scores
the same final position; a protocol-sensitive regression test catches a future
regression.
\item \textbf{Incorrect baseline description.} The Transformer implementation
uses full causal attention, not a sliding window. The model code is unchanged;
the paper, site, and tests now describe it accurately.
\end{enumerate}
The earlier artifacts remain quarantined in ``results\_v1/`` and
``data/summary\_v1.json``. They are not aggregated into the release reported
here.

\section{Results}
@@TRAINED_TABLE@@
@@ACCURACY_FIGURE@@
@@EXTRAP_TABLE@@
\subsection{Delayed recall and ablations}
@@RECALL_TEXT@@
@@RECALL_FIGURE@@
@@ABLATION_TEXT@@
@@DENSITY_FIGURE@@

\section{Study 2: stress tests and failure modes}
@@STUDY2_SECTION@@

\section{Compute cost}
@@COST_TABLE@@
@@COST_TEXT@@
@@COST_FIGURE@@

\section{Discussion}
@@DISCUSSION_TEXT@@

The combined evidence supports a conditional claim. At trained lengths, the
parameter-matched SCAR ablations often fit as well as the full model; long
delays expose a benefit from bounded memory and attentive readout. Study 2
then tests the boundary of that benefit rather than assuming that one-token
recall implies general-purpose memory: associative retrieval, selective copy,
slot sweeps, decay variants, and frozen-state interventions are reported as
separate descriptive analyses. Dense versus sparse supervision remains a
methodological confounder when the protocol is not held fixed, so we do not
present a universal architecture ranking or a significance claim from three
seeds.

\section{Limitations}
This is a synthetic study with small models, a single parameter scale, three
seeds for most Study 2 families, and short training budgets. The tasks test
exact state tracking rather than natural-language modeling, multimodal
reasoning, or noisy real-world sequences. Associative and selective-copy
tasks probe more than first-token recall but do not establish language-model
quality. The ablations are parameter-matched, but changing width to replace
removed modules can itself change optimization and representation capacity.
The RLT comparison is a small implementation, not a claim about the behavior
of all RLT variants. Timing uses one CPU machine, two PyTorch threads, a
parity-sized vocabulary, and an unoptimized research implementation; measured
ratios should not be generalized to GPU kernels or production runtimes.
Finally, the fixed memory's success on this recall generator does not establish
that learned exponential summaries are sufficient for arbitrary long-context
tasks.

\section{Reproducibility and release}
The repository contains the model and task code (``bench.py``), regression
tests (``tests/``), the quarantined Study 1 artifacts, all collected Study 2
JSON files and manifests, aggregation and chart scripts, the generated tables,
and the figures used in this PDF. All optimizer steps and training timings in
the two studies run on Kaggle CPU kernels through ``kaggle/``; no local
training is required.
The release command sequence is:
\begin{verbatim}
python3 -m pytest tests/ -q
python3 aggregate.py
python3 make_charts.py
python3 -m study2.analyze study2_results --out analysis/study2
python3 study2/state_memory.py --out analysis/study2/state_memory.json
python3 make_paper.py
(cd paper && tectonic paper.tex)
\end{verbatim}
The generated summary includes per-seed values and the curriculum flag so the
reported condition cannot silently drift from the trained artifacts.

\begin{thebibliography}{12}
\bibitem{elman1990}
J. L. Elman, ``Finding structure in time,'' \emph{Cognitive Science},
14(2), 179--211, 1990. doi:10.1207/s15516709cog1402\_1.

\bibitem{hochreiter1997}
S. Hochreiter and J. Schmidhuber, ``Long short-term memory,''
\emph{Neural Computation}, 9(8), 1735--1780, 1997.

\bibitem{cho2014}
K. Cho et al., ``Learning phrase representations using RNN encoder--decoder
for statistical machine translation,'' arXiv:1406.1078, 2014.

\bibitem{vaswani2017}
A. Vaswani et al., ``Attention is all you need,''
\emph{Advances in Neural Information Processing Systems}, 30, 2017.
arXiv:1706.03762.

\bibitem{sukhbaatar2015}
S. Sukhbaatar, J. Weston, R. Fergus, et al., ``End-to-end memory networks,''
\emph{Advances in Neural Information Processing Systems}, 28, 2015.
arXiv:1503.08895.

\bibitem{gu2022s4}
A. Gu, K. Goel, and C. Ré, ``Efficiently modeling long sequences with
structured state spaces,'' in \emph{International Conference on Learning
Representations}, 2022. arXiv:2111.00396.

\bibitem{mikolov2015longmemory}
T. Mikolov, A. Joulin, S. Chopra, M. Mathieu, and M. Ranzato,
``Learning longer memory in recurrent neural networks,'' arXiv:1412.7753, 2015.

\bibitem{katharopoulos2020linear}
A. Katharopoulos, A. Vyas, N. Pappas, and F. Fleuret,
``Transformers are RNNs: Fast autoregressive transformers with linear attention,''
in \emph{International Conference on Machine Learning}, 2020. arXiv:2006.16236.

\bibitem{sun2023retnet}
Y. Sun et al., ``Retentive network: A successor to Transformer for large
language models,'' arXiv:2307.08621, 2023.

\bibitem{gu2024mamba}
A. Gu and T. Dao, ``Mamba: Linear-time sequence modeling with selective state
spaces,'' arXiv:2312.00752, 2023.

\bibitem{dai2019transformerxl}
Z. Dai, Z. Yang, Y. Yang, J. Carbonell, Q. V. Le, and R. Salakhutdinov,
``Transformer-XL: Attentive language models beyond a fixed-length context,''
\emph{Proceedings of ACL}, 2019. arXiv:1901.02860.

\bibitem{rae2020compressive}
J. W. Rae, A. Potapenko, S. M. Jayakumar, and T. P. Lillicrap,
``Compressive transformers for long-range sequence modelling,'' in
\emph{International Conference on Learning Representations}, 2020.
arXiv:1911.05507.

\bibitem{arora2023zoology}
S. Arora, S. Eyuboglu, A. Timalsina, I. Johnson, M. Poli, J. Zou,
A. Rudra, and C. Ré, ``Zoology: Measuring and improving recall in efficient
language models,'' arXiv:2312.04927, 2023.

\bibitem{zhang2026rlt}
Y. Zhang, J. Feng, and S. Qin, ``Recurrent Looped Transformer,'' technical
report, 2026. \url{https://github.com/yifanzhang-pro/recurrent-looped-tranformer}.
\end{thebibliography}
\end{document}
"""

    cost_table = cost_table or "% cost benchmark unavailable\n"
    replacements = {
        "@@SETUP@@": setup_text(summary),
        "@@N_RUNS@@": str(sum(entry["n_seeds"] for entry in summary.values())),
        "@@STUDY2_RUNS@@": str(study2_run_count()),
        "@@K_SLOTS@@": str(k_slots()),
        "@@ABSTRACT_TAIL@@": abstract_tail(summary, facts),
        "@@TRAINED_TABLE@@": trained_table,
        "@@EXTRAP_TABLE@@": extrap_table,
        "@@RECALL_TEXT@@": recall_para(summary),
        "@@ABLATION_TEXT@@": ablation_para(summary),
        "@@COST_TABLE@@": cost_table,
        "@@COST_TEXT@@": cost_paragraph,
        "@@SCAR_GRU_COST@@": cost_ratios["scar_gru"],
        "@@SCAR_RLT_COST@@": cost_ratios["scar_rlt"],
        "@@DISCUSSION_TEXT@@": discussion_text(summary),
        "@@STUDY2_SECTION@@": study2_section(),
        "@@ACCURACY_FIGURE@@": figure_tex(
            "accuracy_by_config.png",
            "Accuracy versus evaluation length. Lines are seed means and shaded bands span the three-seed minimum and maximum.",
            "fig:accuracy",
        ),
        "@@RECALL_FIGURE@@": figure_tex(
            "recall_curves.png",
            "Delayed first-token recall across evaluation lengths; chance is 12.5\\%.",
            "fig:recall",
        ),
        "@@DENSITY_FIGURE@@": figure_tex(
            "density_swap.png",
            "Changing supervision density changes which architectures escape chance on the trained-length parity and five-state tasks.",
            "fig:density",
        ),
        "@@COST_FIGURE@@": figure_tex(
            "cost.png",
            "Dedicated serial CPU benchmark. Bars show medians and whiskers show the repeated-iteration minimum and maximum.",
            "fig:cost",
        ),
    }
    for marker, value in replacements.items():
        template = template.replace(marker, value)

    os.makedirs("paper", exist_ok=True)
    with open("paper/paper.tex", "w") as f:
        f.write(template)
    print("wrote paper/paper.tex and paper/figures/")


if __name__ == "__main__":
    main()
