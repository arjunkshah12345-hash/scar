# SCAR Study 2 Protocol (v3)

Status: frozen before Study 2 result collection  
Protocol version: `v3.0`  
Study 1 commit: `eccca7b` / corrected release lineage  
Study 2 branch: `research/v3`

This document freezes the follow-up protocol before inspecting Study 2 results.
Study 1 remains an immutable historical result. New artifacts must carry this
protocol version and a unique experiment ID; they must never be copied into the
Study 1 `results/` directory.

## 1. Scientific questions

The primary question is:

> When can constant-size learned memory preserve information under context-length
> extrapolation, and which SCAR components cause that behavior?

The secondary methodological question is:

> Can training-length accuracy hide large differences in memory retention?

The preregistered hypotheses are:

* H1 — Full SCAR will retain delayed-recall accuracy farther beyond its training
  length than the parameter-matched carrier-only and no-recall variants.
* H2 — The separation will persist across train lengths and when performance is
  plotted against evaluation/training-length ratio, rather than only at one
  absolute length.
* H3 — Increasing the number of independent items to retain will expose a
  capacity limit even when single-item duration remains strong.
* H4 — Long-horizon performance will depend on a non-collapsed distribution of
  timescales and on the attentive read path, not merely on extra parameters.
* H5 — Dense supervision and curriculum can change whether a model discovers a
  length-general algorithm; they are separate interventions, not nuisance
  details.

The protocol permits all outcomes, including: SCAR failing on richer memory
tasks, a strong GRU matching it with a better curriculum, learned decays
collapsing, or the apparent advantage being specific to first-token recall.

## 2. Study 1: immutable corrected baseline

Study 1 is the checked-in corrected release and is not rerun or silently
rewritten:

* 9 architectures × 5 fixed task/supervision conditions × 3 seeds = 135 runs;
* 75–92K parameter target;
* parity and five-state tracking plus delayed first-token recall;
* dense versus sparse supervision and evaluation beyond training length;
* SCAR, SCAR-carrier, and SCAR-no-recall ablations;
* dedicated cost measurement;
* separate 27-run parity curriculum archive in `results_curriculum/`.

The v1 artifacts stay quarantined in `results_v1/`. Study 1 headline values
remain the existing 512-operation recall result: SCAR 100.0 ± 0.0%,
SCAR-carrier 56.6 ± 18.5%, and SCAR-no-recall 30.5 ± 32.4%. The measured
training cost remains 3.12× GRU and 0.40× RLT-lite; the original ~1× GRU cost
hypothesis is falsified.

## 3. Shared Study 2 rules

### 3.1 Training and cloud policy

* All optimizer steps and timing involving training run in Kaggle kernels.
* Local execution may inspect code, run tests, aggregate JSON, make plots, and
  compile the paper. Local scripts must fail closed before an optimizer step.
* Every Kaggle driver clones the exact public commit named in its metadata.
* Kaggle CPU is the default. GPU is allowed only when a protocol entry names it
  explicitly and records the hardware.
* Training uses AdamW, learning rate 3e-3, weight decay 0.01, 200-step linear
  warmup, cosine decay to 0.1×, gradient norm clipping at 1.0, and batch size
  64 unless a study entry states otherwise.
* The default budget is 2,500 optimizer steps. A longer budget is a new
  protocol entry, not an unrecorded rescue.
* Seeds are integer seeds and control Python/NumPy/PyTorch generation. The
  task generator, model initialization, training stream, and evaluation stream
  record their derived seeds.

### 3.2 Models

The primary comparison set is:

* GRU;
* LSTM;
* full causal Transformer;
* RLT-lite;
* SCAR;
* SCAR-carrier (memory removed, widened carrier);
* SCAR-no-recall (memory written but read path removed).

The original Elman and TokenMerge baselines remain in Study 1. They are added
to a Study 2 matrix only when the experiment manifest explicitly requires them.
All model factories report parameter count. Ablation comparisons must include
the parameter count and the exact width-selection rule.

### 3.3 Artifact contract

Each raw JSON artifact must contain:

```text
study, protocol_version, experiment_id, git_commit,
model, variant, seed, task, task_parameters,
train_context, eval_contexts, eval_examples,
params, optimizer, lr, weight_decay, warmup_steps,
batch_size, steps, training_examples, training_tokens,
supervision, curriculum, metrics, raw_metrics,
train_seconds, train_ms_per_step, inference_ms_per_example,
env.torch, env.numpy, env.python, env.platform,
env.device, env.cpu, env.gpu, env.torch_threads, timestamp_utc
```

Collectors must reject missing required fields, malformed experiment IDs,
duplicate `(experiment_id, model, seed)`, incomplete seed sets, and artifacts
whose recorded commit does not match the submitted kernel commit. Completed
artifacts are never overwritten silently; a rerun writes a new attempt path
and is admitted only after deterministic deduplication.

### 3.4 Evaluation and statistics

* Evaluation sequences are generated independently from training sequences.
* Primary accuracy is exact final-answer accuracy; report chance level.
* Report every seed, mean, sample standard deviation, median, min, max, and
  paired per-seed differences against the named comparator.
* Central Study 2 claims use at least five seeds where practical. Secondary
  matrices use three seeds and are labelled accordingly.
* For central paired comparisons, compute a paired bootstrap 95% confidence
  interval over seed-level differences and an exact or permutation test only as
  a supplement. No “statistically tied” language is permitted.
* No seed, length, or task is excluded after seeing outcomes. A failed run is a
  failed run and is reported separately from a scientifically justified rerun.

## 4. Preregistered Study 2A — recall failure envelope

Task: delayed first-token recall, with one uniformly sampled symbol followed by
independent distractors. Training context: 64 operations. Evaluation contexts:
64, 128, 256, 512, 1,024, 2,048, and 4,096 operations. Use 4,096 fresh
examples per length, five seeds (0–4), and the seven-model primary comparison.

Primary plot: accuracy versus evaluation length on a log-scaled x-axis, with
seed intervals. Primary summary: first length at which accuracy drops below
90%, and area under the accuracy-vs-log-length curve. Secondary plot: accuracy
against evaluation/training-length ratio.

This study tests the existing 512-operation result farther out. It does not
declare success merely because SCAR reaches 512.

## 5. Preregistered Study 2B — train/test ratio

Train separate models at 32, 64, and 128 operations. Evaluate each at 1×, 2×,
4×, 8×, and 16× its training length, capped at 4,096. Reuse the five-seed
64-operation runs from Study 2A; run three new seeds (0–2) for 32 and 128.
Use GRU, Transformer, RLT-lite, SCAR, SCAR-carrier, and SCAR-no-recall.

The primary result is the ratio curve, not an absolute-length leaderboard.
Training length is part of the experiment ID, so results from different train
lengths cannot collide.

## 6. Preregistered Study 2C — associative key/value recall

Each example contains `n_pairs` key/value pairs followed by a query key. Keys
and values use disjoint token ranges; the query marker is distinct. Pair order
is random, the queried key occurs exactly once, and the answer is the paired
value at the final position. Evaluation uses fresh keys, pair order, and
distractors. No answer token appears in the query marker or key vocabulary.

Primary matrix:

* pairs: 1, 2, 4, 8, 16, 32;
* train at 4 pairs and evaluate up to 32 pairs;
* train at 64 total input tokens and evaluate up to 2,048 tokens where feasible;
* GRU, LSTM, Transformer, RLT-lite, SCAR, SCAR-carrier, SCAR-no-recall;
* three seeds for the exploratory sweep, five seeds for any headline claim.

Report both pair-count extrapolation and token-length extrapolation. This is
the first test of whether SCAR does more than preserve one distinguished token.

## 7. Preregistered Study 2D — selective copy, capacity, and interference

Use a marked-copy task in which 1, 2, 4, 8, 16, or 32 marked symbols are
embedded among distractors. The answer is the marked subsequence in order.
Vary delay and distractor entropy independently. Also include an interference
variant with conflicting marked values and repeated distractors. Accuracy is
exact-sequence accuracy and per-item accuracy.

The primary capacity plot is retained items versus exact/per-item accuracy at
a fixed long delay. The primary interference plot is accuracy versus distractor
count and entropy. Failures are expected evidence, not grounds for changing
the task.

## 8. Preregistered Study 2E — SCAR mechanism

### Slot count

Train SCAR with k = 1, 2, 4, 8, 16, 32, and 64 slots on the single-item recall
and associative-recall probes. Report parameters, persistent state bytes,
training cost, streaming inference cost, and long-horizon accuracy.

### Timescales

Compare learned multi-timescale decays, fixed logarithmic decays, learned
single decay, fixed single decay, and narrower/wider learned ranges. Record
initial and final lambda, half-life, slot norm, and slot correlation.

### Test-time interventions

For trained SCAR models, separately mask fastest slots, slowest slots, shuffle
slots, equalize decays, and inject controlled noise into selected slots. These
are evaluation-only interventions on frozen checkpoints and do not replace
training ablations.

### Usage analysis

When tracing is enabled, record per-slot write magnitude, read attention,
attention entropy, memory norm, and slot correlation by position. Trace files
are stored separately from headline result JSON and are never used to select
successful runs.

## 9. Supervision and curriculum study

Keep curriculum separate from supervision density. For parity and associative
recall where semantics permit, compare final-only, 5%, 10%, 25%, 50%, and 100%
target density, plus the existing 4→8→16→32 curriculum. Use the same model,
seed, optimizer, and training budget across conditions. The result is a curve
of algorithm-discovery rate versus gradient density, not a claim that a sparse
condition is equivalent to a curriculum condition.

## 10. Baselines and richer evaluation

After the primary matrices are complete, add a faithful small state-space or
selective-state baseline only if its implementation and parameter accounting
can be independently checked. A baseline that cannot be reproduced is not
included in the primary claim.

A small TinyStories or WikiText-2 language-modeling study is exploratory. It
must report validation perplexity, context-length extrapolation, persistent
state, and failure cases. Synthetic results must not be generalized to
language-model superiority. If the language study is not completed on Kaggle,
the paper explicitly says so rather than implying coverage.

## 11. Memory-state accounting

For every primary model, report parameter bytes separately from persistent
streaming state bytes. Measure state as the tensors required to continue
inference for one stream after processing a prefix; token-level KV caches are
counted at their actual context length. Plot state bytes versus context length.
SCAR's state should be constant in context length; this claim is verified by
the implementation and measured accounting, not by parameter count.

## 12. Exclusions and reruns

The following are never mixed into primary tables: Study 1 v1 artifacts,
curriculum runs in the fixed-density directory, incomplete Kaggle outputs,
artifacts from a different commit, and exploratory tasks whose protocol was
changed after result inspection. A rerun is permitted only for infrastructure
failure, corrupted artifact, or a predeclared numerical bug. The original JSON
and reason remain archived.

## 13. Analysis and paper rule

The final paper is rewritten after Study 2 analysis. It must state whether the
evidence supports a strong, conditional, mechanistic, methodological, or
negative result. It must preserve Study 1's negative findings and the 3.12×
GRU cost result. It may not claim universal efficiency, language-model
superiority, or significance without the corresponding evidence.
