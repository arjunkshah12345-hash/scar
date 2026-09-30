# SCAR arXiv submission checklist

Do not upload the paper until every Study 2 family is complete and validated.

- [x] Every Study 2 family has internally consistent artifact/manifest
      commits, and each recorded commit is an ancestor of the published
      `research/v3` head; rerun any family if behavior-changing code differs.
- [x] Every declared family passes `python3 -m study2.validate` with its exact
      expected count; manifests are provenance metadata, not result rows:
      recall-length 35, ratio32 18, ratio128 18, associative-recall 21,
      selective-copy 42, mechanism 33, intervention 5.
- [x] `python3 -m study2.analyze study2_results --out analysis/study2` has
      produced per-seed summaries, bootstrap intervals, intervention summaries,
      and exact-sequence plots where applicable.
- [x] The paper reports per-seed values and uncertainty, but does not call
      cells “statistically tied” without a declared statistical test.
- [x] The 512-token Study 1 recall result, the 3.12x-GRU cost result, and the
      falsified ~1x-GRU hypothesis remain visible alongside Study 2 findings.
- [x] `python3 -m pytest tests/ -q` passes and `git diff --check` is clean.
- [x] `paper/paper.tex` compiles with `tectonic`; all referenced figures and
      bibliography entries are present.
- [x] `python3 make_paper.py` reproduces the committed `paper/paper.tex` and
      figures from the validated artifacts without modifying the worktree.
- [x] The clean arXiv package in `arxiv/` independently compiles with
      `tectonic paper.tex` and contains only `paper.tex`, `README.md`,
      `figures/`, and the compiled verification PDF; it contains no checkpoints,
      local training directories, browser data, or temporary Kaggle logs.
- [x] Upload `paper.tex`, `figures/`, and bibliography/source files only. Do
      not upload checkpoints, local training directories, browser data, or
      temporary Kaggle logs.

## Final release record

- Title: `SCAR: Constant-Size Learned Memory for Length Extrapolation`
- Subtitle: `Controlled tests of recall, capacity, and memory mechanisms`
- Study 2 evidence: 172 validated runs across seven families, with provenance
  recorded in `study2_results/` and generated summaries in `analysis/study2/`.
- Paper PDF: `paper/paper.pdf` (15 pages, independently checked visually).
- arXiv source package: `arxiv/`.
- Recommended categories: `cs.LG` primary; `stat.ML` secondary if desired.
- Rebuild commands: `python3 make_paper.py`,
  `(cd paper && SOURCE_DATE_EPOCH=0 tectonic paper.tex)`, and
  `./scripts/build_arxiv_package.sh` (the release scripts pin the PDF epoch for
  reproducible generated files).

## Submission metadata

- Author: Arjun K. Shah.
- Affiliation: no institution is claimed in the repository; leave the
  affiliation blank or enter the author's truthful current affiliation at
  submission. Do not invent an institutional affiliation.
- Primary category: `cs.LG`.
- Suggested cross-list: `stat.ML`, only if the author wants the statistical
  learning audience and the arXiv form permits it.
- Comments field: 15 pages; code and compact machine-readable results are
  available at `https://github.com/arjunkshah12345-hash/scar`.
- Endorsement: a first-time `cs.LG` submitter may need an arXiv endorsement;
  check the account's endorsement status before starting submission.

### Abstract

Can a recurrent model preserve useful information beyond its training context
without retaining a token-level memory? We study SCAR (State-Carrier with
Attentive Recall), which combines a GRU carrier with a constant-size bank of
16 learned-decay exponential slots and an attentive read head. The corrected
Study 1 release contains 135 matched runs; the cloud-only Study 2 matrix adds
172 preregistered stress-test runs spanning length extrapolation, train/test
ratios, associative recall, selective copying, slot/decay mechanisms, and
frozen-memory interventions. The central result is not an across-the-board
accuracy win: in-distribution recall hides a separation that appears at long
horizons, while multi-item retrieval and perturbation tests reveal capacity and
robustness limits. The dedicated CPU benchmark falsifies the original
approximately one-times-GRU cost hypothesis: SCAR is slower than a GRU but
faster than RLT-lite in the measured setting. The result is a controlled
synthetic study of when bounded learned memory helps, and when it does not.
In the corrected Study 1 delayed-recall probe, SCAR reaches 100.0% at 512
operations, compared with 56.6% for the carrier-only ablation and 30.5% when
the memory is written but not read. In the preregistered follow-up, the same
model is evaluated at lengths through 4,096, reaches 72.8% +/- 13.8% at the
4,096-operation endpoint, and reaches 11.4% +/- 0.6% at 32 key/value pairs;
selective-copy and intervention sweeps expose capacity and perturbation limits.

### Manual submission steps

1. Confirm the author name and truthful affiliation in the arXiv form.
2. Upload the source contents of `arxiv/` (`paper.tex`, `README.md`, and
   `figures/`); `arxiv/paper.pdf` is only the local compile verification.
3. Select `cs.LG`, optionally cross-list to `stat.ML`, and review the generated
   PDF and metadata preview.
4. Add the repository URL in the comments or abstract-adjacent project field
   if available, then submit manually. No automated arXiv submission is run by
   this repository.
