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
