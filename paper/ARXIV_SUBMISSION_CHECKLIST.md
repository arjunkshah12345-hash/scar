# SCAR arXiv submission checklist

Do not upload the paper until every Study 2 family is complete and validated.

- [ ] `study2_results/` contains only artifacts whose `git_commit` matches the
      frozen `research/v3` release commit.
- [ ] Every declared family passes `python3 -m study2.validate` with its exact
      expected count; manifests are provenance metadata, not result rows.
- [ ] `python3 -m study2.analyze study2_results --out analysis/study2` has
      produced per-seed summaries, bootstrap intervals, and exact-sequence
      plots where applicable.
- [ ] The paper reports per-seed values and uncertainty, but does not call
      cells “statistically tied” without a declared statistical test.
- [ ] The 512-token Study 1 recall result, the 3.12x-GRU cost result, and the
      falsified ~1x-GRU hypothesis remain visible alongside Study 2 findings.
- [ ] `python3 -m pytest tests/ -q` passes and `git diff --check` is clean.
- [ ] `paper/paper.tex` compiles with `tectonic`; all referenced figures and
      bibliography entries are present.
- [ ] Upload `paper.tex`, `figures/`, and bibliography/source files only. Do
      not upload checkpoints, local training directories, browser data, or
      temporary Kaggle logs.
