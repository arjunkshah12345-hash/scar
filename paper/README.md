# arXiv package

`paper.tex` is generated from the checked-in JSON artifacts by running
`python3 make_paper.py` from the repository root after all Study 2 families
have been collected and analyzed. The generated figures live in
`paper/figures/` and are copied from `charts/` and `analysis/study2/`.

For an arXiv source upload, include `paper.tex` and every referenced file in
`figures/`. `paper.pdf` is the local compiled preview. No model checkpoints or
training datasets are required: all optimizer steps ran on Kaggle CPU kernels
and the repository retains compact JSON results instead.

Compile locally with:

```sh
cd paper
tectonic paper.tex
```
