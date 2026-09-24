# arXiv package

`paper.tex` is generated from the checked-in JSON artifacts by running
`python3 make_paper.py` from the repository root. The generated figures live in
`paper/figures/` and are copied from `charts/`.

For an arXiv source upload, include `paper.tex` and the four files in
`figures/`. `paper.pdf` is the local compiled preview. No model checkpoints or
training datasets are required: the sweep ran on Kaggle CPU kernels and the
repository retains compact JSON results instead.

Compile locally with:

```sh
cd paper
tectonic paper.tex
```
