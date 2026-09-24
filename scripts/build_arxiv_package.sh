#!/usr/bin/env bash
set -euo pipefail

# Build the minimal, self-contained arXiv source directory from the committed
# paper. This performs no training and refuses to package an incomplete paper.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/arxiv"
cd "$ROOT"

command -v tectonic >/dev/null || {
  echo "error: tectonic is required" >&2
  exit 1
}
[[ -f paper/paper.tex ]] || { echo "error: paper/paper.tex is missing" >&2; exit 1; }
[[ -f paper/paper.pdf ]] || { echo "error: compile paper/paper.tex first" >&2; exit 1; }
rg -q '\\begin\{thebibliography\}' paper/paper.tex \
  || { echo "error: bibliography is missing" >&2; exit 1; }
rg -q 'Study 2' paper/paper.tex \
  || { echo "error: final paper does not contain the completed Study 2 section" >&2; exit 1; }

mkdir -p "$OUT/figures"
find "$OUT/figures" -mindepth 1 -maxdepth 1 -type f -delete
find "$OUT" -maxdepth 1 -type f \( -name '*.tex' -o -name '*.pdf' -o -name '*.bib' -o -name '*.sty' \) -delete

cp paper/paper.tex "$OUT/paper.tex"
cp paper/README.md "$OUT/README.md"
cp paper/figures/* "$OUT/figures/"

(cd "$OUT" && tectonic paper.tex >/dev/null)
echo "built $OUT (source and independently compiled PDF)"
