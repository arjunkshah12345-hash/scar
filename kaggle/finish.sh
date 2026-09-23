#!/bin/bash
# Finalize: run after kaggle/collect.sh has pulled the sweep results.
cd "$(dirname "$0")/.." || exit 1
set -e
python3 -m pytest tests/ -q
python3 aggregate.py
python3 make_charts.py
python3 build_site.py
python3 make_paper.py
(cd paper && tectonic paper.tex)
git add -A
git commit -m "v2 sweep results + regenerated summary, charts, site, paper (corrected harness)"
git push origin main
echo DONE
