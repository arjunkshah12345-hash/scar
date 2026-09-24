#!/usr/bin/env bash
set -euo pipefail

# Publish the completed, cloud-trained Study 2 branch as a PR to main.
# This script never trains locally. It validates collected Kaggle artifacts,
# runs tests/analysis, pushes the branch, and queues a PR through github-safety.
#
# Usage:
#   ./scripts/publish_study2_pr.sh
#   RUN_GITHUB_WORKER=1 ./scripts/publish_study2_pr.sh
#
# A clean tree is required: commit the validated result JSONs and generated
# paper/analysis outputs before running this helper.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_BRANCH="${BASE_BRANCH:-main}"
RELEASE_BRANCH="${RELEASE_BRANCH:-research/v3}"
PR_TITLE="research: publish SCAR Study 2 protocol and results"
DEST="${DEST:-study2_results}"

cd "$ROOT"

die() {
  echo "error: $*" >&2
  exit 1
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  sed -n '1,24p' "$0"
  exit 0
fi

command -v git >/dev/null || die "git is required"
command -v python3 >/dev/null || die "python3 is required"
command -v github-safety >/dev/null || die "github-safety is required"
command -v gh-safe >/dev/null || die "gh-safe is required"

[[ -f EXPERIMENT_PROTOCOL_V3.md ]] || die "run this from the SCAR checkout"
[[ -d "$DEST" ]] || die "missing $DEST; collect Kaggle outputs first"

if [[ -n "$(git status --porcelain)" ]]; then
  git status --short >&2
  die "working tree is dirty; commit the validated Study 2 release first"
fi

git fetch origin "$BASE_BRANCH" --quiet
current_branch="$(git branch --show-current)"
[[ "$current_branch" == "$RELEASE_BRANCH" ]] \
  || die "checkout $RELEASE_BRANCH before publishing (currently $current_branch)"

declare -a FAMILIES=(
  "recall_length:35"
  "associative_recall:21"
  "ratio32:18"
  "ratio128:18"
  "mechanism:33"
  "intervention:5"
  "selective_copy:42"
)

for spec in "${FAMILIES[@]}"; do
  IFS=: read -r family expected <<<"$spec"
  path="$DEST/$family"
  [[ -d "$path" ]] || die "missing Study 2 family: $path"
  python3 -m study2.validate "$path" --expected-count "$expected"
done

python3 -m pytest tests/ -q
python3 -m study2.analyze "$DEST" --out analysis/study2
python3 study2/state_memory.py --out analysis/study2/state_memory.json
git diff --check "origin/$BASE_BRANCH...HEAD"

# Refuse accidental local training. The normal benchmark guard exits before
# any optimizer step outside Kaggle; this check verifies the guard remains in
# place without invoking a training run.
rg -q 'Refusing local training' run_all.py \
  || die "local-training guard is missing from run_all.py"
rg -q 'KAGGLE_KERNEL_RUN_TYPE|/kaggle/working' bench.py \
  || die "cloud-only training guard is missing from bench.py"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "Validation complete; dry run did not push or queue a PR."
  exit 0
fi

git push --set-upstream origin "$RELEASE_BRANCH"

safety_status="$(github-safety status)"
python3 -c 'import json, sys; s=json.load(sys.stdin); raise SystemExit("GitHub safety silence is active") if s.get("silenceActive") else None' <<<"$safety_status"

remote_url="$(git remote get-url origin)"
repo="${remote_url#https://github.com/}"
repo="${repo#http://github.com/}"
repo="${repo#git@github.com:}"
repo="${repo%.git}"
[[ "$repo" == */* ]] || die "could not determine GitHub repository from origin: $remote_url"

existing_url="$(gh-safe pr list --repo "$repo" --base "$BASE_BRANCH" --head "$RELEASE_BRANCH" \
  --state open --json url --jq '.[0].url // empty' 2>/dev/null || true)"
if [[ -n "$existing_url" ]]; then
  echo "Open PR already exists: $existing_url"
  exit 0
fi

pr_body="$(mktemp -t scar-study2-pr-body.XXXXXX)"
trap 'rm -f "$pr_body"' EXIT
cat >"$pr_body" <<EOF
## Summary

- Publishes the frozen SCAR Study 2 protocol and cloud-collected results.
- Keeps the seven Study 2 families separated by experiment identity.
- Includes provenance manifests, validation, analysis, paper updates, and the
  local-training guard.

## Verification

- All expected Kaggle artifacts validate with matching commit provenance.
- `python3 -m pytest tests/ -q`
- `python3 -m study2.analyze study2_results --out analysis/study2`
- `python3 study2/state_memory.py --out analysis/study2/state_memory.json`
- No optimizer steps are run locally; training is Kaggle-only.
EOF

queue_result="$(github-safety queue pr \
  --repo "$repo" \
  --title "$PR_TITLE" \
  --body-file "$pr_body" \
  --base "$BASE_BRANCH" \
  --head "$RELEASE_BRANCH")"
echo "$queue_result"

if [[ "${RUN_GITHUB_WORKER:-0}" == "1" ]]; then
  github-safety worker --once
else
  echo "PR is queued. To let the safety worker create it now, run:"
  echo "  RUN_GITHUB_WORKER=1 $0"
fi
