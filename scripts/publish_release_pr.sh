#!/usr/bin/env bash
set -euo pipefail

# Prepare the final SCAR release branch, apply the curriculum-output routing
# fix if it is missing, run the release contract, push the branch, and queue a
# GitHub PR. No training is performed locally.
#
# Usage:
#   ./scripts/publish_release_pr.sh
#   RUN_GITHUB_WORKER=1 ./scripts/publish_release_pr.sh
#
# Override the defaults when needed:
#   BASE_BRANCH=main RELEASE_BRANCH=release/final-v2 \
#     ./scripts/publish_release_pr.sh

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_BRANCH="${BASE_BRANCH:-main}"
RELEASE_BRANCH="${RELEASE_BRANCH:-release/final-v2}"
PR_TITLE="release: separate curriculum artifact routing"

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

[[ -f run_all.py ]] || die "run this from the SCAR checkout"

if [[ -n "$(git status --porcelain)" ]]; then
  git status --short >&2
  die "working tree is dirty; commit or stash your changes first"
fi

git fetch origin "$BASE_BRANCH" --quiet

# Use the existing release branch. Do not silently manufacture a release from
# main: the branch must already contain the reviewed release bundle.
current_branch="$(git branch --show-current)"
if [[ "$current_branch" != "$RELEASE_BRANCH" ]]; then
  if git show-ref --verify --quiet "refs/heads/$RELEASE_BRANCH"; then
    git switch "$RELEASE_BRANCH"
  else
    git fetch origin "$RELEASE_BRANCH" --quiet \
      || die "missing origin/$RELEASE_BRANCH; create the reviewed release branch first"
    git switch --track "origin/$RELEASE_BRANCH"
  fi
fi

if [[ -n "$(git status --porcelain)" ]]; then
  die "release branch is dirty after checkout"
fi

# This is the one code fix required on top of f084286. Keep it idempotent so
# the script is safe to run against a branch that already has c2010a9.
if ! rg -q '^def output_dir\(config\):' run_all.py; then
  git apply --unidiff-zero <<'PATCH'
diff --git a/run_all.py b/run_all.py
--- a/run_all.py
+++ b/run_all.py
@@ -30,1 +30,6 @@
-    return f"{model}_{task}_{density}{suffix}_seed{seed}"
+    return f"{model}_{task}_{density}{suffix}_seed{seed}"
+
+def output_dir(config):
+    return "results_curriculum" if CONFIGS[config][2] else "results"
+
+
@@ -35,1 +40,2 @@
-    out = os.path.join("results", f"{tag}.json")
+    out_dir = output_dir(config)
+    out = os.path.join(out_dir, f"{tag}.json")
@@ -43,1 +48,1 @@
-           "--device", "cpu", "--out", "results"]
+           "--device", "cpu", "--out", out_dir]
@@ -68,1 +73,2 @@
-    os.makedirs("results", exist_ok=True)
+    for out_dir in {output_dir(config) for config in selected}:
+        os.makedirs(out_dir, exist_ok=True)
PATCH
fi

# Release branches before c2010a9 have the test file but not these two route
# assertions. Add them independently so this remains idempotent.
if ! rg -q 'run_all\.output_dir\("parity_sparse"\)' tests/test_release.py; then
  git apply --unidiff-zero <<'PATCH'
diff --git a/tests/test_release.py b/tests/test_release.py
--- a/tests/test_release.py
+++ b/tests/test_release.py
@@ -25,1 +25,3 @@
-    assert run_all.result_tag("parity_sparse_curriculum", "scar", 0) == "scar_parity_sparse_curriculum_seed0"
+    assert run_all.result_tag("parity_sparse_curriculum", "scar", 0) == "scar_parity_sparse_curriculum_seed0"
+    assert run_all.output_dir("parity_sparse") == "results"
+    assert run_all.output_dir("parity_sparse_curriculum") == "results_curriculum"
PATCH
fi

rg -q '^def output_dir\(config\):' run_all.py \
  || die "routing fix was not applied"
rg -q 'run_all\.output_dir\("parity_sparse_curriculum"\) == "results_curriculum"' tests/test_release.py \
  || die "curriculum routing regression test is missing"

git diff --check
python3 -m pytest tests/test_release.py -q

# This intentionally exits before training because run_all.py must refuse
# local execution. It is a guard check, not a training invocation.
guard_log="$(mktemp -t scar-local-training-guard.XXXXXX)"
pr_body=""
trap 'rm -f "$guard_log" "$pr_body"' EXIT
if python3 run_all.py --configs parity_sparse_curriculum --workers 1 >"$guard_log" 2>&1; then
  cat "$guard_log" >&2
  die "run_all.py did not refuse local training"
fi
rg -q 'Refusing local training' "$guard_log" \
  || { cat "$guard_log" >&2; die "local-training guard produced an unexpected error"; }

if ! git diff --quiet -- run_all.py tests/test_release.py; then
  git add run_all.py tests/test_release.py
  git commit -m "fix: route curriculum artifacts to dedicated directory"
fi

if git diff --quiet "origin/$BASE_BRANCH" HEAD; then
  echo "Release tree already matches origin/$BASE_BRANCH; no PR is needed."
  exit 0
fi

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "Dry run complete; branch is ready but was not pushed and no PR was queued."
  git status --short
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

pr_body="$(mktemp -t scar-pr-body.XXXXXX)"
cat >"$pr_body" <<EOF
## Summary

- Routes curriculum results to `results_curriculum/` instead of contaminating `results/`.
- Keeps the five-condition release contract at 135 fixed-length runs.
- Adds a regression assertion for both output directories.

## Verification

- `python3 -m pytest tests/test_release.py -q`
- `git diff --check`
- Local `run_all.py` training guard verified; training remains Kaggle-only.
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
