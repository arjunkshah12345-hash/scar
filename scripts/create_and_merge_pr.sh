#!/usr/bin/env bash
set -euo pipefail

# Idempotently create one PR and queue its merge.
# No training, comments, retries, or safety bypasses.
#
# Usage:
#   ./scripts/create_and_merge_pr.sh
#
# Optional:
#   BASE_BRANCH=main HEAD_BRANCH=release/scientific-integrity-pass \
#     RUN_GITHUB_WORKER=0 ./scripts/create_and_merge_pr.sh

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_BRANCH="${BASE_BRANCH:-main}"
HEAD_BRANCH="${HEAD_BRANCH:-$(git -C "$ROOT" branch --show-current)}"
PR_TITLE="${PR_TITLE:-docs: record Study 2 primary-release scope}"
RUN_GITHUB_WORKER="${RUN_GITHUB_WORKER:-1}"
MERGE_METHOD="${MERGE_METHOD:-squash}"

cd "$ROOT"

die() {
  echo "error: $*" >&2
  exit 1
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  sed -n '1,20p' "$0"
  exit 0
fi

for command_name in git jq github-safety gh-safe; do
  command -v "$command_name" >/dev/null || die "$command_name is required"
done

[[ "$HEAD_BRANCH" != "$BASE_BRANCH" ]] || die "HEAD_BRANCH must differ from BASE_BRANCH"
[[ "$(git branch --show-current)" == "$HEAD_BRANCH" ]] \
  || die "checkout $HEAD_BRANCH first"
dirty="$(git status --porcelain --untracked-files=all -- \
  ':(exclude)scripts/create_pr_once.sh')"
[[ -z "$dirty" ]] \
  || { printf '%s\n' "$dirty" >&2; die "working tree is dirty; commit validated changes first"; }

git fetch origin "$BASE_BRANCH" --quiet
if git diff --quiet "origin/$BASE_BRANCH...HEAD"; then
  die "branch has no changes relative to origin/$BASE_BRANCH"
fi

remote_url="$(git remote get-url origin)"
repo="${remote_url#https://github.com/}"
repo="${repo#http://github.com/}"
repo="${repo#git@github.com:}"
repo="${repo%.git}"
[[ "$repo" == */* ]] || die "could not determine repository from origin"

safety_json="$(github-safety status)"
safety_root="$(jq -r '.root' <<<"$safety_json")"
[[ "$(jq -r '.silenceActive' <<<"$safety_json")" != "true" ]] \
  || die "GitHub safety silence is active"

pending_pr_id() {
  local queue_file="$safety_root/pr-queue.json"
  [[ -f "$queue_file" ]] || return 0
  jq -r --arg repo "$repo" --arg base "$BASE_BRANCH" --arg head "$HEAD_BRANCH" '
    [.[] | select(
      .type == "pr_create" and
      (.status == "queued" or .status == "processing") and
      .repo == $repo and
      (.base // "main") == $base and
      .head == $head
    )] | .[0].id // empty
  ' "$queue_file"
}

pending_merge_id() {
  local queue_file="$safety_root/merge-queue.json"
  [[ -f "$queue_file" ]] || return 0
  jq -r --arg repo "$repo" --arg pr "$1" '
    [.[] | select(
      .type == "pr_merge" and
      (.status == "queued" or .status == "processing") and
      .repo == $repo and
      (.pr | tostring) == $pr
    )] | .[0].id // empty
  ' "$queue_file"
}

pr_json="$(gh-safe pr list --repo "$repo" --base "$BASE_BRANCH" --head "$HEAD_BRANCH" \
  --state all --json number,state,url)"
pr_number="$(jq -r '.[0].number // empty' <<<"$pr_json")"
pr_state="$(jq -r '.[0].state // empty' <<<"$pr_json")"

if [[ -n "$pr_number" && "$pr_state" != "OPEN" ]]; then
  die "matching PR #$pr_number is $pr_state; refusing a replacement"
fi

if [[ -z "$pr_number" ]]; then
  queued_pr="$(pending_pr_id)"
  if [[ -n "$queued_pr" ]]; then
    echo "PR creation already queued: $queued_pr"
  else
    body_file="$(mktemp -t scar-pr-body.XXXXXX)"
    cat >"$body_file" <<'EOF'
## Summary

- Publishes the final SCAR scientific-integrity wording pass.
- Describes the 172 artifacts as the completed primary Study 2 release.
- Records unrun preregistered secondary conditions explicitly.
- Regenerates the paper and independently compiling arXiv package.

## Verification

- All seven exact Study 2 validators pass: 172 artifacts total.
- python3 -m pytest tests/ -q: 38 passed.
- No optimizer steps were run locally; training remains Kaggle-only.
EOF
    github-safety queue pr \
      --repo "$repo" \
      --title "$PR_TITLE" \
      --body-file "$body_file" \
      --base "$BASE_BRANCH" \
      --head "$HEAD_BRANCH"
  fi

  if [[ "$RUN_GITHUB_WORKER" == "1" ]]; then
    github-safety worker --once
  fi

  pr_json="$(gh-safe pr list --repo "$repo" --base "$BASE_BRANCH" --head "$HEAD_BRANCH" \
    --state all --json number,state,url)"
  pr_number="$(jq -r '.[0].number // empty' <<<"$pr_json")"
  if [[ -z "$pr_number" ]]; then
    echo "PR is queued; merge will be attempted after GitHub creates it and CI runs."
    exit 0
  fi
fi

pr_view="$(gh-safe pr view "$pr_number" --repo "$repo" \
  --json state,mergeable,mergeStateStatus,url)"
mergeable="$(jq -r '.mergeable' <<<"$pr_view")"
merge_state="$(jq -r '.mergeStateStatus' <<<"$pr_view")"
pr_url="$(jq -r '.url' <<<"$pr_view")"

if [[ "$mergeable" != "MERGEABLE" || "$merge_state" != "CLEAN" ]]; then
  echo "PR $pr_url is not merge-ready: $mergeable / $merge_state"
  exit 0
fi

checks_json="$(gh-safe pr checks "$pr_number" --repo "$repo" \
  --json name,state,bucket)"
check_count="$(jq 'length' <<<"$checks_json")"
bad_checks="$(jq -r '[.[] | select(.bucket != "pass") | .name] | join(", ")' <<<"$checks_json")"
if (( check_count == 0 )); then
  echo "No CI checks reported; merge not queued."
  exit 0
fi
if [[ -n "$bad_checks" ]]; then
  echo "Checks are not all green: $bad_checks"
  exit 0
fi

queued_merge="$(pending_merge_id "$pr_number")"
if [[ -n "$queued_merge" ]]; then
  echo "Merge already queued: $queued_merge"
else
  github-safety queue merge --repo "$repo" --pr "$pr_number" --method "$MERGE_METHOD"
fi

if [[ "$RUN_GITHUB_WORKER" == "1" ]]; then
  github-safety worker --once
fi

echo "Create-and-merge workflow submitted for PR #$pr_number: $pr_url"
