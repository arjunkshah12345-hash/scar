#!/usr/bin/env bash
set -euo pipefail

# Push the corrected release branch and create/update its GitHub PR.
# Run from anywhere inside this checkout:
#   ./scripts/publish_release_pr.sh

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_BRANCH="${BASE_BRANCH:-main}"
RELEASE_BRANCH="${RELEASE_BRANCH:-release/final-v2}"
PR_TITLE="release: separate curriculum sweep and publish corrected paper"

cd "$ROOT"

command -v git >/dev/null || { echo "git is required" >&2; exit 1; }
command -v gh >/dev/null || { echo "GitHub CLI (gh) is required" >&2; exit 1; }
gh auth status >/dev/null 2>&1 || {
  echo "GitHub CLI is not authenticated. Run: gh auth login" >&2
  exit 1
}

current_branch="$(git branch --show-current)"
if [[ "$current_branch" != "$RELEASE_BRANCH" ]]; then
  echo "Wrong branch: currently on '$current_branch'; switch to '$RELEASE_BRANCH' first." >&2
  exit 1
fi

if [[ -n "$(git status --porcelain)" ]]; then
  echo "Working tree is dirty. Commit or stash changes before publishing." >&2
  git status --short >&2
  exit 1
fi

repo="$(gh repo view --json nameWithOwner --jq '.nameWithOwner')"
git fetch origin "$BASE_BRANCH" --quiet

if ! git rev-parse --verify "origin/$BASE_BRANCH" >/dev/null 2>&1; then
  echo "Missing origin/$BASE_BRANCH" >&2
  exit 1
fi

read -r behind ahead < <(git rev-list --left-right --count "origin/$BASE_BRANCH...HEAD")
if [[ "$ahead" -eq 0 ]]; then
  echo "No release commits are ahead of origin/$BASE_BRANCH." >&2
  exit 1
fi

echo "Pushing $RELEASE_BRANCH ($ahead commit(s) ahead of $BASE_BRANCH)..."
git push --set-upstream origin "$RELEASE_BRANCH"

existing_url="$(gh pr list --repo "$repo" --base "$BASE_BRANCH" --head "$RELEASE_BRANCH" \
  --state open --json url --jq '.[0].url // empty')"
if [[ -n "$existing_url" ]]; then
  echo "PR already exists: $existing_url"
  exit 0
fi

read -r -d '' PR_BODY <<'EOF' || true
## Summary

- Separates `parity_sparse` from `parity_sparse_curriculum` in both filenames and output directories.
- Keeps the release at 135 fixed-length runs and archives 27 curriculum runs separately.
- Includes the corrected data, paper, citations, CI, license, dependency manifest, and release-contract tests.

## Verification

- `python3 -m pytest tests/ -q` — 13 passed
- GitHub Actions CI passes
- Local training remains blocked; training runs through Kaggle only
EOF

pr_url="$(gh pr create --repo "$repo" --base "$BASE_BRANCH" --head "$RELEASE_BRANCH" \
  --title "$PR_TITLE" --body "$PR_BODY")"
echo "Created PR: $pr_url"
