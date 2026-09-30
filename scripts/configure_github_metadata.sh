#!/usr/bin/env bash
set -euo pipefail

# One-time public GitHub page polish. This performs a direct repository
# metadata write, so it is intentionally opt-in and is not run by Codex.
#
# Usage:
#   RUN_GITHUB_METADATA=1 ./scripts/configure_github_metadata.sh

REPO="${REPO:-arjunkshah12345-hash/scar}"
DESCRIPTION="SCAR: State-Carrier with Attentive Recall for constant-size long-context memory"
HOMEPAGE="https://github.com/${REPO}/tree/main/paper"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  sed -n '1,18p' "$0"
  exit 0
fi

command -v gh >/dev/null || { echo "error: gh is required" >&2; exit 1; }

if [[ "${RUN_GITHUB_METADATA:-0}" != "1" ]]; then
  cat >&2 <<'EOF'
This script performs a direct GitHub repository-metadata write.
Review it, then run:

  RUN_GITHUB_METADATA=1 ./scripts/configure_github_metadata.sh
EOF
  exit 2
fi

gh repo edit "$REPO" \
  --description "$DESCRIPTION" \
  --homepage "$HOMEPAGE" \
  --add-topic sequence-modeling \
  --add-topic recurrent-neural-networks \
  --add-topic long-context \
  --add-topic machine-learning

echo "Updated GitHub description, homepage, and topics for $REPO"
