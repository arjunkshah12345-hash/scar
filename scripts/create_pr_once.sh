#!/usr/bin/env bash
set -euo pipefail

# Backward-compatible name. The real idempotent workflow now also queues the
# merge after CI is green.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "$ROOT/scripts/create_and_merge_pr.sh" "$@"
