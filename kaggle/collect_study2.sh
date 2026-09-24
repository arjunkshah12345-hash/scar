#!/usr/bin/env bash
set -euo pipefail

# Collect Study 2 artifacts from Kaggle as kernels finish. This script only
# downloads JSON/log outputs and runs local validation; it never trains.
# Run from the repository root:
#   bash kaggle/collect_study2.sh

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

POLL_SECONDS="${POLL_SECONDS:-60}"
MAX_POLLS="${MAX_POLLS:-720}"
DEST="${DEST:-study2_results}"

declare -a KERNELS=(
  "scar-v3-recall-length:recall_length:35"
  "scar-v3-associative-recall:associative_recall:21"
  "scar-v3-ratio32:ratio32:18"
  "scar-v3-ratio128:ratio128:18"
  "scar-v3-mechanism:mechanism:33"
  "scar-v3-intervention:intervention:5"
  "scar-v3-selective-copy:selective_copy:42"
)

collect_one() {
  local kernel="$1"
  local family="$2"
  local expected="$3"
  local tmp="/tmp/scar-v3-output-${family}"
  local dest="$DEST/$family"
  mkdir -p "$tmp" "$dest"
  kaggle kernels output "aks1321/$kernel" -p "$tmp" --force >/dev/null 2>&1 || true
  find "$tmp" -type f -name 'v3*.json' -exec cp {} "$dest"/ \;
  local count
  count="$(find "$dest" -maxdepth 1 -type f -name 'v3*.json' | wc -l | tr -d ' ')"
  echo "$family: $count/$expected artifacts"
  if [[ "$count" == "$expected" ]]; then
    python3 -m study2.validate "$dest" --expected-count "$expected"
    return 0
  fi
  return 1
}

for poll in $(seq 1 "$MAX_POLLS"); do
  complete=0
  for spec in "${KERNELS[@]}"; do
    IFS=: read -r kernel family expected <<<"$spec"
    status="$(kaggle kernels status "aks1321/$kernel" 2>&1 | tail -n 1)"
    echo "[$(date +%H:%M)] $family: $status"
    if [[ "$status" == *COMPLETE* || "$status" == *ERROR* ]]; then
      if collect_one "$kernel" "$family" "$expected"; then
        complete=$((complete + 1))
      fi
    fi
  done
  if [[ "$complete" -eq "${#KERNELS[@]}" ]]; then
    python3 -m study2.analyze "$DEST" --out analysis/study2
    echo "Study 2 collection and analysis complete."
    exit 0
  fi
  [[ "$poll" -lt "$MAX_POLLS" ]] || break
  sleep "$POLL_SECONDS"
done

echo "Study 2 collection is incomplete; leaving partial artifacts for the next poll." >&2
exit 1
