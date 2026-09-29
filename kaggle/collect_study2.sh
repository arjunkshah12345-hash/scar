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
  # Recall-length is split by model after the monolithic 35-run kernel hit
  # Kaggle's runtime ceiling. The final family contract remains 7 models x 5
  # seeds = 35 artifacts.
  "scar-v3-recall-length-gru:recall_length:5:35"
  "scar-v3-recall-length-lstm:recall_length:5:35"
  "scar-v3-recall-length-transformer-seed0:recall_length:1:35"
  "scar-v3-recall-length-transformer-seed1:recall_length:1:35"
  "scar-v3-recall-length-transformer-seed2:recall_length:1:35"
  "scar-v3-recall-length-transformer-seed3:recall_length:1:35"
  "scar-v3-recall-length-transformer-seed4:recall_length:1:35"
  "scar-v3-recall-length-rlt-seed0:recall_length:1:35"
  "scar-v3-recall-length-rlt-seed1:recall_length:1:35"
  "scar-v3-recall-length-rlt-seed2:recall_length:1:35"
  "scar-v3-recall-length-rlt-seed3:recall_length:1:35"
  "scar-v3-recall-length-rlt-seed4:recall_length:1:35"
  "scar-v3-recall-length-scar:recall_length:5:35"
  "scar-v3-recall-length-scar-carrier:recall_length:5:35"
  "scar-v3-recall-length-scar-norecall:recall_length:5:35"
  "scar-v3-associative-recall:associative_recall:21:21"
  "scar-v3-ratio32:ratio32:18:18"
  "scar-v3-ratio128:ratio128:18:18"
  "scar-v3-mechanism:mechanism:33:33"
  "scar-v3-intervention:intervention:5:5"
  "scar-v3-selective-copy:selective_copy:42:42"
)

finalize_recall_manifest() {
  local dest="$DEST/recall_length"
  python3 - "$dest" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
paths = sorted(p for p in root.glob("v3*.json"))
if len(paths) != 35:
    raise SystemExit(f"recall_length: found {len(paths)} artifacts, expected 35")
rows = [json.loads(path.read_text()) for path in paths]
first = rows[0]
manifest = {
    "study": "study2",
    "protocol_version": first["protocol_version"],
    "experiment_family": "v3A_recall_length",
    "git_commit": first["git_commit"],
    "models": sorted({row["model"] for row in rows}),
    "seeds": sorted({row["seed"] for row in rows}),
    "train_ops": first["task_parameters"]["train_ops"],
    "eval_lengths": first["eval_contexts"],
    "eval_examples": first["eval_examples"],
    "expected_count": len(rows),
}
(root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
PY
}

collect_one() {
  local kernel="$1"
  local family="$2"
  local expected="$3"
  local final_expected="$4"
  local status="$5"
  local tmp="/tmp/scar-v3-output-${kernel//\//-}-$$"
  local marker="/tmp/scar-v3-collected-${kernel//\//-}-$$"
  local dest="$DEST/$family"
  local split=0
  [[ "$expected" != "$final_expected" ]] && split=1
  rm -rf "$tmp"
  mkdir -p "$tmp" "$dest"
  local count
  count="$(find "$dest" -maxdepth 1 -type f -name 'v3*.json' | wc -l | tr -d ' ')"
  if [[ "$status" == *COMPLETE* && -f "$marker" ]]; then
    if [[ "$split" -eq 0 ]]; then
      python3 -m study2.validate "$dest" --expected-count "$final_expected"
    fi
    echo "$family/$kernel: $count/$final_expected artifacts already collected this poll run"
    return 0
  fi
  # A failed kernel may still have produced a complete, validated family
  # before failing during notebook packaging. Reuse that evidence. A newly
  # COMPLETE kernel, however, must refresh the destination: otherwise a clean
  # rerun can be silently ignored when the old family already has the same
  # expected count.
  if [[ "$split" -eq 0 && "$status" == *ERROR* && "$count" == "$expected" ]]; then
    python3 -m study2.validate "$dest" --expected-count "$final_expected"
    echo "$family: $count/$final_expected artifacts already collected"
    touch "$marker"
    return 0
  fi
  kaggle kernels output "aks1321/$kernel" -p "$tmp" --force >/dev/null 2>&1 || true
  local selected_count
  if [[ "$family" == "mechanism" ]]; then
    selected_count="$(find "$tmp" -type f \( \
      -path "*/study2-results/slot_sweep/v3*.json" -o \
      -path "*/study2-results/decay_sweep/v3*.json" \
    \) | wc -l | tr -d ' ')"
  else
    selected_count="$(find "$tmp" -type f \
      -path "*/study2-results/$family/v3*.json" | wc -l | tr -d ' ')"
  fi
  if [[ "$status" == *COMPLETE* && "$selected_count" != "$expected" ]]; then
    echo "$family: COMPLETE kernel returned $selected_count/$expected family artifacts" >&2
    return 1
  fi
  if [[ "$status" == *COMPLETE* && "$split" -eq 0 ]]; then
    # Replace, rather than append to, a destination from an earlier failed
    # collection attempt.
    find "$dest" -maxdepth 1 -type f -name 'v3*.json' -delete
  fi
  if [[ "$family" == "mechanism" ]]; then
    # The mechanism driver exposes two output subfamilies in the Kaggle
    # bundle; never copy the repository checkout or unrelated family files.
    find "$tmp" -type f \( \
      -path "*/study2-results/slot_sweep/v3*.json" -o \
      -path "*/study2-results/decay_sweep/v3*.json" \
    \) -exec cp {} "$dest"/ \;
  else
    # Kaggle packages the complete /kaggle/working tree. Select only the
    # published output directory, not committed JSONs from the checkout.
    find "$tmp" -type f \
      -path "*/study2-results/$family/v3*.json" \
      -exec cp {} "$dest"/ \;
  fi
  if [[ "$split" -eq 0 ]]; then
    local manifest
    manifest="$(find "$tmp" -type f -path "*/study2-results/$family/manifest.json" | head -n 1)"
    if [[ -z "$manifest" ]]; then
      # Mechanism's combined family manifest was created in the cloned
      # checkout by older driver versions; it is still acceptable provenance
      # when the artifact paths above came from the published output bundle.
      manifest="$(find "$tmp" -type f -path "*/study2_results/$family/manifest.json" | head -n 1)"
    fi
    if [[ -z "$manifest" && "$family" == "mechanism" ]]; then
      manifest="$(find "$tmp" -type f \( \
        -path "*/study2_results/slot_sweep/manifest.json" -o \
        -path "*/study2_results/decay_sweep/manifest.json" \
      \) | head -n 1)"
    fi
    if [[ -n "$manifest" ]]; then cp "$manifest" "$dest/manifest.json"; fi
  fi
  count="$(find "$dest" -maxdepth 1 -type f -name 'v3*.json' | wc -l | tr -d ' ')"
  echo "$family/$kernel: $count/$final_expected artifacts"
  if [[ "$split" -eq 1 && "$selected_count" == "$expected" ]]; then
    touch "$marker"
    return 0
  fi
  if [[ "$count" == "$final_expected" ]]; then
    python3 -m study2.validate "$dest" --expected-count "$final_expected"
    touch "$marker"
    return 0
  fi
  return 1
}

for poll in $(seq 1 "$MAX_POLLS"); do
  complete=0
  for spec in "${KERNELS[@]}"; do
    IFS=: read -r kernel family expected final_expected <<<"$spec"
    status="$( { kaggle kernels status "aks1321/$kernel" 2>&1 || true; } | tail -n 1)"
    echo "[$(date +%H:%M)] $family: $status"
    if [[ "$status" == *COMPLETE* || "$status" == *ERROR* ]]; then
      if collect_one "$kernel" "$family" "$expected" "$final_expected" "$status"; then
        complete=$((complete + 1))
      fi
    fi
  done
  if [[ "$complete" -eq "${#KERNELS[@]}" ]]; then
    finalize_recall_manifest
    python3 -m study2.validate "$DEST/recall_length" --expected-count 35
    python3 -m study2.analyze "$DEST" --out analysis/study2
    echo "Study 2 collection and analysis complete."
    exit 0
  fi
  [[ "$poll" -lt "$MAX_POLLS" ]] || break
  sleep "$POLL_SECONDS"
done

echo "Study 2 collection is incomplete; leaving partial artifacts for the next poll." >&2
exit 1
