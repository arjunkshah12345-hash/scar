#!/bin/bash
# Poll Kaggle kernels; download sweep results as kernels complete; push the
# cost kernel (cc) when a CPU session slot frees. Run from repo root:
#   nohup bash kaggle/collect.sh > /tmp/collect.log 2>&1 &
cd "$(dirname "$0")/.." || exit 1
TAGS="pd ps fd fs rc pc"
CC_PUSHED=0
DONE=""
for i in $(seq 1 240); do   # up to 4 hours of 60s polls
  for tag in $TAGS; do
    case ",$DONE," in *,"$tag",*) continue;; esac
    st=$(kaggle kernels status "aks1321/scar-sweep-$tag" 2>&1 | tail -1)
    echo "[$(date +%H:%M)] $tag: $st"
    if echo "$st" | grep -q 'COMPLETE'; then
      mkdir -p "/tmp/out_$tag"
      kaggle kernels output "aks1321/scar-sweep-$tag" -p "/tmp/out_$tag" >/dev/null 2>&1
      # results may be at output root or under scar/results
      for base in "/tmp/out_$tag/results" "/tmp/out_$tag/scar/results"; do
        [ -d "$base" ] && cp "$base"/*.json results/ 2>/dev/null && echo "[$(date +%H:%M)] $tag: copied results from $base"
      done
      DONE="$DONE,$tag"
    elif echo "$st" | grep -q 'ERROR'; then
      echo "[$(date +%H:%M)] $tag: kernel ERROR - inspect output manually"
      mkdir -p "/tmp/out_$tag"
      kaggle kernels output "aks1321/scar-sweep-$tag" -p "/tmp/out_$tag" >/dev/null 2>&1
      for base in "/tmp/out_$tag/results" "/tmp/out_$tag/scar/results"; do
        [ -d "$base" ] && cp "$base"/*.json results/ 2>/dev/null && echo "[$(date +%H:%M)] $tag: copied partial results from $base"
      done
      DONE="$DONE,$tag"
    fi
  done
  all=1
  for tag in $TAGS; do case ",$DONE," in *,"$tag",*) ;; *) all=0;; esac; done
  if [ $all -eq 1 ]; then
    if [ $CC_PUSHED -eq 0 ]; then
      echo "[$(date +%H:%M)] all sweep kernels done; pushing cost kernel cc"
      kaggle kernels push -p kaggle/cc && CC_PUSHED=1
    else
      st=$(kaggle kernels status aks1321/scar-sweep-cc 2>&1 | tail -1)
      echo "[$(date +%H:%M)] cc: $st"
      if echo "$st" | grep -q 'COMPLETE'; then
        mkdir -p /tmp/out_cc
        kaggle kernels output aks1321/scar-sweep-cc -p /tmp/out_cc >/dev/null 2>&1
        for f in /tmp/out_cc/cost_bench.json /tmp/out_cc/scar/data/cost_bench.json; do
          [ -f "$f" ] && cp "$f" data/cost_bench.json && echo "[$(date +%H:%M)] cc: copied cost_bench.json"
        done
        echo "[$(date +%H:%M)] ALL COLLECTED"
        break
      fi
    fi
  fi
  sleep 60
done
echo "collector exiting; results/ contains: $(ls results/*.json 2>/dev/null | wc -l | tr -d ' ') files"
