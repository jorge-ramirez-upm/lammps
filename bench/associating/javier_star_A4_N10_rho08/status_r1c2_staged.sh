#!/usr/bin/env bash
set -euo pipefail
: "${OUT:?Set OUT=/path/to/r1c2_runs}"
for stage in 1 2 3; do
  dir="$OUT/stage$stage"
  [[ -d "$dir" ]] || continue
  echo "stage $stage"
  [[ -f "$dir/complete" ]] && echo "complete" || echo "incomplete"
  [[ -f "$dir/provenance.txt" ]] && cat "$dir/provenance.txt"
  for file in production.raw production.gt production.com events.dat initial.network final.network production.restart; do
    [[ -e "$dir/$file" ]] && ls -lh "$dir/$file"
  done
done
