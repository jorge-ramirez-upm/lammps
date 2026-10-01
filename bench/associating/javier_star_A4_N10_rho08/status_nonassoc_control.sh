#!/usr/bin/env bash
set -euo pipefail
: "${OUT:?Set OUT=/path/to/nonassoc_control_runs}"
cat "$OUT/provenance.txt"
if [[ -f "$OUT/production/control_nonassoc.complete" ]]; then
  echo "production: complete"
else
  echo "production: incomplete"
fi
for file in "$OUT"/production/control_nonassoc.{raw,gt,lammpstrj,lammpstrj.gz,restart}; do
  [[ -e "$file" ]] && ls -lh "$file"
done
