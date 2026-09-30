#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "$0")" && pwd); out=${1:-$root/r1c1_runs}
echo "output=$out"
[[ -f "$out/provenance.txt" ]] && { echo "provenance:"; sed 's/^/  /' "$out/provenance.txt"; } || echo "provenance: missing"
for stage in equil production; do
  d="$out/$stage"; marker="$d/$stage.complete"; state=not-started
  [[ -d "$d" ]] && state=incomplete; [[ -f "$marker" ]] && state=complete
  steps=$(awk '/^[[:space:]]*[0-9]+/{x=$1} END{if (x=="") print "-"; else print x}' "$d/$stage.log" 2>/dev/null || echo -)
  size=$(du -sh "$d" 2>/dev/null | cut -f1 || echo -)
  printf '%s\tstate=%s\tlast_step=%s\tsize=%s\n' "$stage" "$state" "$steps" "$size"
  if [[ -f "$marker" && -f "$out/provenance.txt" ]]; then
    [[ "$(<"$marker")" == "$(<"$out/provenance.txt")" ]] && echo "  provenance_match=yes" || echo "  provenance_match=no"
  fi
done
