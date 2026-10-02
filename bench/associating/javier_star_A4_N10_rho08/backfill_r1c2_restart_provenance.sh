#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "$0")" && pwd)
OUT=${OUT:-$root/r1c2_runs}
stages=("$@")
if [[ ${#stages[@]} -eq 0 ]]; then stages=(1 2 3); fi
for stage in "${stages[@]}"; do
  [[ "$stage" =~ ^[1-4]$ ]] || { echo "invalid stage: $stage" >&2; exit 2; }
  stage_dir="$OUT/stage$stage"
  complete="$stage_dir/complete"
  output="$stage_dir/production.restart"
  record="$stage_dir/output_restart_provenance.txt"
  [[ -s "$complete" ]] || { echo "missing complete marker: $complete" >&2; exit 3; }
  grep -q "stage=$stage " "$complete" || { echo "invalid completion stage in $complete" >&2; exit 3; }
  [[ -s "$output" ]] || { echo "missing output restart: $output" >&2; exit 3; }
  output_path=$(realpath "$output")
  output_sha256=$(sha256sum "$output" | awk '{print $1}')
  if [[ -e "$record" ]]; then
    grep -q "output_restart_path=$output_path " "$record" || { echo "mismatched existing record: $record" >&2; exit 3; }
    grep -q "output_restart_sha256=$output_sha256" "$record" || { echo "stale existing record: $record" >&2; exit 3; }
    echo "verified $record"
    continue
  fi
  source_sha256=$(sha256sum "$complete" | awk '{print $1}')
  tmp="$record.tmp.$$"
  printf 'stage=%s output_restart_path=%s output_restart_sha256=%s source_complete_path=%s source_complete_sha256=%s migration=legacy-output-restart-backfill\n' \
    "$stage" "$output_path" "$output_sha256" "$(realpath "$complete")" "$source_sha256" > "$tmp"
  mv "$tmp" "$record"
  echo "backfilled $record"
done
