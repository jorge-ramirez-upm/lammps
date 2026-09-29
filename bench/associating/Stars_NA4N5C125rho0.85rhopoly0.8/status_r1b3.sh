#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "$0")" && pwd)
for d in "$root"/r1b3_runs/replica*; do [[ -e "$d" ]] || continue; s=not-started; [[ -f "$d/equil.complete" ]] && s=equil-complete; [[ -f "$d/production.raw" ]] && s=production-incomplete; [[ -f "$d/production.complete" ]] && s=production-complete; raw=$(du -h "$d/production.raw" 2>/dev/null|cut -f1||echo -); last=$(awk '/^[[:space:]]*[0-9]+/{x=$1} END{print x+0}' "$d/production.log" 2>/dev/null||echo -); printf '%s\t%s\traw=%s\tlast-step=%s\n' "$(basename "$d")" "$s" "$raw" "$last"; done
