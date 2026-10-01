#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "$0")" && pwd)
: "${OUT:?Set OUT=/path/to/r1c2_event_logging_benchmark.*}"
cat "$OUT/provenance.txt"
python3 "$root/analyze_r1c2_event_logging_benchmark.py" "$OUT"
