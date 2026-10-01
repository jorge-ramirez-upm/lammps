#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "$0")" && pwd)
: "${LMP:?Set LMP=/path/to/rebuilt/lmp}"
: "${START_RESTART:?Set START_RESTART=/path/to/trusted/restart}"
MPI_NP=${MPI_NP:-8}
BENCH_STEPS=${BENCH_STEPS:-100000}
REPS=${REPS:-3}
LANGEVIN_SEED=${LANGEVIN_SEED:-48279}
KINETICS_SEED=${KINETICS_SEED:-492845}
OUT=${OUT:-$root/r1c2_event_logging_benchmark.$(date +%Y%m%dT%H%M%S)}
positive(){ [[ $1 =~ ^[1-9][0-9]*$ ]] || { echo "$2 must be a positive integer: $1" >&2; exit 2; }; }
positive "$MPI_NP" MPI_NP; positive "$BENCH_STEPS" BENCH_STEPS; positive "$REPS" REPS
[[ -e "$OUT" ]] && { echo "refusing to overwrite existing OUT=$OUT" >&2; exit 3; }
mkdir -p "$OUT"
lmp_path=$(realpath "$LMP")
repo_sha=$(git -C "$root/../../.." rev-parse HEAD)
lmp_sha256=$(sha256sum "$lmp_path" | awk '{print $1}')
cat > "$OUT/provenance.txt" <<EOF
git_sha=$repo_sha
lmp_path=$lmp_path
lmp_sha256=$lmp_sha256
mpi_np=$MPI_NP
bench_steps=$BENCH_STEPS
repetitions=$REPS
langevin_seed=$LANGEVIN_SEED
kinetics_seed=$KINETICS_SEED
start_restart=$(realpath "$START_RESTART")
EOF

run_case(){
  local label=$1 rep=$2
  local dir="$OUT/$label/rep$(printf '%02d' "$rep")"
  mkdir -p "$dir"
  /usr/bin/time -f '%e' -o "$dir/wall_seconds" mpirun -np "$MPI_NP" "$lmp_path" \
    -log "$dir/lammps.log" -screen none \
    -var RESTART "$(realpath "$START_RESTART")" -var STEPS "$BENCH_STEPS" \
    -var LOGGING "$([[ "$label" == on ]] && echo 1 || echo 0)" \
    -var EVENT_LOG "$dir/events.dat" \
    -var INITIAL_NETWORK "$dir/initial.network" \
    -var FINAL_NETWORK "$dir/final.network" \
    -var FINAL_RESTART "$dir/final.restart" \
    -var LANGEVIN_SEED "$LANGEVIN_SEED" -var KINETICS_SEED "$KINETICS_SEED" \
    -in "$root/in.r1c2_event_logging_benchmark.lmp"
}

mkdir -p "$OUT/warmup"
/usr/bin/time -f '%e' -o "$OUT/warmup/wall_seconds" mpirun -np "$MPI_NP" "$lmp_path" \
  -log "$OUT/warmup/lammps.log" -screen none \
  -var RESTART "$(realpath "$START_RESTART")" -var STEPS "$BENCH_STEPS" \
  -var LOGGING 0 -var EVENT_LOG "$OUT/warmup/events.dat" \
  -var INITIAL_NETWORK "$OUT/warmup/initial.network" \
  -var FINAL_NETWORK "$OUT/warmup/final.network" -var FINAL_RESTART "$OUT/warmup/final.restart" \
  -var LANGEVIN_SEED "$LANGEVIN_SEED" -var KINETICS_SEED "$KINETICS_SEED" \
  -in "$root/in.r1c2_event_logging_benchmark.lmp"

for rep in $(seq 1 "$REPS"); do
  run_case off "$rep"
  run_case on "$rep"
done
python3 "$root/analyze_r1c2_event_logging_benchmark.py" "$OUT"
