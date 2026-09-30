#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "$0")" && pwd)
: "${LMP:?Set LMP=/path/to/lmp}"
: "${RESTART:?Set RESTART to an existing B3 replica equil.restart}"
MPI_NP=${MPI_NP:-8}; STEPS=${STEPS:-10000}; STATE_EVERY=${STATE_EVERY:-1000}
LANGEVIN_SEED=${LANGEVIN_SEED:-284729}; KINETICS_SEED=${KINETICS_SEED:-384729}
restart=$(realpath "$RESTART")
out=${OUT:-$root/r1b32_run}; rm -rf "$out"; mkdir -p "$out/single" "$out/multi"
for case in single multi; do
  (cd "$out/$case"; mpirun -np "$MPI_NP" "$LMP" -log log.lammps \
    -var RESTART "$restart" -var STEPS "$STEPS" -var STATE_EVERY "$STATE_EVERY" \
    -var LANGEVIN_SEED "$LANGEVIN_SEED" -var KINETICS_SEED "$KINETICS_SEED" \
    -in "$root/in.r1b32_${case}.lmp")
done
python3 "$root/analyze_r1b32_pressure_compute.py" \
  "$out/single/pressure.raw" "$out/multi/pressure.raw" \
  --single-run0 "$out/single/run0.dat" --multi-run0 "$out/multi/run0.dat" \
  --single-state "$out/single/state.lammpstrj" --multi-state "$out/multi/state.lammpstrj" \
  --out "$out/comparison.json"
