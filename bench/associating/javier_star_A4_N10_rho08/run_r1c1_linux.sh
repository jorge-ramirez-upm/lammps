#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "$0")" && pwd)
: "${LMP:?Set LMP=/path/to/lmp}"
MPI_NP=${MPI_NP:-8}; EQUIL_STEPS=${EQUIL_STEPS:-100000}; PROD_STEPS=${PROD_STEPS:-1000000}
TRAJ_EVERY=${TRAJ_EVERY:-10000}; NETWORK_EVERY=${NETWORK_EVERY:-10000}
VELOCITY_SEED=${VELOCITY_SEED:-4839876}; LANGEVIN_SEED=${LANGEVIN_SEED:-48279}; KINETICS_SEED=${KINETICS_SEED:-492845}
OUT=${OUT:-$root/r1c1_runs}; F=${F:-Stars_NA4N10C1000rho0.85rhopoly0.8.equilibrated}
START_RESTART=${START_RESTART:-$root/${F}.r1a.equil_cont.restart}; REUSE_RESTART=${REUSE_RESTART:-1}
positive(){ [[ $1 =~ ^[1-9][0-9]*$ ]] || { echo "$2 must be a positive integer: $1" >&2; exit 2; }; }
for n in MPI_NP EQUIL_STEPS PROD_STEPS TRAJ_EVERY NETWORK_EVERY; do positive "${!n}" "$n"; done
[[ "$REUSE_RESTART" == 0 || "$REUSE_RESTART" == 1 ]] || { echo "REUSE_RESTART must be 0 or 1" >&2; exit 2; }
lmp_path=$(realpath "$LMP")
repo_sha=$(git -C "$root/../../.." rev-parse HEAD)
lmp_sha256=$(sha256sum "$lmp_path" | awk '{print $1}')
if "$lmp_path" -h 2>&1 | grep -q COMPRESS; then compress=1; else compress=0; fi
if [[ "${TRAJ_EXT:-lammpstrj.gz}" == *.gz && $compress == 1 ]]; then traj_gz=1; traj_ext=lammpstrj.gz; else traj_gz=0; traj_ext=lammpstrj; fi
if [[ "$REUSE_RESTART" == 1 && -s "$START_RESTART" ]]; then start_mode=restart; start_identity=$(realpath "$START_RESTART")
else start_mode=equilibrate; start_identity="$root/${F}.lammpsdat"; fi
identity="git_sha=$repo_sha lmp_sha256=$lmp_sha256 lmp_path=$lmp_path mpi_np=$MPI_NP equil_steps=$EQUIL_STEPS prod_steps=$PROD_STEPS traj_every=$TRAJ_EVERY network_every=$NETWORK_EVERY velocity_seed=$VELOCITY_SEED langevin_seed=$LANGEVIN_SEED kinetics_seed=$KINETICS_SEED start_mode=$start_mode start=$start_identity traj_ext=$traj_ext"
mkdir -p "$OUT"
if [[ -f "$OUT/provenance.txt" ]]; then
  [[ "$(<"$OUT/provenance.txt")" == "$identity" ]] || { echo "$OUT provenance/config mismatch; use a new OUT" >&2; exit 3; }
else
  printf '%s\n' "$identity" > "$OUT/provenance.txt"
fi
stage(){ local dir=$1 marker=$2 required=$3; shift 3; mkdir -p "$dir"
  if [[ -f "$dir/$marker" ]]; then
    [[ "$(<"$dir/$marker")" == "$identity" ]] || { echo "$dir/$marker provenance/config mismatch; use a new OUT" >&2; exit 3; }
    echo "$(basename "$dir"): complete (skip)"; return
  fi
  find "$dir" -mindepth 1 -maxdepth 1 -type f -print -quit | grep -q . && { echo "$dir contains partial output without a valid marker; use a new OUT" >&2; exit 3; } || true
  (cd "$dir"; "$@")
  for file in $required; do [[ -s "$dir/$file" ]] || { echo "$dir/$file missing; refusing completion marker" >&2; exit 4; }; done
  printf '%s' "$identity" > "$dir/$marker"
}
if [[ "$start_mode" == equilibrate ]]; then
  stage "$OUT/equil" equil.complete "equil.restart" mpirun -np "$MPI_NP" "$lmp_path" -log equil.log \
    -var DATA "$root/${F}.lammpsdat" -var EQUIL_STEPS "$EQUIL_STEPS" \
    -var VELOCITY_SEED "$VELOCITY_SEED" -var LANGEVIN_SEED "$LANGEVIN_SEED" \
    -var KINETICS_SEED "$KINETICS_SEED" -in "$root/in.r1c1_equilibrate.lmp"
  restart="$OUT/equil/equil.restart"
else
  restart="$START_RESTART"
fi
stage "$OUT/production" production.complete "production.raw production.gt production.restart" mpirun -np "$MPI_NP" "$lmp_path" -log production.log \
  -var RESTART "$restart" -var PROD_STEPS "$PROD_STEPS" -var TRAJ_EVERY "$TRAJ_EVERY" \
  -var NETWORK_EVERY "$NETWORK_EVERY" -var TRAJ_GZ "$traj_gz" \
  -var LANGEVIN_SEED "$LANGEVIN_SEED" -var KINETICS_SEED "$KINETICS_SEED" \
  -in "$root/in.r1c1_production.lmp"
printf '%s\n' "R1-C1 ready: start_mode=$start_mode trajectory=$traj_ext output=$OUT/production"
