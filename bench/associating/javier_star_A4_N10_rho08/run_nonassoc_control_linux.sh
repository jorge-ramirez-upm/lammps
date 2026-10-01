#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "$0")" && pwd)
: "${LMP:?Set LMP=/path/to/rebuilt/lmp}"
MPI_NP=${MPI_NP:-8}
PROD_STEPS=${PROD_STEPS:-500000}
COM_EVERY=${COM_EVERY:-100}
TRAJ_EVERY=${TRAJ_EVERY:-100}
FULL_TRAJ=${FULL_TRAJ:-0}
ONLINE=${ONLINE:-1}
LANGEVIN_SEED=${LANGEVIN_SEED:-48279}
DATA=${DATA:-$root/Stars_NA4N10C1000rho0.85rhopoly0.8.equilibrated.lammpsdat}
OUT=${OUT:-$root/nonassoc_control_runs}
positive(){ [[ $1 =~ ^[1-9][0-9]*$ ]] || { echo "$2 must be a positive integer: $1" >&2; exit 2; }; }
positive "$MPI_NP" MPI_NP; positive "$PROD_STEPS" PROD_STEPS; positive "$COM_EVERY" COM_EVERY; positive "$TRAJ_EVERY" TRAJ_EVERY
[[ "$ONLINE" == 0 || "$ONLINE" == 1 ]] || { echo "ONLINE must be 0 or 1" >&2; exit 2; }
[[ "$FULL_TRAJ" == 0 || "$FULL_TRAJ" == 1 ]] || { echo "FULL_TRAJ must be 0 or 1" >&2; exit 2; }
[[ -s "$DATA" ]] || { echo "DATA does not exist or is empty: $DATA" >&2; exit 2; }
[[ -e "$OUT" ]] && { echo "refusing to overwrite existing OUT=$OUT" >&2; exit 3; }
mkdir -p "$OUT/production"
lmp_path=$(realpath "$LMP")
data_path=$(realpath "$DATA")
repo_sha=$(git -C "$root/../../.." rev-parse HEAD)
lmp_sha256=$(sha256sum "$lmp_path" | awk '{print $1}')
com_frames=$((PROD_STEPS / COM_EVERY + 1))
traj_frames=$((PROD_STEPS / TRAJ_EVERY + 1))
stars=1000
atoms=41000
estimated_com_bytes=$((com_frames * stars * 64))
estimated_full_atom_bytes=$((traj_frames * atoms * 50))
prefix="$OUT/production/control_nonassoc"
printf '%s\n' \
  "git_sha=$repo_sha" \
  "lmp_path=$lmp_path" \
  "lmp_sha256=$lmp_sha256" \
  "mpi_np=$MPI_NP" \
  "data_path=$data_path" \
  "data_sha256=$(sha256sum "$data_path" | awk '{print $1}')" \
  "prod_steps=$PROD_STEPS" "dt=0.01" "temperature=1.0" "langevin_damping=2.0" \
  "stress_every=1" "com_every=$COM_EVERY" "com_dt=$(awk "BEGIN {printf \"%.12g\", $COM_EVERY * 0.01}")" \
  "full_atom_trajectory=$FULL_TRAJ" "trajectory_every=$TRAJ_EVERY" \
  "online_correlator=$ONLINE" \
  "trajectory_dt=$(awk "BEGIN {printf \"%.12g\", $TRAJ_EVERY * 0.01}")" \
  "com_frames=$com_frames" "estimated_uncompressed_com_bytes=$estimated_com_bytes" \
  "estimated_uncompressed_full_atom_bytes=$estimated_full_atom_bytes" > "$OUT/provenance.txt"
mpirun -np "$MPI_NP" "$lmp_path" -log "$OUT/production/control_nonassoc.log" -screen none \
  -var DATA "$data_path" -var PROD_STEPS "$PROD_STEPS" -var COM_EVERY "$COM_EVERY" -var TRAJ_EVERY "$TRAJ_EVERY" \
  -var FULL_TRAJ "$FULL_TRAJ" -var ONLINE "$ONLINE" -var OUT_PREFIX "$prefix" -var LANGEVIN_SEED "$LANGEVIN_SEED" \
  -in "$root/in.nonassoc_control.lmp"
printf '%s\n' "git_sha=$repo_sha" "lmp_sha256=$lmp_sha256" "prod_steps=$PROD_STEPS" > "$OUT/production/control_nonassoc.complete"
printf '%s\n' "control production complete: $OUT/production"
