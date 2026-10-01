#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "$0")" && pwd)
: "${LMP:?Set LMP=/path/to/rebuilt/lmp}"
MPI_NP=${MPI_NP:-8}
PROD_STEPS=${PROD_STEPS:-500000}
TRAJ_EVERY=${TRAJ_EVERY:-100}
ONLINE=${ONLINE:-1}
LANGEVIN_SEED=${LANGEVIN_SEED:-48279}
DATA=${DATA:-$root/Stars_NA4N10C1000rho0.85rhopoly0.8.equilibrated.lammpsdat}
OUT=${OUT:-$root/nonassoc_control_runs}
positive(){ [[ $1 =~ ^[1-9][0-9]*$ ]] || { echo "$2 must be a positive integer: $1" >&2; exit 2; }; }
positive "$MPI_NP" MPI_NP; positive "$PROD_STEPS" PROD_STEPS; positive "$TRAJ_EVERY" TRAJ_EVERY
[[ "$ONLINE" == 0 || "$ONLINE" == 1 ]] || { echo "ONLINE must be 0 or 1" >&2; exit 2; }
[[ -s "$DATA" ]] || { echo "DATA does not exist or is empty: $DATA" >&2; exit 2; }
[[ -e "$OUT" ]] && { echo "refusing to overwrite existing OUT=$OUT" >&2; exit 3; }
mkdir -p "$OUT/production"
lmp_path=$(realpath "$LMP")
data_path=$(realpath "$DATA")
repo_sha=$(git -C "$root/../../.." rev-parse HEAD)
lmp_sha256=$(sha256sum "$lmp_path" | awk '{print $1}')
if "$lmp_path" -h 2>&1 | grep -q COMPRESS; then traj_gz=1; traj_ext=lammpstrj.gz; else traj_gz=0; traj_ext=lammpstrj; fi
frames=$((PROD_STEPS / TRAJ_EVERY))
atoms=41000
estimated_uncompressed_bytes=$((frames * atoms * 50))
prefix="$OUT/production/control_nonassoc"
printf '%s\n' \
  "git_sha=$repo_sha" \
  "lmp_path=$lmp_path" \
  "lmp_sha256=$lmp_sha256" \
  "mpi_np=$MPI_NP" \
  "data_path=$data_path" \
  "data_sha256=$(sha256sum "$data_path" | awk '{print $1}')" \
  "prod_steps=$PROD_STEPS" "dt=0.01" "temperature=1.0" "langevin_damping=2.0" \
  "stress_every=1" "trajectory_every=$TRAJ_EVERY" \
  "online_correlator=$ONLINE" \
  "trajectory_dt=$(awk "BEGIN {printf \"%.12g\", $TRAJ_EVERY * 0.01}")" "trajectory_ext=$traj_ext" \
  "trajectory_frames=$frames" \
  "estimated_uncompressed_trajectory_bytes=$estimated_uncompressed_bytes" > "$OUT/provenance.txt"
mpirun -np "$MPI_NP" "$lmp_path" -log "$OUT/production/control_nonassoc.log" -screen none \
  -var DATA "$data_path" -var PROD_STEPS "$PROD_STEPS" -var TRAJ_EVERY "$TRAJ_EVERY" \
  -var TRAJ_GZ "$traj_gz" -var ONLINE "$ONLINE" -var OUT_PREFIX "$prefix" -var LANGEVIN_SEED "$LANGEVIN_SEED" \
  -in "$root/in.nonassoc_control.lmp"
printf '%s\n' "git_sha=$repo_sha" "lmp_sha256=$lmp_sha256" "prod_steps=$PROD_STEPS" > "$OUT/production/control_nonassoc.complete"
printf '%s\n' "control production complete: $OUT/production"
