#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "$0")" && pwd)
: "${LMP:?Set LMP=/path/to/lmp}"
MPI_NP=${MPI_NP:-8}; EQUIL_STEPS=${EQUIL_STEPS:-500000}; PROD_STEPS=${PROD_STEPS:-1000000}
TRAJ_EVERY=${TRAJ_EVERY:-10000}; NETWORK_EVERY=${NETWORK_EVERY:-10000}; TRAJ_EXT=${TRAJ_EXT:-lammpstrj}
PARALLEL_REPLICAS=${PARALLEL_REPLICAS:-1}; runs="$root/r1b3_runs"
seed(){ case "$1" in 1) echo '184729 284729 384729';;2) echo '184730 284730 384730';;3) echo '184731 284731 384731';;4) echo '184732 284732 384732';;5) echo '184733 284733 384733';;6) echo '184734 284734 384734';;esac; }
one(){ local r=$1 d="$runs/replica$(printf '%02d' "$r")"; read -r v l k <<<"$(seed "$r")"; mkdir -p "$d"
  if [[ ! -f "$d/equil.complete" ]]; then (cd "$d"; rm -f equil.log; mpirun -np "$MPI_NP" "$LMP" -log equil.log -var DATA "$root/Stars_NA4N5C125rho0.85rhopoly0.8.equilibrated.lammpsdat" -var EQUIL_STEPS "$EQUIL_STEPS" -var TRAJ_EVERY "$TRAJ_EVERY" -var NETWORK_EVERY "$NETWORK_EVERY" -var TRAJ_EXT "$TRAJ_EXT" -var VELOCITY_SEED "$v" -var LANGEVIN_SEED "$l" -var KINETICS_SEED "$k" -in "$root/in.r1b3_equilibrate.lmp"; test -s equil.restart; touch equil.complete); fi
  if [[ ! -f "$d/production.complete" ]]; then (cd "$d"; rm -f production.log; mpirun -np "$MPI_NP" "$LMP" -log production.log -var PROD_STEPS "$PROD_STEPS" -var TRAJ_EVERY "$TRAJ_EVERY" -var NETWORK_EVERY "$NETWORK_EVERY" -var TRAJ_EXT "$TRAJ_EXT" -var LANGEVIN_SEED "$l" -var KINETICS_SEED "$k" -in "$root/in.r1b3_production.lmp"; test -s production.restart; touch production.complete); fi; }
export -f one seed; export root runs LMP MPI_NP EQUIL_STEPS PROD_STEPS TRAJ_EVERY NETWORK_EVERY TRAJ_EXT
if ((PARALLEL_REPLICAS>1)); then seq 1 6 | xargs -n1 -P "$PARALLEL_REPLICAS" bash -c 'one "$0"'; else for r in {1..6}; do one "$r"; done; fi
