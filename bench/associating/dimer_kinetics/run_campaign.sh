#!/usr/bin/env bash
# Usage: ./run_campaign.sh LMP [pilot|full|cadence|density-pilot|density-full] [NPROC]
set -euo pipefail
root=$(cd "$(dirname "$0")" && pwd)
lmp=${1:?LAMMPS executable required}; mode=${2:-pilot}; np=${3:-1}
production=${PRODUCTION:-100000}; warmup=${WARMUP:-20000}; replicas=${REPLICAS:-4}
run() {
  local rho=$1 ea=$2 ee=$3 every=$4 rep=$5
  local tag="rho${rho}_Ea${ea}_Ee${ee}_N${every}_r${rep}"
  # Preserve the committed rho=0.05 K1 files, whose historic names lack rho.
  if [[ $rho == 0.05 ]]; then tag="Ea${ea}_Ee${ee}_N${every}_r${rep}"; fi
  local out="$root/results/${tag}.dat" log="$root/results/${tag}.log"
  [[ -s $out ]] && { echo "skip $tag"; return; }
  echo "run $tag"
  local seed=$((410000 + rep * 1000 + ea * 100 + ee * 10 + every))
  local cmd=("$lmp" -in "$root/in.dimer_kinetics.lmp" -log "$log" -var n 256 -var rho "$rho" -var T 1.0 -var nu0 20 -var Ea "$ea" -var Ee "$ee" -var Nevery "$every" -var warmup "$warmup" -var production "$production" -var sample 100 -var seed "$seed" -var seed2 "$((seed + 11))" -var out "$out" -var damp "${DAMP:-2.0}")
  if ((np > 1)); then mpirun -np "$np" "${cmd[@]}"; else "${cmd[@]}"; fi
}
family() {
  local rho=$1 rep
  for rep in $(seq 1 "$replicas"); do
    for ea in 2 3 4 5 6; do run "$rho" "$ea" 4 100 "$rep"; done
    for ee in 2 4 6 8; do run "$rho" 4 "$ee" 100 "$rep"; done
  done
}
case $mode in
  pilot) for rep in 1 2; do run 0.05 4 4 100 "$rep"; done ;;
  cadence) for every in 50 100 200; do for rep in $(seq 1 "$replicas"); do run 0.05 4 4 "$every" "$rep"; done; done ;;
  full) family 0.05; for every in 50 100 200; do for rep in $(seq 1 "$replicas"); do run 0.05 4 4 "$every" "$rep"; done; done ;;
  density-pilot) for rho in 0.025 0.10 0.20; do for rep in 1 2; do run "$rho" 4 4 100 "$rep"; done; done ;;
  density-full) for rho in 0.025 0.10 0.20; do family "$rho"; done ;;
  *) echo "mode must be pilot, full, cadence, density-pilot, or density-full" >&2; exit 2 ;;
esac
