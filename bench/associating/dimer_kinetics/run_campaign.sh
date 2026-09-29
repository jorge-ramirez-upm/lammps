#!/usr/bin/env bash
# Usage: ./run_campaign.sh LMP [pilot|full|cadence] [NPROC]
set -euo pipefail
root=$(cd "$(dirname "$0")" && pwd)
lmp=${1:?LAMMPS executable required}; mode=${2:-pilot}; np=${3:-1}
run() {
  local ea=$1 ee=$2 every=$3 rep=$4
  local tag="Ea${ea}_Ee${ee}_N${every}_r${rep}"
  local out="$root/results/${tag}.dat" log="$root/results/${tag}.log"
  [[ -s $out ]] && { echo "skip $tag"; return; }
  echo "run $tag"
  local cmd=("$lmp" -in "$root/in.dimer_kinetics.lmp" -log "$log" -var n 256 -var rho 0.05 -var T 1.0 -var nu0 20 -var Ea "$ea" -var Ee "$ee" -var Nevery "$every" -var warmup 20000 -var production 100000 -var sample 100 -var seed "$((410000 + rep * 1000 + ea * 100 + ee * 10 + every))" -var seed2 "$((410011 + rep * 1000 + ea * 100 + ee * 10 + every))" -var out "$out" -var damp "${DAMP:-2.0}")
  if ((np > 1)); then mpirun -np "$np" "${cmd[@]}"; else "${cmd[@]}"; fi
}
case $mode in
  pilot) for rep in 1 2; do run 4 4 100 "$rep"; done ;;
  cadence) for every in 50 100 200; do for rep in 1 2 3 4; do run 4 4 "$every" "$rep"; done; done ;;
  full) for ea in 2 3 4 5 6; do for rep in 1 2 3 4; do run "$ea" 4 100 "$rep"; done; done
        for ee in 2 4 6 8; do for rep in 1 2 3 4; do run 4 "$ee" 100 "$rep"; done; done
        for every in 50 100 200; do for rep in 1 2 3 4; do run 4 4 "$every" "$rep"; done; done ;;
  *) echo "mode must be pilot, full, or cadence" >&2; exit 2 ;;
esac
