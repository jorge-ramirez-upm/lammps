#!/usr/bin/env bash
set -euo pipefail

if [[ ${RUN_BENCHMARKS:-0} != 1 ]]; then
  echo "refusing to benchmark: set RUN_BENCHMARKS=1 explicitly" >&2
  exit 2
fi

mode=${1:-timing}
config=${2:-kk1}
replica=${3:-1}
case "$mode" in timing|production) ;; *) echo "mode: timing|production" >&2; exit 2 ;; esac
case "$config" in cpu|kk1|kk2) ;; *) echo "config: cpu|kk1|kk2" >&2; exit 2 ;; esac

here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../../../.." && pwd)
data="$repo/bench/associating/javier_star_A4_N10_rho08/Stars_NA4N10C1000rho0.85rhopoly0.8.equilibrated.lammpsdat"
cpu="$repo/build-r1a/lmp"
kk="$repo/build-kokkos-cuda/lmp"
input="$here/in.nonassoc_timing.lmp"
[[ $mode == production ]] && input="$repo/bench/associating/javier_star_A4_N10_rho08/in.nonassoc_control.lmp"
[[ -x $cpu && -f $data && -x $kk ]] || { echo "missing executable/data" >&2; exit 1; }

steps=${MEASURE_STEPS:-100000}
warmup=${WARMUP_STEPS:-1000}
seed=${LANGEVIN_SEED:-812731}
online=${ONLINE:-1}
root="$here/runs/$mode/$config/replica-$replica"
mkdir -p "$root"
exe=$cpu; np=8; suffix=cpu; gpu=0; extra=()
case "$config" in
  kk1) exe=$kk; np=1; suffix=kk; gpu=1; extra=(-k on g 1 -sf kk -pk kokkos neigh half);;
  kk2) exe=$kk; np=2; suffix=kk; gpu=1; extra=(-k on g 1 -sf kk -pk kokkos neigh half);;
esac
args=(-var DATA "$data" -var LANGEVIN_SEED "$seed" -var WARMUP_STEPS "$warmup" -var MEASURE_STEPS "$steps" -var OUT_PREFIX "$root/output" -var ONLINE 0 -var FULL_TRAJ 0 -var PROD_STEPS "$steps" -var COM_EVERY 100 -var TRAJ_EVERY 100)
if [[ $mode == production ]]; then
  args=(-var DATA "$data" -var LANGEVIN_SEED "$seed" -var PROD_STEPS "$steps" -var OUT_PREFIX "$root/output" -var ONLINE "$online" -var FULL_TRAJ 0 -var COM_EVERY 100 -var TRAJ_EVERY 100)
fi
export CUDA_ROOT=${CUDA_ROOT:-/usr/local/cuda-12.8}
export PATH="$CUDA_ROOT/bin:$PATH"
mpirun_args=()
[[ $(id -u) -eq 0 ]] && mpirun_args+=(--allow-run-as-root)
if ! { time -p mpirun "${mpirun_args[@]}" -np "$np" "$exe" "${extra[@]}" -in "$input" "${args[@]}"; } 2> "$root/wall.time" | tee "$root/log.lammps"; then
  echo "benchmark command failed; see $root/wall.time and $root/log.lammps" >&2
  exit 1
fi
loop=$(awk '/Loop time of/ {v=$4} END {if (v != "") print v}' "$root/log.lammps")
[[ -n $loop ]] || { echo "benchmark produced no LAMMPS timing record" >&2; exit 1; }
wall=$(awk '$1 == "real" {print $2}' "$root/wall.time")
steps_per_second=$(awk -v n="$steps" -v t="$loop" 'BEGIN {if (t > 0) printf "%.9g", n/t}')
sha=$(sha256sum "$exe" | awk '{print $1}')
git_sha=$(git -C "$repo" rev-parse HEAD)
printf 'mode=%s\nconfig=%s\nreplica=%s\nmpi_ranks=%s\nopenmp_threads=1\ngpu_count=%s\nsuffix=%s\nmeasure_steps=%s\nloop_wall_seconds=%s\nwall_seconds=%s\nsteps_per_second=%s\nexecutable=%s\nexecutable_sha256=%s\ngit_sha=%s\n' "$mode" "$config" "$replica" "$np" "$gpu" "$suffix" "$steps" "$loop" "$wall" "$steps_per_second" "$exe" "$sha" "$git_sha" > "$root/metadata.txt"
cat "$root/metadata.txt"
