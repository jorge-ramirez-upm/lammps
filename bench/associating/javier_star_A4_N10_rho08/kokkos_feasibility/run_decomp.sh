#!/usr/bin/env bash
set -euo pipefail

if [[ ${RUN_BENCHMARKS:-0} != 1 ]]; then
  echo "refusing to benchmark: set RUN_BENCHMARKS=1 explicitly" >&2
  exit 2
fi
config=${1:?config cpu|kk1}
case "$config" in cpu|kk1) ;; *) echo "config: cpu|kk1" >&2; exit 2 ;; esac
diag=${2:?case baseline|com|stress|stresscom|correlate}
case "$diag" in baseline|com|stress|stresscom|correlate) ;; *) echo "invalid diagnostic case" >&2; exit 2 ;; esac
replica=${3:?replica number}
stress_every=${4:-${STRESS_EVERY:-1}}

here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../../../.." && pwd)
data="$repo/bench/associating/javier_star_A4_N10_rho08/Stars_NA4N10C1000rho0.85rhopoly0.8.equilibrated.lammpsdat"
cpu="$repo/build-r1a/lmp"
kk="$repo/build-kokkos-cuda/lmp"
[[ -x $cpu && -x $kk && -f $data ]] || { echo "missing executable/data" >&2; exit 1; }

steps=${MEASURE_STEPS:-50000}
warmup=${WARMUP_STEPS:-1000}
seed=${LANGEVIN_SEED:-812731}
run_tag=$diag
if [[ $diag == stress && $stress_every != 1 ]]; then
  run_tag="stress-every-$stress_every"
fi
root="$here/runs/decomp/$run_tag/$config/replica-$replica"
mkdir -p "$root"
exe=$cpu; np=8; gpu=0; extra=()
if [[ $config == kk1 ]]; then
  exe=$kk; np=1; gpu=1; extra=(-k on g 1 -sf kk -pk kokkos neigh half)
fi
raw=0; com=0; online=0
case "$diag" in
  com) com=1 ;;
  stress) raw=1 ;;
  stresscom) raw=1; com=1 ;;
  correlate) raw=1; com=1; online=1 ;;
esac

export CUDA_ROOT=${CUDA_ROOT:-/usr/local/cuda-12.8}
export PATH="$CUDA_ROOT/bin:$PATH"
mpirun_args=()
[[ $(id -u) -eq 0 ]] && mpirun_args+=(--allow-run-as-root)
args=(-var DATA "$data" -var LANGEVIN_SEED "$seed" -var WARMUP_STEPS "$warmup" -var MEASURE_STEPS "$steps" -var OUT_PREFIX "$root/output" -var RAW "$raw" -var COM "$com" -var ONLINE "$online" -var STRESS_EVERY "$stress_every" -var COM_EVERY 100 -var RAW_FILE "$here/raw_stress.in" -var COM_FILE "$here/com_output.in" -var CORR_FILE "$here/correlate.in")
if ! { time -p mpirun "${mpirun_args[@]}" -np "$np" "$exe" "${extra[@]}" -in "$here/in.nonassoc_decomp.lmp" "${args[@]}"; } 2>"$root/wall.time" | tee "$root/log.lammps"; then
  echo "benchmark failed; see $root/log.lammps" >&2
  exit 1
fi
loop=$(awk '/Loop time of/ {v=$4} END {if (v != "") print v}' "$root/log.lammps")
[[ -n $loop ]] || { echo "no LAMMPS timing record" >&2; exit 1; }
wall=$(awk '$1 == "real" {print $2}' "$root/wall.time")
sps=$(awk -v n="$steps" -v t="$loop" 'BEGIN {if (t > 0) printf "%.9g", n/t}')
sha=$(sha256sum "$exe" | awk '{print $1}')
git_sha=$(git -C "$repo" rev-parse HEAD)
{
  printf 'diag=%s\nconfig=%s\nreplica=%s\nmpi_ranks=%s\nopenmp_threads=1\ngpu_count=%s\n' "$diag" "$config" "$replica" "$np" "$gpu"
  printf 'stress_every=%s\ncom_every=100\nraw_stress=%s\ncom_output=%s\ncorrelator=%s\n' "$stress_every" "$raw" "$com" "$online"
  printf 'measure_steps=%s\nloop_wall_seconds=%s\nwall_seconds=%s\nsteps_per_second=%s\n' "$steps" "$loop" "$wall" "$sps"
  printf 'executable=%s\nexecutable_sha256=%s\ngit_sha=%s\n' "$exe" "$sha" "$git_sha"
} > "$root/metadata.txt"
cat "$root/metadata.txt"
