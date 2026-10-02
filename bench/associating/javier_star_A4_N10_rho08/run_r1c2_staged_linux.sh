#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "$0")" && pwd)
: "${LMP:?Set LMP=/path/to/rebuilt/lmp}"
: "${AUTHORIZE_R1C2:?Set AUTHORIZE_R1C2=YES only on the dedicated host to launch a stage}"
[[ "$AUTHORIZE_R1C2" == YES ]] || { echo "refusing R1-C2 launch without AUTHORIZE_R1C2=YES" >&2; exit 2; }
MPI_NP=${MPI_NP:-8}
STAGE=${STAGE:-1}
COM_EVERY=${COM_EVERY:-100}
ONLINE=${ONLINE:-1}
LANGEVIN_SEED=${LANGEVIN_SEED:-48279}
KINETICS_SEED=${KINETICS_SEED:-492845}
OUT=${OUT:-$root/r1c2_runs}
START_RESTART=${START_RESTART:-$root/r1c1_runs/production/production.restart}
positive(){ [[ $1 =~ ^[1-9][0-9]*$ ]] || { echo "$2 must be a positive integer: $1" >&2; exit 2; }; }
positive "$MPI_NP" MPI_NP; positive "$COM_EVERY" COM_EVERY
[[ "$ONLINE" == 0 || "$ONLINE" == 1 ]] || { echo "ONLINE must be 0 or 1" >&2; exit 2; }
[[ "$STAGE" =~ ^[1234]$ ]] || { echo "STAGE must be 1, 2, 3, or 4" >&2; exit 2; }
case "$STAGE" in
  1) STAGE_STEPS=${STAGE_STEPS:-1000000}; DEFAULT_RESTART="$START_RESTART"; WRITE_INITIAL=1 ;;
  2) STAGE_STEPS=${STAGE_STEPS:-1000000}; DEFAULT_RESTART="$OUT/stage1/production.restart"; WRITE_INITIAL=0 ;;
  3) STAGE_STEPS=${STAGE_STEPS:-2000000}; DEFAULT_RESTART="$OUT/stage2/production.restart"; WRITE_INITIAL=0 ;;
  4) STAGE_STEPS=${STAGE_STEPS:-5000000}; DEFAULT_RESTART="$OUT/stage3/production.restart"; WRITE_INITIAL=0 ;;
esac
if [[ "$STAGE" -gt 1 ]]; then
  previous=$((STAGE - 1))
  [[ -f "$OUT/stage$previous/complete" ]] || { echo "Stage $previous is not complete; refusing Stage $STAGE" >&2; exit 3; }
fi
positive "$STAGE_STEPS" STAGE_STEPS
RESTART=${RESTART:-$DEFAULT_RESTART}
[[ -s "$RESTART" ]] || { echo "restart does not exist or is empty: $RESTART" >&2; exit 2; }
if [[ "$STAGE" == 4 ]]; then
  expected_restart=$(realpath "$DEFAULT_RESTART")
  restart_candidate=$(realpath "$RESTART")
  [[ "$restart_candidate" == "$expected_restart" ]] || {
    echo "Stage 4 requires the Stage 3 restart: $expected_restart" >&2; exit 3;
  }
  grep -q 'stage=3 ' "$OUT/stage3/complete" || {
    echo "Stage 3 completion provenance is invalid; refusing Stage 4" >&2; exit 3;
  }
  recorded_restart_sha=$(sed -n 's/.*restart_sha256=\([^ ]*\).*/\1/p' "$OUT/stage3/complete")
  [[ -n "$recorded_restart_sha" && "$recorded_restart_sha" == "$(sha256sum "$RESTART" | awk '{print $1}')" ]] || {
    echo "Stage 3 restart does not match completion provenance; refusing Stage 4" >&2; exit 3;
  }
fi
lmp_path=$(realpath "$LMP"); restart_path=$(realpath "$RESTART")
repo_sha=$(git -C "$root/../../.." rev-parse HEAD)
lmp_sha256=$(sha256sum "$lmp_path" | awk '{print $1}')
restart_sha256=$(sha256sum "$restart_path" | awk '{print $1}')
stage_dir="$OUT/stage$STAGE"
[[ ! -e "$stage_dir" ]] || { echo "refusing to overwrite existing $stage_dir" >&2; exit 3; }
mkdir -p "$stage_dir"
prefix="$stage_dir/production"
event_log="$stage_dir/events.dat"
initial_network="$stage_dir/initial.network"
final_network="$stage_dir/final.network"
initial_timestep=unknown
if [[ -f "$OUT/stage$((STAGE - 1))/production.raw" ]]; then
  initial_timestep=$(tail -n 1 "$OUT/stage$((STAGE - 1))/production.raw" | awk '{print $1}')
fi
final_timestep=unknown
if [[ "$initial_timestep" =~ ^[0-9]+$ ]]; then final_timestep=$((initial_timestep + STAGE_STEPS)); fi
identity="git_sha=$repo_sha lmp_path=$lmp_path lmp_sha256=$lmp_sha256 mpi_np=$MPI_NP stage=$STAGE stage_steps=$STAGE_STEPS dt=0.01 initial_timestep=$initial_timestep final_timestep=$final_timestep com_every=$COM_EVERY online=$ONLINE langevin_seed=$LANGEVIN_SEED kinetics_seed=$KINETICS_SEED restart=$restart_path restart_sha256=$restart_sha256 event_logging=enabled write_initial=$WRITE_INITIAL out_dir=$stage_dir raw_path=$prefix.raw gt_path=$prefix.gt com_path=$prefix.com event_path=$event_log final_network_path=$final_network restart_path=$prefix.restart"
printf '%s\n' "$identity" > "$stage_dir/provenance.txt"
mpirun -np "$MPI_NP" "$lmp_path" -log "$stage_dir/lammps.log" -screen none \
  -var RESTART "$restart_path" -var STAGE_STEPS "$STAGE_STEPS" \
  -var COM_EVERY "$COM_EVERY" -var ONLINE "$ONLINE" -var LANGEVIN_SEED "$LANGEVIN_SEED" -var KINETICS_SEED "$KINETICS_SEED" \
  -var EVENT_LOG "$event_log" -var INITIAL_NETWORK "$initial_network" -var FINAL_NETWORK "$final_network" \
  -var WRITE_INITIAL "$WRITE_INITIAL" -var OUT_PREFIX "$prefix" -var FINAL_RESTART "$prefix.restart" \
  -in "$root/in.r1c2_stage.lmp"
required_files=("$prefix.raw" "$prefix.com" "$event_log" "$final_network" "$prefix.restart")
if [[ "$ONLINE" == 1 ]]; then required_files+=("$prefix.gt"); fi
for file in "${required_files[@]}"; do
  [[ -s "$file" ]] || { echo "required stage output missing: $file" >&2; exit 4; }
done
if [[ "$WRITE_INITIAL" == 1 ]]; then [[ -s "$initial_network" ]] || { echo "initial network missing" >&2; exit 4; }; fi
printf '%s\n' "$identity" > "$stage_dir/complete"
printf '%s\n' "R1-C2 stage $STAGE complete: $stage_dir"
