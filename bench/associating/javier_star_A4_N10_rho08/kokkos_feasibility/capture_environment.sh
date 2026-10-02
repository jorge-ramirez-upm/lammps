#!/usr/bin/env bash
set -eu

out=${1:-environment.txt}
nvcc_bin=$(command -v nvcc || find /usr/local -path '*/bin/nvcc' -type f -print -quit 2>/dev/null || true)
{
  date -Is
  echo "hostname=$(hostname)"
  echo "os=$( . /etc/os-release && printf '%s %s' "$NAME" "$VERSION_ID" )"
  echo "kernel=$(uname -srvm)"
  echo "cpu_model=$(lscpu | awk -F: '/Model name/ {sub(/^ +/,"",$2); print $2; exit}')"
  echo "cpu_logical=$(nproc)"
  echo "cpu_physical=$(lscpu | awk -F: '/Core\(s\) per socket/ {gsub(/ /,"",$2); c=$2} /Socket\(s\)/ {gsub(/ /,"",$2); s=$2} END {if (c && s) print c*s}')"
  echo "compiler=$(c++ --version | head -1)"
  echo "mpi=$(mpirun --version | head -1)"
  echo "cmake=$(cmake --version | head -1)"
  echo "cuda_root=${CUDA_ROOT:-unset}"
  echo "nvcc=${nvcc_bin:-missing}"
  [[ -z $nvcc_bin ]] || "$nvcc_bin" --version 2>&1
  nvidia-smi 2>&1 || true
} > "$out"
