# KOKKOS/CUDA feasibility benchmark

This directory is isolated from the R1-C2 production paths. It does not modify
`src/` or the trusted `build-r1a/lmp`. The reference data and force/integration
settings are the Javier nonassociating full-system control: 43,563 atoms,
`rho_poly=0.8`, `rho=0.85`, `T=1`, `dt=0.01`, WCA/LJ-cut, FENE backbone, and
Langevin damping 2.

## Verified support in this checkout

| style | KOKKOS variant | benchmark result |
|---|---|---|
| `lj/cut` | `pair_lj_cut_kokkos.cpp` | device |
| `bond_style fene` | `bond_fene_kokkos.cpp` | device |
| `nve` | `fix_nve_kokkos.cpp` | device |
| `langevin` | `fix_langevin_kokkos.cpp` | device |
| `compute pressure` | no `*_kokkos` source | host fallback |
| `compute com/chunk` | no `*_kokkos` source | host fallback |
| `fix print`, `fix ave/time` | no `*_kokkos` source | host fallback |
| `fix ave/correlate/long` | EXTRA-FIX, no KOKKOS source | host fallback |
| `compute stress/atom` | no `*_kokkos` source | host fallback |
| `compute msd` | no `*_kokkos` source | host fallback |

The suffix is supplied at launch: `-k on -sf kk`; LAMMPS uses the standard
style when a `/kk` variant does not exist. The KOKKOS package also requires
`run_style verlet/kk`, supplied by `-sf kk` before input processing.

## Configuration/build

The successful isolated configuration was:

```bash
CUDA_ROOT=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH \
cmake -S cmake -B build-kokkos-cuda -C cmake/presets/kokkos-cuda.cmake \
  -D CMAKE_BUILD_TYPE=Release -D CMAKE_CXX_STANDARD=20 \
  -D CMAKE_CXX_COMPILER=$PWD/lib/kokkos/bin/nvcc_wrapper \
  -D CMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc \
  -D Kokkos_ARCH_PASCAL61=ON -D BUILD_MPI=ON -D BUILD_OMP=OFF \
  -D PKG_MOLECULE=ON -D PKG_KOKKOS=ON -D BUILD_SHARED_LIBS=OFF \
  -D BUILD_TESTING=OFF -D PKG_EXTRA-FIX=ON
CUDA_ROOT=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH \
cmake --build build-kokkos-cuda --parallel 2
```

The separate executable is `build-kokkos-cuda/lmp`. The build uses bundled
Kokkos 5.2.1, CUDA 12.8, and `PASCAL61` for the NVIDIA TITAN Xp. `nvcc` must
be in `PATH` when building because the bundled wrapper resolves it at build
time.

Capture the environment before a benchmark:

```bash
./capture_environment.sh environment.txt
```

## Benchmark commands

No timing matrix is run automatically. Each invocation below is one replica;
use at least one warm-up and three replicas per configuration when authorized.
Defaults are 1,000 warm-up and 100,000 measured steps; override with
`WARMUP_STEPS` and `MEASURE_STEPS`.

```bash
cd bench/associating/javier_star_A4_N10_rho08/kokkos_feasibility
RUN_BENCHMARKS=1 ./run.sh timing cpu 1
RUN_BENCHMARKS=1 ./run.sh timing kk1 1
RUN_BENCHMARKS=1 ./run.sh timing kk2 1
```

`cpu` uses the trusted `build-r1a/lmp` at MPI NP 8. `kk1` is one MPI rank and
one GPU; `kk2` is two MPI ranks sharing one GPU. `-sf kk` accelerates the four
supported standard styles and leaves pressure diagnostics on the host. The
timing input keeps the force/integration model but omits raw stress and COM
files. `production` instead uses the existing control input and representative
outputs:

```bash
RUN_BENCHMARKS=1 ./run.sh production cpu 1
RUN_BENCHMARKS=1 ./run.sh production kk1 1
```

Each run writes an isolated directory under `runs/` with the log, wall-time
capture, metadata, executable SHA-256, and Git SHA. Do not compare timing-only
and production-like rows as the same workload. Compute median wall time and
speedup as `CPU median / candidate median`; report steps/s and spread from the
metadata rows.

Physical validation should use short common windows and compare temperature,
potential energy, six pressure components, zero-lag stress statistics, and star
COM MSD with stochastic/floating-point tolerances. Do not require bitwise
identity and do not implement associating KOKKOS styles in this milestone.

The completed diagnostic decomposition is summarized in
[`kokkos_decomposition_report.md`](kokkos_decomposition_report.md) and
[`decomposition_summary.csv`](decomposition_summary.csv). Re-run an individual
decomposition replica with:

```bash
RUN_BENCHMARKS=1 MEASURE_STEPS=50000 ./run_decomp.sh kk1 baseline 1
RUN_BENCHMARKS=1 MEASURE_STEPS=50000 ./run_decomp.sh kk1 stress 1 10
```

The `stress` fourth argument is the raw-stress cadence; generated `runs/`
artifacts are intentionally ignored because they include large raw/COM files.
