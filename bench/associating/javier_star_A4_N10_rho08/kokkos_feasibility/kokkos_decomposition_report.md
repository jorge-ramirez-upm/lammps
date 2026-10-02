# KOKKOS diagnostic overhead decomposition

Date: 2026-10-03

This report covers the nonassociating Javier full-system control only. It does not modify or benchmark `pair associating` or `fix associating/kinetics`.

## Result

The one-GPU KOKKOS path is useful for force/integration work, but every-step raw stress diagnostics erase that advantage for this workflow. The identifiable dominant cost is the standard host-side `compute pressure` fallback invoked by `fix print` (and by the correlator when enabled), not COM output or the small files themselves.

| case | CPU-8 median loop (s) | GPU-1 median loop (s) | CPU/GPU speedup |
|---|---:|---:|---:|
| dynamics baseline | 79.005 | 43.591 | 1.81x |
| COM only | 80.597 | 46.222 | 1.74x |
| raw stress every step | 89.463 | 151.167 | 0.59x |
| raw stress + COM | 90.580 | 152.376 | 0.59x |
| raw stress + COM + correlate/long | 91.491 | 150.284 | 0.61x |

The three-replica raw timings and cadence sweep are in [decomposition_summary.csv](decomposition_summary.csv). Loop timing is the LAMMPS `Loop time` value; shell wall time is retained in each replica directory separately.

GPU raw-stress cadence medians were 151.167 s at every step, 55.076 s at every 10 steps, and 45.453 s at every 100 steps for 50,000 measured steps. This frequency dependence is consistent with a per-invocation synchronization/fallback cost.

## Physical-equivalence checks

All timed inputs used the same data file and model: 43,563 atoms, 40,000 FENE bonds, WCA `lj/cut` cutoff `2^(1/6)`, FENE coefficients `(30, 1.5, 1, 1)`, `dt=0.01`, Langevin `(1,1,2)`, `rho=0.85`, and the same 8-rank CPU or 1-rank/1-GPU execution choices. The logs retain the same thermo cadence and six pressure components.

CPU and GPU runs were not expected to be trajectory-identical. Across the matrix, temperature remained near 1, potential energy remained on the `7.94e5` scale, and pressure remained near 5 with fluctuating shear components. No atom-count, timestep, coefficient, or systematic thermodynamic discrepancy was observed.

## Why stress synchronizes

`src/compute_pressure.cpp` has no KOKKOS counterpart in this checkout. Its vector path calls `virial_compute()` and performs `MPI_Allreduce` on the six host virial components ([compute_pressure.cpp](../../../../src/compute_pressure.cpp#L283)). The raw input evaluates `c_press[1..6]` every step through standard `fix print`; `FixPrint::end_of_step()` performs variable substitution on every scheduled print ([fix_print.cpp](../../../../src/fix_print.cpp#L153)).

LAMMPS KOKKOS explicitly turns on `auto_sync` around non-KOKKOS computes/fixes ([modify_kokkos.cpp](../../../../src/KOKKOS/modify_kokkos.cpp#L56)), and the atom layer synchronizes device data to host under that condition ([atom_kokkos.cpp](../../../../src/KOKKOS/atom_kokkos.cpp#L244)). Therefore, in this input, six global stress components every step do force the standard host pressure path and a device/host synchronization on every print invocation.

COM output is also standard `compute chunk/atom` + `compute com/chunk` + `fix ave/time`; there are no KOKKOS `compute_com_chunk`, `compute_property_chunk`, `fix_print`, or `fix_ave_time` styles in `src/KOKKOS`. It is sampled every 100 steps and is comparatively inexpensive. `fix ave/correlate/long` is standard EXTRA-FIX host code; its `end_of_step()` evaluates its values and variables ([fix_ave_correlate_long.cpp](../../../../src/EXTRA-FIX/fix_ave_correlate_long.cpp#L450)), but its incremental cost was small relative to raw every-step stress.

## Build and integrity

Only the separate KOKKOS executable was rebuilt:

```text
cmake -S cmake -B build-kokkos-cuda -D PKG_EXTRA-FIX=ON
cmake --build build-kokkos-cuda --parallel 8
```

The benchmarked executable is `build-kokkos-cuda/lmp`, SHA-256 `a5ca23b063da0cf0643a04331dcde00c0938b9d597798956b0b9978870ac403c`. The trusted CPU executable was not rebuilt or touched: `build-r1a/lmp`, SHA-256 `e403ee6a1a793b1fca3974a34b0a12c4e5c79fe2f948b4141265815072bb7862`. All replicas record executable hashes and Git SHA in their `metadata.txt` files.

## Decision

This is situation **B**, with a practical consequence close to **C** for the intended every-step stress workflow: one identifiable standard diagnostic, `compute pressure`/raw stress, dominates and should be optimized or scientifically redesigned first. The timing-only result is in the useful range, but the actual production-like diagnostic result is CPU-favored (the short 5,000-step online comparison was 8.398 s CPU versus 15.220 s GPU loop time).

Do not proceed to custom associating KOKKOS styles yet. The next justified step is a scientific/implementation decision about how to obtain the required fast stress statistics without invoking the standard host pressure path every MD step; only after that should a mixed associating CPU/GPU benchmark be reconsidered.
