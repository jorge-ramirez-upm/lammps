# Device-resident stress buffer design study

Status: analysis only. No `src/` files, `src/ASSOCIATING/*`, production executable, or KOKKOS executable were modified or rebuilt for this study.

## Conclusion

**FEASIBLE BUT REQUIRES MAJOR LAMMPS/KOKKOS CHANGES.**

The GPU already performs the expensive pair and FENE interaction reductions on device, but the current KOKKOS force-style API returns each six-component `EV_FLOAT` reduction to host C++ state on every force call. A new host-side fix alone cannot avoid that transfer. The smallest credible proof of concept is a rheology-specific KOKKOS stress sampler plus a narrow device-resident reduction/buffer hook in the standard KOKKOS force path. It should initially target one MPI rank and one GPU and write raw blocks for offline correlation.

This is feasible because the required data is only six scalars per step and the existing KOKKOS kernels already calculate the required pair and FENE virials. It is not a trivial `compute pressure/kk` addition because the current host-facing compute API and force reduction results impose synchronization at their boundaries.

## Exact current path

### Scheduling the virial work

`Integrate::ev_set()` checks registered global virial consumers and sets `vflag_global` to the configured virial style when their `matchstep()` is true ([`src/integrate.cpp`](../../../../src/integrate.cpp#L106)). The Verlet-KOKKOS loop passes this flag to pair and bond styles ([`src/KOKKOS/verlet_kokkos.cpp`](../../../../src/KOKKOS/verlet_kokkos.cpp#L434)). With raw stress every step, the pressure consumer causes the virial flag to be active every step; with a block sampler, that request must still remain active every step so the force styles compute complete stress contributions.

### LJ pair virial

`PairLJCutKokkos::compute()` calls `ev_init()` and then `pair_compute()` ([`src/KOKKOS/pair_lj_cut_kokkos.cpp`](../../../../src/KOKKOS/pair_lj_cut_kokkos.cpp#L65)). `PairComputeFunctor` defines `EV_FLOAT` as its reduction value, containing energy and `virial[6]` ([`src/KOKKOS/pair_kokkos.h`](../../../../src/KOKKOS/pair_kokkos.h#L54)). The neighbor kernel accumulates the six pair virial terms in `ev.v[]`; `pair_compute_neighlist()` invokes `Kokkos::parallel_reduce()` ([`src/KOKKOS/pair_kokkos.h`](../../../../src/KOKKOS/pair_kokkos.h#L953)).

The reduction is device-executed, but its result is returned as the host-side `EV_FLOAT ev`. `PairLJCutKokkos::compute()` immediately copies `ev.v[0..5]` into the base-class host array `virial[6]` ([`src/KOKKOS/pair_lj_cut_kokkos.cpp`](../../../../src/KOKKOS/pair_lj_cut_kokkos.cpp#L111)). Thus the current path has no persistent device-resident global pair virial available to a later host fix.

For the alternative F·r path, `pair_virial_fdotr_compute()` also performs a device `parallel_reduce()` and then copies the six result values into `fpair->virial[]` on the host ([`src/KOKKOS/pair_kokkos.h`](../../../../src/KOKKOS/pair_kokkos.h#L1003)). The ordinary base implementation is host-only ([`src/pair.cpp`](../../../../src/pair.cpp#L1832)).

### FENE bond virial

`BondFENEKokkos::compute()` obtains the bond list and executes a device `Kokkos::parallel_reduce()` when `evflag` is set ([`src/KOKKOS/bond_fene_kokkos.cpp`](../../../../src/KOKKOS/bond_fene_kokkos.cpp#L70)). Its device `ev_tally()` forms all six bond virial products, including the Newton-bond half contributions when required ([`src/KOKKOS/bond_fene_kokkos.cpp`](../../../../src/KOKKOS/bond_fene_kokkos.cpp#L360)). The returned `ev.v[]` is then copied into the host `virial[6]` array ([`src/KOKKOS/bond_fene_kokkos.cpp`](../../../../src/KOKKOS/bond_fene_kokkos.cpp#L145).

The bond style also copies a one-value error flag to host every call ([`src/KOKKOS/bond_fene_kokkos.cpp`](../../../../src/KOKKOS/bond_fene_kokkos.cpp#L138). That is not the six-stress bottleneck, but it confirms that the current force-style boundary is not fully device-resident.

### Pressure formation

There is no `src/KOKKOS/compute_pressure_kokkos.*` in this checkout. Standard `ComputePressure::compute_vector()` invokes the temperature vector and `virial_compute()` ([`src/compute_pressure.cpp`](../../../../src/compute_pressure.cpp#L283)). `virial_compute()` sums the host `vptr[]` contributions and calls `MPI_Allreduce()` on six host doubles ([`src/compute_pressure.cpp`](../../../../src/compute_pressure.cpp#L331)). This is the exact host operation that the raw six-variable `fix print` path exposes every step.

Temperature has a KOKKOS variant, but it does not solve the cadence problem. `ComputeTempKokkos::compute_vector()` reduces six kinetic components on the device, assigns the reduction result to host `double t[6]`, and then calls `MPI_Allreduce()` into the host vector ([`src/KOKKOS/compute_temp_kokkos.cpp`](../../../../src/KOKKOS/compute_temp_kokkos.cpp#L98)). A single MPI rank removes inter-rank network traffic, but not completion of the device reduction, host result materialization, or the host pressure API.

`compute pressure` has empty KOKKOS data masks and is not marked KOKKOS-native ([`src/compute_pressure.cpp`](../../../../src/compute_pressure.cpp#L140)). KOKKOS fallback handling enables `auto_sync` around non-KOKKOS computes/fixes ([`src/KOKKOS/modify_kokkos.cpp`](../../../../src/KOKKOS/modify_kokkos.cpp#L56)); the atom layer synchronizes device data to host under that condition ([`src/KOKKOS/atom_kokkos.cpp`](../../../../src/KOKKOS/atom_kokkos.cpp#L244)).

## Architecture comparison

### A. `compute pressure/kk`

This is useful only if the consumer is also device-aware. A KOKKOS pressure compute could combine device pair, bond, and kinetic reductions, but the normal LAMMPS `compute_vector()` API returns a host `double *vector`. Calling it from ordinary `fix print` or `fix ave/correlate/long` every timestep would still complete the device work and copy six values to host every timestep.

Therefore this architecture alone does not remove the synchronization. It becomes useful as a lower-level component only when paired with a device consumer or buffer API.

### B. KOKKOS-native buffered stress sampler

Recommended architecture. The initial proof of concept should be a narrow six-channel sampler, conceptually `fix stress/buffer/kk`, but implemented with a force-path hook rather than by reading host `compute pressure` values.

Minimal device data layout:

```text
Kokkos::View<double*[6], DeviceType> stress_buffer;  // [NBUF][Pxx..Pyz]
Kokkos::View<double[6],  DeviceType> pair_step;
Kokkos::View<double[6],  DeviceType> bond_step;
Kokkos::View<double[6],  DeviceType> kinetic_step;
```

The first implementation can avoid three persistent temporaries by having the pair and bond KOKKOS reductions write device-side six-component results exposed through a small sampler interface, followed by one device combine kernel after force evaluation and one device kinetic reduction after final integration. The sampler writes the final pressure/stress row at an index derived from the timestep, not from a host counter.

At `step % NBUF == NBUF-1`, the sampler marks the block complete. The host then performs one block copy, writes a binary record containing the starting timestep and `[NBUF][6]` values, and advances the block epoch. The host must not inspect the current row during ordinary steps.

This is cleaner than modifying the correlator because stress generation and cadence remain independent of the downstream analysis. The required standard-force changes are real, however: current `/kk` styles expose only host `virial[6]` after their reduction.

### C. KOKKOS-aware `fix ave/correlate/long`

This is less attractive as the first implementation. The existing correlator is host code: `end_of_step()` loops through computes, fixes, and variables, evaluates each value, updates its ring/correlation state, and writes output ([`src/EXTRA-FIX/fix_ave_correlate_long.cpp`](../../../../src/EXTRA-FIX/fix_ave_correlate_long.cpp#L450)). A KOKKOS version would need a block-consumption API, new scheduling semantics, and careful preservation of its multi-tau time-origin rules.

It is possible later to add `consume_block(start_step, n, six_values)`, but coupling the first proof of concept to the correlator would make it harder to separate synchronization, buffering, file I/O, and correlation costs. Raw buffered output is the simpler first downstream consumer.

## Narrow rheology sampler

The sampler need only reproduce the six global stress channels:

```text
Pxx Pyy Pzz Pxy Pxz Pyz
```

It must include, without approximation:

- kinetic tensor from the current velocities and masses;
- LJ pair virial from `pair_lj_cut/kk`;
- permanent FENE bond virial from `bond_fene/kk`;
- volume and unit conversion using the same factors as `ComputePressure`.

For this nonassociating proof of concept there is no transient associating term. The sampler should not use only `F·r` as a shortcut unless it is proven equivalent for the exact force/newton/group configuration. The existing standard path distinguishes explicit pair/bond virial tallying and F·r fallback, so reusing the six explicit terms is safer.

## Single-rank design and buffer semantics

The first target is one MPI rank and one GPU. The device reduction produces the global six values locally; the host `MPI_Allreduce` is removed from the per-step path. At flush, one host transfer obtains the complete block. For later multi-rank support, each rank could produce a device block and use a batched host or GPU-aware MPI reduction over `[NBUF][6]`; that is deliberately out of scope for the proof of concept.

Recommended initial sizes:

| NBUF | device payload |
|---:|---:|
| 100 | 4.8 kB |
| 1,000 | 48 kB |
| 10,000 | 480 kB |

Memory is irrelevant; synchronization frequency is the design variable. Start validation at `NBUF=100`, then test 1,000 and 10,000. A single buffer is sufficient for the first correctness proof. Double buffering is useful only after correctness: at a flush, swap the full and filling buffers, enqueue a device-to-host copy for the full buffer, and continue filling the other buffer. It requires explicit stream/fence ownership and a guarantee that the old buffer is never overwritten before its copy completes.

Flush rules must be deterministic:

1. derive the row from the absolute MD timestep and fixed `NBUF`;
2. write exactly one row per timestep;
3. flush only after the row for the block's final timestep is complete;
4. store block start step and row count in the host record;
5. reject a repeated or noncontiguous block during offline validation.

## Correlation strategy

Raw binary block output is the recommended first consumer. It minimizes coupling and makes it possible to compare every stress value against CPU output before any correlation algorithm is involved. Six scalar host values per timestep are cheap to correlate after transfer; the existing decomposition showed the expensive part is obtaining them through the host pressure path, not six-value arithmetic.

The existing `fix ave/correlate/long` path should be added only after raw buffered values pass validation. The clean interface would be a block API that feeds rows to a host correlator without invoking `compute pressure` or any per-timestep host variable evaluation. No GPU multi-tau implementation is justified at this stage.

## Correctness plan

1. Run a deterministic short NVE test from the same data file, with Langevin disabled, identical coefficients, timestep, atom count, and output cadence.
2. Record CPU trusted six-channel pressure and unbuffered KOKKOS six-channel pressure for every step to establish the actual floating-point envelope.
3. Record buffered KOKKOS values with `NBUF=100`, 1,000, and 10,000 and verify exact step coverage, no duplicate rows, and no missing rows.
4. Compare each channel using absolute error, RMS error, maximum error, and relative error against the unbuffered KOKKOS reference. Set the acceptance envelope to ten times the measured unbuffered CPU/GPU discrepancy, with a documented absolute floor for shear components near zero; do not invent a tolerance before measuring it.
5. Compare `G(0)`, short-time `G(t)`, and integrated `eta(t_c)` at several short cutoffs. Require the buffered/unbuffered difference to remain within the measured CPU/GPU numerical envelope plus a separately reported statistical confidence interval.
6. Check the isotropy identity and report componentwise means, `G(0)`, short-time curves, and `eta(t_c)` rather than accepting only aggregate agreement.

The existing stochastic Langevin comparison remains a secondary statistical check; it cannot establish per-timestep equality. The deterministic NVE check is the required implementation gate.

## Future associating extension

`PairAssociating` explicitly disables the F·r virial shortcut and calls `ev_tally_full()` for its shifted FENE contribution ([`src/ASSOCIATING/pair_associating.cpp`](../../../../src/ASSOCIATING/pair_associating.cpp#L15)). Its future `/kk` implementation would need to contribute that virial into the same device stress accumulator; leaving it host-side would reintroduce synchronization whenever the associating force is active.

`FixAssociatingKinetics::end_of_step()` runs only when `ntimestep % nevery == 0` ([`src/ASSOCIATING/fix_associating_kinetics.cpp`](../../../../src/ASSOCIATING/fix_associating_kinetics.cpp#L198)), and current production uses `nevery=100`. It could remain a host-side event/state update initially, provided its required synchronization is accepted every 100 steps and does not force the every-step stress buffer to flush. The standard LJ/FENE stress can remain device-resident every step only if all force contributions to the stress are device-compatible.

## Expected performance and risk

The completed decomposition measured GPU loop medians of 43.59 s for the 50,000-step dynamics baseline, 55.08 s with stress every 10 steps, and 45.45 s with stress every 100 steps. A successful buffered implementation should approach the 43.59 s baseline plus one block transfer per `NBUF` steps. A reasonable engineering target is recovery of at least 90–95% of timing-only throughput at `NBUF=1,000–10,000`; this is a target, not a measurement.

Main risks:

- changing pair/bond KOKKOS reduction ownership and lifetime;
- preserving Newton-pair/Newton-bond half-counting exactly;
- sampling kinetic stress at the same integration point as trusted pressure;
- ordering pair, bond, and final-velocity contributions without host visibility;
- CUDA stream/fence correctness for double buffering;
- future non-KOKKOS force styles silently reintroducing host synchronization;
- extending to MPI ranks without changing the observable.

## Next implementation step

Do not start with `compute pressure/kk` or `fix ave/correlate/long`. First prototype a single-rank `StressAccumulatorKK` interface in the standard KOKKOS force path that exposes device six-component pair and FENE reductions, appends one complete pressure row per step, and writes raw blocks. Keep it behind a separate build and leave all associating files unchanged. Rebuild only `build-kokkos-cuda` after any source change, then run the deterministic NVE validation before performance tuning or correlator integration.
