# K1: reversible-dimer kinetic and thermodynamic validation

This is an isolated WCA sticker-fluid validation of `pair associating` plus
`fix associating/kinetics`. It does not use or alter the star-polymer benchmark.
All beads are identical, may have one partner, and therefore represent
`A + A <=> B` exactly at the species-count level.

## Chosen pilot/production setup

`N=256`, `rho=0.05`, cubic side `17.235`, `T=1`, WCA cutoff `2^(1/6)`,
`dt=0.005`, `nu0=20`, association cutoff equal to WCA cutoff, and `20,000`
WCA/Langevin warm-up steps are used. Production is `100,000` steps (`500` MD
time), sampled every 100 steps, with four independent replicas. This dilute
fluid has mean separation about 2.7, avoids persistent many-body contacts, but
has enough collision opportunities at the chosen accelerated validation
prefactor. The pilot is the central condition `(Ea,Ee,Nevery)=(4,4,100)`.

The production grid is K1-A: `Ee=4`, `Ea=2,3,4,5,6`; K1-B: `Ea=4`,
`Ee=2,4,6,8`; and K1-C: `(Ea,Ee)=(4,4)`, `Nevery=50,100,200`, all with four
replicas. Duplicate central cases are safely skipped. A completed nonempty
`results/*.dat` is the restart marker.

## Theory and fit

Let `C=[A]0=[A]+2[B]`. The model is

`dB/dt = kf (C-2B)^2 - kb B`.

For positive `kf,kb`, define

`D=kb^2+8 kb kf C`,
`b±=(4 kf C+kb ± sqrt(D))/(8 kf)`, and
`R(t)=(b-/b+) exp(-sqrt(D)t)`.

For the deliberately unassociated initial state, the analytic fit curve is

`B(t)=(b- - R(t)b+) / (1-R(t))`.

`analysis/fit_dimer_kinetics.py` directly least-squares fits this complete
transient in log-rate coordinates for every replica; it reports `kf`, `kb`,
`Keq_kin=kf/kb`, and a block-mean direct equilibrium value
`Keq_eq=<[B]/[A]^2>`. Across-replica standard errors are in
`*_conditions.csv`. It also fails if `A+2B` or
`B(t)-B(0) = (creations-breaks)(t)-(creations-breaks)(0)` is not conserved.

The requested tests are assessed by the reported log-linear slopes:
`ln(kf)=Cf-Ea/T`, `ln(kb)` versus `Ee`, and
`ln(Keq)=CK+Ee/T`; intercepts are deliberately fitted, not imposed. K1-C
compares the rates in MD-time units and both equilibrium constants.

## Linux-host commands

From the repository root:

```bash
cmake -S cmake -B build-k1 -D PKG_ASSOCIATING=on -D BUILD_MPI=on -D BUILD_TESTING=on
cmake --build build-k1 -j"$(nproc)"
./bench/associating/dimer_kinetics/run_campaign.sh ./build-k1/lmp pilot 1
python3 bench/associating/dimer_kinetics/analysis/fit_dimer_kinetics.py \
  'bench/associating/dimer_kinetics/results/Ea4_Ee4_N100_r*.dat' \
  --summary bench/associating/dimer_kinetics/results/pilot.csv
./bench/associating/dimer_kinetics/run_campaign.sh ./build-k1/lmp full 1
python3 bench/associating/dimer_kinetics/analysis/fit_dimer_kinetics.py \
  'bench/associating/dimer_kinetics/results/*.dat' \
  --summary bench/associating/dimer_kinetics/results/k1.csv
```

Use a suitable MPI rank count in the final argument on a dedicated host.
Expected outputs are one `.dat` and `.log` per condition/replica,
`pilot.csv`, `k1.csv`, and their `_conditions.csv` companions. The campaign
has 40 unique trajectories (the central condition is shared); at 120,000
steps each it is 4.80 million MD steps total. Wall cost should be estimated
from the two-trajectory pilot (`pilot wall time * 20`) because the global
association scheduler is deliberately not optimized in K1.

A limitation: this validates an effective, well-mixed macroscopic rate only.
At higher density, strong binding, or cadence beyond the collision-resolution
window, spatial correlations and the discrete reaction scheduler can make
these fitted constants state- and cadence-dependent rather than intrinsic.
