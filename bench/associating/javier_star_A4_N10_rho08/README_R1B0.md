# R1-B0: nonassociating-star stress validation

R1-B0 validates the six-channel stress machinery in the supplied KG/WCA star
configuration before interpreting associating rheology.  The reference inputs
use the same data, FENE backbone, WCA potential, `T=1`, `dt=0.01`, NVE plus
Langevin damping 2.0, and neighbor settings as R1-A, but define neither
`pair_style associating` nor `fix associating/kinetics`.  Type-2 stickers are
ordinary WCA beads and no transient bonds are created.

`compute pressure` ordering is `xx yy zz xy xz yz`.  With
`Nxy=Pxx-Pyy`, `Nxz=Pxx-Pzz`, and `Nyz=Pyy-Pzz`, pressure versus Cauchy-stress
sign conventions cancel in these autocorrelations.  R1-B0 tests
`C_Nab(t) = 4 C_ab(t)` statistically, orientation averages
`C_shear=(Cxy+Cxz+Cyz)/3`, `C_N=(CNxy+CNxz+CNyz)/3`, and
`R_iso=C_N/(4 C_shear)` only while the shear signal exceeds 5% of its initial
value.  The R1-A canonical estimator is retained, with a **plus** normal term:
`G=V sum(C_shear_channels)/(5 kBT) + V sum(C_normal_channels)/(30 kBT)`.

## Commands

```bash
cd bench/associating/javier_star_A4_N10_rho08
F=Stars_NA4N10C1000rho0.85rhopoly0.8.equilibrated
mpirun -np 8 ../../../build-r1a/lmp -var F "$F" -var REF_EQUIL_STEPS 50000 -log "$F".r1b0.equil.log -in in.r1b0_equilibrate.lmp
mpirun -np 8 ../../../build-r1a/lmp -var F "$F" -var REF_STRESS_STEPS 200000 -log "$F".r1b0.stress.log -in in.r1b0_stress.lmp
python3 analyze_r1b0.py "$F".r1b0.raw "$F".r1b0.gt --out-prefix "$F".r1b0
```

The raw per-step file is validation-only.  The analysis uses an unbiased FFT
autocorrelation at native `dt=0.01`, compares its nearest early-time lag with
the online multi-tau output, writes numerical CSVs, and makes shear, normal/4,
orientation-average, online/offline, and modulus plots when matplotlib exists.
Long-lag differences are retained in the CSV but not treated as an algorithmic
failure once the correlation is noise-dominated.

## Timing and virial regression

Timing runs use the same nonassociating data and report LAMMPS loop time:

```bash
for MODE in 0 1 2; do
  mpirun -np 8 ../../../build-r1a/lmp -var F "$F" -var MODE "$MODE" -var REF_TIMING_STEPS 10000 -log "$F".r1b0.timing.$MODE.log -in in.r1b0_timing.lmp | tee "$F".r1b0.timing.$MODE.log
done
ctest --test-dir ../../../build-associating-tests -R AssociatingPairVirial --output-on-failure
```

Mode 0 is MD only, mode 1 evaluates pressure every step, and mode 2 adds the
six-scalar multi-tau correlator.  The deterministic CTest places an associated
pair at an oblique separation and compares all six `V*P` tensor components to
`r_alpha F_beta`; failure blocks associating rheology.

R1-A associating production now accepts `-var TRAJ_EVERY` and
`-var NETWORK_EVERY` (both default 10000).  It writes compressed unwrapped
polymer coordinates (`id mol type xu yu zu`) and synchronized network
snapshots.  These snapshots intentionally remain separate files because
`write_associating_network` is a validated atomic snapshot writer; archive
them after a production run rather than changing its detailed-balance path.

Gate: pass only if the virial regression passes, online/offline early bins
agree within statistical precision, and isotropic means/tensor identities are
consistent within finite sampling.  Otherwise distinguish an online mismatch
from finite-size/equilibration statistics before returning to R1-A.

## Local validation result

Starting from the supplied configuration, an 8-rank 10,000-step thermostat
settle followed by a 100,000-step reference correlation run passed gate A.
The pressure means were `Pxx=5.13919`, `Pyy=5.14204`, `Pzz=5.14031`; shear
means were 0.00149, -0.000805, and -0.000465. `R_iso(0)=0.9937`; over 17
early bins above the signal floor its mean was 0.967 +/- 0.053. The maximum
early-bin online/offline relative difference was 0.782% (all six channels),
with absolute differences 0.93e-6--5.85e-6. Thus the ordinary KG/WCA system
passes isotropy and correlator agreement within this prototype's sampling.

The cleaned timing benchmark uses no per-step text output in mode 1; rerun the documented three-mode command for host-specific timings.
pressure), and 18.76 s (pressure plus correlator). Pressure evaluation/global
reduction costs about 20% over bare MD; the six-scalar multi-tau addition costs
only about 1.7% beyond that. The pair-virial CTest passed all xx, yy, zz, xy,
xz, and yz components. Therefore anomalous R1-A normal/shear behavior is not
an ordinary correlator defect; inspect associating virial/network statistics
before any long associating production.
