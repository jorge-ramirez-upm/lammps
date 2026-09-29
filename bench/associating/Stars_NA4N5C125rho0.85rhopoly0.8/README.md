# R1-B1: reduced associating stress-isotropy validation

This diagnostic system has 2,789 atoms: 125 A=4, N=5 stars (2,625 polymer
beads including 500 type-2 stickers), 2,500 permanent FENE bonds, and 164
solvent beads.  It uses the validated R1-A association model: WCA, KG FENE
(30,1.5), associating FENE (30,1.5,Ee=8), `Ea=4`, `nu0=10`, `Nevery=100`,
`dt=.01`, and NVE/Langevin damping 2.  It is a stress diagnostic, not rheology
production.

`in.equilibrate.lmp` starts without transient bonds, equilibrates the network,
and writes energy/bond/turnover diagnostics, low-frequency coordinates, and
network snapshots. `in.stress_validation.lmp` starts from that restart; each
replica has explicit independent Langevin and kinetics seeds. Replicas sharing
one restart are not fully independent at t=0; use separated equilibration
snapshots for the six-replica production default when possible.

Run:
```bash
F=Stars_NA4N5C125rho0.85rhopoly0.8.equilibrated
mpirun -np 8 ../../../build-r1a/lmp -var F "$F" -var EQUIL_STEPS 100000 -in in.equilibrate.lmp
for r in 1 2 3 4 5 6; do mpirun -np 8 ../../../build-r1a/lmp -var F "$F" -var REPLICA $r -var LANGEVIN_SEED $((48279+r)) -var KINETICS_SEED $((492845+r)) -var REF_STRESS_STEPS 200000 -in in.stress_validation.lmp; done
python3 analyze_stress_validation.py "$F".r1b1.r*.raw --out "$F".r1b1
```

The analysis uses unbiased FFT autocorrelations and compares them to the
unchanged online multi-tau correlator. It reports replica mean and SEM for
`C_s`, `C_N/4`, `D=C_N/4-C_s`, and uses `|C_s|>2 SEM(C_s)` for ratios. The
canonical R1-A modulus is `G_shear=V sum(Cshear)/5kBT`,
`G_normal=V sum(Cnormal)/30kBT`, and their sum.

R1-B0 established that the nonassociating KG/WCA reference and online
correlator satisfy isotropy, and the associating pair-virial tensor CTest
passes. The full R1-A single trajectory agreed at t=0 but separated after its
shear signal decayed. R1-B1 tests whether replica statistics resolve that.

## Local smoke

A 100k-step equilibration reached an active population around 230 while
creation/breaking continued. Three 100k stress replicas (same restart,
different streams) had online/offline maximum early relative differences below
0.039%. However `R_iso(0)=0.792` and the largest signal-window `|D|/SEM` was
16.9. This is a preliminary **B** indication, not a final six-independent-
replica gate: generate separated starting states and complete six 200k
replicas before declaring a model-state violation.

## R1-B2 result (1,000,000 steps)

B2 used 100 complete 10,000-step blocks (dt=.01) on 8 MPI ranks. The total
block-mean R0 is 1.00195 +/- 0.00297 SEM, so one is statistically compatible.
Component R0 (block mean +/- SEM): kinetic 1.00236 +/- .00326; WCA 1.02107 +/-
.00707; permanent FENE 1.00388 +/- .00351; associating FENE 1.04353 +/-.01679.
The maximum instantaneous tensor reconstruction residual is 1.78e-14. The
largest shear cross term is WCA--associating, -0.002897, which cancels a
substantial part of their individual shear variances rather than causing an
anisotropy. Permanent-bond orientation is isotropic (maximum mean diagonal
deviation .00853 from 1/3); active-bond orientation remains somewhat anisotropic
(.0460), but the total stress is isotropic. Adjacent 100-time-unit snapshots
have bond persistence Q=0.958 +/- .014 (SD). Gate: **A**, subject to the
finite-sample active-network orientation caveat.

### Method and reproduction

R1-B2 tests whether the B1 discrepancy already exists at zero lag, upstream of
the correlation estimator. Run `mpirun -np 8 ../../../build-r1a/lmp -var F
"$F" -var B2_STEPS 1000000 -in in.zero_lag_isotropy.lmp`, then
`python3 analyze_zero_lag_isotropy.py "$F".r1b2.raw --trajectory
"$F".r1b2.lammpstrj --data "$F".lammpsdat --out "$F".r1b2`. The input writes
every-step total, kinetic (`ke`), WCA (`pair/hybrid lj/cut`), permanent (`bond`),
and associating (`pair/hybrid associating`) pressure tensors. These are direct
LAMMPS global-virial selections; the component sum check excludes hidden stress
terms. The analysis uses Cs0=mean(sum(Pshear^2)/3),
Cn0=mean(sum(N^2)/3), and R0=Cn0/(4Cs0), with 10,000-step contiguous blocks
for SEM. It also writes component means, cross terms, stationarity tables,
orientation tensors, snapshot persistence, CSV outputs, and optional plots.

## R1-B2.1: time-dependent isotropy from the stationary B2 trajectory

R1-B2.1 reuses the B2 million-step raw trajectory; no new MD is required.
`analyze_time_dependent_isotropy.py` computes unbiased FFT ACFs independently
in contiguous 20x50k and 10x100k blocks, then writes their means and SEMs for
`Cs`, `Cn/4`, `D=Cn/4-Cs`, and the signal-gated ratio. A lag is useful only if
`abs(Cs)>2 SEM(Cs)`. The script also defines asymmetric network persistence as
`Q(dt)=mean(|E(t) intersection E(t+dt)|/|E(t)|)` over all valid snapshot pairs.

```bash
python3 analyze_time_dependent_isotropy.py "$F".r1b2.raw --out "$F".r1b21 \
  --network-glob "$F".r1b2.network.*.dat
```

Both block schemes reproduce `Riso(0)=1.001635`, consistent with B2. The
useful window reaches the common 250-time-unit output limit. The largest useful
`abs(D)/SEM(D)` is 4.80 (50k blocks, time 60.16) and 8.92 (100k blocks, time
3.65); their location and sign are not stable against block size. Respectively
90.4% and 88.7% of useful lags are within 2 SEM of zero.

The persistence curve gives `Q(100)=0.9581`; it first falls below 0.9 at 300,
below 0.75 at 800, and below 0.5 at 1800 time units. Thus 50k steps (500 time
units) and 100k steps (1000 time units) remain strongly network-correlated
(`Q=0.819` and `0.675`), so their nominal block SEMs are not independent-sample
errors. Gate: **B** — apparent finite-time deviations remain, but are dominated
by persistence/block dependence; longer stationary sampling is required before
claiming a time-dependent anisotropy.

## R1-B2.2: gapped-window isotropy test

No MD was run. `analyze_gapped_isotropy.py` applies the same unbiased FFT ACF
to five 50k-step windows (maximum lag 25k) and treats them as weakly correlated
sampling units. Scheme A starts at steps `10000, 240000, 470000, 700000,
930000`; scheme B is the deterministic offset `110000, 320000, 530000,
740000, 950000`. Every window has length 50k steps; their ends are start plus
50k. The corresponding nearest start separations are 2300 and 2100 time units,
where mean asymmetric network persistence is 0.403 and 0.431, respectively.

```bash
python3 analyze_gapped_isotropy.py "$F".r1b2.raw --out "$F".r1b22 \
  --network-glob "$F".r1b2.network.*.dat
```

The window-mean zero-lag ratios are 1.00250 (A) and 1.00139 (B), preserving
the B2 result. Both schemes remain useful through the 250-time-unit analysis
limit. Their largest `abs(D)/SEM(D)` values are 21.14 at time 236.68 with
negative D (A), and 14.38 at time 179.83 with positive D (B). Only 74.8% and
75.1% of useful lags are within 2 SEM. Their all-pair mean Q values are 0.227
and 0.252, so these are weakly—not exactly—independent windows.

Unlike B2.1 contiguous blocks (maxima 4.80 at 60.16 for 50k, and 8.92 at 3.65
for 100k), gapping does not produce a common sign, magnitude, or lag location.
Gate: **B, inconclusive**. Independent long trajectories remain necessary;
neither correlated contiguous blocks nor these five weakly correlated windows
support a robust finite-time discrepancy.

## R1-B3: independent-ensemble workflow

R1-B3 is deliberately not run here. It creates six independent associating
networks from the original unassociated data file, equilibrates each for
`EQUIL_STEPS=500000` (5000 time units), and then produces `PROD_STEPS=1000000`
steps. Replica seeds `(velocity, Langevin, kinetics)` are respectively
`(184729,284729,384729)` through `(184734,284734,384734)` in replica order.

```bash
LMP=/path/to/lmp MPI_NP=8 ./run_r1b3_linux.sh
./status_r1b3.sh
python3 analyze_r1b3_ensemble.py r1b3_runs --out r1b3_runs/r1b3
```

The launcher defaults to sequential replicas; set `PARALLEL_REPLICAS` only
when host resources allow it. It uses `equil.complete` and `production.complete`
markers, skips only marked stages on resume, and never treats a partial raw
file as complete. Outputs live under ignored `r1b3_runs/replicaNN/`. Plain
unwrapped trajectories are the default. Set `TRAJ_EXT=lammpstrj.gz` only with
a LAMMPS build that includes COMPRESS support.

The analysis uses unbiased raw-stress FFT ACFs and the six replicas—not blocks—
as statistical units. It writes per-replica zero-lag ratios and ensemble means,
SD, and SEM for Cs, CN/4, D, and signal-gated Riso. It must also pass early-lag
online/offline comparison and equilibration-drift review before assigning the
A/B/C scientific gate; no R1-B3 outcome is claimed until those runs complete.
