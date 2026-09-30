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

## R1-B3.1: zero-lag B2/B3 forensic comparison

No MD is part of this check.  Run the forensic analyzer only where the existing
ignored raw results are present:

```bash
python3 analyze_r1b31_zero_lag.py \
  Stars_NA4N5C125rho0.85rhopoly0.8.r1b2.raw r1b3_runs \
  --out r1b3_runs/r1b31
```

The primary calculation is directly from instantaneous samples:
`Cs0=mean((Pxy^2+Pxz^2+Pyz^2)/3)` and
`CN0/4=mean(((Pxx-Pyy)^2+(Pxx-Pzz)^2+(Pyy-Pzz)^2)/12)`.
The FFT value is only an independent lag-zero check.  The three CSV outputs
contain the compact B2/B3 thermodynamic and tensor-mean table, all consecutive
100k-step and half-trajectory values, and restart continuity evidence.  The
online comparison uses the actual zero-lag row of each `production.gt`.

The observed headline values that motivated this check are:

| existing data | independent units | R0 mean | spread |
|---|---:|---:|---:|
| B2, one 1M-step trajectory | 100 x 10k blocks | 1.00195 | 0.00297 SEM |
| B3, six 1M-step replicas | 6 replicas | 0.8782157 | 0.0102303 SD |

The ignored B2/B3 raw outputs are not stored in this source checkout, so the
individual-replica, window, thermodynamic, online, and continuity rows cannot be
truthfully reconstructed from the headline aggregate.  Consequently B3.1 is
**not yet classifiable as A--D from the tracked files alone**.  In particular,
the aggregate mismatch must not be called D without first demonstrating both
stationarity and restart continuity.  No finite-time `D(t)` interpretation and
no R1-B3 gate follows from this observation.

### Side-by-side input audit

The following is the complete list of input differences that can affect the
sampled ensemble or the reported total tensor; output-only differences are
included explicitly so that they are not mistaken for physics changes.

* **Starting ensemble:** B2 reads the already associated
  `${F}.r1b1.equil.restart`; B3 equilibration reads the original unassociated
  `${DATA}`, creates new Gaussian velocities, runs 500k steps, and production
  reads that replica's `equil.restart`.  Thus starting coordinates, velocities,
  transient network, and amount/history of equilibration differ.
* **Random streams:** B2 uses fixed Langevin/kinetics seeds 58280/592846.  Each
  B3 replica supplies distinct velocity/Langevin/kinetics seeds.  Production
  reuses its replica's named Langevin and kinetics seed when restoring fixes.
* **Image handling:** B3 equilibration calls `reset_atoms image stars`; B2 does
  not.  This changes image flags/unwrapped output, and should not change wrapped
  forces or the pressure tensor, but is retained as a protocol difference.
* **Time origin and staging:** B2 resets the timestep after reading its restart.
  B3 equilibration does not explicitly reset it; B3 production resets it after
  reading the staged restart.  B3 has an explicit equilibration-to-production
  restart boundary that B2 does not have within its measured trajectory.
* **Tensor definition:** B2's `compute ptotal all pressure thermo_temp` and
  B3's `compute press all pressure thermo_temp` request the same total pressure
  definition.  B2 additionally computes selected kinetic, WCA, permanent-bond,
  and associating tensors; these diagnostic computes do not alter forces.
* **Sampling/output fixes:** both write instantaneous total tensor values every
  step. B2 also writes PE and kinetics counters in the same row and takes
  network/coordinate snapshots every 10k. B3 writes temperature, PE, and
  counters every 1k in a separate diagnostic file, takes the same 10k
  snapshots, and runs `ave/correlate/long`. These are output/sampling
  differences, not changes to the pressure tensor.
* **Run organization:** B2's loop assumes an integral number of 10k blocks.
  B3 supports a final remainder and has separate equilibration and production
  loops.  At the present 1M/500k lengths there is no remainder.

Everything else that controls dynamics is identical in the three inputs:
units, atom style, boundaries, special bonds, FENE and WCA/associating
coefficients, neighbor/communication settings, timestep, NVE plus Langevin
(T=1, damping=2, zero net random force), and associating kinetics parameters.
Therefore none of the audited text alone establishes a pressure-definition
error.  The starting-ensemble/equilibration difference is a viable **C** only
if the existing raw diagnostics reproduce a corresponding stationary-state
difference; restart behavior is **B** only if the continuity table demonstrates
a discontinuity rather than ordinary turnover between non-immediate snapshots.

## R1-B3.2: controlled pressure-compute test

This test deliberately reuses one existing B3 `equil.restart`; it performs no
equilibration.  The single input defines only `ptotal`, while the multi input
defines the exact B2 set (`ptotal`, `pke`, `pwca`, `pperm`, and `passoc`).  Both
write a dedicated `run0.dat` before advancing time, every-step pressure/energy/
kinetics data, and sorted coordinate/velocity snapshots.  They use identical
Langevin and kinetics seeds and otherwise identical commands.

```bash
LMP=/path/to/lmp \
RESTART=$PWD/r1b3_runs/replica01/equil.restart \
MPI_NP=8 ./run_r1b32_linux.sh
cat r1b32_run/comparison.json
```

`comparison.json` reports the exact per-component maximum total-pressure
difference, run-zero component-sum residual, mean tensor, mean diagonal
difference, six population variances, `Cs(0)`, `CN(0)/4`, and `R0`.  It also
compares the sorted state dumps byte for byte.  The analyzer exits nonzero for a tensor difference above its default `1e-12`
roundoff tolerance or any state-file difference; `--atol` can set a documented
alternative tolerance. Bitwise total-tensor equality is reported separately.

The existing B3 restarts and raw outputs are ignored run artifacts and are not
present in this checkout.  Therefore no B3.2 numerical result is claimed in
the tracked repository: the exact six run-zero and trajectory differences
must be read from `comparison.json` produced beside the existing restart.
If the primary comparison fails, isolate the observer by making four copies of
the single input and adding, in order, just `pke`, `pwca`, `pperm`, or `passoc`
(and its columns in `fix raw`), always starting from the same restart with the
same rank count and seeds.  Do not chain final restarts.  The first one-at-a-
time case whose total tensor differs identifies the compute requiring a
smaller reproducer; no physics or virial implementation should be changed on
the strength of this diagnostic alone.
