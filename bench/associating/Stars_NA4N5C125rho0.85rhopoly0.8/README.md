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
