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
