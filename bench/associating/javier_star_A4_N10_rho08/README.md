# Javier star-polymer associating-sticker validation

This is Javier's thesis-system starting point in KG LJ units: 1,000 four-arm
stars with arm length `N=10`, polymer density `rho_poly=0.8`, and total density
`rho=0.85`. The supplied `Stars_NA4N10C1000rho0.85rhopoly0.8.equilibrated.lammpsdat`
is already equilibrated and has no transient bonds.

The data-file header and atom records give this mapping:

| Type | Count | Meaning |
| --- | ---: | --- |
| 1 | 37,000 | ordinary polymer monomer |
| 2 | 4,000 | terminal sticker (four per star) |
| 3 | 2,563 | solvent |

Permanent backbone topology remains unchanged: `bond_style fene` with
`K=30`, `R0=1.5`, `epsilon=sigma=1`, and `special_bonds fene`. The input uses
`pair_style hybrid/overlay`: `lj/cut` supplies WCA excluded volume for every
type pair through `r_WCA=2^(1/6)`, while `associating` adds only its shifted
log-FENE attraction for active type-2 pairs, with `Ee=8`.

Reactions operate only on type-2 stickers with `Ea=4`, `T=1.0`,
`r_assoc=2^(1/6)`, `timestep=0.01`, and `Nevery=100`. The 100-step cadence is
the remembered production setup; here it is the initial cadence to validate
the realistic configuration. `nu0=10` is intentionally benchmark-specific:
a repository search found no Javier legacy reversible-bond/tau-leap prefactor
or documented transferable value. It is selected to make the short run show
creation and break events, and is not physically calibrated or numerically
equivalent to the old tau-leap implementation.

From this directory, run:

```
mpirun -np 1 ../../../build-associating/lmp -in in.associating.lmp
mpirun -np 2 ../../../build-associating/lmp -in in.associating.lmp
```

The thermo columns after energy are `f_kinetics[1]` active transient bonds,
`f_kinetics[2]` cumulative creations, and `f_kinetics[3]` cumulative breaks.
The 5,000-step run is a short validation (50 chemical sweeps), not a
production benchmark.

## R1-C1: corrected-virial full-system equilibrium rheology pilot

R1-C1 is a single full-system stress-correlation pilot after the
`PairAssociating::no_virial_fdotr_compute = 1` correction. It preserves the
1,000-star, `N=10`, `rho_poly=0.8`, `rho=0.85` model, `Ee=8`, `Ea=4`, `T=1`,
`dt=0.01`, `Nevery=100`, `r_assoc=2^(1/6)`, `nu0=10`, and Langevin damping 2.
The pilot defaults to 100,000 equilibration steps when starting from the
original no-transient-bond data and 1,000,000 production steps, with raw six
component pressure every step, online `fix ave/correlate/long`, unwrapped type
1/2 coordinates, and network snapshots every 10,000 steps.

If the validated associating continuation restart
`Stars_NA4N10C1000rho0.85rhopoly0.8.equilibrated.r1a.equil_cont.restart` is
present, the launcher reuses it by default and records that choice. This
avoids unnecessary re-equilibration; set `REUSE_RESTART=0` to perform the
100,000-step fresh equilibration. R1-C1 never reuses old stress files.

Run on the dedicated Linux host:

```bash
LMP=~/lammps/build-r1a/lmp \
MPI_NP=8 EQUIL_STEPS=100000 PROD_STEPS=1000000 \
./run_r1c1_linux.sh
./status_r1c1.sh
python3 analyze_r1c1_full_rheology.py \
  r1c1_runs/production/production.raw \
  --online r1c1_runs/production/production.gt \
  --out r1c1_runs/r1c1
```

The analyzer validates the seven-column raw format, computes `Cs`, `CN/4`,
`D`, the rotationally averaged `G(t)`, zero-lag `R_iso`, pressure means,
diagnostic modulus crossings, a cumulative Green--Kubo integral, a tail noise
floor, and an explicitly diagnostic plateau check. It compares early offline
lags with the online multi-tau file. It does not claim a zero-shear viscosity
from this single trajectory; its purpose is to decide whether R1-C2 needs
multiple replicas, longer trajectories, or both.

Every output root records the repository Git SHA, executable path and
SHA-256, MPI rank count, seeds, run lengths, cadence, start state, and
trajectory mode. Completion markers must match that identity exactly; a
configuration or executable change requires a new output root. The launcher
uses compressed polymer trajectories when the executable advertises
`COMPRESS`, otherwise it falls back to plain text. This prevents the stale
executable incident seen before R1-B4. Any future source change under `src/`
requires rebuilding the executable before running R1-C1.

## Pre-R1-C2 event logging gate

Sparse network snapshots cannot resolve short sticker detachment and
reattachment flickers, so the kinetics fix can optionally record each accepted
creation or break directly from its existing replicated global sweep:

```
fix kinetics stickers associating/kinetics 100 492845 10.0 4.0 1.0 ${rwca} \
  event_log r1c2/events.dat
```

The file is written only by rank 0, is opened once with buffered output, and
contains `timestep event_type sticker_i sticker_j molecule_i molecule_j`, with
`event_type` equal to `C` or `B` and canonical `sticker_i < sticker_j`.
An event stream must be combined with an initial active network. Capture that
state with the existing traversal before the run:

```
write_associating_network r1c2/initial.network fix kinetics
```

The event stream is a bare accepted-event history; it is not Javier's
renormalized bond lifetime by itself. Replay begins from the initial network
and applies events sequentially, so temporary detachments followed by
reattachment to the same partner can be recognized later.

The instrumentation gate is prepared but not yet evaluated on the dedicated
host. Run `benchmark_r1c2_event_logging.sh` with a rebuilt executable and the
trusted R1-C1 restart. It performs one warm-up and alternating OFF/ON
measurements, keeps outputs separate, records executable provenance, and applies
the provisional <=2% PASS, <=5% ACCEPTABLE, >5% FAIL/redesign thresholds.
No long R1-C2 production run is authorized by this section.

### Frozen dedicated-host result

The dedicated-host gate completed with the same executable, starting restart,
seeds, and configuration for three OFF and three ON runs at MPI rank count 8
and 100,000 MD steps. The machine-readable result is
`r1c2_event_logging_benchmark_result.json`.

| mode | wall times (s) | median wall (s) | median steps/s |
| --- | --- | ---: | ---: |
| OFF | 259.66, 259.84, 260.35 | 259.84 | 384.8522 |
| ON | 260.36, 260.30, 260.47 | 260.36 | 384.0836 |

The measured median slowdown is 0.2001%, well below the provisional 2% PASS
threshold. Both modes accepted 1,034 creations and 1,047 breaks. OFF logged
zero events; ON logged 2,081 events in 60,986 bytes, or 29.31 bytes/event.
The 0.20% value should not be read as high-precision: it is effectively
negligible compared with the observed run-to-run timing scatter. The gate is
**PASS**, and event logging is frozen as enabled for future associating
production runs.

The instrumentation implementation was commit `0a01eab2de`; the benchmark
provenance Git SHA was `695c713a9ab8b50f73d9c6d41329a967c2843223`, and the
dedicated-host executable SHA-256 was
`e403ee6a1a793b1fca3974a34b0a12c4e5c79fe2f948b4141265815072bb7862`.

## Nonassociating full-system control

The next control is prepared but not run at production length. It reads the
original no-transient-association data file
`Stars_NA4N10C1000rho0.85rhopoly0.8.equilibrated.lammpsdat`, preserving 1,000
four-arm stars, 37,000 type-1 beads, 4,000 type-2 terminal sticker beads,
2,563 type-3 solvent beads, 43,563 atoms, and 40,000 permanent FENE bonds.
Type 2 remains chemically inactive but is not remapped; every type pair uses
only WCA `lj/cut` excluded volume. There is no transient association style,
kinetics fix, or network output in `in.nonassoc_control.lmp`.

The planned production is `PROD_STEPS=500000`, `dt=0.01` (`T=5000`), with
ordinary six-component pressure printed every step and the same six-channel
`fix ave/correlate/long` estimator as R1-C1. Diffusion output is now written
directly by LAMMPS with `compute chunk/atom molecule` and `compute com/chunk`
for the 1,000-star polymer group. At the default `COM_EVERY=100`, each frame
contains `row molecule_id xu yu zu` at `Delta t=1`; `nchunk once`, `ids once`,
and `compress yes` make the row-to-molecule mapping deterministic, while the
coordinates are unwrapped. The default 5,000-frame COM file is estimated at
about 320 MB uncompressed (the launcher records the estimate). The old full
atom dump is disabled; set `FULL_TRAJ=1` only for debugging/regression.

Prepare/run on the dedicated host with:

```
LMP=~/lammps/build-r1a/lmp MPI_NP=8 PROD_STEPS=500000 \
  ./run_nonassoc_control_linux.sh
OUT=nonassoc_control_runs ./status_nonassoc_control.sh
```

The launcher rejects an existing output root and records Git SHA, executable
SHA-256, input-data SHA-256, seeds, stress/COM cadence, and run length. It
does not reuse an associating restart and performs no extra thermalization.
The analyzer command after production is:

```
python3 analyze_nonassoc_control.py \
  nonassoc_control_runs/production/control_nonassoc.raw \
  nonassoc_control_runs/production/control_nonassoc.com \
  --out-dir nonassoc_control_runs/analysis
```

It reads the per-star unwrapped COM rows directly and produces
nested-duration rheology, fixed-cutoff Green–Kubo, block uncertainty, star-COM
MSD, local logarithmic slope, and diffusion diagnostics. A short regression
smoke test compares this output with COMs reconstructed from an optional atom
dump; the latter is not needed for production analysis.
The analyzer makes no single terminal-time claim from a `1/e` crossing: it
reports logarithmically binned block-SEM and block-SD loss-of-resolution lags,
the largest contiguous statistically resolved slow-tail range, and a
cumulative integral truncated at the first binned SEM crossing. Fixed-cutoff
Green–Kubo analysis separates prefix-duration convergence at fixed `t_c` from
a plateau in `t_c`: it reports `T_min` at 10%, 15%, and 25% tolerances, the
required `T/t_c`, and tests a viscosity plateau only with duration-converged
estimates. The cutoff grid extends through 1000 where available; without a
supported plateau the run estimates fixed-cutoff integrals but does not
establish zero-shear viscosity. Diffusion fits are restricted to windows
ending by `T/5`; candidate-window `D` values are reported with mean/median
local exponents in the 250–500, 500–1000, and 250–1000 windows. A stricter
sustained `|alpha-1|<=0.1` diagnostic is reported separately from any claim
of perfectly asymptotic diffusion. No full production simulation or long
associating R1-C2 continuation has been launched.

## Frozen nonassociating control and staged R1-C2 preparation

The completed nonassociating control is frozen in
`nonassoc_control_conclusions.json`. It gives `G(0)≈66.34`, approximately
unity zero-lag isotropy, a coarse block-resolved slow-tail range to `t≈136.8`,
and no defensible single terminal time. The descriptive integral relaxation
diagnostic is `0.1756±0.0194`, but is dominated by the short-time modulus.
Fixed-cutoff Green–Kubo convergence empirically requires roughly `T/t_c≈50`
for useful intermediate cutoffs (`t_c=10–50`); the cutoff values do not form
a convincing plateau, so `eta_0` remains not established. The frozen control
diffusion result is `D_nonassoc≈1.77e-3`, with a few-percent fit-window
variation and motion approaching/consistent with Fickian behavior, not a claim
of a perfect asymptotic `alpha=1` plateau.

The staged continuation is prepared but not authorized or launched.
`run_r1c2_staged_linux.sh` requires `AUTHORIZE_R1C2=YES`, starts from the
trusted `r1c1_runs/production/production.restart`, never resets the timestep,
and uses stages of 1,000,000, 1,000,000, and 2,000,000 steps for cumulative
`T=20000, 30000, 50000`. Each stage has separate raw stress, online
correlation, direct per-star unwrapped COM, event log, final network, restart,
provenance, and completion files. Stage 1 additionally writes
`initial.network` before chemistry advances.

The initial-network command is preceded by `run 0` after pair, dynamics, and
kinetics setup. This initializes communication/ghost state without advancing
the timestep or executing an end-of-step chemistry sweep; production stress,
COM, and correlation fixes are defined afterward. The first failed attempt
left `r1c2_runs/stage1/` without a completion marker. It must be inspected and
then removed manually before retrying Stage 1; the launcher intentionally
refuses to overwrite it.

The exact event format is `timestep event_type sticker_i sticker_j molecule_i
molecule_j`, with `C/B` events and canonical sticker IDs. The staged analyzer
`analyze_r1c2_staged.py` concatenates R1-C1 and stage raw stress files, removes
only exact duplicate boundary rows, rejects gaps and inconsistent overlaps,
replays the initial network plus events, and reports bare and
Javier-renormalized survival. Same-partner detach/reattach is merged;
third-partner binding terminates the pending renormalized episode. Lifetimes
are explicitly left-censored at the R1-C2 start and right-censored at the
final observation where appropriate.

Stage 1 command, when explicitly authorized on the dedicated Linux host:

```bash
AUTHORIZE_R1C2=YES LMP=~/lammps/build-r1a/lmp MPI_NP=8 STAGE=1 \
  ./run_r1c2_staged_linux.sh
```

Stage 1 has since completed. Its repaired analysis is written to
`r1c2_runs/analysis_stage1_repaired2/` and reports `G(0)=68.1857` and
`R_iso(0)=0.997537`, with exact network replay valid. Bare episodes have
`1774` left-censored records (equal to the `1774` initial edges), while the
renormalized survival has median `1967` time units and `1/e` time `2844`
time units; both retain explicit censoring metadata and timestep columns.
The observed COM segment begins at cumulative step `1000000` but is analyzed
with its first frame as the local MSD origin. It gives candidate associating
diffusion fits around `5.1–6.6e-4`, not yet stable and not strictly Fickian.
The expanded fixed-cutoff diagnostics find a descriptive duration-converged
range through `t_c=500`; this is not an automatic zero-shear-viscosity claim.
Stage 2 has now also completed. Its repaired analysis is written to
`r1c2_runs/analysis_stage2_repaired2/`; the duration grid reaches `T=30000`,
and the longer prefixes remove the former apparent `t_c=500–1000` plateau.
The Stage-2 summary therefore reports `eta_0_status=not established`.
Candidate COM slopes are mutually similar around `4.7e-4`, but the measured
logarithmic exponent remains about `0.62` over the principal windows, so this
is labeled an effective candidate slope rather than a stable long-time
diffusion coefficient. The diagnostic stage decision was `CONTINUE`; Stage 3
has since completed at cumulative `T=50000`.

Stage 3 analysis at cumulative `T=50000` extends the stress cutoff and
logarithmic-tail scans through `5000`. Cutoffs `1500` and `2000` have
block-supported rows but fail the 25% duration/block stability test; `3000`
and `5000` are retained as descriptive values but are unsupported by the
safe-lag/block criteria. The first enlarged-range loss of stress resolution
is at about `t=2911` by both block SEM and SD. COM alpha rises from about
`0.65` (`1000–2000`) to `0.89` (`8000–16000`), a trend toward 1 without
passing the strict asymptotic Fickian criterion; the effective candidate slope
therefore remains non-asymptotic. Sticker-lifetime summaries are frozen
separately from unresolved rheology and diffusion. The analyzer also writes a
planning-only `T=100000` section; `T/t_c≈50` is used there as a nonassociating
empirical heuristic, not as a convergence claim.

Stage 4 is prepared but intentionally not launched. After confirming the
Stage-3 completion marker and restart checksum, launch it from this directory
with:

```bash
AUTHORIZE_R1C2=YES LMP=~/lammps/build-r1a/lmp MPI_NP=8 STAGE=4 \
  ./run_r1c2_staged_linux.sh
```

It defaults to `5,000,000` steps from
`r1c2_runs/stage3/production.restart`, writes the Stage-4 raw/GT/COM/event,
network, restart, provenance, and completion files under
`r1c2_runs/stage4/`, and refuses overwrite or a mismatched Stage-3 restart.
The cumulative analyzer should then be run with `--stage-dir` for stages 1,
2, 3, and 4; its existing dynamic duration grid reaches `T=100000`.

Because the existing Stage 1–3 completion markers predate output-restart
hashes, run this one-time migration before Stage 4:

```bash
OUT=$PWD/r1c2_runs ./backfill_r1c2_restart_provenance.sh 1 2 3
```

It preserves the original `complete` files and creates immutable
`output_restart_provenance.txt` sidecars after checking each completion marker,
restart path, and SHA-256. Re-running the command only verifies those sidecars;
it does not launch MD.
