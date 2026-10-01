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
`fix ave/correlate/long` estimator as R1-C1. Polymer types 1 and 2 are dumped
as unwrapped `id mol type xu yu zu` every 100 steps (`Delta t=1`). This is
5,000 frames × 41,000 polymer atoms; using a conservative 50 bytes/atom-line
estimate gives about 10.25 GB uncompressed. The launcher selects compressed
output when the executable supports it and records the estimate in
`provenance.txt`.

Prepare/run on the dedicated host with:

```
LMP=~/lammps/build-r1a/lmp MPI_NP=8 PROD_STEPS=500000 \
  ./run_nonassoc_control_linux.sh
OUT=nonassoc_control_runs ./status_nonassoc_control.sh
```

The launcher rejects an existing output root and records Git SHA, executable
SHA-256, input-data SHA-256, seeds, stress/trajectory cadence, compression,
and run length. It does not reuse an associating restart and performs no extra
thermalization. The analyzer command after production is:

```
python3 analyze_nonassoc_control.py \
  nonassoc_control_runs/production/control_nonassoc.raw \
  nonassoc_control_runs/production/control_nonassoc.lammpstrj.gz \
  --out-dir nonassoc_control_runs/analysis
```

It produces nested-duration rheology, fixed-cutoff Green–Kubo, block
uncertainty, star-COM MSD, local logarithmic slope, and diffusion diagnostics.
The slow rheological time is not defined by the first microscopic `G/G0=0.1`
crossing; the analyzer reports a slow-reference diagnostic beginning after
the local-force drop and withholds a terminal-time claim when block noise
arrives first. No full production simulation or long associating R1-C2
continuation has been launched.
