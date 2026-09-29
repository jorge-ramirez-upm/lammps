# R1-A: equilibrium rheology prototype

R1-A establishes a reproducible equilibration-to-stress-correlation workflow
for Javier's realistic 1,000-star associating KG/WCA system. It is a prototype,
not a terminal-rheology production result. It preserves 4,000 type-2 stickers,
`Ee=8`, `Ea=4`, `T=1`, `dt=0.01`, `Nevery=100`, `nu0=10`, association cutoff
`2^(1/6)`, permanent FENE backbone, WCA excluded volume, and Langevin damping
2.0.

## Two separate stages

`in.r1a_equilibrate.lmp` starts from the supplied no-transient-bond data,
activates association, and performs network/conformational equilibration. It
records temperature, potential energy, active bonds, cumulative creates and
breaks, per-star Rg, and per-star COM MSD. Every 10,000 steps it writes a
network snapshot. `analyze_r1a_equil.py` classifies those snapshots as self
(same molecule ID) or intermolecular bonds. A plateau in active bonds alone is
not an equilibration claim; all diagnostics and their time trends must be
reviewed. The stage writes `$F.r1a.equil.restart`.

`in.r1a_stress.lmp` starts from that restart and begins a **fresh** correlation
measurement after `reset_timestep 0`. Diagnostics and `.gt` output are kept
separate. This avoids accumulating rheology while the initial reversible
network is being formed.

## Stress convention and canonical estimator

The current `compute pressure` documentation and source were checked: its
six-vector is `xx yy zz xy xz yz`, in intensive pressure units. R1-A names
these components `sigma`; LAMMPS calls them pressure. A continuum convention
that calls Cauchy stress the negative of pressure changes every channel's sign,
but same-channel autocorrelations and consistently formed normal differences
are unchanged in \(G(t)\).

\[
N_{xy}=\sigma_{xx}-\sigma_{yy},\quad N_{xz}=\sigma_{xx}-\sigma_{zz},\quad
N_{yz}=\sigma_{yy}-\sigma_{zz}.
\]

The required rotational average is

\[
G(t)=\frac{V}{5k_BT}(C_{xy}+C_{xz}+C_{yz})+
\frac{V}{30k_BT}(C_{Nxy}+C_{Nxz}+C_{Nyz}).
\]

The normal differences are not independent; the coefficient above already
accounts for the rotational average. R1-A deliberately does not substitute a
three-shear estimator.

## Correlator and restart behavior

The verified syntax is

```lammps
fix corr all ave/correlate/long 1 100000 \
  v_sxy v_sxz v_syz v_nxy v_nxz v_nyz \
  type auto file ${F}.gt overwrite ncorr 40
```

`type auto` retains the six autocorrelations separately, in that input order.
`ncorr=40` gives a very long formal multi-tau range, but it is not evidence of
useful terminal data. `fix ave/correlate/long` writes its state to LAMMPS binary
restarts and can continue an interrupted accumulation only when recreated with
identical settings. The prototype instead uses the equilibrated restart to
start a new measurement. `$F` is the data-file basename without `.lammpsdat`.

`analyze_r1a_stress.py` writes `$F.r1a_modulus.csv`, retaining six channels,
`G_shear`, `G_normal`, and total `G`. It labels a lag useful only while the
conservative available-origin upper bound is at least 10% of the total run;
the correlator file does not itself expose exact multi-tau bin counts. It also reports useful-lag RMS channel scatter normalized by the corresponding zero-lag mean. Isotropy means noisy
compatibility, not pointwise equality; normal averaging is judged by comparing
those three raw normal channels with their average.

## Commands

From the repository root:

```bash
cmake -S cmake -B build-r1a -D PKG_ASSOCIATING=on -D PKG_EXTRA-FIX=on -D PKG_MOLECULE=on -D BUILD_MPI=on
cmake --build build-r1a -j"$(nproc)"
cd bench/associating/javier_star_A4_N10_rho08
F=Stars_NA4N10C1000rho0.85rhopoly0.8.equilibrated
mpirun -np 1 ../../../build-r1a/lmp -var F "$F" -var EQUIL_STEPS 200000 -in in.r1a_equilibrate.lmp
python3 analyze_r1a_equil.py "$F".r1a.network.*.dat --out "$F".r1a_network_summary.csv
mpirun -np 1 ../../../build-r1a/lmp -var F "$F" -var PILOT_STEPS 1000000 -in in.r1a_stress.lmp
python3 analyze_r1a_stress.py "$F".gt --volume 51250.5882353 --run-steps 1000000
```

The box volume is `43563/0.85 = 51250.5882353`. The 200,000-step network
stage and 1,000,000-step correlation stage are prototype defaults for a
dedicated CPU host, not a claimed production length. A serious production
should be extended until the *useful*, not formal, lag reaches terminal decay,
with multiple independent blocks/replicas before integrating viscosity.

## Prototype interpretation

A completed prototype must show finite stationary mean stress channels, no
FENE/topology errors, valid reciprocal transient bonds, and mutually noisy but
non-pathological tensor channels. It cannot establish terminal relaxation if
`G(t)` has not decayed within the useful-lag window. Do not report viscosity
from an unconverged tail.

## Local smoke result (not rheology production)

A one-rank local smoke used 10,000 association-equilibration steps followed by
a fresh 10,000-step correlation run with a temporary `Nfreq=10000` edit only to force a small
file write; the documented prototype remains `NFREQ=100000`. The equilibration
turn-on had active bonds `0 -> 1552`, with the potential energy still falling
from about 794861 to 785939. Its final snapshot contained 80 self and 1472
intermolecular transient bonds. Temperature stayed near 1, and no FENE or
network-validity error occurred, but these monotonic observables show that the
starting restart was **not stationary**. The resulting 89-lag `.gt` file
verified the six-channel syntax and postprocessor, reaching a useful local
lag of 81.92 MD time at the conservative 10% available-origin threshold. It
does not approach terminal relaxation; large relative shear-channel scatter at
late lag is expected from this extremely short nonstationary smoke.

On this CPU the 10,000-step stages each cost roughly two minutes. The next
dedicated-host run should first execute the documented 200,000-step
network/conformational stage and inspect its diagnostics; only then should a
fresh 1,000,000-step correlation prototype be launched. That prototype is
expected to provide useful lag near 9000 MD time under the 10% criterion, but
still cannot be assumed to resolve terminal decay. A serious production length
must be chosen after inspecting that result, not by integrating its tail now.
