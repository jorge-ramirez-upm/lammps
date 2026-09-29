# K1: reversible-dimer kinetic and thermodynamic validation

This is a self-contained validation of `pair associating` plus
`fix associating/kinetics` in an isolated WCA sticker fluid. It does not alter
the star-polymer benchmark. Identical stickers have at most one partner, so
this is exactly the species reaction \(A+A \leftrightharpoons B\), with free
stickers \(A\) and reversible dimers \(B\).

## Reaction model, transient, and consistency test

\[
[A]+2[B]=C, \qquad
\frac{d[B]}{dt}=k_f[A]^2-k_b[B]=k_f(C-2[B])^2-k_b[B].
\]

\(k_f\) is the effective macroscopic second-order forward rate, \(k_b\) the
effective macroscopic first-order backward rate,
\(K_{\rm eq}^{\rm kin}=k_f/k_b\), and
\(K_{\rm eq}^{\rm eq}=[B]_{\rm eq}/[A]_{\rm eq}^2\). Agreement of the two
constant estimates is important: one is constrained by the full transient and
the other is directly measured from equilibrium concentrations.

The fitting script uses the analytic transient rather than initial slopes.
With

\[
D=k_b^2+8k_bk_fC,\quad
b_\pm=(4k_fC+k_b\pm\sqrt D)/(8k_f),\quad
R(t)=(b_-/b_+)e^{-\sqrt D t},
\]

and the deliberately unassociated \(B(0)=0\),

\[
B(t)=(b_--R(t)b_+)/(1-R(t)).
\]

It fits each replica in log-rate coordinates, obtains a block equilibrium
estimate from the final half, and records SSE, \(R^2\), mass conservation, and
creation/break versus active-dimer consistency. Poor fit quality, structured
residuals, or disagreeing equilibrium constants signal a failure of the
well-mixed interpretation, rather than merely uncertain rates.

## Finite-step chemistry and Metropolis energetics

Chemistry is attempted once every \(N_{\rm every}\) MD steps:

\[
\Delta t_{\rm chem}=N_{\rm every}\Delta t,\qquad
q(E_a)=1-\exp[-\nu_0e^{-E_a/T}\Delta t_{\rm chem}].
\]

Production evaluates this as `-expm1(-x)`, preserving small-probability
precision. The approximation \(q\simeq\nu_0e^{-E_a/T}\Delta t_{\rm chem}\)
holds only in the small-step limit. At the standard point
\(\nu_0=20\), \(N_{\rm every}=100\), \(\Delta t=0.005\), their product is
10, so low \(E_a\) may show visible finite-step curvature against a naive
\(\ln({\rm rate})\sim-E_a/T\) law.

For separation \(r\),

\[
\Delta U(r)=U_{\rm FENE}(r)-U_{\rm FENE}(r_*)-E_e,
\]

and formation/breaking use \(q\) times

\[
A_f=\min[1,e^{-\Delta U/T}],\qquad A_b=\min[1,e^{+\Delta U/T}].
\]

The implementation uses sign branches so downhill acceptance returns one
without evaluating an exponential larger than one. \(E_e\) thus affects both
effective rates. It is not generally justified to demand separately
\(k_f\propto e^{-E_a/T}\) and \(k_b\propto e^{-(E_a+E_e)/T}\) over the whole
practical range; detailed balance constrains their equilibrium ratio most
directly.

## Physical meaning of the effective constants

\[
k_f(E_a,E_e,\rho)\sim C_f(\rho)q(E_a)\langle A_f\rangle_{\rm encounter},
\qquad
k_b(E_a,E_e,\rho)\sim q(E_a)\langle A_b\rangle_{\rm bound}.
\]

\(C_f\) includes reactive volume, WCA pair correlations, diffusion/encounter
frequency, repeated encounters, and cadence. The bound average samples the
bound separation distribution, explaining why the \(\ln k_b\)--\(E_e\) slope
need not be \(-1/T\). These are effective macroscopic rates, not bare
microscopic prefactors.

The reported \(k_f/q(E_a)\) and \(k_b/q(E_a)\) separate the known finite-step
activation factor from encounter/configurational effects. At fixed \(E_e\) in
a dilute fluid they should be substantially less \(E_a\)-dependent than raw
rates if that factor dominates. This is a validation diagnostic, not a claim
that either quantity must be constant.

At low density the concentration equilibrium constant is expected to obey

\[
K_c=[B]/[A]^2\simeq K_0(\rho)e^{E_e/T},\qquad
\partial\ln K_c/\partial E_e\simeq1/T.
\]

It is not automatically a thermodynamic constant. In a nonideal WCA fluid,
\(K_{\rm therm}=a_B/a_A^2=K_c\gamma_B/\gamma_A^2\). Density can shift the
intercept, and eventually the slope, of \(\ln K_c\) versus \(E_e\). No
prior density-independence of \(K_c\) is assumed.

## Current K1 findings at \(\rho=0.05\)

The committed four-replica production data remain the baseline in `results/`
and are summarized in [RESULTS.md](RESULTS.md):

\[
\frac{d\ln k_f}{dE_a}\approx-0.921,\qquad
\frac{d\ln k_b}{dE_e}\approx-0.664,\qquad
\frac{d\ln K_{\rm eq}}{dE_e}\approx0.984.
\]

At \(T=1\), the last result is particularly significant because the expected
thermodynamic slope is approximately \(+1\). \(K_{\rm eq}^{\rm kin}\) and
\(K_{\rm eq}^{\rm eq}\) generally agree closely, whereas individual effective
rates retain additional dynamical/configurational physics. The low-event
\(E_a=6\) point is interpreted cautiously; these data do not establish an
exact raw-rate Arrhenius law.

## Setup and K1-D density campaign

All runs have \(N=256\), \(T=1\), \(\Delta t=0.005\), \(\nu_0=20\), Langevin
damping 2.0, WCA cutoff \(2^{1/6}\), 20,000 WCA warm-up steps, 100,000
production steps (500 MD time), and sampling every 100 steps. K1-D covers
\(\rho=0.025,0.05,0.10,0.20\), with four replicas, but does not repeat K1-C
cadence tests. At each density D1 is \(E_e=4, E_a=2,3,4,5,6\), and D2 is
\(E_a=4, E_e=2,4,6,8\); the shared central condition is run once.

The existing unprefixed files are the \(\rho=0.05\) baseline. New files use
`rho<density>_Ea...`; a nonempty `.dat` is the restart marker. New-density
central pilots use two replicas and 100,000 steps. If their event totals or
fit quality are inadequate, request a clearly documented longer rerun, e.g.
`PRODUCTION=200000`; do not silently mix production lengths. The fitter uses
physical MD time from `dt`.

## Linux-host commands

```bash
cmake -S cmake -B build-k1 -D PKG_ASSOCIATING=on -D BUILD_MPI=on -D BUILD_TESTING=on
cmake --build build-k1 -j"$(nproc)"

# Central pilots: rho=0.025, 0.10, 0.20; two replicas each
./bench/associating/dimer_kinetics/run_campaign.sh ./build-k1/lmp density-pilot 1

# Baseline plus pilots: raw, condition, and density-trend CSV outputs
python3 bench/associating/dimer_kinetics/analysis/fit_dimer_kinetics.py \
  'bench/associating/dimer_kinetics/results/*.dat' \
  --summary bench/associating/dimer_kinetics/results/k1d.csv

# Only after reviewing pilots: restartable D1/D2 production at new densities
./bench/associating/dimer_kinetics/run_campaign.sh ./build-k1/lmp density-full 1
python3 bench/associating/dimer_kinetics/analysis/fit_dimer_kinetics.py \
  'bench/associating/dimer_kinetics/results/*.dat' \
  --summary bench/associating/dimer_kinetics/results/k1d.csv

# Optional: matplotlib only
python3 bench/associating/dimer_kinetics/analysis/plot_density.py \
  bench/associating/dimer_kinetics/results/k1d_conditions.csv
```

The numeric path requires only Python's standard library. `k1d.csv` contains
raw replica results; `k1d_conditions.csv` supplies replica uncertainties for
rates, normalized rates, and equilibrium constants; `k1d_trends.csv` gives
per-density fits for \(\ln k_f\), exact \(\ln q\), normalized rates,
\(\ln k_b\), and both equilibrium constants. The new production grid is 96
trajectories, with the six reusable pilots already included; it is never
launched automatically by the analysis.

## K1-D local central-pilot result

Two 100,000-step replicas were run locally at each new density for
\((E_a,E_e,N_{\rm every})=(4,4,100)\). Mean creation/break counts were
232/220 at \(\rho=0.025\), 715/676 at \(\rho=0.10\), and 1045/986 at
\(\rho=0.20\). The corresponding \(K_{\rm eq}^{\rm kin}\) and direct values
were 2.57/2.62, 2.95/2.87, and 3.25/3.37. Mass and active-dimer diagnostic
checks passed exactly to the analysis tolerance.

The dilute pilot has enough reversible events to retain 100,000 steps for the
four-replica central production condition; no density-specific length was
silently introduced. Its transient \(R^2\) was only about 0.24 because the
concentration excursion is small compared with equilibrium fluctuations, so
SSE, event totals, replica uncertainty, and kinetic/equilibrium agreement
must accompany \(R^2\). At \(\rho=0.20\), \(R^2\) was about 0.60: this is not
by itself a demonstrated failure, but makes fit-quality and residual review an
explicit K1-D criterion. The slow \(E_a=6\) and strongly bound \(E_e=8\)
production cases remain the likely cases requiring a documented extension.
