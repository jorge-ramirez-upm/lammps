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
