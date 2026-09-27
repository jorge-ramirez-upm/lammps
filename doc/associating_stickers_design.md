# Design: monovalent transient stickers in a Kremer--Grest melt

## Frozen physical model

This proposal targets the LAMMPS **2 Sep 2026** checkout (`patch_24Jan2020-32726-g12ed046a49`) and fixes the first physical model before any C++ implementation.  It is for unentangled or entangled associating-polymer melts in KG reduced LJ units: `epsilon = sigma = m = k_B = 1`.

All beads have WCA excluded volume,

```
U_WCA(r) = 4[(1/r)^12 - (1/r)^6] + 1,  r < r_WCA = 2^(1/6),
             0,                        r >= r_WCA.
```

Permanent backbone bonds remain ordinary LAMMPS `bond_style fene`, with `K=30` and `R0=1.5`, including their normal KG repulsive LJ term.  A transient sticker association is monovalent state held by a fix.  It never creates, deletes, or alters a LAMMPS bond, angle, dihedral, improper, special-neighbor list, or permanent polymer topology.

For a transient association, add **only** the logarithmic FENE term:

```
U_FENE(r) = -1/2 K R0^2 ln[1 - (r/R0)^2],  0 <= r < R0,
U_assoc(r) = U_FENE(r) - U_FENE(r_star) - E_e.
```

`r_star` is the minimum of `U_WCA + U_FENE` for `K=30`, `R0=1.5`; calculate it from the analytic functions, never a hard-coded decimal.  `E_e > 0` is the stabilization at that minimum.  The transient force is only

```
F_assoc_vector = -K (x_i-x_j) / [1-(r/R0)^2].
```

The chemical-state energies at fixed coordinates are

```
U_A(r) = U_WCA(r)
U_B(r) = U_WCA(r) + U_FENE(r) - U_FENE(r_star) - E_e.
```

Thus `Delta_U = U_B-U_A = U_assoc`: WCA cancels from the chemical energy difference but remains physically present exactly once in both states.  An associated state at `r >= R0` is invalid and must raise the normal FENE error, never clamp or automatically rupture.

`bond_style fene` cannot implement the transient contribution.  In this checkout, [`src/MOLECULE/bond_fene.cpp`](../src/MOLECULE/bond_fene.cpp) explicitly adds both log-FENE and WCA LJ.  Reusing it would double-count WCA; adding a transient topology bond would also change `special_bonds` behavior.  Both routes are forbidden.

## Required configuration and invariants

The intended additive configuration is equivalent to:

```
units lj
bond_style fene
bond_coeff ... 30.0 1.5 1.0 1.0
pair_style hybrid/overlay lj/cut 1.122462048309373 associating 1.5
pair_coeff * * lj/cut 1.0 1.0 1.122462048309373
pair_coeff * * associating 30.0 1.5 E_e
special_bonds fene
```

`special_bonds fene` gives permanent 1--2 backbone pairs their WCA contribution through `bond_style fene`; every other bead pair receives WCA through `lj/cut`.  The transient style provides no WCA and never applies or changes `special_lj`.  The first version rejects permanent 1--2 special neighbors as association candidates: they are normally omitted from simple pair lists and are not intended non-backbone sticker associations.

The invariants are:

* Exactly one KG WCA contribution per bead pair.
* A reciprocal transient association adds one shifted log-FENE term and no LJ/WCA term.
* `tagint partner[i] == 0` means free; otherwise it is the ID of exactly one reciprocal partner.
* Transient state never changes permanent topology or its `special_bonds` classification.
* A bound pair is force-active for all `r < R0`, including `r > r_assoc`.
* Equilibrium mode has no distance-triggered forced rupture.

## Association range and detailed-balance kinetics

`r_assoc` is an explicit formation-cutoff parameter.  Its initial reference value is `2^(1/6)`, but it is never hard-coded.  It limits formation candidates only.  The pair-style neighbor cutoff is `R0`, not `r_assoc`, so an associated pair remains in the force list until its FENE limit.

Within `r < r_assoc`, local detailed balance requires

```
k_c(r)/k_b(r) = exp[-Delta_U(r)/T].
```

The first model uses explicit LJ-unit temperature `T` and attempt frequency `nu`:

```
k_c(r) = nu min(1, exp[-Delta_U(r)/T])
k_b(r) = nu min(1, exp[ Delta_U(r)/T])
p = 1 - exp(-k * Nevery * dt).
```

The temperature is an input, never an instantaneous kinetic temperature.  At `r >= r_assoc`, both transition rates are zero.  A bound pair can stretch outside the reaction domain, retain its FENE force, and dissociate after returning: this is gated equilibrium kinetics, not forced rupture.

Suggested syntax:

```
fix ID sticker-group associating/kinetics Nevery seed nu T r_assoc
pair_style associating R0
pair_coeff * * K R0 E_e
```

The pair owns `K`, `R0`, `E_e`, `r_star`, and `Delta_U(r)`; the fix queries it rather than duplicating coefficients.

## State, communication, and kinetic scheduler

`fix associating/kinetics` owns `tagint partner[nmax]` and implements grow/copy/set, border, exchange, and restart callbacks.  It packs tag IDs through `ubuf`.  After a state change it calls `comm->forward_comm(this)`, keeping ghosts current without a neighbor rebuild.

The fix requests an occasional full list with fixed cutoff `r_assoc` for kinetic candidates.  `pair associating` requests a standard half list at `R0`, filters it for reciprocal partner IDs, and performs normal Newton force and energy/virial tallying.  It must not use `special_lj` to scale the transient force and initially rejects rRESPA inner/middle use.

The earlier mutual-nomination scheme is removed: in a crowded melt its formation probability depends on competing candidates while its reverse move does not, so it violates detailed balance.  The CPU reference must use a serial random-scan Metropolis sweep over unordered eligible edges:

1. Build free--free formation and associated breakage edges once by global tag order.
2. Process a reproducible global random permutation keyed by seed, timestep, and pair IDs; recheck endpoints before every move.
3. Accept with `p_c` or `p_b`, update both owning ranks transactionally, then refresh ghosts before a conflicting edge.

This preserves the specified gated detailed balance and monovalency.  It may require explicit MPI owner-to-owner messages; writing a ghost is not sufficient.  A parallel coloring/event scheduler is future work only after it reproduces this serial kernel's stationary distribution and rates.

## Minimal file boundary and future Kokkos path

Add only these new files:

| File | Role |
| --- | --- |
| `src/ASSOCIATING/fix_associating_kinetics.h/.cpp` | State, serial kinetics, MPI transaction, migration/restart, ghost communication. |
| `src/ASSOCIATING/pair_associating.h/.cpp` | Shifted log-FENE force, `Delta_U`, `r_star`, and `R0` force list. |
| `src/KOKKOS/pair_associating_kokkos.h/.cpp` | Future CUDA/Kokkos force implementation. |

No core LAMMPS source changes are required.  The Kokkos pair derives from the CPU pair plus `KokkosBase`; the fix supplies a `DualView<tagint*>` synchronized from host kinetics before device force calculation.  GPU kinetics is a separate future feature.

## Exact local reference sources

| Need | Files | Pattern |
| --- | --- | --- |
| KG FENE definition | `src/MOLECULE/bond_fene.h`, `src/MOLECULE/bond_fene.cpp`, `doc/src/bond_fene.rst` | Log term, WCA term to exclude, and FENE-limit behavior. |
| WCA pair loop | `src/pair_lj_cut.h`, `src/pair_lj_cut.cpp` | Traversal, Newton handling, energy/virial tally. |
| Permanent topology exclusions | `doc/src/special_bonds.rst` | 1--2 exclusions and WCA ownership. |
| Per-atom state | `src/fix_property_atom.h`, `src/fix_property_atom.cpp` | Grow/border/exchange/restart and `ubuf`. |
| Partner IDs | `src/fix_neigh_history.h`, `src/fix_neigh_history.cpp` | `tagint` partner storage and restart layout. |
| Fixed full list | `src/fix_group.cpp`, `src/neigh_request.h`, `src/neigh_list.h` | `REQ_FULL`, `REQ_OCCASIONAL`, `set_cutoff_fixed`. |
| Kokkos pair/data | `src/KOKKOS/pair_lj_cut_kokkos.h/.cpp`, `src/KOKKOS/pair_bondval_kokkos.h/.cpp` | Dual views, masks, device lists, per-atom buffers. |
| Kokkos migration | `src/KOKKOS/fix_neigh_history_kokkos.h/.cpp` | Future device exchange/restart. |

## Acceptance tests

1. At several `r < R0`, compare free force/energy with WCA and associated force/energy with `U_WCA + U_FENE - U_FENE(r_star) - E_e` analytically.
2. Below `r_WCA`, verify associated minus free energy equals `U_assoc` with no second WCA term; a `bond_fene`-like extra WCA must fail.
3. Minimize `U_WCA+U_FENE`, verify `r_star`, `U_assoc(r_star)=-E_e`, and associated total `U_WCA(r_star)-E_e`.
4. Verify formation, breakage, migration, and restart leave permanent bond lists, special-neighbor counts, and backbone FENE energy unchanged; reject permanent 1--2 sticker candidates.
5. Form below `r_assoc`, stretch to `r_assoc < r < R0`, and verify force and energy persist without rupture; require the FENE error at or beyond `R0`.
6. At fixed separations and temperatures, measure `k_c/k_b` against `exp[-U_assoc/T]`; for three stickers compare serial-scheduler probabilities with explicit Boltzmann enumeration.
7. Test reciprocal monovalency, MPI migration without a neighbor rebuild, restart equivalence, and Newton pair on/off.
8. Future: compare CPU and CUDA `/kk` forces, energy, virial, and per-atom energy with frozen kinetics, then after host updates and migration.

## Architectural concern before coding

Exact detailed balance and melt-scale performance conflict here.  The serial random-scan reference kernel is physically correct, but global ordering and remote MPI updates may be expensive.  Proceed with it only if it is acceptable for initial system sizes.  Otherwise, design and validate a parallel scheduler against its Boltzmann distribution before implementing kinetics.
