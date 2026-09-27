# Design: monovalent transient associating stickers

## Scope and version inspected

This proposal targets this clean checkout, which reports `LAMMPS_VERSION` as
**2 Sep 2026** (git `patch_24Jan2020-32726-g12ed046a49`).  It adds reversible,
monovalent sticker associations between selected atoms.  An association is
state held by a fix and a force evaluated by a pair style; it is never a
LAMMPS bond.

Consequently, this feature must not edit `atom->num_bond`, `bond_atom`,
`bond_type`, special-neighbor lists, molecular topology, or any permanent
polymer topology.  Existing bond, angle, dihedral, and improper interactions
remain exactly as read from the data file.

The first implementation should deliberately cover one sticker type and one
association potential.  Multiple sticker chemistries, multivalency, kinetic
models with history, and dynamic topology are separate future work.

## Smallest implementation boundary

Add one self-contained package (suggested name `ASSOCIATING`) and only these
new implementation files:

| File | Responsibility |
| --- | --- |
| `src/ASSOCIATING/fix_associating_kinetics.h` | Registers `fix associating/kinetics`; declares the per-atom partner state and a small accessor for the pair style. |
| `src/ASSOCIATING/fix_associating_kinetics.cpp` | Owns association/breakage decisions, migration, restart, and ghost-state communication. |
| `src/ASSOCIATING/pair_associating.h` | Registers `pair_style associating`. |
| `src/ASSOCIATING/pair_associating.cpp` | Applies the transient association force only; it does not create or delete topology. |
| `src/KOKKOS/pair_associating_kokkos.h` | Future `pair_style associating/kk` declaration. |
| `src/KOKKOS/pair_associating_kokkos.cpp` | Future Kokkos implementation, including device mirrors and device communication. |

The two Kokkos files are a planned second phase, not a prerequisite for the
CPU implementation.  No core LAMMPS file needs an edit: style registration is
in each new header's existing `FIX_CLASS`/`PAIR_CLASS` block, and normal
package discovery builds the added files.  Documentation, examples, and tests
can likewise live under the new package and `unittest/` without changing
existing source files.

## State and ownership

`fix associating/kinetics` stores exactly one primary per-atom datum:

```
tagint partner[nmax]       // 0: unbound; otherwise global atom ID of partner
```

Although this is integer state, it must be `tagint`, not `int`: a LAMMPS atom
ID can exceed the range of `int`.  A nonzero value is valid only when it is
reciprocal (`partner[i] == tag[j]` and `partner[j] == tag[i]`).  This makes
monovalency structural: no atom has capacity for a second transient partner.

The fix allocates and initializes the array with `grow_arrays()`, copies it in
`copy_arrays()`, and zeros it in `set_arrays()`.  It implements
`pack_border()`/`unpack_border()` so pair calculations can inspect the partner
of ghost atoms.  It implements `pack_exchange()`/`unpack_exchange()` so an
atom takes its association state when it migrates between MPI ranks.  It also
implements `pack_restart()`, `unpack_restart()`, `size_restart()`, and
`maxsize_restart()` so a restart preserves live associations.  Encode the
`tagint` in a `double` communication slot with `ubuf`, as LAMMPS does for
integer payloads.

The fix exposes a narrow, read-only accessor returning `partner` and an
explicit `refresh_ghosts()` method.  The pair looks up the required fix by
style/ID during `init_style()` and errors if it is absent, duplicated, or is
configured for a different group.  It receives neither a writable atom-state
pointer nor authority to run kinetics.

After each state change, the fix calls `comm->forward_comm(this)`.  Thus the
next force evaluation sees matching local and ghost state even when the
neighbor list was not rebuilt.  This is essential; relying only on border
communication at neighbor rebuilding produces stale cross-rank associations.

## User-facing syntax

Keep the initial syntax small and explicit:

```
fix ID group-ID associating/kinetics Nevery seed kon koff rreact
pair_style associating rcut
pair_coeff * * k r0 rcut
```

`fix` acts only on atoms in its group.  `Nevery` is the interval between
kinetic updates; `kon` and `koff` are rates in the current LAMMPS time units;
`rreact` is the formation cutoff and must be no larger than the pair cutoff.
The pair style contributes a harmonic transient tether

```
U(r) = 1/2 k (r-r0)^2,          r < rcut,
```

for reciprocal associated pairs only.  It should use the usual pair energy
and virial tallying.  The production implementation must state whether this
is intended for `pair_style hybrid` (recommended: yes, as an additive style)
and reject `special_bonds` scaling for the transient interaction: it is not a
permanent bond.

## Kinetic update algorithm

Run the fix at `POST_FORCE`, so a state selected at step `t` affects the pair
force beginning at step `t+1`; this avoids changing force state partway
through a force evaluation.  Request an occasional full neighbor list with a
cutoff at least `rreact` in `init()`/`init_list()`.  At each `Nevery` update:

1. Break each existing association with probability
   `p_off = 1 - exp(-koff * Nevery * dt)`.  Generate the variate from a
   stateless hash of `(seed, timestep, min(tag_i,tag_j), max(tag_i,tag_j))`.
   Both owners therefore make the same break decision without messages.
2. For every currently free local sticker, enumerate free candidates in the
   full list that are in the fix group and within `rreact`.  Each candidate
   pair independently passes a formation trial with
   `p_on = 1 - exp(-kon * Nevery * dt)`, keyed by the same unordered pair and
   timestep hash.
3. A free atom nominates the passing candidate with the lowest deterministic
   priority hash.  Forward-communicate nominations as a second temporary
   `tagint` fix array (or reuse an internally allocated `proposal[nmax]`).
4. Form an association only when the nomination is mutual.  Both owning ranks
   then write the reciprocal `partner` values locally.  Clear proposals and
   forward-communicate `partner`.

This mutual-winner rule is intentionally simple and MPI decomposition
independent: it requires no remote writes, all-to-all tag lookup, or topology
transactions.  Competition reduces the effective formation rate in dense
sticker regions, so `kon` is an attempted-pair rate, not an unconditional
macroscopic association rate.  The manual must say this plainly.  If an
exact non-mutual proposal/acceptance kinetic scheme becomes necessary, add it
as a separate mode after measuring that need; do not complicate the first
implementation.

Use `tag[i] != tag[j]`, reciprocal-state checks, and the full-list group mask
in all phases.  If a stored partner disappears (e.g., atom deletion by an
unrelated fix), clear the local state at the next update and warn once; this
is preferable to applying a force from a dangling tag.

## Pair force algorithm

`pair associating` requests a standard half neighbor list and loops like
`pair_lj_cut`.  For each neighbor `j`, it applies the tether only if both
partner IDs are reciprocal.  To count each association once, use the normal
half-list ownership or, for a full-list configuration, a `tag[i] < tag[j]`
guard.  The pair reads coordinates, types only if type-dependent coefficients
are later added, forces, and the fix's ghost-refreshed partner array.  It
does not allocate a second persistent per-atom association array and does not
call any bond or special-neighbor API.

The pair should set `single_enable = 0` initially and document that it cannot
report an association from `single()` without fix state.  It should be
compatible with Newton pair on and off; test both.  Reject rRESPA inner/middle
levels in the first version if a correct splitting policy is not implemented.

## Future CUDA / Kokkos design

Implement `pair associating/kk` by following `pair_bondval/kk`, rather than
copying a CPU loop into a CUDA lambda:

* derive from the CPU pair and `KokkosBase`, register device/host aliases, set
  `kokkosable`, data masks, and Kokkos neighbor-list flags;
* give the fix a `DualView<tagint*>` for `partner` and synchronize it after
  host kinetics and after exchange/border operations;
* make the pair read a device view of partner IDs and atom tags, then execute
  the reciprocal check and tether tally in the Kokkos neighbor functor;
* retain host `fix associating/kinetics` initially.  Before each CUDA pair
  call it syncs the owner-updated partner view to device; after kinetic updates
  and fix forward communication it marks the host view modified.  No device
  atom-state mutation is needed in this phase;
* if kinetics later moves to the GPU, add `fix associating/kinetics/kk` as a
  separate pair of files using `fix_neigh_history/kk` as the exchange/restart
  reference.  Do not make the CPU fix depend on Kokkos.

This separation keeps the initial CPU design small while preserving the data
layout and synchronization boundary required by CUDA.

## Reference implementations in this checkout

| Need | Exact reference files | Pattern to reuse |
| --- | --- | --- |
| Simple pair loop | `src/pair_lj_cut.h`, `src/pair_lj_cut.cpp` | Style registration, neighbor traversal, Newton handling, energy/virial tally, coefficients. |
| Per-atom pair communication | `src/EXTRA-PAIR/pair_bondval.h`, `src/EXTRA-PAIR/pair_bondval.cpp` | `comm_forward`/`comm_reverse`, per-atom scratch allocation, and a staged pair calculation. |
| Fixed-size per-atom state across ghosting, migration, and restart | `src/fix_property_atom.h`, `src/fix_property_atom.cpp` | `grow/copy/set`, border, exchange, restart hooks, and `ubuf` encoding. |
| Variable per-atom partner IDs across migration/restart | `src/fix_neigh_history.h`, `src/fix_neigh_history.cpp` | Partner IDs as `tagint`, restart record layout, and ownership-aware history. |
| Simple Kokkos pair | `src/KOKKOS/pair_lj_cut_kokkos.h`, `src/KOKKOS/pair_lj_cut_kokkos.cpp` | `DualView` parameters, device/host aliases, data masks, and Kokkos neighbor dispatch. |
| Kokkos pair with per-atom communication | `src/KOKKOS/pair_bondval_kokkos.h`, `src/KOKKOS/pair_bondval_kokkos.cpp` | Device per-atom buffers, communication callbacks, scatter views, and staged force kernels. |
| Kokkos per-atom migration/restart | `src/KOKKOS/fix_neigh_history_kokkos.h`, `src/KOKKOS/fix_neigh_history_kokkos.cpp` | Kokkos exchange hooks and device mirror ownership. |

## Tests

Add tests with the new package, avoiding changes to unrelated test sources.

1. **Two-sticker force and topology test.**  Two sticker atoms at a known
   separation with manually initialized/restarted reciprocal state must give
   the harmonic analytic force, energy, and virial.  Assert `nbonds`, each
   atom's permanent bond count, bond list, and special-neighbor counts are
   unchanged before and after repeated association updates.
2. **Monovalency and competition test.**  Place three stickers within
   `rreact`.  Across many seeded updates, assert every nonzero partner ID is
   reciprocal and no atom has more than one partner.  Repeat with a different
   MPI decomposition and require the same partner-ID trajectory for a fixed
   seed.
3. **Kinetics limits.**  With `kon=0`, no association forms; with `koff=0`,
   an established association never breaks; with a very large `koff`, survival
   matches `exp(-koff*Nevery*dt)` statistically.  At low density, measure the
   attempted-pair formation probability against
   `1-exp(-kon*Nevery*dt)` within a binomial confidence interval.
4. **MPI migration and ghost test.**  Drive an associated pair across a
   processor boundary without forcing a neighbor rebuild each timestep.
   Verify reciprocal state, force, and energy survive migration and change
   immediately after a break/form event.  Run with Newton pair on and off.
5. **Restart equivalence test.**  Compare an uninterrupted seeded run with a
   run split by `write_restart`/`read_restart`; partner IDs, energy, and
   positions after the same number of steps must match.
6. **CPU/Kokkos equivalence test (future phase).**  For a fixed initial state
   and no kinetic events, compare CPU `associating` with CUDA
   `associating/kk` forces, energy, virial, and per-atom energy within the
   established Kokkos precision tolerance.  Then exercise host kinetics plus
   device force evaluation across migration.

The first three tests are the minimum acceptance set.  The MPI migration and
restart tests are required before claiming production readiness; the Kokkos
comparison is required before enabling the `/kk` style.
