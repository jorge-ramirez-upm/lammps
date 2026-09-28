# Transient-network observability

`write_associating_network FILE fix FIX-ID` writes one instantaneous,
MPI-decomposition-independent snapshot of the active transient network. It is
an explicit command, so its output times are selected in the input script and
are independent of the reaction cadence `Nevery`.

The command reads the associating fix's authoritative `partner[]` state through
its `active_network()` API. It does not create LAMMPS bonds, alter permanent
topology, or perform disk I/O in the chemistry kernel. Each record is written
once globally, sorted by canonical endpoint tags:

```
# timestep 1000 bonds 608
# tag_i tag_j molecule_i molecule_j
11 39586 1 966
```

Thus `tag_i < tag_j` holds for every row. Molecule IDs support self-bond,
bridge, multiple-edge, loop, molecular-graph, cluster, and percolation
analysis without inferring connectivity from permanent topology.

## Javier demonstration

From this directory:

```
mpirun -np 1 ../../../build-associating/lmp -in in.network_demo.lmp -log network-demo.log
python3 verify_network.py network-demo.log network.1000.dat network.2000.dat network.3000.dat
```

The demo writes at 1,000, 2,000, and 3,000 MD steps. The verifier checks every
snapshot against `f_kinetics[1]`, confirms canonical unique edges, confirms
that a sticker is used at most once, and reports self plus intermolecular
bonds. The validated demonstration gave 608 = 29 + 579, 905 = 43 + 862, and
1123 = 56 + 1067 active bonds, respectively.
