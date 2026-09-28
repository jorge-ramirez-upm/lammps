# B2.4 replicated-scheduler performance characterization

This procedure uses the validated Javier `A=4`, `N=10`, 1,000-star system
(43,563 particles, 4,000 terminal stickers) with `Ee=8`, `Ea=4`, `T=1`,
`dt=0.01`, `Nevery=100`, benchmark-only `nu0=10`, `r_assoc=2^(1/6)`, and
`neighbor 0.6 bin`. It characterizes the existing replicated MPI reference
scheduler; it does not change the chemistry algorithm.

## Reproducible procedure

From this directory, build the common representative state once, then use that
same restart for every timed run:

```
mpirun -np 1 ../../../build-associating/lmp -in in.b24_warmup.lmp
for n in 1 2 4 8; do
  mpirun -np $n ../../../build-associating/lmp -in in.b24_timing.lmp -log b24.$n.log
done
```

`in.b24_warmup.lmp` runs 50,000 steps (500 chemical sweeps) from the supplied,
association-free equilibrated data and writes `javier_b24_warmup.restart`.
Its active-bond trace rises rapidly from 0 to 1,694 at 20k steps, then is in
1,769--1,775 over 44k--50k steps (1,769 at the saved state). This is a useful
representative range for B2.4, not a claim of rigorous thermodynamic
equilibration. `in.b24_timing.lmp` reads exactly that restart and runs 10,000
steps / 100 sweeps at every rank count, with thermo only at the endpoints.

Appending `timing` to `fix associating/kinetics` enables the two
`ASSOCIATING_TIMING` summary lines. Disabled is the default and preserves the
scientific scheduler. Timings are cumulative maxima across ranks, i.e. the
critical path, and cover local extraction (including its occasional neighbor
build), all-gathers, global reconstruction/validation, `process_sweep()`, and
partner writeback plus forward communication.

## Raw results

The table records the LAMMPS timed-loop wall time, not process startup or I/O.
All runs have 43,563 particles, 4,000 stickers, 10,000 MD steps, and 100
chemistry sweeps.

| ranks | candidate edges mean [min,max] | active bonds mean [min,max] | creations | breaks | loop wall s | s/MD step | s/sweep | kinetics s | kinetics % |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2151.39 [2097,2198] | 1781.08 [1769,1794] | 111 | 86 | 124.1670 | 0.012417 | 1.241670 | 2.192230 | 1.77 |
| 2 | 2140.36 [2082,2191] | 1777.12 [1770,1785] | 110 | 103 | 65.6708 | 0.006567 | 0.656708 | 1.181100 | 1.80 |
| 4 | 2130.43 [2085,2183] | 1774.90 [1769,1779] | 106 | 99 | 38.0848 | 0.003808 | 0.380848 | 0.705878 | 1.85 |
| 8 | 2135.08 [2076,2189] | 1780.47 [1770,1786] | 105 | 95 | 25.0813 | 0.002508 | 0.250813 | 0.480643 | 1.92 |

| ranks | extraction s | all-gather s | reconstruction s | process sweep s | writeback/forward s |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2.057575 | 0.000643 | 0.042253 | 0.027644 | 0.064114 |
| 2 | 1.073163 | 0.004247 | 0.042886 | 0.027768 | 0.033920 |
| 4 | 0.603477 | 0.011118 | 0.046110 | 0.030029 | 0.020108 |
| 8 | 0.364211 | 0.018235 | 0.055100 | 0.035579 | 0.015014 |

At this 4,000-sticker scale, kinetics remains below 2% of total run time, so
it is not yet a significant total-runtime cost. Local extraction dominates at
one rank. By eight ranks, all-gather, reconstruction, and sweep processing do
not scale while extraction does; the critical chemistry cost improves only
4.56x from 1 to 8 ranks and the all-gather component grows 28.4x. This is
quantitative evidence that the replicated reference scheduler will need a
scalable B3 replacement for larger systems/rank counts, but it does not
justify optimizing or changing it in B2.4.
