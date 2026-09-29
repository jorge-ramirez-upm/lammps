T = 1
rho = 0.05
N = 256
dt = 0.005
nu0 = 20
Langevin damp = 2.0
production = 100000 steps
replicas = 4

**Commands to reproduce everything**

- Start the campaign:

```
DAMP=2.0 \
./bench/associating/dimer_kinetics/run_campaign.sh \
  ./build-k1/lmp full 1
```

- When all trajectories have finished:

```
python3 bench/associating/dimer_kinetics/analysis/fit_dimer_kinetics.py \
  'bench/associating/dimer_kinetics/results/*.dat' \
  --summary bench/associating/dimer_kinetics/results/k1.csv
```

- Inspect the aggregate table:

```
column -s, -t \
  bench/associating/dimer_kinetics/results/k1_conditions.csv
```

**Fitted slopes**

$$ \frac{d \ln k_f}{dE_a} = -0.9213 $$
$$ \frac{d \ln k_b}{dE_e} = -0.6641 $$
$$ \frac{d \ln K_{eq}}{dE_e} = 0.9840 $$

