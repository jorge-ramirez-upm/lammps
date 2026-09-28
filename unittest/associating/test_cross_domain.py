#!/usr/bin/env python3
"""Focused MPI regression for associating's normal ghost communication.

Usage: python3 test_cross_domain.py /path/to/lmp
"""
import math
import subprocess
import sys


def run(lmp, distance, expect_error=False):
    deck = f"""units lj
atom_style atomic
atom_modify map yes
processors 2 1 1
region box block 0 10 0 10 0 10
create_box 1 box
create_atoms 1 single 4.3 5 5
create_atoms 1 single {4.3 + distance} 5 5
mass 1 1
pair_style associating
pair_coeff * * 30 1.5 1
fix a all associating/kinetics pair 1 2
neighbor 0.3 bin
neigh_modify every 1 delay 0 check no
compute fmax all reduce max fx
thermo_style custom step pe c_fmax
thermo 1
run 0
"""
    result = subprocess.run(["mpirun", "-np", "2", lmp, "-log", "none", "-echo", "none"], input=deck,
                            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if expect_error:
        assert result.returncode != 0 and "Associating FENE bond exceeded R0" in result.stdout, result.stdout
        return
    assert result.returncode == 0, result.stdout
    line = next(line for line in result.stdout.splitlines()
                if line.split() and line.split()[0] == "0" and len(line.split()) == 3)
    _, energy, force = map(float, line.split())
    r0, k, ee = 1.5, 30.0, 1.0
    lo, hi = 0.5, 2.0 ** (1.0 / 6.0)
    for _ in range(100):
        mid = (lo + hi) / 2
        derivative = -48 / mid ** 13 + 24 / mid ** 7 + k * mid / (1 - mid * mid / r0 ** 2)
        if derivative < 0: lo = mid
        else: hi = mid
    rstar = (lo + hi) / 2
    expected_energy = -.5 * k * r0 ** 2 * math.log(1 - distance ** 2 / r0 ** 2)
    expected_energy -= -.5 * k * r0 ** 2 * math.log(1 - rstar ** 2 / r0 ** 2) + ee
    expected_force = k * distance / (1 - distance ** 2 / r0 ** 2)
    assert math.isclose(energy, expected_energy, rel_tol=1e-10), (energy, expected_energy, result.stdout)
    assert math.isclose(force, expected_force, rel_tol=1e-10), (force, expected_force)


lmp = sys.argv[1]
run(lmp, 1.4)                 # r_assoc < r < R0, across the x-domain boundary
run(lmp, 1.51, expect_error=True)  # after a fresh neighbor build, still in the R0+skin halo
