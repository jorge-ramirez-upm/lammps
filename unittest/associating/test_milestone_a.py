#!/usr/bin/env python3
"""Additional Milestone A mechanics regressions.  Usage: python3 $0 lmp"""
import math
import os
import subprocess
import sys
import tempfile


def lmp(exe, deck, ranks=1):
    p = subprocess.run(["mpirun", "-np", str(ranks), exe, "-log", "none", "-echo", "none"],
                       input=deck, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    assert p.returncode == 0, p.stdout
    return p.stdout


def value(out):
    for line in out.splitlines():
        fields = line.split()
        if len(fields) >= 2 and fields[0] == "0":
            try:
                return float(fields[1])
            except ValueError:
                pass
    raise AssertionError(out)


def wca(r):
    return 4 * (r ** -12 - r ** -6) + 1 if r < 2 ** (1 / 6) else 0


def fene(r):
    k, r0 = 30., 1.5
    lo, hi = .5, 2 ** (1 / 6)
    for _ in range(100):
        m = (lo + hi) / 2
        d = -48/m**13 + 24/m**7 + k*m/(1-m*m/r0**2)
        lo, hi = (m, hi) if d < 0 else (lo, m)
    rs = (lo + hi) / 2
    return -.5*k*r0*r0*math.log(1-r*r/r0**2) + .5*k*r0*r0*math.log(1-rs*rs/r0**2) - 1


def pair_deck(r, associated=False):
    style = "hybrid/overlay lj/cut 1.122462048309373 associating" if associated else "lj/cut 1.122462048309373"
    coeff = "pair_coeff * * lj/cut 1 1 1.122462048309373\npair_coeff * * associating 30 1.5 1\nfix a all associating/kinetics debug_pair 1 2" if associated else "pair_coeff * * 1 1 1.122462048309373"
    return f"""units lj
atom_style atomic
atom_modify map yes
region b block 0 10 0 10 0 10
create_box 1 b
create_atoms 1 single 4 5 5
create_atoms 1 single {4+r} 5 5
mass 1 1
pair_style {style}
{coeff}
pair_modify shift yes
thermo_style custom step pe press
thermo_modify norm no
run 0
"""


def test_wca(exe):
    for r in (.9, 1.05, 1.2):
        free, bound = value(lmp(exe, pair_deck(r))), value(lmp(exe, pair_deck(r, True)))
        assert math.isclose(free, wca(r), rel_tol=1e-5), (r, free, wca(r))
        assert math.isclose(bound-free, fene(r), rel_tol=1e-5), (r, free, bound, fene(r))


def test_restart(exe):
    path = tempfile.mktemp(suffix=".restart")
    out = lmp(exe, f"""units lj
atom_style atomic
atom_modify map yes
region b block 0 10 0 10 0 10
create_box 1 b
create_atoms 1 single 4 5 5
create_atoms 1 single 5.1 5 5
mass 1 1
pair_style associating
pair_coeff * * 30 1.5 1
fix a all associating/kinetics debug_pair 1 2
thermo_style custom step pe press
thermo_modify norm no
run 0
write_restart {path}
clear
read_restart {path}
pair_style associating
pair_coeff * * 30 1.5 1
fix a all associating/kinetics
thermo_style custom step pe press
thermo_modify norm no
run 0
""")
    os.unlink(path)
    vals = [float(x.split()[1]) for x in out.splitlines() if len(x.split()) == 3 and x.split()[0] == "0"]
    assert len(vals) == 2 and math.isclose(vals[0], vals[1], rel_tol=1e-7), out


def test_migration(exe):
    out = lmp(exe, """units lj
atom_style atomic
atom_modify map yes
processors 2 1 1
region b block 0 10 0 10 0 10
create_box 1 b
create_atoms 1 single 4.9 5 5
create_atoms 1 single 6.0 5 5
mass 1 1
pair_style associating
pair_coeff * * 30 1.5 1
fix a all associating/kinetics debug_pair 1 2
fix n all move linear 0.2 0 0
timestep .5
thermo_style custom step pe
thermo_modify norm no
thermo 1
run 2
""", 2)
    assert "partner outside communication range" not in out.lower(), out


def test_kg_topology(exe):
    def deck(assoc):
        return f"""units lj
atom_style bond
atom_modify map yes
region b block 0 10 0 10 0 10
create_box 1 b bond/types 1 extra/bond/per/atom 2
create_atoms 1 single 4 5 5
create_atoms 1 single 4.97 5 5
create_atoms 1 single 4.9 5 5
mass 1 1
create_bonds single/bond 1 1 2
create_bonds single/bond 1 2 3
bond_style fene
bond_coeff 1 30 1.5 1 1
special_bonds fene
pair_style hybrid/overlay lj/cut 1.122462048309373 associating
pair_coeff * * lj/cut 1 1 1.122462048309373
pair_coeff * * associating 30 1.5 1
pair_modify shift yes
{'fix a all associating/kinetics debug_pair 1 3' if assoc else 'fix a all associating/kinetics'}
thermo_style custom step ebond
thermo_modify norm no
run 0
"""
    base, bound = value(lmp(exe, deck(False))), value(lmp(exe, deck(True)))
    assert math.isclose(base, bound, rel_tol=1e-7), (base, bound)


exe = sys.argv[1]
test_wca(exe)
test_restart(exe)
test_migration(exe)
test_kg_topology(exe)
