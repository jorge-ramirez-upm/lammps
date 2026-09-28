#!/usr/bin/env python3
"""Regression coverage for write_associating_network. Usage: python3 $0 lmp"""
import os
import subprocess
import sys
import tempfile

LMP = sys.argv[1]


def run(deck, ranks=1):
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "network.dat")
        result = subprocess.run(["mpirun", "-np", str(ranks), LMP, "-log", "none", "-screen", "none"],
                                input=deck.format(path=path, restart=os.path.join(tmp, "state.restart")),
                                text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        assert result.returncode == 0, result.stdout
        with open(path) as fp:
            edges = [tuple(map(int, line.split())) for line in fp if line and not line.startswith("#")]
        return edges


def base(atoms, molecules):
    lines = ["units lj", "atom_style bond", "atom_modify map yes", "region b block 0 10 0 10 0 10",
             "create_box 1 b"]
    for xyz in atoms:
        lines.append("create_atoms 1 single {} {} {}".format(*xyz))
    lines += ["mass 1 1"]
    for i, molecule in enumerate(molecules, 1):
        lines.append(f"set atom {i} mol {molecule}")
    return "\n".join(lines) + "\n"


def debug_case(molecules, ranks=1, migration=False):
    deck = base([(4.5, 5, 5), (5.5, 5, 5)], molecules)
    if migration: deck = deck.replace("region b", "processors 2 1 1\nregion b")
    lines = [deck, "pair_style associating",
             "pair_coeff * * 30 1.5 1", "fix a all associating/kinetics debug_pair 1 2"]
    if migration:
        lines += ["fix move all move linear .2 0 0", "timestep .5", "run 6"]
    else:
        lines += ["run 0"]
    lines += ["write_associating_network {path} fix a"]
    return run("\n".join(lines), ranks)


def assert_canonical(edges):
    assert all(a < b for a, b, _, _ in edges), edges
    assert len({(a, b) for a, b, _, _ in edges}) == len(edges), edges
    stickers = [tag for edge in edges for tag in edge[:2]]
    assert len(stickers) == len(set(stickers)), edges


def test_intramolecular():
    assert debug_case([7, 7]) == [(1, 2, 7, 7)]


def test_intermolecular():
    assert debug_case([7, 9]) == [(1, 2, 7, 9)]


def test_multiple_and_uniqueness():
    deck = base([(2, 5, 5), (2.9, 5, 5), (7, 5, 5), (7.9, 5, 5)], [1, 1, 2, 3]) + """pair_style associating
pair_coeff * * 30 1.5 100
fix a all associating/kinetics 1 19 1e9 0 1 1.2
fix hold all move linear 0 0 0
run 1
write_associating_network {path} fix a
"""
    edges = run(deck)
    assert edges == [(1, 2, 1, 1), (3, 4, 2, 3)], edges
    assert_canonical(edges)


def test_cross_domain():
    deck = base([(4.5, 5, 5), (5.5, 5, 5)], [10, 20]).replace("region b", "processors 2 1 1\nregion b") + """
pair_style associating
pair_coeff * * 30 1.5 1
fix a all associating/kinetics debug_pair 1 2
run 0
write_associating_network {path} fix a
"""
    edges = run(deck, 2)
    assert edges == [(1, 2, 10, 20)], edges
    assert_canonical(edges)


def test_migration():
    edges = debug_case([10, 20], 2, migration=True)
    assert edges == [(1, 2, 10, 20)], edges


def test_restart():
    deck = base([(4.5, 5, 5), (5.5, 5, 5)], [10, 20]) + """pair_style associating
pair_coeff * * 30 1.5 1
fix a all associating/kinetics debug_pair 1 2
run 0
write_restart {restart}
clear
read_restart {restart}
pair_style associating
pair_coeff * * 30 1.5 1
fix a all associating/kinetics
run 0
write_associating_network {path} fix a
"""
    edges = run(deck)
    assert edges == [(1, 2, 10, 20)], edges
    assert_canonical(edges)


test_intramolecular()
test_intermolecular()
test_multiple_and_uniqueness()
test_cross_domain()
test_migration()
test_restart()
