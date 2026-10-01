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


def event_case(logging, ranks=2):
    with tempfile.TemporaryDirectory() as tmp:
        event = os.path.join(tmp, "events.dat")
        initial = os.path.join(tmp, "initial.network")
        final = os.path.join(tmp, "final.network")
        option = f" event_log {event}" if logging else ""
        event_base = (base([(4.5, 5, 5), (5.5, 5, 5)], [10, 20])
                      .replace("create_box 1 b", "create_box 2 b")
                      .replace("create_atoms 1", "create_atoms 2")
                      .replace("mass 1 1", "mass 1 1\nmass 2 1"))
        deck = event_base + f"""pair_style hybrid/overlay lj/cut 1.122462 associating
pair_coeff * * lj/cut 1 1 1.122462
pair_coeff * * associating 30 1.5 8
fix k all associating/kinetics 1 19 1e9 0 1000000 1.2{option}
write_associating_network {initial} fix k
run 4
write_associating_network {final} fix k
"""
        result = subprocess.run(["mpirun", "-np", str(ranks), LMP, "-log", "none", "-screen", "none"],
                                input=deck, text=True, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT)
        assert result.returncode == 0, result.stdout
        def read_edges(path):
            with open(path) as fp:
                return [tuple(map(int, line.split())) for line in fp
                        if line and not line.startswith("#")]
        events = []
        if logging:
            with open(event) as fp:
                events = [line.split() for line in fp if line and not line.startswith("#")]
        return read_edges(initial), read_edges(final), events


def test_event_logging_and_replay():
    initial, final, events = event_case(True)
    assert initial == []
    assert final == []
    assert [row[1] for row in events] == ["C", "B", "C", "B"]
    assert all(int(row[2]) < int(row[3]) for row in events)
    active = set(initial)
    for row in events:
        edge = tuple(map(int, row[2:4]))
        if row[1] == "C":
            assert edge not in active
            active.add(edge)
        else:
            assert edge in active
            active.remove(edge)
    assert active == set(final)


def test_event_logging_off_on_equivalence():
    initial_off, final_off, events_off = event_case(False)
    initial_on, final_on, events_on = event_case(True)
    assert events_off == []
    assert initial_off == initial_on == []
    assert final_off == final_on == []
    assert events_on


def test_event_logging_restart_continuity():
    with tempfile.TemporaryDirectory() as tmp:
        event1 = os.path.join(tmp, "events.1.dat")
        event2 = os.path.join(tmp, "events.2.dat")
        restart = os.path.join(tmp, "state.restart")
        initial2 = os.path.join(tmp, "initial.2.network")
        final2 = os.path.join(tmp, "final.2.network")
        event_base = (base([(4.5, 5, 5), (5.5, 5, 5)], [10, 20])
                      .replace("create_box 1 b", "create_box 2 b")
                      .replace("create_atoms 1", "create_atoms 2")
                      .replace("mass 1 1", "mass 1 1\nmass 2 1"))
        deck = event_base + f"""pair_style hybrid/overlay lj/cut 1.122462 associating
pair_coeff * * lj/cut 1 1 1.122462
pair_coeff * * associating 30 1.5 8
fix k all associating/kinetics 1 19 1e9 0 1000000 1.2 event_log {event1}
run 1
write_restart {restart}
clear
read_restart {restart}
pair_style hybrid/overlay lj/cut 1.122462 associating
pair_coeff * * lj/cut 1 1 1.122462
pair_coeff * * associating 30 1.5 8
fix k all associating/kinetics 1 19 1e9 0 1000000 1.2 event_log {event2}
write_associating_network {initial2} fix k
run 1
write_associating_network {final2} fix k
"""
        result = subprocess.run(["mpirun", "-np", "2", LMP, "-log", "none", "-screen", "none"],
                                input=deck, text=True, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT)
        assert result.returncode == 0, result.stdout
        def read_events(path):
            with open(path) as fp:
                return [line.split() for line in fp if line and not line.startswith("#")]
        assert read_events(event1)[0][:2] == ["1", "C"]
        assert read_events(event2)[0][:2] == ["2", "B"]
        with open(initial2) as fp:
            initial_edges = [tuple(map(int, line.split())) for line in fp
                             if line and not line.startswith("#")]
        with open(final2) as fp:
            final_edges = [tuple(map(int, line.split())) for line in fp
                           if line and not line.startswith("#")]
        assert initial_edges == [(1, 2, 10, 20)]
        assert final_edges == []
        active = {(edge[0], edge[1]) for edge in initial_edges}
        for row in read_events(event2):
            edge = (int(row[2]), int(row[3]))
            if row[1] == "C": active.add(edge)
            else: active.remove(edge)
        assert active == {(edge[0], edge[1]) for edge in final_edges}


def test_event_log_refuses_existing_file():
    with tempfile.TemporaryDirectory() as tmp:
        event = os.path.join(tmp, "events.dat")
        event_base = (base([(4.5, 5, 5), (5.5, 5, 5)], [1, 2])
                      .replace("create_box 1 b", "create_box 2 b")
                      .replace("create_atoms 1", "create_atoms 2")
                      .replace("mass 1 1", "mass 1 1\nmass 2 1"))
        deck = event_base + f"""pair_style hybrid/overlay lj/cut 1.122462 associating
pair_coeff * * lj/cut 1 1 1.122462
pair_coeff * * associating 30 1.5 8
fix k all associating/kinetics 1 19 1e9 0 1000000 1.2 event_log {event}
run 1
"""
        first = subprocess.run(["mpirun", "-np", "1", LMP, "-log", "none", "-screen", "none"],
                               input=deck, text=True, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT)
        second = subprocess.run(["mpirun", "-np", "1", LMP, "-log", "none", "-screen", "none"],
                                input=deck, text=True, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT)
        assert first.returncode == 0, first.stdout
        assert second.returncode != 0, second.stdout


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
test_event_logging_and_replay()
test_event_logging_off_on_equivalence()
test_event_logging_restart_continuity()
test_event_log_refuses_existing_file()
