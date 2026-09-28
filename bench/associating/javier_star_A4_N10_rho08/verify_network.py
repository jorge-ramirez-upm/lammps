#!/usr/bin/env python3
"""Verify canonical transient-network snapshots against associating thermo."""
import sys

active = {}
for line in open(sys.argv[1]):
    fields = line.split()
    if len(fields) == 7:
        try: active[int(fields[0])] = int(fields[4])
        except ValueError: pass
for path in sys.argv[2:]:
    lines = open(path).read().splitlines()
    header = lines[0].split()
    timestep, count = int(header[2]), int(header[4])
    edges = [tuple(map(int, line.split())) for line in lines[2:] if line]
    assert len(edges) == count == active[timestep], (path, len(edges), count, active.get(timestep))
    assert all(a < b for a, b, _, _ in edges), edges
    assert len({(a, b) for a, b, _, _ in edges}) == count, edges
    stickers = [tag for edge in edges for tag in edge[:2]]
    assert len(stickers) == len(set(stickers)), edges
    self_bonds = sum(m1 == m2 for _, _, m1, m2 in edges)
    assert self_bonds + (count-self_bonds) == count
    print(f"{path}: total={count} self={self_bonds} intermolecular={count-self_bonds}")
