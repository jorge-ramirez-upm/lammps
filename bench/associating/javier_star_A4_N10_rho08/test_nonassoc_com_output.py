#!/usr/bin/env python3
"""Small LAMMPS regression: direct COM output equals xu atom reconstruction."""
import os
import pathlib
import subprocess
import tempfile
import unittest

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
LMP = os.environ.get("LMP", str(HERE.parents[2] / "build-associating" / "lmp"))


def read_com(path):
    frames = []
    with open(path) as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            step, count = map(int, line.split())
            rows = [handle.readline().split() for _ in range(count)]
            frames.append((step, np.array([[float(x) for x in row[2:5]] for row in rows])))
    return frames


def read_atoms(path):
    frames = []
    with open(path) as handle:
        while True:
            line = handle.readline()
            if not line:
                return frames
            if line.strip() != "ITEM: TIMESTEP":
                continue
            step = int(handle.readline())
            if handle.readline().strip() != "ITEM: NUMBER OF ATOMS":
                raise AssertionError("bad atom dump")
            count = int(handle.readline())
            handle.readline()
            for _ in range(3):
                handle.readline()
            columns = {name: i for i, name in enumerate(handle.readline().split()[2:])}
            centers, counts = {}, {}
            for _ in range(count):
                fields = handle.readline().split()
                mol = int(fields[columns["mol"]])
                centers.setdefault(mol, np.zeros(3))
                centers[mol] += [float(fields[columns[x]]) for x in ("xu", "yu", "zu")]
                counts[mol] = counts.get(mol, 0) + 1
            molecules = sorted(centers)
            frames.append((step, np.array([centers[m] / counts[m] for m in molecules])))


@unittest.skipUnless(pathlib.Path(LMP).is_file(), "set LMP to a built LAMMPS executable")
class NonassocCOMSmokeTest(unittest.TestCase):
    def test_com_matches_unwrapped_atom_reconstruction(self):
        with tempfile.TemporaryDirectory() as temp:
            out = pathlib.Path(temp) / "run"
            env = os.environ.copy()
            env.update({"LMP": LMP, "MPI_NP": "1", "PROD_STEPS": "20",
                        "COM_EVERY": "10", "TRAJ_EVERY": "10", "FULL_TRAJ": "1",
                        "ONLINE": "0", "OUT": str(out)})
            subprocess.run([str(HERE / "run_nonassoc_control_linux.sh")],
                           cwd=HERE, env=env, check=True, stdout=subprocess.DEVNULL)
            direct = read_com(out / "production/control_nonassoc.com")
            atom = read_atoms(out / "production/control_nonassoc.lammpstrj")
            self.assertEqual([step for step, _ in direct], [step for step, _ in atom])
            self.assertEqual(len(direct), 3)  # initial frame plus two COM samples
            for (_, expected), (_, reconstructed) in zip(direct, atom):
                # fix ave/time's default six-decimal formatting bounds the
                # comparison, while xu remains unwrapped across PBCs.
                np.testing.assert_allclose(expected, reconstructed, rtol=0, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
