#!/usr/bin/env python3
import os
import pathlib
import shutil
import subprocess
import tempfile
import unittest

HERE = pathlib.Path(__file__).resolve().parent
LMP = pathlib.Path(os.environ.get("LMP", str(HERE.parents[2] / "build-associating" / "lmp")))
RESTART = HERE / "r1c1_runs" / "production" / "production.restart"


@unittest.skipUnless(LMP.is_file() and RESTART.is_file() and shutil.which("mpirun"),
                     "requires local LAMMPS executable, R1-C1 restart, and mpirun")
class Run0NetworkRegression(unittest.TestCase):
    def test_run0_preserves_restart_network_and_timestep(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            result = subprocess.run([
                "mpirun", "-np", "2", str(LMP), "-log", str(root / "run.log"), "-screen", "none",
                "-var", "RESTART", str(RESTART), "-var", "NETWORK1", str(root / "network1.dat"),
                "-var", "NETWORK2", str(root / "network2.dat"), "-var", "LANGEVIN_SEED", "48279",
                "-var", "KINETICS_SEED", "492845", "-in", str(HERE / "in.r1c2_run0_network_regression.lmp")
            ], cwd=HERE, text=True, capture_output=True, check=False)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual((root / "network1.dat").read_bytes(), (root / "network2.dat").read_bytes())
            header = (root / "network1.dat").read_text().splitlines()[0]
            self.assertIn("# timestep", header)
            log = (root / "run.log").read_text()
            self.assertNotIn("ERROR", log)
            thermo_rows = [line.split() for line in log.splitlines()
                           if line.split() and line.split()[0] == "1000000" and len(line.split()) >= 4]
            self.assertGreaterEqual(len(thermo_rows), 2)
            self.assertEqual(thermo_rows[-1][1:4], thermo_rows[-2][1:4])


if __name__ == "__main__":
    unittest.main()
