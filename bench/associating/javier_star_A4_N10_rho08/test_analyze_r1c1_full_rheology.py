#!/usr/bin/env python3
"""Small data-only checks for the R1-C1 analyzer."""

import importlib.util
import pathlib
import tempfile
import unittest

import numpy as np


HERE = pathlib.Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("r1c1", HERE / "analyze_r1c1_full_rheology.py")
r1c1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r1c1)


class R1C1Test(unittest.TestCase):
    def test_isotropy_and_modulus(self):
        n = 256
        rng = np.random.default_rng(41)
        raw = np.zeros((n, 7))
        raw[:, 0] = np.arange(n)
        raw[:, 1:4] = rng.normal(scale=np.sqrt(2.0), size=(n, 3))
        raw[:, 4:7] = rng.normal(size=(n, 3))
        result, time, cs, cn, d, g, eta, useful = r1c1.analyze(raw, 10.0, 1.0, 0.1)
        self.assertEqual(len(time), n)
        self.assertEqual(len(eta), n)
        self.assertAlmostEqual(result["G_0"], 10.0 * (3.0 / 5.0 * cs[0] + 2.0 / 5.0 * cn[0]))
        self.assertTrue(np.all(np.isfinite(g)))

    def test_format_rejects_missing_channel(self):
        with tempfile.NamedTemporaryFile(mode="w+") as handle:
            np.savetxt(handle, np.zeros((4, 6)))
            handle.flush()
            with self.assertRaises(ValueError):
                r1c1.load_raw(handle.name)

    def test_crossing_and_integration(self):
        time = np.arange(5.0)
        self.assertAlmostEqual(r1c1.first_crossing(time, np.array([1., .8, .4, .05, -.1]), .1), 2.857142857142857)
        raw = np.zeros((5, 7)); raw[:, 0] = np.arange(5)
        raw[:, 1] = np.arange(5) + 1.0
        raw[:, 2] = raw[:, 1]; raw[:, 3] = raw[:, 1]
        result, _, _, _, _, _, eta, _ = r1c1.analyze(raw, 1.0, 1.0, 1.0)
        self.assertAlmostEqual(eta[-1], 0.0)
        self.assertIn("G_over_G0_crossings", result)

    def test_nested_duration_and_fixed_cutoff(self):
        n = 2001
        raw = np.zeros((n, 7)); raw[:, 0] = np.arange(n)
        raw[:, 1:4] = 2.0
        durations, cutoffs = r1c1.duration_rows(raw, [0.5, 20.0], 1.0, 1.0, 0.01,
                                                 max_lag=0.5)
        self.assertEqual({row["duration"] for row in durations}, {0.5, 20.0})
        self.assertTrue(any(row["cutoff"] == 20 for row in cutoffs))

    def test_block_statistics(self):
        raw = np.zeros((41, 7)); raw[:, 0] = np.arange(41)
        raw[:, 1:4] = np.arange(41)[:, None]
        rows = r1c1.block_rows(raw, [2], 1.0, 1.0, 1.0)
        summary = r1c1.block_summary(rows)
        self.assertEqual(len([row for row in summary if row["lag"] == 1]), 1)
        self.assertGreaterEqual(summary[0]["G_sd"], 0.0)

    def test_network_overlap_and_sampled_survival(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            snapshots = [
                (0, [(1, 2), (3, 4)]),
                (100, [(1, 2), (5, 6)]),
                (200, [(5, 6)]),
            ]
            for timestep, edges in snapshots:
                path = pathlib.Path(directory) / f"network.{timestep}.dat"
                path.write_text(f"# timestep {timestep} bonds {len(edges)}\n# tag_i tag_j molecule_i molecule_j\n" +
                                "".join(f"{a} {b} 1 2\n" for a, b in edges))
                paths.append(path)
            info, rows = r1c1.network_persistence(paths, 100.0)
            self.assertEqual(info["status"], "ok")
            self.assertAlmostEqual(rows[1]["Q_edge_overlap"], 0.5)
            self.assertAlmostEqual(rows[2]["sampled_continuous_survival"], 0.0)

    def test_network_missing_and_malformed(self):
        info, rows = r1c1.network_persistence([], 100.0)
        self.assertEqual(info["status"], "missing")
        with tempfile.NamedTemporaryFile(mode="w+") as handle:
            handle.write("not a network\n")
            handle.flush()
            info, rows = r1c1.network_persistence([pathlib.Path(handle.name)], 100.0)
        self.assertEqual(info["status"], "malformed")


if __name__ == "__main__":
    unittest.main()
