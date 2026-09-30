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


if __name__ == "__main__":
    unittest.main()
