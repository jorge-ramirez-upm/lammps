#!/usr/bin/env python3
import importlib.util
import pathlib
import tempfile
import unittest

import numpy as np


HERE = pathlib.Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("control", HERE / "analyze_nonassoc_control.py")
control = importlib.util.module_from_spec(spec)
spec.loader.exec_module(control)


class NonassocControlTest(unittest.TestCase):
    def test_nested_duration_rows_and_fixed_cutoffs(self):
        n = 1001
        raw = np.zeros((n, 7)); raw[:, 0] = np.arange(n)
        raw[:, 1:4] = 1.0
        summary, rows, cutoffs, blocks, block_summary, uncertainty = control._rheology_rows(
            raw, [1.0, 5.0], [1.0, 2.0], [2], 10.0, 1.0, 0.01, 2.0)
        self.assertEqual([item["duration"] for item in summary], [1.0, 5.0])
        self.assertTrue(any(item["cutoff"] == 2.0 for item in cutoffs))
        self.assertTrue(block_summary)

    def test_known_fickian_msd(self):
        rng = np.random.default_rng(19)
        dt = 0.1
        diffusion = 0.25
        increments = rng.normal(scale=np.sqrt(2.0 * diffusion * dt), size=(256, 200, 3))
        com = np.cumsum(increments, axis=0)
        time, msd = control.msd_from_com(com, dt)
        alpha, result = control.diffusion_diagnostics(time, msd)
        self.assertAlmostEqual(result["D"], diffusion, delta=0.06)
        self.assertGreater(np.nanmedian(alpha[-50:]), 0.8)
        self.assertLess(np.nanmedian(alpha[-50:]), 1.2)

    def test_malformed_trajectory(self):
        with tempfile.NamedTemporaryFile(mode="w+") as handle:
            handle.write("ITEM: TIMESTEP\n0\n")
            handle.flush()
            with self.assertRaises(ValueError):
                list(control.dump_frames(handle.name, expected_atoms=0))


if __name__ == "__main__":
    unittest.main()
