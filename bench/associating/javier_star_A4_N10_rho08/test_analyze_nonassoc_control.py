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

    def test_direct_com_output_preserves_ids_and_unwrapped_crossing(self):
        with tempfile.NamedTemporaryFile(mode="w+") as handle:
            handle.write("# Time-averaged data for fix starcom\n")
            handle.write("# TimeStep Number-of-rows\n")
            handle.write("# Row c_star_ids c_star_com[1] c_star_com[2] c_star_com[3]\n")
            handle.write("0 2\n1 11 17.9 0.0 0.0\n2 12 -18.2 0.0 0.0\n")
            handle.write("100 2\n1 11 18.4 0.0 0.0\n2 12 -17.7 0.0 0.0\n")
            handle.flush()
            frames = list(control.com_frames(handle.name, expected_stars=2))
        self.assertEqual([frame[2] for frame in frames], [[11, 12], [11, 12]])
        self.assertEqual(frames[1][0], 100)
        self.assertAlmostEqual(frames[1][1][0, 0], 18.4)

    def test_malformed_com_output(self):
        with tempfile.NamedTemporaryFile(mode="w+") as handle:
            handle.write("0 not-a-count\n")
            handle.flush()
            with self.assertRaises(ValueError):
                list(control.com_frames(handle.name, expected_stars=0))


if __name__ == "__main__":
    unittest.main()
