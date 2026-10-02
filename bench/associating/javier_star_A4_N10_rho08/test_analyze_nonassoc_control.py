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
        summary, rows, cutoffs, blocks, block_summary, uncertainty, tail_bins, terminal, duration = control._rheology_rows(
            raw, [1.0, 5.0], [1.0, 2.0], [2], 10.0, 1.0, 0.01, 2.0)
        self.assertEqual([item["duration"] for item in summary], [1.0, 5.0])
        self.assertTrue(any(item["cutoff"] == 2.0 for item in cutoffs))
        self.assertTrue(any(item["source"] == "blocks" and item["eta_sem"] is not None for item in cutoffs))
        stability = control.fixed_cutoff_stability(cutoffs)
        self.assertTrue(stability["cutoffs"])
        self.assertTrue(block_summary)
        self.assertTrue(tail_bins)
        self.assertTrue(duration)
        self.assertIn("first_G_below_block_SEM", terminal)

    def test_known_fickian_msd(self):
        rng = np.random.default_rng(19)
        dt = 0.1
        diffusion = 0.25
        increments = rng.normal(scale=np.sqrt(2.0 * diffusion * dt), size=(256, 200, 3))
        com = np.cumsum(increments, axis=0)
        time, msd = control.msd_from_com(com, dt)
        alpha, result = control.diffusion_diagnostics(time, msd)
        self.assertAlmostEqual(result["D"], diffusion, delta=0.06)
        self.assertTrue(result["D_candidates"])
        self.assertLessEqual(max(item["t_max"] for item in result["D_candidates"]), time[-1] / 5 + 1e-12)
        self.assertGreater(np.nanmedian(alpha[-50:]), 0.8)
        self.assertLess(np.nanmedian(alpha[-50:]), 1.2)

    def test_diffusion_requires_sustained_window(self):
        time = np.arange(0.0, 10.1, 0.1)
        msd = 6.0 * 0.5 * time
        msd[1:5] *= np.array([0.2, 0.4, 0.7, 0.9])
        _, result = control.diffusion_diagnostics(time, msd, sustained_points=7)
        self.assertIsNotNone(result["fickian_onset"])
        self.assertGreaterEqual(result["fickian_window_end"], result["fickian_onset"])
        self.assertTrue(all(item["t_max"] <= 2.0 + 1e-12 for item in result["D_candidates"]))

    def test_terminal_diagnostic_does_not_invent_tau(self):
        rows = []
        bins = []
        block_rows = []
        for lag, mean in ((0.0, 4.0), (1.0, 2.0), (2.0, 0.2), (3.0, -0.1), (4.0, 0.3)):
            for block in (1, 2, 3, 4, 5):
                block_rows.append({"blocks": 5, "block": block, "lag": lag,
                                   "G": mean, "eta": mean * (lag + 1)})
            rows.append({"blocks": 5, "lag": lag, "G_mean": mean,
                         "G_sd": 0.4, "G_sem": 0.2,
                         "G_sd_over_abs_mean": abs(0.4 / mean) if mean else None})
            bins.append({"blocks": 5, "lag_start": lag, "lag_end": lag + 1.0,
                         "G_mean": mean, "G_sd": 0.4, "G_sem": 0.2, "n_samples": 5})
        diagnostic = control.terminal_diagnostic(bins, block_rows, [5], full_g0=4.0)
        self.assertIsNone(diagnostic.get("tau_term"))
        self.assertEqual(diagnostic["first_G_below_block_SEM"], 2.0)
        self.assertEqual(diagnostic["first_G_below_block_SD"], 2.0)
        self.assertEqual(diagnostic["longest_sustained_positive_interval"]["end"], 3.0)
        self.assertEqual(diagnostic["largest_contiguous_resolved_range"]["end"], 2.0)

    def test_cutoff_stability_is_prefix_contiguous(self):
        rows = []
        for cutoff, stable in ((1.0, True), (2.0, True), (5.0, False), (10.0, True)):
            for duration in (100.0, 200.0):
                rows.append({"source": "nested", "cutoff": cutoff, "duration": duration,
                             "eta_cutoff": 1.0 if stable else (1.0 if duration == 100 else 2.0),
                             "safe_lag_fraction": cutoff / duration})
            rows.append({"source": "blocks", "cutoff": cutoff, "blocks": 5,
                         "eta_cutoff": 1.0, "eta_sem": 0.1,
                         "safe_lag_fraction": cutoff / 1000.0})
        result = control.fixed_cutoff_stability(rows)
        self.assertEqual(result["largest_contiguous_stable_cutoff"], 2.0)
        self.assertEqual(result["individually_stable_cutoffs"], [1.0, 2.0, 10.0])
        self.assertEqual(result["isolated_later_passes_non_converged"], [10.0])

    def test_unsupported_late_cutoff_cannot_converge(self):
        rows = [{"source": "nested", "cutoff": 3000.0, "duration": 50000.0,
                 "eta_cutoff": 10.0, "safe_lag_fraction": 0.06}]
        result = control.fixed_cutoff_duration_convergence(rows)[0]
        self.assertFalse(result["supported_by_block_criteria"])
        self.assertIsNone(result["T_min_25pct"])
        self.assertEqual(result["status"], "unsupported_safe_lag_or_blocks")

    def test_later_alpha_windows_and_trend_are_reported(self):
        time = np.arange(0.0, 40000.1, 10.0)
        msd = time ** 0.6
        _, result = control.diffusion_diagnostics(time, msd)
        windows = result["alpha_in_requested_windows"]
        self.assertEqual([row["window"] for row in windows[-6:]],
                         ["1000-2000", "2000-4000", "4000-8000", "8000-16000",
                          "16000-32000", "32000-45000"])
        self.assertEqual(result["alpha_later_window_trend"], "persistently_subdiffusive")

    def test_duration_convergence_uses_suffix_not_first_prefix(self):
        rows = []
        for cutoff, values in ((1.0, (0.5, 1.0, 1.02)), (2.0, (1.0, 1.01, 1.02))):
            for duration, value in zip((10.0, 20.0, 40.0), values):
                rows.append({"source": "nested", "cutoff": cutoff, "duration": duration,
                             "eta_cutoff": value, "safe_lag_fraction": cutoff / duration})
            rows.append({"source": "blocks", "cutoff": cutoff, "blocks": 5,
                         "eta_cutoff": 1.0, "eta_sem": 0.05,
                         "safe_lag_fraction": cutoff / 100.0})
        result = control.fixed_cutoff_duration_convergence(rows)
        first = result[0]
        self.assertEqual(first["T_min_10pct"], 20.0)
        self.assertEqual(first["T_min_25pct"], 20.0)
        self.assertEqual(result[1]["T_min_10pct"], 10.0)

    def test_plateau_requires_duration_converged_values(self):
        rows = []
        for cutoff in (1.0, 2.0, 5.0, 10.0):
            rows.append({"cutoff": cutoff, "T_min_25pct": 100.0,
                         "T_longest": 1000.0, "eta_from_longest_T": 1.0,
                         "block_SEM": 0.05})
        result = control.cutoff_plateau(rows)
        self.assertTrue(result["plateau_exists"])
        self.assertEqual(result["plateau_cutoffs"], [1.0, 2.0, 5.0, 10.0])

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
