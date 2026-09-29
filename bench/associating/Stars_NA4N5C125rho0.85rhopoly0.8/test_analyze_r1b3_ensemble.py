#!/usr/bin/env python3
"""Synthetic regression tests for the independent-ensemble analysis."""

import importlib.util
import pathlib
import unittest

import numpy as np


HERE = pathlib.Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "analyze_r1b3_ensemble", HERE / "analyze_r1b3_ensemble.py"
)
ANALYSIS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ANALYSIS)


class R1B3SyntheticTest(unittest.TestCase):
    def test_six_isotropic_stress_trajectories(self):
        """Isotropic Gaussian tensors have Cn/4 equal to Cs at zero lag."""
        n_lags = 20000
        cs_all = []
        cn_all = []
        d_all = []
        for replica in range(6):
            rng = np.random.default_rng(9173 + replica)
            raw = np.empty((n_lags, 10))
            raw[:, 0] = np.arange(n_lags)
            # Independent diagonal stresses of variance two make each normal
            # difference have variance four; shear stresses have variance one.
            raw[:, 1:4] = rng.normal(scale=np.sqrt(2), size=(n_lags, 3))
            raw[:, 4:7] = rng.normal(size=(n_lags, 3))
            raw[:, 7] = raw[:, 1] - raw[:, 2]
            raw[:, 8] = raw[:, 1] - raw[:, 3]
            raw[:, 9] = raw[:, 2] - raw[:, 3]
            aa, cs, cn, difference = ANALYSIS.replica_correlations(raw)
            self.assertEqual(aa.shape, (n_lags, 6))
            self.assertEqual(cs.shape, (n_lags,))
            self.assertEqual(cn.shape, (n_lags,))
            self.assertEqual(difference.shape, (n_lags,))
            cs_all.append(cs)
            cn_all.append(cn)
            d_all.append(difference)

        cs_all = np.stack(cs_all)
        cn_all = np.stack(cn_all)
        d_all = np.stack(d_all)
        self.assertEqual(cs_all.shape, (6, n_lags))
        self.assertEqual(cn_all.shape, (6, n_lags))
        self.assertEqual(d_all.shape, (6, n_lags))
        self.assertAlmostEqual(cn_all[:, 0].mean() / cs_all[:, 0].mean(), 1.0, delta=0.03)


if __name__ == "__main__":
    unittest.main()
