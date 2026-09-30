#!/usr/bin/env python3
"""Small, data-independent checks for the R1-B3.1 forensic analyzer."""

import importlib.util
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "r1b31", HERE / "analyze_r1b31_zero_lag.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_direct_known_value_and_fft_agreement():
    stress = np.array([
        [3.0, 1.0, 2.0, 1.0, 2.0, 3.0],
        [1.0, 4.0, 2.0, -2.0, 1.0, -1.0],
        [2.0, 2.0, 2.0, 0.5, -0.5, 1.5],
    ])
    shear_squared_sum = 14.0 + 6.0 + 2.75
    normal_squared_sum = 6.0 + 14.0 + 0.0
    expected_cs = shear_squared_sum / 9.0
    expected_cn4 = normal_squared_sum / 9.0 / 4.0
    direct = MODULE.r0(stress)
    transformed = MODULE.fft_r0(stress)
    np.testing.assert_allclose(direct[:2], (expected_cs, expected_cn4))
    np.testing.assert_allclose(transformed, direct, rtol=2e-15, atol=2e-15)


def test_edges_are_undirected_and_ignore_headers(tmp_path):
    path = tmp_path / "network.dat"
    path.write_text("# a b\n2 1\n3 4 99\n1 2\n", encoding="utf-8")
    assert MODULE.edges(path) == {(1, 2), (3, 4)}
