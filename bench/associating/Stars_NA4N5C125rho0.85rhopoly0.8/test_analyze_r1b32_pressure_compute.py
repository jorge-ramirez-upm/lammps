import json
import subprocess
import sys

import numpy as np


def test_identical_total_and_component_reconstruction(tmp_path):
    total = np.array([[4., 5., 6., .1, .2, .3], [5., 6., 7., .2, .3, .4]])
    prefix = np.array([[0., -10., 20., 1., 2.], [1., -9., 21., 2., 3.]])
    single = np.column_stack((prefix, total))
    pieces = [total * .1, total * .2, total * .3, total * .4]
    multi = np.column_stack((prefix, total, *pieces))
    one, many, out = tmp_path / "one.raw", tmp_path / "many.raw", tmp_path / "out.json"
    np.savetxt(one, single); np.savetxt(many, multi)
    script = __file__.replace("test_analyze_", "analyze_")
    subprocess.run([sys.executable, script, str(one), str(many), "--out", str(out)], check=True)
    result = json.loads(out.read_text())
    assert result["total_tensor_bitwise_equal"]
    assert result["max_abs_total_difference"] == 0
    assert result["max_abs_reconstruction_residual"] < 1e-15
    assert result["R0_difference"] == 0


def test_difference_fails_and_is_reported(tmp_path):
    prefix = np.array([[0., -10., 20., 1., 2.]])
    total = np.array([[4., 5., 6., .1, .2, .3]])
    one = np.column_stack((prefix, total))
    changed = total.copy(); changed[0, 4] += .01
    many = np.column_stack((prefix, changed, total * .1, total * .2, total * .3, total * .4))
    p1, p2, out = tmp_path / "one.raw", tmp_path / "many.raw", tmp_path / "out.json"
    np.savetxt(p1, one); np.savetxt(p2, many)
    script = __file__.replace("test_analyze_", "analyze_")
    process = subprocess.run([sys.executable, script, str(p1), str(p2), "--out", str(out)])
    assert process.returncode != 0
    assert np.isclose(json.loads(out.read_text())["component_max_abs_total_difference"]["xz"], .01)
