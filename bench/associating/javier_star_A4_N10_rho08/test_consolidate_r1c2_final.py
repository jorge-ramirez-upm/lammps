import importlib.util
import pathlib
import tempfile
import unittest


HERE = pathlib.Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("final", HERE / "consolidate_r1c2_final.py")
final = importlib.util.module_from_spec(spec)
spec.loader.exec_module(final)


class FinalConsolidationTest(unittest.TestCase):
    def test_late_prefix_plateau_ignores_unconverged_prefixes(self):
        summary = {"rheology": {"duration_convergence": [
            {"cutoff": c, "T_min_25pct": 50.0, "block_SEM": 2.0,
             "supported_by_block_criteria": True} for c in (1500.0, 2000.0, 3000.0, 5000.0)]}}
        rows = []
        for c, values in {1500.0: (10.0, 10.0), 2000.0: (19.0, 20.0),
                          3000.0: (20.0, 21.0), 5000.0: (21.0, 22.0)}.items():
            rows.extend({"source": "nested", "cutoff": str(c), "duration": str(t),
                         "safe_lag_fraction": "0.1", "eta_cutoff": str(v)}
                        for t, v in ((40.0, values[0]), (50.0, values[1])))
        result = final.late_prefix_plateau(summary, rows)
        self.assertEqual(result["recommended_interval"], [2000.0, 3000.0, 5000.0])
        self.assertAlmostEqual(result["eta0_estimate"], 21.0)

    def test_markdown_declares_production_sufficient(self):
        report = {"production_sufficiency": {"status": "PRODUCTION_SUFFICIENT"},
                  "rheology": {"late_prefix_plateau": {"recommended_interval": [2000.0, 5000.0],
                                                           "eta0_estimate": 75.0, "eta0_uncertainty": 13.0,
                                                           "sensitivity_range": [68.0, 80.0]}},
                  "diffusion": {"D": 4.4e-4, "formal_strict_exponent_criterion_preserved": False},
                  "sticker_times": {"associating": {"bare": {"median_time": 1167.0, "one_over_e_time": 1692.0},
                                                       "renormalized": {"median_time": 1867.0, "one_over_e_time": 2699.0}}},
                  "comparison": {"G0": {"associating": 68.0, "nonassociating": 66.0},
                                  "COM_diffusion": {"associating_over_nonassociating": 0.25},
                                  "sticker_times": {"associating": {"bare": {"median_time": 1167.0, "one_over_e_time": 1692.0},
                                                                        "renormalized": {"median_time": 1867.0, "one_over_e_time": 2699.0}}}}}
        text = final.markdown(report)
        self.assertIn("PRODUCTION_SUFFICIENT", text)
        self.assertIn("effective_candidate_slope", text)


if __name__ == "__main__":
    unittest.main()
