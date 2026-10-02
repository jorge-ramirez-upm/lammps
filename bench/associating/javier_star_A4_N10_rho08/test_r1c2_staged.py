#!/usr/bin/env python3
import importlib.util
import os
import pathlib
import subprocess
import re
import tempfile
import unittest

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("staged", HERE / "analyze_r1c2_staged.py")
staged = importlib.util.module_from_spec(spec)
spec.loader.exec_module(staged)


class R1C2StagedTest(unittest.TestCase):
    def test_concatenation_duplicate_gap_and_overlap(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            a = root / "a.raw"; b = root / "b.raw"; gap = root / "gap.raw"; mismatch = root / "mismatch.raw"
            rows = lambda steps: "# step Pxx Pyy Pzz Pxy Pxz Pyz\n" + "\n".join(
                f"{step} 1 1 1 0 0 0" for step in steps) + "\n"
            a.write_text(rows([0, 1, 2])); b.write_text(rows([2, 3])); gap.write_text(rows([5])); mismatch.write_text(rows([1, 4]))
            self.assertEqual(len(staged.concatenate_raw([a, b])), 4)
            with self.assertRaises(ValueError): staged.concatenate_raw([a, gap])
            with self.assertRaises(ValueError): staged.concatenate_raw([a, mismatch])

    def test_com_continuation_duplicate_and_gap(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            content = ("# TimeStep Number-of-rows\n# Row ids x y z\n"
                       "0 2\n1 11 0 0 0\n2 12 1 0 0\n"
                       "100 2\n1 11 1 0 0\n2 12 2 0 0\n")
            a = root / "a.com"; b = root / "b.com"; gap = root / "gap.com"
            a.write_text(content)
            b.write_text("# TimeStep Number-of-rows\n# Row ids x y z\n100 2\n1 11 1 0 0\n2 12 2 0 0\n200 2\n1 11 2 0 0\n2 12 3 0 0\n")
            gap.write_text(content.replace("0 2", "300 2", 1))
            self.assertEqual(len(staged.concatenate_com([a, b])), 3)
            with self.assertRaises(ValueError): staged.concatenate_com([a, gap])

    def test_network_replay_and_lifetimes(self):
        initial = {(9, 10)}
        events = [
            staged.Event(1, "C", 1, 2, 1, 1),
            staged.Event(4, "B", 1, 2, 1, 1),
            staged.Event(5, "C", 1, 2, 1, 1),
            staged.Event(6, "B", 1, 2, 1, 1),
            staged.Event(8, "C", 1, 3, 1, 2),
            staged.Event(9, "B", 1, 3, 1, 2),
            staged.Event(10, "C", 4, 5, 2, 3),
        ]
        self.assertEqual(staged.replay_network(initial, events), {(4, 5), (9, 10)})
        records = staged.lifetime_records(initial, events, end_timestep=100)
        renorm = records["renormalized"]
        self.assertTrue(any(row["termination"] == "third_partner" and row["duration"] == 5 for row in renorm))
        self.assertTrue(any(row["right_censored"] for row in renorm))
        self.assertTrue(any(row["left_censored"] for row in records["bare"]))

    def test_bare_restarts_but_renormalized_episode_continues(self):
        initial = {(1, 2)}
        events = [
            staged.Event(10, "B", 1, 2, 1, 1),
            staged.Event(11, "C", 1, 2, 1, 1),
            staged.Event(14, "B", 1, 2, 1, 1),
        ]
        records = staged.lifetime_records(initial, events, end_timestep=20)
        bare = records["bare"]
        renorm = records["renormalized"]
        self.assertEqual(sum(row["left_censored"] for row in bare), 1)
        observed_bare = [row for row in bare if not row["right_censored"]]
        self.assertEqual([(row["start"], row["end"], row["left_censored"])
                          for row in observed_bare], [(None, 10, True), (11, 14, False)])
        self.assertEqual(len(renorm), 1)
        self.assertEqual((renorm[0]["start"], renorm[0]["end"], renorm[0]["duration"],
                          renorm[0]["right_censored"]), (None, 20, None, True))

    def test_repeated_flickers_and_third_partner(self):
        initial = {(1, 2)}
        events = [
            staged.Event(2, "B", 1, 2, 1, 1),
            staged.Event(3, "C", 1, 2, 1, 1),
            staged.Event(5, "B", 1, 2, 1, 1),
            staged.Event(6, "C", 1, 2, 1, 1),
            staged.Event(8, "B", 1, 2, 1, 1),
            staged.Event(9, "C", 1, 3, 1, 2),
        ]
        records = staged.lifetime_records(initial, events, end_timestep=20)
        bare_observed = [row for row in records["bare"] if not row["right_censored"]]
        self.assertEqual(len(bare_observed), 3)
        self.assertEqual(sum(row["left_censored"] for row in records["bare"]), 1)
        renorm_observed = [row for row in records["renormalized"] if not row["right_censored"]]
        self.assertEqual(len(renorm_observed), 1)
        self.assertEqual(renorm_observed[0]["termination"], "third_partner")
        self.assertEqual(renorm_observed[0]["end"], 8)

    def test_bare_and_renormalized_survival(self):
        initial = set()
        events = [staged.Event(1, "C", 1, 2, 1, 1), staged.Event(4, "B", 1, 2, 1, 1)]
        records = staged.lifetime_records(initial, events, end_timestep=10)
        curve = staged.kaplan_meier(records["bare"])
        self.assertEqual(curve[-1]["events"], 1)
        self.assertEqual(staged.characteristic_times(curve)["median_steps"], 3.0)
        self.assertEqual(staged.characteristic_times(curve)["median_time"], 0.03)
        self.assertEqual(curve[-1]["time"], 0.03)

    def test_malformed_event_and_network_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            bad_event = root / "bad.events"; bad_event.write_text("1 C 2 1 1 1\n")
            bad_network = root / "bad.network"; bad_network.write_text("2 1 1 1\n")
            with self.assertRaises(ValueError): staged.read_events(bad_event)
            with self.assertRaises(ValueError): staged.read_network(bad_network)

    def test_launcher_requires_explicit_authorization(self):
        launcher = HERE / "run_r1c2_staged_linux.sh"
        environment = os.environ.copy()
        environment.pop("AUTHORIZE_R1C2", None)
        environment["LMP"] = "/bin/true"
        result = subprocess.run([str(launcher)], cwd=HERE, env=environment,
                                text=True, capture_output=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("AUTHORIZE_R1C2", result.stderr)

    def test_stage_input_variables_are_supplied(self):
        input_text = (HERE / "in.r1c2_stage.lmp").read_text()
        variables = set(re.findall(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", input_text))
        internally_defined = {"rwca", "press", "pxx", "pyy", "pzz", "pxy", "pxz", "pyz",
                              "nxy", "nxz", "nyz", "st", "temp_now", "ep", "star_chunks",
                              "star_ids", "star_com"}
        launcher_text = (HERE / "run_r1c2_staged_linux.sh").read_text()
        supplied = set(re.findall(r"-var\s+([A-Za-z_][A-Za-z0-9_]*)", launcher_text))
        missing = variables - internally_defined - supplied
        self.assertEqual(missing, set())
        self.assertIn('-var FINAL_RESTART "$prefix.restart"', launcher_text)


if __name__ == "__main__":
    unittest.main()
