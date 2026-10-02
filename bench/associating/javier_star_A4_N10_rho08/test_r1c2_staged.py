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
    def test_duration_grid_reaches_available_prefix_and_future_stages(self):
        self.assertEqual(staged.duration_grid(30000),
                         [5000.0, 10000.0, 15000.0, 20000.0, 25000.0, 30000.0])
        self.assertEqual(staged.duration_grid(50000)[-3:], [40000.0, 45000.0, 50000.0])
        self.assertEqual(staged.duration_grid(100000)[-3:], [90000.0, 95000.0, 100000.0])

    def test_longer_prefix_can_remove_apparent_cutoff_convergence(self):
        rows = []
        for duration, eta in ((5000.0, 10.0), (10000.0, 10.2), (20000.0, 20.0)):
            rows.append({"source": "nested", "duration": duration, "cutoff": 500.0,
                         "eta_cutoff": eta, "safe_lag_fraction": 0.1})
        convergence = staged.nonassoc.fixed_cutoff_duration_convergence(rows)
        self.assertIsNone(convergence[0]["T_min_25pct"])

    def test_nonfickian_motion_cannot_stop_stage(self):
        decision = staged.adaptive_stage_decision(
            {"eta0_status": "supported descriptively"},
            {"D_candidate_stable": True, "strict_exponent_criterion_passed": False},
            {"one_over_e_time": 2844.0})
        self.assertEqual(decision["recommendation"], "CONTINUE")
        self.assertIn("asymptotic Fickian", decision["reasons"][0])

    def test_extension_plan_is_explicitly_heuristic(self):
        plan = staged.extension_planning(
            {"duration_convergence": [{"cutoff": 2000.0,
                                        "supported_by_block_criteria": True,
                                        "T_min_25pct": None}]},
            {"com_duration": 40000.0}, 50000.0)
        self.assertTrue(plan["heuristic_is_not_convergence"])
        self.assertEqual(plan["rheology_cutoffs"][0]["planned_T_over_cutoff"], 50.0)
        self.assertEqual(plan["com"]["planned_observed_duration"], 90000.0)

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

    def test_four_stage_concatenation_and_event_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            paths = []
            for index, steps in enumerate(([0, 1], [1, 2], [2, 3], [3, 4]), 1):
                path = root / f"stage{index}.raw"
                path.write_text("# step Pxx Pyy Pzz Pxy Pxz Pyz\n" +
                                "\n".join(f"{step} 1 1 1 0 0 0" for step in steps) + "\n")
                paths.append(path)
            combined = staged.concatenate_raw(paths)
            self.assertEqual(int(combined[-1, 0]), 4)
            initial = {(1, 2)}
            events = [staged.Event(2, "B", 1, 2, 1, 1),
                      staged.Event(3, "C", 1, 3, 1, 2)]
            self.assertEqual(staged.replay_network(initial, events), {(1, 3)})

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

    def test_stage4_defaults_gate_and_provenance_contract(self):
        launcher = (HERE / "run_r1c2_staged_linux.sh").read_text()
        self.assertIn('4) STAGE_STEPS=${STAGE_STEPS:-5000000}', launcher)
        self.assertIn('DEFAULT_RESTART="$OUT/stage3/production.restart"', launcher)
        self.assertIn('previous_complete="$OUT/stage$previous/complete"', launcher)
        self.assertIn('input_restart_sha256=', launcher)
        self.assertIn('output_restart_sha256=', launcher)
        self.assertIn('initial_timestep=', launcher)
        self.assertIn('final_timestep=', launcher)
        self.assertIn('event_path=$event_log', launcher)

    def test_stage4_refuses_without_stage3_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            environment = os.environ.copy()
            environment.update({"AUTHORIZE_R1C2": "YES", "LMP": "/bin/true",
                                "OUT": directory, "STAGE": "4"})
            result = subprocess.run([str(HERE / "run_r1c2_staged_linux.sh")],
                                    cwd=HERE, env=environment, text=True,
                                    capture_output=True)
        self.assertEqual(result.returncode, 3)
        self.assertIn("Stage 3 is not complete", result.stderr)

    def test_stage4_uses_stage3_as_only_resume_source_and_no_initial_network(self):
        launcher = (HERE / "run_r1c2_staged_linux.sh").read_text()
        self.assertIn('[[ "$input_restart_path" == "$expected_restart" ]]', launcher)
        self.assertIn('output_record="$OUT/stage$previous/output_restart_provenance.txt"', launcher)
        self.assertIn('4) STAGE_STEPS=${STAGE_STEPS:-5000000}; DEFAULT_RESTART="$OUT/stage3/production.restart"; WRITE_INITIAL=0', launcher)

    def test_legacy_restart_provenance_backfill_is_immutable_and_verified(self):
        script = HERE / "backfill_r1c2_restart_provenance.sh"
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory); stage = root / "stage3"; stage.mkdir()
            complete = stage / "complete"; complete.write_text("git_sha=old stage=3 stage_steps=2\n")
            restart = stage / "production.restart"; restart.write_bytes(b"restart")
            env = os.environ.copy(); env["OUT"] = str(root)
            first = subprocess.run([str(script), "3"], env=env, text=True, capture_output=True)
            self.assertEqual(first.returncode, 0, first.stderr)
            record = stage / "output_restart_provenance.txt"
            self.assertTrue(record.exists())
            original_complete = complete.read_text()
            first_record = record.read_text()
            second = subprocess.run([str(script), "3"], env=env, text=True, capture_output=True)
            self.assertEqual(second.returncode, 0, second.stderr)
            self.assertEqual(complete.read_text(), original_complete)
            self.assertEqual(record.read_text(), first_record)
            self.assertIn("output_restart_sha256=", first_record)

    def test_stage4_restart_gate_accepts_correct_and_rejects_modified_or_different_restart(self):
        launcher = HERE / "run_r1c2_staged_linux.sh"
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory); stage3 = root / "stage3"; stage3.mkdir()
            restart = stage3 / "production.restart"; restart.write_bytes(b"stage3 restart")
            stage2 = root / "stage2"; stage2.mkdir()
            stage2_restart = stage2 / "production.restart"; stage2_restart.write_bytes(b"stage2 restart")
            (stage3 / "production.raw").write_text("5000000 1 1 1 0 0 0\n")
            digest = subprocess.check_output(["sha256sum", str(restart)], text=True).split()[0]
            stage2_digest = subprocess.check_output(["sha256sum", str(stage2_restart)], text=True).split()[0]
            (stage3 / "complete").write_text(
                f"stage=3 input_restart_path={stage2_restart.resolve()} input_restart_sha256={stage2_digest} "
                f"output_restart_path={restart.resolve()} output_restart_sha256={digest}\n")
            fake_bin = root / "bin"; fake_bin.mkdir()
            fake_mpirun = fake_bin / "mpirun"
            fake_mpirun.write_text("""#!/bin/sh
while [ $# -gt 0 ]; do
  if [ "$1" = -var ]; then key=$2; value=$3; eval "$key=\\\"$value\\\""; shift 3
  else shift
  fi
done
printf '5000000 1 1 1 0 0 0\\n' > "$OUT_PREFIX.raw"
printf 'x\\n' > "$OUT_PREFIX.gt"
printf 'x\\n' > "$OUT_PREFIX.com"
printf 'x\\n' > "$EVENT_LOG"
printf '1 2\\n' > "$FINAL_NETWORK"
printf 'stage4 restart\\n' > "$FINAL_RESTART"
""")
            fake_mpirun.chmod(0o755)
            env = os.environ.copy(); env.update({"AUTHORIZE_R1C2": "YES", "LMP": "/bin/true",
                                                   "OUT": str(root), "STAGE": "4",
                                                   "PATH": f"{fake_bin}:{env['PATH']}"})
            accepted = subprocess.run([str(launcher)], cwd=HERE, env=env, text=True, capture_output=True)
            self.assertEqual(accepted.returncode, 0, accepted.stderr)
            self.assertIn("output_restart_sha256=", (root / "stage4" / "complete").read_text())
            (root / "stage4").rename(root / "stage4_saved")
            restart.write_bytes(b"modified")
            rejected_hash = subprocess.run([str(launcher)], cwd=HERE, env=env, text=True, capture_output=True)
            self.assertEqual(rejected_hash.returncode, 3)
            self.assertIn("does not match Stage 3 output provenance", rejected_hash.stderr)
            other = root / "other.restart"; other.write_bytes(b"stage3 restart")
            restart.write_bytes(b"stage3 restart")
            env["RESTART"] = str(other)
            rejected_path = subprocess.run([str(launcher)], cwd=HERE, env=env, text=True, capture_output=True)
            self.assertEqual(rejected_path.returncode, 3)
            self.assertIn("requires the previous stage output restart", rejected_path.stderr)

    def test_stage4_input_preserves_snapshot_boundary_without_new_history_origin(self):
        input_text = (HERE / "in.r1c2_stage.lmp").read_text()
        self.assertIn('write_associating_network ${FINAL_NETWORK} fix kinetics', input_text)
        self.assertIn('if "${WRITE_INITIAL} == 1" then "write_associating_network ${INITIAL_NETWORK} fix kinetics"', input_text)

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
