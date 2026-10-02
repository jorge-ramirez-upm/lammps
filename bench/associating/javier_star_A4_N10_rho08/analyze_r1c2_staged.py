#!/usr/bin/env python3
"""Offline R1-C2 stage concatenation, replay, and bond-survival analysis."""
import argparse
import csv
import json
import importlib.util
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("nonassoc", HERE / "analyze_nonassoc_control.py")
nonassoc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nonassoc)
spec_rheology = importlib.util.spec_from_file_location("r1c1", HERE / "analyze_r1c1_full_rheology.py")
r1c1 = importlib.util.module_from_spec(spec_rheology)
spec_rheology.loader.exec_module(r1c1)


def edge(first, second):
    return (first, second) if first < second else (second, first)


@dataclass(frozen=True)
class Event:
    timestep: int
    kind: str
    sticker_i: int
    sticker_j: int
    molecule_i: int
    molecule_j: int

    @property
    def pair(self):
        return edge(self.sticker_i, self.sticker_j)


def read_events(path):
    events = []
    with open(path) as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.split()
            if len(fields) != 6 or fields[1] not in ("C", "B"):
                raise ValueError(f"malformed event at {path}:{line_number}")
            try:
                values = [int(fields[index]) for index in (0, 2, 3, 4, 5)]
            except ValueError as exc:
                raise ValueError(f"malformed event values at {path}:{line_number}") from exc
            if values[1] >= values[2]:
                raise ValueError(f"event stickers are not canonical at {path}:{line_number}")
            events.append(Event(values[0], fields[1], values[1], values[2], values[3], values[4]))
    if any(events[index].timestep > events[index + 1].timestep for index in range(len(events) - 1)):
        raise ValueError(f"event timestamps are not chronological: {path}")
    return events


def read_network(path):
    active = set()
    with open(path) as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.split()
            if len(fields) < 2:
                raise ValueError(f"malformed network row at {path}:{line_number}")
            try:
                first, second = int(fields[0]), int(fields[1])
            except ValueError as exc:
                raise ValueError(f"malformed network row at {path}:{line_number}") from exc
            if first >= second:
                raise ValueError(f"network edge is not canonical at {path}:{line_number}")
            pair = (first, second)
            if pair in active:
                raise ValueError(f"duplicate network edge at {path}:{line_number}")
            active.add(pair)
    return active


def replay_network(initial, events):
    active = set(initial)
    for event in events:
        if event.kind == "C":
            if event.pair in active:
                raise ValueError(f"creation of active edge at timestep {event.timestep}: {event.pair}")
            active.add(event.pair)
        else:
            if event.pair not in active:
                raise ValueError(f"break of inactive edge at timestep {event.timestep}: {event.pair}")
            active.remove(event.pair)
    return active


def concatenate_raw(paths):
    """Concatenate stepwise raw stress files, removing only exact boundaries."""
    arrays = []
    previous_last = None
    for path in paths:
        data = np.loadtxt(path, comments="#", ndmin=2)
        if data.size == 0:
            continue
        if data.shape[1] < 7:
            raise ValueError(f"raw stress file needs seven columns: {path}")
        steps = data[:, 0].astype(np.int64)
        if np.any(np.diff(steps) <= 0):
            raise ValueError(f"raw stress timestamps are not strictly increasing: {path}")
        if previous_last is not None:
            if steps[0] == previous_last:
                data = data[1:]
                steps = steps[1:]
            elif steps[0] != previous_last + 1:
                raise ValueError(f"gap or inconsistent overlap before {path}: {previous_last} -> {steps[0]}")
        if len(data):
            arrays.append(data[:, :7]); previous_last = int(data[-1, 0])
    return np.vstack(arrays) if arrays else np.empty((0, 7))


def concatenate_com(paths):
    """Concatenate staged COM files with the same boundary rules as stress."""
    frames = []
    previous_step = None
    cadence = None
    molecules = None
    for path in paths:
        current = list(nonassoc.com_frames(path, expected_stars=0))
        for step, coordinates, ids in current:
            if molecules is None:
                molecules = ids
            elif ids != molecules:
                raise ValueError(f"COM molecule mapping changed in {path}")
            if previous_step is not None:
                if step == previous_step:
                    if frames[-1][1].shape != coordinates.shape or not np.array_equal(frames[-1][1], coordinates):
                        raise ValueError(f"inconsistent duplicated COM boundary at {path}")
                    continue
                if cadence is None:
                    cadence = step - previous_step
                if step != previous_step + cadence:
                    raise ValueError(f"COM gap or overlap before {path}: {previous_step} -> {step}")
            frames.append((step, coordinates, ids)); previous_step = step
    return frames


def _record(start, end, left, right, termination):
    return {"start": start, "end": end, "duration": (end - start) if start is not None and end is not None else None,
            "left_censored": bool(left), "right_censored": bool(right),
            "termination": termination, "observed": not right}


def lifetime_records(initial, events, end_timestep=None):
    """Return bare and renormalized records with explicit censoring flags."""
    if end_timestep is None:
        end_timestep = events[-1].timestep if events else 0
    # Bare episodes and renormalized episodes deliberately have separate
    # state.  A same-partner reattachment merges only the renormalized
    # episode; every accepted creation starts a new bare episode.
    bare_active = {pair: (None, True) for pair in initial}
    renorm_active = {pair: (None, True) for pair in initial}
    bare = []
    pending = {}
    renormalized = []
    for event in events:
        pair = event.pair
        if event.kind == "B":
            if pair not in bare_active or pair not in renorm_active:
                raise ValueError(f"break of inactive edge at timestep {event.timestep}: {pair}")
            start, left = bare_active.pop(pair)
            bare.append(_record(start, event.timestep, left, False, "break"))
            renorm_start, renorm_left = renorm_active.pop(pair)
            pending[pair] = (renorm_start, renorm_left, event.timestep)
            continue
        # A third-partner creation closes any unresolved same-partner flicker.
        for pending_pair in list(pending):
            if event.pair != pending_pair and (event.sticker_i in pending_pair or event.sticker_j in pending_pair):
                start, left, break_time = pending.pop(pending_pair)
                renormalized.append(_record(start, break_time, left, False, "third_partner"))
        if pair in pending:
            start, left, _ = pending.pop(pair)
            renorm_active[pair] = (start, left)
            bare_active[pair] = (event.timestep, False)
        else:
            if pair in bare_active or pair in renorm_active:
                raise ValueError(f"creation of active edge at timestep {event.timestep}: {pair}")
            bare_active[pair] = (event.timestep, False)
            renorm_active[pair] = (event.timestep, False)
    for pair, (start, left) in bare_active.items():
        bare.append(_record(start, end_timestep, left, True, "right_censored"))
    for pair, (start, left) in renorm_active.items():
        renormalized.append(_record(start, end_timestep, left, True, "right_censored"))
    for start, left, _ in pending.values():
        renormalized.append(_record(start, end_timestep, left, True, "pending_reattachment"))
    return {"bare": bare, "renormalized": renormalized}


def kaplan_meier(records, dt=0.01):
    """KM curve for records with known durations; left-censored records are excluded."""
    usable = [row for row in records if row["duration"] is not None and not row["left_censored"]]
    times = sorted({row["duration"] for row in usable})
    survival = 1.0; output = []
    for time in times:
        at_risk = sum(row["duration"] >= time for row in usable)
        events = sum(row["duration"] == time and not row["right_censored"] for row in usable)
        censored = sum(row["duration"] == time and row["right_censored"] for row in usable)
        if at_risk:
            survival *= 1.0 - events / at_risk
        output.append({"time_steps": float(time), "time": float(time * dt),
                       "survival": float(survival), "at_risk": int(at_risk),
                       "events": int(events), "censored": int(censored)})
    return output


def characteristic_times(curve):
    result = {}
    for level, name in ((0.5, "median"), (1 / np.e, "one_over_e")):
        crossing = next((row for row in curve if row["survival"] <= level), None)
        result[f"{name}_steps"] = crossing["time_steps"] if crossing else None
        result[f"{name}_time"] = crossing["time"] if crossing else None
        result[f"{name}_at_risk"] = crossing["at_risk"] if crossing else None
    return result


def lifetime_summary(records, curve, dt=0.01):
    total = len(records)
    observed = sum(not row["right_censored"] for row in records)
    left = sum(row["left_censored"] for row in records)
    right = sum(row["right_censored"] for row in records)
    resolved = [row["duration"] for row in records
                if row["duration"] is not None and not row["left_censored"] and not row["right_censored"]]
    maximum = max(resolved) if resolved else None
    return {"total_episodes": total, "observed_terminations": observed,
            "left_censored_count": left, "left_censored_fraction": left / total if total else None,
            "right_censored_count": right, "right_censored_fraction": right / total if total else None,
            "median_steps": characteristic_times(curve)["median_steps"],
            "median_time": characteristic_times(curve)["median_time"],
            "median_at_risk": characteristic_times(curve)["median_at_risk"],
            "one_over_e_steps": characteristic_times(curve)["one_over_e_steps"],
            "one_over_e_time": characteristic_times(curve)["one_over_e_time"],
            "one_over_e_at_risk": characteristic_times(curve)["one_over_e_at_risk"],
            "maximum_resolved_steps": maximum,
            "maximum_resolved_time": maximum * dt if maximum is not None else None}


def write_csv(path, fields, rows):
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def duration_grid(available_duration, spacing=5000.0):
    """Return cumulative prefixes through the longest available trajectory."""
    if available_duration < 0 or spacing <= 0:
        raise ValueError("duration and spacing must be positive")
    count = int(np.floor((available_duration + 1e-9) / spacing))
    durations = [float(spacing * index) for index in range(1, count + 1)]
    if not durations or available_duration - durations[-1] > 1e-8:
        durations.append(float(available_duration))
    return durations


def adaptive_stage_decision(rheology, diffusion, renormalized_summary):
    """Diagnostic continuation gate; never launches a later stage."""
    reasons = []
    if rheology.get("eta0_status") != "supported descriptively":
        reasons.append("zero-shear viscosity plateau is not established")
    if not diffusion.get("strict_exponent_criterion_passed", False):
        reasons.append("asymptotic Fickian COM diffusion is not established")
    if renormalized_summary.get("one_over_e_time") is None:
        reasons.append("renormalized survival has no resolved 1/e crossing")
    return {"recommendation": "CONTINUE" if reasons else "STOP", "reasons": reasons,
            "diagnostic_only": True, "next_stage_not_launched": True}


def extension_planning(rheology, diffusion, current_total, target_total=100000.0):
    """Plan only; T/t_c~50 is a heuristic, not a convergence claim."""
    cutoffs = []
    for row in rheology.get("duration_convergence", []):
        cutoff = row["cutoff"]
        cutoffs.append({"cutoff": cutoff,
                        "current_T_over_cutoff": current_total / cutoff,
                        "planned_T_over_cutoff": target_total / cutoff,
                        "heuristic_T_over_cutoff_50_passes": target_total / cutoff >= 50,
                        "currently_supported": row.get("supported_by_block_criteria", False),
                        "currently_resolved": row.get("T_min_25pct") is not None and
                                              row.get("supported_by_block_criteria", False)})
    current_com = diffusion.get("com_duration", current_total - 10000.0)
    return {"status": "planning_only", "current_total_T": current_total,
            "planned_total_T": target_total, "heuristic": "T/t_c approximately 50",
            "heuristic_is_not_convergence": True,
            "rheology_cutoffs": cutoffs,
            "com": {"current_observed_duration": current_com,
                    "planned_observed_duration": target_total - (current_total - current_com),
                    "material_resolution_gain": "likely for later alpha windows and unresolved long-time dynamics"}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--r1c1-raw", required=True)
    parser.add_argument("--stage-dir", action="append", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--volume", type=float, default=43563.0 / 0.85)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--dt", type=float, default=0.01)
    args = parser.parse_args()
    stages = [Path(path) for path in args.stage_dir]
    raw_paths = [Path(args.r1c1_raw)] + [stage / "production.raw" for stage in stages]
    raw = concatenate_raw(raw_paths)
    com_frames = concatenate_com([stage / "production.com" for stage in stages])
    initial = read_network(stages[0] / "initial.network")
    events = []
    active = set(initial)
    for stage in stages:
        stage_events = read_events(stage / "events.dat")
        active = replay_network(active, stage_events)
        final_network = stage / "final.network"
        if final_network.exists() and active != read_network(final_network):
            raise ValueError(f"network replay mismatch at {stage}")
        events.extend(stage_events)
    if any(events[index].timestep > events[index + 1].timestep for index in range(len(events) - 1)):
        raise ValueError("stage event streams are not chronological")
    final = replay_network(initial, events)
    records = lifetime_records(initial, events, int(raw[-1, 0]) if len(raw) else None)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    np.savetxt(out / "r1c2_combined.raw", raw, header="step Pxx Pyy Pzz Pxy Pxz Pyz", comments="")
    stress_summary = {}
    rheology = {}
    if len(raw):
        stress_time, cs, cn, difference, modulus, eta = r1c1._modulus(
            raw, args.volume, args.temperature, args.dt)
        available_duration = float(stress_time[-1])
        durations = duration_grid(available_duration)
        cutoffs = [1, 2, 5, 10, 20, 50, 100, 200, 300, 500, 750, 1000, 1500, 2000, 3000, 5000]
        (duration_summary, duration_rows, cutoff_rows, block_rows, block_summary,
            uncertainty, tail_bins, terminal, duration_convergence) = nonassoc._rheology_rows(
            raw, durations, cutoffs, (4, 5, 10), args.volume, args.temperature, args.dt, 5000.0)
        cutoff_stability = nonassoc.fixed_cutoff_stability(cutoff_rows)
        plateau = nonassoc.cutoff_plateau(duration_convergence)
        stress_summary = {"G0": float(modulus[0]),
                          "R_iso_0": float(cn[0] / cs[0]) if cs[0] else None,
                          "fixed_cutoff_eta_full_trajectory": {str(cutoff): float(eta[min(int(round(cutoff / args.dt)), len(eta) - 1)])
                                                                for cutoff in cutoffs if cutoff <= stress_time[-1]},
                          "definition": "offline modulus from concatenated raw stress; fixed-cutoff values are descriptive and do not by themselves establish eta_0"}
        rheology = {"duration_summary": duration_summary, "duration_convergence": duration_convergence,
                    "cutoff_stability": cutoff_stability, "cutoff_plateau": plateau,
                    "slow_tail": terminal,
                    "eta0_status": plateau["eta0_status"]}
        write_csv(out / "r1c2_duration_convergence.csv", list(duration_convergence[0]) if duration_convergence else ["cutoff"], duration_convergence)
        write_csv(out / "r1c2_fixed_cutoff.csv",
                  ["source", "duration", "cutoff", "blocks", "eta_cutoff", "eta_sd", "eta_sem", "safe_lag_fraction"], cutoff_rows)
        write_csv(out / "r1c2_slow_tail_bins.csv",
                  ["blocks", "lag_start", "lag_end", "lag_center", "G_mean", "G_sd", "G_sem", "n_samples", "n_lag_points"], tail_bins)
    else:
        rheology = {"eta0_status": "not established", "reason": "empty concatenated stress trajectory"}
    if com_frames:
        com = np.stack([frame[1] for frame in com_frames])
        com_time, com_msd = nonassoc.msd_from_com(com, (com_frames[1][0] - com_frames[0][0]) * args.dt)
        alpha, diffusion = nonassoc.diffusion_diagnostics(com_time, com_msd)
        diffusion["D_nonassoc_reference"] = 1.77e-3
        diffusion["D_vs_nonassoc_ratio"] = diffusion["D"] / 1.77e-3 if diffusion["D"] is not None else None
        diffusion["com_duration"] = float(com_time[-1])
        diffusion["coefficient_status"] = ("asymptotic_candidate" if diffusion.get("strict_exponent_criterion_passed")
                                            else "effective_candidate_slope")
        diffusion["wording"] = ("candidate effective diffusion slope; linear-fit stability does not establish a long-time diffusion coefficient while the strict Fickian criterion fails"
                                 if not diffusion.get("strict_exponent_criterion_passed") else
                                 "diffusion fit is supported by the strict sustained Fickian diagnostic, without claiming a perfect alpha=1 plateau")
        write_csv(out / "r1c2_com_msd.csv", ["time", "g_CM", "alpha"],
                  (dict(time=float(t), g_CM=float(g), alpha=float(a)) for t, g, a in zip(com_time, com_msd, alpha)))
    else:
        diffusion = {"D": None, "D_candidate_stable": False, "strict_exponent_criterion_passed": False,
                     "asymptotic_fickian_confirmed": False, "reason": "no COM frames"}
    for name in ("bare", "renormalized"):
        curve = kaplan_meier(records[name], args.dt)
        write_csv(out / f"r1c2_{name}_survival.csv",
                  ["time_steps", "time", "survival", "at_risk", "events", "censored"], curve)
    bare_summary = lifetime_summary(records["bare"], kaplan_meier(records["bare"], args.dt), args.dt)
    renormalized_summary = lifetime_summary(records["renormalized"], kaplan_meier(records["renormalized"], args.dt), args.dt)
    decision = adaptive_stage_decision(rheology, diffusion, renormalized_summary)
    planning = extension_planning(rheology, diffusion,
                                  float(raw[-1, 0] * args.dt) if len(raw) else 0.0)
    summary = {"raw_rows": int(len(raw)), "raw_start": int(raw[0, 0]) if len(raw) else None,
               "raw_end": int(raw[-1, 0]) if len(raw) else None,
               "stress": stress_summary,
               "com_frames": len(com_frames),
               "com_start": int(com_frames[0][0]) if com_frames else None,
               "com_end": int(com_frames[-1][0]) if com_frames else None,
               "stage_count": len(stages), "event_count": len(events),
               "initial_edges": len(initial), "final_edges": len(final),
               "final_network_replay_valid": True,
               "left_censored": {name: sum(row["left_censored"] for row in records[name]) for name in records},
               "right_censored": {name: sum(row["right_censored"] for row in records[name]) for name in records},
               "rheology": rheology,
               "diffusion": diffusion,
               "bare_lifetime": bare_summary,
               "renormalized_lifetime": renormalized_summary,
               "sticker_lifetime_freeze": {"status": "frozen_for_current_observation",
                                            "bare": bare_summary,
                                            "renormalized": renormalized_summary,
                                            "unresolved_quantities": ["rheology", "COM diffusion"]},
               "bare_characteristic_times": characteristic_times(kaplan_meier(records["bare"], args.dt)),
               "renormalized_characteristic_times": characteristic_times(kaplan_meier(records["renormalized"], args.dt)),
               "event_lifetime_censoring": "R1-C2 event-derived lifetimes are left-censored at the R1-C2 start; active observations at the final stage are right-censored.",
               "renormalized_definition": "same-partner detach/reattach is merged; a third-partner creation terminates the pending renormalized episode at the break time.",
               "network_replay": "initial network plus sequential C/B events; exact final-state replay required",
               "stage_decision": decision,
               "extension_to_T100000_planning": planning}
    (out / "r1c2_staged.summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
