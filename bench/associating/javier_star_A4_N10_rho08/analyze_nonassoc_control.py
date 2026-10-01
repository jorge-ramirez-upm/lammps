#!/usr/bin/env python3
"""Analyze nonassociating control stress and star-COM trajectories."""
import argparse
import csv
import gzip
import importlib.util
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("r1c1", HERE / "analyze_r1c1_full_rheology.py")
r1c1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r1c1)


def _rheology_rows(raw, durations, cutoffs, blocks, volume, temperature, dt, max_lag):
    duration_rows, cutoff_rows, duration_summary = [], [], []
    for requested in durations:
        count = min(len(raw), int(round(requested / dt)) + 1)
        if count < 2:
            continue
        time, cs, cn, difference, modulus, eta = r1c1._modulus(
            raw[:count], volume, temperature, dt)
        normalized = modulus / modulus[0] if modulus[0] else np.full_like(modulus, np.nan)
        duration = float(time[-1])
        for lag in np.arange(0.0, min(max_lag, duration) + dt / 2, dt):
            index = min(int(round(lag / dt)), len(modulus) - 1)
            duration_rows.append({"duration": duration, "lag": float(lag),
                                  "G0": float(modulus[0]),
                                  "R_iso_0": float(cn[0] / cs[0]) if cs[0] else None,
                                  "G": float(modulus[index]),
                                  "G_over_G0": float(normalized[index]),
                                  "eta": float(eta[index])})
        for cutoff in cutoffs:
            if cutoff <= duration:
                index = min(int(round(cutoff / dt)), len(eta) - 1)
                cutoff_rows.append({"duration": duration, "cutoff": cutoff,
                                    "eta_cutoff": float(eta[index]),
                                    "safe_lag_fraction": cutoff / duration})
        duration_summary.append({"duration": duration, "G0": float(modulus[0]),
                                 "R_iso_0": float(cn[0] / cs[0]) if cs[0] else None,
                                 "G_over_G0_crossing_0.1": r1c1.first_crossing(time, normalized, .1),
                                 "G_over_G0_crossing_0.01": r1c1.first_crossing(time, normalized, .01)})

    block_rows = []
    # Use the full common lag grid for the block-resolved tail diagnostic.
    lags = tuple(np.arange(0.0, max_lag + dt / 2.0, dt))
    block_cutoff_rows = []
    for nblocks in blocks:
        for block, values in enumerate(np.array_split(raw, nblocks), 1):
            if len(values) < 2:
                continue
            time, _, _, _, modulus, eta = r1c1._modulus(values, volume, temperature, dt)
            for lag in lags:
                if lag <= time[-1]:
                    index = int(round(lag / dt))
                    block_rows.append({"blocks": nblocks, "block": block, "lag": lag,
                                       "block_duration": float(time[-1]), "G": float(modulus[index]),
                                       "eta": float(eta[index])})
    block_summary = []
    for nblocks in blocks:
        for lag in lags:
            rows = [row for row in block_rows if row["blocks"] == nblocks and row["lag"] == lag]
            values = np.array([row["G"] for row in rows])
            etas = np.array([row["eta"] for row in rows])
            if not len(values):
                continue
            mean = float(values.mean())
            sd = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            block_summary.append({"blocks": nblocks, "lag": lag, "G_mean": mean,
                                  "G_sd": sd, "G_sem": sd / math.sqrt(len(values)),
                                  "G_sd_over_abs_mean": sd / abs(mean) if mean else None,
                                  "eta_mean": float(etas.mean()),
                                  "eta_sd": float(etas.std(ddof=1)) if len(etas) > 1 else 0.0,
                                  "eta_sem": float(etas.std(ddof=1) / math.sqrt(len(etas))) if len(etas) > 1 else 0.0})
    # Nested-duration values are retained, while these rows add block SEMs for
    # every requested fixed cutoff.  eta_final at the full trajectory length
    # is deliberately never used as a convergence diagnostic.
    for nblocks in blocks:
        rows = [row for row in block_rows if row["blocks"] == nblocks]
        for cutoff in cutoffs:
            values = [row for row in rows if abs(row["lag"] - cutoff) <= dt / 2]
            if not values:
                continue
            etas = np.asarray([row["eta"] for row in values])
            mean = float(etas.mean())
            sd = float(etas.std(ddof=1)) if len(etas) > 1 else 0.0
            block_cutoff_rows.append({"source": "blocks", "duration": float(values[0]["block_duration"]),
                                      "cutoff": float(cutoff), "blocks": nblocks,
                                      "eta_cutoff": mean, "eta_sd": sd,
                                      "eta_sem": sd / math.sqrt(len(etas)) if len(etas) > 1 else 0.0,
                                      "safe_lag_fraction": float(cutoff / values[0]["block_duration"])})
    # Put the duration rows after the block rows with a common schema.
    cutoff_rows = [{"source": "nested", "duration": row["duration"], "cutoff": row["cutoff"],
                    "blocks": None, "eta_cutoff": row["eta_cutoff"], "eta_sd": None,
                    "eta_sem": None, "safe_lag_fraction": row["safe_lag_fraction"]}
                   for row in cutoff_rows] + block_cutoff_rows
    uncertainty = {str(n): next((row["lag"] for row in block_summary
                                 if row["blocks"] == n and row["G_sd_over_abs_mean"] is not None
                                 and row["G_sd_over_abs_mean"] >= 1.0), None) for n in blocks}
    terminal = terminal_diagnostic(block_summary, block_rows, blocks)
    return duration_summary, duration_rows, cutoff_rows, block_rows, block_summary, uncertainty, terminal


def _first_lag(rows, predicate):
    for row in rows:
        if predicate(row):
            return float(row["lag"])
    return None


def terminal_diagnostic(block_summary, block_rows, blocks):
    """Describe the slow stress tail without asserting a single terminal time."""
    preferred = 5 if 5 in blocks else (max(blocks) if blocks else None)
    rows = sorted((row for row in block_summary if row["blocks"] == preferred), key=lambda row: row["lag"])
    if not rows:
        return {"status": "unresolved", "block_scheme": preferred}
    below_sem = _first_lag(rows[1:], lambda row: row["G_mean"] <= row["G_sem"])
    below_sd = _first_lag(rows[1:], lambda row: row["G_mean"] <= row["G_sd"])
    positive_runs = []
    run = []
    for row in rows:
        if row["G_mean"] > 0:
            run.append(row)
        elif run:
            positive_runs.append(run); run = []
    if run:
        positive_runs.append(run)
    positive_run = max(positive_runs, key=len) if positive_runs else []
    positive_end = positive_run[-1]["lag"] if positive_run else None
    positive_start = positive_run[0]["lag"] if positive_run else None
    # Truncate the integral at the first loss of block-SEM resolution; later
    # noisy re-crossings do not extend the claimed resolved window.
    resolved_end = below_sem if below_sem is not None else rows[-1]["lag"]
    integral = None
    if resolved_end is not None and resolved_end > 0:
        eta_rows = [row for row in block_rows if row["blocks"] == preferred and
                    row["lag"] <= resolved_end + 1e-12]
        if eta_rows:
            by_block = {}
            for row in eta_rows:
                by_block[row["block"]] = row
            eta = np.asarray([row["eta"] for row in by_block.values()])
            mean = float(eta.mean()); sd = float(eta.std(ddof=1)) if len(eta) > 1 else 0.0
            g0 = rows[0]["G_mean"]
            if g0 > 0:
                integral = {"cutoff": float(resolved_end), "tau_int": mean / g0,
                            "tau_int_sem": sd / math.sqrt(len(eta)) / g0 if len(eta) > 1 else 0.0,
                            "definition": "block mean cumulative integral divided by G(0), truncated when block mean first falls below its SEM"}
    return {"status": "descriptive_only", "block_scheme": preferred,
            "first_G_below_block_SEM": below_sem,
            "first_G_below_block_SD": below_sd,
            "longest_sustained_positive_interval": {"start": positive_start, "end": positive_end,
                                                       "duration": (positive_end - positive_start) if positive_run else None},
            "integral_relaxation": integral,
            "definition": "No single tau_term is claimed; all ranges are block-resolved diagnostics."}


def fixed_cutoff_stability(cutoff_rows):
    """Compare nested-duration spread and block SEM at each fixed cutoff."""
    result = []
    cutoffs = sorted({row["cutoff"] for row in cutoff_rows})
    for cutoff in cutoffs:
        nested = np.asarray([row["eta_cutoff"] for row in cutoff_rows
                             if row["cutoff"] == cutoff and row["source"] == "nested"
                             and row["safe_lag_fraction"] <= 0.2])
        block = [row for row in cutoff_rows if row["cutoff"] == cutoff and row["source"] == "blocks"
                 and row["safe_lag_fraction"] <= 0.2]
        if not len(nested) or not block:
            continue
        nested_mean = float(nested.mean())
        nested_relative_span = float((nested.max() - nested.min()) / max(abs(nested_mean), 1e-30))
        block_relative_sem = float(max(row["eta_sem"] for row in block) /
                                   max(abs(np.mean([row["eta_cutoff"] for row in block])), 1e-30))
        result.append({"cutoff": float(cutoff), "nested_relative_span": nested_relative_span,
                       "block_relative_sem": block_relative_sem,
                       "stable_at_25_percent": bool(nested_relative_span <= 0.25 and block_relative_sem <= 0.25)})
    stable = [row["cutoff"] for row in result if row["stable_at_25_percent"]]
    return {"criteria": "only durations with cutoff/T <= 0.2 are compared; nested duration span and maximum block SEM each <= 25% of the cutoff integral",
            "cutoffs": result, "largest_stable_cutoff": max(stable) if stable else None}


def _open_dump(path):
    return gzip.open(path, "rt") if str(path).endswith(".gz") else open(path)


def com_frames(path, expected_stars=1000):
    """Yield (timestep, COM[star, xyz]) from fix ave/time vector output.

    Each frame is a timestep/row-count pair followed by rows containing the
    deterministic compressed-chunk row, original molecule ID, and unwrapped
    x/y/z.  The molecule ID is retained in the file rather than inferred from
    atom dumps.
    """
    with _open_dump(path) as handle:
        while True:
            line = handle.readline()
            if not line:
                return
            if line.startswith("#") or not line.strip():
                continue
            first = line.split()
            if len(first) != 2:
                raise ValueError("malformed COM frame header")
            try:
                step, count = int(first[0]), int(first[1])
            except ValueError as exc:
                raise ValueError("malformed COM timestep/row-count header") from exc
            if expected_stars and count != expected_stars:
                raise ValueError(f"expected {expected_stars} stars, got {count}")
            rows = []
            for expected_row in range(1, count + 1):
                fields = handle.readline().split()
                if len(fields) != 5:
                    raise ValueError("malformed COM row")
                try:
                    row, molecule = int(fields[0]), int(fields[1])
                    point = [float(value) for value in fields[2:5]]
                except ValueError as exc:
                    raise ValueError("malformed COM row values") from exc
                if row != expected_row:
                    raise ValueError("COM rows are not in deterministic order")
                rows.append((molecule, point))
            molecules = [molecule for molecule, _ in rows]
            if len(set(molecules)) != count:
                raise ValueError("duplicate molecule ID in COM frame")
            yield step, np.asarray([point for _, point in rows], dtype=float), molecules


def dump_frames(path, expected_atoms=41000):
    """Backward-compatible alias removed from the production path."""
    raise ValueError("atom trajectory parsing is disabled; use the direct COM output")


def msd_from_com(com, dt):
    com = np.asarray(com, dtype=float)
    if com.ndim != 3 or com.shape[2] != 3 or com.shape[0] < 2:
        raise ValueError("COM array must have shape (frames, stars, 3) with >=2 frames")
    nframes = com.shape[0]
    size = 1 << (2 * nframes - 1).bit_length()
    result = np.zeros(nframes)
    for star in range(com.shape[1]):
        for coordinate in range(3):
            values = com[:, star, coordinate]
            transformed = np.fft.rfft(values, size)
            products = np.fft.irfft(transformed * transformed.conj(), size)[:nframes]
            products /= np.arange(nframes, 0, -1)
            squares = values * values
            prefix = np.concatenate(([0.0], np.cumsum(squares)))
            lags = np.arange(nframes)
            first = (prefix[nframes - lags] - prefix[0]) / (nframes - lags)
            second = (prefix[nframes] - prefix[lags]) / (nframes - lags)
            result += first + second - 2.0 * products
    result /= com.shape[1]
    return np.arange(nframes, dtype=float) * dt, result


def _rolling_log_exponent(time, msd, window=9):
    """Local log-log slope from a centered rolling regression."""
    time = np.asarray(time, dtype=float); msd = np.asarray(msd, dtype=float)
    alpha = np.full_like(msd, np.nan)
    valid = (time > 0) & (msd > 0)
    half = max(2, int(window) // 2)
    for index in range(half, len(time) - half):
        sample = slice(index - half, index + half + 1)
        if np.all(valid[sample]):
            alpha[index] = np.polyfit(np.log(time[sample]), np.log(msd[sample]), 1)[0]
    return alpha


def diffusion_diagnostics(time, msd, max_fraction=0.2, exponent_tolerance=0.2,
                          sustained_points=7):
    valid = (time > 0) & (msd > 0)
    alpha = _rolling_log_exponent(time, msd)
    limit = float(time[-1] * max_fraction)
    candidates = (("T/20_to_T/10", 0.05, 0.10),
                  ("T/10_to_T/5", 0.10, 0.20),
                  ("T/20_to_T/5", 0.05, 0.20))
    fits = []
    for name, lo_fraction, hi_fraction in candidates:
        lo = time[-1] * lo_fraction; hi = min(time[-1] * hi_fraction, limit)
        selected = valid & (time >= lo) & (time <= hi)
        if np.count_nonzero(selected) < 4:
            continue
        slope, intercept = np.polyfit(time[selected], msd[selected], 1)
        fits.append({"window": name, "t_min": float(lo), "t_max": float(hi),
                     "D": float(slope / 6.0), "slope": float(slope),
                     "points": int(np.count_nonzero(selected))})
    diffusion = float(np.median([fit["D"] for fit in fits])) if fits else None
    d_values = np.asarray([fit["D"] for fit in fits], dtype=float)
    stable = bool(len(d_values) > 1 and np.ptp(d_values) / max(abs(np.mean(d_values)), 1e-30) <= 0.25)
    good = valid & (time <= limit) & np.isfinite(alpha) & (abs(alpha - 1.0) <= exponent_tolerance)
    onset = None; end = None
    run_start = None
    for index, is_good in enumerate(good):
        if is_good and run_start is None:
            run_start = index
        if (not is_good or index == len(good) - 1) and run_start is not None:
            run_end = index if is_good and index == len(good) - 1 else index - 1
            if run_end - run_start + 1 >= sustained_points:
                onset = float(time[run_start]); end = float(time[run_end]); break
            run_start = None
    return alpha, {"D": diffusion, "D_candidates": fits,
                   "D_candidate_stable": stable,
                   "D_fit_max_fraction_of_total_lag": max_fraction,
                   "fickian_onset": onset, "fickian_window_end": end,
                   "fickian_window_points": sustained_points,
                   "fickian_exponent_tolerance": exponent_tolerance,
                   "definition": "D is the median of candidate linear MSD fits restricted to T/20..T/5; onset is the first sustained rolling log-log exponent window consistent with 1."}


def write_csv(path, fields, rows):
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("raw")
    parser.add_argument("com_output")
    parser.add_argument("--out-dir", default=".")
    parser.add_argument("--volume", type=float, default=43563.0 / 0.85)
    parser.add_argument("--stars", type=int, default=1000)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--durations", default="100,250,500,1000,2500,5000")
    parser.add_argument("--cutoffs", default="1,2,5,10,20,50,100,200")
    parser.add_argument("--blocks", default="4,5,10")
    parser.add_argument("--max-lag", type=float, default=200.0)
    parser.add_argument("--assoc-correlations")
    args = parser.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    raw = r1c1.load_raw(args.raw)
    durations = [float(x) for x in args.durations.split(",") if x]
    cutoffs = [float(x) for x in args.cutoffs.split(",") if x]
    blocks = [int(x) for x in args.blocks.split(",") if x]
    duration_summary, duration_rows, cutoff_rows, block_rows, block_summary, uncertainty, terminal = _rheology_rows(
        raw, durations, cutoffs, blocks, args.volume, args.temperature, args.dt, args.max_lag)
    cutoff_stability = fixed_cutoff_stability(cutoff_rows)
    frame_data = list(com_frames(args.com_output, expected_stars=args.stars))
    if not frame_data:
        raise ValueError("no trajectory frames")
    steps = np.array([row[0] for row in frame_data])
    if len(set(np.diff(steps))) != 1:
        raise ValueError("trajectory timesteps are not uniformly spaced")
    com = np.stack([row[1] for row in frame_data])
    time, msd = msd_from_com(com, float(np.diff(steps)[0]) * args.dt)
    alpha, diffusion = diffusion_diagnostics(time, msd)
    stress_time, cs, cn, difference, modulus, eta = r1c1._modulus(
        raw, args.volume, args.temperature, args.dt)
    write_csv(out / "control_nonassoc.correlations.csv",
              ["time", "Cs", "Cn_over4", "D", "G", "eta", "useful"],
              (dict(time=float(t), Cs=float(s), Cn_over4=float(n), D=float(d),
                    G=float(g), eta=float(e), useful=1)
               for t, s, n, d, g, e in zip(stress_time, cs, cn, difference, modulus, eta)))
    write_csv(out / "control_nonassoc.duration_convergence.csv", list(duration_rows[0]), duration_rows)
    write_csv(out / "control_nonassoc.fixed_cutoff.csv",
              ["source", "duration", "cutoff", "blocks", "eta_cutoff", "eta_sd", "eta_sem", "safe_lag_fraction"],
              cutoff_rows)
    write_csv(out / "control_nonassoc.block_statistics.csv", list(block_rows[0]), block_rows)
    write_csv(out / "control_nonassoc.block_summary.csv", list(block_summary[0]), block_summary)
    write_csv(out / "control_nonassoc.msd.csv", ["time", "g_CM", "alpha"],
              (dict(time=float(t), g_CM=float(g), alpha=float(a)) for t, g, a in zip(time, msd, alpha)))
    (out / "control_nonassoc.diffusion.json").write_text(json.dumps(diffusion, indent=2, sort_keys=True) + "\n")
    report = {"rows": len(raw), "raw_duration": float((len(raw) - 1) * args.dt),
              "durations": duration_summary, "block_uncertainty_lag": uncertainty,
              "terminal_diagnostic": terminal,
              "fixed_cutoff_stability": cutoff_stability,
              "tau_term": None,
              "tau_term_status": "not claimed",
              "tau_term_definition": "No single terminal time is inferred from a 1/e crossing; use the block-resolved SEM/SD crossings, positive interval, and truncated integral diagnostic.",
              "com_frames": len(frame_data), "stars": int(com.shape[1]),
              "com_duration": float(time[-1]), "diffusion": diffusion,
              "no_zero_shear_viscosity_claim": True}
    if args.assoc_correlations:
        assoc = np.genfromtxt(args.assoc_correlations, names=True, delimiter=",")
        common = np.arange(0.0, min(stress_time[-1], assoc["time"][-1]) + args.dt / 2, args.dt)
        write_csv(out / "control_nonassoc_vs_assoc.csv", ["time", "G_nonassoc", "G_assoc"],
                  (dict(time=float(t), G_nonassoc=float(np.interp(t, stress_time, modulus)),
                        G_assoc=float(np.interp(t, assoc["time"], assoc["G"]))) for t in common))
        report["association_comparison"] = str(out / "control_nonassoc_vs_assoc.csv")
    (out / "control_nonassoc.summary.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
