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
                                 "eta_final": float(eta[-1]),
                                 "G_over_G0_crossing_0.1": r1c1.first_crossing(time, normalized, .1),
                                 "G_over_G0_crossing_0.01": r1c1.first_crossing(time, normalized, .01)})

    block_rows = []
    lags = tuple(sorted(set(cutoffs)))
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
                                  "eta_sd": float(etas.std(ddof=1)) if len(etas) > 1 else 0.0})
    uncertainty = {str(n): next((row["lag"] for row in block_summary
                                 if row["blocks"] == n and row["G_sd_over_abs_mean"] is not None
                                 and row["G_sd_over_abs_mean"] >= 1.0), None) for n in blocks}
    return duration_summary, duration_rows, cutoff_rows, block_rows, block_summary, uncertainty


def _open_dump(path):
    return gzip.open(path, "rt") if str(path).endswith(".gz") else open(path)


def dump_frames(path, expected_atoms=41000):
    with _open_dump(path) as handle:
        while True:
            line = handle.readline()
            if not line:
                return
            if line.strip() != "ITEM: TIMESTEP":
                continue
            step = int(handle.readline())
            if handle.readline().strip() != "ITEM: NUMBER OF ATOMS":
                raise ValueError("malformed trajectory atom-count header")
            count = int(handle.readline())
            if expected_atoms and count != expected_atoms:
                raise ValueError(f"expected {expected_atoms} polymer atoms, got {count}")
            if not handle.readline().startswith("ITEM: BOX BOUNDS"):
                raise ValueError("malformed trajectory box header")
            for _ in range(3):
                if not handle.readline():
                    raise ValueError("truncated trajectory box")
            header = handle.readline().split()
            if header[:2] != ["ITEM:", "ATOMS"]:
                raise ValueError("malformed trajectory atom header")
            columns = {name: index for index, name in enumerate(header[2:])}
            required = ("id", "mol", "type", "xu", "yu", "zu")
            if any(name not in columns for name in required):
                raise ValueError("trajectory needs id mol type xu yu zu")
            centers, counts = {}, {}
            for _ in range(count):
                fields = handle.readline().split()
                if len(fields) <= max(columns.values()):
                    raise ValueError("truncated trajectory atom record")
                if int(fields[columns["type"]]) not in (1, 2):
                    raise ValueError("polymer trajectory contains a non-polymer type")
                mol = int(fields[columns["mol"]])
                point = np.array([float(fields[columns[name]]) for name in ("xu", "yu", "zu")])
                centers[mol] = centers.get(mol, np.zeros(3)) + point
                counts[mol] = counts.get(mol, 0) + 1
            molecules = sorted(centers)
            if not molecules or any(counts[mol] != 41 for mol in molecules):
                raise ValueError("each star must have 41 type-1/type-2 beads")
            yield step, np.array([centers[mol] / counts[mol] for mol in molecules])


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


def diffusion_diagnostics(time, msd):
    valid = (time > 0) & (msd > 0)
    alpha = np.full_like(msd, np.nan)
    alpha[valid] = np.gradient(np.log(msd[valid]), np.log(time[valid]))
    indices = np.flatnonzero(valid)
    tail = indices[int(len(indices) * .75):] if len(indices) else []
    if len(tail) >= 2:
        slope, _ = np.polyfit(time[tail], msd[tail], 1)
        diffusion = float(slope / 6.0)
    else:
        slope = diffusion = None
    onset = None
    for index in range(1, len(alpha) - 5):
        window = alpha[index:index + 5]
        if np.all(np.isfinite(window)) and .8 <= np.median(window) <= 1.2:
            onset = float(time[index]); break
    return alpha, {"D": diffusion, "late_msd_slope": float(slope) if slope is not None else None,
                   "fickian_onset": onset,
                   "definition": "late linear MSD slope divided by 6; onset is the first five-point alpha window with median 0.8..1.2"}


def write_csv(path, fields, rows):
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("raw")
    parser.add_argument("trajectory")
    parser.add_argument("--out-dir", default=".")
    parser.add_argument("--volume", type=float, default=43563.0 / 0.85)
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
    duration_summary, duration_rows, cutoff_rows, block_rows, block_summary, uncertainty = _rheology_rows(
        raw, durations, cutoffs, blocks, args.volume, args.temperature, args.dt, args.max_lag)
    frame_data = list(dump_frames(args.trajectory))
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
    reference_index = min(int(round(1.0 / args.dt)), len(modulus) - 1)
    slow_reference = modulus[reference_index]
    slow_candidate = None
    if slow_reference > 0 and stress_time[-1] > stress_time[reference_index]:
        slow_candidate = r1c1.first_crossing(
            stress_time[reference_index:], modulus[reference_index:] / slow_reference, 1.0 / math.e)
    uncertainty_lag = uncertainty.get("5")
    tau_term = (slow_candidate if slow_candidate is not None and
                (uncertainty_lag is None or slow_candidate < uncertainty_lag) else None)
    write_csv(out / "control_nonassoc.correlations.csv",
              ["time", "Cs", "Cn_over4", "D", "G", "eta", "useful"],
              (dict(time=float(t), Cs=float(s), Cn_over4=float(n), D=float(d),
                    G=float(g), eta=float(e), useful=1)
               for t, s, n, d, g, e in zip(stress_time, cs, cn, difference, modulus, eta)))
    write_csv(out / "control_nonassoc.duration_convergence.csv", list(duration_rows[0]), duration_rows)
    write_csv(out / "control_nonassoc.fixed_cutoff.csv", list(cutoff_rows[0]), cutoff_rows)
    write_csv(out / "control_nonassoc.block_statistics.csv", list(block_rows[0]), block_rows)
    write_csv(out / "control_nonassoc.block_summary.csv", list(block_summary[0]), block_summary)
    write_csv(out / "control_nonassoc.msd.csv", ["time", "g_CM", "alpha"],
              (dict(time=float(t), g_CM=float(g), alpha=float(a)) for t, g, a in zip(time, msd, alpha)))
    (out / "control_nonassoc.diffusion.json").write_text(json.dumps(diffusion, indent=2, sort_keys=True) + "\n")
    report = {"rows": len(raw), "raw_duration": float((len(raw) - 1) * args.dt),
              "durations": duration_summary, "block_uncertainty_lag": uncertainty,
              "slow_reference_lag": min(1.0, float(stress_time[-1])),
              "slow_reference_1_over_e_candidate": slow_candidate,
              "tau_term": tau_term,
              "tau_term_status": "resolved before block uncertainty" if tau_term is not None else "unresolved or statistically ambiguous",
              "tau_term_definition": "slow-reference analysis starts at t=1; a terminal time is claimed only if the slow-component 1/e crossing precedes block uncertainty",
              "trajectory_frames": len(frame_data), "stars": int(com.shape[1]),
              "trajectory_duration": float(time[-1]), "diffusion": diffusion,
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
