#!/usr/bin/env python3
"""Diagnostic R1-C1 analysis; this does not estimate a final viscosity."""

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np


LABELS = ("Pxx", "Pyy", "Pzz", "Pxy", "Pxz", "Pyz")
NAMES = ("xy", "xz", "yz", "nxy", "nxz", "nyz")
DIAGNOSTIC_LAGS = (1, 2, 5, 10, 20, 50, 100, 200, 500)
FIXED_CUTOFFS = (20, 50, 100, 200, 500)


def load_raw(path):
    values = np.loadtxt(path, comments="#")
    if values.ndim == 1:
        values = values[None, :]
    if values.ndim != 2 or values.shape[1] != 7 or values.shape[0] < 2:
        raise ValueError("raw stress must have >=2 rows and exactly 7 columns: step + six pressures")
    if not np.all(np.isfinite(values)) or not np.all(np.diff(values[:, 0]) == 1):
        raise ValueError("raw stress steps must be finite consecutive integers")
    return values


def autocorrelation(values):
    n = len(values)
    size = 1 << (2 * n - 1).bit_length()
    transformed = np.fft.rfft(values, size)
    return np.fft.irfft(transformed * transformed.conj(), size)[:n] / np.arange(n, 0, -1)


def correlations(raw):
    pressure = raw[:, 1:7]
    channels = np.column_stack((pressure[:, 3], pressure[:, 4], pressure[:, 5],
                                pressure[:, 0] - pressure[:, 1],
                                pressure[:, 0] - pressure[:, 2],
                                pressure[:, 1] - pressure[:, 2]))
    acf = np.column_stack([autocorrelation(channels[:, i]) for i in range(6)])
    cs = acf[:, :3].mean(axis=1)
    cn_over4 = acf[:, 3:].mean(axis=1) / 4.0
    difference = cn_over4 - cs
    return acf, cs, cn_over4, difference


def first_crossing(time, values, level):
    indices = np.flatnonzero(values <= level)
    if not len(indices):
        return None
    i = int(indices[0])
    if i == 0:
        return float(time[0])
    x0, x1, y0, y1 = time[i - 1], time[i], values[i - 1], values[i]
    if y1 == y0:
        return float(x1)
    return float(x0 + (level - y0) * (x1 - x0) / (y1 - y0))


def cumulative_integral(values, dt):
    return np.concatenate(([0.0], np.cumsum((values[1:] + values[:-1]) * 0.5 * dt)))


def _modulus(raw, volume, temperature, dt):
    acf, cs, cn_over4, difference = correlations(raw)
    time = np.arange(len(raw), dtype=float) * dt
    modulus = volume / (5.0 * temperature) * acf[:, :3].sum(axis=1)
    modulus += volume / (30.0 * temperature) * acf[:, 3:].sum(axis=1)
    return time, cs, cn_over4, difference, modulus, cumulative_integral(modulus, dt)


def duration_rows(raw, durations, volume, temperature, dt, max_lag=500.0):
    """Return common-lag duration rows and fixed-cutoff values."""
    rows, cutoffs = [], []
    for requested in durations:
        count = min(len(raw), int(round(requested / dt)) + 1)
        if count < 2:
            continue
        time, cs, cn, difference, modulus, eta = _modulus(
            raw[:count], volume, temperature, dt)
        duration = float(time[-1])
        limit = min(max_lag, duration)
        for lag in np.unique(np.r_[0.0, np.arange(dt, limit + dt / 2, dt)]):
            i = min(int(round(lag / dt)), len(modulus) - 1)
            rows.append({"duration": duration, "lag": float(lag), "G": float(modulus[i]),
                         "eta": float(eta[i])})
        for cutoff in FIXED_CUTOFFS:
            if cutoff <= duration:
                i = min(int(round(cutoff / dt)), len(eta) - 1)
                cutoffs.append({"duration": duration, "cutoff": cutoff,
                                "eta_cutoff": float(eta[i]),
                                "safe_lag_fraction": float(cutoff / duration)})
    return rows, cutoffs


def block_rows(raw, schemes, volume, temperature, dt):
    rows = []
    for blocks in schemes:
        for block, values in enumerate(np.array_split(raw, blocks), 1):
            if len(values) < 2:
                continue
            time, _, _, _, modulus, eta = _modulus(values, volume, temperature, dt)
            for lag in DIAGNOSTIC_LAGS:
                if lag <= time[-1]:
                    i = int(round(lag / dt))
                    rows.append({"scheme": f"{blocks}x", "blocks": blocks,
                                 "block": block, "block_duration": float(time[-1]),
                                 "lag": lag, "G": float(modulus[i]),
                                 "eta": float(eta[i])})
    return rows


def parse_network_snapshot(path):
    lines = Path(path).read_text().splitlines()
    if not lines or not lines[0].startswith("# timestep "):
        raise ValueError(f"malformed network snapshot: {path}")
    match = re.match(r"# timestep (\d+) bonds (\d+)$", lines[0])
    if not match or len(lines) < 2:
        raise ValueError(f"malformed network snapshot: {path}")
    timestep, expected = map(int, match.groups())
    edges = set()
    for line in lines[2:]:
        fields = line.split()
        if len(fields) != 4:
            raise ValueError(f"malformed network edge: {path}")
        try:
            a, b = int(fields[0]), int(fields[1])
        except ValueError as exc:
            raise ValueError(f"malformed network edge: {path}") from exc
        if a == b:
            raise ValueError(f"self-edge in network snapshot: {path}")
        edges.add(tuple(sorted((a, b))))
    if len(edges) != expected:
        raise ValueError(f"network bond count mismatch: {path}")
    return timestep, edges


def network_persistence(paths, snapshot_dt=100.0):
    if not paths:
        return {"status": "missing", "message": "no network snapshots supplied"}, []
    snapshots = []
    try:
        snapshots = sorted((parse_network_snapshot(path) for path in paths))
    except (OSError, ValueError) as exc:
        return {"status": "malformed", "message": str(exc)}, []
    rows = []
    for lag_index in range(len(snapshots)):
        overlaps, continuous = [], []
        for start in range(len(snapshots) - lag_index):
            initial = snapshots[start][1]
            if not initial:
                continue
            current = snapshots[start][1].copy()
            for stop in range(start + 1, start + lag_index + 1):
                current &= snapshots[stop][1]
            overlaps.append(len(initial & snapshots[start + lag_index][1]) / len(initial))
            continuous.append(len(current) / len(initial))
        rows.append({"lag": lag_index * snapshot_dt,
                     "snapshot_lag": lag_index,
                     "Q_edge_overlap": float(np.mean(overlaps)),
                     "sampled_continuous_survival": float(np.mean(continuous)),
                     "origins": len(overlaps),
                     "mean_edges": float(np.mean([len(s[1]) for s in snapshots]))})
    info = {"status": "ok", "snapshots": len(snapshots), "cadence": snapshot_dt}
    time = np.array([row["lag"] for row in rows])
    for name in ("Q_edge_overlap", "sampled_continuous_survival"):
        values = np.array([row[name] for row in rows])
        info[name + "_crossings"] = {
            str(level): first_crossing(time, values, level)
            for level in (0.9, 0.75, 0.5, 1.0 / np.e)
        }
    return info, rows


def block_summary(rows):
    summary = []
    for scheme in sorted({r["blocks"] for r in rows}):
        selected = [r for r in rows if r["blocks"] == scheme]
        for lag in DIAGNOSTIC_LAGS:
            values = [r["G"] for r in selected if r["lag"] == lag]
            etas = [r["eta"] for r in selected if r["lag"] == lag]
            if values:
                mean = float(np.mean(values))
                sd = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
                summary.append({"blocks": scheme, "lag": lag, "G_mean": mean,
                                "G_sd": sd, "G_sem": sd / np.sqrt(len(values)),
                                "G_sd_over_abs_mean": sd / abs(mean) if mean else None,
                                "eta_mean": float(np.mean(etas)),
                                "eta_sd": float(np.std(etas, ddof=1)) if len(etas) > 1 else 0.0})
    return summary


def online_compare(acf, path, dt, max_time):
    online = np.loadtxt(path, comments="#")
    if online.ndim == 1:
        online = online[None, :]
    if online.ndim != 2 or online.shape[1] != 7:
        raise ValueError("online correlator must have 7 columns: time + six correlations")
    selected = online[:, 0] <= max_time
    lags = np.rint(online[selected, 0] / dt).astype(int)
    if np.any(lags < 0) or np.any(lags >= len(acf)):
        raise ValueError("online correlator contains lags outside raw stress range")
    delta = np.abs(acf[lags] - online[selected, 1:7])
    return {
        "samples": int(delta.shape[0]),
        "max_abs": float(delta.max()) if delta.size else None,
        "mean_abs": float(delta.mean()) if delta.size else None,
        "relative_note": "not used; unstable near zero crossings",
    }


def analyze(raw, volume, temperature, dt, online_path=None, online_max_time=1000.0):
    acf, cs, cn_over4, difference = correlations(raw)
    time = np.arange(len(raw), dtype=float) * dt
    modulus = volume / (5.0 * temperature) * acf[:, :3].sum(axis=1)
    modulus += volume / (30.0 * temperature) * acf[:, 3:].sum(axis=1)
    g0 = float(modulus[0])
    normalized = modulus / g0 if g0 else np.full_like(modulus, np.nan)
    tail = modulus[max(1, int(0.9 * len(modulus))):]
    noise_floor = float(1.4826 * np.median(np.abs(tail - np.median(tail))))
    if noise_floor == 0.0:
        noise_floor = float(np.std(tail))
    useful = np.abs(modulus) > 3.0 * noise_floor if noise_floor else np.ones(len(modulus), bool)
    eta = cumulative_integral(modulus, dt)
    plateau = False
    if np.any(useful):
        end = int(np.flatnonzero(useful)[-1])
        window = max(1000, min(10000, max(2, end // 10)))
        if end >= window and abs(eta[end]) > 0:
            slope = abs(eta[end] - eta[end - window]) / (window * dt)
            plateau = bool(slope * (window * dt) <= 0.05 * abs(eta[end]) and
                           np.all(useful[end - window:end + 1]))
    result = {
        "rows": int(len(raw)), "duration": float(time[-1]),
        "mean_pressure": dict(zip(LABELS, raw[:, 1:7].mean(axis=0).tolist())),
        "R_iso_0": float(cn_over4[0] / cs[0]) if cs[0] else None,
        "Cs_0": float(cs[0]), "Cn_over4_0": float(cn_over4[0]), "G_0": g0,
        "noise_floor_MAD_tail": noise_floor,
        "useful_lag_first": float(time[np.flatnonzero(useful)[0]]) if np.any(useful) else None,
        "useful_lag_last": float(time[np.flatnonzero(useful)[-1]]) if np.any(useful) else None,
        "useful_lags": int(useful.sum()),
        "G_over_G0_crossings": {str(level): first_crossing(time, normalized, level)
                                for level in (0.1, 0.01, 0.001)},
        "eta_final_diagnostic": float(eta[-1]),
        "apparent_plateau_before_noise_dominated": plateau,
        "plateau_note": "diagnostic only; no zero-shear viscosity is claimed",
    }
    if online_path:
        result["online_offline"] = online_compare(acf, online_path, dt, online_max_time)
    return result, time, cs, cn_over4, difference, modulus, eta, useful


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("raw")
    parser.add_argument("--online", help="fix ave/correlate/long output")
    parser.add_argument("--out", default="r1c1")
    parser.add_argument("--volume", type=float, default=43563.0 / 0.85)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--online-max-time", type=float, default=1000.0)
    parser.add_argument("--network-glob", action="append", default=[],
                        help="network snapshot glob; repeat for multiple globs")
    parser.add_argument("--network-dt", type=float, default=100.0)
    parser.add_argument("--durations", default="1250,2500,5000,10000")
    parser.add_argument("--block-schemes", default="4,5,10")
    args = parser.parse_args()
    if args.volume <= 0 or args.temperature <= 0 or args.dt <= 0:
        parser.error("volume, temperature, and dt must be positive")
    raw = load_raw(args.raw)
    result, time, cs, cn, difference, modulus, eta, useful = analyze(
        raw, args.volume, args.temperature, args.dt, args.online, args.online_max_time)
    result["input"] = str(Path(args.raw))
    out_root = Path(args.out)
    out_root.parent.mkdir(parents=True, exist_ok=True)
    durations = [float(value) for value in args.durations.split(",") if value]
    schemes = [int(value) for value in args.block_schemes.split(",") if value]
    duration_data, cutoff_data = duration_rows(raw, durations, args.volume, args.temperature, args.dt)
    blocks = block_rows(raw, schemes, args.volume, args.temperature, args.dt)
    block_stats = block_summary(blocks)
    network_paths = []
    for pattern in args.network_glob:
        network_paths.extend(Path().glob(pattern) if not Path(pattern).is_absolute()
                             else Path(pattern).parent.glob(Path(pattern).name))
    network_info, network_data = network_persistence(sorted(set(network_paths)), args.network_dt)
    ratios = [r["G_sd_over_abs_mean"] for r in block_stats if r["lag"] == 500 and r["G_sd_over_abs_mean"] is not None]
    block_uncertainty_lags = {
        str(blocks): next((r["lag"] for r in block_stats
                           if r["blocks"] == blocks and r["G_sd_over_abs_mean"] is not None
                           and r["G_sd_over_abs_mean"] >= 1.0), None)
        for blocks in schemes
    }
    fixed_stable = False
    for cutoff in FIXED_CUTOFFS:
        values = [r["eta_cutoff"] for r in cutoff_data if r["cutoff"] == cutoff]
        if len(values) >= 2 and abs(values[-1] - values[-2]) <= 0.1 * max(abs(values[-1]), 1e-30):
            fixed_stable = True
    if network_info["status"] == "ok" and fixed_stable and not any(r >= 1 for r in ratios):
        gate = "C1-A"
    elif network_info["status"] == "ok" and not any(r >= 1 for r in ratios):
        gate = "C1-B"
    else:
        gate = "C1-C"
    result.update({
        "input": str(Path(args.raw)),
        "duration_windows": durations,
        "fixed_cutoffs": FIXED_CUTOFFS,
        "network": network_info,
        "network_lifetime_limitation": (
            "Network snapshots are spaced by 100 time units; edge overlap and sampled "
            "continuous survival cannot reconstruct the thesis-style renormalized bond "
            "lifetime. Detachment/rebinding between snapshots is unresolved."),
        "slow_tail_assessment": (
            "The long positive G(t) tail is not assigned a single relaxation time: "
            "block uncertainty becomes comparable to the mean before the tail is resolved."),
        "block_uncertainty_lag_G_sd_over_abs_mean_ge_1": block_uncertainty_lags,
        "tau_slow": None,
        "trajectory_over_tau_slow": None,
        "trajectory_span_classification": "not defensibly classifiable; unresolved slow tail, conservatively C1-C",
        "interpretation_gate": gate,
        "interpretation_note": (
            "No independent-replica count is inferred from T/tau; block statistics estimate "
            "effective information in one ergodic trajectory."),
    })
    output_dir = out_root.parent
    summary_path = output_dir / "r1c1_convergence.summary.json"
    duration_path = output_dir / "r1c1_duration_convergence.csv"
    cutoff_path = output_dir / "r1c1_fixed_cutoff.csv"
    block_path = output_dir / "r1c1_block_statistics.csv"
    block_summary_path = output_dir / "r1c1_block_summary.csv"
    network_path = output_dir / "r1c1_network_persistence.csv"
    with open(str(out_root) + ".correlations.csv", "w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(["time", "Cs", "Cn_over4", "D", "G", "eta", "useful"])
        writer.writerows(zip(time, cs, cn, difference, modulus, eta, useful.astype(int)))
    def write_csv(path, fieldnames, rows):
        with open(path, "w", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    write_csv(duration_path,
              ["duration", "lag", "G", "eta"], duration_data)
    write_csv(cutoff_path,
              ["duration", "cutoff", "eta_cutoff", "safe_lag_fraction"], cutoff_data)
    write_csv(block_path,
              ["scheme", "blocks", "block", "block_duration", "lag", "G", "eta"], blocks)
    write_csv(block_summary_path,
              ["blocks", "lag", "G_mean", "G_sd", "G_sem", "G_sd_over_abs_mean",
               "eta_mean", "eta_sd"], block_stats)
    write_csv(network_path,
              ["lag", "snapshot_lag", "Q_edge_overlap", "sampled_continuous_survival",
               "origins", "mean_edges"], network_data)
    result["outputs"] = {
        "summary": str(summary_path),
        "duration_convergence": str(duration_path),
        "fixed_cutoff": str(cutoff_path),
        "block_statistics": str(block_path),
        "block_summary": str(block_summary_path),
        "network_persistence": str(network_path),
    }
    with open(summary_path, "w") as output:
        json.dump(result, output, indent=2, sort_keys=True)
        output.write("\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
