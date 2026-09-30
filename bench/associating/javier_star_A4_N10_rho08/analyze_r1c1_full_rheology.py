#!/usr/bin/env python3
"""Diagnostic R1-C1 analysis; this does not estimate a final viscosity."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


LABELS = ("Pxx", "Pyy", "Pzz", "Pxy", "Pxz", "Pyz")
NAMES = ("xy", "xz", "yz", "nxy", "nxz", "nyz")


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
    eta = np.concatenate(([0.0], np.cumsum((modulus[1:] + modulus[:-1]) * 0.5 * dt)))
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
    args = parser.parse_args()
    if args.volume <= 0 or args.temperature <= 0 or args.dt <= 0:
        parser.error("volume, temperature, and dt must be positive")
    raw = load_raw(args.raw)
    result, time, cs, cn, difference, modulus, eta, useful = analyze(
        raw, args.volume, args.temperature, args.dt, args.online, args.online_max_time)
    result["input"] = str(Path(args.raw))
    with open(args.out + ".summary.json", "w") as output:
        json.dump(result, output, indent=2, sort_keys=True)
        output.write("\n")
    with open(args.out + ".correlations.csv", "w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(["time", "Cs", "Cn_over4", "D", "G", "eta", "useful"])
        writer.writerows(zip(time, cs, cn, difference, modulus, eta, useful.astype(int)))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
