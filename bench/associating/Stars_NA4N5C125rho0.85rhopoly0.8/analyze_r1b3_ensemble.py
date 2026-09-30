#!/usr/bin/env python3
"""Analyze R1-B3 using the six replicas (never internal blocks) as samples."""

import argparse
import csv
import os

import numpy as np


NREPLICAS = 6
# production.raw is: step, pxx, pyy, pzz, pxy, pxz, pyz, nxy, nxz, nyz.
# Thus columns 4:10 are exactly the six quantities passed, in this order, to
# fix ave/correlate/long in in.r1b3_production.lmp.
CORRELATION_COLUMNS = slice(4, 10)


def ac(x):
    """Return the unbiased autocorrelation at every available lag."""
    n = len(x)
    m = 1 << (2 * n - 1).bit_length()
    f = np.fft.rfft(x, m)
    return np.fft.irfft(f * f.conj(), m)[:n] / np.arange(n, 0, -1)


def load(path):
    x = np.loadtxt(path, comments="#")
    return x[None, :] if x.ndim == 1 else x


def replica_correlations(raw):
    """Calculate Cs, Cn/4, and their difference for one replica."""
    if raw.ndim != 2 or raw.shape[0] == 0 or raw.shape[1] < 10:
        raise ValueError("production.raw must be a nonempty table with at least 10 columns")
    channels = raw[:, CORRELATION_COLUMNS]
    assert channels.shape == (raw.shape[0], 6)
    aa = np.array([ac(channels[:, i]) for i in range(6)]).T
    n_lags = raw.shape[0]
    assert aa.shape == (n_lags, 6), "raw ACF array must have shape (n_lags, 6)"
    cs = np.mean(aa[:, :3], axis=1)
    cn_over4 = np.mean(aa[:, 3:], axis=1) / 4
    difference = cn_over4 - cs
    for name, value in (("Cs", cs), ("Cn_over4", cn_over4), ("D", difference)):
        assert value.ndim == 1 and len(value) == n_lags, (
            f"per-replica {name} must be one-dimensional with length n_lags"
        )
    return aa, cs, cn_over4, difference


def required_raw_paths(runs):
    """Return only the six explicitly named, completed production runs."""
    paths = []
    for replica in range(1, NREPLICAS + 1):
        directory = os.path.join(runs, f"replica{replica:02d}")
        marker = os.path.join(directory, "production.complete")
        raw = os.path.join(directory, "production.raw")
        if not os.path.isfile(marker):
            raise FileNotFoundError(f"required completion marker is missing: {marker}")
        if not os.path.isfile(raw) or os.path.getsize(raw) == 0:
            raise FileNotFoundError(f"required nonempty raw file is missing: {raw}")
        paths.append(raw)
    return paths


def compare_online(aa, gt_path, dt):
    """Compare online output only at compatible, represented integer lags."""
    g = load(gt_path)
    if g.ndim != 2 or g.shape[1] != 7 or g.shape[0] == 0:
        raise ValueError(f"{gt_path} must have columns time and six correlations")
    lag_float = g[:, 0] / dt
    lag = np.rint(lag_float).astype(int)
    if not np.allclose(lag_float, lag, rtol=0, atol=1e-8):
        raise ValueError(f"{gt_path} contains times incompatible with dt={dt}")
    if np.any(lag < 0) or np.any(lag >= aa.shape[0]):
        raise ValueError(f"{gt_path} contains lags outside the raw ACF range")
    online = g[:, 1:7]
    offline = aa[lag, :]
    assert offline.shape == online.shape
    absolute = np.abs(offline - online)
    # Relative errors convey no useful information near a zero crossing.  Scale
    # the cutoff independently for each correlation channel.
    cutoff = np.maximum(
        np.max(np.abs(online), axis=0) * np.sqrt(np.finfo(float).eps),
        np.finfo(float).eps,
    )
    stable = np.abs(online) > cutoff[None, :]
    relative = absolute[stable] / np.abs(online[stable])
    return {
        "online_max_abs": float(np.max(absolute)),
        "online_max_rel": float(np.max(relative)) if relative.size else np.nan,
        "online_relative_values": int(relative.size),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="?", default="r1b3_runs")
    parser.add_argument("--out", default="r1b3")
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--sustained-min-lags", type=int, default=1000,
                        help="minimum consecutive significant lags for sustained-sign reporting")
    args = parser.parse_args()
    if args.dt <= 0:
        parser.error("--dt must be positive")

    records = []
    checks = []
    n_lags = None
    for path in required_raw_paths(args.runs):
        aa, cs, cn_over4, difference = replica_correlations(load(path))
        if n_lags is None:
            n_lags = len(cs)
        elif len(cs) != n_lags:
            raise ValueError("all six production.raw files must have the same number of rows")
        records.append((cs, cn_over4, difference))
        check = {
            "replica": os.path.basename(os.path.dirname(path)),
            "R0": cn_over4[0] / cs[0],
        }
        gt = os.path.splitext(path)[0] + ".gt"
        if os.path.exists(gt):
            check.update(compare_online(aa, gt, args.dt))
        checks.append(check)

    cs = np.stack([record[0] for record in records])
    cn = np.stack([record[1] for record in records])
    difference = np.stack([record[2] for record in records])
    for name, value in (("Cs", cs), ("Cn_over4", cn), ("D", difference)):
        assert value.shape == (NREPLICAS, n_lags), (
            f"ensemble {name} must have shape (6, n_lags)"
        )

    mean = lambda x: x.mean(axis=0)
    sd = lambda x: x.std(axis=0, ddof=1)
    sem = lambda x: sd(x) / np.sqrt(NREPLICAS)
    cm, nm, dm = mean(cs), mean(cn), mean(difference)
    cs_sd, cn_sd, d_sd = sd(cs), sd(cn), sd(difference)
    ce, ne, de = sem(cs), sem(cn), sem(difference)
    good = np.abs(cm) > 2 * ce
    ratio = np.full(n_lags, np.nan)
    ratio[good] = nm[good] / cm[good]
    rows = [
        {
            "time": i * args.dt, "Cs": cm[i], "Cs_sd": cs_sd[i], "Cs_sem": ce[i],
            "Cn_over4": nm[i], "Cn_over4_sd": cn_sd[i], "Cn_over4_sem": ne[i],
            "D": dm[i], "D_sd": d_sd[i], "D_sem": de[i], "R_iso": ratio[i],
            "useful": int(good[i]),
        }
        for i in range(n_lags)
    ]
    with open(args.out + ".ensemble.csv", "w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    with open(args.out + ".replicas.csv", "w", newline="") as output:
        fields = sorted(set().union(*(check.keys() for check in checks)))
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(checks)

    summary = {
        "replicas": NREPLICAS,
        "R0_mean": float(np.mean([check["R0"] for check in checks])),
        "R0_sd": float(np.std([check["R0"] for check in checks], ddof=1)),
    }
    online_abs = [check["online_max_abs"] for check in checks if "online_max_abs" in check]
    online_rel = [check["online_max_rel"] for check in checks if "online_max_rel" in check]
    if not np.any(good):
        summary.update(status="insufficient ensemble precision", useful_lags=0)
    else:
        # A zero SEM with a nonzero difference is legitimately infinite evidence.
        sig = np.divide(np.abs(dm[good]), de[good], out=np.zeros_like(dm[good]),
                        where=de[good] != 0)
        sig[(de[good] == 0) & (dm[good] != 0)] = np.inf
        useful_indices = np.flatnonzero(good)
        maximum = int(np.argmax(sig))
        i = useful_indices[maximum]
        significant = good & (np.abs(dm) > 2 * de)
        max_run = run = 0
        max_run_start = None
        for j in np.flatnonzero(significant):
            if run and j == previous + 1 and np.sign(dm[j]) == np.sign(dm[previous]):
                run += 1
            else:
                run = 1
                run_start = j
            if run > max_run:
                max_run = run
                max_run_start = run_start
            previous = j
        summary.update(
            status="ok", useful_lags=int(good.sum()),
            useful_time_first=float(useful_indices[0] * args.dt),
            useful_time_last=float(useful_indices[-1] * args.dt),
            max_abs_D_over_sem=float(sig[maximum]), max_time=i * args.dt,
            max_D=float(dm[i]), fraction_within_1=float(np.mean(sig <= 1)),
            fraction_within_2=float(np.mean(sig <= 2)),
            max_significant_same_sign_lags=int(max_run),
            max_significant_same_sign_time=(None if max_run_start is None else
                                            float(max_run_start * args.dt)),
            sustained_same_sign_nonzero=bool(max_run >= args.sustained_min_lags),
        )
    summary["online_max_abs"] = None if not online_abs else float(max(online_abs))
    summary["online_max_rel"] = None if not online_rel else float(max(online_rel))
    with open(args.out + ".summary.csv", "w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=summary)
        writer.writeheader()
        writer.writerow(summary)
    print(summary)


if __name__ == "__main__":
    main()
