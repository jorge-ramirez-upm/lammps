#!/usr/bin/env python3
"""Forensic, zero-lag-only comparison of existing R1-B2 and R1-B3 data.

This program deliberately performs no dynamics and makes no finite-lag gate.
The primary R0 calculation is the literal mean of instantaneous stress
products.  An FFT is evaluated only as an implementation cross-check.
"""

import argparse
import csv
import glob
import os
from pathlib import Path

import numpy as np


def load(path):
    value = np.loadtxt(path, comments="#")
    value = value[None, :] if value.ndim == 1 else value
    if value.ndim != 2 or not len(value):
        raise ValueError(f"empty or malformed table: {path}")
    return value


def r0(stress):
    """Return Cs(0), Cn(0)/4, and their ratio, without an FFT."""
    shear = stress[:, 3:6]
    normal = np.column_stack((stress[:, 0] - stress[:, 1],
                              stress[:, 0] - stress[:, 2],
                              stress[:, 1] - stress[:, 2]))
    cs = np.mean(shear * shear)
    cn4 = np.mean(normal * normal) / 4.0
    return cs, cn4, cn4 / cs


def fft_r0(stress):
    """Independently obtain lag zero through the FFT autocorrelation route."""
    channels = np.column_stack((stress[:, 3:6],
                                stress[:, 0] - stress[:, 1],
                                stress[:, 0] - stress[:, 2],
                                stress[:, 1] - stress[:, 2]))
    n = len(channels)
    size = 1 << (2 * n - 1).bit_length()
    zero = np.array([
        np.fft.irfft(np.fft.rfft(x, size) *
                     np.fft.rfft(x, size).conj(), size)[0] / n
        for x in channels.T
    ])
    cs, cn4 = zero[:3].mean(), zero[3:].mean() / 4.0
    return cs, cn4, cn4 / cs


def diagnostics(path):
    if not os.path.isfile(path):
        return None
    table = load(path)
    if table.shape[1] < 6:
        raise ValueError(f"diagnostics need step,temp,pe,active,created,broken: {path}")
    return table[:, :6]


def edges(path):
    result = set()
    with open(path, encoding="utf-8") as source:
        for line in source:
            fields = line.split()
            if len(fields) >= 2:
                try:
                    a, b = int(fields[0]), int(fields[1])
                except ValueError:
                    continue
                result.add(tuple(sorted((a, b))))
    return result


def online_zero(path):
    table = load(path)
    row = table[np.argmin(np.abs(table[:, 0]))]
    if not np.isclose(row[0], 0.0):
        raise ValueError(f"no zero-lag row in {path}")
    if len(row) != 7:
        raise ValueError(f"expected lag plus six channels in {path}")
    cs = row[1:4].mean()
    cn4 = row[4:7].mean() / 4.0
    return cs, cn4, cn4 / cs


def snapshot_step(path):
    return int(Path(path).name.split(".")[-2])


def add_window(rows, dataset, label, stress, start, stop):
    cs, cn4, ratio = r0(stress[start:stop])
    rows.append(dict(dataset=dataset, window=label, start_row=start,
                     stop_row=stop, samples=stop-start, Cs0=cs,
                     Cn0_over4=cn4, R0=ratio))


def means(stress, diag):
    out = {"pxx": stress[:, 0].mean(), "pyy": stress[:, 1].mean(),
           "pzz": stress[:, 2].mean(), "pxy": stress[:, 3].mean(),
           "pxz": stress[:, 4].mean(), "pyz": stress[:, 5].mean()}
    if diag is not None:
        out.update(temperature=diag[:, 1].mean(), pe=diag[:, 2].mean(),
                   active=diag[:, 3].mean())
    return out


def write_csv(path, rows):
    if not rows:
        return
    fields = []
    for row in rows:
        fields.extend(key for key in row if key not in fields)
    with open(path, "w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("b2_raw", help="existing *.r1b2.raw file")
    parser.add_argument("runs", nargs="?", default="r1b3_runs")
    parser.add_argument("--out", default="r1b31")
    parser.add_argument("--chunk-steps", type=int, default=100000)
    args = parser.parse_args()
    if args.chunk_steps <= 0:
        parser.error("--chunk-steps must be positive")

    windows, summary, continuity = [], [], []
    b2 = load(args.b2_raw)
    if b2.shape[1] < 11:
        raise ValueError("B2 raw table needs at least 11 columns")
    # B2: step,pe,active,created,broken,total xx,yy,zz,xy,xz,yz,...
    datasets = [("B2", b2[:, 5:11], b2[:, 0], None,
                 {"pe": b2[:, 1].mean(), "active": b2[:, 2].mean()})]

    for replica in range(1, 7):
        name = f"B3-replica{replica:02d}"
        directory = os.path.join(args.runs, f"replica{replica:02d}")
        raw_path = os.path.join(directory, "production.raw")
        raw = load(raw_path)
        if raw.shape[1] < 10:
            raise ValueError(f"B3 raw table needs 10 columns: {raw_path}")
        diag = diagnostics(os.path.join(directory, "production.diagnostics"))
        datasets.append((name, raw[:, 1:7], raw[:, 0], diag, {}))

        eqdiag = diagnostics(os.path.join(directory, "equil.diagnostics"))
        record = {"dataset": name}
        if eqdiag is not None and diag is not None:
            record.update(equil_last_active=eqdiag[-1, 3],
                          production_first_active=diag[0, 3],
                          active_jump=diag[0, 3] - eqdiag[-1, 3],
                          equil_last_step=eqdiag[-1, 0],
                          production_first_step=diag[0, 0])
        before = sorted(glob.glob(os.path.join(directory, "equil.network.*.dat")),
                        key=snapshot_step)
        after = sorted(glob.glob(os.path.join(directory, "production.network.*.dat")),
                       key=snapshot_step)
        if before and after:
            eb, ea = edges(before[-1]), edges(after[0])
            record.update(network_before=os.path.basename(before[-1]),
                          network_after=os.path.basename(after[0]),
                          edges_before=len(eb), edges_after=len(ea),
                          common_edges=len(eb & ea), exact_edge_set=int(eb == ea))
            # Production snapshots normally start at 10k, not immediately at
            # restart; never mislabel turnover during that interval as a jump.
            record["edge_comparison_immediate"] = int(snapshot_step(after[0]) == 0)
        continuity.append(record)

    for name, stress, step, diag, extras in datasets:
        if len(stress) != len(step):
            raise AssertionError("stress/step length mismatch")
        direct = r0(stress)
        via_fft = fft_r0(stress)
        if not np.allclose(direct, via_fft, rtol=2e-13, atol=1e-14):
            raise AssertionError(f"direct and FFT lag zero disagree for {name}")
        record = dict(dataset=name, samples=len(stress), R0_direct=direct[2],
                      R0_fft=via_fft[2], direct_fft_abs=abs(direct[2]-via_fft[2]))
        record.update(means(stress, diag))
        record.update(extras)
        half = len(stress) // 2
        first_r0, second_r0 = r0(stress[:half])[2], r0(stress[half:])[2]
        record.update(first_half_R0=first_r0, second_half_R0=second_r0,
                      half_change=second_r0-first_r0,
                      second_half_closer_to_one=int(
                          abs(second_r0-1.0) < abs(first_r0-1.0)))
        if name.startswith("B3"):
            online = online_zero(os.path.join(
                args.runs, name.replace("B3-", ""), "production.gt"))
            record.update(R0_online=online[2],
                          raw_online_abs=abs(direct[2]-online[2]))
        summary.append(record)

        first_step = int(step[0])
        bins = ((step.astype(int) - first_step) // args.chunk_steps)
        for value in np.unique(bins):
            index = np.flatnonzero(bins == value)
            add_window(windows, name, f"chunk-{value + 1}", stress,
                       index[0], index[-1] + 1)
        chunk_r0 = [row["R0"] for row in windows
                    if row["dataset"] == name and row["window"].startswith("chunk-")]
        record["chunk_R0_slope_per_100k"] = (
            float(np.polyfit(np.arange(len(chunk_r0)), chunk_r0, 1)[0])
            if len(chunk_r0) > 1 else np.nan)
        add_window(windows, name, "first-half", stress, 0, half)
        add_window(windows, name, "second-half", stress, half, len(stress))

    write_csv(args.out + ".summary.csv", summary)
    write_csv(args.out + ".windows.csv", windows)
    write_csv(args.out + ".restart.csv", continuity)

    b3 = [row for row in summary if row["dataset"].startswith("B3")]
    ratios = np.array([row["R0_direct"] for row in b3])
    first = np.array([row["R0"] for row in windows
                      if row["dataset"].startswith("B3") and row["window"] == "first-half"])
    second = np.array([row["R0"] for row in windows
                       if row["dataset"].startswith("B3") and row["window"] == "second-half"])
    print("individual B3 R0:", " ".join(f"{x:.9g}" for x in ratios))
    print(f"B3 replica mean +/- SD: {ratios.mean():.9g} +/- {ratios.std(ddof=1):.9g}")
    print(f"B2 R0: {summary[0]['R0_direct']:.9g}")
    print(f"B3 half means: first={first.mean():.9g}, second={second.mean():.9g}, "
          f"change={second.mean()-first.mean():+.9g}")
    print("B3 ensemble moves toward 1 in the second half:" ,
          abs(second.mean()-1.0) < abs(first.mean()-1.0))
    print("Wrote", args.out + ".summary.csv,", args.out + ".windows.csv, and",
          args.out + ".restart.csv")


if __name__ == "__main__":
    main()
