#!/usr/bin/env python3
"""Compare the R1-B3.2 single- and multi-pressure-compute controls."""
import argparse
import json
from pathlib import Path

import numpy as np

LABELS = ("xx", "yy", "zz", "xy", "xz", "yz")


def load(path):
    values = np.loadtxt(path, comments="#")
    return values[None, :] if values.ndim == 1 else values


def r0(tensor):
    shear = tensor[:, 3:6]
    normal = np.column_stack((tensor[:, 0] - tensor[:, 1],
                              tensor[:, 0] - tensor[:, 2],
                              tensor[:, 1] - tensor[:, 2]))
    cs = np.mean(shear * shear)
    cn_over_four = np.mean(normal * normal) / 4.0
    return cs, cn_over_four, cn_over_four / cs if cs else float("nan")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("single")
    parser.add_argument("multi")
    parser.add_argument("--single-state")
    parser.add_argument("--multi-state")
    parser.add_argument("--single-run0")
    parser.add_argument("--multi-run0")
    parser.add_argument("--out", default="r1b32.json")
    parser.add_argument("--atol", type=float, default=1.0e-12,
                        help="allowed absolute total-tensor difference")
    args = parser.parse_args()

    single, multi = load(args.single), load(args.multi)
    if single.shape[1] != 11 or multi.shape[1] != 35:
        raise SystemExit("expected 11-column single and 35-column multi raw files")
    if not np.array_equal(single[:, 0], multi[:, 0]):
        raise SystemExit("sample steps differ")
    total_single, total_multi = single[:, 5:11], multi[:, 5:11]
    total_delta = total_multi - total_single
    components = multi[:, 11:17] + multi[:, 17:23] + multi[:, 23:29] + multi[:, 29:35]
    reconstruction = total_multi - components
    run0_delta = run0_reconstruction = None
    if args.single_run0 or args.multi_run0:
        if not (args.single_run0 and args.multi_run0):
            raise SystemExit("both run-0 paths are required")
        single0, multi0 = load(args.single_run0), load(args.multi_run0)
        run0_delta = multi0[0, 1:7] - single0[0, 1:7]
        run0_reconstruction = multi0[0, 1:7] - sum(
            (multi0[0, start:start + 6] for start in (7, 13, 19, 25)))
    state_equal = None
    if args.single_state or args.multi_state:
        if not (args.single_state and args.multi_state):
            raise SystemExit("both state paths are required")
        state_equal = Path(args.single_state).read_bytes() == Path(args.multi_state).read_bytes()

    cs_s, cn_s, ratio_s = r0(total_single)
    cs_m, cn_m, ratio_m = r0(total_multi)
    result = {
        "samples": len(single),
        "state_files_bitwise_equal": state_equal,
        "total_tensor_bitwise_equal": bool(np.array_equal(total_single, total_multi)),
        "max_abs_total_difference": float(np.max(np.abs(total_delta))),
        "max_abs_reconstruction_residual": float(np.max(np.abs(reconstruction))),
        "component_max_abs_total_difference": dict(zip(LABELS, np.max(np.abs(total_delta), axis=0).tolist())),
        "component_run0_total_difference": None if run0_delta is None else dict(zip(LABELS, run0_delta.tolist())),
        "component_run0_reconstruction_residual": None if run0_reconstruction is None else dict(zip(LABELS, run0_reconstruction.tolist())),
        "single_mean_total": dict(zip(LABELS, total_single.mean(axis=0).tolist())),
        "multi_mean_total": dict(zip(LABELS, total_multi.mean(axis=0).tolist())),
        "mean_diagonal_difference": float(total_delta[:, :3].mean()),
        "single_variance": dict(zip(LABELS, total_single.var(axis=0).tolist())),
        "multi_variance": dict(zip(LABELS, total_multi.var(axis=0).tolist())),
        "variance_difference": dict(zip(LABELS, (total_multi.var(axis=0) - total_single.var(axis=0)).tolist())),
        "single_Cs0": float(cs_s), "multi_Cs0": float(cs_m),
        "Cs0_difference": float(cs_m - cs_s),
        "single_CN0_over_4": float(cn_s), "multi_CN0_over_4": float(cn_m),
        "single_R0": float(ratio_s), "multi_R0": float(ratio_m),
        "R0_difference": float(ratio_m - ratio_s),
    }
    Path(args.out).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    run0_bad = run0_delta is not None and (np.max(np.abs(run0_delta)) > args.atol or
                                            np.max(np.abs(run0_reconstruction)) > args.atol)
    if result["max_abs_total_difference"] > args.atol or state_equal is False or run0_bad:
        raise SystemExit("single and multi cases differ; add component computes one at a time")


if __name__ == "__main__":
    main()
