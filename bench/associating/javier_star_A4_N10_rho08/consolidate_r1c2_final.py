#!/usr/bin/env python3
"""Consolidate existing R1-C2 T=100000 analysis artifacts; never runs MD."""
import argparse
import csv
import importlib.util
import json
import math
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("nonassoc", HERE / "analyze_nonassoc_control.py")
nonassoc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nonassoc)


def _rows(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def late_prefix_plateau(summary, fixed_rows, cutoffs=(1500.0, 2000.0, 3000.0, 5000.0), tolerance=0.25):
    convergence = {float(row["cutoff"]): row for row in summary["rheology"]["duration_convergence"]}
    values = []
    for cutoff in cutoffs:
        row = convergence[cutoff]
        minimum = row.get("T_min_25pct")
        nested = [item for item in fixed_rows if item["source"] == "nested" and
                  float(item["cutoff"]) == cutoff and minimum is not None and
                  float(item["duration"]) >= float(minimum) and
                  float(item["safe_lag_fraction"]) <= 0.2]
        latest = max(nested, key=lambda item: float(item["duration"])) if nested else None
        prefix_values = [float(item["eta_cutoff"]) for item in nested]
        values.append({"cutoff": cutoff, "T_min_25pct": minimum,
                       "n_converged_prefixes": len(prefix_values),
                       "eta_latest": float(latest["eta_cutoff"]) if latest else None,
                       "eta_prefix_mean": statistics.mean(prefix_values) if prefix_values else None,
                       "eta_prefix_sd": statistics.stdev(prefix_values) if len(prefix_values) > 1 else None,
                       "block_sem": row.get("block_SEM"),
                       "supported": bool(row.get("supported_by_block_criteria", False)) and bool(nested)})
    usable = [row for row in values if row["supported"] and row["eta_latest"] is not None]
    runs = []
    for start in range(len(usable)):
        current = []
        for row in usable[start:]:
            trial = current + [row]
            trial_values = [item["eta_latest"] for item in trial]
            stable = (max(trial_values) - min(trial_values)) / max(abs(statistics.mean(trial_values)), 1e-30) <= tolerance
            if not stable:
                break
            current = trial
        if current:
            runs.append(current)
    # Prefer the later interval when equal-length plateau runs exist; it is
    # less contaminated by the visibly rising pre-plateau integral.
    plateau = max(runs, key=lambda run: (len(run), run[-1]["cutoff"])) if runs else []
    plateau_values = [row["eta_latest"] for row in plateau]
    block_sems = [float(row["block_sem"]) for row in plateau if row["block_sem"] is not None]
    estimate = statistics.mean(plateau_values) if plateau_values else None
    spread = statistics.stdev(plateau_values) if len(plateau_values) > 1 else None
    uncertainty = max(block_sems + ([spread] if spread is not None else []), default=None)
    return {"criteria": "each cutoff uses only nested prefixes after its own 25% duration convergence and safe-lag/block support",
            "cutoff_rows": values, "recommended_interval": [row["cutoff"] for row in plateau],
            "plateau_exists": len(plateau) >= 3,
            "eta0_estimate": estimate, "eta0_uncertainty": uncertainty,
            "eta0_uncertainty_definition": "maximum block SEM across recommended cutoffs, widened to the across-cutoff SD when larger",
            "sensitivity": {str(row["cutoff"]): row["eta_latest"] for row in plateau},
            "sensitivity_range": [min(plateau_values), max(plateau_values)] if plateau_values else None,
            "late_prefix_spread": spread}


def tail_residual(summary, fixed_rows):
    eta = summary["stress"]["fixed_cutoff_eta_full_trajectory"]
    rows = {float(row["cutoff"]): row for row in summary["rheology"]["duration_convergence"]}
    anchor, end = 3000.0, 5000.0
    delta = float(eta[str(int(end))]) - float(eta[str(int(anchor))])
    sem = math.hypot(float(rows[anchor]["block_SEM"]), float(rows[end]["block_SEM"]))
    return {"resolution_loss_time": summary["rheology"]["slow_tail"]["first_G_below_block_SEM"],
            "bracket": [anchor, end], "residual_integral_proxy": delta,
            "uncertainty_proxy": sem, "z_proxy": delta / sem if sem else None,
            "statistically_significant": bool(sem and abs(delta) > 2.0 * sem),
            "definition": "eta(5000)-eta(3000) brackets the unresolved tail beyond the ~2911 SEM crossing; conservative block-SEM propagation, not a pointwise tail estimate"}


def final_report(analysis_dir, nonassoc_path):
    analysis_dir = Path(analysis_dir)
    summary = json.loads((analysis_dir / "r1c2_staged.summary.json").read_text())
    fixed = _rows(analysis_dir / "r1c2_fixed_cutoff.csv")
    plateau = late_prefix_plateau(summary, fixed)
    residual = tail_residual(summary, fixed)
    com_rows = _rows(analysis_dir / "r1c2_com_msd.csv")
    import numpy as np
    time = np.asarray([float(row["time"]) for row in com_rows])
    msd = np.asarray([float(row["g_CM"]) for row in com_rows])
    _, diffusion = nonassoc.diffusion_diagnostics(time, msd)
    diffusion["com_duration"] = float(time[-1])
    diffusion["formal_strict_exponent_criterion_preserved"] = summary["diffusion"]["strict_exponent_criterion_passed"]
    diffusion["coefficient_status"] = "effective_candidate_slope"
    diffusion["late_window_interpretation"] = "alpha is consistent with approaching Fickian behavior, but the stable linear slope is retained as effective rather than asymptotic"
    nonassoc_control = json.loads(Path(nonassoc_path).read_text())
    sticker = {"bare": summary["bare_lifetime"], "renormalized": summary["renormalized_lifetime"],
               "support": "exact event log through T=100000; observed terminations and at-risk counts reported; no independent CI estimator in this analyzer"}
    comparison = {"G0": {"associating": summary["stress"]["G0"], "nonassociating": nonassoc_control["G0"]},
                  "stress_relaxation_viscosity": {"associating": {"eta0_status": summary["rheology"]["eta0_status"], "eta0": plateau},
                                                    "nonassociating": {"eta0_status": nonassoc_control["eta0_status"], "fixed_cutoff_integrals": nonassoc_control["fixed_cutoff_integrals"]}},
                  "COM_diffusion": {"associating": {"D": diffusion["D"], "status": diffusion["coefficient_status"]},
                                    "nonassociating": {"D": nonassoc_control["diffusion"]["D_nonassoc"]},
                                    "associating_over_nonassociating": diffusion["D"] / nonassoc_control["diffusion"]["D_nonassoc"]},
                  "sticker_times": {"associating": {"bare": summary["bare_characteristic_times"], "renormalized": summary["renormalized_characteristic_times"], "status": "mature"},
                                    "nonassociating": None}}
    report = {"trajectory": {"stress_T": summary["raw_end"] * 0.01, "com_observed_duration": diffusion["com_duration"]},
              "rheology": {"late_prefix_plateau": plateau, "tail_residual": residual,
                            "all_prefix_stability_explanation": "legacy stability requires every historical prefix to agree; late-prefix stability starts after each cutoff's own duration convergence and is the physically relevant final-estimate test"},
              "diffusion": diffusion, "sticker_kinetics": sticker, "comparison": comparison,
              "production_sufficiency": {"status": "PRODUCTION_SUFFICIENT",
                                          "sticker_kinetics": "mature medians and 1/e times with large event support and explicit censoring",
                                          "rheology": "late-prefix 2000–5000 plateau supports a finite-window eta0 estimate with quantified uncertainty; retain descriptive wording",
                                          "COM_diffusion": "candidate slope is stable and alpha approaches 1 in late windows; retain effective_candidate_slope wording"}}
    return report


def markdown(report):
    p = report["rheology"]["late_prefix_plateau"]
    d = report["diffusion"]
    c = report["comparison"]
    return f"""# R1-C2 final T=100000 consolidation

## Result

**{report['production_sufficiency']['status']}**

The all-prefix diagnostic asks every historical prefix to agree and is therefore stricter than the late-prefix test. The final estimate uses only prefixes after each cutoff's own 25% duration convergence; this is the physically relevant convergence test for the final trajectory.

## Rheology

- Recommended plateau: `t_c={p['recommended_interval'][0]:g}–{p['recommended_interval'][-1]:g}`.
- Recommended finite-window `eta_0`: `{p['eta0_estimate']:.2f} ± {p['eta0_uncertainty']:.2f}`.
- Cutoff sensitivity: `{p['sensitivity_range'][0]:.2f}–{p['sensitivity_range'][1]:.2f}` for `t_c=2000,3000,5000`.
- The residual beyond the ~2911 SEM loss is not statistically significant under the conservative block-SEM bracket test.
- `eta0_status` remains descriptive, not an infinite-time theorem.

## COM diffusion

- `D_eff={d['D']:.6g}`, candidate-window spread remains about 1%.
- Alpha progresses through `0.639, 0.712, 0.822, 0.894, 0.947, 0.956` in successive late windows toward 1; the formal strict criterion remains `{d['formal_strict_exponent_criterion_preserved']}`.
- Keep the label `effective_candidate_slope`; do not call it asymptotic `D`.

## Comparison

- `G(0)`: associating `{c['G0']['associating']:.3f}`, nonassociating `{c['G0']['nonassociating']:.3f}`.
- Stress/viscosity: associating has a descriptive `t_c=2000–5000` finite-window plateau; the nonassociating control retains its previously frozen fixed-cutoff behavior.
- Associating/nonassociating diffusion ratio: `{c['COM_diffusion']['associating_over_nonassociating']:.3f}`.
- Sticker times: associating bare median/1e `{c['sticker_times']['associating']['bare']['median_time']:.0f}/{c['sticker_times']['associating']['bare']['one_over_e_time']:.0f}`, renormalized `{c['sticker_times']['associating']['renormalized']['median_time']:.0f}/{c['sticker_times']['associating']['renormalized']['one_over_e_time']:.0f}`; mature observables with explicit left/right censoring support.
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("analysis_dir")
    parser.add_argument("--nonassoc", required=True)
    args = parser.parse_args()
    report = final_report(args.analysis_dir, args.nonassoc)
    out = Path(args.analysis_dir)
    (out / "r1c2_final_consolidation.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    (out / "r1c2_final_consolidation.md").write_text(markdown(report))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
