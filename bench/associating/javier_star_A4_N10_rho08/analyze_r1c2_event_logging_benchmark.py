#!/usr/bin/env python3
"""Summarize OFF/ON event-logging benchmark runs without external packages."""
import csv
import json
import re
import statistics
import sys
from pathlib import Path


def metric(path):
    directory = Path(path)
    wall = float((directory / "wall_seconds").read_text().strip())
    log = (directory / "lammps.log").read_text()
    match = re.findall(r"ASSOCIATING_TIMING .*?accepted_creations=(\d+) accepted_breaks=(\d+)", log)
    created, broken = map(int, match[-1]) if match else (None, None)
    events = directory / "events.dat"
    event_count = sum(1 for line in events.read_text().splitlines() if line and not line.startswith("#")) if events.exists() else 0
    event_bytes = events.stat().st_size if events.exists() else 0
    return {"wall_seconds": wall, "steps_per_second": None, "accepted_creations": created,
            "accepted_breaks": broken, "logged_events": event_count,
            "event_log_bytes": event_bytes,
            "bytes_per_event": event_bytes / event_count if event_count else 0.0}


def main():
    root = Path(sys.argv[1])
    steps = int(next(line.split("=", 1)[1] for line in (root / "provenance.txt").read_text().splitlines()
                     if line.startswith("bench_steps=")))
    rows = []
    for label in ("off", "on"):
        for directory in sorted((root / label).glob("rep*")):
            row = {"mode": label, "rep": directory.name, **metric(directory)}
            row["steps_per_second"] = steps / row["wall_seconds"]
            rows.append(row)
    medians = {}
    for label in ("off", "on"):
        values = [row["wall_seconds"] for row in rows if row["mode"] == label]
        medians[label] = {"wall_seconds_median": statistics.median(values),
                          "wall_seconds_min": min(values), "wall_seconds_max": max(values),
                          "steps_per_second_median": steps / statistics.median(values)}
    off = medians["off"]["wall_seconds_median"]
    on = medians["on"]["wall_seconds_median"]
    summary = {"steps": steps, "runs": rows, "medians": medians,
               "slowdown_percent": 100.0 * (on - off) / off,
               "performance_gate": ("PASS" if (on-off)/off <= 0.02 else
                                     "ACCEPTABLE" if (on-off)/off <= 0.05 else "FAIL / redesign")}
    with (root / "benchmark.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    (root / "benchmark.summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
