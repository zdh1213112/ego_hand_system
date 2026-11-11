#!/usr/bin/env python3
"""Export nearest IMU associations; retain native VIO times and output rate."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from ego_data.time_sync import nearest_indices, sampling_summary


def associate(trajectory: Path, imu: Path, output: Path, max_delta_ns=2_000_000):
    # Both inputs are already on the IMU time axis. Never apply camera delay here.
    with trajectory.open(newline="") as stream:
        rows = list(csv.reader(stream))
    if not rows or not rows[0][0].startswith("#timestamp"):
        raise ValueError("expected native Basalt EuRoC CSV with integer nanosecond timestamps")
    times = [int(row[0]) for row in rows[1:] if row]
    with imu.open(newline="") as stream:
        it = [int(row["timestamp_ns"]) for row in csv.DictReader(stream)]
    matches = nearest_indices(times, it, max_delta_ns)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "vio_imu_association.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["vio_index", "vio_timestamp_ns", "nearest_imu_index",
                         "nearest_imu_timestamp_ns", "imu_minus_vio_ns", "matched"])
        for index, (timestamp, match) in enumerate(zip(times, matches)):
            writer.writerow([index, timestamp, match if match is not None else "",
                             it[match] if match is not None else "",
                             it[match]-timestamp if match is not None else "", int(match is not None)])
    deltas = [abs(it[m]-t) for t, m in zip(times, matches) if m is not None]
    exact_times = set(it)
    summary = {"trajectory": str(trajectory.resolve()), "imu": str(imu.resolve()),
               "vio": sampling_summary(times), "imu_sampling": sampling_summary(it),
               "matched_count": len(deltas), "unmatched_count": len(times)-len(deltas),
               "exact_imu_timestamp_count": sum(t in exact_times for t in times),
               "max_allowed_delta_ns": max_delta_ns,
               "max_matched_abs_delta_ns": max(deltas) if deltas else None,
               "association_only": True, "trajectory_retimed": False,
               "timestamp_meaning": "native image state epoch on IMU time axis; not host output time",
               "output_rate_meaning": "native visual update rate; nearest matching does not produce IMU-rate VIO"}
    (output / "vio_imu_timing.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--imu", type=Path, required=True, help="adapted IMU CSV: timestamp_ns column")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-delta-us", type=int, default=2000)
    args = parser.parse_args()
    print(json.dumps(associate(args.trajectory, args.imu, args.output, args.max_delta_us * 1000), indent=2))
