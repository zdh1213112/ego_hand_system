#!/usr/bin/env python3
"""Derive each rigidly attached camera's world pose from the common IMU trajectory."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from fuse_basalt_hand_trajectory import (load_basalt_trajectory, pose_from_basalt_json,
                                        quaternion_xyzw_from_matrix)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    result = args.result.resolve()
    calibration = json.loads((result / "dataset/calibration.json").read_text())["value0"]
    trajectory = load_basalt_trajectory(result / "trajectory.csv")
    directory = result / "camera_poses"
    directory.mkdir(exist_ok=True)
    first_timestamp = min(trajectory)
    first_rows = []
    for index, entry in enumerate(calibration["T_imu_cam"]):
        transform = pose_from_basalt_json(entry)
        destination = directory / f"cam_{index}_world_poses.csv"
        if destination.exists():
            raise FileExistsError(destination)
        with destination.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("timestamp_ns", "time_s", "x_m", "y_m", "z_m", "qw", "qx", "qy", "qz"))
            for timestamp, pose in sorted(trajectory.items()):
                camera = pose @ transform
                quaternion = quaternion_xyzw_from_matrix(camera[:3, :3])
                writer.writerow((timestamp, (timestamp - first_timestamp) * 1e-9, *camera[:3, 3],
                                 quaternion[3], *quaternion[:3]))
                if timestamp == first_timestamp:
                    first_rows.append({"camera": f"cam_{index}", "position_world_m": camera[:3, 3].tolist(),
                                       "forward_axis_world": camera[:3, 2].tolist()})
    summary = {"formula": "T_world_cam_i(t) = T_world_imu(t) @ T_imu_cam_i",
               "pose_count_per_camera": len(trajectory), "camera_count": len(first_rows),
               "meaning": "Derived camera poses from one jointly estimated rigid-body motion; not independent VIO runs",
               "initial_camera_poses": first_rows}
    (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
