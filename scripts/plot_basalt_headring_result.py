#!/usr/bin/env python3
"""Create standalone trajectory/map figures and a clearly range-limited PLY preview."""
from __future__ import annotations
import argparse
import csv
import json
import os
from pathlib import Path
import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    result = args.result.resolve()
    os.environ.setdefault("MPLCONFIGDIR", str(result / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = list(csv.DictReader((result / "headring_poses.csv").open()))
    positions = np.asarray([[float(row[k]) for k in ("x_m", "y_m", "z_m")] for row in rows])
    times = np.asarray([float(row["time_s"]) for row in rows])
    relative = positions - positions[0]
    angles = np.asarray([[float(row[k]) for k in ("roll_deg", "pitch_deg", "yaw_deg")] for row in rows])
    angles = np.rad2deg(np.unwrap(np.deg2rad(angles), axis=0))
    angles -= angles[0]
    plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": .25})
    fig = plt.figure(figsize=(12, 9), constrained_layout=True)
    ax = fig.add_subplot(221, projection="3d")
    ax.plot(*relative.T, color="#1967b3", linewidth=1.4)
    ax.scatter(*relative[0], c="#218738", label="start")
    ax.scatter(*relative[-1], c="#cc2929", label="end")
    ax.set(xlabel="X (m)", ylabel="Y (m)", zlabel="Z (m)", title="Headring reference trajectory (cam_0)")
    ax.legend()
    ax = fig.add_subplot(222)
    ax.scatter(relative[:, 0], relative[:, 1], c=times, s=6, cmap="viridis")
    ax.set(xlabel="X (m)", ylabel="Y (m)", title="Top view; color follows time", aspect="equal")
    ax = fig.add_subplot(223)
    for i, name in enumerate(("X", "Y", "Z")):
        ax.plot(times, relative[:, i], label=name)
    ax.set(xlabel="Time (s)", ylabel="Displacement from start (m)", title="Position")
    ax.legend()
    ax = fig.add_subplot(224)
    for i, name in enumerate(("roll", "pitch", "yaw")):
        ax.plot(times, angles[:, i], label=name)
    ax.set(xlabel="Time (s)", ylabel="Unwrapped Euler-angle change (deg)", title="Orientation")
    ax.legend()
    fig.suptitle("Four-camera + IMU VIO | local gravity-aligned world | no external ground truth", fontsize=13)
    fig.savefig(result / "headring_trajectory.png", dpi=170)
    plt.close(fig)

    map_dir = result / "map"
    landmarks = np.loadtxt(map_dir / "sparse_landmarks.csv", delimiter=",", skiprows=1)
    points = landmarks[:, 1:4]
    keep = np.linalg.norm(points - positions.mean(axis=0), axis=1) <= 10.0
    near = points[keep]
    with (map_dir / "sparse_map_10m.ply").open("w") as stream:
        stream.write("ply\nformat ascii 1.0\ncomment display subset within 10 m of mean VIO cam_0 position\n")
        stream.write(f"element vertex {len(near)}\nproperty double x\nproperty double y\nproperty double z\nend_header\n")
        for point in near:
            stream.write(" ".join(format(v, ".16g") for v in point) + "\n")
    optimized = list(csv.DictReader((map_dir / "keyframe_headring_poses.csv").open()))
    optimized = np.asarray([[float(row[k]) for k in ("x_m", "y_m", "z_m")] for row in optimized])
    fig = plt.figure(figsize=(13, 6), constrained_layout=True)
    ax = fig.add_subplot(121, projection="3d")
    scatter = ax.scatter(*near.T, c=near[:, 2], cmap="viridis", s=3, alpha=.5)
    ax.plot(*optimized.T, color="#d13522", linewidth=2, label="optimized keyframes")
    ax.set(xlabel="World X (m)", ylabel="World Y (m)", zlabel="World Z (m)", title="NFR sparse map (3D)")
    ax.view_init(elev=25, azim=-55)
    ax.legend()
    ax = fig.add_subplot(122)
    ax.scatter(near[:, 0], near[:, 1], c=near[:, 2], cmap="viridis", s=4, alpha=.4)
    ax.plot(optimized[:, 0], optimized[:, 1], color="#d13522", linewidth=2, label="optimized keyframes")
    ax.set(xlabel="World X (m)", ylabel="World Y (m)", title="Map top view", aspect="equal")
    ax.legend()
    fig.colorbar(scatter, ax=fig.axes[:2], label="World Z (m)", shrink=.7)
    fig.suptitle(f"{len(points):,} unique landmarks; {len(near):,} shown within 10 m | sparse geometry, no surface mesh", fontsize=12)
    fig.savefig(result / "sparse_map_preview.png", dpi=170)
    plt.close(fig)
    summary = {"unique_landmarks": len(points), "preview_points_within_10m": int(keep.sum()),
               "preview_excluded_far_points": int((~keep).sum()),
               "filter": "display subset only; 10 m Euclidean radius around mean VIO cam_0; original map retained"}
    (map_dir / "preview_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
