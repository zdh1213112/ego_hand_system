#!/usr/bin/env python3
"""Compare rectified WiLoR predictions with Nokov's raw-camera marker projection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def records(path: Path) -> dict[tuple[int, str], dict]:
    return {
        (int(row["sync_index"]), str(row["camera"])): row
        for row in (json.loads(line) for line in path.open())
    }


def mapped_points(points: np.ndarray, map_x: np.ndarray, map_y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    height, width = map_x.shape
    valid = (
        np.isfinite(points).all(axis=1)
        & (points[:, 0] >= 0) & (points[:, 0] < width - 1)
        & (points[:, 1] >= 0) & (points[:, 1] < height - 1)
    )
    coordinates = np.where(np.isfinite(points), points, 0)
    x0 = np.clip(np.floor(coordinates[:, 0]).astype(np.int32), 0, width - 2)
    y0 = np.clip(np.floor(coordinates[:, 1]).astype(np.int32), 0, height - 2)
    dx = np.clip(coordinates[:, 0] - x0, 0, 1)
    dy = np.clip(coordinates[:, 1] - y0, 0, 1)
    def sample(image: np.ndarray) -> np.ndarray:
        return (
            image[y0, x0] * (1 - dx) * (1 - dy)
            + image[y0, x0 + 1] * dx * (1 - dy)
            + image[y0 + 1, x0] * (1 - dx) * dy
            + image[y0 + 1, x0 + 1] * dx * dy
        )
    x, y = sample(map_x), sample(map_y)
    return np.stack((x, y), axis=-1), valid


def summary(errors: list[float]) -> dict:
    values = np.asarray(errors, dtype=np.float64)
    return {
        "points": int(values.size),
        "mean_px": float(values.mean()),
        "median_px": float(np.median(values)),
        "p95_px": float(np.percentile(values, 95)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old", type=Path, required=True)
    parser.add_argument("--new", type=Path, required=True)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--rectification", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    old, new = records(args.old), records(args.new)
    observations = {
        int(row["sync_index"]): row
        for row in (json.loads(line) for line in args.observations.open())
    }
    with np.load(args.rectification) as archive:
        maps = {name: archive[name] for name in (
            "map_left_x", "map_left_y", "map_right_x", "map_right_y"
        )}
    result: dict[str, dict] = {}
    for camera, prefix in (("camera0", "left"), ("camera1", "right")):
        errors = {"label": [], "before": [], "after": []}
        improved = 0
        frame_count = 0
        for sync, cam in sorted(old):
            if cam != camera or (sync, cam) not in new or sync not in observations:
                continue
            marker = np.asarray(
                observations[sync]["hands"][0]["views"][camera]["joints_2d"],
                dtype=object,
            )
            marker_valid = np.array([
                item is not None and len(item) == 2 and all(v is not None for v in item)
                for item in marker
            ])
            marker_xy = np.array([
                item if valid else [np.nan, np.nan]
                for item, valid in zip(marker, marker_valid)
            ], dtype=np.float64)
            if marker_xy.shape != (21, 2):
                raise ValueError(f"Unexpected marker shape at sync={sync}")
            rect_points = {
                "label": np.asarray(old[(sync, cam)]["gt_joints_2d"], dtype=np.float64),
                "before": np.asarray(old[(sync, cam)]["pred_joints_2d"], dtype=np.float64),
                "after": np.asarray(new[(sync, cam)]["pred_joints_2d"], dtype=np.float64),
            }
            mapped = {
                name: mapped_points(points, maps[f"map_{prefix}_x"], maps[f"map_{prefix}_y"])
                for name, points in rect_points.items()
            }
            valid = marker_valid.copy()
            for _, point_valid in mapped.values():
                valid &= point_valid
            if not valid.any():
                continue
            frame_count += 1
            distances = {
                name: np.linalg.norm(point_xy[valid] - marker_xy[valid], axis=1)
                for name, (point_xy, _) in mapped.items()
            }
            for name, values in distances.items():
                errors[name].extend(values.tolist())
            improved += int(np.count_nonzero(distances["after"] < distances["before"]))
        result[camera] = {
            "frames": frame_count,
            "improved_points": improved,
            **{name: summary(values) for name, values in errors.items()},
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
