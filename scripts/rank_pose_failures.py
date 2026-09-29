#!/usr/bin/env python3
"""Rank the most suspicious final 3-D hand-pose frames.

This is an inspection aid, not a new filter.  It compares two final-fusion
JSONL files frame-by-frame and reports large per-joint velocity/acceleration
events together with the multiview quality and per-camera reprojection
diagnostics saved by the fusion stage.  The report is intended to select a
small set of frames for human cam2 2-D verification before changing the
fusion algorithm.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _load_jsonl(path: Path) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                rows[int(row["sync_index"])] = row
    return rows


def _points(value: Any) -> np.ndarray:
    points = np.asarray(value, dtype=np.float64)
    if points.shape != (21, 3):
        return np.full((21, 3), np.nan, dtype=np.float64)
    return points


def _hands_by_side(row: dict[str, Any]) -> dict[int, dict[str, Any]]:
    output: dict[int, dict[str, Any]] = {}
    for hand in row.get("hands", []):
        side = int(hand.get("side", -1))
        if side in (0, 1) and side not in output:
            output[side] = hand
    return output


def _camera_quality(hand: dict[str, Any]) -> dict[str, Any]:
    quality = hand.get("quality", {})
    diagnostics = hand.get("consensus_view_diagnostics", {})
    cameras: dict[str, dict[str, Any]] = {}
    for camera_id, value in diagnostics.items():
        if not isinstance(value, dict):
            continue
        cameras[str(camera_id)] = {
            "median_px": value.get("median_px"),
            "p95_px": value.get("p95_px"),
            "inlier_joint_count": value.get("inlier_joint_count"),
            "sam3_bad": value.get("sam3_bad"),
            "sam3_joint_support": value.get("sam3_joint_support"),
        }
    condition = np.asarray(quality.get("ray_condition_numbers", []), dtype=np.float64)
    angles = np.asarray(quality.get("ray_min_angles_degrees", []), dtype=np.float64)
    condition = condition[np.isfinite(condition)]
    angles = angles[np.isfinite(angles)]
    return {
        "camera_ids": list(hand.get("camera_ids", [])),
        "active_camera_count": len(hand.get("camera_ids", [])),
        "reprojection_median_px": quality.get("multiview_reprojection_median_px"),
        "reprojection_p95_px": quality.get("multiview_reprojection_p95_px"),
        "min_inlier_view_count": min(hand.get("inlier_view_counts", [0]), default=0),
        "condition_number_max": float(np.max(condition)) if len(condition) else None,
        "condition_number_p95": float(np.percentile(condition, 95)) if len(condition) else None,
        "min_ray_angle_min_degrees": float(np.min(angles)) if len(angles) else None,
        "consensus_rejected_camera_ids": list(hand.get("consensus_rejected_camera_ids", [])),
        "cameras": cameras,
    }


def _events(rows: dict[int, dict[str, Any]]) -> dict[tuple[int, int, int], dict[str, Any]]:
    by_side: dict[int, dict[int, tuple[np.ndarray, dict[str, Any]]]] = {0: {}, 1: {}}
    for sync_index, row in rows.items():
        for side, hand in _hands_by_side(row).items():
            by_side[side][sync_index] = (_points(hand.get("joints_base_m", [])), hand)

    events: dict[tuple[int, int, int], dict[str, Any]] = {}
    for side, tracks in by_side.items():
        indices = sorted(tracks)
        for position in range(1, len(indices) - 1):
            previous_index, current_index, next_index = indices[position - 1:position + 2]
            if current_index - previous_index != 1 or next_index - current_index != 1:
                continue
            previous = tracks[previous_index][0]
            current = tracks[current_index][0]
            following = tracks[next_index][0]
            finite = (
                np.isfinite(previous).all(axis=1)
                & np.isfinite(current).all(axis=1)
                & np.isfinite(following).all(axis=1)
            )
            step_before = np.linalg.norm(current - previous, axis=1) * 1000.0
            step_after = np.linalg.norm(following - current, axis=1) * 1000.0
            acceleration = np.linalg.norm(following - 2.0 * current + previous, axis=1) * 1000.0
            hand = tracks[current_index][1]
            quality = _camera_quality(hand)
            for joint in np.flatnonzero(finite):
                events[(current_index, side, int(joint))] = {
                    "sync_index": current_index,
                    "side": side,
                    "joint": int(joint),
                    "step_before_mm": float(step_before[joint]),
                    "step_after_mm": float(step_after[joint]),
                    "acceleration_mm": float(acceleration[joint]),
                    "quality": quality,
                }
    return events


def _rank(events: dict[tuple[int, int, int], dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    values = list(events.values())
    values.sort(
        key=lambda item: (
            float(item.get("acceleration_mm", 0.0)),
            max(float(item.get("step_before_mm", 0.0)), float(item.get("step_after_mm", 0.0))),
        ),
        reverse=True,
    )
    return values[: max(int(limit), 0)]


def _merge_comparison(
    baseline: dict[tuple[int, int, int], dict[str, Any]],
    assisted: dict[tuple[int, int, int], dict[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    ranked = _rank(assisted, max(limit * 4, limit))
    output: list[dict[str, Any]] = []
    for event in ranked:
        key = (event["sync_index"], event["side"], event["joint"])
        row = dict(event)
        row["baseline"] = baseline.get(key)
        output.append(row)
        if len(output) >= limit:
            break
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--assisted", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--top", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline_rows = _load_jsonl(args.baseline)
    assisted_rows = _load_jsonl(args.assisted)
    baseline_events = _events(baseline_rows)
    assisted_events = _events(assisted_rows)
    report = {
        "baseline": str(args.baseline),
        "assisted": str(args.assisted),
        "frame_counts": {
            "baseline_rows": len(baseline_rows),
            "assisted_rows": len(assisted_rows),
            "common_rows": len(set(baseline_rows) & set(assisted_rows)),
        },
        "event_counts": {
            "baseline": len(baseline_events),
            "assisted": len(assisted_events),
        },
        "top_assisted_acceleration_events": _merge_comparison(
            baseline_events, assisted_events, args.top,
        ),
        "top_baseline_acceleration_events": _merge_comparison(
            assisted_events, baseline_events, args.top,
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "frame_counts": report["frame_counts"],
        "event_counts": report["event_counts"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
