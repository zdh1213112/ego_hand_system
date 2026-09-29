#!/usr/bin/env python3
"""Evaluate a leave-one-view target-camera correction and final fusion.

The default report is calibration-based: it compares raw/optimized target
2-D observations to a leave-out 3-D reference and reports final 3-D temporal
motion.  If a JSONL file with ``sync_index``/``hands``/``joints_base_m`` is
provided via ``--ground-truth``, the same report additionally contains true
absolute 3-D joint errors.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from camera_models import project_points
from ego_data.calibration import CameraCalibration


def _load_jsonl(path: Path) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                rows[int(row["sync_index"])] = row
    return rows


def _points_2d(value: Any) -> np.ndarray:
    points = np.asarray(value, dtype=np.float64)
    return points if points.shape == (21, 2) else np.full((21, 2), np.nan)


def _points_3d(value: Any) -> np.ndarray:
    points = np.asarray(value, dtype=np.float64)
    return points if points.shape == (21, 3) else np.full((21, 3), np.nan)


def _project(calibration: CameraCalibration, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rotation = calibration.T_base_camera[:3, :3]
    center = calibration.T_base_camera[:3, 3]
    points_camera = (rotation.T @ (points - center).T).T
    projected, valid = project_points(calibration, points_camera)
    projected = np.asarray(projected, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    projected[~valid] = np.nan
    return projected, valid


def _select_hand(record: dict[str, Any], side: int, projected: np.ndarray, valid: np.ndarray) -> dict[str, Any] | None:
    best = None
    best_key = (float("inf"), float("inf"))
    for hand in record.get("hands", []):
        if hand.get("is_right") is None or int(hand["is_right"]) != side:
            continue
        observed = _points_2d(hand.get("joints_2d", []))
        usable = valid & np.isfinite(observed).all(axis=1) & np.isfinite(projected).all(axis=1)
        if int(np.count_nonzero(usable)) < 12:
            continue
        median = float(np.median(np.linalg.norm(observed[usable] - projected[usable], axis=1)))
        key = (median, -float(hand.get("confidence", 0.0)))
        if key < best_key:
            best_key = key
            best = hand
    return best


def _stats(values: list[float], *, scale: float = 1.0) -> dict[str, Any]:
    data = np.asarray(values, dtype=np.float64)
    data = data[np.isfinite(data)]
    if not len(data):
        return {"sample_count": 0}
    data = data * scale
    return {
        "sample_count": int(len(data)),
        "p50": float(np.percentile(data, 50)),
        "p95": float(np.percentile(data, 95)),
        "p99": float(np.percentile(data, 99)),
        "max": float(np.max(data)),
        "over_50": int(np.count_nonzero(data > 50.0)),
        "over_80": int(np.count_nonzero(data > 80.0)),
        "over_100": int(np.count_nonzero(data > 100.0)),
    }


def _target_2d_report(
    raw_rows: dict[int, dict[str, Any]],
    optimized_rows: dict[int, dict[str, Any]],
    reference_rows: dict[int, dict[str, Any]],
    calibration: CameraCalibration,
) -> dict[str, Any]:
    before: list[float] = []
    after: list[float] = []
    changed_before: list[float] = []
    changed_after: list[float] = []
    shifts: list[float] = []
    changed_joints = 0
    matched_hands = 0
    for sync_index in sorted(set(raw_rows) & set(optimized_rows) & set(reference_rows)):
        raw_record = raw_rows[sync_index]
        optimized_record = optimized_rows[sync_index]
        for fused_hand in reference_rows[sync_index].get("hands", []):
            side = int(fused_hand.get("side", -1))
            if side not in (0, 1):
                continue
            reference_points = _points_3d(fused_hand.get("joints_base_m", []))
            projected, valid = _project(calibration, reference_points)
            raw_hand = _select_hand(raw_record, side, projected, valid)
            optimized_hand = _select_hand(optimized_record, side, projected, valid)
            if raw_hand is None or optimized_hand is None:
                continue
            matched_hands += 1
            raw = _points_2d(raw_hand.get("joints_2d", []))
            optimized = _points_2d(optimized_hand.get("joints_2d", []))
            usable = valid & np.isfinite(projected).all(axis=1)
            usable &= np.isfinite(raw).all(axis=1) & np.isfinite(optimized).all(axis=1)
            if not np.any(usable):
                continue
            before_values = np.linalg.norm(raw[usable] - projected[usable], axis=1)
            after_values = np.linalg.norm(optimized[usable] - projected[usable], axis=1)
            shift_values = np.linalg.norm(optimized[usable] - raw[usable], axis=1)
            before.extend(before_values.tolist())
            after.extend(after_values.tolist())
            shifts.extend(shift_values.tolist())
            changed = shift_values > 1e-6
            changed_joints += int(np.count_nonzero(changed))
            changed_before.extend(before_values[changed].tolist())
            changed_after.extend(after_values[changed].tolist())
    reduction = None
    if before and after and np.median(before) > 1e-9:
        reduction = float(100.0 * (np.median(before) - np.median(after)) / np.median(before))
    return {
        "matched_hand_count": matched_hands,
        "changed_joint_count": changed_joints,
        "all_joints_residual_before_px": _stats(before),
        "all_joints_residual_after_px": _stats(after),
        "changed_joints_residual_before_px": _stats(changed_before),
        "changed_joints_residual_after_px": _stats(changed_after),
        "target_2d_shift_px": _stats(shifts),
        "median_residual_reduction_percent": reduction,
    }


def _motion_report(path: Path) -> dict[str, Any]:
    rows = _load_jsonl(path)
    by_side: dict[int, list[tuple[int, np.ndarray]]] = {0: [], 1: []}
    for sync_index, row in rows.items():
        for hand in row.get("hands", []):
            side = int(hand.get("side", -1))
            points = _points_3d(hand.get("joints_base_m", []))
            if side in by_side:
                by_side[side].append((sync_index, points))
    output: dict[str, Any] = {}
    for side, values in by_side.items():
        values.sort(key=lambda item: item[0])
        steps: list[float] = []
        accelerations: list[float] = []
        for index in range(1, len(values)):
            if values[index][0] - values[index - 1][0] != 1:
                continue
            first, second = values[index - 1][1], values[index][1]
            finite = np.isfinite(first).all(axis=1) & np.isfinite(second).all(axis=1)
            steps.extend(np.linalg.norm(second[finite] - first[finite], axis=1).tolist())
        for index in range(1, len(values) - 1):
            if (
                values[index][0] - values[index - 1][0] != 1
                or values[index + 1][0] - values[index][0] != 1
            ):
                continue
            first, second, third = values[index - 1][1], values[index][1], values[index + 1][1]
            finite = (
                np.isfinite(first).all(axis=1)
                & np.isfinite(second).all(axis=1)
                & np.isfinite(third).all(axis=1)
            )
            accelerations.extend(
                np.linalg.norm(third[finite] - 2.0 * second[finite] + first[finite], axis=1).tolist()
            )
        output[str(side)] = {
            "step_mm": _stats(steps, scale=1000.0),
            "acceleration_mm": _stats(accelerations, scale=1000.0),
        }
    return output


def _ray_geometry_by_joint(path: Path) -> list[dict[str, Any]]:
    conditions: list[list[float]] = [[] for _ in range(21)]
    angles: list[list[float]] = [[] for _ in range(21)]
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            for hand in row.get("hands", []):
                quality = hand.get("quality", {})
                condition_values = quality.get("ray_condition_numbers", [])
                angle_values = quality.get("ray_min_angles_degrees", [])
                if len(condition_values) == 21:
                    for joint, value in enumerate(condition_values):
                        if isinstance(value, (int, float)) and np.isfinite(value):
                            conditions[joint].append(float(value))
                if len(angle_values) == 21:
                    for joint, value in enumerate(angle_values):
                        if isinstance(value, (int, float)) and np.isfinite(value):
                            angles[joint].append(float(value))
    report = []
    for joint in range(21):
        condition = conditions[joint]
        angle = angles[joint]
        report.append({
            "joint": joint,
            "sample_count": len(condition),
            "condition_number_p50": float(np.percentile(condition, 50)) if condition else None,
            "condition_number_p95": float(np.percentile(condition, 95)) if condition else None,
            "condition_number_p99": float(np.percentile(condition, 99)) if condition else None,
            "condition_ge_1000": int(sum(value >= 1000.0 for value in condition)),
            "min_ray_angle_p50_degrees": float(np.percentile(angle, 50)) if angle else None,
        })
    return report


def _ground_truth_report(
    optimized_rows: dict[int, dict[str, Any]], ground_truth_rows: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    errors: list[float] = []
    for sync_index in sorted(set(optimized_rows) & set(ground_truth_rows)):
        estimate_by_side = {
            int(hand.get("side", -1)): _points_3d(hand.get("joints_base_m", []))
            for hand in optimized_rows[sync_index].get("hands", [])
        }
        truth_by_side = {
            int(hand.get("side", -1)): _points_3d(hand.get("joints_base_m", []))
            for hand in ground_truth_rows[sync_index].get("hands", [])
        }
        for side in (0, 1):
            if side not in estimate_by_side or side not in truth_by_side:
                continue
            finite = np.isfinite(estimate_by_side[side]).all(axis=1) & np.isfinite(truth_by_side[side]).all(axis=1)
            errors.extend(np.linalg.norm(estimate_by_side[side][finite] - truth_by_side[side][finite], axis=1).tolist())
    return {"available": bool(errors), "absolute_joint_error_mm": _stats(errors, scale=1000.0) if errors else None}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-camera", default="camera2")
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--raw-predictions", required=True, type=Path)
    parser.add_argument("--optimized-predictions", required=True, type=Path)
    parser.add_argument("--leaveout-fusion", required=True, type=Path)
    parser.add_argument("--original-fusion", required=True, type=Path)
    parser.add_argument("--optimized-fusion", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--ground-truth", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    calibration = CameraCalibration.load(args.dataset / "calibration" / f"{args.target_camera}.json")
    raw_rows = _load_jsonl(args.raw_predictions / args.target_camera / "predictions.jsonl")
    optimized_rows = _load_jsonl(args.optimized_predictions / args.target_camera / "predictions.jsonl")
    reference_rows = _load_jsonl(args.leaveout_fusion / "accepted.jsonl")
    original_summary = json.loads((args.original_fusion / "summary.json").read_text(encoding="utf-8"))
    optimized_summary = json.loads((args.optimized_fusion / "summary.json").read_text(encoding="utf-8"))
    report: dict[str, Any] = {
        "schema_version": 1,
        "target_camera": args.target_camera,
        "calibration_based_target_2d": _target_2d_report(
            raw_rows, optimized_rows, reference_rows, calibration,
        ),
        "fusion_summary_comparison": {
            "original": {
                key: original_summary.get(key)
                for key in (
                    "accepted_frame_count", "rejected_frame_count", "accepted_rate",
                    "reprojection_median_px", "selected_view_count_median",
                    "valid_joint_inlier_view_count_median",
                )
            },
            "optimized": {
                key: optimized_summary.get(key)
                for key in (
                    "accepted_frame_count", "rejected_frame_count", "accepted_rate",
                    "reprojection_median_px", "selected_view_count_median",
                    "valid_joint_inlier_view_count_median",
                )
            },
        },
        "final_3d_motion": {
            "original": _motion_report(args.original_fusion / "accepted.jsonl"),
            "optimized": _motion_report(args.optimized_fusion / "accepted.jsonl"),
        },
        "optimized_ray_geometry_by_joint": _ray_geometry_by_joint(
            args.optimized_fusion / "accepted.jsonl"
        ),
        "ground_truth": (
            _ground_truth_report(
                _load_jsonl(args.optimized_fusion / "accepted.jsonl"),
                _load_jsonl(args.ground_truth),
            )
            if args.ground_truth else {"available": False, "absolute_joint_error_mm": None}
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
