#!/usr/bin/env python3
"""Fuse six-camera dual-hypothesis WiLoR observations with native-DS RANSAC."""

from __future__ import annotations

import argparse
from collections import Counter
import itertools
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from camera_models import project_points  # noqa: E402
from camera_models.double_sphere import unproject  # noqa: E402
from ego_data.calibration import CameraCalibration  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--min-views", type=int, default=3)
    parser.add_argument("--ransac-threshold-px", type=float, default=20.0)
    parser.add_argument("--max-reprojection-median-px", type=float, default=15.0)
    parser.add_argument("--max-reprojection-p95-px", type=float, default=40.0)
    parser.add_argument("--min-valid-joints", type=int, default=12)
    parser.add_argument("--ambiguity-margin", type=float, default=1.0)
    parser.add_argument(
        "--baseline-cameras", nargs=2, default=("camera2", "camera3"),
        metavar=("LEFT", "RIGHT"),
        help="stereo pair used for the cross-view comparison",
    )
    parser.add_argument("--max-frames", type=int, default=0)
    return parser.parse_args()


def _load_jsonl(path: Path) -> dict[int, dict[str, Any]]:
    result = {}
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            result[int(row["sync_index"])] = row
    return result


def _groups(row: dict[str, Any]) -> dict[int, dict[int, dict[str, Any]]]:
    groups: dict[int, dict[int, dict[str, Any]]] = {}
    for hand in row.get("hands", []):
        groups.setdefault(int(hand["detection_index"]), {})[int(hand["is_right"])] = hand
    return groups


def _ray_in_base(camera: CameraCalibration, pixel: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    ray_camera, valid = unproject(camera, np.asarray(pixel, dtype=np.float64)[None])
    if not bool(valid[0]):
        return None
    rotation = camera.T_base_camera[:3, :3]
    center = camera.T_base_camera[:3, 3]
    direction = rotation @ ray_camera[0]
    direction /= np.linalg.norm(direction)
    return center, direction


def _intersect_rays(rays: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray | None:
    matrix = np.zeros((3, 3), dtype=np.float64)
    target = np.zeros(3, dtype=np.float64)
    identity = np.eye(3)
    for center, direction in rays:
        projector = identity - np.outer(direction, direction)
        matrix += projector
        target += projector @ center
    if np.linalg.cond(matrix) > 1e8:
        return None
    return np.linalg.solve(matrix, target)


def _errors(
    point_base: np.ndarray,
    observations: list[tuple[str, CameraCalibration, np.ndarray]],
) -> np.ndarray:
    result = np.full(len(observations), np.inf, dtype=np.float64)
    for index, (_, camera, pixel) in enumerate(observations):
        rotation = camera.T_base_camera[:3, :3]
        center = camera.T_base_camera[:3, 3]
        point_camera = rotation.T @ (point_base - center)
        ray = _ray_in_base(camera, pixel)
        if ray is None or np.dot(point_base - center, ray[1]) <= 0:
            continue
        projected, valid = project_points(camera, point_camera[None])
        if bool(valid[0]) and np.all(np.isfinite(projected[0])):
            result[index] = np.linalg.norm(projected[0] - pixel)
    return result


def triangulate_ransac(
    observations: list[tuple[str, CameraCalibration, np.ndarray]],
    threshold_px: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rays = [_ray_in_base(camera, pixel) for _, camera, pixel in observations]
    best: tuple[int, float, np.ndarray, np.ndarray] | None = None
    for first, second in itertools.combinations(range(len(observations)), 2):
        if rays[first] is None or rays[second] is None:
            continue
        point = _intersect_rays([rays[first], rays[second]])
        if point is None:
            continue
        errors = _errors(point, observations)
        inliers = np.isfinite(errors) & (errors <= threshold_px)
        count = int(inliers.sum())
        score = float(np.median(errors[inliers])) if count else float("inf")
        candidate = (count, -score, point, inliers)
        if best is None or candidate[:2] > best[:2]:
            best = candidate
    if best is None or best[0] < 2:
        return np.full(3, np.nan), np.full(len(observations), np.inf), np.zeros(len(observations), bool)
    inlier_rays = [rays[index] for index in np.flatnonzero(best[3]) if rays[index] is not None]
    point = _intersect_rays(inlier_rays)
    if point is None:
        point = best[2]
    errors = _errors(point, observations)
    inliers = np.isfinite(errors) & (errors <= threshold_px)
    return point, errors, inliers


def _triangulate_hand(
    side: int,
    selected: dict[str, dict[str, Any]],
    calibrations: dict[str, CameraCalibration],
    threshold_px: float,
    min_views: int,
) -> dict[str, Any]:
    cameras = tuple(selected)
    points = np.full((21, 3), np.nan, dtype=np.float64)
    errors = np.full((21, len(cameras)), np.inf, dtype=np.float64)
    inlier_mask = np.zeros((21, len(cameras)), dtype=bool)
    inlier_counts = np.zeros(21, dtype=np.int32)
    for joint in range(21):
        observations = [
            (camera, calibrations[camera], np.asarray(selected[camera]["joints_2d"][joint], dtype=np.float64))
            for camera in cameras
        ]
        point, joint_errors, inliers = triangulate_ransac(observations, threshold_px)
        errors[joint] = joint_errors
        inlier_mask[joint] = inliers
        inlier_counts[joint] = int(inliers.sum())
        if inlier_counts[joint] >= min_views:
            points[joint] = point
    finite_errors = errors[np.isfinite(errors)]
    inlier_errors = errors[inlier_mask & np.isfinite(errors)]
    valid_joints = int(np.count_nonzero(np.isfinite(points).all(axis=1)))
    median = float(np.median(inlier_errors)) if len(inlier_errors) else float("inf")
    p95 = float(np.percentile(inlier_errors, 95)) if len(inlier_errors) else float("inf")
    return {
        "side": side,
        "cameras": list(cameras),
        "points": points,
        "errors": errors,
        "inlier_mask": inlier_mask,
        "inlier_counts": inlier_counts,
        "valid_joints": valid_joints,
        "median_px": median,
        "p95_px": p95,
        "all_view_median_px": (
            float(np.median(finite_errors)) if len(finite_errors) else float("inf")
        ),
        "all_view_p95_px": (
            float(np.percentile(finite_errors, 95)) if len(finite_errors) else float("inf")
        ),
    }


def _baseline_comparison(
    result: dict[str, Any], selected: dict[str, dict[str, Any]],
    calibrations: dict[str, CameraCalibration], baseline_cameras: tuple[str, str],
) -> dict[str, Any]:
    if any(camera not in selected for camera in baseline_cameras):
        return {"available_joint_count": 0}
    stereo_points = np.full((21, 3), np.nan, dtype=np.float64)
    cross_view_errors = []
    differences_mm = []
    for joint in range(21):
        observations = [
            (
                camera, calibrations[camera],
                np.asarray(selected[camera]["joints_2d"][joint], dtype=np.float64),
            )
            for camera in baseline_cameras
        ]
        rays = [
            _ray_in_base(calibration, pixel)
            for _, calibration, pixel in observations
        ]
        if any(ray is None for ray in rays):
            continue
        point = _intersect_rays([ray for ray in rays if ray is not None])
        if point is None:
            continue
        stereo_points[joint] = point
        all_observations = [
            (
                camera, calibrations[camera],
                np.asarray(hand["joints_2d"][joint], dtype=np.float64),
            )
            for camera, hand in selected.items()
        ]
        errors = _errors(point, all_observations)
        cross_view_errors.extend(errors[np.isfinite(errors)].tolist())
        multiview_point = result["points"][joint]
        if np.all(np.isfinite(multiview_point)):
            differences_mm.append(float(np.linalg.norm(point - multiview_point) * 1000.0))
    available = int(np.count_nonzero(np.isfinite(stereo_points).all(axis=1)))
    return {
        "camera_ids": list(baseline_cameras),
        "available_joint_count": available,
        "cross_view_reprojection_median_px": (
            float(np.median(cross_view_errors)) if cross_view_errors else None
        ),
        "cross_view_reprojection_p95_px": (
            float(np.percentile(cross_view_errors, 95)) if cross_view_errors else None
        ),
        "multiview_3d_difference_median_mm": (
            float(np.median(differences_mm)) if differences_mm else None
        ),
    }


def _serialize_hand(
    result: dict[str, Any], selected: dict[str, dict[str, Any]],
    calibrations: dict[str, CameraCalibration], baseline_cameras: tuple[str, str],
) -> dict[str, Any]:
    return {
        "side": result["side"],
        "camera_ids": result["cameras"],
        "joints_base_m": result["points"].tolist(),
        "inlier_view_counts": result["inlier_counts"].tolist(),
        "quality": {
            "valid_joint_count": result["valid_joints"],
            "multiview_reprojection_median_px": result["median_px"],
            "multiview_reprojection_p95_px": result["p95_px"],
            "all_view_reprojection_median_px": result["all_view_median_px"],
            "all_view_reprojection_p95_px": result["all_view_p95_px"],
        },
        "stereo_baseline_comparison": _baseline_comparison(
            result, selected, calibrations, baseline_cameras
        ),
        "views": {
            camera: {
                "detection_index": int(hand["detection_index"]),
                "bbox_xyxy": hand["bbox_xyxy"],
                "confidence": hand["confidence"],
                "detector_is_right": hand.get("detector_is_right"),
                "joints_2d": hand["joints_2d"],
                "inlier_joint_count": int(
                    result["inlier_mask"][:, result["cameras"].index(camera)].sum()
                ),
            }
            for camera, hand in selected.items()
        },
    }


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"output already exists: {args.output}")
    if args.min_views < 2 or args.min_valid_joints < 1 or args.max_frames < 0:
        raise ValueError("invalid min-views/min-valid-joints/max-frames")
    dataset = args.dataset.resolve()
    prediction_root = args.predictions.resolve()
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    cameras = tuple(manifest["camera_ids"])
    baseline_cameras = tuple(args.baseline_cameras)
    if any(camera not in cameras for camera in baseline_cameras):
        raise ValueError(f"baseline cameras must be selected dataset cameras: {baseline_cameras}")
    calibrations = {
        camera: CameraCalibration.load(dataset / "calibration" / f"{camera}.json")
        for camera in cameras
    }
    prediction_rows = {
        camera: _load_jsonl(prediction_root / camera / "predictions.jsonl")
        for camera in cameras
    }
    frame_ids = sorted(set.intersection(*(set(rows) for rows in prediction_rows.values())))
    frame_ids = frame_ids[: args.max_frames or None]
    output = args.output.resolve()
    output.mkdir(parents=True)
    accepted = []
    rejected = []
    reasons: Counter[str] = Counter()
    for sync_index in frame_ids:
        groups = {camera: _groups(prediction_rows[camera][sync_index]) for camera in cameras}
        active = []
        detection_ids = {}
        for camera in cameras:
            complete = [
                detection for detection, sides in groups[camera].items()
                if 0 in sides and 1 in sides
            ]
            complete.sort(
                key=lambda detection: float(groups[camera][detection][0].get("confidence", 0)),
                reverse=True,
            )
            if len(complete) >= 2:
                active.append(camera)
                detection_ids[camera] = complete[:2]
        if len(active) < args.min_views:
            reason = "too_few_cameras_with_two_hands"
            rejected.append({"sync_index": sync_index, "reason": reason, "active_cameras": active})
            reasons[reason] += 1
            continue
        assignments = []
        for flips in itertools.product((0, 1), repeat=len(active)):
            selected_by_side = {0: {}, 1: {}}
            for camera, flip in zip(active, flips):
                first, second = detection_ids[camera]
                left_detection, right_detection = (first, second) if flip == 0 else (second, first)
                selected_by_side[0][camera] = groups[camera][left_detection][0]
                selected_by_side[1][camera] = groups[camera][right_detection][1]
            hands = [
                _triangulate_hand(
                    side, selected_by_side[side], calibrations,
                    args.ransac_threshold_px, args.min_views,
                )
                for side in (0, 1)
            ]
            missing = sum(21 - hand["valid_joints"] for hand in hands)
            cost = sum(hand["median_px"] + 0.2 * hand["p95_px"] for hand in hands) + 100.0 * missing
            assignments.append((float(cost), flips, hands, selected_by_side))
        assignments.sort(key=lambda item: item[0])
        best = assignments[0]
        margin = assignments[1][0] - best[0] if len(assignments) > 1 else float("inf")
        if np.isfinite(margin) and margin < args.ambiguity_margin:
            reason = "assignment_ambiguous"
            rejected.append({"sync_index": sync_index, "reason": reason, "margin": margin})
            reasons[reason] += 1
            continue
        quality_reasons = []
        for hand in best[2]:
            if hand["valid_joints"] < args.min_valid_joints:
                quality_reasons.append("too_few_valid_joints")
            if hand["median_px"] > args.max_reprojection_median_px:
                quality_reasons.append("reprojection_median_too_large")
            if hand["p95_px"] > args.max_reprojection_p95_px:
                quality_reasons.append("reprojection_p95_too_large")
        if quality_reasons:
            quality_reasons = sorted(set(quality_reasons))
            rejected.append({
                "sync_index": sync_index, "reasons": quality_reasons,
                "assignment_cost": best[0], "assignment_margin": margin,
                "hands": [
                    {
                        "side": hand["side"], "valid_joint_count": hand["valid_joints"],
                        "inlier_median_px": hand["median_px"],
                        "inlier_p95_px": hand["p95_px"],
                        "all_view_p95_px": hand["all_view_p95_px"],
                    }
                    for hand in best[2]
                ],
            })
            reasons.update(quality_reasons)
            continue
        accepted.append({
            "sync_index": sync_index,
            "active_camera_count": len(active),
            "active_cameras": active,
            "assignment_cost": best[0],
            "assignment_margin": margin,
            "hands": [
                _serialize_hand(
                    hand, best[3][side], calibrations, baseline_cameras
                )
                for side, hand in enumerate(best[2])
            ],
        })
    (output / "accepted.jsonl").write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in accepted),
        encoding="utf-8",
    )
    (output / "rejected.jsonl").write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rejected),
        encoding="utf-8",
    )
    all_medians = [
        hand["quality"]["multiview_reprojection_median_px"]
        for row in accepted for hand in row["hands"]
    ]
    all_p95 = [
        hand["quality"]["multiview_reprojection_p95_px"]
        for row in accepted for hand in row["hands"]
    ]
    stereo_cross_view = [
        hand["stereo_baseline_comparison"]["cross_view_reprojection_median_px"]
        for row in accepted for hand in row["hands"]
        if hand["stereo_baseline_comparison"].get("cross_view_reprojection_median_px") is not None
    ]
    multiview_cross_view = [
        hand["quality"]["all_view_reprojection_median_px"]
        for row in accepted for hand in row["hands"]
        if hand["stereo_baseline_comparison"].get("cross_view_reprojection_median_px") is not None
    ]
    differences_mm = [
        hand["stereo_baseline_comparison"]["multiview_3d_difference_median_mm"]
        for row in accepted for hand in row["hands"]
        if hand["stereo_baseline_comparison"].get("multiview_3d_difference_median_mm") is not None
    ]
    active_camera_counts = [row["active_camera_count"] for row in accepted]
    joint_inlier_view_counts = [
        count
        for row in accepted for hand in row["hands"]
        for count in hand["inlier_view_counts"]
        if count >= args.min_views
    ]
    stereo_cross_median = float(np.median(stereo_cross_view)) if stereo_cross_view else None
    multiview_cross_median = float(np.median(multiview_cross_view)) if multiview_cross_view else None
    summary = {
        "schema_version": 1,
        "stage": "wilor_native_ds_multiview_fusion",
        "camera_ids": list(cameras),
        "processed_frame_count": len(frame_ids),
        "accepted_frame_count": len(accepted),
        "rejected_frame_count": len(rejected),
        "accepted_rate": len(accepted) / max(len(frame_ids), 1),
        "rejected_reason_counts": dict(reasons),
        "reprojection_median_px": float(np.median(all_medians)) if all_medians else None,
        "reprojection_p95_px": float(np.median(all_p95)) if all_p95 else None,
        "accepted_active_camera_count_median": (
            float(np.median(active_camera_counts)) if active_camera_counts else None
        ),
        "valid_joint_inlier_view_count_median": (
            float(np.median(joint_inlier_view_counts)) if joint_inlier_view_counts else None
        ),
        "valid_joint_inlier_view_count_distribution": {
            str(count): joint_inlier_view_counts.count(count)
            for count in sorted(set(joint_inlier_view_counts))
        },
        "stereo_baseline_comparison": {
            "camera_ids": list(baseline_cameras),
            "comparable_hand_count": len(stereo_cross_view),
            "stereo_cross_view_median_px": stereo_cross_median,
            "multiview_cross_view_median_px": multiview_cross_median,
            "cross_view_median_improvement_percent": (
                100.0 * (1.0 - multiview_cross_median / stereo_cross_median)
                if stereo_cross_median and multiview_cross_median is not None else None
            ),
            "stereo_multiview_3d_difference_median_mm": (
                float(np.median(differences_mm)) if differences_mm else None
            ),
        },
        "parameters": {
            "min_views": args.min_views,
            "ransac_threshold_px": args.ransac_threshold_px,
            "max_reprojection_median_px": args.max_reprojection_median_px,
            "max_reprojection_p95_px": args.max_reprojection_p95_px,
            "min_valid_joints": args.min_valid_joints,
            "ambiguity_margin": args.ambiguity_margin,
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
