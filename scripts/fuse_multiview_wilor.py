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
from camera_models.double_sphere import unproject as unproject_double_sphere  # noqa: E402
from camera_models.kb import unproject as unproject_kb  # noqa: E402
from ego_data.calibration import CameraCalibration  # noqa: E402


# A very small ray angle makes depth essentially unconstrained.  The previous
# 0.75 degree gate was enough for the synthetic near-parallel unit test, but
# still allowed badly conditioned pairs in real recordings.  At the working
# distance of the GEN cameras a 1.25 degree angle is comfortably below the
# normal inter-camera angle while removing the unstable tail.
MIN_TRIANGULATION_ANGLE_DEG = 1.25
MAX_TRIANGULATED_BASE_DISTANCE_M = 3.0
MAX_TRIANGULATED_HAND_RADIUS_M = 0.30
MAX_PAIR_POINT_SPREAD_M = 0.12
# A ray intersection a few centimetres in front of the camera is usually the
# result of pairing two different detector boxes.  It can nevertheless have
# an almost perfect reprojection score, so image error alone must not accept
# it.  The GEN recordings put the glove well outside this near-camera volume;
# keep the threshold conservative enough for close hand motions.
MIN_TRIANGULATED_CAMERA_DISTANCE_M = 0.12
# A 21-joint hand with a palm span outside this interval is not a plausible
# physical hand.  This catches cross-camera box pairings whose individual
# rays happen to meet at a compact/oversized pseudo-pose.
MIN_TRIANGULATED_PALM_WIDTH_M = 0.035
MAX_TRIANGULATED_PALM_WIDTH_M = 0.18


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
    if camera.model == "DS":
        unproject = unproject_double_sphere
    elif camera.model == "KB":
        unproject = unproject_kb
    else:
        raise ValueError(f"unsupported camera model: {camera.model}")
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


def regularized_ray_intersection(
    rays: list[tuple[np.ndarray, np.ndarray]],
    prior: np.ndarray,
    weight: float,
) -> np.ndarray | None:
    """Intersect rays while softly anchoring the solution to a 3-D prior.

    Two cameras with a small baseline constrain image position well but leave
    depth poorly conditioned.  A temporal prior turns that ill-conditioned
    intersection into the regularized least-squares problem

    ``sum ||(I-dd^T)(p-c)||² + weight ||p-prior||²``.

    ``weight`` is dimensionless relative to the ray normal equations.  The
    caller is responsible for deciding when the prior is trustworthy; this
    function only solves the bounded linear system and rejects non-finite
    inputs.
    """
    prior = np.asarray(prior, dtype=np.float64)
    if prior.shape != (3,) or not np.all(np.isfinite(prior)) or len(rays) < 2:
        return None
    lam = max(float(weight), 0.0)
    matrix = np.eye(3) * lam
    target = lam * prior
    identity = np.eye(3)
    valid = 0
    for center, direction in rays:
        center = np.asarray(center, dtype=np.float64)
        direction = np.asarray(direction, dtype=np.float64)
        norm = float(np.linalg.norm(direction))
        if (
            center.shape != (3,)
            or direction.shape != (3,)
            or not np.all(np.isfinite(center))
            or not np.all(np.isfinite(direction))
            or norm <= 1e-9
        ):
            continue
        direction = direction / norm
        projector = identity - np.outer(direction, direction)
        matrix += projector
        target += projector @ center
        valid += 1
    if valid < 2:
        return None
    try:
        point = np.linalg.solve(matrix, target)
    except np.linalg.LinAlgError:
        return None
    return point if np.all(np.isfinite(point)) else None


def _maximum_ray_angle_deg(
    rays: list[tuple[np.ndarray, np.ndarray]],
) -> float:
    maximum = 0.0
    for first, second in itertools.combinations(rays, 2):
        cosine = float(np.clip(np.dot(first[1], second[1]), -1.0, 1.0))
        maximum = max(maximum, float(np.degrees(np.arccos(cosine))))
    return maximum


def _pairwise_triangulation_points(
    rays: list[tuple[np.ndarray, np.ndarray]],
) -> list[np.ndarray]:
    """Return well-conditioned pair intersections for a ray set.

    A least-squares intersection can have a low reprojection error even when
    different camera pairs imply materially different depths.  Keeping the
    pair solutions lets the caller detect that ambiguity before accepting a
    depth branch.
    """
    points: list[np.ndarray] = []
    for first, second in itertools.combinations(rays, 2):
        cosine = float(np.clip(np.dot(first[1], second[1]), -1.0, 1.0))
        angle = float(np.degrees(np.arccos(cosine)))
        if angle < MIN_TRIANGULATION_ANGLE_DEG:
            continue
        point = _intersect_rays([first, second])
        if point is not None and np.all(np.isfinite(point)):
            points.append(point)
    return points


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
    # (inlier count, negative median error, pair angle, point, inlier mask)
    best: tuple[int, float, float, np.ndarray, np.ndarray] | None = None
    for first, second in itertools.combinations(range(len(observations)), 2):
        if rays[first] is None or rays[second] is None:
            continue
        cosine = float(np.clip(np.dot(rays[first][1], rays[second][1]), -1.0, 1.0))
        pair_angle = float(np.degrees(np.arccos(cosine)))
        if pair_angle < MIN_TRIANGULATION_ANGLE_DEG:
            continue
        point = _intersect_rays([rays[first], rays[second]])
        if point is None:
            continue
        errors = _errors(point, observations)
        inliers = np.isfinite(errors) & (errors <= threshold_px)
        count = int(inliers.sum())
        score = float(np.median(errors[inliers])) if count else float("inf")
        # Prefer a better-conditioned pair when support and reprojection are
        # tied.  This is a tie breaker only; image agreement remains primary.
        candidate = (count, -score, pair_angle, point, inliers)
        if best is None or candidate[:3] > best[:3]:
            best = candidate
    if best is None or best[0] < 2:
        return np.full(3, np.nan), np.full(len(observations), np.inf), np.zeros(len(observations), bool)
    inlier_rays = [rays[index] for index in np.flatnonzero(best[4]) if rays[index] is not None]
    point = _intersect_rays(inlier_rays)
    if point is None:
        point = best[3]
    errors = _errors(point, observations)
    inliers = np.isfinite(errors) & (errors <= threshold_px)
    final_rays = [
        rays[index] for index in np.flatnonzero(inliers)
        if rays[index] is not None
    ]
    if (
        len(final_rays) < 2
        or _maximum_ray_angle_deg(final_rays) < MIN_TRIANGULATION_ANGLE_DEG
        or not np.all(np.isfinite(point))
        or float(np.linalg.norm(point)) > MAX_TRIANGULATED_BASE_DISTANCE_M
    ):
        return (
            np.full(3, np.nan),
            errors,
            np.zeros(len(observations), dtype=bool),
        )
    # Reject intersections that lie essentially on a camera centre.  Such a
    # point is a classic low-baseline/depth branch: every selected pixel can
    # still be explained, but the recovered depth is physically meaningless.
    ray_distances = np.asarray(
        [np.linalg.norm(point - rays[index][0]) for index in np.flatnonzero(inliers)
         if rays[index] is not None],
        dtype=np.float64,
    )
    if (
        ray_distances.size < 2
        or not np.all(np.isfinite(ray_distances))
        or float(np.min(ray_distances)) < MIN_TRIANGULATED_CAMERA_DISTANCE_M
    ):
        return (
            np.full(3, np.nan),
            errors,
            np.zeros(len(observations), dtype=bool),
        )
    pair_points = _pairwise_triangulation_points(final_rays)
    if len(pair_points) >= 2:
        pair_array = np.asarray(pair_points, dtype=np.float64)
        pair_centre = np.median(pair_array, axis=0)
        pair_spread = np.linalg.norm(pair_array - pair_centre, axis=1)
        # Pair disagreement is a depth-branch ambiguity, not ordinary pixel
        # noise.  Reject only a high quantile so one noisy camera does not
        # discard an otherwise well-supported point.
        if float(np.percentile(pair_spread, 75)) > MAX_PAIR_POINT_SPREAD_M:
            return (
                np.full(3, np.nan),
                errors,
                np.zeros(len(observations), dtype=bool),
            )
    return point, errors, inliers


def _reject_gross_hand_geometry(
    points: np.ndarray,
    inlier_mask: np.ndarray,
    inlier_counts: np.ndarray,
) -> int:
    """Invalidate isolated triangulations outside a physical hand envelope."""
    rejected = 0
    for _ in range(2):
        finite = np.isfinite(points).all(axis=1)
        joints = np.flatnonzero(finite)
        if len(joints) < 5:
            break
        centre = np.median(points[joints], axis=0)
        distances = np.linalg.norm(points[joints] - centre, axis=1)
        bad = joints[distances > MAX_TRIANGULATED_HAND_RADIUS_M]
        if not len(bad):
            break
        points[bad] = np.nan
        inlier_mask[bad] = False
        inlier_counts[bad] = 0
        rejected += len(bad)
    return rejected


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
    geometry_rejected_joints = _reject_gross_hand_geometry(
        points, inlier_mask, inlier_counts
    )
    # Keep the conditioning information alongside the reconstructed points.
    # Reprojection error alone is not enough to identify an unstable stereo
    # branch: two nearly parallel rays can intersect at an arbitrary depth and
    # still explain both pixels almost perfectly.  These per-joint summaries
    # are consumed by the guided selector and are also serialized for audit.
    joint_ray_angles = np.full(21, np.nan, dtype=np.float64)
    joint_pair_spreads = np.full(21, np.nan, dtype=np.float64)
    # Conditioning of the palm anchors determines whether the whole hand is
    # trustworthy.  Computing pair intersections for all 21 joints inside the
    # already combinatorial candidate enumeration was needlessly expensive;
    # retain the full-shaped arrays for compatibility but evaluate the five
    # palm joints only.
    conditioning_joints = (0, 5, 9, 13, 17)
    for joint in conditioning_joints:
        if not np.all(np.isfinite(points[joint])):
            continue
        ray_values = []
        for index, camera in enumerate(cameras):
            if not inlier_mask[joint, index]:
                continue
            pixel_values = np.asarray(
                selected[camera].get("joints_2d", [])[joint], dtype=np.float64
            )
            if pixel_values.shape != (2,) or not np.all(np.isfinite(pixel_values)):
                continue
            ray = _ray_in_base(calibrations[camera], pixel_values)
            if ray is not None:
                ray_values.append(ray)
        if len(ray_values) < 2:
            continue
        angles = []
        pair_points = []
        for first, second in itertools.combinations(ray_values, 2):
            cosine = float(np.clip(np.dot(first[1], second[1]), -1.0, 1.0))
            angles.append(float(np.degrees(np.arccos(cosine))))
            if angles[-1] >= MIN_TRIANGULATION_ANGLE_DEG:
                pair_point = _intersect_rays([first, second])
                if pair_point is not None and np.all(np.isfinite(pair_point)):
                    pair_points.append(pair_point)
        if angles:
            # The largest inlier-ray angle is the useful conditioning angle:
            # it is the pair RANSAC prefers when image support is tied.
            joint_ray_angles[joint] = max(angles)
        if len(pair_points) >= 2:
            pair_array = np.asarray(pair_points, dtype=np.float64)
            pair_centre = np.median(pair_array, axis=0)
            joint_pair_spreads[joint] = float(np.percentile(
                np.linalg.norm(pair_array - pair_centre, axis=1), 75
            ))
    geometry_rejection_reasons: list[str] = []
    finite_points = np.isfinite(points).all(axis=1)
    palm_indices = np.asarray([0, 5, 9, 13, 17], dtype=np.int64)
    finite_palm = finite_points[palm_indices]
    palm_width = float("nan")
    camera_distance = float("nan")
    if np.count_nonzero(finite_palm) >= 3:
        palm_values = points[palm_indices[finite_palm]]
        palm_center = np.median(palm_values, axis=0)
        if np.all(np.isfinite(palm_center)):
            centers = np.asarray(
                [calibrations[camera].T_base_camera[:3, 3] for camera in cameras],
                dtype=np.float64,
            )
            camera_distance = float(np.min(
                np.linalg.norm(palm_center[None] - centers, axis=1)
            )) if len(centers) else float("nan")
        if finite_points[5] and finite_points[17]:
            palm_width = float(np.linalg.norm(points[5] - points[17]))
        if np.isfinite(camera_distance) and (
            camera_distance < MIN_TRIANGULATED_CAMERA_DISTANCE_M
        ):
            geometry_rejection_reasons.append("near_camera_hand_center")
        if np.isfinite(palm_width) and (
            palm_width < MIN_TRIANGULATED_PALM_WIDTH_M
            or palm_width > MAX_TRIANGULATED_PALM_WIDTH_M
        ):
            geometry_rejection_reasons.append("implausible_palm_width")
    # The physical envelope is a hand-level gate.  Do not leave a partially
    # valid pseudo-hand for later temporal code to propagate when its palm is
    # already impossible; retaining the reason makes the rejection auditable.
    if geometry_rejection_reasons:
        geometry_rejected_joints += int(np.count_nonzero(finite_points))
        points[:] = np.nan
        inlier_mask[:] = False
        inlier_counts[:] = 0
    finite_errors = errors[np.isfinite(errors)]
    inlier_errors = errors[inlier_mask & np.isfinite(errors)]
    valid_joints = int(np.count_nonzero(np.isfinite(points).all(axis=1)))
    median = float(np.median(inlier_errors)) if len(inlier_errors) else float("inf")
    p95 = float(np.percentile(inlier_errors, 95)) if len(inlier_errors) else float("inf")
    finite_angles = joint_ray_angles[np.isfinite(joint_ray_angles)]
    finite_spreads = joint_pair_spreads[np.isfinite(joint_pair_spreads)]
    palm_indices = np.asarray([0, 5, 9, 13, 17], dtype=np.int64)
    finite_palm = np.isfinite(points[palm_indices]).all(axis=1)
    palm_angles = joint_ray_angles[palm_indices]
    palm_angles = palm_angles[np.isfinite(palm_angles)]
    palm_spreads = joint_pair_spreads[palm_indices]
    palm_spreads = palm_spreads[np.isfinite(palm_spreads)]
    palm_support = inlier_counts[palm_indices][finite_palm]
    support_values = inlier_counts[np.isfinite(points).all(axis=1)]
    return {
        "side": side,
        "cameras": list(cameras),
        "points": points,
        "errors": errors,
        "inlier_mask": inlier_mask,
        "inlier_counts": inlier_counts,
        "valid_joints": valid_joints,
        "geometry_rejected_joints": geometry_rejected_joints,
        "geometry_rejection_reasons": geometry_rejection_reasons,
        "palm_width_m": palm_width,
        "camera_distance_m": camera_distance,
        "median_px": median,
        "p95_px": p95,
        "all_view_median_px": (
            float(np.median(finite_errors)) if len(finite_errors) else float("inf")
        ),
        "all_view_p95_px": (
            float(np.percentile(finite_errors, 95)) if len(finite_errors) else float("inf")
        ),
        "joint_ray_angle_deg": joint_ray_angles,
        "joint_pair_spread_m": joint_pair_spreads,
        "ray_angle_min_deg": (
            float(np.min(finite_angles)) if len(finite_angles) else float("nan")
        ),
        "ray_angle_p25_deg": (
            float(np.percentile(finite_angles, 25)) if len(finite_angles) else float("nan")
        ),
        "ray_angle_median_deg": (
            float(np.median(finite_angles)) if len(finite_angles) else float("nan")
        ),
        "palm_ray_angle_min_deg": (
            float(np.min(palm_angles)) if len(palm_angles) else float("nan")
        ),
        "palm_ray_angle_p25_deg": (
            float(np.percentile(palm_angles, 25)) if len(palm_angles) else float("nan")
        ),
        "palm_ray_angle_median_deg": (
            float(np.median(palm_angles)) if len(palm_angles) else float("nan")
        ),
        "pair_spread_p75_m": (
            float(np.percentile(finite_spreads, 75)) if len(finite_spreads) else 0.0
        ),
        "pair_spread_max_m": (
            float(np.max(finite_spreads)) if len(finite_spreads) else 0.0
        ),
        "palm_pair_spread_p75_m": (
            float(np.percentile(palm_spreads, 75)) if len(palm_spreads) else 0.0
        ),
        "palm_three_view_count": int(np.count_nonzero(palm_support >= 3)),
        "palm_two_view_fraction": (
            float(np.mean(palm_support == 2)) if len(palm_support) else 1.0
        ),
        "two_view_joint_fraction": (
            float(np.mean(support_values == 2)) if len(support_values) else 1.0
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
    def serialize_view(camera: str, hand: dict[str, Any]) -> dict[str, Any]:
        view = {
            "detection_index": int(hand["detection_index"]),
            "bbox_xyxy": hand["bbox_xyxy"],
            "confidence": hand["confidence"],
            "detector_is_right": hand.get("detector_is_right"),
            "joints_2d": hand["joints_2d"],
            "inlier_joint_count": int(
                result["inlier_mask"][:, result["cameras"].index(camera)].sum()
            ),
        }
        if "wilor_joints_2d" in hand:
            view["wilor_joints_2d"] = hand["wilor_joints_2d"]
        if "marker_assist" in hand:
            view["marker_assist"] = hand["marker_assist"]
        return view

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
            "geometry_rejected_joint_count": int(
                result.get("geometry_rejected_joints", 0)
            ),
            "ray_angle_min_deg": result.get("ray_angle_min_deg"),
            "ray_angle_p25_deg": result.get("ray_angle_p25_deg"),
            "ray_angle_median_deg": result.get("ray_angle_median_deg"),
            "palm_ray_angle_min_deg": result.get("palm_ray_angle_min_deg"),
            "palm_ray_angle_p25_deg": result.get("palm_ray_angle_p25_deg"),
            "palm_ray_angle_median_deg": result.get("palm_ray_angle_median_deg"),
            "pair_spread_p75_m": result.get("pair_spread_p75_m"),
            "pair_spread_max_m": result.get("pair_spread_max_m"),
            "palm_pair_spread_p75_m": result.get("palm_pair_spread_p75_m"),
            "palm_three_view_count": result.get("palm_three_view_count"),
            "palm_two_view_fraction": result.get("palm_two_view_fraction"),
            "two_view_joint_fraction": result.get("two_view_joint_fraction"),
        },
        "stereo_baseline_comparison": _baseline_comparison(
            result, selected, calibrations, baseline_cameras
        ),
        "views": {
            camera: serialize_view(camera, hand)
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
