#!/usr/bin/env python3
"""Conservative temporal guard for ambiguous multiview 3D hand tracks.

The calibrated 2D observations can be perfectly smooth while a two-view ray
intersection changes depth dramatically when the baseline is small.  This
module only changes the exported 3D joints.  Camera-native boxes and WiLoR
2D joints remain untouched, so a bad depth estimate cannot move the reliable
2D visualisation.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from camera_models import project_points


PALM_JOINTS = np.asarray((0, 5, 9, 13, 17), dtype=np.int32)
ALGORITHM = "bidirectional_trusted_anchor_3d_guard_v2"


def _palm_center(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if points.shape != (21, 3):
        return np.full(3, np.nan, dtype=np.float64)
    values = points[PALM_JOINTS]
    finite = np.isfinite(values).all(axis=1)
    if np.count_nonzero(finite) < 3:
        return np.full(3, np.nan, dtype=np.float64)
    return np.median(values[finite], axis=0)


def _bbox_iou(first: Any, second: Any) -> float:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    if first.shape != (4,) or second.shape != (4,):
        return 0.0
    left = max(float(first[0]), float(second[0]))
    top = max(float(first[1]), float(second[1]))
    right = min(float(first[2]), float(second[2]))
    bottom = min(float(first[3]), float(second[3]))
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, float(first[2] - first[0])) * max(
        0.0, float(first[3] - first[1])
    )
    second_area = max(0.0, float(second[2] - second[0])) * max(
        0.0, float(second[3] - second[1])
    )
    union = first_area + second_area - intersection
    return intersection / union if union > 1e-9 else 0.0


def _duplicate_detector_views(row: dict[str, Any], iou_threshold: float) -> bool:
    """Detect a same-frame left/right assignment to one physical box."""
    hands = {
        int(hand.get("side", -1)): hand
        for hand in row.get("hands", [])
        if int(hand.get("side", -1)) in (0, 1)
    }
    if set(hands) != {0, 1}:
        return False
    first_views = hands[0].get("views", {})
    second_views = hands[1].get("views", {})
    for camera, first in first_views.items():
        second = second_views.get(camera)
        if second is None:
            continue
        if (
            first.get("detection_index") is not None
            and first.get("detection_index") == second.get("detection_index")
        ):
            return True
        if _bbox_iou(
            first.get("bbox_xyxy", []), second.get("bbox_xyxy", [])
        ) >= iou_threshold:
            return True
    return False


def _hand_info(
    hand: dict[str, Any], duplicate: bool,
) -> dict[str, Any]:
    points = np.asarray(hand.get("joints_base_m", []), dtype=np.float64)
    counts = np.asarray(hand.get("inlier_view_counts", []), dtype=np.float64)
    palm_counts = (
        counts[PALM_JOINTS]
        if counts.shape == (21,)
        else np.zeros(len(PALM_JOINTS), dtype=np.float64)
    )
    quality = hand.get("quality", {})
    reprojection = float(
        quality.get("multiview_reprojection_median_px", float("inf"))
    )
    return {
        "points": points,
        "center": _palm_center(points),
        "camera_count": len(hand.get("camera_ids", [])),
        "palm_min_inliers": int(np.min(palm_counts)) if len(palm_counts) else 0,
        "reprojection_median_px": reprojection,
        "duplicate": bool(duplicate),
    }


def _motion_stats(centers: list[np.ndarray], frame_ids: list[int]) -> dict[str, Any]:
    values: list[float] = []
    for first, second, first_id, second_id in zip(
        centers[:-1], centers[1:], frame_ids[:-1], frame_ids[1:]
    ):
        if not (np.isfinite(first).all() and np.isfinite(second).all()):
            continue
        gap = max(int(second_id) - int(first_id), 1)
        values.append(float(np.linalg.norm(second - first) * 1000.0 / gap))
    array = np.asarray(values, dtype=np.float64)
    return {
        "sample_count": int(len(array)),
        "median_mm_per_frame": float(np.median(array)) if len(array) else None,
        "p95_mm_per_frame": float(np.percentile(array, 95)) if len(array) else None,
        "p99_mm_per_frame": float(np.percentile(array, 99)) if len(array) else None,
        "max_mm_per_frame": float(np.max(array)) if len(array) else None,
        "over_50mm_count": int(np.count_nonzero(array > 50.0)),
        "over_80mm_count": int(np.count_nonzero(array > 80.0)),
        "over_100mm_count": int(np.count_nonzero(array > 100.0)),
    }


def _translate_hand(
    hand: dict[str, Any], current: np.ndarray, target: np.ndarray,
) -> int:
    points = np.asarray(hand.get("joints_base_m", []), dtype=np.float64)
    if points.shape != (21, 3) or not (
        np.isfinite(current).all() and np.isfinite(target).all()
    ):
        return 0
    finite = np.isfinite(points).all(axis=1)
    if not np.any(finite):
        return 0
    points[finite] += target - current
    hand["joints_base_m"] = points.tolist()
    return int(np.count_nonzero(finite))


def _translated_points(
    hand: dict[str, Any], current: np.ndarray, target: np.ndarray,
) -> np.ndarray:
    points = np.asarray(hand.get("joints_base_m", []), dtype=np.float64)
    if points.shape != (21, 3) or not (
        np.isfinite(current).all() and np.isfinite(target).all()
    ):
        return points
    output = points.copy()
    finite = np.isfinite(output).all(axis=1)
    output[finite] += target - current
    return output


def _reprojection_stats(
    hand: dict[str, Any], points: np.ndarray,
    calibrations: Mapping[str, Any] | None,
) -> tuple[float, float] | None:
    """Return candidate median/p95 pixel residual over selected camera views."""
    if calibrations is None or points.shape != (21, 3):
        return None
    residuals: list[float] = []
    finite_points = np.isfinite(points).all(axis=1)
    if not np.any(finite_points):
        return None
    for camera_id, view in hand.get("views", {}).items():
        camera = calibrations.get(camera_id)
        if camera is None or int(view.get("inlier_joint_count", 0)) <= 0:
            continue
        observations = np.asarray(view.get("joints_2d", []), dtype=np.float64)
        if observations.shape != (21, 2):
            continue
        rotation = camera.T_base_camera[:3, :3]
        center = camera.T_base_camera[:3, 3]
        points_camera = (rotation.T @ (points - center).T).T
        projected, valid = project_points(camera, points_camera)
        valid = (
            np.asarray(valid, dtype=bool)
            & finite_points
            & np.isfinite(observations).all(axis=1)
            & np.isfinite(projected).all(axis=1)
        )
        if np.any(valid):
            residuals.extend(
                np.linalg.norm(projected[valid] - observations[valid], axis=1).tolist()
            )
    if not residuals:
        return None
    values = np.asarray(residuals, dtype=np.float64)
    return float(np.median(values)), float(np.percentile(values, 95))


def _annotate_replacement(
    hand: dict[str, Any], method: str, residual_m: float,
    references: tuple[int, ...], corrected_joint_count: int,
) -> None:
    hand["temporal_guard"] = {
        "applied": True,
        "method": method,
        "residual_m": float(residual_m),
        "reference_sync_indices": [int(value) for value in references],
        "corrected_joint_count": int(corrected_joint_count),
    }
    anatomy = hand.setdefault("anatomy_refinement", {})
    anatomy["applied"] = True
    anatomy["temporal_guard_applied"] = True
    anatomy["temporal_guard_method"] = method
    anatomy["temporal_guard_corrected_joint_count"] = int(corrected_joint_count)


def _annotate_clear(hand: dict[str, Any], method: str) -> None:
    hand["joints_base_m"] = np.full((21, 3), np.nan, dtype=np.float64).tolist()
    hand["temporal_guard"] = {"applied": True, "method": method}
    anatomy = hand.setdefault("anatomy_refinement", {})
    anatomy["applied"] = True
    anatomy["temporal_guard_applied"] = True
    anatomy["temporal_guard_method"] = method
    anatomy["temporal_guard_corrected_joint_count"] = 0


def guard_temporal_rows(
    rows: list[dict[str, Any]],
    *,
    radius: int = 24,
    replacement_threshold_m: float = 0.060,
    min_anchor_cameras: int = 5,
    min_anchor_palm_inliers: int = 3,
    max_anchor_reprojection_px: float = 12.0,
    max_anchor_speed_m_per_frame: float = 0.050,
    duplicate_iou: float = 0.90,
    clear_unrecoverable_endpoint: bool = True,
    calibrations: Mapping[str, Any] | None = None,
    max_reprojection_median_px: float = 35.0,
    max_reprojection_p95_px: float = 100.0,
    max_local_reprojection_ratio: float = 1.5,
) -> dict[str, Any]:
    """Repair depth spikes using only bidirectional, high-support anchors.

    A replacement is a rigid translation of the current hand, preserving the
    WiLoR articulation/2D evidence.  If an endpoint has no future support and
    is implausibly far from its last trusted pose, its 3D joints are cleared;
    the detector box and 2D skeleton remain available to the renderer.
    """
    if not rows:
        return {
            "enabled": True,
            "algorithm": ALGORITHM,
            "corrected_hand_count": 0,
            "cleared_hand_count": 0,
        }
    ordered = sorted(rows, key=lambda row: int(row["sync_index"]))
    frame_ids = [int(row["sync_index"]) for row in ordered]
    radius = max(int(radius), 1)
    threshold = max(float(replacement_threshold_m), 1e-6)
    infos: dict[tuple[int, int], dict[str, Any]] = {}
    duplicate_count = 0
    for index, row in enumerate(ordered):
        duplicate = _duplicate_detector_views(row, duplicate_iou)
        duplicate_count += int(duplicate)
        for side in (0, 1):
            hand = next(
                (value for value in row.get("hands", [])
                 if int(value.get("side", -1)) == side),
                None,
            )
            if hand is not None:
                infos[(index, side)] = _hand_info(hand, duplicate)

    before_centers = {
        side: [infos[(index, side)]["center"].copy() for index in range(len(ordered))]
        for side in (0, 1)
    }
    anchor_indices: dict[int, list[int]] = {0: [], 1: []}
    for side in (0, 1):
        for index in range(len(ordered)):
            info = infos.get((index, side))
            if info is None:
                continue
            if (
                np.isfinite(info["center"]).all()
                and info["camera_count"] >= min_anchor_cameras
                and info["palm_min_inliers"] >= min_anchor_palm_inliers
                and info["reprojection_median_px"] <= max_anchor_reprojection_px
                and not info["duplicate"]
            ):
                anchor_indices[side].append(index)

    corrections: list[tuple[int, int, np.ndarray, float, tuple[int, int], str]] = []
    reprojection_rejected: list[dict[str, Any]] = []
    # Pass 1: use nearest trusted anchors on both sides of an ambiguous frame.
    for side in (0, 1):
        anchors = anchor_indices[side]
        for index in range(len(ordered)):
            info = infos.get((index, side))
            if info is None or index in anchors:
                continue
            current = info["center"]
            if not np.isfinite(current).all():
                continue
            sync_index = frame_ids[index]
            before = [
                candidate for candidate in anchors
                if frame_ids[candidate] < sync_index
                and sync_index - frame_ids[candidate] <= radius
            ]
            after = [
                candidate for candidate in anchors
                if frame_ids[candidate] > sync_index
                and frame_ids[candidate] - sync_index <= radius
            ]
            if not before or not after:
                continue
            first = max(before, key=lambda value: frame_ids[value])
            second = min(after, key=lambda value: frame_ids[value])
            gap = frame_ids[second] - frame_ids[first]
            if gap <= 0:
                continue
            first_center = before_centers[side][first]
            second_center = before_centers[side][second]
            if not (np.isfinite(first_center).all() and np.isfinite(second_center).all()):
                continue
            endpoint_speed = np.linalg.norm(second_center - first_center) / gap
            if endpoint_speed > max_anchor_speed_m_per_frame:
                continue
            alpha = (sync_index - frame_ids[first]) / gap
            target = (1.0 - alpha) * first_center + alpha * second_center
            residual = float(np.linalg.norm(current - target))
            if residual <= threshold:
                continue
            corrections.append(
                (index, side, target, residual, (first, second), "trusted_anchor_interpolation")
            )

    corrected_hand_count = 0
    corrected_joint_count = 0
    correction_records: list[dict[str, Any]] = []
    cleared_hand_count = 0
    cleared_records: list[dict[str, Any]] = []
    for index, side, target, residual, references, method in corrections:
        info = infos[(index, side)]
        hand = next(
            value for value in ordered[index]["hands"]
            if int(value.get("side", -1)) == side
        )
        candidate_points = _translated_points(hand, info["center"], target)
        reprojection = _reprojection_stats(hand, candidate_points, calibrations)
        if reprojection is not None and (
            reprojection[0] > max_reprojection_median_px
            or reprojection[1] > max_reprojection_p95_px
        ):
            _annotate_clear(hand, "clear_temporal_candidate_reprojection")
            cleared_hand_count += 1
            cleared_records.append({
                "sync_index": frame_ids[index],
                "side": side,
                "method": "clear_temporal_candidate_reprojection",
            })
            reprojection_rejected.append({
                "sync_index": frame_ids[index],
                "side": side,
                "median_px": reprojection[0],
                "p95_px": reprojection[1],
            })
            continue
        count = _translate_hand(hand, info["center"], target)
        if count <= 0:
            continue
        _annotate_replacement(
            hand, method, residual,
            tuple(frame_ids[reference] for reference in references), count,
        )
        corrected_hand_count += 1
        corrected_joint_count += count
        correction_records.append({
            "sync_index": frame_ids[index],
            "side": side,
            "method": method,
            "residual_mm": 1000.0 * residual,
            "reference_sync_indices": [frame_ids[value] for value in references],
        })

    # Pass 2: catch an isolated spike whose adjacent frames are both usable,
    # even when those frames do not meet the stricter five-camera anchor rule.
    for side in (0, 1):
        centers = [
            _palm_center(next(
                value for value in row["hands"]
                if int(value.get("side", -1)) == side
            ).get("joints_base_m", []))
            for row in ordered
        ]
        for index in range(1, len(ordered) - 1):
            info = infos.get((index, side))
            if info is None or not np.isfinite(centers[index]).all():
                continue
            previous, following = centers[index - 1], centers[index + 1]
            if not (np.isfinite(previous).all() and np.isfinite(following).all()):
                continue
            first_id, current_id, second_id = (
                frame_ids[index - 1], frame_ids[index], frame_ids[index + 1]
            )
            gap = second_id - first_id
            if gap <= 0:
                continue
            endpoint_speed = np.linalg.norm(following - previous) / gap
            if endpoint_speed > max_anchor_speed_m_per_frame:
                continue
            alpha = (current_id - first_id) / gap
            target = (1.0 - alpha) * previous + alpha * following
            residual = float(np.linalg.norm(centers[index] - target))
            if residual <= threshold:
                continue
            hand = next(
                value for value in ordered[index]["hands"]
                if int(value.get("side", -1)) == side
            )
            candidate_points = _translated_points(hand, centers[index], target)
            reprojection = _reprojection_stats(hand, candidate_points, calibrations)
            if reprojection is not None and (
                reprojection[0] > max_reprojection_median_px
                or reprojection[1] > max_reprojection_p95_px
            ):
                _annotate_clear(hand, "clear_temporal_candidate_reprojection")
                cleared_hand_count += 1
                cleared_records.append({
                    "sync_index": current_id,
                    "side": side,
                    "method": "clear_temporal_candidate_reprojection",
                })
                reprojection_rejected.append({
                    "sync_index": current_id,
                    "side": side,
                    "median_px": reprojection[0],
                    "p95_px": reprojection[1],
                })
                continue
            current_reprojection = _reprojection_stats(
                hand, np.asarray(hand.get("joints_base_m", []), dtype=np.float64),
                calibrations,
            )
            if (
                current_reprojection is not None
                and reprojection is not None
                and reprojection[0]
                > current_reprojection[0] * max_local_reprojection_ratio + 3.0
            ):
                continue
            count = _translate_hand(hand, centers[index], target)
            if count <= 0:
                continue
            _annotate_replacement(
                hand, "local_bidirectional_spike", residual,
                (first_id, second_id), count,
            )
            corrected_hand_count += 1
            corrected_joint_count += count
            centers[index] = target
            correction_records.append({
                "sync_index": current_id,
                "side": side,
                "method": "local_bidirectional_spike",
                "residual_mm": 1000.0 * residual,
                "reference_sync_indices": [first_id, second_id],
            })

    if clear_unrecoverable_endpoint:
        # A final low-support frame cannot be validated in the future.  Keep
        # its 2D evidence, but do not export a known depth jump as 3D truth.
        for side in (0, 1):
            for index in range(max(0, len(ordered) - 2), len(ordered)):
                info = infos.get((index, side))
                if info is None or index in anchor_indices[side]:
                    continue
                # Do not erase a merely sparse but still four-view endpoint.
                # This cleanup is reserved for the two-view (or worse) tail
                # where no independent future check exists.
                if info["camera_count"] > 2:
                    continue
                future_anchor = [
                    value for value in anchor_indices[side]
                    if frame_ids[value] > frame_ids[index]
                ]
                previous_anchor = [
                    value for value in anchor_indices[side]
                    if frame_ids[value] < frame_ids[index]
                ]
                if future_anchor or not previous_anchor:
                    continue
                previous = max(previous_anchor, key=lambda value: frame_ids[value])
                current = info["center"]
                previous_center = before_centers[side][previous]
                if not (
                    np.isfinite(current).all()
                    and np.isfinite(previous_center).all()
                ):
                    continue
                if np.linalg.norm(current - previous_center) <= threshold:
                    continue
                hand = next(
                    value for value in ordered[index]["hands"]
                    if int(value.get("side", -1)) == side
                )
                _annotate_clear(hand, "clear_untrusted_temporal_endpoint")
                cleared_hand_count += 1
                cleared_records.append({
                    "sync_index": frame_ids[index],
                    "side": side,
                    "method": "clear_untrusted_temporal_endpoint",
                })

    after_centers = {side: [] for side in (0, 1)}
    for row in ordered:
        for side in (0, 1):
            hand = next(
                (value for value in row.get("hands", [])
                 if int(value.get("side", -1)) == side),
                None,
            )
            after_centers[side].append(
                _palm_center(hand.get("joints_base_m", []))
                if hand is not None else np.full(3, np.nan)
            )
    return {
        "enabled": True,
        "algorithm": ALGORITHM,
        "parameters": {
            "radius": radius,
            "replacement_threshold_m": threshold,
            "min_anchor_cameras": min_anchor_cameras,
            "min_anchor_palm_inliers": min_anchor_palm_inliers,
            "max_anchor_reprojection_px": max_anchor_reprojection_px,
            "max_anchor_speed_m_per_frame": max_anchor_speed_m_per_frame,
            "duplicate_iou": duplicate_iou,
            "clear_unrecoverable_endpoint": bool(clear_unrecoverable_endpoint),
            "max_reprojection_median_px": max_reprojection_median_px,
            "max_reprojection_p95_px": max_reprojection_p95_px,
            "max_local_reprojection_ratio": max_local_reprojection_ratio,
        },
        "duplicate_detector_frame_count": duplicate_count,
        "trusted_anchor_frame_count": {
            str(side): len(anchor_indices[side]) for side in (0, 1)
        },
        "corrected_hand_count": corrected_hand_count,
        "corrected_joint_count": corrected_joint_count,
        "cleared_hand_count": cleared_hand_count,
        "reprojection_rejected_count": len(reprojection_rejected),
        "reprojection_rejected": reprojection_rejected,
        "corrections": correction_records,
        "cleared": cleared_records,
        "motion_before": {
            str(side): _motion_stats(before_centers[side], frame_ids)
            for side in (0, 1)
        },
        "motion_after": {
            str(side): _motion_stats(after_centers[side], frame_ids)
            for side in (0, 1)
        },
    }
