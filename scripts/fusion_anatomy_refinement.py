#!/usr/bin/env python3
"""Offline anatomical and temporal refinement for fused 21-joint hands."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np

from camera_models import project_points
from stabilize_hand_3d import (
    PALM_FRAME_JOINTS,
    acceleration_metric,
    active_ranges,
    bone_error_metric,
    constrain_bones,
    displacement_metric,
    estimate_bone_lengths,
    palm_normalized_3d_coordinates,
    palm_normalized_3d_step_metric,
    reject_bone_outliers,
    reject_observation_outliers,
    smooth_3d_landmarks_in_palm_frame,
    stabilize_once,
)


ALGORITHM = "robust_temporal_fixed_bone_continuity_guarded_v3"


@dataclass(frozen=True)
class FusionAnatomyConfig:
    outlier_window: int = 4
    outlier_distance_m: float = 0.10
    max_hand_radius_m: float = 0.20
    bone_outlier_absolute_m: float = 0.025
    bone_outlier_relative: float = 0.45
    max_gap: int = 3
    smoothing_radius: int = 2
    local_shape_strength: float = 0.55
    bone_iterations: int = 20
    bone_strength: float = 0.85
    final_bone_iterations: int = 12
    final_bone_strength: float = 0.80
    reliable_adjustment_blend: float = 0.35
    max_reliable_adjustment_m: float = 0.020
    max_reprojection_regression_px: float = 15.0
    max_reprojection_shift_px: float = 35.0
    large_reprojection_improvement_ratio: float = 0.70
    temporal_radius: int = 4
    temporal_min_neighbors_per_side: int = 2
    temporal_palm_residual_m: float = 0.070
    temporal_palm_strong_residual_m: float = 0.140
    temporal_global_min_rejected_joints: int = 6
    temporal_local_residual: float = 0.90
    temporal_local_min_joints: int = 2
    temporal_blend: float = 0.85
    temporal_max_global_adjustment_m: float = 0.80
    temporal_max_local_adjustment_m: float = 0.12
    correction_threshold_m: float = 0.002

    def validate(self) -> None:
        if min(
            self.outlier_window,
            self.max_gap,
            self.smoothing_radius,
            self.bone_iterations,
            self.final_bone_iterations,
            self.temporal_radius,
            self.temporal_min_neighbors_per_side,
            self.temporal_global_min_rejected_joints,
            self.temporal_local_min_joints,
        ) < 0:
            raise ValueError("anatomy windows and iterations must be non-negative")
        if min(
            self.outlier_distance_m,
            self.max_hand_radius_m,
            self.bone_outlier_absolute_m,
            self.bone_outlier_relative,
            self.max_reliable_adjustment_m,
            self.max_reprojection_regression_px,
            self.max_reprojection_shift_px,
            self.temporal_palm_residual_m,
            self.temporal_palm_strong_residual_m,
            self.temporal_local_residual,
            self.temporal_max_global_adjustment_m,
            self.temporal_max_local_adjustment_m,
            self.correction_threshold_m,
        ) <= 0.0:
            raise ValueError("anatomy thresholds must be positive")
        for name, value in (
            ("local-shape-strength", self.local_shape_strength),
            ("bone-strength", self.bone_strength),
            ("final-bone-strength", self.final_bone_strength),
            ("reliable-adjustment-blend", self.reliable_adjustment_blend),
            ("large-reprojection-improvement-ratio", self.large_reprojection_improvement_ratio),
            ("temporal-blend", self.temporal_blend),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _weighted_rigid_alignment(
    source: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    weights = np.maximum(np.asarray(weights, dtype=np.float64), 1e-6)
    weights /= weights.sum()
    source_center = np.sum(source * weights[:, None], axis=0)
    target_center = np.sum(target * weights[:, None], axis=0)
    covariance = (source - source_center).T @ (
        (target - target_center) * weights[:, None]
    )
    u, _singular, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T
    translation = target_center - rotation @ source_center
    return rotation, translation


def _preserve_palm_pose(
    refined: np.ndarray,
    raw: np.ndarray,
    observed: np.ndarray,
    accepted: np.ndarray,
    confidence: np.ndarray,
) -> np.ndarray:
    output = refined.copy()
    for side in range(refined.shape[0]):
        for frame in range(refined.shape[1]):
            if not np.any(observed[side, frame]):
                continue
            anchors = PALM_FRAME_JOINTS[
                accepted[side, frame, PALM_FRAME_JOINTS]
                & np.isfinite(refined[side, frame, PALM_FRAME_JOINTS]).all(axis=1)
            ]
            if len(anchors) >= 3:
                rotation, translation = _weighted_rigid_alignment(
                    refined[side, frame, anchors],
                    raw[side, frame, anchors],
                    confidence[side, frame, anchors],
                )
                finite = np.isfinite(output[side, frame]).all(axis=1)
                output[side, frame, finite] = (
                    rotation @ output[side, frame, finite].T
                ).T + translation
            elif (
                observed[side, frame, 0]
                and np.all(np.isfinite(output[side, frame, 0]))
            ):
                output[side, frame] += (
                    raw[side, frame, 0] - output[side, frame, 0]
                )
    return output


def _fill_from_rigid_neighbors(
    points: np.ndarray,
    valid: np.ndarray,
    raw: np.ndarray,
    observed: np.ndarray,
    accepted: np.ndarray,
    confidence: np.ndarray,
    radius: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fill rejected joints only when two nearby poses support the same estimate."""
    output = points.copy()
    output_valid = valid.copy()
    filled = np.zeros_like(valid)
    if radius <= 0:
        return output, output_valid, filled
    for side in range(points.shape[0]):
        for frame in range(points.shape[1]):
            missing = np.flatnonzero(observed[side, frame] & ~output_valid[side, frame])
            if not len(missing):
                continue
            anchors = PALM_FRAME_JOINTS[accepted[side, frame, PALM_FRAME_JOINTS]]
            if len(anchors) < 3:
                continue
            lo = max(0, frame - radius)
            hi = min(points.shape[1], frame + radius + 1)
            neighbor_frames = [
                index for index in range(lo, hi)
                if index != frame
                and np.all(output_valid[side, index, anchors])
            ]
            if len(neighbor_frames) < 2:
                continue
            alignments: dict[int, tuple[np.ndarray, np.ndarray]] = {}
            for neighbor in neighbor_frames:
                alignments[neighbor] = _weighted_rigid_alignment(
                    output[side, neighbor, anchors],
                    raw[side, frame, anchors],
                    confidence[side, frame, anchors],
                )
            for joint in missing:
                estimates = []
                weights = []
                for neighbor in neighbor_frames:
                    if not output_valid[side, neighbor, joint]:
                        continue
                    rotation, translation = alignments[neighbor]
                    estimates.append(
                        rotation @ output[side, neighbor, joint] + translation
                    )
                    weights.append(radius + 1 - abs(neighbor - frame))
                if len(estimates) < 2:
                    continue
                estimates_array = np.asarray(estimates, dtype=np.float64)
                centre = np.median(estimates_array, axis=0)
                distances = np.linalg.norm(estimates_array - centre, axis=1)
                if len(estimates_array) >= 3:
                    limit = max(0.025, float(np.median(distances)) * 2.5)
                    inliers = distances <= limit
                    if np.count_nonzero(inliers) >= 2:
                        estimates_array = estimates_array[inliers]
                        weights = np.asarray(weights)[inliers].tolist()
                output[side, frame, joint] = np.average(
                    estimates_array, axis=0, weights=np.asarray(weights)
                )
                output_valid[side, frame, joint] = True
                filled[side, frame, joint] = True
    return output, output_valid, filled


def _arrays_from_rows(
    rows: list[dict[str, Any]], camera_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[tuple[int, int], dict[str, Any]]]:
    frame_count = max((int(row["sync_index"]) for row in rows), default=-1) + 1
    raw = np.full((2, frame_count, 21, 3), np.nan, dtype=np.float64)
    observed = np.zeros((2, frame_count, 21), dtype=bool)
    confidence = np.zeros((2, frame_count, 21), dtype=np.float64)
    hands: dict[tuple[int, int], dict[str, Any]] = {}
    for row in rows:
        frame = int(row["sync_index"])
        for hand in row["hands"]:
            side = int(hand["side"])
            points = np.asarray(hand["joints_base_m"], dtype=np.float64)
            support = np.asarray(hand["inlier_view_counts"], dtype=np.float64)
            if points.shape != (21, 3) or support.shape != (21,):
                raise ValueError(
                    f"invalid fused hand at sync={frame}, side={side}"
                )
            finite = np.isfinite(points).all(axis=1)
            raw[side, frame] = points
            observed[side, frame] = finite
            confidence[side, frame] = np.where(
                finite,
                np.clip(support / max(camera_count, 2), 0.10, 1.0),
                0.0,
            )
            hands[(frame, side)] = hand
    return raw, observed, confidence, hands


def _affected_finger_scope(rejected: np.ndarray) -> np.ndarray:
    """Limit weak adjustments to fingers that actually contain an outlier."""
    affected = np.zeros_like(rejected)
    finger_groups = ((1, 5), (5, 9), (9, 13), (13, 17), (17, 21))
    for side in range(rejected.shape[0]):
        for frame in range(rejected.shape[1]):
            if rejected[side, frame, 0]:
                affected[side, frame] = True
                continue
            for start, end in finger_groups:
                if np.any(rejected[side, frame, start:end]):
                    affected[side, frame, start:end] = True
    return affected


def _theil_sen_zero_prediction(
    times: np.ndarray, values: np.ndarray,
) -> np.ndarray:
    """Robustly predict a vector at time zero from surrounding samples."""
    times = np.asarray(times, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    slopes = []
    for first in range(len(times)):
        for second in range(first + 1, len(times)):
            delta = times[second] - times[first]
            if abs(delta) > 1e-9:
                slopes.append((values[second] - values[first]) / delta)
    if not slopes:
        return np.median(values, axis=0)
    slope = np.median(np.asarray(slopes), axis=0)
    time_shape = (-1,) + (1,) * (values.ndim - 1)
    return np.median(values - times.reshape(time_shape) * slope, axis=0)


def _temporal_neighbor_frames(
    frame: int,
    frame_valid: np.ndarray,
    radius: int,
    minimum_per_side: int,
) -> np.ndarray:
    lo = max(0, frame - radius)
    hi = min(len(frame_valid), frame + radius + 1)
    before = np.flatnonzero(frame_valid[lo:frame]) + lo
    after = np.flatnonzero(frame_valid[frame + 1:hi]) + frame + 1
    if len(before) < minimum_per_side or len(after) < minimum_per_side:
        return np.empty(0, dtype=np.int32)
    return np.concatenate((before, after)).astype(np.int32)


def _cap_vector(delta: np.ndarray, maximum_m: float) -> np.ndarray:
    length = float(np.linalg.norm(delta))
    if length <= maximum_m or length <= 1e-12:
        return delta
    return delta * (maximum_m / length)


def _apply_robust_temporal_refinement(
    points: np.ndarray,
    observed: np.ndarray,
    rejected: np.ndarray,
    confidence: np.ndarray,
    bone_lengths: np.ndarray,
    config: FusionAnatomyConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Repair whole-palm and local-pose spikes supported by both time directions."""
    output = points.copy()
    _local, _origins, _axes, _scales, frame_valid = (
        palm_normalized_3d_coordinates(points, observed)
    )
    global_corrected = np.zeros(points.shape[:2], dtype=bool)
    global_residual = np.full(points.shape[:2], np.nan, dtype=np.float64)
    local_corrected = np.zeros(observed.shape, dtype=bool)
    local_residual = np.full(points.shape[:2], np.nan, dtype=np.float64)
    minimum = config.temporal_min_neighbors_per_side

    # Repair rigid palm-pose spikes first. A line prediction tolerates a short
    # run of bad frames and does not lag smooth, fast hand motion.
    for side in range(points.shape[0]):
        for frame in range(points.shape[1]):
            if not frame_valid[side, frame]:
                continue
            neighbors = _temporal_neighbor_frames(
                frame, frame_valid[side], config.temporal_radius, minimum
            )
            if not len(neighbors):
                continue
            times = neighbors.astype(np.float64) - frame
            predicted_anchors = _theil_sen_zero_prediction(
                times, points[side, neighbors][:, PALM_FRAME_JOINTS]
            )
            current_anchors = points[side, frame, PALM_FRAME_JOINTS]
            residual = float(np.median(np.linalg.norm(
                current_anchors - predicted_anchors, axis=1
            )))
            global_residual[side, frame] = residual
            rejected_count = int(np.count_nonzero(rejected[side, frame]))
            weak_evidence = (
                residual > config.temporal_palm_residual_m
                and rejected_count >= config.temporal_global_min_rejected_joints
            )
            strong_evidence = residual > config.temporal_palm_strong_residual_m
            if not (weak_evidence or strong_evidence):
                continue
            rotation, translation = _weighted_rigid_alignment(
                current_anchors,
                predicted_anchors,
                confidence[side, frame, PALM_FRAME_JOINTS],
            )
            current = output[side, frame]
            rigid_target = (rotation @ current.T).T + translation
            deltas = rigid_target - current
            median_delta = np.median(deltas[observed[side, frame]], axis=0)
            capped_translation = _cap_vector(
                median_delta, config.temporal_max_global_adjustment_m
            )
            deltas += capped_translation - median_delta
            output[side, frame, observed[side, frame]] += (
                config.temporal_blend * deltas[observed[side, frame]]
            )
            global_corrected[side, frame] = True

    # Repair articulation spikes in each frame's palm coordinates. Palm anchors
    # are excluded so this pass cannot create global hand motion.
    local_after, origins, axes, scales, frame_valid = (
        palm_normalized_3d_coordinates(output, observed)
    )
    finger_joints = np.setdiff1d(np.arange(21), PALM_FRAME_JOINTS)
    for side in range(points.shape[0]):
        for frame in range(points.shape[1]):
            if not frame_valid[side, frame]:
                continue
            neighbors = _temporal_neighbor_frames(
                frame, frame_valid[side], config.temporal_radius, minimum
            )
            if not len(neighbors):
                continue
            times = neighbors.astype(np.float64) - frame
            predicted = _theil_sen_zero_prediction(
                times, local_after[side, neighbors]
            )
            distances = np.linalg.norm(
                local_after[side, frame] - predicted, axis=1
            )
            local_residual[side, frame] = float(
                np.nanmedian(distances[finger_joints])
            )
            outlier_joints = finger_joints[
                observed[side, frame, finger_joints]
                & np.isfinite(predicted[finger_joints]).all(axis=1)
                & (distances[finger_joints] > config.temporal_local_residual)
            ]
            if len(outlier_joints) < config.temporal_local_min_joints:
                continue
            for joint in outlier_joints:
                target = origins[side, frame] + scales[side, frame] * (
                    predicted[joint] @ axes[side, frame]
                )
                delta = _cap_vector(
                    target - output[side, frame, joint],
                    config.temporal_max_local_adjustment_m,
                )
                output[side, frame, joint] += config.temporal_blend * delta
                local_corrected[side, frame, joint] = True

    if np.any(local_corrected):
        temporal_confidence = np.maximum(confidence, 0.80)
        temporal_confidence[local_corrected] = 0.05
        constrained = constrain_bones(
            output,
            observed,
            temporal_confidence,
            bone_lengths,
            config.final_bone_iterations,
            config.final_bone_strength,
        )
        output[local_corrected] = constrained[local_corrected]
    output[~observed] = np.nan
    return output, global_corrected, local_corrected, global_residual, local_residual


def _palm_step_metric(points: np.ndarray, valid: np.ndarray) -> dict[str, Any]:
    centers = np.full(points.shape[:2] + (3,), np.nan, dtype=np.float64)
    frame_valid = np.all(valid[:, :, PALM_FRAME_JOINTS], axis=2)
    centers[frame_valid] = np.median(
        points[:, :, PALM_FRAME_JOINTS][frame_valid], axis=1
    )
    samples = []
    for side in range(points.shape[0]):
        consecutive = frame_valid[side, 1:] & frame_valid[side, :-1]
        if np.any(consecutive):
            samples.extend((1000.0 * np.linalg.norm(
                centers[side, 1:][consecutive]
                - centers[side, :-1][consecutive],
                axis=1,
            )).tolist())
    values = np.asarray(samples, dtype=np.float64)
    return {
        "sample_count": int(len(values)),
        "median_mm": float(np.median(values)) if len(values) else None,
        "p95_mm": float(np.percentile(values, 95)) if len(values) else None,
        "p99_mm": float(np.percentile(values, 99)) if len(values) else None,
        "max_mm": float(np.max(values)) if len(values) else None,
    }


def _project_base_point(camera: Any, point_base: np.ndarray) -> np.ndarray | None:
    rotation = camera.T_base_camera[:3, :3]
    center = camera.T_base_camera[:3, 3]
    point_camera = rotation.T @ (point_base - center)
    pixels, valid = project_points(camera, point_camera[None])
    return pixels[0] if bool(valid[0]) else None


def _apply_reprojection_guard(
    refined: np.ndarray,
    raw: np.ndarray,
    hands: Mapping[tuple[int, int], dict[str, Any]],
    calibrations: Mapping[str, Any] | None,
    config: FusionAnatomyConfig,
    trusted_temporal_hands: np.ndarray | None = None,
) -> tuple[np.ndarray, int, int, dict[str, Any] | None]:
    if not calibrations:
        return refined, 0, 0, None
    output = refined.copy()
    limited_count = 0
    reverted_count = 0
    for (frame, side), hand in hands.items():
        if (
            trusted_temporal_hands is not None
            and trusted_temporal_hands[side, frame]
        ):
            continue
        views = [
            (calibrations[camera_id], np.asarray(view["joints_2d"], dtype=np.float64))
            for camera_id, view in hand.get("views", {}).items()
            if camera_id in calibrations
            and int(view.get("inlier_joint_count", 0)) > 0
            and np.asarray(view.get("joints_2d", [])).shape == (21, 2)
        ]
        if not views:
            continue
        for joint in range(21):
            start = raw[side, frame, joint]
            target = refined[side, frame, joint]
            if not (np.all(np.isfinite(start)) and np.all(np.isfinite(target))):
                continue
            delta = target - start
            if float(np.linalg.norm(delta)) <= config.correction_threshold_m:
                continue
            evidence = []
            for camera, observations in views:
                observation = observations[joint]
                raw_pixel = _project_base_point(camera, start)
                if raw_pixel is None or not np.all(np.isfinite(observation)):
                    continue
                evidence.append((camera, observation, raw_pixel))
            if len(evidence) < 2:
                output[side, frame, joint] = start
                limited_count += 1
                reverted_count += 1
                continue
            raw_residual = float(np.median([
                np.linalg.norm(raw_pixel - observation)
                for _camera, observation, raw_pixel in evidence
            ]))

            def allowed(alpha: float) -> bool:
                point = start + alpha * delta
                shifts = []
                residuals = []
                for camera, observation, raw_pixel in evidence:
                    pixel = _project_base_point(camera, point)
                    if pixel is None:
                        return False
                    shifts.append(float(np.linalg.norm(pixel - raw_pixel)))
                    residuals.append(float(np.linalg.norm(pixel - observation)))
                residual_median = float(np.median(residuals))
                strongly_supported = (
                    residual_median
                    <= config.large_reprojection_improvement_ratio * raw_residual
                )
                return (
                    (max(shifts) <= config.max_reprojection_shift_px or strongly_supported)
                    and residual_median
                    <= raw_residual + config.max_reprojection_regression_px
                )

            if allowed(1.0):
                continue
            low, high = 0.0, 1.0
            for _ in range(10):
                middle = 0.5 * (low + high)
                if allowed(middle):
                    low = middle
                else:
                    high = middle
            output[side, frame, joint] = start + low * delta
            limited_count += 1
            reverted_count += int(low < 0.05)

    residual_before = []
    residual_after = []
    projected_shift = []
    for (frame, side), hand in hands.items():
        for camera_id, view in hand.get("views", {}).items():
            if (
                camera_id not in calibrations
                or int(view.get("inlier_joint_count", 0)) <= 0
            ):
                continue
            observations = np.asarray(view.get("joints_2d", []), dtype=np.float64)
            if observations.shape != (21, 2):
                continue
            for joint in range(21):
                before = _project_base_point(
                    calibrations[camera_id], raw[side, frame, joint]
                )
                after = _project_base_point(
                    calibrations[camera_id], output[side, frame, joint]
                )
                if (
                    before is None
                    or after is None
                    or not np.all(np.isfinite(observations[joint]))
                ):
                    continue
                residual_before.append(float(np.linalg.norm(
                    before - observations[joint]
                )))
                residual_after.append(float(np.linalg.norm(
                    after - observations[joint]
                )))
                projected_shift.append(float(np.linalg.norm(after - before)))

    def summarize(values: list[float]) -> dict[str, float | int | None]:
        array = np.asarray(values, dtype=np.float64)
        return {
            "sample_count": int(len(array)),
            "median_px": float(np.median(array)) if len(array) else None,
            "p95_px": float(np.percentile(array, 95)) if len(array) else None,
            "p99_px": float(np.percentile(array, 99)) if len(array) else None,
            "max_px": float(np.max(array)) if len(array) else None,
        }

    return output, limited_count, reverted_count, {
        "residual_before": summarize(residual_before),
        "residual_after": summarize(residual_after),
        "raw_to_refined_shift": summarize(projected_shift),
    }


def refine_accepted_rows(
    rows: list[dict[str, Any]],
    camera_count: int,
    config: FusionAnatomyConfig,
    calibrations: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Modify accepted rows in place and return sequence-level diagnostics."""
    config.validate()
    if not rows:
        return {
            "enabled": True,
            "algorithm": ALGORITHM,
            "parameters": config.to_dict(),
            "corrected_joint_count": 0,
            "corrected_hand_count": 0,
        }
    raw, observed, confidence, hands = _arrays_from_rows(rows, camera_count)
    ranges = active_ranges(observed)
    accepted, rejected = reject_observation_outliers(
        raw,
        observed,
        ranges,
        config.outlier_window,
        config.outlier_distance_m,
        config.max_hand_radius_m,
    )
    preliminary = raw.copy()
    preliminary[~accepted] = np.nan
    preliminary_confidence = confidence.copy()
    preliminary_confidence[~accepted] = 0.0
    bone_lengths = estimate_bone_lengths(
        preliminary, accepted, preliminary_confidence
    )
    accepted, bone_rejected = reject_bone_outliers(
        raw,
        accepted,
        preliminary_confidence,
        bone_lengths,
        config.bone_outlier_absolute_m,
        config.bone_outlier_relative,
    )
    rejected |= bone_rejected
    prepared = stabilize_once(
        raw,
        accepted,
        confidence,
        config.max_gap,
        config.smoothing_radius,
        12,
        0.80,
    )

    candidate_seed, candidate_valid, rigid_neighbor_filled = _fill_from_rigid_neighbors(
        prepared["stabilized"],
        prepared["valid"],
        raw,
        observed,
        accepted,
        confidence,
        max(config.outlier_window, config.max_gap),
    )
    candidate_confidence = prepared["confidence"].copy()
    candidate_confidence[rigid_neighbor_filled] = 0.20
    candidate = constrain_bones(
        candidate_seed,
        candidate_valid,
        candidate_confidence,
        bone_lengths,
        config.bone_iterations,
        config.bone_strength,
    )
    candidate = smooth_3d_landmarks_in_palm_frame(
        candidate,
        candidate_valid,
        config.smoothing_radius,
        config.local_shape_strength,
    )
    candidate = constrain_bones(
        candidate,
        candidate_valid,
        candidate_confidence,
        bone_lengths,
        config.final_bone_iterations,
        config.final_bone_strength,
    )
    candidate = _preserve_palm_pose(
        candidate, raw, observed, accepted, confidence
    )

    # Rejected observations are replaced only when the temporal pass produced a
    # finite substitute. Reliable observations receive a small, capped blend
    # toward the anatomical candidate instead of being overwritten wholesale.
    # This keeps per-frame multiview evidence dominant while repairing isolated
    # depth spikes and unstable finger shapes.
    candidate_finite = np.isfinite(candidate).all(axis=-1)
    refined = raw.copy()
    reconstructable_outliers = rejected & observed & candidate_finite
    refined[reconstructable_outliers] = candidate[reconstructable_outliers]
    reliable = (
        accepted
        & observed
        & candidate_finite
        & _affected_finger_scope(rejected)
    )
    reliable_delta = candidate - raw
    reliable_norm = np.linalg.norm(reliable_delta, axis=-1)
    reliable_scale = np.minimum(
        config.reliable_adjustment_blend,
        config.max_reliable_adjustment_m / np.maximum(reliable_norm, 1e-12),
    )
    refined[reliable] = (
        raw[reliable]
        + reliable_delta[reliable] * reliable_scale[reliable, None]
    )
    refined[~observed] = np.nan
    (
        refined,
        reprojection_limited,
        reprojection_reverted,
        reprojection_diagnostics,
    ) = _apply_reprojection_guard(refined, raw, hands, calibrations, config)

    before_temporal = refined.copy()
    (
        temporal_candidate,
        temporal_global_corrected,
        temporal_local_corrected,
        temporal_global_residual,
        temporal_local_residual,
    ) = _apply_robust_temporal_refinement(
        refined,
        observed,
        rejected,
        confidence,
        bone_lengths,
        config,
    )
    (
        refined,
        temporal_reprojection_limited,
        temporal_reprojection_reverted,
        temporal_reprojection_diagnostics,
    ) = _apply_reprojection_guard(
        temporal_candidate,
        raw,
        hands,
        calibrations,
        config,
        trusted_temporal_hands=(
            temporal_global_corrected
            & (temporal_global_residual > config.temporal_palm_strong_residual_m)
            & (
                np.count_nonzero(rejected, axis=2)
                >= config.temporal_global_min_rejected_joints
            )
        ),
    )
    trusted_temporal_hands = (
        temporal_global_corrected
        & (temporal_global_residual > config.temporal_palm_strong_residual_m)
        & (
            np.count_nonzero(rejected, axis=2)
            >= config.temporal_global_min_rejected_joints
        )
    )

    displacements = np.linalg.norm(refined - raw, axis=-1)
    corrected = observed & (displacements > config.correction_threshold_m)
    repaired_outliers = reconstructable_outliers & corrected
    corrected_hands = 0
    for (frame, side), hand in hands.items():
        values = displacements[side, frame, observed[side, frame]]
        frame_corrected = corrected[side, frame]
        corrected_count = int(np.count_nonzero(frame_corrected))
        corrected_hands += int(corrected_count > 0)
        hand["unrefined_joints_base_m"] = hand["joints_base_m"]
        hand["joints_base_m"] = refined[side, frame].tolist()
        hand["anatomy_refinement"] = {
            "applied": corrected_count > 0,
            "algorithm": ALGORITHM,
            "corrected_joint_count": corrected_count,
            "rejected_observation_count": int(
                np.count_nonzero(rejected[side, frame])
            ),
            "interpolated_joint_count": int(
                np.count_nonzero(prepared["interpolated"][side, frame])
            ),
            "rigid_neighbor_filled_count": int(
                np.count_nonzero(rigid_neighbor_filled[side, frame])
            ),
            "repaired_outlier_count": int(
                np.count_nonzero(repaired_outliers[side, frame])
            ),
            "unrepaired_outlier_count": int(
                np.count_nonzero(
                    rejected[side, frame] & ~repaired_outliers[side, frame]
                )
            ),
            "temporal_global_corrected": bool(
                temporal_global_corrected[side, frame]
            ),
            "temporal_global_reprojection_override": bool(
                trusted_temporal_hands[side, frame]
            ),
            "temporal_global_residual_mm": (
                float(1000.0 * temporal_global_residual[side, frame])
                if np.isfinite(temporal_global_residual[side, frame]) else None
            ),
            "temporal_local_corrected_joint_count": int(
                np.count_nonzero(temporal_local_corrected[side, frame])
            ),
            "temporal_local_residual": (
                float(temporal_local_residual[side, frame])
                if np.isfinite(temporal_local_residual[side, frame]) else None
            ),
            "displacement_median_mm": (
                float(np.median(values) * 1000.0) if len(values) else None
            ),
            "displacement_p95_mm": (
                float(np.percentile(values, 95) * 1000.0) if len(values) else None
            ),
            "displacement_max_mm": (
                float(np.max(values) * 1000.0) if len(values) else None
            ),
        }

    return {
        "enabled": True,
        "algorithm": ALGORITHM,
        "parameters": config.to_dict(),
        "input_observation_count": int(np.count_nonzero(observed)),
        "accepted_observation_count": int(np.count_nonzero(accepted)),
        "rejected_observation_count": int(np.count_nonzero(rejected)),
        "interpolated_joint_count": int(
            np.count_nonzero(prepared["interpolated"])
        ),
        "rigid_neighbor_filled_count": int(
            np.count_nonzero(rigid_neighbor_filled)
        ),
        "reprojection_limited_joint_count": reprojection_limited,
        "reprojection_reverted_joint_count": reprojection_reverted,
        "reprojection_diagnostics": reprojection_diagnostics,
        "temporal_reprojection_limited_joint_count": temporal_reprojection_limited,
        "temporal_reprojection_reverted_joint_count": temporal_reprojection_reverted,
        "temporal_reprojection_diagnostics": temporal_reprojection_diagnostics,
        "temporal_global_corrected_hand_count": int(
            np.count_nonzero(temporal_global_corrected)
        ),
        "temporal_global_reprojection_override_hand_count": int(
            np.count_nonzero(trusted_temporal_hands)
        ),
        "temporal_local_corrected_joint_count": int(
            np.count_nonzero(temporal_local_corrected)
        ),
        "repaired_outlier_count": int(np.count_nonzero(repaired_outliers)),
        "unrepaired_outlier_count": int(
            np.count_nonzero(rejected & ~repaired_outliers)
        ),
        "corrected_joint_count": int(np.count_nonzero(corrected)),
        "corrected_hand_count": corrected_hands,
        "bone_lengths_mm": (bone_lengths * 1000.0).tolist(),
        "bone_error_before": bone_error_metric(raw, observed, bone_lengths),
        "bone_error_after": bone_error_metric(refined, observed, bone_lengths),
        "acceleration_median_before_mm": acceleration_metric(raw, observed),
        "acceleration_median_after_mm": acceleration_metric(refined, observed),
        "palm_normalized_step_before": palm_normalized_3d_step_metric(
            raw, observed
        ),
        "palm_normalized_step_after": palm_normalized_3d_step_metric(
            refined, observed
        ),
        "palm_normalized_step_before_temporal": palm_normalized_3d_step_metric(
            before_temporal, observed
        ),
        "palm_step_before": _palm_step_metric(raw, observed),
        "palm_step_before_temporal": _palm_step_metric(before_temporal, observed),
        "palm_step_after": _palm_step_metric(refined, observed),
        "displacement": displacement_metric(refined, raw, observed),
        "reliable_adjustment_displacement": displacement_metric(
            refined, raw, reliable
        ),
        "repaired_outlier_displacement": displacement_metric(
            refined, raw, repaired_outliers
        ),
    }
