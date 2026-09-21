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


ALGORITHM = "robust_temporal_fixed_bone_continuity_guarded_v4"
TRACKING_ALGORITHM = "confidence_guided_bidirectional_hand_tracking_v1"


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
    temporal_min_palm_scale_m: float = 0.030
    temporal_max_palm_scale_m: float = 0.180
    temporal_max_reference_hand_radius_m: float = 0.30
    allow_strong_reprojection_override: bool = False
    tracking_radius: int = 6
    tracking_min_neighbors_per_side: int = 2
    tracking_min_good_joints: int = 18
    tracking_min_reference_support: float = 3.0
    tracking_max_reference_corrections: int = 8
    # Three-camera support is a normal observation for the 1--4 subset.  A
    # value of four made every such frame look "weak" to the tracker and
    # allowed a temporal template to move an otherwise valid hand by 6--10 cm.
    tracking_apply_support: float = 3.0
    tracking_max_reprojection_median_px: float = 12.0
    tracking_max_reprojection_p95_px: float = 90.0
    tracking_palm_residual_m: float = 0.060
    tracking_shape_residual: float = 0.75
    tracking_blend: float = 0.75
    tracking_max_adjustment_m: float = 0.10
    # Low-redundancy/partial observations may still need finger repair, but a
    # temporal template must not translate the whole hand on its own.  Branch
    # selection is the authority for a rigid depth change; this gate limits
    # the fallback tracker to a small local correction.
    tracking_low_redundancy_max_adjustment_m: float = 0.035
    tracking_allow_low_redundancy_rigid: bool = False
    # Isolated depth-branch failures can affect one palm joint while the
    # remaining palm anchors are still plausible. This pass works per joint
    # and therefore does not require a complete palm frame.
    temporal_joint_spike_radius: int = 6
    temporal_joint_spike_min_neighbors_per_side: int = 2
    temporal_joint_spike_max_joints_per_hand: int = 6
    temporal_joint_spike_distance_m: float = 0.12
    temporal_joint_spike_velocity_scale: float = 3.0
    temporal_joint_spike_max_local_speed_m: float = 0.080
    temporal_joint_spike_margin_m: float = 0.025
    temporal_joint_spike_strong_distance_m: float = 0.14
    temporal_joint_spike_blend: float = 0.90
    temporal_joint_spike_max_adjustment_m: float = 0.35
    # Hand-level zero-phase guard for isolated depth-branch jumps. This is
    # deliberately separate from the per-joint guard: a wrong wrist/palm
    # branch can move most joints together while preserving low reprojection.
    temporal_palm_spike_radius: int = 12
    temporal_palm_spike_min_neighbors_per_side: int = 2
    temporal_palm_spike_distance_m: float = 0.050
    temporal_palm_spike_velocity_scale: float = 2.5
    temporal_palm_spike_margin_m: float = 0.020
    temporal_palm_spike_strong_distance_m: float = 0.13
    temporal_palm_spike_blend: float = 0.95
    temporal_palm_spike_max_adjustment_m: float = 0.35
    # Final short-window zero-phase filter catches both one-frame and short
    # two-frame branches after all anatomical passes have run.
    final_spike_radius: int = 3
    final_spike_min_neighbors_per_side: int = 1
    final_spike_distance_m: float = 0.10
    final_spike_velocity_scale: float = 3.0
    final_spike_margin_m: float = 0.020
    final_spike_blend: float = 1.0
    final_spike_max_joints_per_hand: int = 18
    # Final hand-centre trajectory guard.  This is intentionally separate
    # from the per-joint short filter: a depth branch can move a partially
    # observed hand as a rigid block while every individual finger remains
    # locally plausible.
    final_palm_radius: int = 6
    final_palm_min_neighbors_per_side: int = 1
    final_palm_distance_m: float = 0.060
    final_palm_velocity_scale: float = 2.2
    final_palm_margin_m: float = 0.015
    final_palm_blend: float = 0.90
    final_palm_max_adjustment_m: float = 0.14
    final_palm_severe_distance_m: float = 0.20
    final_palm_severe_max_adjustment_m: float = 0.42
    final_palm_min_joints: int = 3
    final_palm_run_max_length: int = 3
    final_palm_run_distance_m: float = 0.055
    final_palm_run_velocity_scale: float = 2.5
    final_palm_run_margin_m: float = 0.015
    # Conservative zero-phase smoothing for ordinary detector jitter.  This
    # is separate from the spike guards: those guards replace only obvious
    # excursions, while this pass gently reduces small frame-to-frame tremor
    # on otherwise reliable joints.
    gesture_smoothing_enabled: bool = True
    gesture_smoothing_radius: int = 2
    gesture_smoothing_blend: float = 0.30
    gesture_smoothing_max_motion_m: float = 0.080
    gesture_smoothing_max_residual_m: float = 0.045
    gesture_smoothing_max_adjustment_m: float = 0.022
    correction_threshold_m: float = 0.002
    # Anatomical constraints can occasionally move one reliable joint by a
    # very large amount when a low-baseline depth branch is present.  Keep
    # coherent whole-hand temporal corrections available, but cap isolated
    # post-processing moves so the repair stage cannot create a new jump.
    max_post_refinement_joint_adjustment_m: float = 0.120
    max_post_refinement_rigid_adjustment_m: float = 0.350

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
            self.tracking_radius,
            self.tracking_min_neighbors_per_side,
            self.tracking_min_good_joints,
            self.temporal_joint_spike_radius,
            self.temporal_joint_spike_min_neighbors_per_side,
            self.temporal_joint_spike_max_joints_per_hand,
            self.temporal_palm_spike_radius,
            self.temporal_palm_spike_min_neighbors_per_side,
            self.final_spike_radius,
            self.final_spike_min_neighbors_per_side,
            self.final_spike_max_joints_per_hand,
            self.final_palm_radius,
            self.final_palm_min_neighbors_per_side,
            self.final_palm_min_joints,
            self.final_palm_run_max_length,
            self.gesture_smoothing_radius,
        ) < 0:
            raise ValueError("anatomy windows and iterations must be non-negative")
        if self.tracking_min_good_joints < 1 or self.tracking_min_good_joints > 21:
            raise ValueError("tracking-min-good-joints must be in [1, 21]")
        if self.tracking_max_reference_corrections < 0:
            raise ValueError("tracking-max-reference-corrections must be non-negative")
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
            self.temporal_min_palm_scale_m,
            self.temporal_max_palm_scale_m,
            self.temporal_max_reference_hand_radius_m,
            self.tracking_min_reference_support,
            self.tracking_apply_support,
            self.tracking_max_reprojection_median_px,
            self.tracking_max_reprojection_p95_px,
            self.tracking_palm_residual_m,
            self.tracking_shape_residual,
            self.tracking_max_adjustment_m,
            self.tracking_low_redundancy_max_adjustment_m,
            self.temporal_joint_spike_distance_m,
            self.temporal_joint_spike_velocity_scale,
            self.temporal_joint_spike_max_local_speed_m,
            self.temporal_joint_spike_margin_m,
            self.temporal_joint_spike_strong_distance_m,
            self.temporal_joint_spike_blend,
            self.temporal_joint_spike_max_adjustment_m,
            self.temporal_palm_spike_distance_m,
            self.temporal_palm_spike_velocity_scale,
            self.temporal_palm_spike_margin_m,
            self.temporal_palm_spike_strong_distance_m,
            self.temporal_palm_spike_blend,
            self.temporal_palm_spike_max_adjustment_m,
            self.final_spike_distance_m,
            self.final_spike_velocity_scale,
            self.final_spike_margin_m,
            self.correction_threshold_m,
            self.final_palm_distance_m,
            self.final_palm_velocity_scale,
            self.final_palm_margin_m,
            self.final_palm_blend,
            self.final_palm_max_adjustment_m,
            self.final_palm_severe_distance_m,
            self.final_palm_severe_max_adjustment_m,
            self.final_palm_run_distance_m,
            self.final_palm_run_velocity_scale,
            self.final_palm_run_margin_m,
            self.gesture_smoothing_blend,
            self.gesture_smoothing_max_motion_m,
            self.gesture_smoothing_max_residual_m,
            self.gesture_smoothing_max_adjustment_m,
            self.max_post_refinement_joint_adjustment_m,
            self.max_post_refinement_rigid_adjustment_m,
        ) <= 0.0:
            raise ValueError("anatomy thresholds must be positive")
        for name, value in (
            ("local-shape-strength", self.local_shape_strength),
            ("bone-strength", self.bone_strength),
            ("final-bone-strength", self.final_bone_strength),
            ("reliable-adjustment-blend", self.reliable_adjustment_blend),
            ("large-reprojection-improvement-ratio", self.large_reprojection_improvement_ratio),
            ("temporal-blend", self.temporal_blend),
            ("tracking-blend", self.tracking_blend),
            ("temporal-joint-spike-blend", self.temporal_joint_spike_blend),
            ("temporal-palm-spike-blend", self.temporal_palm_spike_blend),
            ("final-spike-blend", self.final_spike_blend),
            ("gesture-smoothing-blend", self.gesture_smoothing_blend),
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
            if hand.get("synthetic_track_fill"):
                # Interpolated dropout recovery is useful as a temporal
                # display/track bridge, but must not become a high-confidence
                # anatomical reference or override real observations.
                confidence[side, frame, finite] = 0.10
            hands[(frame, side)] = hand
    return raw, observed, confidence, hands


def _temporal_segments_from_rows(
    rows: list[dict[str, Any]], frame_count: int,
) -> np.ndarray:
    """Build per-side segment IDs emitted by the image-track identity pass.

    Older result files do not contain segment metadata; those rows are treated
    as one continuous segment so the public refinement helpers retain their
    historical behaviour.  A missing hand remains ``-1`` and therefore cannot
    accidentally bridge two re-entry segments.
    """
    segments = np.full((2, frame_count), -1, dtype=np.int32)
    for row in rows:
        frame = int(row.get("sync_index", -1))
        if frame < 0 or frame >= frame_count:
            continue
        for hand in row.get("hands", []):
            try:
                side = int(hand.get("side", -1))
            except (TypeError, ValueError):
                continue
            if side not in (0, 1):
                continue
            value = hand.get("temporal_segment_id", 0)
            try:
                value = int(value)
            except (TypeError, ValueError):
                value = 0
            segments[side, frame] = value
    # Synthetic/direct callers do not provide metadata.  Assigning a common
    # default segment only to finite frames preserves the optional nature of
    # this guard while still letting explicit ``-1`` gaps block references.
    segments[segments < 0] = 0
    return segments


def _segment_boundary_frames(
    observed: np.ndarray, segment_ids: np.ndarray,
) -> np.ndarray:
    """Return frames adjacent to a physical image-track segment change."""
    boundary = np.zeros(observed.shape[:2], dtype=bool)
    if segment_ids.shape != observed.shape[:2]:
        return boundary
    for side in range(observed.shape[0]):
        finite_frames = np.flatnonzero(np.any(observed[side], axis=1))
        for before, after in zip(finite_frames[:-1], finite_frames[1:]):
            if int(segment_ids[side, before]) == int(segment_ids[side, after]):
                continue
            boundary[side, int(before)] = True
            boundary[side, int(after)] = True
    return boundary


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
    segment_ids: np.ndarray | None = None,
) -> np.ndarray:
    lo = max(0, frame - radius)
    hi = min(len(frame_valid), frame + radius + 1)
    valid = np.asarray(frame_valid, dtype=bool).copy()
    if segment_ids is not None:
        ids = np.asarray(segment_ids)
        if ids.shape == valid.shape and 0 <= frame < len(ids):
            valid &= ids == ids[frame]
    before = np.flatnonzero(valid[lo:frame]) + lo
    after = np.flatnonzero(valid[frame + 1:hi]) + frame + 1
    if len(before) < minimum_per_side or len(after) < minimum_per_side:
        return np.empty(0, dtype=np.int32)
    return np.concatenate((before, after)).astype(np.int32)


def _cap_vector(delta: np.ndarray, maximum_m: float) -> np.ndarray:
    length = float(np.linalg.norm(delta))
    if length <= maximum_m or length <= 1e-12:
        return delta
    return delta * (maximum_m / length)


def _physical_hand_frames(
    points: np.ndarray,
    valid: np.ndarray,
    config: FusionAnatomyConfig,
) -> np.ndarray:
    """Return frames whose finite joints form a plausible temporal reference."""
    result = np.zeros(points.shape[:2], dtype=bool)
    for side in range(points.shape[0]):
        for frame in range(points.shape[1]):
            joints = np.flatnonzero(
                valid[side, frame]
                & np.isfinite(points[side, frame]).all(axis=1)
            )
            if len(joints) < 12 or not np.all(
                valid[side, frame, PALM_FRAME_JOINTS]
            ):
                continue
            current = points[side, frame]
            palm_scale = float(np.linalg.norm(current[5] - current[17]))
            if not (
                config.temporal_min_palm_scale_m
                <= palm_scale
                <= config.temporal_max_palm_scale_m
            ):
                continue
            centre = np.median(current[joints], axis=0)
            radius = float(np.max(np.linalg.norm(current[joints] - centre, axis=1)))
            if radius <= config.temporal_max_reference_hand_radius_m:
                result[side, frame] = True
    return result


def _final_hand_envelope_valid(
    points: np.ndarray,
    observed: np.ndarray,
    config: FusionAnatomyConfig,
    minimum_joints: int = 12,
) -> bool:
    """Check the delivered pose for a disconnected/over-sized hand.

    The temporal reference gate is intentionally strict about all palm
    anchors, which is useful while choosing neighbours but too strict for a
    partially visible hand.  The final output gate therefore only requires a
    small set of finite joints and three palm anchors.  It still rejects the
    failure mode where a reprojection guard restores some fingers after a
    rigid palm correction, leaving a physically impossible long hand.
    """
    values = np.asarray(points, dtype=np.float64)
    mask = np.asarray(observed, dtype=bool)
    if values.shape != (21, 3) or mask.shape != (21,):
        return False
    finite = mask & np.isfinite(values).all(axis=1)
    if int(np.count_nonzero(finite)) < int(minimum_joints):
        return False
    palm = np.asarray(PALM_FRAME_JOINTS, dtype=np.int32)
    palm_finite = finite[palm]
    if int(np.count_nonzero(palm_finite)) < 3:
        return True
    centre = np.median(values[palm[palm_finite]], axis=0)
    radius = float(np.max(np.linalg.norm(values[finite] - centre, axis=1)))
    if not np.isfinite(radius) or radius > config.temporal_max_reference_hand_radius_m:
        return False
    # Only enforce palm width when both canonical width anchors are visible;
    # a side-on/occluded hand may legitimately have fewer than five anchors.
    if finite[5] and finite[17]:
        scale = float(np.linalg.norm(values[5] - values[17]))
        if not (
            config.temporal_min_palm_scale_m
            <= scale
            <= config.temporal_max_palm_scale_m
        ):
            return False
    return True


def _capped_updates(
    start: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    maximum_m: float,
) -> np.ndarray:
    output = start.copy()
    deltas = target - start
    norms = np.linalg.norm(deltas, axis=-1)
    scales = np.minimum(1.0, maximum_m / np.maximum(norms, 1e-12))
    output[mask] = start[mask] + deltas[mask] * scales[mask][:, None]
    return output


def _guard_post_refinement_displacements(
    refined: np.ndarray,
    raw: np.ndarray,
    observed: np.ndarray,
    config: FusionAnatomyConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Limit incoherent anatomy moves while retaining rigid track recovery.

    The anatomical solver may move a single rejected finger to satisfy a bone
    constraint.  If its parent/depth branch is wrong, that move can be larger
    than the original frame-to-frame motion and is visible as a new jump.  A
    coherent rigid correction (most joints moving by a similar amount) is
    allowed up to the larger hand-level cap; isolated corrections are clipped
    or reverted at the smaller joint-level cap.
    """
    output = np.asarray(refined, dtype=np.float64).copy()
    raw = np.asarray(raw, dtype=np.float64)
    observed = np.asarray(observed, dtype=bool)
    clipped = np.zeros(observed.shape, dtype=bool)
    reverted = np.zeros(observed.shape, dtype=bool)
    joint_limit = float(config.max_post_refinement_joint_adjustment_m)
    rigid_limit = float(config.max_post_refinement_rigid_adjustment_m)
    for side in range(output.shape[0]):
        for frame in range(output.shape[1]):
            finite = (
                observed[side, frame]
                & np.isfinite(raw[side, frame]).all(axis=1)
                & np.isfinite(output[side, frame]).all(axis=1)
            )
            if not np.any(finite):
                continue
            deltas = output[side, frame] - raw[side, frame]
            norms = np.linalg.norm(deltas, axis=1)
            values = norms[finite]
            if not len(values):
                continue
            median = float(np.median(values))
            raw_points = raw[side, frame]
            raw_finite = finite & np.isfinite(raw_points).all(axis=1)
            raw_centre = np.median(raw_points[raw_finite], axis=0) if np.any(raw_finite) else np.full(3, np.nan)
            raw_radius = np.linalg.norm(raw_points - raw_centre, axis=1)
            # A point that is already outside the physical hand envelope is a
            # genuine reconstruction outlier, not an aggressive temporal
            # correction.  Allow the anatomical repair to move such a point
            # by the larger bounded amount (the ordinary isolated-joint cap
            # would otherwise prevent recovery of obvious test/recording
            # spikes).
            gross_raw = (
                np.isfinite(raw_centre).all()
                and raw_finite
                & np.isfinite(raw_radius)
                & (raw_radius > config.max_hand_radius_m)
            )
            # A rigid correction has a compact displacement distribution.  Do
            # not let one large finger hide inside an otherwise small update.
            coherent = bool(
                median >= 0.045
                and np.percentile(values, 75) <= max(1.5 * median, 0.060)
                and np.count_nonzero(values >= 0.5 * median) >= max(5, int(0.5 * len(values)))
            )
            limit = rigid_limit if coherent else joint_limit
            offending = finite & (norms > limit) & ~gross_raw
            if np.any(offending):
                # Clipping is preferable to silently retaining a metre-scale
                # correction.  For an extremely incoherent move, revert the
                # offending joint completely so reprojection/anatomy stages do
                # not leave a disconnected bone.
                for joint in np.flatnonzero(offending):
                    if not coherent and norms[joint] > 2.0 * joint_limit:
                        output[side, frame, joint] = raw[side, frame, joint]
                        reverted[side, frame, joint] = True
                        continue
                    if norms[joint] > 1e-12:
                        output[side, frame, joint] = (
                            raw[side, frame, joint]
                            + deltas[joint] * (limit / norms[joint])
                        )
                        clipped[side, frame, joint] = True
            gross_offending = gross_raw & (norms > rigid_limit)
            for joint in np.flatnonzero(gross_offending):
                if norms[joint] <= 1e-12:
                    continue
                output[side, frame, joint] = (
                    raw[side, frame, joint]
                    + deltas[joint] * (rigid_limit / norms[joint])
                )
                clipped[side, frame, joint] = True
    output[~observed] = np.nan
    return output, clipped, reverted


def _trajectory_centers(
    points: np.ndarray,
    observed: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return robust palm centres and the number of supporting anchors."""
    points = np.asarray(points, dtype=np.float64)
    observed = np.asarray(observed, dtype=bool)
    centers = np.full(points.shape[:2] + (3,), np.nan, dtype=np.float64)
    counts = np.zeros(points.shape[:2], dtype=np.int32)
    palm = np.asarray(PALM_FRAME_JOINTS, dtype=np.int32)
    for side in range(points.shape[0]):
        for frame in range(points.shape[1]):
            finite = observed[side, frame, palm] & np.isfinite(
                points[side, frame, palm]
            ).all(axis=1)
            counts[side, frame] = int(np.count_nonzero(finite))
            if counts[side, frame] >= 3:
                centers[side, frame] = np.median(
                    points[side, frame, palm[finite]], axis=0
                )
    return centers, counts


def _nearest_trajectory_neighbors(
    centers: np.ndarray,
    frame: int,
    radius: int,
) -> tuple[int | None, int | None]:
    """Find the nearest finite trajectory samples on either side."""
    previous = None
    for index in range(frame - 1, max(-1, frame - radius - 1), -1):
        if np.all(np.isfinite(centers[index])):
            previous = index
            break
    following = None
    for index in range(frame + 1, min(len(centers), frame + radius + 1)):
        if np.all(np.isfinite(centers[index])):
            following = index
            break
    return previous, following


def _local_trajectory_edges(
    centers: np.ndarray,
    frame: int,
    radius: int = 3,
) -> tuple[np.ndarray, tuple[int | None, int | None]]:
    """Return normalized incoming/outgoing palm edges around one frame."""
    previous, following = _nearest_trajectory_neighbors(centers, frame, radius)
    values: list[float] = []
    if previous is not None:
        values.append(float(np.linalg.norm(
            centers[frame] - centers[previous]
        ) / max(frame - previous, 1)))
    if following is not None:
        values.append(float(np.linalg.norm(
            centers[following] - centers[frame]
        ) / max(following - frame, 1)))
    return np.asarray(values, dtype=np.float64), (previous, following)


def _atomic_hand_trajectory_guard(
    refined: np.ndarray,
    raw: np.ndarray,
    observed: np.ndarray,
    hands: Mapping[tuple[int, int], dict[str, Any]],
    config: FusionAnatomyConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Reject a post-processing update that makes a hand's local trajectory worse.

    The individual anatomy/tracking passes operate on different subsets of
    joints.  A palm correction can therefore survive while fingers remain on
    the old depth branch, or a tracker can move a frame toward a stale
    template and create a larger edge than the original fusion.  This final
    *atomic* check compares the complete candidate hand with the original
    fused hand in a short zero-phase neighbourhood.  When the local edge
    energy or maximum edge becomes worse, the entire hand frame is restored to
    the raw observation; no mixed palm/finger state is serialized.
    """
    output = np.asarray(refined, dtype=np.float64).copy()
    raw = np.asarray(raw, dtype=np.float64)
    observed = np.asarray(observed, dtype=bool)
    reverted = np.zeros(output.shape[:2], dtype=bool)
    raw_centers, raw_counts = _trajectory_centers(raw, observed)
    candidate_centers, candidate_counts = _trajectory_centers(
        output, observed
    )
    tolerance = 1e-6
    # A 12 mm energy tolerance avoids churn from ordinary millimetre-level
    # anatomical changes, while the max-edge guard catches a correction that
    # simply moves a jump from one side of the frame to the other.
    energy_tolerance = 0.012
    max_edge_tolerance = 0.018
    for (frame, side), _hand in hands.items():
        if not (
            np.all(np.isfinite(raw_centers[side, frame]))
            and np.all(np.isfinite(candidate_centers[side, frame]))
            and raw_counts[side, frame] >= 3
            and candidate_counts[side, frame] >= 3
        ):
            continue
        centre_delta = float(np.linalg.norm(
            candidate_centers[side, frame] - raw_centers[side, frame]
        ))
        if centre_delta <= max(config.correction_threshold_m, 0.005):
            continue
        raw_edges, raw_neighbours = _local_trajectory_edges(
            raw_centers[side], frame
        )
        candidate_edges, candidate_neighbours = _local_trajectory_edges(
            candidate_centers[side], frame
        )
        if not len(raw_edges) or not len(candidate_edges):
            continue
        # Compare the same available side(s).  If a neighbour was itself
        # corrected, the candidate sequence is intentionally evaluated as a
        # whole; this prevents two adjacent bad updates from hiding their
        # combined discontinuity.
        raw_energy = float(np.sum(raw_edges ** 2))
        candidate_energy = float(np.sum(candidate_edges ** 2))
        raw_max = float(np.max(raw_edges))
        candidate_max = float(np.max(candidate_edges))
        worsened_energy = candidate_energy > raw_energy + energy_tolerance ** 2
        worsened_edge = candidate_max > raw_max + max_edge_tolerance
        # A very large rigid displacement is unsafe even when one side of the
        # bracket is missing.  The geometric branch selector is the only stage
        # allowed to choose such a move, and it has already run before this
        # post-processing guard.
        excessive = centre_delta > config.max_post_refinement_rigid_adjustment_m
        if not (worsened_energy or worsened_edge or excessive):
            continue
        output[side, frame] = raw[side, frame]
        output[side, frame, ~observed[side, frame]] = np.nan
        reverted[side, frame] = True
    return output, reverted


def _final_center_step_guard(
    refined: np.ndarray,
    raw: np.ndarray,
    observed: np.ndarray,
    hands: Mapping[tuple[int, int], dict[str, Any]],
    config: FusionAnatomyConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Prevent refinement from creating a new whole-hand centre jump.

    The atomic guard compares local energy, which is intentionally tolerant of
    a coherent rigid repair.  That tolerance previously allowed a 80--125 mm
    tracking update to move a discontinuity from one edge of a frame to the
    other.  This last guard compares the candidate and raw centre transitions
    over exactly the same neighbouring observations.  A post-processing
    update is retained only when it does not materially increase either
    adjacent normalized step; branch selection—not anatomy—is responsible for
    large depth changes.
    """
    output = np.asarray(refined, dtype=np.float64).copy()
    raw = np.asarray(raw, dtype=np.float64)
    observed = np.asarray(observed, dtype=bool)
    reverted = np.zeros(output.shape[:2], dtype=bool)
    raw_centers, raw_counts = _trajectory_centers(raw, observed)
    candidate_centers, candidate_counts = _trajectory_centers(output, observed)
    radius = max(int(config.final_palm_radius), 1)
    # Ordinary anatomical corrections can move a centre by a few mm.  A
    # 15-mm tolerance protects those changes while still catching visible
    # tracking jumps and the old 350-mm rigid-update tail.
    step_tolerance = 0.015
    for (frame, side), _hand in hands.items():
        if not (
            raw_counts[side, frame] >= 3
            and candidate_counts[side, frame] >= 3
            and np.all(np.isfinite(raw_centers[side, frame]))
            and np.all(np.isfinite(candidate_centers[side, frame]))
        ):
            continue
        centre_delta = float(np.linalg.norm(
            candidate_centers[side, frame] - raw_centers[side, frame]
        ))
        if centre_delta <= max(config.correction_threshold_m, 0.005):
            continue
        previous, following = _nearest_trajectory_neighbors(
            raw_centers[side], frame, radius
        )
        raw_edges: list[float] = []
        candidate_edges: list[float] = []
        for neighbour in (previous, following):
            if neighbour is None or not np.all(np.isfinite(
                candidate_centers[side, neighbour]
            )):
                continue
            gap = max(abs(frame - neighbour), 1)
            raw_edges.append(float(np.linalg.norm(
                raw_centers[side, frame] - raw_centers[side, neighbour]
            )) / gap)
            candidate_edges.append(float(np.linalg.norm(
                candidate_centers[side, frame]
                - candidate_centers[side, neighbour]
            )) / gap)
        worsened = False
        if raw_edges and candidate_edges:
            raw_values = np.asarray(raw_edges, dtype=np.float64)
            candidate_values = np.asarray(candidate_edges, dtype=np.float64)
            # A valid repair can very slightly lengthen the already-short
            # edge while dramatically shrinking the bad edge.  Compare the
            # maximum and the total energy instead of vetoing on any one
            # neighbour.
            worsened = bool(
                np.max(candidate_values) > np.max(raw_values) + step_tolerance
                or np.sum(candidate_values ** 2)
                > np.sum(raw_values ** 2) + step_tolerance ** 2
            )
        excessive = centre_delta > config.max_post_refinement_rigid_adjustment_m
        if excessive or worsened:
            output[side, frame] = raw[side, frame]
            output[side, frame, ~observed[side, frame]] = np.nan
            reverted[side, frame] = True
            candidate_centers[side, frame] = raw_centers[side, frame]
    return output, reverted


def _limit_refinement_center_displacement(
    refined: np.ndarray,
    raw: np.ndarray,
    observed: np.ndarray,
    maximum_m: float = 0.10,
    exempt_hands: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Cap the rigid component introduced by post-processing.

    A true large depth correction is handled by the explicit bidirectional
    palm pass and remains subject to the atomic trajectory test.  Other
    anatomy/tracking stages should not translate a hand by more than 100 mm in
    one frame.  Applying the cap to the whole finite pose preserves bones and
    articulation, unlike clipping palm joints independently.
    """
    output = np.asarray(refined, dtype=np.float64).copy()
    raw = np.asarray(raw, dtype=np.float64)
    observed = np.asarray(observed, dtype=bool)
    limited = np.zeros(output.shape[:2], dtype=bool)
    raw_centers, raw_counts = _trajectory_centers(raw, observed)
    candidate_centers, candidate_counts = _trajectory_centers(output, observed)
    limit = max(float(maximum_m), 0.0)
    if limit <= 0.0:
        return output, limited
    for side in range(output.shape[0]):
        for frame in range(output.shape[1]):
            if (
                exempt_hands is not None
                and exempt_hands.shape == output.shape[:2]
                and bool(exempt_hands[side, frame])
            ):
                continue
            if not (
                raw_counts[side, frame] >= 3
                and candidate_counts[side, frame] >= 3
                and np.all(np.isfinite(raw_centers[side, frame]))
                and np.all(np.isfinite(candidate_centers[side, frame]))
            ):
                continue
            delta = candidate_centers[side, frame] - raw_centers[side, frame]
            distance = float(np.linalg.norm(delta))
            if distance <= limit or distance <= 1e-12:
                continue
            correction = delta * (limit / distance) - delta
            finite = observed[side, frame] & np.isfinite(
                output[side, frame]
            ).all(axis=1)
            output[side, frame, finite] += correction
            limited[side, frame] = True
    output[~observed] = np.nan
    return output, limited


def _drop_unreconstructable_hands(
    refined: np.ndarray,
    observed: np.ndarray,
    rejected: np.ndarray,
    candidate_finite: np.ndarray,
    minimum_observed_joints: int = 12,
) -> tuple[np.ndarray, np.ndarray]:
    """Hide a whole-hand outlier when no finite temporal reconstruction exists.

    A frame whose every observed joint is rejected must not fall back to the
    original triangulation: that value is precisely the one the outlier pass
    declared unsafe.  Keeping it creates a physically impossible one-frame
    jump (typically when a hand exits the cameras).  We retain partial hands,
    and only clear frames with enough observations to represent a hand but no
    finite candidate for any of them.
    """
    output = refined.copy()
    dropped = np.zeros(observed.shape[:2], dtype=bool)
    for side in range(observed.shape[0]):
        for frame in range(observed.shape[1]):
            observed_joints = observed[side, frame]
            count = int(np.count_nonzero(observed_joints))
            if count < minimum_observed_joints:
                continue
            rejected_joints = observed_joints & rejected[side, frame]
            if not np.all(rejected_joints[observed_joints]):
                continue
            # A partially finite candidate is not enough for an all-rejected
            # hand: in particular, leaving the original palm anchors while
            # replacing a few fingers still creates a large depth/identity
            # jump.  Keep the hand only when every observed joint (including
            # all palm anchors) has a temporal reconstruction.
            if np.all(candidate_finite[side, frame, observed_joints]):
                continue
            output[side, frame, observed_joints] = np.nan
            dropped[side, frame] = True
    return output, dropped


def _apply_robust_temporal_refinement(
    points: np.ndarray,
    observed: np.ndarray,
    rejected: np.ndarray,
    confidence: np.ndarray,
    bone_lengths: np.ndarray,
    config: FusionAnatomyConfig,
    segment_ids: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Repair whole-palm and local-pose spikes supported by both time directions."""
    output = points.copy()
    _local, _origins, _axes, _scales, frame_valid = (
        palm_normalized_3d_coordinates(points, observed)
    )
    physical_frames = _physical_hand_frames(points, observed, config)
    reliable_joint = (
        observed
        & ~rejected
        & np.isfinite(points).all(axis=-1)
    )
    reliable_palm_frames = (
        frame_valid
        & physical_frames
        & np.all(reliable_joint[:, :, PALM_FRAME_JOINTS], axis=2)
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
            if not (frame_valid[side, frame] and physical_frames[side, frame]):
                continue
            neighbors = _temporal_neighbor_frames(
                frame,
                reliable_palm_frames[side],
                config.temporal_radius,
                minimum,
                segment_ids[side] if segment_ids is not None else None,
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
            proposed = current.copy()
            proposed[observed[side, frame]] += (
                config.temporal_blend * deltas[observed[side, frame]]
            )
            proposed = _capped_updates(
                current,
                proposed,
                observed[side, frame],
                config.temporal_max_global_adjustment_m,
            )
            candidate_sequence = output[side:side + 1, frame:frame + 1].copy()
            candidate_sequence[0, 0] = proposed
            candidate_valid = observed[side:side + 1, frame:frame + 1]
            if not _physical_hand_frames(
                candidate_sequence, candidate_valid, config
            )[0, 0]:
                continue
            output[side, frame] = proposed
            global_corrected[side, frame] = True

    # Repair articulation spikes in each frame's palm coordinates. Palm anchors
    # are excluded so this pass cannot create global hand motion.
    local_after, origins, axes, scales, frame_valid = (
        palm_normalized_3d_coordinates(output, observed)
    )
    physical_after_global = _physical_hand_frames(output, observed, config)
    local_reference_frames = frame_valid & physical_after_global
    before_local = output.copy()
    finger_joints = np.setdiff1d(np.arange(21), PALM_FRAME_JOINTS)
    for side in range(points.shape[0]):
        for frame in range(points.shape[1]):
            if not local_reference_frames[side, frame]:
                continue
            predicted = np.full((21, 3), np.nan, dtype=np.float64)
            for joint in finger_joints:
                joint_reference_frames = (
                    local_reference_frames[side]
                    & reliable_joint[side, :, joint]
                )
                neighbors = _temporal_neighbor_frames(
                    frame,
                    joint_reference_frames,
                    config.temporal_radius,
                    minimum,
                    segment_ids[side] if segment_ids is not None else None,
                )
                if not len(neighbors):
                    continue
                times = neighbors.astype(np.float64) - frame
                predicted[joint] = _theil_sen_zero_prediction(
                    times, local_after[side, neighbors, joint]
                )
            distances = np.linalg.norm(
                local_after[side, frame] - predicted, axis=1
            )
            finite_finger_distances = distances[
                finger_joints[np.isfinite(distances[finger_joints])]
            ]
            if not len(finite_finger_distances):
                continue
            local_residual[side, frame] = float(
                np.median(finite_finger_distances)
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
        output = _capped_updates(
            before_local,
            constrained,
            local_corrected,
            config.temporal_max_local_adjustment_m,
        )
    output[~observed] = np.nan
    return output, global_corrected, local_corrected, global_residual, local_residual


def _tracking_good_frames(
    points: np.ndarray,
    observed: np.ndarray,
    hands: Mapping[tuple[int, int], dict[str, Any]],
    config: FusionAnatomyConfig,
) -> np.ndarray:
    """Select reliable poses that may be used as bidirectional track references."""
    physical = _physical_hand_frames(points, observed, config)
    good = np.zeros(points.shape[:2], dtype=bool)
    for (frame, side), hand in hands.items():
        finite = observed[side, frame] & np.isfinite(points[side, frame]).all(axis=1)
        support = np.asarray(hand.get("inlier_view_counts", []), dtype=np.float64)
        if support.shape != (21,):
            support = np.zeros(21, dtype=np.float64)
        support_values = support[finite]
        quality = hand.get("quality", {})
        # Do not use an under-conditioned stereo branch as a temporal
        # template.  Pair-spread/low-angle solutions can have excellent pixel
        # residuals while carrying the wrong depth; once admitted here they
        # contaminate both directions of the tracker.
        try:
            spread = float(quality.get("palm_pair_spread_p75_m", 0.0))
            angle = float(quality.get("palm_ray_angle_p25_deg", np.nan))
            camera_count = len(hand.get("camera_ids", []))
            palm_three = int(quality.get("palm_three_view_count", 0) or 0)
            two_fraction = float(quality.get("palm_two_view_fraction", 1.0))
        except (TypeError, ValueError):
            spread, angle, camera_count, palm_three, two_fraction = (
                float("inf"), float("nan"), 0, 0, 1.0
            )
        has_depth_metadata = any(
            key in quality
            for key in (
                "palm_pair_spread_p75_m", "palm_ray_angle_p25_deg",
                "palm_three_view_count", "palm_two_view_fraction",
            )
        )
        if has_depth_metadata and (
            (np.isfinite(spread) and spread > 0.090)
            or (np.isfinite(angle) and angle < 4.5)
            or camera_count <= 2
            or (camera_count >= 3 and (palm_three < 2 or two_fraction > 0.60))
        ):
            continue
        valid_count = int(np.count_nonzero(finite))
        if valid_count < config.tracking_min_good_joints or not len(support_values):
            continue
        anatomy = hand.get("anatomy_refinement", {})
        if int(anatomy.get("corrected_joint_count", 0)) > (
            config.tracking_max_reference_corrections
        ):
            continue
        if float(np.median(support_values)) < config.tracking_min_reference_support:
            continue
        median_reprojection = quality.get("multiview_reprojection_median_px")
        p95_reprojection = quality.get("all_view_reprojection_p95_px")
        if median_reprojection is not None and float(median_reprojection) > (
            config.tracking_max_reprojection_median_px
        ):
            continue
        if p95_reprojection is not None and float(p95_reprojection) > (
            config.tracking_max_reprojection_p95_px
        ):
            continue
        if physical[side, frame]:
            good[side, frame] = True
    return good


def _hand_identity_collision(
    frame: int,
    side: int,
    hands: Mapping[tuple[int, int], dict[str, Any]],
    wrist_threshold_px: float = 25.0,
    overlap_threshold: float = 0.55,
) -> bool:
    """Detect a likely left/right swap in a shared camera observation."""
    current = hands.get((frame, side))
    other = hands.get((frame, 1 - side))
    if not current or not other:
        return False
    for camera_id, current_view in current.get("views", {}).items():
        other_view = other.get("views", {}).get(camera_id)
        if other_view is None:
            continue
        current_points = np.asarray(current_view.get("joints_2d", []), dtype=np.float64)
        other_points = np.asarray(other_view.get("joints_2d", []), dtype=np.float64)
        if current_points.shape == (21, 2) and other_points.shape == (21, 2):
            if np.all(np.isfinite(current_points[0])) and np.all(np.isfinite(other_points[0])):
                if float(np.linalg.norm(current_points[0] - other_points[0])) <= wrist_threshold_px:
                    return True
        current_box = np.asarray(current_view.get("bbox_xyxy", []), dtype=np.float64)
        other_box = np.asarray(other_view.get("bbox_xyxy", []), dtype=np.float64)
        if current_box.shape == (4,) and other_box.shape == (4,):
            left = max(current_box[0], other_box[0])
            top = max(current_box[1], other_box[1])
            right = min(current_box[2], other_box[2])
            bottom = min(current_box[3], other_box[3])
            intersection = max(0.0, right - left) * max(0.0, bottom - top)
            area_current = max(0.0, current_box[2] - current_box[0]) * max(0.0, current_box[3] - current_box[1])
            area_other = max(0.0, other_box[2] - other_box[0]) * max(0.0, other_box[3] - other_box[1])
            union = area_current + area_other - intersection
            if union > 1e-6 and intersection / union >= overlap_threshold:
                return True
    return False


def _apply_confidence_guided_tracking(
    points: np.ndarray,
    observed: np.ndarray,
    confidence: np.ndarray,
    hands: Mapping[tuple[int, int], dict[str, Any]],
    config: FusionAnatomyConfig,
    segment_ids: np.ndarray | None = None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, Any],
    np.ndarray,
]:
    """Track unreliable hand poses from good frames on both time directions.

    The predictor first estimates the palm trajectory with a robust zero-phase
    line fit.  Neighbor poses are rigidly aligned to that predicted palm and
    their median supplies the finger configuration.  Only frames with weak
    multi-view support or a large normalized residual are changed; reliable
    fast motions are left untouched.
    """
    output = np.asarray(points, dtype=np.float64).copy()
    observed = np.asarray(observed, dtype=bool)
    confidence = np.asarray(confidence, dtype=np.float64)
    good = _tracking_good_frames(output, observed, hands, config)
    physical = _physical_hand_frames(output, observed, config)
    corrected = np.zeros_like(observed)
    hand_corrected = np.zeros(output.shape[:2], dtype=bool)
    trusted_override = np.zeros(output.shape[:2], dtype=bool)
    palm_residuals = np.full(output.shape[:2], np.nan, dtype=np.float64)
    shape_residuals = np.full(output.shape[:2], np.nan, dtype=np.float64)
    if config.tracking_radius <= 0:
        return (
            output,
            corrected,
            hand_corrected,
            palm_residuals,
            shape_residuals,
            {"algorithm": TRACKING_ALGORITHM, "good_frame_count": int(np.count_nonzero(good))},
            trusted_override,
        )

    palm_joints = PALM_FRAME_JOINTS
    finger_joints = np.setdiff1d(np.arange(21), palm_joints)
    for side in range(output.shape[0]):
        for frame in range(output.shape[1]):
            current = output[side, frame]
            if not np.all(observed[side, frame, palm_joints]) or not np.all(
                np.isfinite(current[palm_joints])
            ):
                continue
            references = _temporal_neighbor_frames(
                frame,
                good[side],
                config.tracking_radius,
                config.tracking_min_neighbors_per_side,
                segment_ids[side] if segment_ids is not None else None,
            )
            if len(references) < 2:
                continue
            reference_palm = output[side, references][:, palm_joints]
            if not np.isfinite(reference_palm).all():
                continue
            predicted_palm = _theil_sen_zero_prediction(
                references.astype(np.float64) - frame,
                reference_palm,
            )
            if not np.isfinite(predicted_palm).all():
                continue
            palm_residual = float(np.median(np.linalg.norm(
                current[palm_joints] - predicted_palm, axis=1
            )))
            palm_residuals[side, frame] = palm_residual

            aligned_poses = []
            aligned_weights = []
            for reference in references:
                pose = output[side, reference]
                finite_palm = np.isfinite(pose[palm_joints]).all(axis=1)
                if not np.all(finite_palm):
                    continue
                rotation, translation = _weighted_rigid_alignment(
                    pose[palm_joints], predicted_palm,
                    np.ones(len(palm_joints), dtype=np.float64),
                )
                aligned = np.full_like(pose, np.nan)
                finite = np.isfinite(pose).all(axis=1)
                aligned[finite] = (
                    rotation @ pose[finite].T
                ).T + translation
                aligned_poses.append(aligned)
                aligned_weights.append(
                    config.tracking_radius + 1 - abs(int(reference) - frame)
                )
            if len(aligned_poses) < 2:
                continue
            aligned_array = np.asarray(aligned_poses, dtype=np.float64)
            predicted = np.full((21, 3), np.nan, dtype=np.float64)
            for joint in range(21):
                values = aligned_array[:, joint]
                values = values[np.isfinite(values).all(axis=1)]
                if len(values):
                    predicted[joint] = np.median(values, axis=0)
            predicted[palm_joints] = predicted_palm
            predicted_valid = np.isfinite(predicted).all(axis=1)
            if not np.all(predicted_valid[palm_joints]):
                continue

            current_rotation, current_translation = _weighted_rigid_alignment(
                current[palm_joints], predicted_palm,
                np.ones(len(palm_joints), dtype=np.float64),
            )
            current_aligned = np.full_like(current, np.nan)
            finite_current = np.isfinite(current).all(axis=1)
            current_aligned[finite_current] = (
                current_rotation @ current[finite_current].T
            ).T + current_translation
            shared = (
                np.isfinite(current_aligned).all(axis=1)
                & predicted_valid
                & np.isin(np.arange(21), finger_joints)
            )
            predicted_scale = float(np.linalg.norm(predicted[5] - predicted[17]))
            if np.any(shared) and predicted_scale > 1e-5:
                # A median over all fingers hides a single bad fingertip: in
                # practice most joints can be perfectly stable while one
                # occluded finger jumps.  Use a high quantile as the hand
                # residual, while retaining the per-joint distances below to
                # limit the actual update to suspicious joints.
                shared_distances = np.linalg.norm(
                    current_aligned[shared] - predicted[shared], axis=1
                ) / predicted_scale
                # There are only sixteen non-palm joints, so even the 95th
                # percentile can still interpolate between many zeros and
                # hide one bad fingertip.  The maximum is safe here because
                # this value is used only together with the low-support gate;
                # the update itself remains per-joint and capped.
                shape_residual = float(np.max(shared_distances))
            else:
                shape_residual = float("inf")
            shape_residuals[side, frame] = shape_residual

            hand = hands.get((frame, side), {})
            support = np.asarray(hand.get("inlier_view_counts", []), dtype=np.float64)
            support_values = support[
                observed[side, frame]
            ] if support.shape == (21,) else np.empty(0)
            median_support = (
                float(np.median(support_values)) if len(support_values) else 0.0
            )
            quality = hand.get("quality", {})
            try:
                median_reprojection = quality.get(
                    "multiview_reprojection_median_px"
                )
                depth_spread = float(
                    quality.get("palm_pair_spread_p75_m", 0.0)
                )
                depth_angle = float(
                    quality.get("palm_ray_angle_p25_deg", np.nan)
                )
                camera_count = len(hand.get("camera_ids", []))
                palm_three = int(quality.get("palm_three_view_count", 0) or 0)
                palm_two_fraction = float(
                    quality.get("palm_two_view_fraction", 1.0)
                )
            except (TypeError, ValueError):
                median_reprojection = None
                depth_spread, depth_angle, camera_count = (
                    float("inf"), float("nan"), 0
                )
                palm_three, palm_two_fraction = 0, 1.0
            valid_count = int(np.count_nonzero(observed[side, frame] & predicted_valid))
            quality_bad = (
                valid_count < config.tracking_min_good_joints
                or median_support < config.tracking_apply_support
            )
            # A very large bidirectional palm residual is incompatible with a
            # physically plausible 30 FPS trajectory even when the individual
            # triangulated joints report high support (e.g. a wrong depth
            # branch selected consistently by several cameras).  Treat this
            # as a hand-level outlier; ordinary residuals still require weak
            # support so genuine fast articulation is preserved.
            severe_palm = palm_residual > max(
                0.15, 2.0 * config.tracking_palm_residual_m
            )
            identity_collision = _hand_identity_collision(frame, side, hands)
            needs_tracking = identity_collision or severe_palm or (quality_bad and (
                palm_residual > config.tracking_palm_residual_m
                or shape_residual > config.tracking_shape_residual
                or valid_count < config.tracking_min_good_joints
            ))
            # Tracking is a repair tool, not a replacement for a reliable
            # multiview measurement.  If the current pose itself has passed
            # the physical/support/reprojection gates, preserve it even when
            # a neighbouring weak branch makes the temporal predictor disagree.
            # This prevents the tracker from manufacturing 100--250 mm jumps
            # on otherwise good fast gestures.
            has_depth_metadata = any(
                key in quality
                for key in (
                    "palm_pair_spread_p75_m", "palm_ray_angle_p25_deg",
                    "palm_three_view_count", "palm_two_view_fraction",
                )
            )
            current_quality_good = bool(
                has_depth_metadata
                and valid_count >= config.tracking_min_good_joints
                and median_support >= config.tracking_apply_support
                and physical[side, frame]
                and (median_reprojection is None or float(median_reprojection) <= 10.0)
                and not (
                    (np.isfinite(depth_spread) and depth_spread > 0.090)
                    or (np.isfinite(depth_angle) and depth_angle < 4.5)
                    or camera_count <= 2
                    or (
                        camera_count >= 3
                        and (palm_three < 2 or palm_two_fraction > 0.60)
                    )
                )
            )
            if current_quality_good:
                continue
            # A 3/4-camera hand whose depth is only mildly weak is still an
            # observation, not an invitation to replace most of the pose with
            # a temporal template.  Tracking may repair isolated fingers, but
            # rigid palm motion is reserved for an explicitly severe (>150
            # mm) bidirectional contradiction.  This prevents 80--125 mm
            # centre shifts on side-on gestures that merely have fewer than
            # four inlier views per joint.
            depth_unsafe = bool(
                (np.isfinite(depth_spread) and depth_spread > 0.090)
                or (np.isfinite(depth_angle) and depth_angle < 4.5)
                or camera_count <= 2
            )
            # Direct unit-test/synthetic callers may omit camera/depth
            # metadata; retain the historical tracker behaviour for those
            # inputs.  Production fused rows always carry the metadata and
            # therefore receive the low-redundancy gate.
            low_redundancy = bool(
                has_depth_metadata
                and (
                    camera_count <= 2
                    or median_support < config.tracking_apply_support
                    or (
                        camera_count == 3
                        and (
                            palm_three < 2
                            or palm_two_fraction > 0.60
                        )
                    )
                )
            )
            if not needs_tracking:
                continue

            # Preserve joints backed by several cameras whenever possible.
            # A low-quality hand may still contain a few reliable palm/finger
            # joints; dragging all 21 joints toward a temporal template would
            # erase genuine fast motion.  Conversely, a high-support isolated
            # finger spike is eligible through the shape residual gate.
            if support.shape == (21,):
                low_support = observed[side, frame] & (
                    support < config.tracking_apply_support
                )
            else:
                low_support = observed[side, frame]
            joint_distances = np.linalg.norm(
                current_aligned - predicted, axis=1
            )
            shape_bad_joints = (
                np.isfinite(joint_distances)
                & (joint_distances > config.tracking_shape_residual * predicted_scale)
            )
            update_mask = low_support | shape_bad_joints
            # A rigid palm displacement is a hand-level failure: use all palm
            # anchors only when the current support is weak, otherwise keep the
            # measured palm pose and repair fingers independently.
            if palm_residual > config.tracking_palm_residual_m and (
                severe_palm or (quality_bad and depth_unsafe)
            ):
                update_mask[palm_joints] = True
            elif not severe_palm:
                update_mask[palm_joints] = False
            if severe_palm:
                # Keep the pose rigid during a depth-branch failure.  The
                # larger cap is only enabled for this strong bidirectional
                # contradiction and remains subject to reprojection checks.
                update_mask = observed[side, frame] & predicted_valid
                if low_redundancy and not config.tracking_allow_low_redundancy_rigid:
                    # A partial/two-view sample is not authoritative for a
                    # whole-hand translation.  Preserve its palm anchors and
                    # let the geometric branch selector (or a later explicit
                    # palm guard) decide whether the depth should change.
                    update_mask[palm_joints] = False
            finite_target = observed[side, frame] & predicted_valid & update_mask
            deltas = predicted - current
            norms = np.linalg.norm(deltas, axis=1)
            # Large branch changes must be resolved by geometric branch
            # selection/depth regularization.  The pose tracker remains a
            # conservative smoothing stage and never jumps farther than its
            # configured local cap.
            adjustment_limit = config.tracking_max_adjustment_m
            if low_redundancy:
                adjustment_limit = min(
                    adjustment_limit,
                    config.tracking_low_redundancy_max_adjustment_m,
                )
            scales = np.minimum(
                1.0,
                adjustment_limit / np.maximum(norms, 1e-12),
            )
            update = finite_target & np.isfinite(scales)
            blend = config.tracking_blend
            current[update] += blend * deltas[update] * scales[update, None]
            output[side, frame] = current
            corrected[side, frame] = update
            hand_corrected[side, frame] = bool(np.any(update))
            # Kept as an explicit diagnostic flag for compatibility, but the
            # caller never uses it to bypass reprojection safety checks.
            trusted_override[side, frame] = False

            # Record the actual target displacement for diagnostics and use by
            # downstream consumers.  This is deliberately computed after the
            # capped blend, so a large rejected correction is distinguishable
            # from a successful small one.

    return (
        output,
        corrected,
        hand_corrected,
        palm_residuals,
        shape_residuals,
        {
            "algorithm": TRACKING_ALGORITHM,
            "good_frame_count": int(np.count_nonzero(good)),
            "corrected_hand_count": int(np.count_nonzero(hand_corrected)),
            "corrected_joint_count": int(np.count_nonzero(corrected)),
            "trusted_override_hand_count": int(np.count_nonzero(trusted_override)),
        },
        trusted_override,
    )


def _apply_bidirectional_joint_spike_guard(
    points: np.ndarray,
    observed: np.ndarray,
    config: FusionAnatomyConfig,
    hands: Mapping[tuple[int, int], dict[str, Any]] | None = None,
    segment_ids: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Repair isolated 3-D branch jumps without requiring a complete palm.

    A stereo solution can put only the wrist or a few palm joints on the wrong
    depth branch while the frame still has a low image reprojection error.  The
    regular temporal passes intentionally skip incomplete palm frames, so this
    final per-joint pass uses physically plausible frames on both sides of the
    sample and a robust Theil--Sen prediction.  The velocity-adaptive threshold
    avoids flattening genuinely fast hand motion while rejecting isolated jumps
    that are several times larger than the local motion.
    """
    source = np.asarray(points, dtype=np.float64)
    output = source.copy()
    corrected = np.zeros_like(observed)
    strong = np.zeros_like(observed)
    residuals = np.full(observed.shape, np.nan, dtype=np.float64)
    if config.temporal_joint_spike_radius <= 0:
        return output, corrected, strong, residuals

    physical = _physical_hand_frames(source, observed, config)
    quality_good = np.zeros(source.shape[:2], dtype=bool)
    if hands:
        for (frame, side), hand in hands.items():
            finite = observed[side, frame] & np.isfinite(
                source[side, frame]
            ).all(axis=1)
            support = np.asarray(hand.get("inlier_view_counts", []), dtype=np.float64)
            median_support = (
                float(np.median(support[finite]))
                if support.shape == (21,) and np.any(finite) else 0.0
            )
            anatomy = hand.get("anatomy_refinement", {})
            reproj = hand.get("quality", {}).get("multiview_reprojection_median_px")
            quality_good[side, frame] = bool(
                physical[side, frame]
                and np.count_nonzero(finite) >= 18
                and median_support >= 2.0
                and int(anatomy.get("rejected_observation_count", 0)) <= 5
                and int(anatomy.get("corrected_joint_count", 0)) <= 10
                and (reproj is None or float(reproj) <= 10.0)
            )
    frame_count = source.shape[1]
    min_neighbors = config.temporal_joint_spike_min_neighbors_per_side
    updates: list[tuple[int, int, int, np.ndarray, float, bool]] = []
    for side in range(source.shape[0]):
        for frame in range(frame_count):
            for joint in range(source.shape[2]):
                current = source[side, frame, joint]
                if not (observed[side, frame, joint] and np.all(np.isfinite(current))):
                    continue
                reference_mask = (
                    (quality_good[side] | physical[side])
                    & observed[side, :, joint]
                    & np.isfinite(source[side, :, joint]).all(axis=1)
                )
                # If enough high-quality anchors exist, exclude corrected or
                # weak frames entirely. This prevents a two/three-frame bad
                # depth run from teaching the predictor its own wrong branch.
                if np.count_nonzero(quality_good[side]) >= (
                    2 * min_neighbors
                ):
                    strict_mask = (
                        quality_good[side]
                        & observed[side, :, joint]
                        & np.isfinite(source[side, :, joint]).all(axis=1)
                    )
                    strict_refs = _temporal_neighbor_frames(
                        frame, strict_mask,
                        config.temporal_joint_spike_radius,
                        min_neighbors,
                        segment_ids[side] if segment_ids is not None else None,
                    )
                    if len(strict_refs) >= 2:
                        reference_mask = strict_mask
                references = _temporal_neighbor_frames(
                    frame,
                    reference_mask,
                    config.temporal_joint_spike_radius,
                    min_neighbors,
                    segment_ids[side] if segment_ids is not None else None,
                )
                if len(references) < 2:
                    continue
                predicted = _theil_sen_zero_prediction(
                    references.astype(np.float64) - frame,
                    source[side, references, joint],
                )
                # When both immediately surrounding physical frames exist,
                # prefer their local linear interpolation. A wider Theil--Sen
                # fit is robust to outliers, but during a fast turn it can
                # extrapolate too far and move the discontinuity to the
                # previous frame. The bracketed estimate is zero-phase and
                # does not introduce that one-frame lag.
                before = references[references < frame]
                after = references[references > frame]
                if len(before) and len(after):
                    previous = int(before[-1])
                    following = int(after[0])
                    if following - previous <= 2:
                        alpha = (frame - previous) / max(following - previous, 1)
                        predicted = (
                            (1.0 - alpha) * source[side, previous, joint]
                            + alpha * source[side, following, joint]
                        )
                        if (
                            np.linalg.norm(
                                source[side, following, joint]
                                - source[side, previous, joint]
                            )
                            > config.temporal_joint_spike_max_local_speed_m
                        ):
                            # A large bracket displacement is likely genuine
                            # rapid motion; do not classify the midpoint as a
                            # spike merely because its linear path is long.
                            predicted = _theil_sen_zero_prediction(
                                references.astype(np.float64) - frame,
                                source[side, references, joint],
                            )
                if not np.all(np.isfinite(predicted)):
                    continue
                residual = float(np.linalg.norm(current - predicted))
                residuals[side, frame, joint] = residual

                # Estimate the locally plausible one-frame displacement only
                # from adjacent reliable reference frames. Gaps are excluded
                # so a missing run cannot inflate the acceptance threshold.
                speeds = []
                for first, second in zip(references[:-1], references[1:]):
                    if int(second) != int(first) + 1:
                        continue
                    first_point = source[side, first, joint]
                    second_point = source[side, second, joint]
                    if np.all(np.isfinite(first_point)) and np.all(np.isfinite(second_point)):
                        speeds.append(float(np.linalg.norm(second_point - first_point)))
                local_speed = float(np.median(speeds)) if speeds else 0.0
                threshold = max(
                    config.temporal_joint_spike_distance_m,
                    config.temporal_joint_spike_velocity_scale * local_speed
                    + config.temporal_joint_spike_margin_m,
                )
                if residual <= threshold:
                    continue

                # The prediction must itself be supported by a compact set of
                # neighbors. This rejects a smooth but highly non-linear turn
                # where a line fit would otherwise pull the current pose away.
                reference_residuals = np.linalg.norm(
                    source[side, references, joint] - predicted,
                    axis=1,
                )
                support_scale = float(np.median(reference_residuals))
                if residual <= max(
                    threshold,
                    2.0 * support_scale + config.temporal_joint_spike_margin_m,
                ):
                    continue
                adjustment = predicted - current
                limit = config.temporal_joint_spike_max_adjustment_m
                norm = float(np.linalg.norm(adjustment))
                if norm > limit and norm > 1e-12:
                    adjustment *= limit / norm
                    predicted = current + adjustment
                    residual = float(np.linalg.norm(adjustment))
                is_strong = residual >= config.temporal_joint_spike_strong_distance_m
                blend = 1.0 if is_strong else config.temporal_joint_spike_blend
                updates.append((side, frame, joint, current + blend * adjustment, residual, is_strong))

    # Apply updates in a second pass; source remains immutable while computing
    # all predictions, so adjacent spikes cannot mask one another.
    update_counts: dict[tuple[int, int], int] = {}
    for side, frame, _joint, _target, _residual, _is_strong in updates:
        update_counts[(side, frame)] = update_counts.get((side, frame), 0) + 1
    for side, frame, joint, target, residual, is_strong in updates:
        if update_counts[(side, frame)] > (
            config.temporal_joint_spike_max_joints_per_hand
        ):
            continue
        output[side, frame, joint] = target
        corrected[side, frame, joint] = True
        strong[side, frame, joint] = is_strong

    return output, corrected, strong, residuals


def _apply_bidirectional_palm_spike_guard(
    points: np.ndarray,
    observed: np.ndarray,
    hands: Mapping[tuple[int, int], dict[str, Any]] | None,
    config: FusionAnatomyConfig,
    segment_ids: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Repair an isolated whole-hand translation/depth branch jump.

    The multiview solver can select a wrong depth solution for the wrist and
    palm together.  Per-joint filtering then sees many individually plausible
    points and cannot restore the hand center.  This pass uses only compact,
    physically valid frames on both sides of the current frame and changes a
    frame only when its palm center is an isolated outlier.  A large immediate
    neighbor displacement is treated as genuine fast motion, so a real turn
    is not flattened.
    """
    source = np.asarray(points, dtype=np.float64)
    output = source.copy()
    corrected = np.zeros(observed.shape[:2], dtype=bool)
    strong = np.zeros(observed.shape[:2], dtype=bool)
    residuals = np.full(observed.shape[:2], np.nan, dtype=np.float64)
    if config.temporal_palm_spike_radius <= 0:
        return output, corrected, strong, residuals

    physical = _physical_hand_frames(source, observed, config)
    palm = np.asarray(PALM_FRAME_JOINTS, dtype=np.int32)
    minimum = config.temporal_palm_spike_min_neighbors_per_side

    # A physical frame can still be the wrong depth branch. Prefer frames with
    # strong detector/multiview support as anchors, and fall back to physical
    # frames only when a short sequence has no such anchors.
    quality_good = np.zeros(source.shape[:2], dtype=bool)
    if hands:
        for (frame, side), hand in hands.items():
            finite = observed[side, frame] & np.isfinite(
                source[side, frame]
            ).all(axis=1)
            support = np.asarray(
                hand.get("inlier_view_counts", []), dtype=np.float64
            )
            median_support = (
                float(np.median(support[finite]))
                if support.shape == (21,) and np.any(finite) else 0.0
            )
            anatomy = hand.get("anatomy_refinement", {})
            rejected_count = int(anatomy.get("rejected_observation_count", 0))
            corrected_count = int(anatomy.get("corrected_joint_count", 0))
            quality = hand.get("quality", {})
            reproj = quality.get("multiview_reprojection_median_px")
            quality_good[side, frame] = bool(
                physical[side, frame]
                and np.count_nonzero(finite) >= 18
                and np.all(np.isfinite(source[side, frame, palm]))
                and median_support >= 2.0
                and rejected_count <= 5
                and corrected_count <= 10
                and (reproj is None or float(reproj) <= 10.0)
            )
    anchor_frames = quality_good | physical
    for side in range(source.shape[0]):
        for frame in range(source.shape[1]):
            current = source[side, frame]
            current_palm = np.isfinite(current[palm]).all(axis=1)
            # The frame being repaired is allowed to fail the physical gate:
            # a depth-branch jump is precisely what often makes its palm scale
            # or radius non-physical. References remain strictly physical.
            if np.count_nonzero(current_palm) < 3:
                continue
            palm_values = source[side][:, palm]
            centers = np.full((source.shape[1], 3), np.nan, dtype=np.float64)
            finite_center = np.isfinite(palm_values).all(axis=2)
            for center_frame in np.flatnonzero(np.any(finite_center, axis=1)):
                centers[center_frame] = np.median(
                    palm_values[center_frame, finite_center[center_frame]],
                    axis=0,
                )
            center_valid = anchor_frames[side] & np.isfinite(centers).all(axis=1)
            references = _temporal_neighbor_frames(
                frame,
                center_valid,
                config.temporal_palm_spike_radius,
                minimum,
                segment_ids[side] if segment_ids is not None else None,
            )
            if len(references) < 2:
                continue
            predicted = _theil_sen_zero_prediction(
                references.astype(np.float64) - frame,
                centers[references],
            )
            if not np.all(np.isfinite(predicted)):
                continue

            # Prefer the bracketed, zero-phase estimate when the nearest
            # physical samples are adjacent. It avoids moving a discontinuity
            # into the previous frame during rapid but valid motion.
            before = references[references < frame]
            after = references[references > frame]
            if len(before) and len(after):
                previous = int(before[-1])
                following = int(after[0])
                if following - previous <= 2:
                    alpha = (frame - previous) / max(following - previous, 1)
                    bracket = (
                        (1.0 - alpha) * centers[previous]
                        + alpha * centers[following]
                    )
                    bracket_step = float(np.linalg.norm(
                        centers[following] - centers[previous]
                    ))
                    if bracket_step <= config.temporal_palm_spike_max_adjustment_m:
                        predicted = bracket

            if not np.all(np.isfinite(centers[frame])):
                continue
            residual = float(np.linalg.norm(centers[frame] - predicted))
            residuals[side, frame] = residual

            # Estimate ordinary local center speed from consecutive physical
            # references. Missing/rejected gaps are intentionally excluded.
            speeds = []
            for first, second in zip(references[:-1], references[1:]):
                if int(second) != int(first) + 1:
                    continue
                speeds.append(float(np.linalg.norm(
                    centers[second] - centers[first]
                )))
            local_speed = float(np.median(speeds)) if speeds else 0.0
            threshold = max(
                config.temporal_palm_spike_distance_m,
                config.temporal_palm_spike_velocity_scale * local_speed
                + config.temporal_palm_spike_margin_m,
            )
            current_quality_good = quality_good[side, frame]
            if not current_quality_good:
                # Low-support/partially observed frames are exactly where a
                # branch jump occurs. Use a lower fixed trigger than for a
                # high-quality physical frame, while still requiring a clear
                # disagreement with both temporal sides below.
                threshold = min(threshold, 0.045)
            if residual <= threshold:
                continue

            # Do not replace a frame merely because a wide-window trajectory
            # disagrees with it. If both immediate neighbors exist and agree
            # with the current center, the wide-window estimate is likely
            # crossing a genuine turn; only an isolated local excursion is a
            # spike. At least one immediate neighbor is also required so that
            # a hand reappearing after a long camera gap is not hallucinated.
            immediate_before = frame > 0 and np.all(
                np.isfinite(centers[frame - 1])
            ) and bool(observed[side, frame - 1, palm].sum() >= 3)
            immediate_after = frame + 1 < source.shape[1] and np.all(
                np.isfinite(centers[frame + 1])
            ) and bool(observed[side, frame + 1, palm].sum() >= 3)
            if not immediate_before and not immediate_after:
                continue
            if immediate_before and immediate_after:
                edge_before_local = float(np.linalg.norm(
                    centers[frame] - centers[frame - 1]
                ))
                edge_after_local = float(np.linalg.norm(
                    centers[frame] - centers[frame + 1]
                ))
                if (
                    edge_before_local <= 0.70 * threshold
                    or edge_after_local <= 0.70 * threshold
                ):
                    continue

            # Require the current sample to be an isolated excursion. If the
            # two sides themselves disagree by a large amount, this is likely
            # a real fast move or a short occlusion run rather than one bad
            # frame, and should be left to tracking.
            if len(before) and len(after):
                previous = int(before[-1])
                following = int(after[0])
                edge_before = float(np.linalg.norm(
                    centers[frame] - centers[previous]
                ))
                edge_after = float(np.linalg.norm(
                    centers[frame] - centers[following]
                ))
                side_step = float(np.linalg.norm(
                    centers[following] - centers[previous]
                ))
                if side_step > max(
                    2.0 * threshold,
                    config.temporal_palm_spike_max_adjustment_m,
                ):
                    continue

            support_residual = float(np.median(np.linalg.norm(
                centers[references] - predicted,
                axis=1,
            )))
            # For a high-quality physical frame, retain the conservative
            # support-consistency test. A low-quality/partial frame is the
            # actual failure mode this guard targets; its surrounding
            # neighbors have already passed the local-isolation check above,
            # so requiring the wide-window samples to form a tight line would
            # incorrectly reject depth-branch corrections during a turn.
            if current_quality_good and residual <= max(
                threshold,
                2.0 * support_residual + config.temporal_palm_spike_margin_m,
            ):
                continue
            if not current_quality_good and not immediate_before and not immediate_after:
                continue
            if not current_quality_good and (not immediate_before or not immediate_after):
                before_gap = frame - int(before[-1]) if len(before) else 999
                after_gap = int(after[0]) - frame if len(after) else 999
                if residual < 0.080 or min(before_gap, after_gap) > 8:
                    continue

            delta = predicted - centers[frame]
            norm = float(np.linalg.norm(delta))
            if norm <= 1e-12:
                continue
            limit = config.temporal_palm_spike_max_adjustment_m
            if norm > limit:
                delta *= limit / norm
                residual = float(np.linalg.norm(delta))
            is_strong = residual >= config.temporal_palm_spike_strong_distance_m
            blend = 1.0 if is_strong else config.temporal_palm_spike_blend
            finite = observed[side, frame] & np.isfinite(current).all(axis=1)
            if np.count_nonzero(finite) < 8:
                continue
            output[side, frame, finite] = (
                current[finite] + blend * delta
            )
            corrected[side, frame] = True
            strong[side, frame] = is_strong

    return output, corrected, strong, residuals


def _apply_final_short_spike_filter(
    points: np.ndarray,
    observed: np.ndarray,
    config: FusionAnatomyConfig,
    segment_ids: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove isolated short-window joint excursions from the final track.

    This is intentionally a last, conservative pass. It uses only the
    already-refined trajectory, a small bracket on each side, and a velocity
    adaptive threshold; no long-window smoothing or causal lag is introduced.
    """
    source = np.asarray(points, dtype=np.float64)
    output = source.copy()
    corrected = np.zeros_like(observed)
    if config.final_spike_radius <= 0:
        return output, corrected
    frame_count = source.shape[1]
    for side in range(source.shape[0]):
        for joint in range(source.shape[2]):
            updates: list[tuple[int, np.ndarray]] = []
            for frame in range(frame_count):
                if not (observed[side, frame, joint]
                        and np.isfinite(source[side, frame, joint]).all()):
                    continue
                before = [
                    k for k in range(
                        frame - 1,
                        max(-1, frame - config.final_spike_radius - 1),
                        -1,
                    ) if observed[side, k, joint]
                    and (
                        segment_ids is None
                        or segment_ids.shape != observed.shape[:2]
                        or segment_ids[side, k] == segment_ids[side, frame]
                    )
                    and np.isfinite(source[side, k, joint]).all()
                ]
                after = [
                    k for k in range(
                        frame + 1,
                        min(frame_count, frame + config.final_spike_radius + 1),
                    ) if observed[side, k, joint]
                    and (
                        segment_ids is None
                        or segment_ids.shape != observed.shape[:2]
                        or segment_ids[side, k] == segment_ids[side, frame]
                    )
                    and np.isfinite(source[side, k, joint]).all()
                ]
                if (
                    len(before) < config.final_spike_min_neighbors_per_side
                    or len(after) < config.final_spike_min_neighbors_per_side
                ):
                    continue
                previous = int(before[0])
                following = int(after[0])
                if following <= previous:
                    continue
                alpha = (frame - previous) / max(following - previous, 1)
                predicted = (
                    (1.0 - alpha) * source[side, previous, joint]
                    + alpha * source[side, following, joint]
                )
                residual = float(np.linalg.norm(
                    source[side, frame, joint] - predicted
                ))
                bracket_step = float(np.linalg.norm(
                    source[side, following, joint]
                    - source[side, previous, joint]
                )) / max(following - previous, 1)
                threshold = max(
                    config.final_spike_distance_m,
                    config.final_spike_velocity_scale * bracket_step
                    + config.final_spike_margin_m,
                )
                if residual <= threshold:
                    continue
                # Require a true excursion: both edges from the current point
                # must be large relative to the local motion. This protects a
                # genuine monotonic movement that merely has a high velocity.
                edge_before = float(np.linalg.norm(
                    source[side, frame, joint]
                    - source[side, previous, joint]
                )) / max(frame - previous, 1)
                edge_after = float(np.linalg.norm(
                    source[side, following, joint]
                    - source[side, frame, joint]
                )) / max(following - frame, 1)
                if min(edge_before, edge_after) <= 0.70 * threshold:
                    continue
                updates.append((frame, predicted))
            if len(updates) > config.final_spike_max_joints_per_hand:
                # Per-joint updates are collected independently; this guard is
                # applied below per frame to avoid replacing an entire pose
                # during a genuine articulated movement.
                pass
            for frame, predicted in updates:
                output[side, frame, joint] = (
                    source[side, frame, joint]
                    + config.final_spike_blend * (
                        predicted - source[side, frame, joint]
                    )
                )
                corrected[side, frame, joint] = True
    # Limit the number of joints altered in any one frame. If more joints
    # disagree together, the hand-level tracking pass is the appropriate owner.
    for side in range(output.shape[0]):
        for frame in range(frame_count):
            indices = np.flatnonzero(corrected[side, frame])
            if len(indices) > config.final_spike_max_joints_per_hand:
                corrected[side, frame, indices[config.final_spike_max_joints_per_hand:]] = False
                output[side, frame, indices[config.final_spike_max_joints_per_hand:]] = source[side, frame, indices[config.final_spike_max_joints_per_hand:]]
    return output, corrected


def _apply_final_palm_trajectory_filter(
    points: np.ndarray,
    observed: np.ndarray,
    hands: Mapping[tuple[int, int], dict[str, Any]] | None,
    config: FusionAnatomyConfig,
    segment_ids: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Remove isolated rigid palm excursions from the final trajectory.

    The ordinary short-window filter works per joint and therefore cannot
    recognise a wrong stereo-depth branch that translates most of a partially
    observed hand together.  Here the palm centre is estimated from whatever
    three or more palm anchors are available.  A frame is changed only when
    the nearest valid samples on both temporal sides predict a compact
    bracket and the current centre is an isolated excursion.  The correction
    is a capped rigid translation, so finger articulation and bone lengths are
    preserved exactly.
    """
    source = np.asarray(points, dtype=np.float64)
    output = source.copy()
    corrected = np.zeros(observed.shape[:2], dtype=bool)
    residuals = np.full(observed.shape[:2], np.nan, dtype=np.float64)
    if config.final_palm_radius <= 0:
        return output, corrected, residuals

    def improves_local_edges(
        current: np.ndarray,
        previous: np.ndarray,
        following: np.ndarray,
        candidate: np.ndarray,
        gap_before: int,
        gap_after: int,
    ) -> bool:
        """Require a palm correction to reduce (or preserve) both local jumps.

        A bracket can be geometrically plausible while still pointing in the
        wrong direction when one of its boundary observations is itself a
        branch switch.  Applying such a correction independently can turn a
        50 mm input step into a 180 mm output step.  Compare normalized edge
        lengths before/after and reject any update that increases the largest
        edge or the local squared-edge energy.
        """
        before_edges = np.asarray((
            np.linalg.norm(current - previous) / max(int(gap_before), 1),
            np.linalg.norm(following - current) / max(int(gap_after), 1),
        ), dtype=np.float64)
        after_edges = np.asarray((
            np.linalg.norm(candidate - previous) / max(int(gap_before), 1),
            np.linalg.norm(following - candidate) / max(int(gap_after), 1),
        ), dtype=np.float64)
        if not (
            np.all(np.isfinite(before_edges)) and np.all(np.isfinite(after_edges))
        ):
            return False
        # A tiny tolerance avoids rejecting numerically neutral corrections,
        # while a strict max-edge guard prevents a correction from moving the
        # discontinuity to the other side of the frame.
        tolerance = 1e-6 + 0.01 * float(np.sum(before_edges ** 2))
        return bool(
            np.sum(after_edges ** 2) <= np.sum(before_edges ** 2) + tolerance
            and np.max(after_edges) <= np.max(before_edges) + 0.005
        )

    palm = np.asarray(PALM_FRAME_JOINTS, dtype=np.int32)
    frame_count = source.shape[1]
    centers = np.full((source.shape[0], frame_count, 3), np.nan, dtype=np.float64)
    center_counts = np.zeros((source.shape[0], frame_count), dtype=np.int32)
    for side in range(source.shape[0]):
        for frame in range(frame_count):
            finite = observed[side, frame, palm] & np.isfinite(
                source[side, frame, palm]
            ).all(axis=1)
            center_counts[side, frame] = int(np.count_nonzero(finite))
            if center_counts[side, frame] >= config.final_palm_min_joints:
                centers[side, frame] = np.median(
                    source[side, frame, palm[finite]], axis=0
                )

    # Metadata is only a quality hint.  A low-support frame gets a lower
    # trigger because that is where wrong branches are most common; a frame
    # with strong support must disagree more clearly before being changed.
    quality_good = np.zeros((source.shape[0], frame_count), dtype=bool)
    weak_frame = np.zeros((source.shape[0], frame_count), dtype=bool)
    if hands:
        for (frame, side), hand in hands.items():
            if not np.all(np.isfinite(centers[side, frame])):
                continue
            valid = observed[side, frame] & np.isfinite(
                source[side, frame]
            ).all(axis=1)
            support = np.asarray(hand.get("inlier_view_counts", []), dtype=np.float64)
            support_values = support[valid] if support.shape == (21,) else np.empty(0)
            median_support = float(np.median(support_values)) if len(support_values) else 0.0
            quality = hand.get("quality", {})
            reproj = quality.get("multiview_reprojection_median_px")
            anatomy = hand.get("anatomy_refinement", {})
            weak_frame[side, frame] = bool(
                np.count_nonzero(valid) < 20
                or median_support < 3.0
                or (reproj is not None and float(reproj) > 10.0)
            )
            quality_good[side, frame] = bool(
                center_counts[side, frame] == len(palm)
                and np.count_nonzero(valid) >= 18
                and median_support >= 3.0
                and (reproj is None or float(reproj) <= 10.0)
                and int(anatomy.get("rejected_observation_count", 0)) <= 5
            )

    for side in range(source.shape[0]):
        all_valid_frames = np.flatnonzero(np.isfinite(centers[side]).all(axis=1))
        for frame in all_valid_frames:
            frame = int(frame)
            if segment_ids is not None and segment_ids.shape == observed.shape[:2]:
                segment = segment_ids[side, frame]
                valid_frames = all_valid_frames[
                    segment_ids[side, all_valid_frames] == segment
                ]
            else:
                valid_frames = all_valid_frames
            before = valid_frames[
                (valid_frames < frame)
                & (valid_frames >= frame - config.final_palm_radius)
            ]
            after = valid_frames[
                (valid_frames > frame)
                & (valid_frames <= frame + config.final_palm_radius)
            ]
            if (
                len(before) < config.final_palm_min_neighbors_per_side
                or len(after) < config.final_palm_min_neighbors_per_side
            ):
                continue
            previous = int(before[-1])
            following = int(after[0])
            if following <= previous:
                continue
            alpha = (frame - previous) / float(following - previous)
            predicted = (
                (1.0 - alpha) * centers[side, previous]
                + alpha * centers[side, following]
            )
            current = centers[side, frame]
            residual = float(np.linalg.norm(current - predicted))
            residuals[side, frame] = residual
            bracket_step = float(np.linalg.norm(
                centers[side, following] - centers[side, previous]
            )) / max(following - previous, 1)
            threshold = max(
                config.final_palm_distance_m,
                config.final_palm_velocity_scale * bracket_step
                + config.final_palm_margin_m,
            )
            if not quality_good[side, frame]:
                threshold = min(threshold, 0.050)
            if residual <= threshold:
                continue

            # A true monotonic movement has one short edge.  An isolated
            # branch excursion has two long edges to its bracket samples.
            edge_before = float(np.linalg.norm(current - centers[side, previous])) / max(frame - previous, 1)
            edge_after = float(np.linalg.norm(centers[side, following] - current)) / max(following - frame, 1)
            if min(edge_before, edge_after) <= 0.65 * threshold:
                continue
            if not quality_good[side, frame] and residual < 0.075:
                continue

            delta = predicted - current
            norm = float(np.linalg.norm(delta))
            if norm <= 1e-12:
                continue
            adjustment_limit = (
                config.final_palm_severe_max_adjustment_m
                if (
                    not quality_good[side, frame]
                    and residual >= config.final_palm_severe_distance_m
                    and (following - previous) <= 5
                    and float(np.linalg.norm(
                        centers[side, following] - centers[side, previous]
                    )) <= 0.15
                )
                else config.final_palm_max_adjustment_m
            )
            if norm > adjustment_limit:
                delta *= adjustment_limit / norm
            finite_joints = observed[side, frame] & np.isfinite(
                source[side, frame]
            ).all(axis=1)
            if np.count_nonzero(finite_joints) < config.final_palm_min_joints:
                continue
            candidate_center = current + config.final_palm_blend * delta
            if not improves_local_edges(
                current,
                centers[side, previous],
                centers[side, following],
                candidate_center,
                frame - previous,
                following - frame,
            ):
                continue
            output[side, frame, finite_joints] = (
                source[side, frame, finite_joints]
                + config.final_palm_blend * delta
            )
            corrected[side, frame] = True
    # A branch can persist for two or three frames, so no single frame is an
    # isolated excursion.  Detect short runs whose centre remains far from a
    # straight interpolation between the reliable samples immediately before
    # and after the run.  Correct the whole run with a rigid translation.  We
    # require both boundary edges to be large and the outside bracket to be
    # compact; this leaves genuine sustained motion untouched.
    for side in range(source.shape[0]):
        valid_mask = np.isfinite(centers[side]).all(axis=1)
        # Only low-quality frames can form a candidate run.  High-quality
        # frames are deliberately excluded even if their motion is fast.
        low_quality = valid_mask & weak_frame[side]
        run_start = 0
        while run_start < frame_count:
            if not low_quality[run_start]:
                run_start += 1
                continue
            run_end = run_start
            while run_end + 1 < frame_count and low_quality[run_end + 1]:
                if (
                    segment_ids is not None
                    and segment_ids.shape == observed.shape[:2]
                    and segment_ids[side, run_end + 1]
                    != segment_ids[side, run_end]
                ):
                    break
                run_end += 1
            run_length = run_end - run_start + 1
            if run_length < 2 or run_length > config.final_palm_run_max_length:
                run_start = run_end + 1
                continue
            previous = run_start - 1
            following = run_end + 1
            if previous < 0 or following >= frame_count or not (
                valid_mask[previous] and valid_mask[following]
            ):
                run_start = run_end + 1
                continue
            if (
                segment_ids is not None
                and segment_ids.shape == observed.shape[:2]
                and not (
                    segment_ids[side, previous]
                    == segment_ids[side, run_start]
                    == segment_ids[side, following]
                )
            ):
                run_start = run_end + 1
                continue
            outside_step = float(np.linalg.norm(
                centers[side, following] - centers[side, previous]
            )) / max(following - previous, 1)
            threshold = max(
                config.final_palm_run_distance_m,
                config.final_palm_run_velocity_scale * outside_step
                + config.final_palm_run_margin_m,
            )
            residual_values = []
            predictions = []
            for current_frame in range(run_start, run_end + 1):
                alpha = (current_frame - previous) / float(following - previous)
                predicted = (
                    (1.0 - alpha) * centers[side, previous]
                    + alpha * centers[side, following]
                )
                predictions.append(predicted)
                residual_values.append(float(np.linalg.norm(
                    centers[side, current_frame] - predicted
                )))
            if max(residual_values) <= threshold:
                run_start = run_end + 1
                continue
            edge_before = float(np.linalg.norm(
                centers[side, run_start] - centers[side, previous]
            ))
            edge_after = float(np.linalg.norm(
                centers[side, following] - centers[side, run_end]
            ))
            if min(edge_before, edge_after) <= 0.65 * threshold:
                run_start = run_end + 1
                continue
            bracket_norm = float(np.linalg.norm(
                centers[side, following] - centers[side, previous]
            ))
            if bracket_norm > config.final_palm_max_adjustment_m:
                run_start = run_end + 1
                continue
            for current_frame, predicted, residual in zip(
                range(run_start, run_end + 1), predictions, residual_values
            ):
                delta = predicted - centers[side, current_frame]
                norm = float(np.linalg.norm(delta))
                if norm <= 1e-12:
                    continue
                if norm > config.final_palm_max_adjustment_m:
                    delta *= config.final_palm_max_adjustment_m / norm
                finite_joints = observed[side, current_frame] & np.isfinite(
                    source[side, current_frame]
                ).all(axis=1)
                if np.count_nonzero(finite_joints) < config.final_palm_min_joints:
                    continue
                candidate_center = centers[side, current_frame] + config.final_palm_blend * delta
                if not improves_local_edges(
                    centers[side, current_frame],
                    centers[side, previous],
                    centers[side, following],
                    candidate_center,
                    current_frame - previous,
                    following - current_frame,
                ):
                    continue
                output[side, current_frame, finite_joints] = (
                    source[side, current_frame, finite_joints]
                    + config.final_palm_blend * delta
                )
                corrected[side, current_frame] = True
                residuals[side, current_frame] = max(
                    residuals[side, current_frame], residual
                )
            run_start = run_end + 1
    return output, corrected, residuals


def _apply_gesture_smoothing(
    points: np.ndarray,
    observed: np.ndarray,
    config: FusionAnatomyConfig,
    segment_ids: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Gently smooth ordinary hand jitter without flattening fast gestures.

    The earlier temporal passes are intentionally anomaly detectors.  A
    reliable WiLoR joint can therefore still move a few millimetres back and
    forth in every frame and never trigger any guard.  This pass uses a
    centred, robust constant-velocity prediction and only applies a small
    bounded blend when the local motion is below the fast-gesture threshold.
    It never bridges an image-track segment boundary or fills a missing joint.
    """
    source = np.asarray(points, dtype=np.float64)
    output = source.copy()
    corrected = np.zeros_like(observed, dtype=bool)
    if not config.gesture_smoothing_enabled or config.gesture_smoothing_radius <= 0:
        return output, corrected
    radius = int(config.gesture_smoothing_radius)
    palm = set(int(index) for index in PALM_FRAME_JOINTS)
    for side in range(source.shape[0]):
        for joint in range(source.shape[2]):
            joint_valid = (
                observed[side, :, joint]
                & np.isfinite(source[side, :, joint]).all(axis=1)
            )
            for frame in np.flatnonzero(joint_valid):
                before = _temporal_neighbor_frames(
                    int(frame), joint_valid, radius, 1,
                    segment_ids[side] if segment_ids is not None else None,
                )
                if not len(before):
                    continue
                # _temporal_neighbor_frames returns both sides.  Require a
                # bracket so a hand entering/leaving the view is untouched.
                previous = before[before < frame]
                following = before[before > frame]
                if not len(previous) or not len(following):
                    continue
                neighbours = np.concatenate((previous, following))
                times = neighbours.astype(np.float64) - float(frame)
                values = source[side, neighbours, joint]
                predicted = _theil_sen_zero_prediction(times, values)
                if not np.all(np.isfinite(predicted)):
                    continue
                previous_frame = int(previous[-1])
                following_frame = int(following[0])
                gap_before = max(int(frame) - previous_frame, 1)
                gap_after = max(following_frame - int(frame), 1)
                before_step = float(np.linalg.norm(
                    source[side, frame, joint]
                    - source[side, previous_frame, joint]
                )) / gap_before
                after_step = float(np.linalg.norm(
                    source[side, following_frame, joint]
                    - source[side, frame, joint]
                )) / gap_after
                if max(before_step, after_step) > config.gesture_smoothing_max_motion_m:
                    continue
                residual = float(np.linalg.norm(
                    source[side, frame, joint] - predicted
                ))
                if residual <= 1e-9 or residual > config.gesture_smoothing_max_residual_m:
                    continue
                # Palm anchors determine global hand motion and are smoothed
                # more weakly than finger joints to avoid visible lag.
                blend = float(config.gesture_smoothing_blend)
                if joint in palm:
                    blend *= 0.45
                delta = _cap_vector(
                    (predicted - source[side, frame, joint]) * blend,
                    config.gesture_smoothing_max_adjustment_m,
                )
                if np.linalg.norm(delta) <= 1e-9:
                    continue
                output[side, frame, joint] = source[side, frame, joint] + delta
                corrected[side, frame, joint] = True
    output[~observed] = np.nan
    return output, corrected


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
    trusted_temporal_joints: np.ndarray | None = None,
    trusted_palm_hands: np.ndarray | None = None,
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
            if (
                trusted_palm_hands is not None
                and trusted_palm_hands[side, frame]
            ):
                # The final palm trajectory guard is a deliberately capped,
                # rigid correction selected from both time directions.  Its
                # purpose is precisely to override a low-support stereo
                # branch that can still have low reprojection error in the
                # selected cameras; retaining the old pixel-shift gate would
                # undo the correction immediately.
                continue
            if (
                trusted_temporal_joints is not None
                and trusted_temporal_joints[side, frame, joint]
            ):
                continue
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
    temporal_segments = _temporal_segments_from_rows(rows, raw.shape[1])
    segment_boundaries = _segment_boundary_frames(observed, temporal_segments)
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
    # Outlier windows are intentionally conservative around a hard image-track
    # re-entry.  The preceding and entering observations belong to different
    # physical image segments; comparing them as one trajectory would mark a
    # perfectly valid pose as an all-joint outlier and then let anatomy move it
    # toward the wrong hand.  Preserve the delivered measurements at both
    # sides of the boundary and let branch selection handle any true depth
    # ambiguity before this stage.
    if np.any(segment_boundaries):
        rejected[segment_boundaries] = False
        accepted[segment_boundaries] = observed[segment_boundaries]
        preliminary_confidence[segment_boundaries] = confidence[segment_boundaries]
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
    refined, dropped_unreconstructable = _drop_unreconstructable_hands(
        refined,
        observed,
        rejected,
        candidate_finite,
    )
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
        temporal_segments,
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
            & config.allow_strong_reprojection_override
            & (temporal_global_residual > config.temporal_palm_strong_residual_m)
            & (
                np.count_nonzero(rejected, axis=2)
                >= config.temporal_global_min_rejected_joints
            )
        ),
    )
    trusted_temporal_hands = (
        temporal_global_corrected
        & config.allow_strong_reprojection_override
        & (temporal_global_residual > config.temporal_palm_strong_residual_m)
        & (
            np.count_nonzero(rejected, axis=2)
            >= config.temporal_global_min_rejected_joints
        )
    )

    (
        tracked_candidate,
        tracking_corrected,
        tracking_hand_corrected,
        tracking_palm_residual,
        tracking_shape_residual,
        tracking_summary,
        tracking_trusted_override,
    ) = _apply_confidence_guided_tracking(
        refined,
        observed,
        confidence,
        hands,
        config,
        temporal_segments,
    )
    (
        palm_spike_candidate,
        palm_spike_corrected,
        palm_spike_strong,
        palm_spike_residual,
    ) = _apply_bidirectional_palm_spike_guard(
        tracked_candidate,
        observed,
        hands,
        config,
        temporal_segments,
    )
    (
        joint_spike_candidate,
        joint_spike_corrected,
        joint_spike_strong,
        joint_spike_residual,
    ) = _apply_bidirectional_joint_spike_guard(
        palm_spike_candidate,
        observed,
        config,
        hands,
        temporal_segments,
    )
    final_spike_candidate, final_spike_corrected = _apply_final_short_spike_filter(
        joint_spike_candidate,
        observed,
        config,
        temporal_segments,
    )
    (
        refined,
        tracking_reprojection_limited,
        tracking_reprojection_reverted,
        tracking_reprojection_diagnostics,
    ) = _apply_reprojection_guard(
        final_spike_candidate,
        raw,
        hands,
        calibrations,
        config,
        # Every tracking correction remains subject to multiview reprojection
        # evidence.  A collision is only a diagnostic trigger; it is not
        # sufficient authority to bypass image-space safety checks.
        trusted_temporal_hands=trusted_temporal_hands,
        # A strong, isolated *per-joint* contradiction may override the image
        # guard for that joint only.  A palm-spike decision is a rigid
        # whole-hand proposal; allowing only its palm anchors through here
        # would let the guard restore the fingers to the old branch and split
        # the hand into two disconnected pieces.  Keep palm corrections under
        # the normal reprojection gate; the final hand-centre pass can still
        # apply a coherent rigid update when the evidence is safe.
        trusted_temporal_joints=joint_spike_strong,
    )
    # Apply the hand-centre guard after reprojection.  Earlier temporal passes
    # can temporarily move an incorrect depth branch close to the track, only
    # for the image-space guard to restore the raw branch.  Running this last
    # lets the bidirectional trajectory decision see the actual delivered
    # pose; its corrections are already rigid, capped, and low-quality-only.
    (
        refined,
        final_palm_corrected,
        final_palm_residual,
    ) = _apply_final_palm_trajectory_filter(
        refined,
        observed,
        hands,
        config,
        temporal_segments,
    )

    # Reduce small, ordinary detector tremor after the anomaly/depth guards.
    # The result is still passed through the same reprojection and displacement
    # protections as every other correction, so smoothing cannot hide a bad
    # branch or move a hand across an image-track boundary.
    (
        gesture_smoothed,
        gesture_smoothing_corrected,
    ) = _apply_gesture_smoothing(
        refined,
        observed,
        config,
        temporal_segments,
    )
    (
        refined,
        gesture_reprojection_limited,
        gesture_reprojection_reverted,
        gesture_reprojection_diagnostics,
    ) = _apply_reprojection_guard(
        gesture_smoothed,
        raw,
        hands,
        calibrations,
        config,
        # Preserve the isolated depth-branch repairs that already passed the
        # dedicated bidirectional spike guard above.  Without forwarding this
        # trust mask, the second reprojection pass can silently pull a strong
        # repair back toward the raw branch even when gesture smoothing did
        # not touch that joint.
        trusted_temporal_joints=joint_spike_strong,
    )

    (
        refined,
        post_refinement_clipped,
        post_refinement_reverted,
    ) = _guard_post_refinement_displacements(
        refined, raw, observed, config
    )

    refined, post_refinement_center_limited = (
        _limit_refinement_center_displacement(
            refined,
            raw,
            observed,
            min(config.tracking_max_adjustment_m, 0.10),
            exempt_hands=temporal_global_corrected,
        )
    )

    # Enforce hand-level atomicity after every per-joint guard.  If the
    # collection of anatomy/tracking corrections makes the local palm
    # trajectory worse (or exceeds the rigid safety cap), restore the whole
    # hand frame instead of serializing a palm/finger mixture.
    refined, atomic_trajectory_reverted = _atomic_hand_trajectory_guard(
        refined, raw, observed, hands, config
    )
    refined, final_center_step_reverted = _final_center_step_guard(
        refined, raw, observed, hands, config
    )

    # A rigid palm proposal and the per-view reprojection guard must be
    # accepted or rejected as one pose.  If only a subset of joints survived
    # the preceding guards, the delivered array can contain a palm on one
    # depth branch and fingers on another.  Validate the final envelope and
    # atomically fall back to the original triangulation for that hand.  This
    # is deliberately a last safety net: normal articulation and partial
    # visibility are retained, while metre-scale disconnected skeletons are
    # never serialized.
    physical_envelope_reverted = np.zeros(refined.shape[:2], dtype=bool)
    physical_envelope_failed = np.zeros(refined.shape[:2], dtype=bool)
    for (frame, side), _hand in hands.items():
        current = refined[side, frame]
        if _final_hand_envelope_valid(
            current, observed[side, frame], config,
        ):
            continue
        raw_pose = raw[side, frame]
        # The raw fused pose is the least-assumptive fallback.  Even when it
        # is itself weak, it cannot inherit a split correction from this
        # refinement stage; downstream branch/identity selection has already
        # run on that observation.
        refined[side, frame] = raw_pose
        physical_envelope_reverted[side, frame] = True
        if not _final_hand_envelope_valid(
            raw_pose, observed[side, frame], config,
        ):
            physical_envelope_failed[side, frame] = True
    physical_envelope_reverted |= atomic_trajectory_reverted
    physical_envelope_reverted |= final_center_step_reverted

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
            "synthetic_track_fill": bool(hand.get("synthetic_track_fill")),
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
            "dropped_unreconstructable": bool(
                dropped_unreconstructable[side, frame]
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
            "tracking_applied": bool(tracking_hand_corrected[side, frame]),
            "tracking_trusted_override": bool(
                tracking_trusted_override[side, frame]
            ),
            "tracking_corrected_joint_count": int(
                np.count_nonzero(tracking_corrected[side, frame])
            ),
            "tracking_palm_residual_mm": (
                float(1000.0 * tracking_palm_residual[side, frame])
                if np.isfinite(tracking_palm_residual[side, frame]) else None
            ),
            "tracking_shape_residual": (
                float(tracking_shape_residual[side, frame])
                if np.isfinite(tracking_shape_residual[side, frame]) else None
            ),
            "joint_spike_corrected_joint_count": int(
                np.count_nonzero(joint_spike_corrected[side, frame])
            ),
            "joint_spike_strong_joint_count": int(
                np.count_nonzero(joint_spike_strong[side, frame])
            ),
            "joint_spike_max_residual_mm": (
                float(np.nanmax(joint_spike_residual[side, frame]) * 1000.0)
                if np.any(np.isfinite(joint_spike_residual[side, frame])) else None
            ),
            "palm_spike_corrected": bool(palm_spike_corrected[side, frame]),
            "palm_spike_strong": bool(palm_spike_strong[side, frame]),
            "palm_spike_residual_mm": (
                float(palm_spike_residual[side, frame] * 1000.0)
                if np.isfinite(palm_spike_residual[side, frame]) else None
            ),
            "final_spike_corrected_joint_count": int(
                np.count_nonzero(final_spike_corrected[side, frame])
            ),
            "final_palm_trajectory_corrected": bool(final_palm_corrected[side, frame]),
            "gesture_smoothing_corrected_joint_count": int(
                np.count_nonzero(gesture_smoothing_corrected[side, frame])
            ),
            "physical_envelope_reverted": bool(
                physical_envelope_reverted[side, frame]
            ),
            "physical_envelope_failed": bool(
                physical_envelope_failed[side, frame]
            ),
            "post_refinement_clipped_joint_count": int(
                np.count_nonzero(post_refinement_clipped[side, frame])
            ),
            "post_refinement_reverted_joint_count": int(
                np.count_nonzero(post_refinement_reverted[side, frame])
            ),
            "post_refinement_center_limited": bool(
                post_refinement_center_limited[side, frame]
            ),
            "atomic_trajectory_reverted": bool(
                atomic_trajectory_reverted[side, frame]
            ),
            "final_center_step_reverted": bool(
                final_center_step_reverted[side, frame]
            ),
            "final_palm_trajectory_residual_mm": (
                float(final_palm_residual[side, frame] * 1000.0)
                if np.isfinite(final_palm_residual[side, frame]) else None
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
        "tracking": {
            **tracking_summary,
            "reprojection_limited_joint_count": tracking_reprojection_limited,
            "reprojection_reverted_joint_count": tracking_reprojection_reverted,
            "reprojection_diagnostics": tracking_reprojection_diagnostics,
        },
        "joint_spike_guard": {
            "algorithm": "bidirectional_per_joint_depth_branch_guard_v1",
            "corrected_joint_count": int(np.count_nonzero(joint_spike_corrected)),
            "strong_corrected_joint_count": int(np.count_nonzero(joint_spike_strong)),
            "max_predicted_residual_mm": (
                float(np.nanmax(joint_spike_residual) * 1000.0)
                if np.any(np.isfinite(joint_spike_residual)) else None
            ),
        },
        "palm_spike_guard": {
            "algorithm": "bidirectional_whole_hand_center_guard_v1",
            "corrected_hand_count": int(np.count_nonzero(palm_spike_corrected)),
            "strong_corrected_hand_count": int(np.count_nonzero(palm_spike_strong)),
            "max_predicted_residual_mm": (
                float(np.nanmax(palm_spike_residual) * 1000.0)
                if np.any(np.isfinite(palm_spike_residual)) else None
            ),
        },
        "final_spike_guard": {
            "algorithm": "bidirectional_short_window_joint_filter_v1",
            "corrected_joint_count": int(np.count_nonzero(final_spike_corrected)),
        },
        "final_palm_trajectory_guard": {
            "algorithm": "bidirectional_rigid_palm_trajectory_filter_v1",
            "corrected_hand_count": int(np.count_nonzero(final_palm_corrected)),
            "max_predicted_residual_mm": (
                float(np.nanmax(final_palm_residual) * 1000.0)
                if np.any(np.isfinite(final_palm_residual)) else None
            ),
        },
        "gesture_smoothing": {
            "algorithm": "zero_phase_bounded_gesture_smoothing_v1",
            "corrected_joint_count": int(
                np.count_nonzero(gesture_smoothing_corrected)
            ),
            "reprojection_limited_joint_count": int(
                gesture_reprojection_limited
            ),
            "reprojection_reverted_joint_count": int(
                gesture_reprojection_reverted
            ),
            "reprojection_diagnostics": gesture_reprojection_diagnostics,
        },
        "physical_envelope_reverted_hand_count": int(
            np.count_nonzero(physical_envelope_reverted)
        ),
        "atomic_trajectory_reverted_hand_count": int(
            np.count_nonzero(atomic_trajectory_reverted)
        ),
        "final_center_step_reverted_hand_count": int(
            np.count_nonzero(final_center_step_reverted)
        ),
        "physical_envelope_failed_hand_count": int(
            np.count_nonzero(physical_envelope_failed)
        ),
        "post_refinement_clipped_joint_count": int(
            np.count_nonzero(post_refinement_clipped)
        ),
        "post_refinement_reverted_joint_count": int(
            np.count_nonzero(post_refinement_reverted)
        ),
        "post_refinement_center_limited_hand_count": int(
            np.count_nonzero(post_refinement_center_limited)
        ),
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
        "dropped_unreconstructable_hand_count": int(
            np.count_nonzero(dropped_unreconstructable)
        ),
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
