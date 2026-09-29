#!/usr/bin/env python3
"""Per-joint temporal stabilisation for ambiguous multiview hand depth.

The WiLoR/SAM observations live in the camera image and are deliberately not
changed here.  A joint can nevertheless have a very different 3-D depth when
its rays are nearly parallel (or when only three views survive matching).  The
functions in this module use the 2-D rays as measurements and a bidirectional
track as a weak depth prior.  A correction is only kept when it is a temporal
outlier and the replacement is physically plausible.  The final low-pass and
bone passes are zero-phase, so the output is suitable for an offline delivery
video rather than a laggy live filter.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from camera_models import project_points, unproject_points
from stabilize_hand_3d import (
    PALM_FRAME_JOINTS,
    SKELETON_EDGES,
    constrain_bones,
    estimate_bone_lengths,
)


ALGORITHM = "per_joint_bidirectional_ray_prior_conditioned_angle_weighted_bone_temporal_v4_visibility"
RAY_CONDITIONING_SOFT_THRESHOLD = 100.0
RAY_CONDITIONING_SATURATION = 600.0
RAY_CONDITIONING_HARD_THRESHOLD = 1000.0
RAY_ANGLE_REFERENCE_DEGREES = 25.0
RAY_ANGLE_WEIGHT_FLOOR = 0.20


def _palm_center(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if points.shape != (21, 3):
        return np.full(3, np.nan, dtype=np.float64)
    values = points[PALM_FRAME_JOINTS]
    values = values[np.isfinite(values).all(axis=1)]
    # Condition-number gating can temporarily remove one or more MCP joints.
    # Fall back to the robust center of the remaining hand points so the
    # articulation guard can still recognize a palm-stable finger branch.
    if len(values) < 3:
        values = points[np.isfinite(points).all(axis=1)]
    if len(values) < 3:
        return np.full(3, np.nan, dtype=np.float64)
    return np.median(values, axis=0)


def _hand_palm(hand: Mapping[str, Any]) -> np.ndarray:
    """Return a robust palm center from a serialized fusion hand."""
    for key in ("unrefined_joints_base_m", "joints_base_m"):
        values = np.asarray(hand.get(key, []), dtype=np.float64)
        if values.shape == (21, 3):
            center = _palm_center(values)
            if np.isfinite(center).all():
                return center
    return np.full(3, np.nan, dtype=np.float64)


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


def _hand_match_cost(
    current: Mapping[str, Any],
    previous: Mapping[str, Any],
    frame_gap: int,
) -> tuple[float, bool]:
    """Compare two physical-hand candidates using 3-D and common 2-D boxes."""
    terms: list[float] = []
    previous_palm = _hand_palm(previous)
    current_palm = _hand_palm(current)
    if np.isfinite(previous_palm).all() and np.isfinite(current_palm).all():
        allowance = 0.06 + 0.10 * max(int(frame_gap), 1)
        terms.append(min(float(np.linalg.norm(current_palm - previous_palm)) / allowance, 6.0))

    box_terms: list[float] = []
    previous_views = previous.get("views", {})
    current_views = current.get("views", {})
    for camera_id in set(previous_views).intersection(current_views):
        previous_box_array = np.asarray(
            previous_views[camera_id].get("bbox_xyxy"), dtype=np.float64
        )
        current_box_array = np.asarray(
            current_views[camera_id].get("bbox_xyxy"), dtype=np.float64
        )
        if previous_box_array.shape != (4,) or current_box_array.shape != (4,):
            continue
        if not (
            np.isfinite(previous_box_array).all()
            and np.isfinite(current_box_array).all()
        ):
            continue
        previous_center = 0.5 * (previous_box_array[:2] + previous_box_array[2:])
        current_center = 0.5 * (current_box_array[:2] + current_box_array[2:])
        previous_diagonal = float(np.linalg.norm(previous_box_array[2:] - previous_box_array[:2]))
        current_diagonal = float(np.linalg.norm(current_box_array[2:] - current_box_array[:2]))
        diagonal = max(0.5 * (previous_diagonal + current_diagonal), 1.0)
        center_distance = float(np.linalg.norm(current_center - previous_center)) / diagonal
        box_terms.append(0.70 * min(center_distance, 4.0) + 0.30 * (1.0 - _bbox_iou(
            previous_box_array, current_box_array
        )))
    if box_terms:
        terms.append(float(np.median(box_terms)))
    if not terms:
        return 1e6, False
    if len(terms) == 1:
        return terms[0], False
    return 0.55 * terms[0] + 0.45 * terms[1], True


def stabilize_hand_identities(
    rows: list[dict[str, Any]],
    *,
    max_frame_gap: int = 8,
    swap_margin: float = 0.25,
    swap_min_cost: float = 0.90,
) -> dict[str, Any]:
    """Keep output identities attached to physical hands across label swaps.

    This swaps complete serialized hand records before anatomy and 3-D
    temporal refinement.  Camera boxes, 2-D skeletons, and SAM3 evidence are
    not changed; only the exported side/track assignment is corrected.
    """
    if not rows:
        return {
            "enabled": True,
            "algorithm": "bidirectional_multiview_bbox_palm_identity_v1",
            "swapped_frame_count": 0,
            "swapped_hand_count": 0,
            "ambiguous_frame_count": 0,
            "max_frame_gap": int(max_frame_gap),
            "swap_margin": float(swap_margin),
            "swap_min_cost": float(swap_min_cost),
        }
    rows.sort(key=lambda row: int(row.get("sync_index", 0)))
    tracks: dict[int, dict[str, Any]] = {}
    last_frame: int | None = None
    swapped_frames = 0
    swapped_hands = 0
    ambiguous_frames = 0
    swap_events: list[int] = []

    for row in rows:
        sync_index = int(row.get("sync_index", 0))
        frame_gap = max(sync_index - last_frame, 1) if last_frame is not None else 1
        if last_frame is not None and frame_gap > max(int(max_frame_gap), 1):
            tracks = {}
        current = {
            int(hand.get("side", -1)): hand
            for hand in row.get("hands", [])
            if int(hand.get("side", -1)) in (0, 1)
        }
        assigned: dict[int, dict[str, Any]]
        if len(current) == 2 and len(tracks) == 2:
            identity_cost = sum(
                _hand_match_cost(current[side], tracks[side], frame_gap)[0]
                for side in (0, 1)
            )
            swap_cost = (
                _hand_match_cost(current[1], tracks[0], frame_gap)[0]
                + _hand_match_cost(current[0], tracks[1], frame_gap)[0]
            )
            if (
                identity_cost >= float(swap_min_cost)
                and swap_cost + float(swap_margin) < identity_cost
                and swap_cost <= identity_cost * 0.78
            ):
                assigned = {0: current[1], 1: current[0]}
                swapped_frames += 1
                swapped_hands += 2
                swap_events.append(sync_index)
            else:
                assigned = current
                if abs(identity_cost - swap_cost) < float(swap_margin):
                    ambiguous_frames += 1
        elif current and tracks:
            # Partial rows are valid when only one physical hand passes the
            # multiview gate.  The old two-hand assumption indexed both
            # tracks unconditionally and crashed on ``tracks[missing_side]``.
            # Match each current hand against the tracks that actually exist;
            # with two current hands, use the one-to-one assignment with the
            # lowest available cost.
            assigned = {}
            remaining_tracks = set(tracks)
            for candidate_side, candidate in current.items():
                if not remaining_tracks:
                    target_side = candidate_side
                    if target_side in assigned:
                        target_side = 1 - target_side
                    assigned[target_side] = candidate
                    continue
                costs = {
                    track_side: _hand_match_cost(
                        candidate, tracks[track_side], frame_gap
                    )[0]
                    for track_side in remaining_tracks
                }
                best_side = min(costs, key=costs.get)
                if costs[best_side] < 1e6:
                    target_side = best_side
                    remaining_tracks.remove(best_side)
                else:
                    target_side = candidate_side
                # A one-hand row may leave only one prior track.  When the
                # next frame contains both hands, the first candidate can
                # claim that track while the second candidate carries the
                # same detector side.  Never overwrite the first hand: the
                # remaining physical side belongs to the second candidate.
                if target_side in assigned:
                    target_side = 1 - target_side
                assigned[target_side] = candidate
        else:
            assigned = current

        if assigned != current:
            row["hands"] = [assigned[side] for side in sorted(assigned)]
        for side, hand in assigned.items():
            hand["side"] = int(side)
            hand["temporal_identity"] = {
                "enabled": True,
                "swapped": bool(assigned.get(side) is not current.get(side)),
                "track_side": int(side),
            }
        tracks = assigned.copy()
        last_frame = sync_index

    return {
        "enabled": True,
        "algorithm": "bidirectional_multiview_bbox_palm_identity_v1",
        "swapped_frame_count": int(swapped_frames),
        "swapped_hand_count": int(swapped_hands),
        "ambiguous_frame_count": int(ambiguous_frames),
        "swap_events_preview": swap_events[:100],
        "max_frame_gap": int(max_frame_gap),
        "swap_margin": float(swap_margin),
        "swap_min_cost": float(swap_min_cost),
    }


def _finite_point(point: Any, palm: np.ndarray, max_radius_m: float) -> bool:
    point = np.asarray(point, dtype=np.float64)
    if point.shape != (3,) or not np.isfinite(point).all():
        return False
    # The base-frame workspace is sub-metre in the recorded sequences.  This
    # catches a ray intersection that escaped the camera volume before it can
    # become the state of a forward/backward tracker.
    if float(np.linalg.norm(point)) > 2.0:
        return False
    return (
        not np.isfinite(palm).all()
        or float(np.linalg.norm(point - palm)) <= max_radius_m
    )


def _rays_for_joint(
    hand: Mapping[str, Any],
    joint: int,
    calibrations: Mapping[str, Any],
) -> list[tuple[Any, np.ndarray, np.ndarray, np.ndarray, float]]:
    """Build calibrated rays and a conservative per-view quality weight.

    ``sam3_assist`` is deliberately used only as a region/visibility prior.
    It never supplies a joint location by itself.  A hard SAM3 outlier is
    therefore not allowed to dominate triangulation, while an absent SAM3
    record keeps the original unit weight and remains backwards compatible.
    """
    rays: list[tuple[Any, np.ndarray, np.ndarray, np.ndarray, float]] = []
    for camera_id, view in hand.get("views", {}).items():
        camera = calibrations.get(camera_id)
        observations = np.asarray(view.get("joints_2d", []), dtype=np.float64)
        if (
            camera is None
            or int(view.get("inlier_joint_count", 0)) <= 0
            or observations.shape != (21, 2)
            or not np.isfinite(observations[joint]).all()
        ):
            continue
        camera_ray, valid = unproject_points(camera, observations[joint:joint + 1])
        if not bool(np.asarray(valid, dtype=bool)[0]):
            continue
        rotation = camera.T_base_camera[:3, :3]
        center = camera.T_base_camera[:3, 3]
        direction = rotation @ np.asarray(camera_ray[0], dtype=np.float64)
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-12 or not np.isfinite(direction).all():
            continue
        view_weight = 1.0
        sam3 = view.get("sam3_assist", {})
        if isinstance(sam3, Mapping) and bool(sam3.get("enabled")):
            # A missing/occluded joint is still usable as a weak ray because
            # SAM3 masks can be conservative at fingertips.  ``hard_outlier``
            # is the stronger signal: it means the 2-D point is outside the
            # tolerant mask and receives only a small residual weight.
            visible_flags = sam3.get("visible_flags", [])
            hard_flags = sam3.get("hard_outlier_flags", [])
            if len(visible_flags) == 21 and not bool(visible_flags[joint]):
                view_weight *= 0.45
            if len(hard_flags) == 21 and bool(hard_flags[joint]):
                view_weight *= 0.18
            score = float(sam3.get("mask_score", 1.0))
            box_iou = float(sam3.get("mask_box_iou_yolo", 1.0))
            view_weight *= float(np.clip(0.65 + 0.35 * score, 0.45, 1.0))
            view_weight *= float(np.clip(0.75 + 0.25 * box_iou, 0.55, 1.0))
        rays.append((camera, center, direction / norm, observations[joint], view_weight))
    return rays


def _ray_angle_weights(
    rays: list[tuple[Any, np.ndarray, np.ndarray, np.ndarray, float]],
) -> np.ndarray:
    """Return geometry/SAM3 weights for each ray.

    A view whose ray is nearly parallel to *all* other selected views has a
    weak depth baseline.  Its contribution is reduced according to the
    largest sine angle available to that view; this keeps a single good
    baseline useful while preventing a cluster of almost-parallel cameras
    from pulling a joint onto an arbitrary depth branch.
    """
    if not rays:
        return np.empty(0, dtype=np.float64)
    if len(rays) == 1:
        return np.asarray([float(np.clip(rays[0][4], 0.0, 1.0))], dtype=np.float64)
    reference = float(np.sin(np.deg2rad(RAY_ANGLE_REFERENCE_DEGREES)))
    diversity: list[float] = []
    for index, ray in enumerate(rays):
        direction = np.asarray(ray[2], dtype=np.float64)
        sines = []
        for other_index, other in enumerate(rays):
            if other_index == index:
                continue
            other_direction = np.asarray(other[2], dtype=np.float64)
            # ``abs(dot)`` makes this an angle between ray axes, so a camera
            # looking from the opposite side is correctly treated as a large
            # baseline rather than a sign error.
            sine = float(np.linalg.norm(np.cross(direction, other_direction)))
            sines.append(sine)
        diversity.append(max(sines, default=0.0))
    geometry = np.clip(np.asarray(diversity, dtype=np.float64) / max(reference, 1e-6),
                       RAY_ANGLE_WEIGHT_FLOOR, 1.0)
    sam3 = np.clip(np.asarray([float(ray[4]) for ray in rays], dtype=np.float64),
                   0.05, 1.0)
    return geometry * sam3


def _ray_pair_angles_degrees(
    rays: list[tuple[Any, np.ndarray, np.ndarray, np.ndarray, float]],
) -> np.ndarray:
    """Return all pairwise ray-axis angles in degrees for diagnostics."""
    values: list[float] = []
    for first in range(len(rays)):
        for second in range(first + 1, len(rays)):
            dot = float(np.clip(abs(np.dot(rays[first][2], rays[second][2])), 0.0, 1.0))
            values.append(float(np.degrees(np.arccos(dot))))
    return np.asarray(values, dtype=np.float64)


def _regularized_ray_point(
    rays: list[tuple[Any, np.ndarray, np.ndarray, np.ndarray, float]],
    prior: np.ndarray,
    weight: float,
) -> np.ndarray:
    identity = np.eye(3, dtype=np.float64)
    matrix = max(float(weight), 1e-6) * identity
    vector = max(float(weight), 1e-6) * np.asarray(prior, dtype=np.float64)
    ray_weights = _ray_angle_weights(rays)
    for ray_index, (_camera, center, direction, _observation, _quality_weight) in enumerate(rays):
        projector = identity - np.outer(direction, direction)
        contribution = float(ray_weights[ray_index]) if ray_index < len(ray_weights) else 1.0
        matrix += contribution * projector
        vector += contribution * (projector @ center)
    try:
        point = np.linalg.solve(matrix, vector)
    except np.linalg.LinAlgError:
        point = np.asarray(prior, dtype=np.float64).copy()
    return point


def _ray_residuals(
    point: np.ndarray,
    rays: list[tuple[Any, np.ndarray, np.ndarray, np.ndarray, float]],
) -> np.ndarray:
    values: list[float] = []
    point = np.asarray(point, dtype=np.float64)
    for camera, center, _direction, observation, _quality_weight in rays:
        rotation = camera.T_base_camera[:3, :3]
        point_camera = rotation.T @ (point - center)
        projected, valid = project_points(camera, point_camera[None])
        if bool(np.asarray(valid, dtype=bool)[0]) and np.isfinite(projected[0]).all():
            values.append(float(np.linalg.norm(projected[0] - observation)))
    return np.asarray(values, dtype=np.float64)


def _quality(
    point: np.ndarray,
    rays: list[tuple[Any, np.ndarray, np.ndarray, np.ndarray]],
) -> tuple[float, float]:
    residuals = _ray_residuals(point, rays)
    if not len(residuals):
        return float("inf"), float("inf")
    return (
        float(np.median(residuals)),
        float(np.percentile(residuals, 95)),
    )


def _ray_condition_number(
    rays: list[tuple[Any, np.ndarray, np.ndarray, np.ndarray, float]],
) -> float:
    """Return the conditioning of the perpendicular-ray least-squares solve.

    A hand joint can have six valid 2-D observations while the corresponding
    rays are almost parallel.  In that case the image residual remains small
    but depth is effectively unconstrained, so an unregularized triangulation
    is free to jump between very different 3-D points.  The eigenvalue ratio
    of ``sum(I - ddᵀ)`` is a compact, calibration-aware measure of that
    ambiguity (smallest eigenvalue = weakly observed direction).
    """
    if len(rays) < 2:
        return float("inf")
    identity = np.eye(3, dtype=np.float64)
    matrix = np.zeros((3, 3), dtype=np.float64)
    for _camera, _center, direction, _observation, _quality_weight in rays:
        matrix += identity - np.outer(direction, direction)
    try:
        eigenvalues = np.linalg.eigvalsh(matrix)
    except np.linalg.LinAlgError:
        return float("inf")
    smallest = max(float(eigenvalues[0]), 1e-9)
    return float(max(eigenvalues[-1], 0.0) / smallest)


def _track_direction(
    points: np.ndarray,
    hands: list[Mapping[str, Any]],
    joint: int,
    frame_ids: np.ndarray,
    calibrations: Mapping[str, Any],
    rays_cache: list[list[list[tuple[Any, np.ndarray, np.ndarray, np.ndarray, float]]]] | None = None,
    *,
    forward: bool,
    ray_prior_weight: float,
    ray_blend: float,
    max_step_m: float,
    max_radius_m: float,
    max_reprojection_median_px: float,
    max_reprojection_p95_px: float,
    max_missing_gap: int,
) -> np.ndarray:
    """Track one joint in one direction without filling rejected intervals."""
    count = len(points)
    output = np.full((count, 3), np.nan, dtype=np.float64)
    state = np.full(3, np.nan, dtype=np.float64)
    velocity = np.zeros(3, dtype=np.float64)
    previous_index: int | None = None
    missing = 0
    indices = range(count) if forward else range(count - 1, -1, -1)
    for index in indices:
        raw = np.asarray(points[index], dtype=np.float64)
        palm = _palm_center(np.asarray(hands[index].get("joints_base_m", [])))
        raw_valid = _finite_point(raw, palm, max_radius_m)
        if not raw_valid:
            missing += 1
            output[index] = np.nan
            if missing > max(int(max_missing_gap), 0):
                state[:] = np.nan
                velocity[:] = 0.0
                previous_index = None
            continue
        missing = 0
        if not np.isfinite(state).all() or previous_index is None:
            state = raw.copy()
            output[index] = state
            previous_index = index
            continue

        delta_frame = max(
            abs(int(frame_ids[index]) - int(frame_ids[previous_index])), 1
        )
        # ``rows`` contains accepted frames only, so a rejected interval is
        # represented by a jump in ``frame_ids`` rather than by explicit
        # NaN samples.  Never extrapolate a velocity across such an interval:
        # the first observation after a gap is a new measurement and must
        # re-anchor the track.  Otherwise a stale velocity can move an entire
        # hand onto a different depth branch (especially after an occlusion).
        if delta_frame > max(int(max_missing_gap), 0) + 1:
            state = raw.copy()
            velocity[:] = 0.0
            output[index] = state
            previous_index = index
            continue
        prediction = state + velocity * delta_frame
        if not _finite_point(prediction, palm, max_radius_m):
            prediction = state.copy()
        rays = (
            rays_cache[index][joint]
            if rays_cache is not None
            else _rays_for_joint(hands[index], joint, calibrations)
        )
        candidate = prediction.copy()
        if len(rays) >= 2:
            # Do not let a nearly parallel set of rays overwrite a trusted
            # temporal state.  The condition number is computed from the
            # actual selected views, so this automatically becomes strict for
            # the troublesome side-1 frames where only a narrow baseline is
            # visible and remains permissive when the views triangulate well.
            conditioning = _ray_condition_number(rays)
            effective_prior_weight = float(ray_prior_weight)
            blend_scale = 1.0
            if conditioning > RAY_CONDITIONING_SOFT_THRESHOLD:
                ambiguity = float(np.clip(
                    (conditioning - RAY_CONDITIONING_SOFT_THRESHOLD)
                    / (RAY_CONDITIONING_SATURATION - RAY_CONDITIONING_SOFT_THRESHOLD),
                    0.0, 1.0,
                ))
                effective_prior_weight *= 1.0 + 7.0 * ambiguity
                blend_scale = 1.0 - 0.82 * ambiguity
            if conditioning >= RAY_CONDITIONING_HARD_THRESHOLD:
                # With an almost unobservable depth axis, even a regularized
                # least-squares point can move to the wrong branch.  Keep the
                # bidirectional temporal prediction as the measurement for
                # this joint; the raw point is still considered below when it
                # agrees with that prediction and has good image support.
                candidate = prediction.copy()
            else:
                solved = _regularized_ray_point(
                    rays, prediction, effective_prior_weight,
                )
                if _finite_point(solved, palm, max_radius_m):
                    move = solved - prediction
                    move_norm = float(np.linalg.norm(move))
                    limit = max_step_m * delta_frame
                    if move_norm > limit and move_norm > 1e-12:
                        solved = prediction + move * (limit / move_norm)
                    blend = float(np.clip(ray_blend * blend_scale, 0.0, 1.0))
                    candidate = (1.0 - blend) * prediction + blend * solved
            if not _finite_point(candidate, palm, max_radius_m):
                candidate = prediction
            candidate_quality = _quality(candidate, rays)
            raw_quality = _quality(raw, rays)
            # A nearby raw point is preferred when it is at least as supported
            # by the image.  Otherwise reject an invalid ray branch.
            if (
                np.linalg.norm(raw - prediction) <= 0.018 * delta_frame
                and raw_quality[0] <= candidate_quality[0] + 2.0
            ):
                candidate = raw
            elif (
                candidate_quality[0] > max_reprojection_median_px
                or candidate_quality[1] > max_reprojection_p95_px
            ) and raw_quality[0] < candidate_quality[0]:
                candidate = raw
        elif np.linalg.norm(raw - prediction) <= 0.018 * delta_frame:
            candidate = raw

        if not _finite_point(candidate, palm, max_radius_m):
            candidate = prediction
        step = (candidate - state) / delta_frame
        velocity = 0.65 * velocity + 0.35 * np.clip(step, -0.06, 0.06)
        state = candidate
        output[index] = state
        previous_index = index
    return output


def _gaussian_smooth_runs(
    points: np.ndarray,
    original_valid: np.ndarray,
    sigma: float,
    frame_ids: np.ndarray | None = None,
) -> np.ndarray:
    """Zero-phase Gaussian smoothing, never crossing an invalid gap."""
    if sigma <= 0.0:
        return points.copy()
    radius = max(int(np.ceil(3.0 * sigma)), 1)
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    kernel /= np.sum(kernel)
    output = points.copy()
    for side in range(points.shape[0]):
        for joint in range(points.shape[2]):
            valid = original_valid[side, :, joint]
            indices = np.flatnonzero(valid)
            if not len(indices):
                continue
            breaks = np.flatnonzero(np.diff(indices) > 1)
            if frame_ids is not None and len(indices) > 1:
                breaks = np.union1d(
                    breaks,
                    np.flatnonzero(np.diff(frame_ids[indices]) > 1),
                )
            runs = np.split(indices, breaks + 1)
            for run in runs:
                if len(run) < 2:
                    continue
                values = points[side, run, joint]
                # Reflecting endpoints avoids pulling the first/last valid
                # point toward zero while still remaining zero-phase.
                padded = np.pad(values, ((radius, radius), (0, 0)), mode="edge")
                output[side, run, joint] = np.asarray([
                    np.sum(
                        padded[index:index + 2 * radius + 1]
                        * kernel[:, None], axis=0
                    )
                    for index in range(len(run))
                ])
    output[~original_valid] = np.nan
    return output


def _remove_isolated_step_spikes(
    points: np.ndarray,
    original_valid: np.ndarray,
    frame_ids: np.ndarray,
    max_step_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Repair a one-frame excursion left by an anatomical projection.

    Bone projection is intentionally applied after temporal filtering, because
    it can restore a physically plausible finger length.  On an ill-conditioned
    ray set that projection can nevertheless move one joint farther than the
    configured per-frame limit.  This pass only touches an *isolated* peak:
    the two neighbouring samples must be present and the direct bridge across
    the sample must be smooth.  Sustained fast motion (both sides moving at a
    similar speed) is retained.
    """
    output = np.asarray(points, dtype=np.float64).copy()
    changed = np.zeros(original_valid.shape, dtype=bool)
    if output.ndim != 3 or output.shape[0] != len(frame_ids):
        return output, changed
    cap = max(float(max_step_m), 1e-6)
    for _ in range(2):
        for joint in range(output.shape[1]):
            for index in range(1, len(frame_ids) - 1):
                if (
                    index < 2
                    or index + 2 >= len(frame_ids)
                    or not original_valid[index, joint]
                    or not original_valid[index - 1, joint]
                    or not original_valid[index + 1, joint]
                    or int(frame_ids[index]) - int(frame_ids[index - 1]) != 1
                    or int(frame_ids[index + 1]) - int(frame_ids[index]) != 1
                ):
                    continue
                previous = output[index - 1, joint]
                current = output[index, joint]
                following = output[index + 1, joint]
                if not (
                    np.isfinite(previous).all()
                    and np.isfinite(current).all()
                    and np.isfinite(following).all()
                ):
                    continue
                incoming = float(np.linalg.norm(current - previous))
                outgoing = float(np.linalg.norm(following - current))
                bridge = float(np.linalg.norm(following - previous) / 2.0)
                peak = max(incoming, outgoing)
                # A finger can show a large 3-D branch while the palm itself
                # moves only a few centimetres.  In that case the joint is an
                # articulation/depth spike even when its two endpoints are
                # farther apart than the conservative global bridge limit.
                previous_palm = _palm_center(output[index - 1])
                following_palm = _palm_center(output[index + 1])
                palm_bridge = (
                    float(np.linalg.norm(following_palm - previous_palm) / 2.0)
                    if np.isfinite(previous_palm).all()
                    and np.isfinite(following_palm).all()
                    else float("inf")
                )
                articulation_spike = (
                    palm_bridge <= 0.060
                    and peak > max(0.080, cap * 1.50)
                )
                if peak <= cap * 1.05 or (
                    bridge > cap * 1.30 and not articulation_spike
                ):
                    continue
                # Compare the peak with the local track rather than applying a
                # hard clamp to a genuinely quick, monotonic gesture.
                local_speeds: list[float] = []
                for neighbour in range(max(1, index - 2), min(len(frame_ids), index + 3)):
                    if neighbour == index:
                        continue
                    if (
                        int(frame_ids[neighbour]) - int(frame_ids[neighbour - 1]) != 1
                        or not original_valid[neighbour, joint]
                        or not original_valid[neighbour - 1, joint]
                    ):
                        continue
                    left = output[neighbour - 1, joint]
                    right = output[neighbour, joint]
                    if np.isfinite(left).all() and np.isfinite(right).all():
                        local_speeds.append(float(np.linalg.norm(right - left)))
                local_median = float(np.median(local_speeds)) if local_speeds else cap
                direction_change = float(np.dot(current - previous, following - current))
                isolated = (
                    articulation_spike
                    or peak > max(cap * 1.05, local_median * 1.45)
                    or direction_change < 0.0
                )
                if not isolated:
                    continue
                target = previous + (following - previous) * 0.5
                if np.linalg.norm(current - target) <= 0.002:
                    continue
                output[index, joint] = target
                changed[index, joint] = True
    return output, changed


def _limit_temporal_steps(
    points: np.ndarray,
    original_valid: np.ndarray,
    frame_ids: np.ndarray,
    max_step_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Repair short 3-D branch runs without manufacturing fixed-size jumps.

    The previous implementation clipped every excessive step to
    ``max_step_m``.  That made a bad depth branch look like a sequence of
    identical 25/40-mm jumps, which is still visibly wrong.  We now detect a
    contiguous run of excessive steps and replace only the interior with a
    zero-phase interpolation between its two trusted endpoints.  A run whose
    endpoints themselves imply genuinely fast motion is retained instead of
    being silently slowed.  Missing joints and frame gaps are never filled.
    """
    output = np.asarray(points, dtype=np.float64).copy()
    changed = np.zeros(original_valid.shape, dtype=bool)
    if output.ndim != 3 or output.shape[0] != len(frame_ids):
        return output, changed
    cap = max(float(max_step_m), 1e-6)

    # Work joint-by-joint and never cross an upstream missing sample.  The
    # endpoint-speed allowance prevents a real, monotonic fast gesture from
    # being flattened merely because it exceeds the conservative branch cap.
    for _ in range(3):
        for joint in range(output.shape[1]):
            valid = (
                original_valid[:, joint]
                & np.isfinite(output[:, joint]).all(axis=1)
            )
            indices = np.flatnonzero(valid)
            if len(indices) < 3:
                continue
            breaks = np.flatnonzero(np.diff(indices) > 1)
            if len(indices) > 1:
                breaks = np.union1d(
                    breaks,
                    np.flatnonzero(np.diff(frame_ids[indices]) > 1),
                )
            runs = np.split(indices, breaks + 1)
            for run in runs:
                if len(run) < 3:
                    continue
                # The run is contiguous in array/frame space by construction.
                values = output[run, joint]
                steps = np.linalg.norm(np.diff(values, axis=0), axis=1)
                excessive = steps > cap
                if not np.any(excessive):
                    continue
                starts = np.flatnonzero(
                    excessive & np.r_[True, ~excessive[:-1]]
                )
                ends = np.flatnonzero(
                    excessive & np.r_[~excessive[1:], True]
                )
                for start_interval, end_interval in zip(starts, ends):
                    left = int(start_interval)
                    right = int(end_interval) + 1
                    # Need one trusted point on each side of the excessive
                    # run; this avoids inventing an endpoint or crossing a
                    # deliberately missing interval.
                    if left <= 0 or right >= len(run) - 1:
                        continue
                    left_index = int(run[left - 1])
                    right_index = int(run[right])
                    if (
                        int(frame_ids[right_index])
                        - int(frame_ids[left_index])
                        != right_index - left_index
                    ):
                        continue
                    endpoint_speed = float(
                        np.linalg.norm(
                            output[right_index, joint]
                            - output[left_index, joint]
                        )
                        / max(right_index - left_index, 1)
                    )
                    # If both endpoints demand a genuinely fast monotonic
                    # motion, retain it; otherwise this is a depth branch.
                    if endpoint_speed > cap * 1.50:
                        continue
                    for index in range(left, right):
                        current_index = int(run[index])
                        alpha = (current_index - left_index) / max(
                            right_index - left_index, 1
                        )
                        target = (
                            (1.0 - alpha) * output[left_index, joint]
                            + alpha * output[right_index, joint]
                        )
                        if np.isfinite(target).all() and float(
                            np.linalg.norm(output[current_index, joint] - target)
                        ) > 0.002:
                            output[current_index, joint] = target
                            changed[current_index, joint] = True

    output[~original_valid] = np.nan
    return output, changed


def _remove_boundary_tail_branches(
    points: np.ndarray,
    original_valid: np.ndarray,
    frame_ids: np.ndarray,
    max_step_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Drop an untrusted branch that starts after a stable run and ends in a gap.

    A depth/association error can be internally smooth for a few samples, so
    the ordinary isolated-spike test cannot identify it.  If a large step
    follows at least two small steps and the finite run immediately terminates
    at a missing frame, there is no trusted endpoint to interpolate toward;
    retaining that tail would expose a whole wrong finger branch.  We keep the
    last trusted sample and mark only the branch tail missing.  A two-sample
    run after an occlusion is handled conservatively by the same rule.
    """
    output = np.asarray(points, dtype=np.float64).copy()
    cleared = np.zeros(original_valid.shape, dtype=bool)
    if output.ndim != 3 or output.shape[0] != len(frame_ids):
        return output, cleared
    cap = max(float(max_step_m), 1e-6)
    # 80 mm/frame is a conservative physical outlier threshold for this
    # sequence; unlike the old 25/40-mm clamp it only decides when to discard
    # an untrusted tail and never clips a retained trajectory.
    large_step = max(0.080, cap * 3.20)
    stable_step = max(0.035, cap * 1.40)
    for joint in range(output.shape[1]):
        valid = original_valid[:, joint] & np.isfinite(output[:, joint]).all(axis=1)
        indices = np.flatnonzero(valid)
        if len(indices) < 2:
            continue
        breaks = np.flatnonzero(np.diff(indices) > 1)
        if len(indices) > 1:
            breaks = np.union1d(
                breaks,
                np.flatnonzero(np.diff(frame_ids[indices]) > 1),
            )
        runs = np.split(indices, breaks + 1)
        for run in runs:
            if len(run) < 2:
                continue
            # A tail is only actionable when the next accepted sample is
            # absent or separated by a rejected frame.  This prevents the
            # guard from deleting a genuinely fast, fully observed gesture.
            run_end = int(run[-1])
            tail_ends_at_gap = (
                run_end + 1 >= len(frame_ids)
                or not original_valid[run_end + 1, joint]
                or not np.isfinite(output[run_end + 1, joint]).all()
                or int(frame_ids[run_end + 1]) - int(frame_ids[run_end]) > 1
            )
            if not tail_ends_at_gap:
                continue
            values = output[run, joint]
            steps = np.linalg.norm(np.diff(values, axis=0), axis=1)
            for step_index, step in enumerate(steps):
                if step <= large_step:
                    continue
                prior = steps[max(0, step_index - 2):step_index]
                stable_history = len(prior) >= 2 and float(np.median(prior)) <= stable_step
                short_reappearance = len(run) <= 2 and step_index == 0
                if not (stable_history or short_reappearance):
                    continue
                # Keep the sample before the branch and remove the remainder
                # of this finite tail.  Do not touch points before the onset.
                for offset in range(step_index + 1, len(run)):
                    index = int(run[offset])
                    if np.isfinite(output[index, joint]).all():
                        output[index, joint] = np.nan
                        cleared[index, joint] = True
                # For a two-sample reappearance the second sample is the
                # branch onset itself and must also be removed.
                if short_reappearance:
                    index = int(run[step_index + 1])
                    if np.isfinite(output[index, joint]).all():
                        output[index, joint] = np.nan
                        cleared[index, joint] = True
                break
    return output, cleared


def _invalidate_excessive_temporal_jumps(
    points: np.ndarray,
    original_valid: np.ndarray,
    frame_ids: np.ndarray,
    max_jump_m: float = 0.080,
) -> tuple[np.ndarray, np.ndarray]:
    """Mark unresolved single-frame jumps missing instead of clipping them.

    This is a last-resort visibility gate, not a motion clamp.  If a point
    still moves farther than the physically conservative threshold after all
    bidirectional, anatomical, and gap-aware repairs, the newer endpoint is
    discarded.  A missing endpoint is preferable to exporting a known wrong
    depth branch, and it also prevents the old fixed-size step artifact.
    """
    output = np.asarray(points, dtype=np.float64).copy()
    cleared = np.zeros(original_valid.shape, dtype=bool)
    if output.ndim != 3 or output.shape[0] != len(frame_ids):
        return output, cleared
    threshold = max(float(max_jump_m), 1e-6)
    # Repeating the scan handles a branch whose first deletion exposes a
    # second excessive edge while still terminating quickly.
    for _ in range(3):
        changed = False
        for joint in range(output.shape[1]):
            for index in range(1, len(frame_ids)):
                if (
                    not original_valid[index, joint]
                    or not original_valid[index - 1, joint]
                    or int(frame_ids[index]) - int(frame_ids[index - 1]) != 1
                    or not np.isfinite(output[index, joint]).all()
                    or not np.isfinite(output[index - 1, joint]).all()
                ):
                    continue
                if float(np.linalg.norm(output[index, joint] - output[index - 1, joint])) <= threshold:
                    continue
                output[index, joint] = np.nan
                cleared[index, joint] = True
                changed = True
        if not changed:
            break
    return output, cleared


def _view_points_for_temporal_guard(view: Mapping[str, Any]) -> np.ndarray:
    """Return the corrected camera-space points used by the view guard.

    ``joints_2d`` is the SAM3/refined observation when available.  The raw
    WiLoR points are a backwards-compatible fallback for older JSONL files.
    This helper intentionally does not alter either observation.
    """
    for key in ("joints_2d", "wilor_joints_2d"):
        points = np.asarray(view.get(key, []), dtype=np.float64)
        if points.shape == (21, 2) and np.isfinite(points).all():
            return points
    return np.full((21, 2), np.nan, dtype=np.float64)


def _bbox_area(view: Mapping[str, Any]) -> float:
    box = np.asarray(view.get("bbox_xyxy", []), dtype=np.float64)
    if box.shape != (4,) or not np.isfinite(box).all():
        return 0.0
    return max(float(box[2] - box[0]), 0.0) * max(float(box[3] - box[1]), 0.0)


def _hand_has_camera_view_jump(
    current: Mapping[str, Any],
    previous: Mapping[str, Any],
    *,
    wrist_jump_px: float,
    expanded_box_ratio: float,
    expanded_box_wrist_px: float,
) -> tuple[bool, list[str]]:
    """Detect a detector association jump before it reaches 3-D.

    A merged two-hand detector box can have a perfectly plausible SAM3 mask
    and reprojection error, yet its wrist/box suddenly moves to the other
    hand.  Such a sample cannot be repaired by 3-D smoothing after
    triangulation.  We therefore invalidate the whole hand for that sample
    when a common camera shows an unmistakable one-frame association jump.
    The thresholds are deliberately high; ordinary hand motion remains
    untouched and a rejected sample is preferable to exporting a wrong pose.
    """
    current_views = current.get("views", {})
    previous_views = previous.get("views", {})
    suspicious: list[str] = []
    for camera in set(current_views).intersection(previous_views):
        current_view = current_views[camera]
        previous_view = previous_views[camera]
        current_points = _view_points_for_temporal_guard(current_view)
        previous_points = _view_points_for_temporal_guard(previous_view)
        if not (
            np.isfinite(current_points).all()
            and np.isfinite(previous_points).all()
        ):
            continue
        wrist = float(np.linalg.norm(current_points[0] - previous_points[0]))
        current_area = _bbox_area(current_view)
        previous_area = _bbox_area(previous_view)
        area_ratio = (
            current_area / max(previous_area, 1.0)
            if current_area > 0.0 and previous_area > 0.0 else 1.0
        )
        # The area condition is useful for a box that suddenly contains both
        # hands.  It still requires a substantial wrist movement, so a normal
        # zoom or crop change does not clear a pose by itself.
        if wrist >= float(wrist_jump_px) or (
            wrist >= float(expanded_box_wrist_px)
            and area_ratio >= float(expanded_box_ratio)
        ):
            suspicious.append(str(camera))
    return bool(suspicious), suspicious


def _invalidate_view_jump_hands(
    points: np.ndarray,
    original_valid: np.ndarray,
    hands: list[Mapping[str, Any]],
    previous_hands: list[Mapping[str, Any] | None],
    frame_ids: np.ndarray,
    *,
    wrist_jump_px: float = 180.0,
    expanded_box_ratio: float = 1.50,
    expanded_box_wrist_px: float = 130.0,
) -> tuple[np.ndarray, np.ndarray, dict[int, list[str]]]:
    """Clear a hand whose shared camera has a one-frame association jump."""
    output = np.asarray(points, dtype=np.float64).copy()
    cleared = np.zeros(original_valid.shape, dtype=bool)
    events: dict[int, list[str]] = {}
    if output.ndim != 3 or output.shape[0] != len(frame_ids):
        return output, cleared, events
    for index in range(1, len(frame_ids)):
        if int(frame_ids[index]) - int(frame_ids[index - 1]) != 1:
            continue
        previous = previous_hands[index]
        current = hands[index]
        if previous is None or current is None:
            continue
        jump, cameras = _hand_has_camera_view_jump(
            current, previous,
            wrist_jump_px=wrist_jump_px,
            expanded_box_ratio=expanded_box_ratio,
            expanded_box_wrist_px=expanded_box_wrist_px,
        )
        if not jump or not np.isfinite(output[index]).any():
            continue
        output[index] = np.nan
        cleared[index] = original_valid[index]
        events[int(frame_ids[index])] = cameras
    return output, cleared, events


def stabilize_joint_temporal_rows(
    rows: list[dict[str, Any]],
    calibrations: Mapping[str, Any],
    *,
    ray_prior_weight: float = 3.0,
    ray_blend: float = 0.80,
    max_step_m: float = 0.040,
    max_radius_m: float = 0.55,
    max_missing_gap: int = 2,
    local_radius: int = 3,
    local_outlier_threshold_m: float = 0.025,
    local_candidate_ratio: float = 0.82,
    smoothing_sigma: float = 1.30,
    bone_iterations: int = 15,
    bone_strength: float = 0.80,
    max_reprojection_median_px: float = 35.0,
    max_reprojection_p95_px: float = 100.0,
    min_hand_joints: int = 18,
    excessive_jump_threshold_m: float = 0.040,
    view_wrist_jump_px: float = 180.0,
    view_expanded_box_ratio: float = 1.50,
    view_expanded_box_wrist_px: float = 130.0,
) -> dict[str, Any]:
    """Stabilize finite 3-D joints in-place while preserving 2-D evidence."""
    if not rows:
        return {"enabled": True, "algorithm": ALGORITHM, "corrected_joint_count": 0}
    ordered = sorted(rows, key=lambda row: int(row["sync_index"]))
    frame_ids = np.asarray([int(row["sync_index"]) for row in ordered], dtype=np.int64)
    corrected = np.zeros((2, len(ordered), 21), dtype=bool)
    original_valid = np.zeros((2, len(ordered), 21), dtype=bool)
    replacement_count = 0
    direction_count = 0
    ray_condition_values: list[float] = []
    ray_min_angle_values: list[float] = []
    ray_count_distribution: dict[str, int] = {}
    sam3_downweighted_ray_count = 0
    boundary_tail_cleared_count = 0
    excessive_jump_cleared_count = 0
    sparse_hand_cleared_count = 0
    sparse_joint_cleared_count = 0
    post_temporal_sparse_hand_cleared_count = 0
    post_temporal_sparse_joint_cleared_count = 0
    view_jump_cleared_count = 0
    view_jump_cleared_joint_count = 0
    view_jump_events: dict[tuple[int, int], list[str]] = {}
    # Accepted rows are sparse when a whole frame is rejected.  The samples
    # immediately before/after a long gap have only one-sided temporal
    # evidence; a bidirectional average there can import the other branch
    # from the far side of an occlusion.  Keep those boundary observations
    # anchored to their anatomical input instead.
    gap_boundary = np.zeros(len(ordered), dtype=bool)
    if len(frame_ids) > 1:
        large_gaps = np.diff(frame_ids) > max(int(max_missing_gap), 0) + 1
        gap_boundary[:-1] |= large_gaps
        gap_boundary[1:] |= large_gaps

    for side in (0, 1):
        hands = [
            next((hand for hand in row.get("hands", [])
                  if int(hand.get("side", -1)) == side), None)
            for row in ordered
        ]
        if not any(hand is not None for hand in hands):
            continue
        hands_present = [hand if hand is not None else {"joints_base_m": []}
                         for hand in hands]
        rays_cache = [
            [
                _rays_for_joint(hand, joint, calibrations)
                for joint in range(21)
            ]
            for hand in hands_present
        ]
        points = np.asarray([
            np.asarray(hand.get("joints_base_m", []), dtype=np.float64)
            if np.asarray(hand.get("joints_base_m", [])).shape == (21, 3)
            else np.full((21, 3), np.nan, dtype=np.float64)
            for hand in hands_present
        ])
        raw_valid = np.isfinite(points).all(axis=2)
        sparse_mask = raw_valid.sum(axis=1) < max(int(min_hand_joints), 0)
        if np.any(sparse_mask):
            # A partial skeleton is more distracting than a short, explicit
            # dropout: after ray conditioning or view rejection, a few MCP /
            # fingertip points can remain and make the rendered hand appear
            # to jump between different articulations.  Treat the hand as a
            # single visibility unit for temporal tracking.
            points[sparse_mask] = np.nan
            sparse_hand_cleared_count += int(np.count_nonzero(sparse_mask))
            sparse_joint_cleared_count += int(np.count_nonzero(sparse_mask)) * 21
        original_valid[side] = np.isfinite(points).all(axis=2)

        # A one-frame merged detector box is not recoverable after the
        # triangulation has already mixed its rays.  Use the camera-native
        # (SAM3-corrected when present) observations to reject that hand
        # before the bidirectional 3-D tracker starts.
        previous_hands: list[Mapping[str, Any] | None] = [None]
        for index in range(1, len(ordered)):
            previous_hands.append(
                hands[index - 1] if hands[index - 1] is not None else None
            )
        view_guard_points, view_guard_cleared, view_events = (
            _invalidate_view_jump_hands(
                points,
                original_valid[side],
                hands_present,
                previous_hands,
                frame_ids,
                wrist_jump_px=view_wrist_jump_px,
                expanded_box_ratio=view_expanded_box_ratio,
                expanded_box_wrist_px=view_expanded_box_wrist_px,
            )
        )
        if view_events:
            points = view_guard_points
            original_valid[side] = np.isfinite(points).all(axis=2)
            view_jump_cleared_count += len(view_events)
            view_jump_cleared_joint_count += int(np.count_nonzero(view_guard_cleared))
            for frame_id, cameras in view_events.items():
                view_jump_events[(side, int(frame_id))] = list(cameras)
        # Record per-joint geometry before any temporal edits.  These values
        # make the summary actionable: a large condition number or a small
        # minimum ray angle identifies a depth-ambiguous joint rather than a
        # generic "bad frame".
        for index, hand in enumerate(hands_present):
            for joint in range(21):
                if not original_valid[side, index, joint]:
                    continue
                rays = rays_cache[index][joint]
                ray_count_distribution[str(len(rays))] = (
                    ray_count_distribution.get(str(len(rays)), 0) + 1
                )
                if len(rays) < 2:
                    continue
                conditioning = _ray_condition_number(rays)
                if np.isfinite(conditioning):
                    ray_condition_values.append(float(conditioning))
                else:
                    ray_condition_values.append(float("inf"))
                angles = _ray_pair_angles_degrees(rays)
                if len(angles):
                    ray_min_angle_values.append(float(np.min(angles)))
                sam3_downweighted_ray_count += sum(
                    float(ray[4]) < 0.999 for ray in rays
                )
        forward = np.full_like(points, np.nan)
        backward = np.full_like(points, np.nan)
        for joint in range(21):
            forward[:, joint] = _track_direction(
                points[:, joint], hands_present, joint, frame_ids, calibrations,
                rays_cache=rays_cache,
                forward=True, ray_prior_weight=ray_prior_weight,
                ray_blend=ray_blend, max_step_m=max_step_m,
                max_radius_m=max_radius_m,
                max_reprojection_median_px=max_reprojection_median_px,
                max_reprojection_p95_px=max_reprojection_p95_px,
                max_missing_gap=max_missing_gap,
            )
            backward[:, joint] = _track_direction(
                points[:, joint], hands_present, joint, frame_ids, calibrations,
                rays_cache=rays_cache,
                forward=False, ray_prior_weight=ray_prior_weight,
                ray_blend=ray_blend, max_step_m=max_step_m,
                max_radius_m=max_radius_m,
                max_reprojection_median_px=max_reprojection_median_px,
                max_reprojection_p95_px=max_reprojection_p95_px,
                max_missing_gap=max_missing_gap,
            )
            direction_count += 2

        candidate = points.copy()
        for joint in range(21):
            for index in range(len(ordered)):
                if not original_valid[side, index, joint]:
                    continue
                if gap_boundary[index]:
                    # Do not combine a one-sided track with the opposite side
                    # of a long rejected interval.  The final assignment below
                    # reapplies this anchor after smoothing/bone projection.
                    continue
                options = [value for value in (forward[index, joint], backward[index, joint])
                           if np.isfinite(value).all()]
                if not options:
                    continue
                candidate_point = np.mean(options, axis=0)
                palm = _palm_center(points[index])
                if not _finite_point(candidate_point, palm, max_radius_m):
                    continue
                # Compare against the original local track.  This is the key
                # branch safeguard: a smooth filtered branch must not replace
                # a raw point that was already continuous.
                before = max(0, index - max(int(local_radius), 1))
                after = min(len(ordered), index + max(int(local_radius), 1) + 1)
                neighbors = [
                    value for neighbor in range(before, after)
                    if neighbor != index
                    and original_valid[side, neighbor, joint]
                    for value in (points[neighbor, joint],)
                ]
                if len(neighbors) < 2:
                    continue
                neighbour_values = np.asarray(neighbors, dtype=np.float64)
                local_target = np.median(neighbour_values, axis=0)
                raw_residual = float(np.linalg.norm(points[index, joint] - local_target))
                candidate_residual = float(np.linalg.norm(candidate_point - local_target))
                # If the current 3-D point is a ray-depth outlier, use the
                # filtered point even when its absolute motion is large.  A
                # small reprojection allowance handles the unavoidable depth
                # ambiguity without accepting a visibly wrong 2-D branch.
                rays = rays_cache[index][joint]
                current_quality = _quality(points[index, joint], rays) if len(rays) >= 2 else (0.0, 0.0)
                candidate_quality = _quality(candidate_point, rays) if len(rays) >= 2 else (0.0, 0.0)
                quality_ok = (
                    len(rays) < 2
                    or (
                        candidate_quality[0] <= current_quality[0] + 8.0
                        and candidate_quality[1] <= current_quality[1] + 30.0
                    )
                )
                if (
                    raw_residual > local_outlier_threshold_m
                    and candidate_residual < raw_residual * local_candidate_ratio
                    and quality_ok
                ):
                    candidate[index, joint] = candidate_point
                    corrected[side, index, joint] = True
                    replacement_count += 1

        # A short robust median pass removes residual one-frame excursions.
        # It does not cross rows whose upstream 3-D point was intentionally
        # cleared, and therefore cannot resurrect an untrusted endpoint.
        for _ in range(3):
            for joint in range(21):
                for index in range(len(ordered)):
                    if not original_valid[side, index, joint] or not np.isfinite(candidate[index, joint]).all():
                        continue
                    lo = max(0, index - max(int(local_radius), 1))
                    hi = min(len(ordered), index + max(int(local_radius), 1) + 1)
                    neighbours = candidate[lo:hi, joint]
                    valid = np.isfinite(neighbours).all(axis=1)
                    if np.count_nonzero(valid) < 3:
                        continue
                    values = neighbours[valid]
                    target = np.median(values, axis=0)
                    spread = np.median(np.linalg.norm(values - target, axis=1))
                    residual = float(np.linalg.norm(candidate[index, joint] - target))
                    if residual > local_outlier_threshold_m and residual > 2.0 * max(spread, 1e-6):
                        candidate[index, joint] = target
                        corrected[side, index, joint] = True

        # Enforce a sequence-specific anatomical scale after temporal edits.
        # Low confidence is assigned to joints that changed, so a correction
        # cannot drag all neighbouring observations to an arbitrary branch.
        support = np.asarray([
            np.asarray(hand.get("inlier_view_counts", []), dtype=np.float64)
            if np.asarray(hand.get("inlier_view_counts", [])).shape == (21,)
            else np.ones(21, dtype=np.float64)
            for hand in hands_present
        ])
        confidence = np.clip(support / 6.0, 0.10, 1.0)
        confidence[corrected[side]] = 0.08
        base_points = points[None]
        valid = np.isfinite(candidate).all(axis=2)[None]
        bone_lengths = estimate_bone_lengths(
            base_points,
            original_valid[side][None],
            confidence[None],
        )
        candidate = constrain_bones(
            candidate[None], valid, confidence[None], bone_lengths,
            max(int(bone_iterations), 0), float(np.clip(bone_strength, 0.0, 1.0)),
        )[0]
        candidate[~original_valid[side]] = np.nan
        candidate = _gaussian_smooth_runs(
            candidate[None], original_valid[side][None], smoothing_sigma,
            frame_ids=frame_ids,
        )[0]
        candidate = constrain_bones(
            candidate[None], np.isfinite(candidate).all(axis=2)[None],
            confidence[None], bone_lengths,
            max(int(bone_iterations), 0), float(np.clip(bone_strength, 0.0, 1.0)),
        )[0]
        candidate[~original_valid[side]] = np.nan
        candidate, spike_corrected = _remove_isolated_step_spikes(
            candidate, original_valid[side], frame_ids, max_step_m,
        )
        corrected[side] |= spike_corrected
        # Re-apply a light anatomical pass after replacing an isolated point;
        # the replacement is an interpolation of its neighbours, so a few
        # iterations are enough and cannot reintroduce the original branch.
        candidate = constrain_bones(
            candidate[None], np.isfinite(candidate).all(axis=2)[None],
            confidence[None], bone_lengths,
            max(min(int(bone_iterations), 6), 0),
            float(np.clip(bone_strength * 0.35, 0.0, 1.0)),
        )[0]
        candidate[~original_valid[side]] = np.nan

        # Final zero-phase continuity guard.  This runs after the last bone
        # pass so anatomical projection cannot reintroduce a depth branch.
        candidate, continuity_corrected = _limit_temporal_steps(
            candidate, original_valid[side], frame_ids, max_step_m,
        )
        corrected[side] |= continuity_corrected
        candidate, boundary_tail_cleared = _remove_boundary_tail_branches(
            candidate, original_valid[side], frame_ids, max_step_m,
        )
        corrected[side] |= boundary_tail_cleared
        boundary_tail_cleared_count += int(np.count_nonzero(boundary_tail_cleared))
        candidate, excessive_jump_cleared = _invalidate_excessive_temporal_jumps(
            candidate, original_valid[side], frame_ids,
            max_jump_m=float(excessive_jump_threshold_m),
        )
        corrected[side] |= excessive_jump_cleared
        excessive_jump_cleared_count += int(np.count_nonzero(excessive_jump_cleared))

        # Ray/reprojection checks can remove joints after the initial
        # conditioning pass.  Re-check the *final* candidate as a hand-level
        # visibility unit; otherwise a 6--17-joint partial skeleton survives
        # and its articulation changes abruptly on the next frame.  This
        # guard runs after every temporal/anatomical pass so no later stage
        # can resurrect that partial hand.
        post_temporal_sparse = np.zeros(len(hands), dtype=bool)
        for index, hand in enumerate(hands):
            if hand is None or sparse_mask[index]:
                continue
            final_count = int(np.isfinite(candidate[index]).all(axis=1).sum())
            if final_count >= max(int(min_hand_joints), 0):
                continue
            post_temporal_sparse[index] = True
            if final_count > 0:
                post_temporal_sparse_hand_cleared_count += 1
                post_temporal_sparse_joint_cleared_count += final_count
                candidate[index] = np.nan
            hand["post_temporal_sparse_visibility_guard"] = {
                "applied": True,
                "minimum_finite_joints": int(min_hand_joints),
                "finite_joint_count_before": final_count,
            }

        for index, hand in enumerate(hands):
            if hand is None:
                continue
            hand["joints_base_m"] = candidate[index].tolist()
            if sparse_mask[index]:
                hand["sparse_visibility_guard"] = {
                    "applied": True,
                    "minimum_finite_joints": int(min_hand_joints),
                    "finite_joint_count_before": int(raw_valid[index].sum()),
                }
            if (side, int(frame_ids[index])) in view_jump_events:
                hand["temporal_view_guard"] = {
                    "applied": True,
                    "cameras": view_jump_events[(side, int(frame_ids[index]))],
                    "wrist_jump_threshold_px": float(view_wrist_jump_px),
                }
            changed = int(np.count_nonzero(
                original_valid[side, index]
                & np.isfinite(candidate[index]).all(axis=1)
                & (
                    np.linalg.norm(
                        candidate[index] - points[index], axis=1
                    ) > 0.002
                )
            ))
            hand["joint_temporal_guard"] = {
                "applied": bool(changed),
                "algorithm": ALGORITHM,
                "corrected_joint_count": changed,
                "preserved_missing_joint_count": int(
                    np.count_nonzero(~original_valid[side, index])
                ),
            }

    return {
        "enabled": True,
        "algorithm": ALGORITHM,
        "parameters": {
            "ray_prior_weight": float(ray_prior_weight),
            "ray_blend": float(ray_blend),
            "max_step_m": float(max_step_m),
            "max_radius_m": float(max_radius_m),
            "max_missing_gap": int(max_missing_gap),
            "local_radius": int(local_radius),
            "local_outlier_threshold_m": float(local_outlier_threshold_m),
            "local_candidate_ratio": float(local_candidate_ratio),
            "smoothing_sigma": float(smoothing_sigma),
            "bone_iterations": int(bone_iterations),
            "bone_strength": float(bone_strength),
            "ray_conditioning_adaptation": True,
            "ray_conditioning_soft_threshold": RAY_CONDITIONING_SOFT_THRESHOLD,
            "ray_conditioning_saturation": RAY_CONDITIONING_SATURATION,
            "ray_conditioning_hard_threshold": RAY_CONDITIONING_HARD_THRESHOLD,
            "ray_angle_weighting": True,
            "ray_angle_reference_degrees": RAY_ANGLE_REFERENCE_DEGREES,
            "ray_angle_weight_floor": RAY_ANGLE_WEIGHT_FLOOR,
            "sam3_ray_quality_weighting": True,
            "isolated_step_spike_guard": True,
            # These are anomaly thresholds, not hard output caps.  The final
            # pass repairs only short branch runs with trusted endpoints and
            # leaves genuinely fast motion unchanged.
            "final_continuity_branch_repair_threshold_m": float(max_step_m),
            "final_continuity_hard_cap": False,
            "gap_boundary_anchor": True,
            "gap_boundary_max_missing_gap": int(max_missing_gap),
            "min_hand_joints": int(min_hand_joints),
            "excessive_jump_threshold_m": float(excessive_jump_threshold_m),
            "view_wrist_jump_px": float(view_wrist_jump_px),
            "view_expanded_box_ratio": float(view_expanded_box_ratio),
            "view_expanded_box_wrist_px": float(view_expanded_box_wrist_px),
        },
        "corrected_joint_count": int(np.count_nonzero(corrected)),
        "corrected_hand_count": int(np.count_nonzero(np.any(corrected, axis=2))),
        "direction_track_count": int(direction_count),
        "preserved_missing_joint_count": int(np.count_nonzero(~original_valid)),
        "gap_boundary_sample_count": int(np.count_nonzero(gap_boundary)),
        "boundary_tail_cleared_joint_count": int(boundary_tail_cleared_count),
        "excessive_jump_cleared_joint_count": int(excessive_jump_cleared_count),
        "excessive_jump_threshold_m": float(excessive_jump_threshold_m),
        "sparse_hand_cleared_count": int(sparse_hand_cleared_count),
        "sparse_joint_cleared_count": int(sparse_joint_cleared_count),
        "post_temporal_sparse_hand_cleared_count": int(
            post_temporal_sparse_hand_cleared_count
        ),
        "post_temporal_sparse_joint_cleared_count": int(
            post_temporal_sparse_joint_cleared_count
        ),
        "view_jump_cleared_hand_count": int(view_jump_cleared_count),
        "view_jump_cleared_joint_count": int(view_jump_cleared_joint_count),
        "view_jump_event_count": int(len(view_jump_events)),
        "ray_geometry": {
            "sample_count": len(ray_condition_values),
            "condition_number_p50": (
                float(np.percentile(ray_condition_values, 50))
                if ray_condition_values else None
            ),
            "condition_number_p95": (
                float(np.percentile(ray_condition_values, 95))
                if ray_condition_values else None
            ),
            "condition_number_p99": (
                float(np.percentile(ray_condition_values, 99))
                if ray_condition_values else None
            ),
            "condition_number_max": (
                float(np.max(ray_condition_values))
                if ray_condition_values else None
            ),
            "condition_ge_100": int(sum(value >= 100.0 for value in ray_condition_values)),
            "condition_ge_600": int(sum(value >= 600.0 for value in ray_condition_values)),
            "condition_ge_1000": int(sum(value >= 1000.0 for value in ray_condition_values)),
            "min_ray_angle_p05_degrees": (
                float(np.percentile(ray_min_angle_values, 5))
                if ray_min_angle_values else None
            ),
            "min_ray_angle_p50_degrees": (
                float(np.percentile(ray_min_angle_values, 50))
                if ray_min_angle_values else None
            ),
            "min_ray_angle_p95_degrees": (
                float(np.percentile(ray_min_angle_values, 95))
                if ray_min_angle_values else None
            ),
            "ray_count_distribution": ray_count_distribution,
            "sam3_downweighted_ray_count": int(sam3_downweighted_ray_count),
        },
    }
