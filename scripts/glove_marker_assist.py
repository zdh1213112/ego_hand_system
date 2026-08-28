#!/usr/bin/env python3
"""Use visible reflective glove markers as conservative WiLoR image evidence.

The marker detector operates only on the RGB image. It never reads NOKOV
files. Marker associations are constrained by detector ownership and finger
topology. Weak associations are retained as hypothesis-selection evidence;
only broad, low-residual support is allowed to move the WiLoR skeleton.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


FINGER_JOINT_INDICES = np.arange(1, 21, dtype=np.int32)
HAND_EDGES = tuple(
    edge
    for finger in range(5)
    for edge in (
        (0, 1 + 4 * finger),
        (1 + 4 * finger, 2 + 4 * finger),
        (2 + 4 * finger, 3 + 4 * finger),
        (3 + 4 * finger, 4 + 4 * finger),
    )
)


class MarkerAssistError(RuntimeError):
    """Association failure with a stable code for aggregate diagnostics."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class MarkerAssistConfig:
    saturation_max: int = 100
    value_min: int = 160
    min_blob_area: int = 3
    max_blob_area: int = 180
    min_blob_size: int = 2
    max_blob_size: int = 25
    max_aspect_ratio: float = 3.0
    min_fill_ratio: float = 0.20
    min_circularity: float = 0.20
    search_padding_px: float = 45.0
    bbox_padding_px: float = 12.0
    seed_distance_px: float = 35.0
    match_distance_px: float = 13.0
    max_shift_px: float = 20.0
    min_matches: int = 3
    min_finger_groups: int = 2
    global_min_matches: int = 8
    global_min_finger_groups: int = 4
    max_global_residual_median_px: float = 7.5
    max_global_residual_p95_px: float = 12.0
    max_applied_shift_px: float = 10.0
    finger_ownership_margin_px: float = 2.0
    min_same_finger_direction_cosine: float = -0.10
    marker_blend: float = 0.15
    max_local_adjustment_px: float = 3.0
    max_bone_length_change_ratio: float = 0.25
    max_bone_length_change_p95_ratio: float = 0.12

    def validate(self) -> None:
        if not (0 <= self.saturation_max <= 255 and 0 <= self.value_min <= 255):
            raise ValueError("HSV thresholds must be in [0, 255]")
        if self.min_blob_area < 1 or self.max_blob_area < self.min_blob_area:
            raise ValueError("invalid blob area limits")
        if self.min_blob_size < 1 or self.max_blob_size < self.min_blob_size:
            raise ValueError("invalid blob size limits")
        if self.max_aspect_ratio < 1.0:
            raise ValueError("max-aspect-ratio must be at least 1")
        if not (0.0 <= self.min_fill_ratio <= 1.0):
            raise ValueError("min-fill-ratio must be in [0, 1]")
        if not (0.0 <= self.min_circularity <= 1.0):
            raise ValueError("min-circularity must be in [0, 1]")
        positive_distances = (
            self.search_padding_px,
            self.bbox_padding_px,
            self.seed_distance_px,
            self.match_distance_px,
            self.max_shift_px,
            self.max_global_residual_median_px,
            self.max_global_residual_p95_px,
            self.max_applied_shift_px,
            self.max_local_adjustment_px,
        )
        if min(positive_distances) <= 0:
            raise ValueError("marker search and correction limits must be positive")
        if self.finger_ownership_margin_px < 0:
            raise ValueError("finger-ownership-margin must be non-negative")
        if not (1 <= self.min_matches <= 20):
            raise ValueError("min-matches must be in [1, 20]")
        if not (1 <= self.min_finger_groups <= 5):
            raise ValueError("min-finger-groups must be in [1, 5]")
        if not (self.min_matches <= self.global_min_matches <= 20):
            raise ValueError("global-min-matches must be in [min-matches, 20]")
        if not (self.min_finger_groups <= self.global_min_finger_groups <= 5):
            raise ValueError(
                "global-min-finger-groups must be in [min-finger-groups, 5]"
            )
        if not (-1.0 <= self.min_same_finger_direction_cosine <= 1.0):
            raise ValueError("same-finger direction cosine must be in [-1, 1]")
        if not (0.0 <= self.marker_blend <= 1.0):
            raise ValueError("marker-blend must be in [0, 1]")
        if not (0.0 <= self.max_bone_length_change_ratio <= 1.0):
            raise ValueError("max bone-length change must be in [0, 1]")
        if not (0.0 <= self.max_bone_length_change_p95_ratio <= 1.0):
            raise ValueError("bone-length P95 change must be in [0, 1]")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BrightBlobs:
    centers: np.ndarray
    areas: np.ndarray
    circularities: np.ndarray
    source_indices: np.ndarray | None = None


@dataclass(frozen=True)
class MarkerAssociation:
    joint_indices: np.ndarray
    component_indices: np.ndarray
    observations: np.ndarray
    shift_px: np.ndarray
    residuals_px: np.ndarray
    finger_group_count: int
    topology_rejected_count: int


def _component_circularity(component_mask: np.ndarray, area: float) -> float:
    contours, _hierarchy = cv2.findContours(
        component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    perimeter = sum(cv2.arcLength(contour, True) for contour in contours)
    if perimeter <= 0.0:
        return 0.0
    return float(4.0 * np.pi * area / (perimeter * perimeter))


def detect_bright_blobs(
    image_bgr: np.ndarray, config: MarkerAssistConfig,
) -> BrightBlobs:
    """Detect compact, approximately round, low-saturation bright components."""
    config.validate()
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("marker detection expects one BGR image")
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    mask = (
        (hsv[:, :, 1] < config.saturation_max)
        & (hsv[:, :, 2] > config.value_min)
    ).astype(np.uint8)
    count, labels, stats, centers = cv2.connectedComponentsWithStats(mask, 8)
    output_centers: list[np.ndarray] = []
    output_areas: list[float] = []
    output_circularities: list[float] = []
    for index in range(1, count):
        x, y, width, height, area = stats[index]
        if not (config.min_blob_area <= area <= config.max_blob_area):
            continue
        if not (
            config.min_blob_size <= width <= config.max_blob_size
            and config.min_blob_size <= height <= config.max_blob_size
        ):
            continue
        aspect = max(width, height) / max(1, min(width, height))
        fill_ratio = float(area) / float(width * height)
        if aspect > config.max_aspect_ratio or fill_ratio < config.min_fill_ratio:
            continue
        local_mask = (labels[y:y + height, x:x + width] == index).astype(np.uint8)
        circularity = _component_circularity(local_mask, float(area))
        if circularity < config.min_circularity:
            continue
        output_centers.append(centers[index])
        output_areas.append(float(area))
        output_circularities.append(circularity)
    if not output_centers:
        return BrightBlobs(
            centers=np.empty((0, 2), dtype=np.float64),
            areas=np.empty(0, dtype=np.float64),
            circularities=np.empty(0, dtype=np.float64),
            source_indices=np.empty(0, dtype=np.int32),
        )
    output = np.asarray(output_centers, dtype=np.float64)
    return BrightBlobs(
        centers=output,
        areas=np.asarray(output_areas, dtype=np.float64),
        circularities=np.asarray(output_circularities, dtype=np.float64),
        source_indices=np.arange(len(output), dtype=np.int32),
    )


def _near_hand_components(
    predicted: np.ndarray, components: np.ndarray, padding_px: float,
) -> tuple[np.ndarray, np.ndarray]:
    finite = np.isfinite(predicted).all(axis=1)
    if not np.any(finite) or len(components) == 0:
        return np.empty((0, 2), dtype=np.float64), np.empty(0, dtype=np.int32)
    values = predicted[finite]
    lower = np.min(values, axis=0) - padding_px
    upper = np.max(values, axis=0) + padding_px
    use = np.all((components >= lower) & (components <= upper), axis=1)
    indices = np.flatnonzero(use).astype(np.int32)
    return components[indices], indices


def _finger_compatible_mask(
    shifted: np.ndarray,
    source_joint_indices: np.ndarray,
    components: np.ndarray,
    margin_px: float,
) -> np.ndarray:
    distances = np.linalg.norm(
        shifted[:, None, :] - components[None, :, :], axis=2
    )
    joint_groups = (source_joint_indices - 1) // 4
    group_distances = np.full((5, len(components)), np.inf, dtype=np.float64)
    for group in range(5):
        use = joint_groups == group
        if np.any(use):
            group_distances[group] = np.min(distances[use], axis=0)
    nearest_group_distance = np.min(group_distances, axis=0)
    return np.asarray([
        group_distances[int(group)] <= nearest_group_distance + margin_px
        for group in joint_groups
    ])


def _hungarian_pairs(
    shifted: np.ndarray,
    source_joint_indices: np.ndarray,
    components: np.ndarray,
    config: MarkerAssistConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    distances = np.linalg.norm(
        shifted[:, None, :] - components[None, :, :], axis=2
    )
    compatible = _finger_compatible_mask(
        shifted, source_joint_indices, components, config.finger_ownership_margin_px
    )
    valid = (distances <= config.match_distance_px) & compatible
    assignment_cost = np.where(valid, distances, 1e6)
    dummy = np.full(
        (len(shifted), len(shifted)), config.match_distance_px + 1e-6,
        dtype=np.float64,
    )
    rows, columns = linear_sum_assignment(np.hstack((assignment_cost, dummy)))
    use = columns < len(components)
    rows = rows[use]
    columns = columns[use]
    if len(rows):
        keep = valid[rows, columns]
        rows = rows[keep]
        columns = columns[keep]
    return rows.astype(np.int32), columns.astype(np.int32), distances


def _remove_same_finger_reversals(
    shifted: np.ndarray,
    source_joint_indices: np.ndarray,
    components: np.ndarray,
    rows: np.ndarray,
    columns: np.ndarray,
    distances: np.ndarray,
    minimum_cosine: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    keep = np.ones(len(rows), dtype=bool)
    rejected = 0
    groups = (source_joint_indices[rows] - 1) // 4
    positions = (source_joint_indices[rows] - 1) % 4
    for group in range(5):
        members = np.flatnonzero(groups == group)
        for first_offset, first_value in enumerate(members):
            if not keep[first_value]:
                continue
            for second_value in members[first_offset + 1:]:
                if not keep[second_value] or positions[first_value] == positions[second_value]:
                    continue
                first, second = int(first_value), int(second_value)
                if positions[first] > positions[second]:
                    first, second = second, first
                predicted_delta = shifted[rows[second]] - shifted[rows[first]]
                observed_delta = components[columns[second]] - components[columns[first]]
                denominator = (
                    float(np.linalg.norm(predicted_delta))
                    * float(np.linalg.norm(observed_delta))
                )
                if denominator < 1e-6:
                    continue
                cosine = float(np.dot(predicted_delta, observed_delta) / denominator)
                if cosine >= minimum_cosine:
                    continue
                first_residual = distances[rows[first], columns[first]]
                second_residual = distances[rows[second], columns[second]]
                remove = first if first_residual >= second_residual else second
                keep[remove] = False
                rejected += 1
    return rows[keep], columns[keep], rejected


def _candidate_shifts(
    predicted: np.ndarray,
    nearby: np.ndarray,
    config: MarkerAssistConfig,
) -> list[np.ndarray]:
    distances = np.linalg.norm(
        predicted[:, None, :] - nearby[None, :, :], axis=2
    )
    nearest = np.argmin(distances, axis=1)
    nearest_distance = distances[np.arange(len(predicted)), nearest]
    use = nearest_distance <= config.seed_distance_px
    if int(np.count_nonzero(use)) < config.min_matches:
        raise MarkerAssistError(
            "too_few_coarse_matches",
            f"only {int(np.count_nonzero(use))} coarse marker matches; "
            f"need {config.min_matches}",
        )
    offsets = nearby[nearest[use]] - predicted[use]
    values = [np.median(offsets, axis=0), *offsets]
    candidates: list[np.ndarray] = []
    seen: set[tuple[int, int]] = set()
    for value in values:
        if float(np.linalg.norm(value)) > config.max_shift_px:
            continue
        key = tuple(np.rint(value).astype(np.int32).tolist())
        if key not in seen:
            candidates.append(np.asarray(value, dtype=np.float64))
            seen.add(key)
    if not candidates:
        smallest = float(np.min(np.linalg.norm(offsets, axis=1)))
        raise MarkerAssistError(
            "coarse_shift_too_large",
            f"smallest coarse marker shift {smallest:.2f}px exceeds "
            f"{config.max_shift_px:.2f}px",
        )
    return candidates


def associate_marker_blobs(
    joints_2d: np.ndarray,
    component_centers: np.ndarray,
    config: MarkerAssistConfig,
    component_source_indices: np.ndarray | None = None,
) -> MarkerAssociation:
    """Estimate a robust translation and topology-constrained assignment."""
    config.validate()
    joints = np.asarray(joints_2d, dtype=np.float64)
    components = np.asarray(component_centers, dtype=np.float64)
    if joints.shape != (21, 2):
        raise ValueError(f"expected WiLoR joints with shape (21, 2), got {joints.shape}")
    predicted = joints[FINGER_JOINT_INDICES]
    finite = np.isfinite(predicted).all(axis=1)
    if int(np.count_nonzero(finite)) < config.min_matches:
        raise MarkerAssistError(
            "too_few_finite_joints", "too few finite WiLoR finger joints"
        )
    predicted = predicted[finite]
    source_joint_indices = FINGER_JOINT_INDICES[finite]
    nearby, nearby_indices = _near_hand_components(
        predicted, components, config.search_padding_px
    )
    if len(nearby) == 0:
        raise MarkerAssistError(
            "no_nearby_blobs", "no bright glove-marker blobs near the WiLoR hand"
        )

    best: tuple[
        tuple[float, ...], np.ndarray, np.ndarray, np.ndarray, np.ndarray
    ] | None = None
    for shift in _candidate_shifts(predicted, nearby, config):
        rows, columns, distances = _hungarian_pairs(
            predicted + shift, source_joint_indices, nearby, config
        )
        groups = len(set(((source_joint_indices[rows] - 1) // 4).tolist()))
        median = (
            float(np.median(distances[rows, columns])) if len(rows) else float("inf")
        )
        score = (float(len(rows)), float(groups), -median, -float(np.linalg.norm(shift)))
        if best is None or score > best[0]:
            best = (score, shift, rows, columns, distances)
    assert best is not None
    _score, shift, rows, columns, distances = best
    if len(rows) >= config.min_matches:
        refined_shift = np.median(nearby[columns] - predicted[rows], axis=0)
        if float(np.linalg.norm(refined_shift)) <= config.max_shift_px:
            shift = refined_shift
            rows, columns, distances = _hungarian_pairs(
                predicted + shift, source_joint_indices, nearby, config
            )
    rows, columns, topology_rejected = _remove_same_finger_reversals(
        predicted + shift,
        source_joint_indices,
        nearby,
        rows,
        columns,
        distances,
        config.min_same_finger_direction_cosine,
    )
    if len(rows) < config.min_matches:
        raise MarkerAssistError(
            "too_few_topology_matches",
            f"only {len(rows)} topology-valid one-to-one marker matches; "
            f"need {config.min_matches}",
        )
    joint_indices = source_joint_indices[rows]
    group_count = len(set(((joint_indices - 1) // 4).tolist()))
    if group_count < config.min_finger_groups:
        raise MarkerAssistError(
            "too_few_finger_groups",
            f"marker matches cover only {group_count} finger groups; "
            f"need {config.min_finger_groups}",
        )
    source_ids = (
        np.arange(len(components), dtype=np.int32)
        if component_source_indices is None
        else np.asarray(component_source_indices, dtype=np.int32)
    )
    observations = nearby[columns]
    residuals = distances[rows, columns]
    return MarkerAssociation(
        joint_indices=joint_indices,
        component_indices=source_ids[nearby_indices[columns]],
        observations=observations,
        shift_px=np.asarray(shift, dtype=np.float64),
        residuals_px=np.asarray(residuals, dtype=np.float64),
        finger_group_count=group_count,
        topology_rejected_count=topology_rejected,
    )


def _bounded_vector(vector: np.ndarray, maximum_norm: float) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= maximum_norm or norm < 1e-9:
        return vector
    return vector * (maximum_norm / norm)


def _bone_length_change(
    original: np.ndarray, adjusted: np.ndarray,
) -> tuple[float, float]:
    changes = []
    for start, end in HAND_EDGES:
        before = float(np.linalg.norm(original[end] - original[start]))
        after = float(np.linalg.norm(adjusted[end] - adjusted[start]))
        if before > 1e-6 and np.isfinite(before) and np.isfinite(after):
            changes.append(abs(after / before - 1.0))
    if not changes:
        return float("inf"), float("inf")
    return float(np.max(changes)), float(np.percentile(changes, 95))


def _association_metadata(
    association: MarkerAssociation,
    original: np.ndarray,
) -> dict[str, Any]:
    raw_residuals = np.linalg.norm(
        original[association.joint_indices] - association.observations, axis=1
    )
    return {
        "matched_marker_count": int(len(association.joint_indices)),
        "matched_joint_indices": association.joint_indices.tolist(),
        "matched_component_indices": association.component_indices.tolist(),
        "matched_blob_centers": association.observations.tolist(),
        "coarse_shift_px": association.shift_px.tolist(),
        "shift_norm_px": float(np.linalg.norm(association.shift_px)),
        "match_residual_median_px": float(np.median(association.residuals_px)),
        "match_residual_p95_px": float(np.percentile(association.residuals_px, 95)),
        "raw_marker_residual_median_px": float(np.median(raw_residuals)),
        "finger_group_count": association.finger_group_count,
        "topology_rejected_match_count": association.topology_rejected_count,
    }


def assist_wilor_hand(
    hand: dict[str, Any],
    blobs: BrightBlobs,
    config: MarkerAssistConfig,
) -> dict[str, Any]:
    """Refine a skeleton only when marker support is broad and low-residual."""
    output = dict(hand)
    original = np.asarray(hand["joints_2d"], dtype=np.float64)
    output["wilor_joints_2d"] = original.tolist()
    try:
        association = associate_marker_blobs(
            original, blobs.centers, config, blobs.source_indices
        )
    except MarkerAssistError as error:
        output["marker_assist"] = {
            "applied": False,
            "evidence_only": False,
            "reason_code": error.code,
            "reason": str(error),
            "candidate_blob_count": int(len(blobs.centers)),
        }
        return output
    except ValueError as error:
        output["marker_assist"] = {
            "applied": False,
            "evidence_only": False,
            "reason_code": "invalid_input",
            "reason": str(error),
            "candidate_blob_count": int(len(blobs.centers)),
        }
        return output

    metadata = _association_metadata(association, original)
    global_reasons = []
    if metadata["matched_marker_count"] < config.global_min_matches:
        global_reasons.append("too_few_global_matches")
    if metadata["finger_group_count"] < config.global_min_finger_groups:
        global_reasons.append("too_few_global_finger_groups")
    if metadata["match_residual_median_px"] > config.max_global_residual_median_px:
        global_reasons.append("global_residual_median_too_large")
    if metadata["match_residual_p95_px"] > config.max_global_residual_p95_px:
        global_reasons.append("global_residual_p95_too_large")
    if global_reasons:
        output["marker_assist"] = {
            "applied": False,
            "evidence_only": True,
            "reason_code": global_reasons[0],
            "reason": "; ".join(global_reasons),
            "candidate_blob_count": int(len(blobs.centers)),
            **metadata,
        }
        return output

    applied_shift = _bounded_vector(
        association.shift_px, config.max_applied_shift_px
    )
    adjusted = original + applied_shift
    local_adjustments = []
    for joint_index, observation in zip(
        association.joint_indices, association.observations
    ):
        joint = int(joint_index)
        local = config.marker_blend * (observation - adjusted[joint])
        local = _bounded_vector(local, config.max_local_adjustment_px)
        adjusted[joint] += local
        local_adjustments.append(float(np.linalg.norm(local)))
    bone_max, bone_p95 = _bone_length_change(original, adjusted)
    local_adjustment_applied = True
    if (
        bone_max > config.max_bone_length_change_ratio
        or bone_p95 > config.max_bone_length_change_p95_ratio
    ):
        adjusted = original + applied_shift
        local_adjustments = []
        bone_max, bone_p95 = _bone_length_change(original, adjusted)
        local_adjustment_applied = False

    assisted_residuals = np.linalg.norm(
        adjusted[association.joint_indices] - association.observations, axis=1
    )
    output["joints_2d"] = adjusted.tolist()
    output["marker_assist"] = {
        "applied": True,
        "evidence_only": False,
        "algorithm": "hsv_owned_topology_hungarian_conservative_v2",
        "reason_code": "correction_applied",
        "candidate_blob_count": int(len(blobs.centers)),
        **metadata,
        "applied_shift_px": applied_shift.tolist(),
        "applied_shift_norm_px": float(np.linalg.norm(applied_shift)),
        "shift_was_clipped": bool(
            np.linalg.norm(applied_shift - association.shift_px) > 1e-6
        ),
        "assisted_marker_residual_median_px": float(
            np.median(assisted_residuals)
        ),
        "local_adjustment_applied": local_adjustment_applied,
        "local_adjustment_norm_max_px": max(local_adjustments, default=0.0),
        "bone_length_change_max_ratio": bone_max,
        "bone_length_change_p95_ratio": bone_p95,
        "marker_blend": config.marker_blend,
    }
    return output


def _subset_blobs(blobs: BrightBlobs, indices: np.ndarray) -> BrightBlobs:
    source = (
        np.arange(len(blobs.centers), dtype=np.int32)
        if blobs.source_indices is None
        else blobs.source_indices
    )
    return BrightBlobs(
        centers=blobs.centers[indices],
        areas=blobs.areas[indices],
        circularities=blobs.circularities[indices],
        source_indices=np.asarray(source, dtype=np.int32)[indices],
    )


def _owned_blob_indices(
    hands: list[dict[str, Any]],
    centers: np.ndarray,
    padding_px: float,
) -> dict[int, np.ndarray]:
    detections: dict[int, np.ndarray] = {}
    for ordinal, hand in enumerate(hands):
        detection = int(hand.get("detection_index", ordinal))
        bbox = hand.get("bbox_xyxy")
        if bbox is not None and detection not in detections:
            detections[detection] = np.asarray(bbox, dtype=np.float64)
    if not detections:
        return {}
    owners: dict[int, list[int]] = {detection: [] for detection in detections}
    for component_index, center in enumerate(centers):
        candidates = []
        for detection, bbox in detections.items():
            lower = bbox[:2] - padding_px
            upper = bbox[2:] + padding_px
            if not np.all((center >= lower) & (center <= upper)):
                continue
            bbox_center = 0.5 * (bbox[:2] + bbox[2:])
            half_size = np.maximum(0.5 * (bbox[2:] - bbox[:2]), 1.0)
            score = float(np.linalg.norm((center - bbox_center) / half_size))
            candidates.append((score, detection))
        if candidates:
            owners[min(candidates)[1]].append(component_index)
    return {
        detection: np.asarray(indices, dtype=np.int32)
        for detection, indices in owners.items()
    }


def assist_wilor_hypotheses(
    image_bgr: np.ndarray,
    hands: list[dict[str, Any]],
    config: MarkerAssistConfig,
) -> tuple[list[dict[str, Any]], BrightBlobs]:
    blobs = detect_bright_blobs(image_bgr, config)
    owned = _owned_blob_indices(hands, blobs.centers, config.bbox_padding_px)
    assisted = []
    for ordinal, hand in enumerate(hands):
        detection = int(hand.get("detection_index", ordinal))
        if owned:
            hand_blobs = _subset_blobs(
                blobs, owned.get(detection, np.empty(0, dtype=np.int32))
            )
        else:
            hand_blobs = blobs
        assisted.append(assist_wilor_hand(hand, hand_blobs, config))
    return assisted, blobs
