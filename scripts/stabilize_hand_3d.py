#!/usr/bin/env python3
"""Prepare stereo hand landmarks for MANO fitting with explicit missing-data states."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np


JOINT_NAMES = (
    "wrist", "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
    "index_mcp", "index_pip", "index_dip", "index_tip",
    "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
    "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip",
)

# A MANO-compatible kinematic tree over MediaPipe's 21 semantic joints.
SKELETON_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
)

PALM_FRAME_JOINTS = np.asarray((0, 5, 9, 13, 17), dtype=np.int32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stabilize EGO stereo 3D hand landmarks before MANO fitting."
    )
    parser.add_argument("--input", required=True, type=Path, help="stereo_landmarks_3d.csv")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-gap", type=int, default=5, help="maximum interpolated gap in frames")
    parser.add_argument("--outlier-window", type=int, default=4, help="temporal outlier radius")
    parser.add_argument(
        "--outlier-distance-m", type=float, default=0.12,
        help="maximum distance from the local temporal median",
    )
    parser.add_argument(
        "--max-hand-radius-m", type=float, default=0.22,
        help="maximum joint distance from the per-frame hand median",
    )
    parser.add_argument(
        "--pixel-outlier-window", type=int, default=4,
        help="temporal radius used to reject isolated 2D landmark-shape jumps",
    )
    parser.add_argument(
        "--pixel-outlier-distance", type=float, default=0.45,
        help="maximum palm-normalized 2D landmark deviation from nearby frames",
    )
    parser.add_argument(
        "--pixel-scale-ratio", type=float, default=1.8,
        help="maximum isolated palm-width ratio relative to nearby frames",
    )
    parser.add_argument(
        "--pixel-smoothing-radius", type=int, default=3,
        help="zero-phase radius for finger-shape smoothing in the moving palm frame",
    )
    parser.add_argument(
        "--pixel-smoothing-strength", type=float, default=0.75,
        help="blend toward the locally smoothed finger shape; palm pose stays per-frame",
    )
    parser.add_argument(
        "--local-shape-smoothing-radius", type=int, default=3,
        help="zero-phase radius for 3D finger articulation in the moving palm frame",
    )
    parser.add_argument(
        "--local-shape-smoothing-strength", type=float, default=0.70,
        help="3D finger-shape smoothing blend; palm anchors stay unchanged",
    )
    parser.add_argument(
        "--bone-outlier-absolute-m", type=float, default=0.05,
        help="minimum gross bone-length residual used for observation rejection",
    )
    parser.add_argument(
        "--bone-outlier-relative", type=float, default=0.80,
        help="gross bone-length residual relative to the stable length",
    )
    parser.add_argument("--window-radius", type=int, default=2, help="temporal smoothing radius")
    parser.add_argument("--bone-iterations", type=int, default=8)
    parser.add_argument("--bone-strength", type=float, default=0.65)
    parser.add_argument(
        "--max-observation-adjustment-m", type=float, default=0.08,
        help="reject an observation if stabilization must move it farther than this",
    )
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--no-video", action="store_true")
    return parser.parse_args()


def confidence_from_row(row: dict) -> float:
    epipolar = float(row["epipolar_error_px"])
    reprojection = float(row["reprojection_error_px"])
    disparity = float(row.get("disparity_px", 24.0))
    handedness = min(float(row["left_handedness_score"]), float(row["right_handedness_score"]))
    geometry = np.exp(-0.5 * (epipolar / 8.0) ** 2) * np.exp(-0.5 * (reprojection / 4.0) ** 2)
    disparity_quality = float(np.clip((disparity - 2.0) / 24.0, 0.10, 1.0))
    refinement_factor = 1.0
    if row.get("refinement_attempted", "0") == "1":
        if row.get("refinement_used", "0") == "1":
            refinement_factor = 0.55 + 0.45 * float(row.get("refinement_quality", 0.0))
        else:
            refinement_factor = 0.38 if int(row["landmark_index"]) in (4, 8, 12, 16, 20) else 0.50
    return float(np.clip(
        geometry * disparity_quality * (0.5 + 0.5 * handedness) * refinement_factor,
        0.02, 1.0,
    ))


def load_stereo_csv(path: Path):
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = list(csv.DictReader(path.open("r", encoding="utf-8", newline="")))
    if not rows:
        raise RuntimeError("input CSV contains no observations")
    required = {
        "pair_index", "track_id", "landmark_index", "valid_3d",
        "x_left_camera_m", "y_left_camera_m", "z_left_camera_m",
        "x_rectified_m", "y_rectified_m", "z_rectified_m",
        "disparity_px", "epipolar_error_px", "reprojection_error_px",
    }
    missing = required - set(rows[0])
    if missing:
        raise RuntimeError(f"missing CSV fields: {sorted(missing)}")

    pair_count = max(int(row["pair_index"]) for row in rows) + 1
    track_ids = sorted({int(row["track_id"]) for row in rows})
    track_to_slot = {track_id: slot for slot, track_id in enumerate(track_ids)}
    shape = (len(track_ids), pair_count, 21)
    raw = np.full(shape + (3,), np.nan, dtype=np.float64)
    raw_rectified = np.full(shape + (3,), np.nan, dtype=np.float64)
    observed = np.zeros(shape, dtype=bool)
    confidence = np.zeros(shape, dtype=np.float64)
    left_pixels = np.full(shape + (2,), np.nan, dtype=np.float64)
    right_pixels = np.full(shape + (2,), np.nan, dtype=np.float64)
    handedness_votes = {track_id: {} for track_id in track_ids}
    pair_metadata: dict[int, dict] = {}

    for row in rows:
        pair = int(row["pair_index"])
        track_id = int(row["track_id"])
        slot = track_to_slot[track_id]
        joint = int(row["landmark_index"])
        pair_metadata.setdefault(pair, {
            "left_index": int(row["left_index"]),
            "right_index": int(row["right_index"]),
        })
        left_pixels[slot, pair, joint] = [
            float(row["left_x_rectified_px"]), float(row["left_y_rectified_px"])
        ]
        right_pixels[slot, pair, joint] = [
            float(row["right_x_rectified_px"]), float(row["right_y_rectified_px"])
        ]
        label = row["left_handedness"]
        handedness_votes[track_id][label] = handedness_votes[track_id].get(label, 0) + 1
        if row["valid_3d"] == "1":
            point = np.asarray([
                float(row["x_left_camera_m"]),
                float(row["y_left_camera_m"]),
                float(row["z_left_camera_m"]),
            ])
            if np.all(np.isfinite(point)):
                raw[slot, pair, joint] = point
                raw_rectified[slot, pair, joint] = np.asarray([
                    float(row["x_rectified_m"]),
                    float(row["y_rectified_m"]),
                    float(row["z_rectified_m"]),
                ])
                observed[slot, pair, joint] = True
                confidence[slot, pair, joint] = confidence_from_row(row)

    handedness = []
    for track_id in track_ids:
        votes = handedness_votes[track_id]
        handedness.append(max(votes, key=votes.get) if votes else "Unknown")
    return {
        "rows": rows,
        "track_ids": np.asarray(track_ids, dtype=np.int32),
        "handedness": np.asarray(handedness),
        "raw": raw,
        "raw_rectified": raw_rectified,
        "observed": observed,
        "confidence": confidence,
        "left_pixels": left_pixels,
        "right_pixels": right_pixels,
        "pair_metadata": pair_metadata,
    }


def load_rectified_camera_contract(input_path: Path, data: dict) -> dict:
    summary_path = input_path.parent / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"stereo summary with P1/P2 is missing: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    p1 = np.asarray(summary.get("p1"), dtype=np.float64)
    p2 = np.asarray(summary.get("p2"), dtype=np.float64)
    if p1.shape != (3, 4) or p2.shape != (3, 4):
        raise RuntimeError("stereo summary does not contain valid 3x4 P1/P2 matrices")

    mask = data["observed"]
    left = data["raw"][mask]
    rectified = data["raw_rectified"][mask]
    finite = np.isfinite(left).all(axis=1) & np.isfinite(rectified).all(axis=1)
    left = left[finite]
    rectified = rectified[finite]
    if len(left) < 20:
        raise RuntimeError("not enough 3D correspondences to recover rectification rotation")
    # Rectification is a rotation about the optical centre: rectified = R1 @ left.
    u, _, vt = np.linalg.svd(rectified.T @ left)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    residual = np.linalg.norm((rotation @ left.T).T - rectified, axis=1)
    if float(np.percentile(residual, 95)) > 1e-5:
        raise RuntimeError("left-to-rectified rotation recovery failed")
    return {
        "p1": p1,
        "p2": p2,
        "left_to_rectified_rotation": rotation,
        "residual_median_m": float(np.median(residual)),
        "residual_p95_m": float(np.percentile(residual, 95)),
        "rectified_size": np.asarray(summary.get("rectified_size", [1600, 1300]), dtype=np.int32),
    }


def active_ranges(observed: np.ndarray) -> list[tuple[int, int]]:
    ranges = []
    for track in range(observed.shape[0]):
        active = np.flatnonzero(np.any(observed[track], axis=1))
        if len(active):
            ranges.append((int(active[0]), int(active[-1])))
        else:
            ranges.append((0, -1))
    return ranges


def detect_temporal_pixel_outliers(
    pixels: np.ndarray,
    radius: int,
    normalized_distance: float,
    maximum_scale_ratio: float,
) -> np.ndarray:
    """Find isolated MediaPipe shape jumps after removing palm translation/scale/rotation."""
    if pixels.ndim != 4 or pixels.shape[2:] != (21, 2):
        raise ValueError(f"unexpected pixel landmark shape: {pixels.shape}")
    rejected = np.zeros(pixels.shape[:-1], dtype=bool)
    if radius <= 0:
        return rejected

    finite = np.isfinite(pixels).all(axis=-1)
    local = np.full_like(pixels, np.nan, dtype=np.float64)
    palm_scale = np.full(pixels.shape[:2], np.nan, dtype=np.float64)
    for track in range(pixels.shape[0]):
        for frame in range(pixels.shape[1]):
            points = pixels[track, frame]
            if not np.all(finite[track, frame, PALM_FRAME_JOINTS]):
                continue
            origin = np.median(points[PALM_FRAME_JOINTS], axis=0)
            palm_axis = points[5] - points[17]
            scale = float(np.linalg.norm(palm_axis))
            if scale < 5.0:
                continue
            axis_x = palm_axis / scale
            axis_y = np.asarray((-axis_x[1], axis_x[0]), dtype=np.float64)
            offsets = points - origin
            local[track, frame, :, 0] = offsets @ axis_x / scale
            local[track, frame, :, 1] = offsets @ axis_y / scale
            palm_scale[track, frame] = scale

    minimum_neighbours = max(4, radius + 1)
    for track in range(pixels.shape[0]):
        for frame in range(radius, pixels.shape[1] - radius):
            neighbour_frames = np.concatenate((
                np.arange(frame - radius, frame),
                np.arange(frame + 1, frame + radius + 1),
            ))
            neighbour_scales = palm_scale[track, neighbour_frames]
            neighbour_scales = neighbour_scales[np.isfinite(neighbour_scales)]
            current_scale = palm_scale[track, frame]
            if np.isfinite(current_scale) and len(neighbour_scales) >= minimum_neighbours:
                median_scale = float(np.median(neighbour_scales))
                scale_ratio = max(current_scale / median_scale, median_scale / current_scale)
                if scale_ratio > maximum_scale_ratio:
                    rejected[track, frame] = finite[track, frame]
                    continue

            for joint in range(21):
                current = local[track, frame, joint]
                if not np.all(np.isfinite(current)):
                    continue
                neighbours = local[track, neighbour_frames, joint]
                neighbours = neighbours[np.isfinite(neighbours).all(axis=1)]
                if len(neighbours) < minimum_neighbours:
                    continue
                median = np.median(neighbours, axis=0)
                coordinate_mad = 1.4826 * np.median(
                    np.abs(neighbours - median), axis=0
                )
                adaptive_distance = 5.0 * float(np.linalg.norm(coordinate_mad))
                threshold = max(normalized_distance, adaptive_distance)
                if np.linalg.norm(current - median) > threshold:
                    rejected[track, frame, joint] = True
    return rejected


def palm_normalized_pixel_coordinates(
    pixels: np.ndarray,
    valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Express image landmarks in each frame's own palm coordinate system."""
    if pixels.ndim != 4 or pixels.shape[2:] != (21, 2):
        raise ValueError(f"unexpected pixel landmark shape: {pixels.shape}")
    if valid.shape != pixels.shape[:-1]:
        raise ValueError("pixel validity shape disagrees with landmark shape")
    local = np.full_like(pixels, np.nan, dtype=np.float64)
    origins = np.full(pixels.shape[:2] + (2,), np.nan, dtype=np.float64)
    axes_x = np.full_like(origins, np.nan)
    axes_y = np.full_like(origins, np.nan)
    scales = np.full(pixels.shape[:2], np.nan, dtype=np.float64)
    frame_valid = np.zeros(pixels.shape[:2], dtype=bool)
    for track in range(pixels.shape[0]):
        for frame in range(pixels.shape[1]):
            if not np.all(valid[track, frame, PALM_FRAME_JOINTS]):
                continue
            points = pixels[track, frame]
            origin = np.median(points[PALM_FRAME_JOINTS], axis=0)
            palm_axis = points[5] - points[17]
            scale = float(np.linalg.norm(palm_axis))
            if not np.isfinite(scale) or scale < 5.0:
                continue
            axis_x = palm_axis / scale
            axis_y = np.asarray((-axis_x[1], axis_x[0]), dtype=np.float64)
            offsets = points - origin
            local[track, frame, :, 0] = offsets @ axis_x / scale
            local[track, frame, :, 1] = offsets @ axis_y / scale
            local[track, frame, ~valid[track, frame]] = np.nan
            origins[track, frame] = origin
            axes_x[track, frame] = axis_x
            axes_y[track, frame] = axis_y
            scales[track, frame] = scale
            frame_valid[track, frame] = True
    return local, origins, axes_x, axes_y, scales, frame_valid


def smooth_pixel_landmarks_in_palm_frame(
    pixels: np.ndarray,
    valid: np.ndarray,
    radius: int,
    strength: float,
) -> np.ndarray:
    """Smooth finger articulation while preserving current palm pose and placement.

    The temporal filter operates on palm-normalized coordinates and is symmetric,
    so camera-space hand motion is retained without introducing a causal lag. The
    current palm centre is preserved, while noisy palm scale/orientation and local
    landmark shape are smoothed together.
    """
    pixels = np.asarray(pixels, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    if radius <= 0 or strength <= 0.0:
        return pixels.copy()
    if not 0.0 <= strength <= 1.0:
        raise ValueError("pixel smoothing strength must be in [0, 1]")
    local, origins, axes_x, axes_y, scales, frame_valid = (
        palm_normalized_pixel_coordinates(pixels, valid)
    )
    filtered = pixels.copy()
    for track in range(pixels.shape[0]):
        for frame in range(pixels.shape[1]):
            if not frame_valid[track, frame]:
                continue
            lo = max(0, frame - radius)
            hi = min(pixels.shape[1], frame + radius + 1)
            candidate_frames = np.arange(lo, hi)
            palm_candidates = candidate_frames[frame_valid[track, candidate_frames]]
            palm_weights = radius + 1 - np.abs(palm_candidates - frame)
            mean_axis = np.average(
                axes_x[track, palm_candidates], axis=0, weights=palm_weights
            )
            mean_axis_length = float(np.linalg.norm(mean_axis))
            if mean_axis_length > 1e-8:
                mean_axis /= mean_axis_length
            else:
                mean_axis = axes_x[track, frame]
            blended_axis = (
                (1.0 - strength) * axes_x[track, frame] + strength * mean_axis
            )
            blended_axis /= max(float(np.linalg.norm(blended_axis)), 1e-8)
            blended_axis_y = np.asarray((-blended_axis[1], blended_axis[0]))
            median_log_scale = float(np.median(np.log(scales[track, palm_candidates])))
            blended_scale = float(np.exp(
                (1.0 - strength) * np.log(scales[track, frame])
                + strength * median_log_scale
            ))
            blended_local = local[track, frame].copy()
            for joint in range(21):
                if not valid[track, frame, joint]:
                    continue
                candidates = candidate_frames[
                    frame_valid[track, candidate_frames]
                    & valid[track, candidate_frames, joint]
                ]
                if len(candidates) < 2:
                    continue
                values = local[track, candidates, joint]
                median = np.median(values, axis=0)
                distances = np.linalg.norm(values - median, axis=1)
                distance_median = float(np.median(distances))
                distance_mad = 1.4826 * float(
                    np.median(np.abs(distances - distance_median))
                )
                robust_limit = max(0.10, distance_median + 3.5 * distance_mad)
                inliers = distances <= robust_limit
                if np.count_nonzero(inliers) >= 2:
                    candidates = candidates[inliers]
                    values = values[inliers]
                temporal_weights = radius + 1 - np.abs(candidates - frame)
                smoothed_local = np.average(values, axis=0, weights=temporal_weights)
                current_local = local[track, frame, joint]
                blended_local[joint] = (
                    (1.0 - strength) * current_local + strength * smoothed_local
                )
            palm_centre_local = np.median(
                blended_local[PALM_FRAME_JOINTS], axis=0
            )
            blended_local -= palm_centre_local
            for joint in np.flatnonzero(valid[track, frame]):
                filtered[track, frame, joint] = origins[track, frame] + blended_scale * (
                    blended_local[joint, 0] * blended_axis
                    + blended_local[joint, 1] * blended_axis_y
                )
            reconstructed_centre = np.median(
                filtered[track, frame, PALM_FRAME_JOINTS], axis=0
            )
            filtered[track, frame, valid[track, frame]] += (
                origins[track, frame] - reconstructed_centre
            )
    filtered[~valid] = np.nan
    return filtered


def palm_normalized_pixel_step_metric(pixels: np.ndarray, valid: np.ndarray) -> dict:
    """Summarize frame-to-frame finger-shape motion independent of palm motion."""
    local, _, _, _, _, frame_valid = palm_normalized_pixel_coordinates(pixels, valid)
    finger_joints = np.setdiff1d(np.arange(21), PALM_FRAME_JOINTS)
    samples = []
    for track in range(pixels.shape[0]):
        for frame in range(1, pixels.shape[1]):
            joint_valid = (
                frame_valid[track, frame - 1]
                & frame_valid[track, frame]
                & valid[track, frame - 1, finger_joints]
                & valid[track, frame, finger_joints]
            )
            if np.any(joint_valid):
                delta = local[track, frame, finger_joints[joint_valid]] - local[
                    track, frame - 1, finger_joints[joint_valid]
                ]
                samples.extend(np.linalg.norm(delta, axis=1).tolist())
    values = np.asarray(samples, dtype=np.float64)
    return {
        "sample_count": int(len(values)),
        "median": float(np.median(values)) if len(values) else None,
        "p95": float(np.percentile(values, 95)) if len(values) else None,
    }


def palm_normalized_3d_coordinates(
    points: np.ndarray,
    valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Express 3D joints in a per-frame rigid palm coordinate system."""
    if points.ndim != 4 or points.shape[2:] != (21, 3):
        raise ValueError(f"unexpected 3D landmark shape: {points.shape}")
    if valid.shape != points.shape[:-1]:
        raise ValueError("3D validity shape disagrees with landmark shape")
    local = np.full_like(points, np.nan, dtype=np.float64)
    origins = np.full(points.shape[:2] + (3,), np.nan, dtype=np.float64)
    axes = np.full(points.shape[:2] + (3, 3), np.nan, dtype=np.float64)
    scales = np.full(points.shape[:2], np.nan, dtype=np.float64)
    frame_valid = np.zeros(points.shape[:2], dtype=bool)
    for track in range(points.shape[0]):
        for frame in range(points.shape[1]):
            if not np.all(valid[track, frame, PALM_FRAME_JOINTS]):
                continue
            current = points[track, frame]
            origin = np.median(current[PALM_FRAME_JOINTS], axis=0)
            axis_x_vector = current[5] - current[17]
            scale = float(np.linalg.norm(axis_x_vector))
            if not np.isfinite(scale) or scale < 1e-5:
                continue
            axis_x = axis_x_vector / scale
            axis_y_vector = current[9] - current[0]
            axis_y_vector -= np.dot(axis_y_vector, axis_x) * axis_x
            axis_y_length = float(np.linalg.norm(axis_y_vector))
            if not np.isfinite(axis_y_length) or axis_y_length < 1e-5:
                continue
            axis_y = axis_y_vector / axis_y_length
            axis_z = np.cross(axis_x, axis_y)
            axis_z_length = float(np.linalg.norm(axis_z))
            if not np.isfinite(axis_z_length) or axis_z_length < 1e-5:
                continue
            axis_z /= axis_z_length
            palm_axes = np.stack((axis_x, axis_y, axis_z), axis=0)
            local[track, frame] = (current - origin) @ palm_axes.T / scale
            local[track, frame, ~valid[track, frame]] = np.nan
            origins[track, frame] = origin
            axes[track, frame] = palm_axes
            scales[track, frame] = scale
            frame_valid[track, frame] = True
    return local, origins, axes, scales, frame_valid


def smooth_3d_landmarks_in_palm_frame(
    points: np.ndarray,
    valid: np.ndarray,
    radius: int,
    strength: float,
) -> np.ndarray:
    """Smooth 3D finger configuration without smoothing global hand motion."""
    points = np.asarray(points, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    if radius <= 0 or strength <= 0.0:
        return points.copy()
    if not 0.0 <= strength <= 1.0:
        raise ValueError("local shape smoothing strength must be in [0, 1]")
    local, origins, axes, scales, frame_valid = palm_normalized_3d_coordinates(
        points, valid
    )
    filtered = points.copy()
    for track in range(points.shape[0]):
        for frame in range(points.shape[1]):
            if not frame_valid[track, frame]:
                continue
            lo = max(0, frame - radius)
            hi = min(points.shape[1], frame + radius + 1)
            candidate_frames = np.arange(lo, hi)
            palm_candidates = candidate_frames[frame_valid[track, candidate_frames]]
            palm_weights = radius + 1 - np.abs(palm_candidates - frame)
            mean_rotation = np.average(
                axes[track, palm_candidates], axis=0, weights=palm_weights
            )
            u, _, vt = np.linalg.svd(mean_rotation)
            mean_rotation = u @ vt
            if np.linalg.det(mean_rotation) < 0:
                u[:, -1] *= -1
                mean_rotation = u @ vt
            blended_rotation = (
                (1.0 - strength) * axes[track, frame] + strength * mean_rotation
            )
            u, _, vt = np.linalg.svd(blended_rotation)
            blended_rotation = u @ vt
            if np.linalg.det(blended_rotation) < 0:
                u[:, -1] *= -1
                blended_rotation = u @ vt
            median_log_scale = float(np.median(np.log(scales[track, palm_candidates])))
            blended_scale = float(np.exp(
                (1.0 - strength) * np.log(scales[track, frame])
                + strength * median_log_scale
            ))
            blended_local = local[track, frame].copy()
            for joint in range(21):
                if not valid[track, frame, joint]:
                    continue
                candidates = candidate_frames[
                    frame_valid[track, candidate_frames]
                    & valid[track, candidate_frames, joint]
                ]
                if len(candidates) < 2:
                    continue
                values = local[track, candidates, joint]
                median = np.median(values, axis=0)
                distances = np.linalg.norm(values - median, axis=1)
                distance_median = float(np.median(distances))
                distance_mad = 1.4826 * float(
                    np.median(np.abs(distances - distance_median))
                )
                robust_limit = max(0.10, distance_median + 3.5 * distance_mad)
                inliers = distances <= robust_limit
                if np.count_nonzero(inliers) >= 2:
                    candidates = candidates[inliers]
                    values = values[inliers]
                temporal_weights = radius + 1 - np.abs(candidates - frame)
                smoothed_local = np.average(values, axis=0, weights=temporal_weights)
                blended_local[joint] = (
                    (1.0 - strength) * local[track, frame, joint]
                    + strength * smoothed_local
                )
            palm_centre_local = np.median(
                blended_local[PALM_FRAME_JOINTS], axis=0
            )
            blended_local -= palm_centre_local
            for joint in np.flatnonzero(valid[track, frame]):
                filtered[track, frame, joint] = (
                    origins[track, frame]
                    + blended_scale * (blended_local[joint] @ blended_rotation)
                )
            reconstructed_centre = np.median(
                filtered[track, frame, PALM_FRAME_JOINTS], axis=0
            )
            filtered[track, frame, valid[track, frame]] += (
                origins[track, frame] - reconstructed_centre
            )
    filtered[~valid] = np.nan
    return filtered


def palm_normalized_3d_step_metric(points: np.ndarray, valid: np.ndarray) -> dict:
    """Summarize local 3D finger-shape changes independent of rigid hand motion."""
    local, _, _, _, frame_valid = palm_normalized_3d_coordinates(points, valid)
    finger_joints = np.setdiff1d(np.arange(21), PALM_FRAME_JOINTS)
    samples = []
    for track in range(points.shape[0]):
        for frame in range(1, points.shape[1]):
            joint_valid = (
                frame_valid[track, frame - 1]
                & frame_valid[track, frame]
                & valid[track, frame - 1, finger_joints]
                & valid[track, frame, finger_joints]
            )
            if np.any(joint_valid):
                delta = local[track, frame, finger_joints[joint_valid]] - local[
                    track, frame - 1, finger_joints[joint_valid]
                ]
                samples.extend(np.linalg.norm(delta, axis=1).tolist())
    values = np.asarray(samples, dtype=np.float64)
    return {
        "sample_count": int(len(values)),
        "median": float(np.median(values)) if len(values) else None,
        "p95": float(np.percentile(values, 95)) if len(values) else None,
    }


def reject_observation_outliers(
    points: np.ndarray,
    observed: np.ndarray,
    ranges: list[tuple[int, int]],
    temporal_radius: int,
    temporal_distance_m: float,
    max_hand_radius_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Reject impossible hand geometry and isolated temporal depth spikes.

    Rejected samples remain available in ``raw_positions_left_camera_m`` and are
    exported through an explicit mask.  They are only excluded from filtering,
    bone estimation and the MANO observation confidence.
    """
    accepted = observed.copy()
    rejected = np.zeros(observed.shape, dtype=bool)

    # A physical hand cannot contain a joint hundreds of millimetres away from
    # the robust centre of the other joints in the same frame.  Run this twice
    # so one extreme sample cannot hide a second, smaller outlier.
    for _ in range(2):
        for track, (start, end) in enumerate(ranges):
            if end < start:
                continue
            for frame in range(start, end + 1):
                joints = np.flatnonzero(accepted[track, frame])
                if len(joints) < 5:
                    continue
                centre = np.median(points[track, frame, joints], axis=0)
                distances = np.linalg.norm(points[track, frame, joints] - centre, axis=1)
                bad = joints[distances > max_hand_radius_m]
                if len(bad):
                    accepted[track, frame, bad] = False
                    rejected[track, frame, bad] = True

    if temporal_radius <= 0:
        return accepted, rejected

    # Stereo depth spikes are often internally plausible for one edge but
    # inconsistent with the same semantic joint in nearby frames.  A symmetric
    # median keeps this an offline, zero-phase preprocessing step.
    for _ in range(2):
        newly_rejected = []
        for track, (start, end) in enumerate(ranges):
            if end < start:
                continue
            for frame in range(start, end + 1):
                lo = max(start, frame - temporal_radius)
                hi = min(end, frame + temporal_radius)
                for joint in np.flatnonzero(accepted[track, frame]):
                    candidate_frames = np.arange(lo, hi + 1)
                    candidate_frames = candidate_frames[candidate_frames != frame]
                    candidate_frames = candidate_frames[accepted[track, candidate_frames, joint]]
                    if len(candidate_frames) < 3:
                        continue
                    local_median = np.median(points[track, candidate_frames, joint], axis=0)
                    distance = np.linalg.norm(points[track, frame, joint] - local_median)
                    if distance > temporal_distance_m:
                        newly_rejected.append((track, frame, joint))
        if not newly_rejected:
            break
        for track, frame, joint in newly_rejected:
            accepted[track, frame, joint] = False
            rejected[track, frame, joint] = True
    return accepted, rejected


def interpolate_short_gaps(points: np.ndarray, valid: np.ndarray, confidence: np.ndarray,
                           ranges: list[tuple[int, int]], max_gap: int):
    result = points.copy()
    result_valid = valid.copy()
    result_confidence = confidence.copy()
    interpolated = np.zeros(valid.shape, dtype=bool)
    for track, (start, end) in enumerate(ranges):
        if end < start:
            continue
        for joint in range(21):
            indices = np.flatnonzero(valid[track, start:end + 1, joint]) + start
            for left, right in zip(indices[:-1], indices[1:]):
                gap = int(right - left - 1)
                if gap <= 0 or gap > max_gap:
                    continue
                for offset in range(1, gap + 1):
                    alpha = offset / (gap + 1)
                    frame = int(left + offset)
                    result[track, frame, joint] = (
                        (1.0 - alpha) * points[track, left, joint]
                        + alpha * points[track, right, joint]
                    )
                    result_valid[track, frame, joint] = True
                    interpolated[track, frame, joint] = True
                    endpoint_confidence = min(confidence[track, left, joint], confidence[track, right, joint])
                    result_confidence[track, frame, joint] = endpoint_confidence * 0.55
    return result, result_valid, result_confidence, interpolated


def temporal_filter(points: np.ndarray, valid: np.ndarray, confidence: np.ndarray,
                    ranges: list[tuple[int, int]], radius: int) -> np.ndarray:
    if radius <= 0:
        return points.copy()
    result = points.copy()
    temporal_kernel = np.asarray([radius + 1 - abs(offset) for offset in range(-radius, radius + 1)])
    for track, (start, end) in enumerate(ranges):
        for frame in range(start, end + 1):
            for joint in range(21):
                if not valid[track, frame, joint]:
                    continue
                lo = max(start, frame - radius)
                hi = min(end, frame + radius)
                candidate_frames = np.arange(lo, hi + 1)
                mask = valid[track, candidate_frames, joint]
                candidate_frames = candidate_frames[mask]
                if not len(candidate_frames):
                    continue
                distances = np.linalg.norm(
                    points[track, candidate_frames, joint] - points[track, frame, joint], axis=1
                )
                candidate_frames = candidate_frames[distances <= 0.12]
                if not len(candidate_frames):
                    continue
                offsets = candidate_frames - frame
                kernel_weights = temporal_kernel[offsets + radius]
                weights = kernel_weights * np.maximum(confidence[track, candidate_frames, joint], 0.05)
                result[track, frame, joint] = np.average(
                    points[track, candidate_frames, joint], axis=0, weights=weights
                )
    return result


def estimate_bone_lengths(points: np.ndarray, observed: np.ndarray, confidence: np.ndarray):
    track_count = points.shape[0]
    lengths = np.full((track_count, len(SKELETON_EDGES)), np.nan, dtype=np.float64)
    samples = [[[] for _ in SKELETON_EDGES] for _ in range(track_count)]
    for track in range(track_count):
        for edge_index, (parent, child) in enumerate(SKELETON_EDGES):
            valid = observed[track, :, parent] & observed[track, :, child]
            valid &= confidence[track, :, parent] >= 0.15
            valid &= confidence[track, :, child] >= 0.15
            edge_lengths = np.linalg.norm(
                points[track, valid, child] - points[track, valid, parent], axis=1
            )
            edge_lengths = edge_lengths[(edge_lengths > 0.008) & (edge_lengths < 0.18)]
            if len(edge_lengths):
                median = np.median(edge_lengths)
                mad = np.median(np.abs(edge_lengths - median))
                tolerance = max(3.5 * 1.4826 * mad, 0.20 * median, 0.004)
                inliers = edge_lengths[np.abs(edge_lengths - median) <= tolerance]
                lengths[track, edge_index] = np.median(inliers)
                samples[track][edge_index] = inliers.tolist()
    if np.any(~np.isfinite(lengths)):
        for edge_index in range(len(SKELETON_EDGES)):
            global_values = [value for track in range(track_count) for value in samples[track][edge_index]]
            fallback = np.median(global_values) if global_values else 0.03
            lengths[~np.isfinite(lengths[:, edge_index]), edge_index] = fallback
    return lengths


def reject_bone_outliers(
    points: np.ndarray,
    accepted: np.ndarray,
    confidence: np.ndarray,
    bone_lengths: np.ndarray,
    absolute_tolerance_m: float,
    relative_tolerance: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove observations participating in physically impossible long/short edges."""
    result = accepted.copy()
    rejected = np.zeros(accepted.shape, dtype=bool)
    for _ in range(2):
        changes = []
        for track in range(points.shape[0]):
            for frame in range(points.shape[1]):
                violations = []
                counts = np.zeros(21, dtype=np.int32)
                for edge_index, (parent, child) in enumerate(SKELETON_EDGES):
                    if not (result[track, frame, parent] and result[track, frame, child]):
                        continue
                    length = np.linalg.norm(
                        points[track, frame, child] - points[track, frame, parent]
                    )
                    target = bone_lengths[track, edge_index]
                    tolerance = max(absolute_tolerance_m, relative_tolerance * target)
                    if abs(length - target) <= tolerance:
                        continue
                    violations.append((parent, child))
                    counts[parent] += 1
                    counts[child] += 1
                if not violations:
                    continue
                bad = set(np.flatnonzero(counts >= 2).tolist())
                for parent, child in violations:
                    if parent in bad or child in bad:
                        continue
                    parent_confidence = confidence[track, frame, parent]
                    child_confidence = confidence[track, frame, child]
                    bad.add(parent if parent_confidence <= child_confidence else child)
                changes.extend((track, frame, joint) for joint in bad)
        if not changes:
            break
        for track, frame, joint in changes:
            result[track, frame, joint] = False
            rejected[track, frame, joint] = True
    return result, rejected


def constrain_bones(points: np.ndarray, valid: np.ndarray, confidence: np.ndarray,
                    bone_lengths: np.ndarray, iterations: int, strength: float) -> np.ndarray:
    result = points.copy()
    for track in range(points.shape[0]):
        for frame in range(points.shape[1]):
            if np.count_nonzero(valid[track, frame]) < 2:
                continue
            for _ in range(iterations):
                for edge_index, (parent, child) in enumerate(SKELETON_EDGES):
                    if not (valid[track, frame, parent] and valid[track, frame, child]):
                        continue
                    vector = result[track, frame, child] - result[track, frame, parent]
                    length = float(np.linalg.norm(vector))
                    if length < 1e-8:
                        continue
                    error = length - bone_lengths[track, edge_index]
                    direction = vector / length
                    parent_confidence = max(confidence[track, frame, parent], 0.05)
                    child_confidence = max(confidence[track, frame, child], 0.05)
                    total = parent_confidence + child_confidence
                    correction = strength * error * direction
                    result[track, frame, parent] += correction * (child_confidence / total)
                    result[track, frame, child] -= correction * (parent_confidence / total)
    return result


def acceleration_metric(points: np.ndarray, valid: np.ndarray) -> float | None:
    values = []
    for track in range(points.shape[0]):
        for frame in range(1, points.shape[1] - 1):
            mask = valid[track, frame - 1] & valid[track, frame] & valid[track, frame + 1]
            if np.any(mask):
                acceleration = points[track, frame + 1, mask] - 2 * points[track, frame, mask] + points[track, frame - 1, mask]
                values.extend((np.linalg.norm(acceleration, axis=1) * 1000.0).tolist())
    return float(np.median(values)) if values else None


def bone_error_metric(points: np.ndarray, valid: np.ndarray, targets: np.ndarray) -> dict:
    absolute_mm = []
    relative = []
    for track in range(points.shape[0]):
        for edge_index, (parent, child) in enumerate(SKELETON_EDGES):
            mask = valid[track, :, parent] & valid[track, :, child]
            current = np.linalg.norm(points[track, mask, child] - points[track, mask, parent], axis=1)
            errors = np.abs(current - targets[track, edge_index])
            absolute_mm.extend((errors * 1000.0).tolist())
            relative.extend((errors / targets[track, edge_index]).tolist())
    return {
        "median_absolute_mm": float(np.median(absolute_mm)) if absolute_mm else None,
        "p95_absolute_mm": float(np.percentile(absolute_mm, 95)) if absolute_mm else None,
        "median_relative": float(np.median(relative)) if relative else None,
        "p95_relative": float(np.percentile(relative, 95)) if relative else None,
    }


def write_csv(path: Path, data: dict, stabilized: np.ndarray, valid: np.ndarray,
              accepted_observed: np.ndarray, rejected: np.ndarray,
              interpolated: np.ndarray, confidence: np.ndarray) -> None:
    fields = [
        "pair_index", "left_index", "right_index", "track_id", "handedness", "landmark_index",
        "joint_name", "source_state", "input_observed", "accepted_observation",
        "outlier_rejected", "output_valid", "confidence",
        "raw_x_m", "raw_y_m", "raw_z_m", "x_left_camera_m", "y_left_camera_m", "z_left_camera_m",
    ]
    raw = data["raw"]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for track_slot, track_id in enumerate(data["track_ids"]):
            for pair in range(stabilized.shape[1]):
                if not np.any(valid[track_slot, pair]) and pair not in data["pair_metadata"]:
                    continue
                metadata = data["pair_metadata"].get(pair, {"left_index": -1, "right_index": -1})
                for joint in range(21):
                    source_state = "observed" if accepted_observed[track_slot, pair, joint] else (
                        "interpolated" if interpolated[track_slot, pair, joint] else "missing"
                    )
                    if rejected[track_slot, pair, joint] and not interpolated[track_slot, pair, joint]:
                        source_state = "rejected_outlier"
                    raw_point = raw[track_slot, pair, joint]
                    point = stabilized[track_slot, pair, joint]
                    output_valid = bool(valid[track_slot, pair, joint])
                    writer.writerow({
                        "pair_index": pair,
                        "left_index": metadata["left_index"],
                        "right_index": metadata["right_index"],
                        "track_id": int(track_id),
                        "handedness": str(data["handedness"][track_slot]),
                        "landmark_index": joint,
                        "joint_name": JOINT_NAMES[joint],
                        "source_state": source_state,
                        "input_observed": int(data["observed"][track_slot, pair, joint]),
                        "accepted_observation": int(accepted_observed[track_slot, pair, joint]),
                        "outlier_rejected": int(rejected[track_slot, pair, joint]),
                        "output_valid": int(output_valid),
                        "confidence": f"{confidence[track_slot, pair, joint]:.8f}",
                        "raw_x_m": f"{raw_point[0]:.9f}" if np.all(np.isfinite(raw_point)) else "nan",
                        "raw_y_m": f"{raw_point[1]:.9f}" if np.all(np.isfinite(raw_point)) else "nan",
                        "raw_z_m": f"{raw_point[2]:.9f}" if np.all(np.isfinite(raw_point)) else "nan",
                        "x_left_camera_m": f"{point[0]:.9f}" if output_valid else "nan",
                        "y_left_camera_m": f"{point[1]:.9f}" if output_valid else "nan",
                        "z_left_camera_m": f"{point[2]:.9f}" if output_valid else "nan",
                    })


def map_point(point: np.ndarray, axis_a: int, axis_b: int, limits, panel) -> tuple[int, int]:
    x0, y0, width, height = panel
    a_min, a_max, b_min, b_max = limits
    x = x0 + int(np.clip((point[axis_a] - a_min) / max(a_max - a_min, 1e-6), 0, 1) * width)
    y = y0 + height - int(np.clip((point[axis_b] - b_min) / max(b_max - b_min, 1e-6), 0, 1) * height)
    return x, y


def render_video(path: Path, raw: np.ndarray, stabilized: np.ndarray, valid: np.ndarray,
                 observed: np.ndarray, track_ids: np.ndarray, handedness: np.ndarray, fps: float) -> None:
    finite = stabilized[np.isfinite(stabilized).all(axis=-1)]
    if not len(finite):
        return
    percentiles = np.percentile(finite, [2, 98], axis=0)
    padding = np.maximum((percentiles[1] - percentiles[0]) * 0.12, 0.025)
    x_limits = (percentiles[0, 0] - padding[0], percentiles[1, 0] + padding[0])
    y_limits = (percentiles[0, 1] - padding[1], percentiles[1, 1] + padding[1])
    z_limits = (percentiles[0, 2] - padding[2], percentiles[1, 2] + padding[2])
    canvas_size = (1280, 720)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, canvas_size)
    if not writer.isOpened():
        raise RuntimeError(f"cannot create {path}")
    panels = [(55, 90, 550, 550), (675, 90, 550, 550)]
    colors = [(40, 220, 40), (40, 170, 255), (255, 80, 180), (255, 220, 40)]
    try:
        for frame in range(stabilized.shape[1]):
            canvas = np.full((canvas_size[1], canvas_size[0], 3), 20, dtype=np.uint8)
            cv2.putText(canvas, f"MANO preparation | pair {frame}", (35, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (240, 240, 240), 2, cv2.LINE_AA)
            cv2.putText(canvas, "Front: X / Y", (panels[0][0], 78), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (210, 210, 210), 2, cv2.LINE_AA)
            cv2.putText(canvas, "Top: X / Z", (panels[1][0], 78), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (210, 210, 210), 2, cv2.LINE_AA)
            for panel in panels:
                cv2.rectangle(canvas, (panel[0], panel[1]), (panel[0] + panel[2], panel[1] + panel[3]),
                              (70, 70, 70), 1)
            for track in range(stabilized.shape[0]):
                color = colors[int(track_ids[track]) % len(colors)]
                for axis_a, axis_b, limits, panel in (
                    (0, 1, (*x_limits, *y_limits), panels[0]),
                    (0, 2, (*x_limits, *z_limits), panels[1]),
                ):
                    for parent, child in SKELETON_EDGES:
                        if valid[track, frame, parent] and valid[track, frame, child]:
                            p0 = map_point(stabilized[track, frame, parent], axis_a, axis_b, limits, panel)
                            p1 = map_point(stabilized[track, frame, child], axis_a, axis_b, limits, panel)
                            cv2.line(canvas, p0, p1, color, 3, cv2.LINE_AA)
                    for joint in range(21):
                        if observed[track, frame, joint] and np.all(np.isfinite(raw[track, frame, joint])):
                            raw_point = map_point(raw[track, frame, joint], axis_a, axis_b, limits, panel)
                            cv2.circle(canvas, raw_point, 3, (105, 105, 105), -1, cv2.LINE_AA)
                        if valid[track, frame, joint]:
                            point = map_point(stabilized[track, frame, joint], axis_a, axis_b, limits, panel)
                            cv2.circle(canvas, point, 4, color, -1, cv2.LINE_AA)
                cv2.putText(canvas, f"T{int(track_ids[track])} {handedness[track]}",
                            (40 + track * 220, 690), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
            writer.write(canvas)
    finally:
        writer.release()


def write_preview_montage(video_path: Path, output_path: Path, frame_count: int) -> None:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot read {video_path} for montage")
    sample_indices = np.linspace(0, max(frame_count - 1, 0), 6, dtype=int)
    samples = []
    try:
        for index in sample_indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = capture.read()
            if not ok:
                continue
            frame = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_AREA)
            samples.append(frame)
    finally:
        capture.release()
    if not samples:
        raise RuntimeError("could not decode any montage frames")
    while len(samples) < 6:
        samples.append(samples[-1].copy())
    montage = np.vstack((np.hstack(samples[:3]), np.hstack(samples[3:6])))
    if not cv2.imwrite(str(output_path), montage):
        raise RuntimeError(f"cannot write {output_path}")


def displacement_metric(stabilized: np.ndarray, raw: np.ndarray, accepted: np.ndarray) -> dict:
    displacement_mm = np.linalg.norm(stabilized[accepted] - raw[accepted], axis=1) * 1000.0
    return {
        "median_mm": float(np.median(displacement_mm)) if len(displacement_mm) else None,
        "p95_mm": float(np.percentile(displacement_mm, 95)) if len(displacement_mm) else None,
        "max_mm": float(np.max(displacement_mm)) if len(displacement_mm) else None,
    }


def stabilize_once(
    raw: np.ndarray,
    accepted_observed: np.ndarray,
    base_confidence: np.ndarray,
    max_gap: int,
    window_radius: int,
    bone_iterations: int,
    bone_strength: float,
):
    cleaned_points = raw.copy()
    cleaned_points[~accepted_observed] = np.nan
    cleaned_confidence = base_confidence.copy()
    cleaned_confidence[~accepted_observed] = 0.0
    ranges = active_ranges(accepted_observed)
    interpolated_points, valid, confidence, interpolated = interpolate_short_gaps(
        cleaned_points, accepted_observed, cleaned_confidence, ranges, max_gap
    )
    smoothed = temporal_filter(
        interpolated_points, valid, confidence, ranges, window_radius
    )
    bone_lengths = estimate_bone_lengths(cleaned_points, accepted_observed, cleaned_confidence)
    stabilized = constrain_bones(
        smoothed, valid, confidence, bone_lengths, bone_iterations, bone_strength
    )
    stabilized[~valid] = np.nan
    return {
        "cleaned_points": cleaned_points,
        "cleaned_confidence": cleaned_confidence,
        "ranges": ranges,
        "valid": valid,
        "confidence": confidence,
        "interpolated": interpolated,
        "bone_lengths": bone_lengths,
        "stabilized": stabilized,
    }


def main() -> int:
    args = parse_args()
    if (args.max_gap < 0 or args.outlier_window < 0 or args.pixel_outlier_window < 0
            or args.pixel_smoothing_radius < 0 or args.local_shape_smoothing_radius < 0
            or args.window_radius < 0
            or args.bone_iterations < 0):
        raise ValueError("gap, radius and iterations must be non-negative")
    if (args.outlier_distance_m <= 0.0 or args.max_hand_radius_m <= 0.0
            or args.pixel_outlier_distance <= 0.0 or args.pixel_scale_ratio <= 1.0
            or args.bone_outlier_absolute_m <= 0.0 or args.bone_outlier_relative <= 0.0
            or args.max_observation_adjustment_m <= 0.0):
        raise ValueError("outlier distance thresholds must be positive")
    if not 0.0 <= args.bone_strength <= 1.0:
        raise ValueError("--bone-strength must be in [0, 1]")
    if not 0.0 <= args.pixel_smoothing_strength <= 1.0:
        raise ValueError("--pixel-smoothing-strength must be in [0, 1]")
    if not 0.0 <= args.local_shape_smoothing_strength <= 1.0:
        raise ValueError("--local-shape-smoothing-strength must be in [0, 1]")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    start_time = time.perf_counter()
    data = load_stereo_csv(args.input.resolve())
    camera_contract = load_rectified_camera_contract(args.input.resolve(), data)
    input_ranges = active_ranges(data["observed"])
    accepted_observed, rejected = reject_observation_outliers(
        data["raw"], data["observed"], input_ranges, args.outlier_window,
        args.outlier_distance_m, args.max_hand_radius_m,
    )
    left_pixel_outliers = detect_temporal_pixel_outliers(
        data["left_pixels"], args.pixel_outlier_window,
        args.pixel_outlier_distance, args.pixel_scale_ratio,
    )
    right_pixel_outliers = detect_temporal_pixel_outliers(
        data["right_pixels"], args.pixel_outlier_window,
        args.pixel_outlier_distance, args.pixel_scale_ratio,
    )
    pixel_geometry_rejected = (
        data["observed"] & (left_pixel_outliers | right_pixel_outliers)
    )
    accepted_observed[pixel_geometry_rejected] = False
    rejected |= pixel_geometry_rejected
    left_pixel_valid = np.isfinite(data["left_pixels"]).all(axis=-1) & ~left_pixel_outliers
    right_pixel_valid = np.isfinite(data["right_pixels"]).all(axis=-1) & ~right_pixel_outliers
    left_pixels_filtered = smooth_pixel_landmarks_in_palm_frame(
        data["left_pixels"], left_pixel_valid,
        args.pixel_smoothing_radius, args.pixel_smoothing_strength,
    )
    right_pixels_filtered = smooth_pixel_landmarks_in_palm_frame(
        data["right_pixels"], right_pixel_valid,
        args.pixel_smoothing_radius, args.pixel_smoothing_strength,
    )
    preliminary_points = data["raw"].copy()
    preliminary_points[~accepted_observed] = np.nan
    preliminary_confidence = data["confidence"].copy()
    preliminary_confidence[~accepted_observed] = 0.0
    preliminary_bone_lengths = estimate_bone_lengths(
        preliminary_points, accepted_observed, preliminary_confidence
    )
    accepted_observed, bone_rejected = reject_bone_outliers(
        data["raw"], accepted_observed, preliminary_confidence, preliminary_bone_lengths,
        args.bone_outlier_absolute_m, args.bone_outlier_relative,
    )
    rejected |= bone_rejected
    prepared = None
    residual_rejected_count = 0
    for _ in range(3):
        prepared = stabilize_once(
            data["raw"], accepted_observed, data["confidence"], args.max_gap,
            args.window_radius, args.bone_iterations, args.bone_strength,
        )
        residual = np.linalg.norm(prepared["stabilized"] - data["raw"], axis=-1)
        residual_bad = accepted_observed & (residual > args.max_observation_adjustment_m)
        if not np.any(residual_bad):
            break
        accepted_observed[residual_bad] = False
        rejected[residual_bad] = True
        residual_rejected_count += int(np.count_nonzero(residual_bad))
    else:
        prepared = stabilize_once(
            data["raw"], accepted_observed, data["confidence"], args.max_gap,
            args.window_radius, args.bone_iterations, args.bone_strength,
        )
    assert prepared is not None
    cleaned_points = prepared["cleaned_points"]
    ranges = prepared["ranges"]
    valid = prepared["valid"]
    confidence = prepared["confidence"]
    interpolated = prepared["interpolated"]
    bone_lengths = prepared["bone_lengths"]
    stabilized_before_local_shape = prepared["stabilized"]
    stabilized = smooth_3d_landmarks_in_palm_frame(
        stabilized_before_local_shape, valid,
        args.local_shape_smoothing_radius, args.local_shape_smoothing_strength,
    )

    csv_path = output / "stabilized_landmarks_3d.csv"
    npz_path = output / "mano_input.npz"
    summary_path = output / "summary.json"
    video_path = output / "stabilized_3d.mp4"
    montage_path = output / "preview_montage.jpg"
    write_csv(csv_path, data, stabilized, valid, accepted_observed, rejected, interpolated, confidence)
    pair_indices = np.arange(stabilized.shape[1], dtype=np.int32)
    left_indices = np.asarray([
        data["pair_metadata"].get(int(pair), {}).get("left_index", -1) for pair in pair_indices
    ], dtype=np.int32)
    right_indices = np.asarray([
        data["pair_metadata"].get(int(pair), {}).get("right_index", -1) for pair in pair_indices
    ], dtype=np.int32)
    np.savez_compressed(
        npz_path,
        positions_left_camera_m=stabilized.astype(np.float32),
        valid=valid,
        observed=accepted_observed,
        input_observed=data["observed"],
        outlier_rejected=rejected,
        interpolated=interpolated,
        confidence=confidence.astype(np.float32),
        raw_positions_left_camera_m=data["raw"].astype(np.float32),
        positions_before_local_shape_filter_m=stabilized_before_local_shape.astype(np.float32),
        left_rectified_px=data["left_pixels"].astype(np.float32),
        right_rectified_px=data["right_pixels"].astype(np.float32),
        left_rectified_px_filtered=left_pixels_filtered.astype(np.float32),
        right_rectified_px_filtered=right_pixels_filtered.astype(np.float32),
        left_rectified_valid=left_pixel_valid,
        right_rectified_valid=right_pixel_valid,
        left_rectified_outlier=left_pixel_outliers,
        right_rectified_outlier=right_pixel_outliers,
        track_ids=data["track_ids"],
        handedness=data["handedness"],
        joint_names=np.asarray(JOINT_NAMES),
        skeleton_edges=np.asarray(SKELETON_EDGES, dtype=np.int32),
        bone_lengths_m=bone_lengths.astype(np.float32),
        pair_indices=pair_indices,
        left_frame_indices=left_indices,
        right_frame_indices=right_indices,
        fps=np.asarray(args.fps, dtype=np.float32),
        left_to_rectified_rotation=camera_contract["left_to_rectified_rotation"].astype(np.float32),
        projection_left_rectified=camera_contract["p1"].astype(np.float32),
        projection_right_rectified=camera_contract["p2"].astype(np.float32),
        rectified_size=camera_contract["rectified_size"],
    )
    if not args.no_video:
        render_video(
            video_path, data["raw"], stabilized, valid, data["observed"],
            data["track_ids"], data["handedness"], args.fps
        )
        write_preview_montage(video_path, montage_path, stabilized.shape[1])

    raw_bone_error = bone_error_metric(data["raw"], data["observed"], bone_lengths)
    accepted_bone_error = bone_error_metric(cleaned_points, accepted_observed, bone_lengths)
    stabilized_bone_error = bone_error_metric(stabilized, valid, bone_lengths)
    input_observed_count = int(np.count_nonzero(data["observed"]))
    observed_count = int(np.count_nonzero(accepted_observed))
    rejected_count = int(np.count_nonzero(rejected))
    interpolated_count = int(np.count_nonzero(interpolated))
    output_count = int(np.count_nonzero(valid))
    complete_input = int(np.count_nonzero(np.all(data["observed"], axis=2)))
    complete_before = int(np.count_nonzero(np.all(accepted_observed, axis=2)))
    complete_after = int(np.count_nonzero(np.all(valid, axis=2)))
    finite_depths = stabilized[valid][:, 2]
    counts_by_track = {
        str(int(track_id)): {
            "input_observed": int(np.count_nonzero(data["observed"][track])),
            "outlier_rejected": int(np.count_nonzero(rejected[track])),
            "accepted_observed": int(np.count_nonzero(accepted_observed[track])),
            "interpolated": int(np.count_nonzero(interpolated[track])),
            "output_valid": int(np.count_nonzero(valid[track])),
            "complete_hand_instances_output": int(np.count_nonzero(np.all(valid[track], axis=1))),
        }
        for track, track_id in enumerate(data["track_ids"])
    }
    counts_by_landmark = {
        str(joint): {
            "joint_name": JOINT_NAMES[joint],
            "input_observed": int(np.count_nonzero(data["observed"][:, :, joint])),
            "outlier_rejected": int(np.count_nonzero(rejected[:, :, joint])),
            "accepted_observed": int(np.count_nonzero(accepted_observed[:, :, joint])),
            "interpolated": int(np.count_nonzero(interpolated[:, :, joint])),
            "output_valid": int(np.count_nonzero(valid[:, :, joint])),
        }
        for joint in range(21)
    }
    summary = {
        "stage": "mano_preparation",
        "input": str(args.input.resolve()),
        "track_ids": data["track_ids"].tolist(),
        "handedness": data["handedness"].tolist(),
        "active_pair_ranges": [list(item) for item in ranges],
        "input_observed_3d_points": input_observed_count,
        "outlier_rejected_3d_points": rejected_count,
        "pixel_geometry_rejected_3d_points": int(np.count_nonzero(pixel_geometry_rejected)),
        "left_pixel_outliers": int(np.count_nonzero(left_pixel_outliers)),
        "right_pixel_outliers": int(np.count_nonzero(right_pixel_outliers)),
        "left_pixel_shape_step_raw": palm_normalized_pixel_step_metric(
            data["left_pixels"], left_pixel_valid
        ),
        "left_pixel_shape_step_filtered": palm_normalized_pixel_step_metric(
            left_pixels_filtered, left_pixel_valid
        ),
        "right_pixel_shape_step_raw": palm_normalized_pixel_step_metric(
            data["right_pixels"], right_pixel_valid
        ),
        "right_pixel_shape_step_filtered": palm_normalized_pixel_step_metric(
            right_pixels_filtered, right_pixel_valid
        ),
        "optimization_residual_rejected_3d_points": residual_rejected_count,
        "observed_3d_points": observed_count,
        "interpolated_3d_points": interpolated_count,
        "output_3d_points": output_count,
        "complete_hand_instances_input": complete_input,
        "complete_hand_instances_before": complete_before,
        "complete_hand_instances_after": complete_after,
        "outlier_rejected_then_interpolated": int(np.count_nonzero(rejected & interpolated)),
        "point_counts_by_track": counts_by_track,
        "point_counts_by_landmark": counts_by_landmark,
        "stabilized_depth_m": {
            "median": float(np.median(finite_depths)) if len(finite_depths) else None,
            "p05": float(np.percentile(finite_depths, 5)) if len(finite_depths) else None,
            "p95": float(np.percentile(finite_depths, 95)) if len(finite_depths) else None,
        },
        "rectified_camera_contract": {
            "left_to_rectified_rotation": camera_contract["left_to_rectified_rotation"].tolist(),
            "rotation_recovery_residual_median_m": camera_contract["residual_median_m"],
            "rotation_recovery_residual_p95_m": camera_contract["residual_p95_m"],
            "projection_left": camera_contract["p1"].tolist(),
            "projection_right": camera_contract["p2"].tolist(),
            "rectified_size": camera_contract["rectified_size"].tolist(),
        },
        "raw_temporal_acceleration_median_mm_per_frame2": acceleration_metric(data["raw"], data["observed"]),
        "accepted_temporal_acceleration_median_mm_per_frame2": acceleration_metric(
            cleaned_points, accepted_observed
        ),
        "stabilized_temporal_acceleration_median_mm_per_frame2": acceleration_metric(stabilized, valid),
        "local_3d_shape_step_before_filter": palm_normalized_3d_step_metric(
            stabilized_before_local_shape, valid
        ),
        "local_3d_shape_step_after_filter": palm_normalized_3d_step_metric(
            stabilized, valid
        ),
        "raw_bone_length_error": raw_bone_error,
        "accepted_bone_length_error": accepted_bone_error,
        "stabilized_bone_length_error": stabilized_bone_error,
        "accepted_observation_displacement_after_stabilization": displacement_metric(
            stabilized, data["raw"], accepted_observed
        ),
        "bone_lengths_m": {
            str(int(data["track_ids"][track])): {
                f"{parent}-{child}": float(bone_lengths[track, edge_index])
                for edge_index, (parent, child) in enumerate(SKELETON_EDGES)
            }
            for track in range(len(data["track_ids"]))
        },
        "parameters": {
            "max_gap": args.max_gap,
            "outlier_window": args.outlier_window,
            "outlier_distance_m": args.outlier_distance_m,
            "max_hand_radius_m": args.max_hand_radius_m,
            "pixel_outlier_window": args.pixel_outlier_window,
            "pixel_outlier_distance": args.pixel_outlier_distance,
            "pixel_scale_ratio": args.pixel_scale_ratio,
            "pixel_smoothing_radius": args.pixel_smoothing_radius,
            "pixel_smoothing_strength": args.pixel_smoothing_strength,
            "local_shape_smoothing_radius": args.local_shape_smoothing_radius,
            "local_shape_smoothing_strength": args.local_shape_smoothing_strength,
            "bone_outlier_absolute_m": args.bone_outlier_absolute_m,
            "bone_outlier_relative": args.bone_outlier_relative,
            "window_radius": args.window_radius,
            "bone_iterations": args.bone_iterations,
            "bone_strength": args.bone_strength,
            "max_observation_adjustment_m": args.max_observation_adjustment_m,
        },
        "elapsed_seconds": time.perf_counter() - start_time,
        "mano_ready_contract": {
            "file": str(npz_path),
            "positions": "positions_left_camera_m [track, pair, 21, 3], metres",
            "valid": "valid [track, pair, 21]",
            "confidence": "confidence [track, pair, 21]",
            "note": "Missing long gaps remain NaN and must be handled by the MANO objective.",
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("EGO MANO preparation")
    print(f"Tracks: {data['track_ids'].tolist()} ({data['handedness'].tolist()})")
    print(
        "Input/rejected/accepted/interpolated/output 3D points: "
        f"{input_observed_count}/{rejected_count}/{observed_count}/{interpolated_count}/{output_count}"
    )
    print(f"Complete hand instances input/accepted/after: {complete_input}/{complete_before}/{complete_after}")
    print(
        "Temporal acceleration median raw/stabilized: "
        f"{summary['raw_temporal_acceleration_median_mm_per_frame2']:.3f}/"
        f"{summary['stabilized_temporal_acceleration_median_mm_per_frame2']:.3f} mm/frame^2"
    )
    print(
        "Bone error median raw/stabilized: "
        f"{raw_bone_error['median_absolute_mm']:.3f}/"
        f"{stabilized_bone_error['median_absolute_mm']:.3f} mm"
    )
    print(f"Stabilized CSV: {csv_path}")
    print(f"MANO input NPZ: {npz_path}")
    if not args.no_video:
        print(f"3D visualization: {video_path}")
        print(f"Preview montage: {montage_path}")
    print(f"Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(1)
