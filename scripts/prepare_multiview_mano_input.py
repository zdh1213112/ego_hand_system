#!/usr/bin/env python3
"""Convert strict six-view fusion into the shared MANO sequence-fit contract."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from camera_models import RectificationOptions, create_stereo_rectification
from camera_models.double_sphere import unproject as unproject_double_sphere
from ego_data.calibration import CameraCalibration, StereoCalibration
from stabilize_hand_3d import (
    acceleration_metric,
    active_ranges,
    bone_error_metric,
    detect_temporal_pixel_outliers,
    displacement_metric,
    estimate_bone_lengths,
    palm_normalized_3d_step_metric,
    reject_bone_outliers,
    reject_observation_outliers,
    smooth_3d_landmarks_in_palm_frame,
    smooth_pixel_landmarks_in_palm_frame,
    stabilize_once,
)


HAND_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fusion", required=True, type=Path, help="strict fusion directory")
    parser.add_argument("--dataset", required=True, type=Path, help="normalized multiview dataset")
    parser.add_argument("--output", required=True, type=Path, help="MANO input NPZ")
    parser.add_argument("--rectification-output", required=True, type=Path)
    parser.add_argument("--left-camera", default="camera2")
    parser.add_argument("--right-camera", default="camera3")
    parser.add_argument("--focal-scale", type=float, default=1.0)
    parser.add_argument("--anatomy-refine", type=int, choices=(0, 1), default=1)
    parser.add_argument("--outlier-window", type=int, default=4)
    parser.add_argument("--outlier-distance-m", type=float, default=0.10)
    parser.add_argument("--max-hand-radius-m", type=float, default=0.20)
    parser.add_argument("--bone-outlier-absolute-m", type=float, default=0.025)
    parser.add_argument("--bone-outlier-relative", type=float, default=0.45)
    parser.add_argument("--max-gap", type=int, default=3)
    parser.add_argument("--smoothing-radius", type=int, default=2)
    parser.add_argument("--local-shape-strength", type=float, default=0.60)
    parser.add_argument("--bone-iterations", type=int, default=12)
    parser.add_argument("--bone-strength", type=float, default=0.80)
    parser.add_argument(
        "--handedness-policy", choices=("auto", "strict", "geometric"), default="auto",
        help=(
            "auto permits geometric identity only for a fusion result configured "
            "with ignore/adaptive; strict preserves the legacy validation"
        ),
    )
    return parser.parse_args()


def _project_rectified(
    points_left: np.ndarray, rotation: np.ndarray, projection: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    rectified = (rotation @ points_left.T).T
    homogeneous = np.concatenate(
        (rectified, np.ones((len(rectified), 1), dtype=np.float64)), axis=1
    )
    projected = (projection @ homogeneous.T).T
    valid = np.isfinite(projected).all(axis=1) & (projected[:, 2] > 1e-8)
    pixels = np.full((len(points_left), 2), np.nan, dtype=np.float64)
    pixels[valid] = projected[valid, :2] / projected[valid, 2:3]
    return pixels, valid


def _rectify_view_pixels(
    camera: CameraCalibration,
    pixels: np.ndarray,
    rectification_rotation: np.ndarray,
    projection: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Map native fisheye/DS pixels to their pinhole rectified ray pixels."""
    pixels = np.asarray(pixels, dtype=np.float64)
    if pixels.shape != (21, 2):
        raise ValueError(f"expected 21x2 image joints, got {pixels.shape}")
    if camera.model == "DS":
        rays, valid = unproject_double_sphere(camera, pixels)
        rectified = (rectification_rotation @ rays.T).T
        valid &= np.isfinite(rectified).all(axis=1) & (rectified[:, 2] > 1e-8)
        output = np.full((21, 2), np.nan, dtype=np.float64)
        projected = (projection[:, :3] @ rectified.T).T
        output[valid] = projected[valid, :2] / projected[valid, 2:3]
        return output, valid
    if camera.model == "KB":
        output = cv2.fisheye.undistortPoints(
            pixels.reshape(-1, 1, 2),
            camera.K,
            camera.distortion,
            R=rectification_rotation,
            P=projection[:, :3],
        )[:, 0]
        valid = np.isfinite(output).all(axis=1)
        output[~valid] = np.nan
        return output, valid
    raise ValueError(f"unsupported camera model for rectification: {camera.model}")


def _load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if "sync_index" not in row or "hands" not in row:
                raise ValueError(f"invalid fusion row at line {line_number}")
            rows.append(row)
    return rows


def _fps(sync_csv: Path) -> float:
    with sync_csv.open("r", encoding="utf-8", newline="") as stream:
        values = [int(row["reference_timestamp_ns"]) for row in csv.DictReader(stream)]
    return 1e9 / float(np.median(np.diff(values))) if len(values) > 1 else 30.0


def main() -> int:
    args = parse_args()
    dataset = args.dataset.resolve()
    fusion = args.fusion.resolve()
    output = args.output.resolve()
    rectification_output = args.rectification_output.resolve()
    if output.exists() or rectification_output.exists():
        raise FileExistsError("MANO input or rectification output already exists")
    if args.left_camera == args.right_camera:
        raise ValueError("left/right cameras must differ")
    if args.focal_scale <= 0:
        raise ValueError("focal-scale must be positive")
    if min(
        args.outlier_window,
        args.max_gap,
        args.smoothing_radius,
        args.bone_iterations,
    ) < 0:
        raise ValueError("anatomy refinement windows/iterations must be non-negative")
    if min(
        args.outlier_distance_m,
        args.max_hand_radius_m,
        args.bone_outlier_absolute_m,
        args.bone_outlier_relative,
    ) <= 0:
        raise ValueError("anatomy refinement thresholds must be positive")
    if not 0.0 <= args.local_shape_strength <= 1.0:
        raise ValueError("local-shape-strength must be in [0, 1]")
    if not 0.0 <= args.bone_strength <= 1.0:
        raise ValueError("bone-strength must be in [0, 1]")

    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("dataset_type") != "normalized_multiview":
        raise ValueError(f"not a normalized multiview dataset: {dataset}")
    camera_ids = tuple(manifest["camera_ids"])
    if args.left_camera not in camera_ids or args.right_camera not in camera_ids:
        raise ValueError("selected export cameras are not present in the dataset")
    pair_count = int(manifest["synchronization"]["frame_count"])
    left = CameraCalibration.load(dataset / "calibration" / f"{args.left_camera}.json")
    right = CameraCalibration.load(dataset / "calibration" / f"{args.right_camera}.json")
    fusion_summary = json.loads((fusion / "summary.json").read_text(encoding="utf-8"))
    fusion_handedness = fusion_summary.get("parameters", {}).get(
        "detector_handedness", "strict"
    )
    allow_geometric_handedness = (
        args.handedness_policy == "geometric"
        or (
            args.handedness_policy == "auto"
            and fusion_handedness in ("ignore", "adaptive")
        )
    )
    stereo = StereoCalibration.from_cameras(left, right)
    rectification = create_stereo_rectification(
        stereo, RectificationOptions(focal_scale=args.focal_scale), "auto"
    )

    positions = np.full((2, pair_count, 21, 3), np.nan, dtype=np.float32)
    confidence = np.zeros((2, pair_count, 21), dtype=np.float32)
    valid = np.zeros((2, pair_count, 21), dtype=bool)
    observed = np.zeros_like(valid)
    left_px = np.full((2, pair_count, 21, 2), np.nan, dtype=np.float32)
    right_px = np.full_like(left_px, np.nan)
    left_px_valid = np.zeros_like(valid)
    right_px_valid = np.zeros_like(valid)
    populated: set[tuple[int, int]] = set()
    identity_observations = 0
    identity_mismatches = 0

    rows = _load_rows(fusion / "accepted.jsonl")
    rotation_base_left = left.T_base_camera[:3, :3]
    center_base_left = left.T_base_camera[:3, 3]
    width, height = rectification.image_size
    for row in rows:
        pair = int(row["sync_index"])
        if not 0 <= pair < pair_count:
            raise ValueError(f"fusion sync_index {pair} outside dataset range")
        for hand in row["hands"]:
            side = int(hand["side"])
            if side not in (0, 1):
                raise ValueError(f"invalid hand side at sync_index {pair}: {side}")
            key = (side, pair)
            if key in populated:
                raise ValueError(f"duplicate fused hand: side={side}, sync_index={pair}")
            populated.add(key)
            for camera, view in hand.get("views", {}).items():
                if int(view.get("inlier_joint_count", 0)) <= 0:
                    continue
                detector_side = view.get("detector_is_right")
                identity_match = (
                    detector_side is not None and int(detector_side) == side
                )
                if not identity_match and not allow_geometric_handedness:
                    raise ValueError(
                        f"fusion is not strict-handedness clean: sync={pair}, side={side}, "
                        f"camera={camera}, detector_is_right={detector_side!r}"
                    )
                if identity_match:
                    identity_observations += 1
                else:
                    identity_mismatches += 1

            points_base = np.asarray(hand["joints_base_m"], dtype=np.float64)
            support = np.asarray(hand["inlier_view_counts"], dtype=np.int32)
            if points_base.shape != (21, 3) or support.shape != (21,):
                raise ValueError(f"invalid fused hand shape at sync_index {pair}")
            points_left = (rotation_base_left.T @ (points_base - center_base_left).T).T
            finite = np.isfinite(points_left).all(axis=1) & (support >= 2)
            left_view = hand.get("views", {}).get(args.left_camera)
            right_view = hand.get("views", {}).get(args.right_camera)
            if left_view is not None and int(left_view.get("inlier_joint_count", 0)) > 0:
                pixels_left, projectable_left = _rectify_view_pixels(
                    left,
                    np.asarray(left_view["joints_2d"], dtype=np.float64),
                    rectification.R1,
                    rectification.P1,
                )
            else:
                pixels_left = np.full((21, 2), np.nan, dtype=np.float64)
                projectable_left = np.zeros(21, dtype=bool)
            if right_view is not None and int(right_view.get("inlier_joint_count", 0)) > 0:
                pixels_right, projectable_right = _rectify_view_pixels(
                    right,
                    np.asarray(right_view["joints_2d"], dtype=np.float64),
                    rectification.R2,
                    rectification.P2,
                )
            else:
                pixels_right = np.full((21, 2), np.nan, dtype=np.float64)
                projectable_right = np.zeros(21, dtype=bool)
            in_left = (
                projectable_left
                & (pixels_left[:, 0] >= 0) & (pixels_left[:, 0] < width)
                & (pixels_left[:, 1] >= 0) & (pixels_left[:, 1] < height)
            )
            in_right = (
                projectable_right
                & (pixels_right[:, 0] >= 0) & (pixels_right[:, 0] < width)
                & (pixels_right[:, 1] >= 0) & (pixels_right[:, 1] < height)
            )
            positions[side, pair] = points_left.astype(np.float32)
            valid[side, pair] = finite
            observed[side, pair] = finite
            confidence[side, pair] = np.where(
                finite, np.clip(support / max(len(camera_ids), 2), 0.1, 1.0), 0.0
            ).astype(np.float32)
            left_px[side, pair] = pixels_left.astype(np.float32)
            right_px[side, pair] = pixels_right.astype(np.float32)
            left_px_valid[side, pair] = in_left
            right_px_valid[side, pair] = in_right

    input_positions = positions.copy()
    input_observed = observed.copy()
    base_confidence = confidence.copy()
    rejected = np.zeros_like(valid)
    interpolated = np.zeros_like(valid)
    left_px_filtered = left_px.copy()
    right_px_filtered = right_px.copy()
    if args.anatomy_refine:
        ranges = active_ranges(observed)
        accepted, rejected = reject_observation_outliers(
            positions,
            observed,
            ranges,
            args.outlier_window,
            args.outlier_distance_m,
            args.max_hand_radius_m,
        )
        left_outliers = detect_temporal_pixel_outliers(
            left_px,
            args.outlier_window,
            0.45,
            1.8,
        )
        right_outliers = detect_temporal_pixel_outliers(
            right_px,
            args.outlier_window,
            0.45,
            1.8,
        )
        left_px_valid &= ~left_outliers
        right_px_valid &= ~right_outliers
        pixel_geometry_rejected = observed & left_outliers & right_outliers
        accepted[pixel_geometry_rejected] = False
        rejected |= pixel_geometry_rejected
        left_px_filtered = smooth_pixel_landmarks_in_palm_frame(
            left_px,
            left_px_valid,
            args.smoothing_radius,
            args.local_shape_strength,
        )
        right_px_filtered = smooth_pixel_landmarks_in_palm_frame(
            right_px,
            right_px_valid,
            args.smoothing_radius,
            args.local_shape_strength,
        )
        preliminary = positions.copy()
        preliminary[~accepted] = np.nan
        preliminary_confidence = confidence.copy()
        preliminary_confidence[~accepted] = 0.0
        bone_lengths = estimate_bone_lengths(
            preliminary, accepted, preliminary_confidence
        )
        accepted, bone_rejected = reject_bone_outliers(
            positions,
            accepted,
            preliminary_confidence,
            bone_lengths,
            args.bone_outlier_absolute_m,
            args.bone_outlier_relative,
        )
        rejected |= bone_rejected
        prepared = stabilize_once(
            positions,
            accepted,
            confidence,
            args.max_gap,
            args.smoothing_radius,
            args.bone_iterations,
            args.bone_strength,
        )
        positions = smooth_3d_landmarks_in_palm_frame(
            prepared["stabilized"],
            prepared["valid"],
            args.smoothing_radius,
            args.local_shape_strength,
        ).astype(np.float32)
        valid = prepared["valid"]
        observed = accepted
        interpolated = prepared["interpolated"]
        fit_confidence = prepared["confidence"]
        pixel_confidence = np.where(
            left_px_valid | right_px_valid,
            0.25 * base_confidence,
            0.0,
        )
        confidence = np.maximum(fit_confidence, pixel_confidence).astype(np.float32)
    else:
        bone_lengths = estimate_bone_lengths(positions, observed, confidence)

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        positions_left_camera_m=positions,
        valid=valid,
        observed=observed,
        input_observed=input_observed,
        outlier_rejected=rejected,
        interpolated=interpolated,
        confidence=confidence,
        raw_positions_left_camera_m=input_positions,
        left_rectified_px=left_px,
        right_rectified_px=right_px,
        left_rectified_px_filtered=left_px_filtered,
        right_rectified_px_filtered=right_px_filtered,
        left_rectified_valid=left_px_valid,
        right_rectified_valid=right_px_valid,
        track_ids=np.asarray([0, 1], dtype=np.int32),
        handedness=np.asarray(["Left", "Right"]),
        bone_lengths_m=bone_lengths,
        skeleton_edges=np.asarray(HAND_EDGES, dtype=np.int32),
        pair_indices=np.arange(pair_count, dtype=np.int32),
        left_frame_indices=np.arange(pair_count, dtype=np.int32),
        right_frame_indices=np.arange(pair_count, dtype=np.int32),
        fps=np.asarray(_fps(dataset / "multiview_frames.csv"), dtype=np.float32),
        left_to_rectified_rotation=rectification.R1.astype(np.float32),
        projection_left_rectified=rectification.P1.astype(np.float32),
        projection_right_rectified=rectification.P2.astype(np.float32),
        rectified_size=np.asarray(rectification.image_size, dtype=np.int32),
    )
    np.savez_compressed(
        rectification_output,
        R1=rectification.R1.astype(np.float32),
        R2=rectification.R2.astype(np.float32),
        P1=rectification.P1.astype(np.float32),
        P2=rectification.P2.astype(np.float32),
        map_left_x=rectification.map_left_x,
        map_left_y=rectification.map_left_y,
        map_right_x=rectification.map_right_x,
        map_right_y=rectification.map_right_y,
        image_size=np.asarray(rectification.image_size, dtype=np.int32),
        left_camera=np.asarray(args.left_camera),
        right_camera=np.asarray(args.right_camera),
    )
    summary = {
        "schema_version": 1,
        "stage": "six_view_fusion_to_mano_input",
        "fusion": str(fusion),
        "dataset": str(dataset),
        "pair_count": pair_count,
        "accepted_frame_count": len(rows),
        "accepted_hand_count": len(populated),
        "input_valid_joint_count": int(input_observed.sum()),
        "valid_joint_count": int(valid.sum()),
        "accepted_observation_count": int(observed.sum()),
        "outlier_rejected_count": int(rejected.sum()),
        "interpolated_joint_count": int(interpolated.sum()),
        "strict_identity_observation_count": identity_observations,
        "strict_identity_mismatch_count": identity_mismatches,
        "handedness_policy": args.handedness_policy,
        "geometric_handedness_allowed": allow_geometric_handedness,
        "left_camera": args.left_camera,
        "right_camera": args.right_camera,
        "rectified_image_size": list(rectification.image_size),
        "rectified_focal_px": float(rectification.P1[0, 0]),
        "coordinate_conversion": "X_left = R_base_left.T @ (X_base - t_base_left)",
        "anatomy_refinement": {
            "enabled": bool(args.anatomy_refine),
            "outlier_window": args.outlier_window,
            "outlier_distance_m": args.outlier_distance_m,
            "max_hand_radius_m": args.max_hand_radius_m,
            "bone_outlier_absolute_m": args.bone_outlier_absolute_m,
            "bone_outlier_relative": args.bone_outlier_relative,
            "max_gap": args.max_gap,
            "smoothing_radius": args.smoothing_radius,
            "local_shape_strength": args.local_shape_strength,
            "bone_iterations": args.bone_iterations,
            "bone_strength": args.bone_strength,
            "bone_error_before": bone_error_metric(
                input_positions, input_observed, bone_lengths
            ),
            "bone_error_after": bone_error_metric(positions, valid, bone_lengths),
            "acceleration_median_before_mm": acceleration_metric(
                input_positions, input_observed
            ),
            "acceleration_median_after_mm": acceleration_metric(positions, valid),
            "palm_normalized_step_before": palm_normalized_3d_step_metric(
                input_positions, input_observed
            ),
            "palm_normalized_step_after": palm_normalized_3d_step_metric(
                positions, valid
            ),
            "accepted_observation_displacement": displacement_metric(
                positions, input_positions, observed
            ),
        },
    }
    output.with_suffix(".json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
