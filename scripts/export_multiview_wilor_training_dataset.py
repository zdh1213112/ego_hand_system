#!/usr/bin/env python3
"""Export rectified schema-4 or native KB schema-6 WiLoR training labels."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import sys
from typing import Any

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from ego_data.dataset import SequentialVideoReader
from ego_data.calibration import CameraCalibration, StereoCalibration
from wilor_label_projection import project_label_points
from wilor_frame_annotations import require_both_hands, assign_image_paths, write_frame_index
from mano_conventions import (
    MIRROR_X,
    WILOR_RIGHT_CANONICAL,
    canonical_rectification_rotation,
    mirror_left_points,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-space", choices=("physical_rectified", "native_raw"), default="physical_rectified")
    parser.add_argument("--require-both-hands", action="store_true", help="keep a camera frame only if both physical hands pass quality checks")
    parser.add_argument("--fusion", required=True, type=Path)
    parser.add_argument("--mano-fit", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--rectification", type=Path, help="required only for physical_rectified export")
    parser.add_argument("--reference-camera", default="camera0")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cameras", nargs="+", default=["camera2", "camera3"])
    parser.add_argument(
        "--view-filter", choices=("legacy", "complete21"), default="legacy",
        help=(
            "legacy accepts partially supported views; complete21 requires all "
            "21 joints to be inliers and visible in the selected camera"
        ),
    )
    parser.add_argument("--min-camera-inlier-joints", type=int, default=12)
    parser.add_argument("--max-reprojection-median-px", type=float, default=25.0)
    parser.add_argument("--max-reprojection-p95-px", type=float, default=60.0)
    parser.add_argument("--max-nokov-fit-median-mm", type=float, default=10.0,
                        help="maximum median MANO-to-NOKOV 3D joint error per frame")
    parser.add_argument("--max-nokov-fit-p95-mm", type=float, default=20.0,
                        help="maximum p95 MANO-to-NOKOV 3D joint error per frame")
    parser.add_argument("--bbox-margin", type=float, default=0.10)
    parser.add_argument("--min-visible-vertex-fraction", type=float, default=0.50)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument(
        "--sample-stride", type=int, default=0,
        help="fixed sync-frame stride; 0 uses motion-adaptive sampling",
    )
    return parser.parse_args()


def _load_fusion(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
    rows.sort(key=lambda row: int(row["sync_index"]))
    return rows


def _load_sync_rows(path: Path) -> dict[int, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return {int(row["sync_index"]): row for row in csv.DictReader(stream)}


def _load_track(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as archive:
        return {name: archive[name].copy() for name in archive.files}


def _project(vertices: np.ndarray, trans: np.ndarray, K: np.ndarray) -> np.ndarray:
    camera = vertices + trans[None]
    homogeneous = (K @ camera.T).T
    if not np.isfinite(homogeneous).all() or np.any(homogeneous[:, 2] <= 1e-8):
        raise ValueError("fitted mesh contains invalid or behind-camera vertices")
    return (homogeneous[:, :2] / homogeneous[:, 2:3]).astype(np.float32)


def _mesh_bbox(
    projected: np.ndarray, image_size: tuple[int, int], margin: float,
    min_visible_fraction: float,
) -> np.ndarray | None:
    width, height = image_size
    visible = (
        np.isfinite(projected).all(axis=1)
        & (projected[:, 0] >= 0.0) & (projected[:, 0] < width)
        & (projected[:, 1] >= 0.0) & (projected[:, 1] < height)
    )
    if float(visible.mean()) < min_visible_fraction:
        return None
    points = projected[visible]
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    size = maximum - minimum
    minimum -= margin * size
    maximum += margin * size
    minimum = np.maximum(minimum, (0.0, 0.0))
    maximum = np.minimum(maximum, (width - 1.0, height - 1.0))
    if np.any(maximum - minimum < 2.0):
        return None
    return np.asarray([minimum[0], minimum[1], maximum[0], maximum[1]], dtype=np.float64)


def _all_points_inside(points: Any, image_size: tuple[int, int]) -> bool:
    """Return whether every 2D point is finite and inside the image."""
    values = np.asarray(points, dtype=np.float64)
    if values.shape != (21, 2) or not np.isfinite(values).all():
        return False
    width, height = image_size
    return bool(
        np.all(values[:, 0] >= 0.0) and np.all(values[:, 0] < width)
        and np.all(values[:, 1] >= 0.0) and np.all(values[:, 1] < height)
    )


def _complete21_source_view(
    view: dict[str, Any], image_size: tuple[int, int],
) -> bool:
    """Require 21 real RANSAC inliers that are inside this source camera."""
    return (
        int(view.get("inlier_joint_count", 0)) == 21
        and _all_points_inside(view.get("joints_2d"), image_size)
    )


def _finite_error(values: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return float("inf"), float("inf")
    return float(np.median(finite)), float(np.percentile(finite, 95))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rectify_mano_geometry(
    local_vertices: np.ndarray,
    local_joints: np.ndarray,
    translation: np.ndarray,
    rotation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Rotate a MANO result while preserving its root-centred convention.

    MANO applies ``global_orient`` around its wrist joint rather than the
    coordinate origin.  Left-multiplying the global rotation therefore moves
    the local root by ``rotation @ root - root``.  Move that offset from the
    local geometry into camera translation so both MANO replay and camera-space
    projection remain exact.
    """
    local_vertices = np.asarray(local_vertices, dtype=np.float32)
    local_joints = np.asarray(local_joints, dtype=np.float32)
    translation = np.asarray(translation, dtype=np.float32)
    rotation = np.asarray(rotation, dtype=np.float32)
    root_offset = (rotation @ local_joints[0] - local_joints[0]).astype(np.float32)
    vertices = (
        (rotation @ local_vertices.T).T - root_offset[None]
    ).astype(np.float32)
    joints = (
        (rotation @ local_joints.T).T - root_offset[None]
    ).astype(np.float32)
    rectified_translation = (
        rotation @ translation + root_offset
    ).astype(np.float32)
    return vertices, joints, rectified_translation, root_offset


def _full_pose_matrices(
    full_pose_axis_angle: np.ndarray,
    rectification_rotation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert the complete mean-inclusive MANO pose to rectified matrices."""
    full_pose = np.asarray(full_pose_axis_angle, dtype=np.float64).reshape(16, 3)
    global_matrix = Rotation.from_rotvec(full_pose[0]).as_matrix()
    global_rectified = (
        np.asarray(rectification_rotation, dtype=np.float64) @ global_matrix
    ).astype(np.float32)[None]
    hand_pose = Rotation.from_rotvec(full_pose[1:]).as_matrix().astype(np.float32)
    return global_rectified, hand_pose


MOTION_MIN_INTERVAL_S = 0.10
MOTION_MAX_INTERVAL_S = 0.50
MOTION_TRANSLATION_THRESHOLD_M = 0.008
MOTION_POSE_THRESHOLD_M = 0.010


def _sync_time_s(sync_rows: dict[int, dict[str, str]], sync_index: int) -> float:
    """Return a stable recording time for a synchronized frame."""
    row = sync_rows.get(sync_index)
    if row is not None:
        try:
            return float(row["reference_timestamp_ns"]) * 1e-9
        except (KeyError, TypeError, ValueError):
            pass
    # Normalized recordings are nominally 30 Hz. This fallback is only used for
    # malformed/missing timestamps and keeps adaptive sampling deterministic.
    return float(sync_index) / 30.0


def _motion_state(group: list[dict[str, Any]]) -> dict[int, dict[str, np.ndarray]]:
    """Build one camera-independent motion state per physical hand."""
    states: dict[int, dict[str, np.ndarray]] = {}
    for item in group:
        side = int(item["side"])
        if side in states or "motion_points" not in item:
            continue
        points = np.asarray(item["motion_points"], dtype=np.float32)
        if points.shape != (21, 3) or not np.isfinite(points).all():
            continue
        states[side] = {
            "root": points[0].copy(),
            "relative": (points - points[0]).astype(np.float32),
        }
    return states


def _motion_change(
    previous: dict[int, dict[str, np.ndarray]],
    current: dict[int, dict[str, np.ndarray]],
) -> tuple[float, float]:
    """Return maximum root and root-aligned hand-shape changes in metres."""
    common = sorted(set(previous) & set(current))
    if not common:
        return float("inf"), float("inf")
    translation = 0.0
    pose = 0.0
    for side in common:
        translation = max(
            translation,
            float(np.linalg.norm(current[side]["root"] - previous[side]["root"])),
        )
        delta = current[side]["relative"] - previous[side]["relative"]
        pose = max(pose, float(np.linalg.norm(delta, axis=1).mean()))
    return translation, pose


def _select_sync_indices(
    pending: list[dict[str, Any]],
    sync_rows: dict[int, dict[str, str]],
    sample_stride: int,
) -> tuple[list[int], dict[str, Any]]:
    """Select synchronized frames without separating cameras or hands.

    A positive stride selects every Nth synchronized frame and also keeps the
    final frame. With stride zero, frames are selected when the fitted hand has
    moved enough, a hand appears/disappears, or a maximum quiet interval has
    elapsed. The motion state is stored in a camera-independent physical frame,
    so both training views remain paired at every selected sync index.
    """
    if sample_stride < 0:
        raise ValueError("sample-stride must be non-negative")
    groups: dict[int, list[dict[str, Any]]] = {}
    for item in pending:
        groups.setdefault(int(item["sync_index"]), []).append(item)
    sync_indices = sorted(groups)
    if not sync_indices:
        return [], {
            "mode": "fixed_stride" if sample_stride else "motion_adaptive",
            "source_sync_count": 0,
            "selected_sync_count": 0,
        }

    if sample_stride > 0:
        origin = sync_indices[0]
        selected = [
            sync_index for sync_index in sync_indices
            if (sync_index - origin) % sample_stride == 0
        ]
        if selected[-1] != sync_indices[-1]:
            selected.append(sync_indices[-1])
        return selected, {
            "mode": "fixed_stride",
            "sample_stride": int(sample_stride),
            "source_sync_count": len(sync_indices),
            "selected_sync_count": len(selected),
        }

    selected = [sync_indices[0]]
    previous_state = _motion_state(groups[sync_indices[0]])
    previous_time = _sync_time_s(sync_rows, sync_indices[0])
    hand_change_count = 0
    motion_change_count = 0
    quiet_interval_count = 0
    for sync_index in sync_indices[1:]:
        current_state = _motion_state(groups[sync_index])
        current_time = _sync_time_s(sync_rows, sync_index)
        elapsed = max(0.0, current_time - previous_time)
        hand_changed = set(current_state) != set(previous_state)
        translation_delta, pose_delta = _motion_change(previous_state, current_state)
        motion_changed = (
            translation_delta >= MOTION_TRANSLATION_THRESHOLD_M
            or pose_delta >= MOTION_POSE_THRESHOLD_M
        )
        quiet_interval = elapsed + 1e-6 >= MOTION_MAX_INTERVAL_S
        should_select = hand_changed or (
            elapsed + 1e-6 >= MOTION_MIN_INTERVAL_S and motion_changed
        ) or quiet_interval
        if not should_select:
            continue
        selected.append(sync_index)
        previous_state = current_state
        previous_time = current_time
        hand_change_count += int(hand_changed)
        motion_change_count += int(motion_changed and not hand_changed)
        quiet_interval_count += int(quiet_interval and not motion_changed and not hand_changed)

    if selected[-1] != sync_indices[-1]:
        selected.append(sync_indices[-1])
    return selected, {
        "mode": "motion_adaptive",
        "source_sync_count": len(sync_indices),
        "selected_sync_count": len(selected),
        "selected_by_hand_presence_change": hand_change_count,
        "selected_by_motion": motion_change_count,
        "selected_by_quiet_interval": quiet_interval_count,
        "min_interval_ms": MOTION_MIN_INTERVAL_S * 1000.0,
        "max_interval_ms": MOTION_MAX_INTERVAL_S * 1000.0,
        "translation_threshold_m": MOTION_TRANSLATION_THRESHOLD_M,
        "pose_threshold_m": MOTION_POSE_THRESHOLD_M,
    }


def _limit_to_sync_groups(
    pending: list[dict[str, Any]], selected_sync_indices: list[int], max_samples: int,
) -> list[dict[str, Any]]:
    """Apply the sample cap without splitting a synchronized image/label group."""
    if max_samples <= 0:
        return pending
    groups: dict[int, list[dict[str, Any]]] = {}
    for item in pending:
        groups.setdefault(int(item["sync_index"]), []).append(item)
    selected: list[dict[str, Any]] = []
    for sync_index in selected_sync_indices:
        group = groups.get(sync_index, [])
        if selected and len(selected) + len(group) > max_samples:
            break
        selected.extend(group)
    return selected


def main() -> int:
    args = parse_args()
    if args.image_space == "native_raw":
        # Backward-compatible command spelling; native export never reads the
        # optional legacy rectification argument or computes rectified geometry.
        from export_native_wilor_training_dataset import main as export_native
        forwarded = []
        for key in ("fusion", "mano_fit", "dataset", "output", "reference_camera",
                    "sample_stride", "max_samples", "bbox_margin", "min_visible_vertex_fraction",
                    "max_nokov_fit_median_mm", "max_nokov_fit_p95_mm",
                    "max_reprojection_median_px", "max_reprojection_p95_px", "jpeg_quality", "view_filter"):
            forwarded += ["--" + key.replace("_", "-"), str(getattr(args, key))]
        forwarded += ["--cameras", *args.cameras]
        if args.require_both_hands:
            forwarded += ["--require-both-hands"]
        previous = sys.argv
        try:
            sys.argv = [previous[0], *forwarded]
            export_native()
        finally:
            sys.argv = previous
        return 0
    if args.rectification is None:
        raise ValueError("physical_rectified export requires --rectification")
    if args.max_samples < 0:
        raise ValueError("max-samples must be non-negative")
    if args.sample_stride < 0:
        raise ValueError("sample-stride must be non-negative")
    if not 1 <= args.min_camera_inlier_joints <= 21:
        raise ValueError("min-camera-inlier-joints must be in [1, 21]")
    if not 0.0 <= args.bbox_margin <= 1.0:
        raise ValueError("bbox-margin must be in [0, 1]")
    if not 0.0 < args.min_visible_vertex_fraction <= 1.0:
        raise ValueError("min-visible-vertex-fraction must be in (0, 1]")
    if not 1 <= args.jpeg_quality <= 100:
        raise ValueError("jpeg-quality must be in [1, 100]")

    fusion = args.fusion.resolve()
    fit_root = args.mano_fit.resolve()
    dataset = args.dataset.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    fit_summary_path = fit_root / "summary.json"
    if not fit_summary_path.is_file():
        raise FileNotFoundError(fit_summary_path)
    fit_summary = json.loads(fit_summary_path.read_text(encoding="utf-8"))
    fit_target = fit_summary.get("fit_target", "mano_joints")
    if fit_target != "mano_joints":
        raise ValueError("this exporter uses joint-to-target gates; surface-marker fits need separate validated observation gates")
    rgb_parameters = fit_summary.get("parameters", {})
    rgb_refined = float(rgb_parameters.get("w_silhouette", 0.0)) > 0

    if fit_summary.get("mano_convention") != WILOR_RIGHT_CANONICAL:
        raise ValueError(
            f"MANO fit must use {WILOR_RIGHT_CANONICAL}, got "
            f"{fit_summary.get('mano_convention')!r}"
        )
    expected_model_mapping = {
        "left": "MANO_RIGHT.pkl", "right": "MANO_RIGHT.pkl",
    }
    if fit_summary.get("mano_model_by_side") != expected_model_mapping:
        raise ValueError(
            f"MANO fit model mapping must be {expected_model_mapping}, got "
            f"{fit_summary.get('mano_model_by_side')!r}"
        )
    model_dir = Path(fit_summary["model_dir"]).expanduser().resolve()
    mano_assets = {}
    for name in ("MANO_RIGHT.pkl",):
        path = model_dir / name
        if not path.is_file():
            raise FileNotFoundError(path)
        mano_assets[name] = {
            "path": str(path),
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
        }
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("dataset_type") != "normalized_multiview":
        raise ValueError(f"not a normalized multiview dataset: {dataset}")
    camera_ids = tuple(manifest["camera_ids"])
    cameras = tuple(dict.fromkeys(args.cameras))
    if any(camera not in camera_ids for camera in cameras):
        raise ValueError("an export camera is not present in the multiview dataset")

    with np.load(args.rectification.resolve()) as archive:
        left_camera = str(archive["left_camera"].item())
        right_camera = str(archive["right_camera"].item())
        rectification = {
            "R1": archive["R1"].astype(np.float32),
            "R2": archive["R2"].astype(np.float32),
            "P1": archive["P1"].astype(np.float32),
            "P2": archive["P2"].astype(np.float32),
            "map_left_x": archive["map_left_x"].astype(np.float32),
            "map_left_y": archive["map_left_y"].astype(np.float32),
            "map_right_x": archive["map_right_x"].astype(np.float32),
            "map_right_y": archive["map_right_y"].astype(np.float32),
            "image_size": tuple(int(value) for value in archive["image_size"]),
        }
    if any(camera not in (left_camera, right_camera) for camera in cameras):
        raise ValueError(
            f"rectification supports only {left_camera}/{right_camera}, got {cameras}"
        )
    native = args.image_space == "native_raw"
    calibrations = {c: CameraCalibration.load(dataset / "calibration" / f"{c}.json") for c in cameras} if native else {}
    if native and any(c.model != "KB" for c in calibrations.values()):
        raise ValueError("native_raw export currently requires KB cameras")
    export_size = tuple(manifest["image_size"]) if native else rectification["image_size"]
    K = rectification["P1"][:, :3].copy().astype(np.float32)
    right_offset = np.linalg.solve(K, rectification["P2"][:, 3]).astype(np.float32)
    if native:
        left_cal = CameraCalibration.load(dataset / "calibration" / f"{left_camera}.json")
        right_cal = CameraCalibration.load(dataset / "calibration" / f"{right_camera}.json")
        stereo = StereoCalibration.from_cameras(left_cal, right_cal)
        np.testing.assert_allclose(rectification["R2"] @ stereo.R_left_to_right,
                                   rectification["R1"], atol=2e-5)
        np.testing.assert_allclose(rectification["R2"] @ stereo.t_left_to_right_m,
                                   right_offset, atol=2e-5)
        if any(cal.image_size != export_size for cal in calibrations.values()):
            raise ValueError("native camera size disagrees with source video")
    canonical_rotation = {
        side: canonical_rectification_rotation(rectification["R1"], side)
        for side in (0, 1)
    }
    canonical_right_offset = {
        0: (MIRROR_X @ right_offset).astype(np.float32),
        1: right_offset.copy(),
    }
    sync_rows = _load_sync_rows(dataset / "multiview_frames.csv")
    fusion_rows = _load_fusion(fusion / "accepted.jsonl")
    fusion_summary = json.loads((fusion / "summary.json").read_text(encoding="utf-8"))
    nokov_reference = fusion_summary.get("stage") == "nokov21_aligned_reference"
    if nokov_reference and args.view_filter != "complete21":
        raise ValueError("NOKOV truth export requires --view-filter complete21")
    tracks = {}
    track_lookup = {}
    required_sides = {
        int(hand["side"]) for row in fusion_rows for hand in row["hands"]
    }
    for side in (0, 1):
        path = fit_root / f"track_{side}.npz"
        if not path.is_file():
            if nokov_reference and side not in required_sides:
                continue
            raise FileNotFoundError(path)
        tracks[side] = _load_track(path)
        track_convention = (
            str(tracks[side]["mano_convention"])
            if "mano_convention" in tracks[side] else None
        )
        if track_convention != WILOR_RIGHT_CANONICAL:
            raise ValueError(
                f"{path} uses {track_convention!r}, expected {WILOR_RIGHT_CANONICAL!r}"
            )
        track_lookup[side] = {
            int(pair): index for index, pair in enumerate(tracks[side]["pair_indices"])
        }

    camera_order = {left_camera: 0, right_camera: 1}
    pending: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for row in fusion_rows:
        sync_index = int(row["sync_index"])
        if sync_index not in sync_rows:
            raise ValueError(f"missing synchronization row {sync_index}")
        for hand in sorted(row["hands"], key=lambda value: int(value["side"])):
            side = int(hand["side"])
            frame_index = track_lookup[side].get(sync_index)
            if frame_index is None:
                rejected.append({
                    "sync_index": sync_index, "side": side, "reason": "missing_mano_fit_frame",
                })
                continue
            track = tracks[side]
            if not bool(track["render_valid"][frame_index]):
                rejected.append({
                    "sync_index": sync_index, "side": side, "reason": "mano_fit_not_render_valid",
                })
                continue
            fit_joint_median_mm = float("nan")
            fit_joint_p95_mm = float("nan")
            if nokov_reference:
                joint_errors = np.asarray(track["joint_error_m"][frame_index], dtype=np.float64)
                joint_errors = joint_errors[np.isfinite(joint_errors)] * 1000.0
                if len(joint_errors) != 21:
                    rejected.append({
                        "sync_index": sync_index, "side": side,
                        "reason": "incomplete_nokov_joint_fit",
                    })
                    continue
                fit_joint_median_mm = float(np.median(joint_errors))
                fit_joint_p95_mm = float(np.percentile(joint_errors, 95))
                if (fit_joint_median_mm > args.max_nokov_fit_median_mm
                        or fit_joint_p95_mm > args.max_nokov_fit_p95_mm):
                    rejected.append({
                        "sync_index": sync_index, "side": side,
                        "reason": "mano_nokov_3d_fit_error",
                        "median_mm": fit_joint_median_mm,
                        "p95_mm": fit_joint_p95_mm,
                    })
                    continue
            translation = track["translation"][frame_index].astype(np.float32)
            local_vertices = (
                track["vertices"][frame_index].astype(np.float32) - translation[None]
            )
            local_joints = (
                track["joints"][frame_index].astype(np.float32) - translation[None]
            )
            if "full_pose_axis_angle" not in track:
                raise ValueError(
                    f"track {side} is missing mean-inclusive full_pose_axis_angle"
                )
            local_rectified, joints_rectified, trans_left, root_offset = (
                _rectify_mano_geometry(
                    local_vertices,
                    local_joints,
                    translation,
                    canonical_rotation[side],
                )
            )
            global_rectified, hand_pose = _full_pose_matrices(
                track["full_pose_axis_angle"][frame_index],
                canonical_rotation[side],
            )
            physical_vertices = mirror_left_points(
                local_rectified, side
            ).astype(np.float32)
            physical_joints = mirror_left_points(
                joints_rectified, side
            ).astype(np.float32)
            motion_translation = mirror_left_points(
                trans_left, side
            ).astype(np.float32)
            motion_points = (
                physical_joints + motion_translation[None]
            ).astype(np.float32)

            for camera in cameras:
                view = hand.get("views", {}).get(camera)
                inlier_joint_count = (
                    0 if view is None else int(view.get("inlier_joint_count", 0))
                )
                if view is None or inlier_joint_count < args.min_camera_inlier_joints:
                    rejected.append({
                        "sync_index": sync_index, "side": side, "camera": camera,
                        "reason": "too_few_camera_inlier_joints",
                        "inlier_joint_count": inlier_joint_count,
                    })
                    continue
                if args.view_filter == "complete21" and not _complete21_source_view(
                    view, tuple(manifest["image_size"])
                ):
                    rejected.append({
                        "sync_index": sync_index, "side": side, "camera": camera,
                        "reason": "camera_view_not_complete21",
                        "inlier_joint_count": inlier_joint_count,
                    })
                    continue
                error_key = (
                    "left_reprojection_error_px" if camera == left_camera
                    else "right_reprojection_error_px"
                )
                median_error, p95_error = _finite_error(track[error_key][frame_index])
                if (median_error > args.max_reprojection_median_px
                        or p95_error > args.max_reprojection_p95_px):
                    rejected.append({
                        "sync_index": sync_index, "side": side, "camera": camera,
                        "reason": "mano_fit_reprojection_error",
                        "median_px": median_error, "p95_px": p95_error,
                    })
                    continue
                canonical_trans = trans_left.copy()
                if camera == right_camera:
                    canonical_trans += canonical_right_offset[side]
                trans = mirror_left_points(
                    canonical_trans, side
                ).astype(np.float32)
                sample_K = K.copy()
                sample_vertices = physical_vertices.copy()
                sample_joints = physical_joints.copy()
                sample_global = global_rectified.copy()
                native_info = {}
                if native:
                    camera_cal = calibrations[camera]
                    inverse = rectification["R1" if camera == left_camera else "R2"].T
                    canonical_inverse = canonical_rectification_rotation(inverse, side)
                    nv, nj, nt, native_root_offset = _rectify_mano_geometry(
                        mirror_left_points(sample_vertices, side),
                        mirror_left_points(sample_joints, side),
                        mirror_left_points(trans, side), canonical_inverse,
                    )
                    sample_vertices = mirror_left_points(nv, side)
                    sample_joints = mirror_left_points(nj, side)
                    trans = mirror_left_points(nt, side)
                    sample_global = (canonical_inverse @ sample_global[0])[None].astype(np.float32)
                    sample_K = camera_cal.K.astype(np.float32)
                    native_info = {"image_space": "native_raw", "camera_model": "KB",
                                   "distortion": camera_cal.distortion.astype(np.float64)}
                projection_info = {"K": sample_K, **native_info}
                try:
                    projected = project_label_points(sample_vertices + trans[None], projection_info).astype(np.float32)
                    projected_joints = project_label_points(sample_joints + trans[None], projection_info).astype(np.float32)
                except ValueError as error:
                    rejected.append({
                        "sync_index": sync_index, "side": side, "camera": camera,
                        "reason": "mesh_projection_failed", "detail": str(error),
                    })
                    continue
                if args.view_filter == "complete21" and not _all_points_inside(
                    projected_joints, export_size
                ):
                    rejected.append({
                        "sync_index": sync_index, "side": side, "camera": camera,
                        "reason": "exported_mano_joints_not_complete21",
                        "inlier_joint_count": inlier_joint_count,
                    })
                    continue
                bbox = _mesh_bbox(
                    projected, export_size, args.bbox_margin,
                    args.min_visible_vertex_fraction,
                )
                if bbox is None:
                    rejected.append({
                        "sync_index": sync_index, "side": side, "camera": camera,
                        "reason": "mesh_not_sufficiently_visible",
                    })
                    continue
                sample = {
                    "bbox": bbox,
                    "vertices": sample_vertices.copy(),
                    "joints_3d": sample_joints.copy(),
                    "joints_2d_array": projected,
                    "side": np.asarray(float(side), dtype=np.float32),
                    "trans": trans,
                    "K": sample_K.copy(),
                    "mano": {
                        "global_orient": sample_global.copy(),
                        "hand_pose": hand_pose.copy(),
                        "betas": track["betas"].astype(np.float32).copy(),
                    },
                }
                sample.update(native_info)
                pending.append({
                    "sync_index": sync_index,
                    "side": side,
                    "camera": camera,
                    "sample": sample,
                    # Camera-independent physical coordinates used only for
                    # motion-adaptive sampling. The exported label keeps the
                    # camera-specific translation above.
                    "motion_points": motion_points,
                    "metadata": {
                        "sync_index": sync_index,
                        "camera": camera,
                        "side": side,
                        "handedness": "right" if side else "left",
                        "physical_side": side,
                        "stored_image_space": args.image_space,
                        "fit_reprojection_space": "physical_rectified",
                        "stored_image_horizontally_flipped": False,
                        "mano_parameter_space": "right_canonical",
                        "view_filter": args.view_filter,
                        "camera_inlier_joint_count": inlier_joint_count,
                        "fit_reprojection_median_px": median_error,
                        "fit_reprojection_p95_px": p95_error,
                        "fit_nokov_joint_median_mm": fit_joint_median_mm if nokov_reference else None,
                        "fit_nokov_joint_p95_mm": fit_joint_p95_mm if nokov_reference else None,
                        "fusion_mode": hand.get("fusion_mode"),
                        "mano_root_offset_m": root_offset.tolist(),
                        "native_root_offset_m": native_root_offset.tolist() if native else None,
                    },
                })

    if args.require_both_hands:
        pending, incomplete_frames = require_both_hands(pending)
        rejected.extend(incomplete_frames)
    pending.sort(
        key=lambda item: (item["sync_index"], item["side"], camera_order[item["camera"]])
    )
    selected_sync_indices, sampling = _select_sync_indices(
        pending, sync_rows, args.sample_stride
    )
    selected_sync_set = set(selected_sync_indices)
    pending = [item for item in pending if item["sync_index"] in selected_sync_set]
    pending = _limit_to_sync_groups(
        pending, selected_sync_indices, args.max_samples
    )
    if not pending:
        raise RuntimeError("no samples passed training export quality checks")
    sampling = dict(sampling)
    sampling.update({
        "exported_sync_count": len({int(item["sync_index"]) for item in pending}),
        "exported_sample_count": len(pending),
    })
    assign_image_paths(pending, shared=native)
    for index, item in enumerate(pending):
        stem = f"{index:06d}"
        item["stem"] = stem
        item["metadata"].update({
            "index": index,
            "image": f"images/{item['image_stem']}.jpg",
            "label": f"labels/{stem}.npy",
            "sampling_mode": sampling["mode"],
            "sample_stride": args.sample_stride,
        })
        if nokov_reference:
            item["metadata"]["source_group"] = fusion_summary["source_group"]
            item["metadata"]["supervision_source"] = (
                "nokov21_rgb_silhouette_mano_fit" if rgb_refined else "nokov21_mano_fit"
            )
            item["metadata"]["camera_support_semantics"] = (
                "observed NOKOV joints projected inside native camera image"
            )

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        image_root = temporary / "images"
        label_root = temporary / "labels"
        image_root.mkdir()
        label_root.mkdir()
        import torch

        for item in pending:
            stored = item["sample"]
            sample = {
                "bbox": stored["bbox"],
                "vertices": stored["vertices"],
                "joints_3d": stored["joints_3d"],
                "joints_2d": torch.from_numpy(stored["joints_2d_array"]),
                "side": stored["side"],
                "trans": stored["trans"],
                "K": stored["K"],
                "mano": stored["mano"],
            }
            if native:
                sample.update({key: stored[key] for key in ("image_space", "camera_model", "distortion")})
            np.save(label_root / f"{item['stem']}.npy", sample, allow_pickle=True)

        image_size = tuple(manifest["image_size"])
        for camera in cameras:
            selected = [item for item in pending if item["camera"] == camera]
            if not selected:
                continue
            reader = SequentialVideoReader(
                dataset / "cameras" / camera / manifest["storage"]["video_filename"],
                image_size,
            )
            try:
                grouped: dict[int, list[dict[str, Any]]] = {}
                for item in selected:
                    grouped.setdefault(item["sync_index"], []).append(item)
                for sync_index in sorted(grouped):
                    source_index = int(sync_rows[sync_index][f"{camera}_frame_index"])
                    frame = reader.read(source_index)
                    if camera == left_camera:
                        map_x = rectification["map_left_x"]
                        map_y = rectification["map_left_y"]
                    else:
                        map_x = rectification["map_right_x"]
                        map_y = rectification["map_right_y"]
                    rectified = frame if native else cv2.remap(
                        frame, map_x, map_y, cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT,
                    )
                    written_images = set()
                    for item in grouped[sync_index]:
                        image_path = image_root / f"{item['image_stem']}.jpg"
                        if image_path not in written_images and not cv2.imwrite(
                            str(image_path), frame if native else rectified,
                            [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality],
                        ):
                            raise RuntimeError(f"cannot write image: {image_path}")
                        written_images.add(image_path)
                        item["metadata"]["source_frame_index"] = source_index
                        item["metadata"]["reference_timestamp_ns"] = int(sync_rows[sync_index]["reference_timestamp_ns"])
            finally:
                reader.close()

        write_frame_index(temporary / "frames.jsonl", [item["metadata"] for item in pending])
        (temporary / "index.jsonl").write_text(
            "".join(
                json.dumps(item["metadata"], ensure_ascii=False, separators=(",", ":")) + "\n"
                for item in pending
            ),
            encoding="utf-8",
        )
        (temporary / "rejected.jsonl").write_text(
            "".join(
                json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                for row in rejected
            ),
            encoding="utf-8",
        )
        side_counts = {
            name: sum(item["side"] == side for item in pending)
            for side, name in ((0, "left"), (1, "right"))
        }
        camera_counts = {
            camera: sum(item["camera"] == camera for item in pending) for camera in cameras
        }
        summary = {
            # Schema 6 adds explicit native camera projection metadata.
            # Legacy rectified exports retain their schema-4 contract.
            "schema_version": 6 if native else 4,
            "stage": "six_view_refined_wilor_training_dataset",
            "schema_reference": "000865.npy",
            "fit_target_semantics": fit_target,
            "rgb_silhouette_refinement": {
                "enabled": rgb_refined,
                "weight": float(rgb_parameters.get("w_silhouette", 0.0)),
                "source": rgb_parameters.get("silhouette_input"),
                "mask_semantics": "automatic native RGB white-glove contours" if rgb_refined else None,
            },
            "sample_count": len(pending),
            "image_layout": "shared_camera_frames" if native else "per_hand",
            "annotation_policy": "require_both_hands" if args.require_both_hands else "available_hands",
            "sampling": sampling,
            "max_samples": args.max_samples,
            "image_count": len(list(image_root.glob("*.jpg"))),
            "label_count": len(list(label_root.glob("*.npy"))),
            "side_counts": side_counts,
            "camera_counts": camera_counts,
            "rejected_count": len(rejected),
            "image_size": list(export_size),
            "K": None if native else K.tolist(),
            "camera_calibrations": {c: {**cal.to_dict(), "sha256": _sha256(dataset / "calibration" / f"{c}.json")} for c, cal in calibrations.items()},
            "source_dataset": str(dataset),
            "source_fit": str(fit_root),
            "source_rectification": str(args.rectification.resolve()),
            "projection_fields": {"joints_2d": "778 mesh vertex pixels (legacy field name)", "joints_3d": "21 physical-side model joints; add trans for camera coordinates"},
            "cameras": list(cameras),
            "view_filter": args.view_filter,
            "source": (
                "aligned NOKOV 21-point reference + shared MANO sequence fit" + (" + native RGB contour refinement" if rgb_refined else "")
                if nokov_reference else
                "strict multiview WiLoR fusion + shared MANO sequence fit"
            ),
            "nokov_reference": nokov_reference,
            "source_group": fusion_summary.get("source_group") if nokov_reference else None,
            "mano_model_by_side": {
                "left": "MANO_RIGHT.pkl",
                "right": "MANO_RIGHT.pkl",
            },
            "mano_pose_representation": "rotation_matrix_full_mean",
            "mano_model": "MANO_RIGHT.pkl",
            "image_space": args.image_space,
            "geometry_space": "physical_side",
            "mano_parameter_space": "right_canonical",
            "side_semantics": "physical_identity",
            "mano_source_revision": fit_summary.get("mano_revision", "unknown"),
            "mano_assets": mano_assets,
            "quality_thresholds": {
                "min_camera_inlier_joints": args.min_camera_inlier_joints,
                "max_reprojection_median_px": args.max_reprojection_median_px,
                "max_reprojection_p95_px": args.max_reprojection_p95_px,
                "min_visible_vertex_fraction": args.min_visible_vertex_fraction,
                "require_complete_camera_joints": args.view_filter == "complete21",
                "max_nokov_fit_median_mm": args.max_nokov_fit_median_mm if nokov_reference else None,
                "max_nokov_fit_p95_mm": args.max_nokov_fit_p95_mm if nokov_reference else None,
            },
        }
        if summary["image_count"] != len({item["image_stem"] for item in pending}) or summary["label_count"] != len(pending):
            raise RuntimeError("paired image/label count mismatch before finalizing output")
        (temporary / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        temporary.rename(output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
