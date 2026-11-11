#!/usr/bin/env python3
"""Prepare aligned NOKOV 21-point truth for the existing MANO sequence fitter.

Input NPZ is produced after EGO/NOKOV time and dynamic head-camera alignment:
  joints_base_m          float [2, F, 21, 3], physical left/right in cam_0 base
  valid                  bool  [2, F, 21], actual observed marker/center validity
  reference_timestamp_ns int64 [F], exactly matching multiview_frames.csv

No synthetic WiLoR prediction is used as a target. Pixel arrays are projections
of the same NOKOV 3-D points for visibility/diagnostics; use --w-2d 0 when
fitting to avoid treating these projections as independent measurements.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from camera_models import RectificationOptions, create_stereo_rectification, project_points
from ego_data.calibration import CameraCalibration, StereoCalibration


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aligned-nokov", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="output mano_input.npz")
    parser.add_argument("--rectification-output", type=Path, required=True)
    parser.add_argument("--fusion-output", type=Path, required=True,
                        help="NOKOV observation rows for the existing strict label exporter")
    parser.add_argument("--source-group", required=True,
                        help="stable recording/session ID, used to keep train and validation separate")
    parser.add_argument("--left-camera", default="camera2")
    parser.add_argument("--right-camera", default="camera3")
    parser.add_argument("--focal-scale", type=float, default=1.0)
    return parser.parse_args()


def _sync_timestamps(path: Path) -> np.ndarray:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return np.asarray(
            [int(row["reference_timestamp_ns"]) for row in csv.DictReader(stream)],
            dtype=np.int64,
        )


def _project(points: np.ndarray, rotation: np.ndarray, projection: np.ndarray,
             size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    rectified = np.einsum("ij,...j->...i", rotation, points)
    homogeneous = np.concatenate(
        [rectified, np.ones((*rectified.shape[:-1], 1), dtype=rectified.dtype)], axis=-1
    )
    projected = np.einsum("ij,...j->...i", projection, homogeneous)
    depth = projected[..., 2]
    good = np.isfinite(projected).all(axis=-1) & (depth > 1e-8)
    pixels = np.full((*points.shape[:-1], 2), np.nan, dtype=np.float32)
    np.divide(projected[..., :2], depth[..., None], out=pixels, where=good[..., None])
    width, height = size
    inside = (good & (pixels[..., 0] >= 0) & (pixels[..., 0] < width)
              & (pixels[..., 1] >= 0) & (pixels[..., 1] < height))
    return pixels, inside


def _nullable_points(points: np.ndarray) -> list:
    return [
        [float(value) for value in point] if np.isfinite(point).all() else None
        for point in points
    ]


def prepare(aligned_path: Path, dataset: Path, output: Path, rectification_output: Path,
            left_camera: str = "camera2", right_camera: str = "camera3",
            focal_scale: float = 1.0, fusion_output: Path | None = None,
            source_group: str = "") -> dict:
    aligned_path = aligned_path.expanduser().resolve()
    dataset = dataset.expanduser().resolve()
    output = output.expanduser().resolve()
    rectification_output = rectification_output.expanduser().resolve()
    if output == rectification_output or output.exists() or rectification_output.exists():
        raise FileExistsError("both output paths must be distinct and not yet exist")
    if fusion_output is not None:
        fusion_output = fusion_output.expanduser().resolve()
        if fusion_output.exists():
            raise FileExistsError(fusion_output)
    if left_camera == right_camera or focal_scale <= 0:
        raise ValueError("select two distinct cameras and a positive focal scale")
    if fusion_output is not None and not source_group.strip():
        raise ValueError("--source-group must identify this recording")
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("dataset_type") != "normalized_multiview":
        raise ValueError("expected a normalized multiview dataset")
    if not manifest.get("calibration", {}).get("available"):
        raise ValueError("camera calibration is unavailable")
    with np.load(aligned_path, allow_pickle=False) as archive:
        required = {"joints_base_m", "valid", "reference_timestamp_ns"}
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"aligned NOKOV NPZ is missing {sorted(missing)}")
        joints_base = archive["joints_base_m"].astype(np.float64)
        observed = archive["valid"].astype(bool)
        timestamps = archive["reference_timestamp_ns"].astype(np.int64)
    expected_timestamps = _sync_timestamps(dataset / "multiview_frames.csv")
    frame_count = len(expected_timestamps)
    if joints_base.shape != (2, frame_count, 21, 3):
        raise ValueError(f"joints_base_m must be [2,{frame_count},21,3]")
    if observed.shape != joints_base.shape[:-1] or timestamps.shape != (frame_count,):
        raise ValueError("valid or reference_timestamp_ns has the wrong shape")
    if not np.array_equal(timestamps, expected_timestamps):
        raise ValueError("NOKOV alignment timestamps do not match the video frame table")
    if np.any(observed & ~np.isfinite(joints_base).all(axis=-1)):
        raise ValueError("observed joints contain NaN/Inf")
    if np.any(observed & (np.linalg.norm(joints_base, axis=-1) > 10.0)):
        raise ValueError("joints_base_m appears not to be in metres")
    if not np.any(observed.sum(axis=-1) >= 12):
        raise ValueError("no hand frame has at least 12 observed NOKOV joints")
    joints_base = joints_base.copy()
    joints_base[~observed] = np.nan
    left = CameraCalibration.load(dataset / "calibration" / f"{left_camera}.json")
    right = CameraCalibration.load(dataset / "calibration" / f"{right_camera}.json")
    rectification = create_stereo_rectification(
        StereoCalibration.from_cameras(left, right),
        RectificationOptions(focal_scale=focal_scale), "auto",
    )
    left_T = left.T_base_camera
    right_T = right.T_base_camera
    positions_left = np.einsum(
        "ij,...j->...i", left_T[:3, :3].T, joints_base - left_T[:3, 3]
    ).astype(np.float32)
    positions_right = np.einsum(
        "ij,...j->...i", right_T[:3, :3].T, joints_base - right_T[:3, 3]
    )
    left_pixels, left_inside = _project(
        positions_left, rectification.R1, rectification.P1, rectification.image_size
    )
    # P2 includes the stereo baseline and projects points in the LEFT rectified
    # frame. Feeding right-camera coordinates here would apply that baseline
    # twice, corrupting the right-view diagnostics and any 2-D fitting loss.
    right_pixels, right_inside = _project(
        positions_left, rectification.R1, rectification.P2, rectification.image_size
    )
    raw_views = {}
    for camera, positions in ((left_camera, positions_left), (right_camera, positions_right)):
        calibration = left if camera == left_camera else right
        native_pixels, native_valid = project_points(calibration, positions)
        width, height = calibration.image_size
        native_valid &= observed & np.isfinite(native_pixels).all(axis=-1)
        native_valid &= ((native_pixels[..., 0] >= 0) & (native_pixels[..., 0] < width)
                         & (native_pixels[..., 1] >= 0) & (native_pixels[..., 1] < height))
        native_pixels[~native_valid] = np.nan
        raw_views[camera] = (native_pixels, native_valid)
    left_inside &= observed
    right_inside &= observed
    confidence = observed.astype(np.float32)
    fps = (1e9 / float(np.median(np.diff(timestamps)))
           if frame_count > 1 else float(manifest.get("fps", 30.0)))
    output.parent.mkdir(parents=True, exist_ok=True)
    rectification_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        positions_left_camera_m=positions_left,
        valid=observed,
        observed=observed,
        confidence=confidence,
        left_rectified_px=left_pixels,
        right_rectified_px=right_pixels,
        left_rectified_valid=left_inside,
        right_rectified_valid=right_inside,
        track_ids=np.asarray([0, 1], dtype=np.int32),
        handedness=np.asarray(["Left", "Right"]),
        pair_indices=np.arange(frame_count, dtype=np.int32),
        fps=np.asarray(fps, dtype=np.float32),
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
        left_camera=np.asarray(left_camera),
        right_camera=np.asarray(right_camera),
    )
    summary = {
        "stage": "aligned_nokov21_to_mano_input",
        "aligned_nokov": str(aligned_path),
        "dataset": str(dataset),
        "frame_count": frame_count,
        "observed_joints_by_side": observed.sum(axis=(1, 2)).astype(int).tolist(),
        "frames_with_21_joints_by_side": observed.all(axis=-1).sum(axis=1).astype(int).tolist(),
        "rectified_visible_joints_by_camera": {
            left_camera: int(left_inside.sum()), right_camera: int(right_inside.sum()),
        },
        "pixel_target_provenance": "projection of NOKOV 3-D; not independent RGB observations",
        "fit_instruction": "run fit_mano_sequence.py with --w-2d 0",
        "source_group": source_group,
    }
    if fusion_output is not None:
        fusion_output.mkdir(parents=True)
        accepted = []
        for frame in range(frame_count):
            hands = []
            for side in (0, 1):
                if int(observed[side, frame].sum()) < 12:
                    continue
                views = {}
                for camera, (pixels, native_valid) in raw_views.items():
                    view_valid = native_valid[side, frame]
                    if not view_valid.any():
                        continue
                    views[camera] = {
                        "inlier_joint_count": int(view_valid.sum()),
                        "joints_2d": _nullable_points(pixels[side, frame]),
                        "detector_is_right": side,
                        "observation_source": "nokov_projected_marker",
                    }
                hands.append({
                    "side": side,
                    "joints_base_m": _nullable_points(joints_base[side, frame]),
                    "observed_joint_count": int(observed[side, frame].sum()),
                    "views": views,
                })
            if hands:
                accepted.append({"sync_index": frame, "hands": hands})
        (fusion_output / "accepted.jsonl").write_text(
            "".join(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n"
                    for row in accepted),
            encoding="utf-8",
        )
        (fusion_output / "summary.json").write_text(
            json.dumps({
                "stage": "nokov21_aligned_reference",
                "aligned_nokov": str(aligned_path),
                "dataset": str(dataset),
                "source_group": source_group,
                "accepted_frames": len(accepted),
                "view_count_semantics": "observed NOKOV joints projected inside native camera image",
                "detector_is_right_semantics": "NOKOV physical hand identity; no RGB detector used",
            }, indent=2) + "\n", encoding="utf-8",
        )
        summary["fusion_output"] = str(fusion_output)
        summary["accepted_frames"] = len(accepted)
    output.with_suffix(".json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> int:
    args = parse_args()
    print(json.dumps(prepare(
        args.aligned_nokov, args.dataset, args.output, args.rectification_output,
        args.left_camera, args.right_camera, args.focal_scale, args.fusion_output,
        args.source_group,
    ), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
