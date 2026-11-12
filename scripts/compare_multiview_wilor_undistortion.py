#!/usr/bin/env python3
"""Compare matched raw/undistorted experiments without claiming ground-truth accuracy."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from camera_models import project_points, unproject_points
from ego_data.calibration import CameraCalibration
from ego_data.dataset import read_csv


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _rows(path: Path) -> dict[int, dict]:
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            key = int(row["sync_index"])
            if key in result:
                raise ValueError(f"duplicate sync_index {key}: {path}")
            result[key] = row
    return result


def _stats(values) -> dict:
    values = np.asarray(list(values), dtype=float)
    return {"count": len(values), "mean": float(np.mean(values)) if len(values) else None,
            "median": float(np.median(values)) if len(values) else None,
            "p95": float(np.percentile(values, 95)) if len(values) else None}


def _fusion_directory(root: Path, config: dict, *, require_complete: bool = True) -> Path:
    if config.get("sam3_glove_assist", {}).get("enabled"):
        name = "fusion_multiview_sam3_assisted"
    elif config.get("glove_marker_assist", {}).get("enabled"):
        name = "fusion_multiview_glove_marker_assisted"
    elif config.get("detector_handedness", "strict") != "strict":
        name = "fusion_multiview_geometric"
    else:
        name = "fusion_multiview"
    path = root / name
    if require_complete and not (path / "summary.json").is_file():
        raise ValueError(f"complete fusion is required: {path}")
    return path


def _inside(pixels: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return (np.isfinite(pixels).all(axis=-1)
            & (pixels[:, 0] >= 0) & (pixels[:, 0] <= size[0] - 1)
            & (pixels[:, 1] >= 0) & (pixels[:, 1] <= size[1] - 1))


def original_pixel_residuals(hand: dict, cameras: dict, raw_cameras: dict) -> dict:
    """Map selected observations and final 3-D projections to the same raw pixels."""
    points = np.asarray(hand.get("joints_base_m", []), dtype=float)
    if points.shape != (21, 3):
        raise ValueError("fusion hand must contain 21 joints_base_m")
    residuals = {}
    for camera_id, view in hand.get("views", {}).items():
        camera, raw = cameras[camera_id], raw_cameras[camera_id]
        observed = np.asarray(view["joints_2d"], dtype=float)
        if observed.shape != (21, 2):
            raise ValueError("fusion view must contain 21 joints_2d")
        rays, ray_valid = unproject_points(camera, observed)
        # Both datasets use the same optical frame; the comparison validates this.
        raw_observed, observation_valid = project_points(raw, rays)
        camera_points = (points - raw.T_base_camera[:3, 3]) @ raw.T_base_camera[:3, :3]
        raw_projected, projection_valid = project_points(raw, camera_points)
        valid = (ray_valid & observation_valid & projection_valid
                 & _inside(observed, camera.image_size) & _inside(raw_observed, raw.image_size)
                 & np.isfinite(raw_projected).all(axis=1))
        errors = np.linalg.norm(raw_projected - raw_observed, axis=1)
        for joint in np.flatnonzero(valid):
            residuals[(camera_id, int(joint))] = float(errors[joint])
    return residuals


def _measure(root: Path, config: dict, dataset: Path, sync_rows: list[dict], raw_cameras: dict) -> tuple[dict, dict]:
    cameras = {name: CameraCalibration.load(dataset / "calibration" / f"{name}.json")
               for name in raw_cameras}
    expected = {int(row["sync_index"]): row for row in sync_rows}
    detections = {}
    if not (root / "wilor_multiview" / "summary.json").is_file():
        raise ValueError(f"inference is incomplete: {root}")
    for camera in cameras:
        predicted = _rows(root / "wilor_multiview" / camera / "predictions.jsonl")
        if predicted.keys() != expected.keys():
            raise ValueError(f"{root}/{camera}: inference does not cover the synchronized frames")
        counts = []
        for index, row in predicted.items():
            sync = expected[index]
            if (int(row["timestamp_ns"]) != int(sync[f"{camera}_timestamp_ns"])
                    or int(row["source_frame_index"]) != int(sync[f"{camera}_frame_index"])):
                raise ValueError(f"{root}/{camera}: inference timestamp/frame mismatch")
            # Each detector box has two handedness hypotheses; count it only once.
            counts.append(len({hand["detection_index"] for hand in row.get("hands", [])}))
        detections[camera] = {"frames_with_detections": sum(count > 0 for count in counts),
                              "frame_detection_rate": sum(count > 0 for count in counts) / len(counts),
                              "physical_box_count": sum(counts)}
    accepted = _rows(_fusion_directory(root, config) / "accepted.jsonl")
    if not accepted.keys() <= expected.keys():
        raise ValueError("fusion contains unexpected sync indices")
    residuals, joint_count, hand_count, complete_hand_count, valid_frames = {}, 0, 0, 0, set()
    for index, row in accepted.items():
        sides = set()
        for hand in row.get("hands", []):
            side = int(hand["side"])
            if side not in (0, 1) or side in sides:
                raise ValueError("invalid/duplicate fused hand side")
            sides.add(side)
            points = np.asarray(hand.get("joints_base_m", []), dtype=float)
            if points.shape != (21, 3):
                raise ValueError("fusion hand must contain 21 joints_base_m")
            finite = int(np.isfinite(points).all(axis=1).sum())
            joint_count += finite
            hand_count += int(finite > 0)
            complete_hand_count += int(finite == 21)
            if finite:
                valid_frames.add(index)
            for (camera, joint), error in original_pixel_residuals(hand, cameras, raw_cameras).items():
                residuals[(index, side, camera, joint)] = error
    return {"synchronized_frames": len(expected), "detections": detections,
            "accepted_frames": len(accepted), "frames_with_finite_3d": len(valid_frames),
            "finite_3d_frame_rate": len(valid_frames) / len(expected),
            "finite_hands": hand_count, "complete_21_joint_hands": complete_hand_count,
            "finite_3d_joints": joint_count,
            "original_pixel_residual_all_selected_views": _stats(residuals.values())}, residuals


def compare_runs(raw_root: Path, undistorted_root: Path) -> tuple[dict, list[dict]]:
    raw_root, undistorted_root = raw_root.resolve(), undistorted_root.resolve()
    raw_config, undistorted_config = _json(raw_root / "run_config.json"), _json(undistorted_root / "run_config.json")
    if raw_config.get("image_preprocessing") or not undistorted_config.get("image_preprocessing"):
        raise ValueError("expected a raw baseline and an undistorted experiment")
    comparable = {key: value for key, value in undistorted_config.items() if key != "image_preprocessing"}
    if raw_config != comparable:
        different = sorted(key for key in set(raw_config) | set(comparable) if raw_config.get(key) != comparable.get(key))
        raise ValueError(f"run configurations differ beyond preprocessing: {different}")
    raw_data, undistorted_data = raw_root / "normalized_multiview", undistorted_root / "undistorted_multiview"
    raw_manifest, undistorted_manifest = _json(raw_data / "manifest.json"), _json(undistorted_data / "manifest.json")
    if raw_manifest.get("undistortion") or not undistorted_manifest.get("undistortion"):
        raise ValueError("dataset preprocessing does not match experiment type")
    for field in ("source", "camera_ids", "reference_camera"):
        if raw_manifest[field] != undistorted_manifest[field]:
            raise ValueError(f"dataset {field} differs")
    sync_rows = read_csv(raw_data / "multiview_frames.csv")
    if not sync_rows or sync_rows != read_csv(undistorted_data / "multiview_frames.csv"):
        raise ValueError("synchronized frame indices/timestamps differ")
    raw_cameras = {}
    for camera in raw_manifest["camera_ids"]:
        raw = CameraCalibration.load(raw_data / "calibration" / f"{camera}.json")
        saved = CameraCalibration.load(undistorted_data / "source_calibration" / f"{camera}.json")
        pinhole = CameraCalibration.load(undistorted_data / "calibration" / f"{camera}.json")
        if (raw.to_dict() != saved.to_dict() or pinhole.model != "PINHOLE"
                or not np.array_equal(raw.T_base_camera, pinhole.T_base_camera)):
            raise ValueError(f"{camera}: source calibration/optical frame differs")
        raw_cameras[camera] = raw
    raw_metrics, raw_errors = _measure(raw_root, raw_config, raw_data, sync_rows, raw_cameras)
    new_metrics, new_errors = _measure(undistorted_root, undistorted_config, undistorted_data, sync_rows, raw_cameras)
    keys = sorted(raw_errors.keys() & new_errors.keys())
    samples = [{"sync_index": key[0], "side": key[1], "camera": key[2], "joint": key[3],
                "raw_residual_px": raw_errors[key], "undistorted_residual_px": new_errors[key],
                "delta_px": new_errors[key] - raw_errors[key]} for key in keys]
    report = {
        "schema_version": 1, "raw_run": str(raw_root), "undistorted_run": str(undistorted_root),
        "preprocessing": undistorted_config["image_preprocessing"],
        "valid_pixel_fraction": {camera: info["valid_pixel_fraction"] for camera, info in
                                 undistorted_manifest["undistortion"]["cameras"].items()},
        "raw": raw_metrics, "undistorted": new_metrics,
        "paired_original_pixel_residual": {"raw": _stats(raw_errors[key] for key in keys),
                                            "undistorted": _stats(new_errors[key] for key in keys),
                                            "delta_undistorted_minus_raw": _stats(new_errors[key] - raw_errors[key] for key in keys)},
        "interpretation": [
            "Detection coverage is not precision/recall: no hand-presence labels are supplied.",
            "Residuals compare final 3D to each run's selected 2D observations in the ORIGINAL image pixels, not to ground truth.",
            "Paired samples require the same sync index, hand side, camera and joint in both runs; verify hand identities visually.",
            "All-selected-view metrics have different sample populations; read them together with coverage and paired metrics.",
            "Pixel thresholds in the shared pipeline have unchanged numeric values but different angular scales after warping.",
            "A pinhole view can crop the original fisheye FOV; masks report valid output pixels, not the fraction of source FOV retained.",
            "Lower residuals or more detections alone do not demonstrate better 3D accuracy; labeled 2D/3D ground truth is required.",
        ],
    }
    return report, samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-run", required=True, type=Path)
    parser.add_argument("--undistorted-run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="new comparison JSON (also writes a paired-sample CSV)")
    args = parser.parse_args()
    output = args.output.resolve()
    csv_path = output.with_suffix(".csv")
    if output == csv_path or output.exists() or csv_path.exists():
        parser.error("use a new .json output path")
    report, samples = compare_runs(args.raw_run, args.undistorted_run)
    output.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["sync_index", "side", "camera", "joint", "raw_residual_px", "undistorted_residual_px", "delta_px"])
        writer.writeheader()
        writer.writerows(samples)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
