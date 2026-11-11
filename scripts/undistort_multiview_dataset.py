#!/usr/bin/env python3
"""Convert synchronized KB/GEN-DS videos to independent pinhole views.

No stereo rotation is applied: camera optical axes and rig extrinsics stay
unchanged. Frame indices and nanosecond synchronization tables are preserved.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

import cv2
import numpy as np

from camera_models import project_points, unproject_points
from ego_data.calibration import CameraCalibration
from ego_data.dataset import SequentialVideoReader, read_csv


def undistortion_options(fov_deg: float, width: int = 0, height: int = 0) -> dict:
    if not np.isfinite(fov_deg) or not 0 < fov_deg < 175:
        raise ValueError("undistort FOV must be finite and in (0, 175) degrees")
    if (width == 0) != (height == 0) or min(width, height) < 0:
        raise ValueError("undistort width/height must both be positive, or both zero")
    if width and (min(width, height) < 2 or width % 2 or height % 2):
        raise ValueError("undistort width/height must be positive even numbers")
    return {
        "mode": "independent_pinhole", "version": 1, "fov_deg": float(fov_deg),
        "width": width, "height": height, "codec": "ffv1", "interpolation": "linear",
    }


def options_from_environment() -> dict | None:
    fov = os.environ.get("EGO_WILOR_UNDISTORT_FOV_DEG", "")
    if not fov:
        return None
    return undistortion_options(
        float(fov), int(os.environ.get("EGO_WILOR_UNDISTORT_WIDTH", "0")),
        int(os.environ.get("EGO_WILOR_UNDISTORT_HEIGHT", "0")),
    )


def make_pinhole_view(
    camera: CameraCalibration, fov_deg: float, output_size: tuple[int, int] | None = None,
) -> tuple[CameraCalibration, np.ndarray, np.ndarray, np.ndarray]:
    if camera.model not in ("KB", "DS"):
        raise ValueError(f"expected a raw KB/DS camera, got {camera.model}")
    width, height = output_size or camera.image_size
    undistortion_options(fov_deg, width, height)
    focal = width / (2.0 * np.tan(np.deg2rad(fov_deg) / 2.0))
    intrinsic = np.asarray([
        [focal, 0.0, (width - 1) / 2.0],
        [0.0, focal, (height - 1) / 2.0], [0.0, 0.0, 1.0],
    ])
    pinhole = replace(camera, model="PINHOLE", image_size=(width, height),
                      K=intrinsic, distortion=np.empty(0))
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    rays, _ = unproject_points(pinhole, np.stack((x, y), axis=-1))
    pixels, projectable = project_points(camera, rays)
    map_x, map_y = pixels[..., 0].astype(np.float32), pixels[..., 1].astype(np.float32)
    raw_width, raw_height = camera.image_size
    valid = (projectable & np.isfinite(pixels).all(axis=-1)
             & (map_x >= 0) & (map_x <= raw_width - 1)
             & (map_y >= 0) & (map_y <= raw_height - 1))
    if not valid.any():
        raise ValueError(f"{camera.camera_id}: no valid pixels at this FOV")
    map_x[~valid] = map_y[~valid] = -1
    return pinhole, map_x, map_y, valid


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_signature(source: Path, manifest: dict) -> dict:
    files = [source / "manifest.json", source / "multiview_frames.csv"]
    videos = {}
    for camera in manifest["camera_ids"]:
        files.extend((source / "calibration" / f"{camera}.json",
                      source / "cameras" / camera / "timestamps.csv"))
        video = source / "cameras" / camera / manifest["storage"]["video_filename"]
        stat = video.stat()
        videos[camera] = {"size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    return {"path": str(source), "sha256": {str(p.relative_to(source)): _sha256(p) for p in files},
            "videos": videos}


def undistort_dataset(source: Path, output: Path, options: dict, reuse: bool = False) -> Path:
    source, output = source.resolve(), output.resolve()
    manifest = json.loads((source / "manifest.json").read_text())
    if manifest.get("dataset_type") != "normalized_multiview":
        raise ValueError("input must be a normalized_multiview dataset")
    if manifest.get("undistortion"):
        raise ValueError("input is already undistorted")
    cameras = manifest["camera_ids"]
    if len(cameras) < 2 or len(set(cameras)) != len(cameras):
        raise ValueError("at least two distinct cameras are required")
    options = undistortion_options(options["fov_deg"], options["width"], options["height"])
    config = {"options": options, "source": _source_signature(source, manifest)}
    if output.exists():
        if not reuse:
            raise FileExistsError(f"output already exists: {output}")
        existing = json.loads((output / "manifest.json").read_text())
        if existing.get("undistortion", {}).get("config") != config:
            raise ValueError("undistortion configuration/source changed; use a new output directory")
        required = [output / "multiview_frames.csv", output / "undistortion.json"]
        for camera in cameras:
            required.extend((output / "calibration" / f"{camera}.json",
                             output / "source_calibration" / f"{camera}.json",
                             output / "cameras" / camera / "video.mkv",
                             output / "cameras" / camera / "timestamps.csv",
                             output / "cameras" / camera / "valid_mask.png"))
        if not all(p.is_file() and p.stat().st_size for p in required):
            raise ValueError("incomplete undistortion output; use a new output directory")
        print(f"[undistort] reuse {output}", flush=True)
        return output
    rows = read_csv(source / "multiview_frames.csv")
    if not rows:
        raise ValueError("no synchronized frames")
    output_size = (options["width"], options["height"]) if options["width"] else tuple(manifest["image_size"])
    # Validate before creating output or doing any video work.
    undistortion_options(options["fov_deg"], *output_size)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        (temporary / "calibration").mkdir()
        (temporary / "source_calibration").mkdir()
        camera_info = {}
        for camera in cameras:
            calibration_path = source / "calibration" / f"{camera}.json"
            calibration = CameraCalibration.load(calibration_path)
            if calibration.camera_id != camera or calibration.image_size != tuple(manifest["image_size"]):
                raise ValueError(f"{camera}: calibration ID/resolution does not match manifest")
            pinhole, map_x, map_y, valid = make_pinhole_view(calibration, options["fov_deg"], output_size)
            indices = [int(row[f"{camera}_frame_index"]) for row in rows]
            if min(indices) < 0 or any(b <= a for a, b in zip(indices, indices[1:])):
                raise ValueError(f"{camera}: frame indices must strictly increase")
            count = indices[-1] + 1
            timestamp_path = source / "cameras" / camera / "timestamps.csv"
            timestamps = read_csv(timestamp_path)
            stamps = {int(row["frame_index"]): int(row["timestamp_ns"]) for row in timestamps}
            for row in rows:
                if stamps.get(int(row[f"{camera}_frame_index"])) != int(row[f"{camera}_timestamp_ns"]):
                    raise ValueError(f"{camera}: synchronization/timestamp table mismatch")
            times = [stamps[i] for i in range(count)]
            deltas = np.diff(times)
            if len(deltas) and np.any(deltas <= 0):
                raise ValueError(f"{camera}: timestamps must increase")
            fps = 1e9 / float(np.median(deltas)) if len(deltas) else 30.0
            directory = temporary / "cameras" / camera
            directory.mkdir(parents=True)
            if not cv2.imwrite(str(directory / "valid_mask.png"), valid.astype(np.uint8) * 255):
                raise RuntimeError("cannot write validity mask")
            reader = SequentialVideoReader(source / "cameras" / camera / manifest["storage"]["video_filename"], calibration.image_size)
            writer = cv2.VideoWriter(str(directory / "video.mkv"), cv2.VideoWriter_fourcc(*"FFV1"), fps, output_size)
            try:
                if not writer.isOpened():
                    raise RuntimeError("OpenCV needs FFV1/Matroska encoding support")
                for index in range(count):
                    frame = cv2.remap(reader.read(index), map_x, map_y, cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_CONSTANT)
                    writer.write(frame)
                    if (index + 1) % 100 == 0:
                        print(f"[undistort] {camera}: {index + 1}/{count}", flush=True)
            finally:
                reader.close()
                writer.release()
            # The CSV is authoritative for timestamps; the video is sequential storage.
            with (directory / "timestamps.csv").open("w", newline="") as stream:
                csv_writer = csv.DictWriter(stream, fieldnames=list(timestamps[0]))
                csv_writer.writeheader()
                csv_writer.writerows(row for row in timestamps if int(row["frame_index"]) < count)
            pinhole.save(temporary / "calibration" / f"{camera}.json")
            shutil.copy2(calibration_path, temporary / "source_calibration" / f"{camera}.json")
            camera_info[camera] = {"source_model": calibration.model, "K": pinhole.K.tolist(),
                                   "frame_count": count, "fps": fps,
                                   "valid_pixel_fraction": float(valid.mean())}
            print(f"[undistort] {camera}: {count} frames, valid pixels {valid.mean():.1%}", flush=True)
        shutil.copy2(source / "multiview_frames.csv", temporary / "multiview_frames.csv")
        shutil.copy2(source / "manifest.json", temporary / "source_manifest.json")
        info = {"config": config, "cameras": camera_info,
                "geometry": "original optical axes and T_base_camera; PINHOLE intrinsics; no stereo rotation",
                "timing": "original frame indices and nanosecond timestamps; CSV is authoritative"}
        (temporary / "undistortion.json").write_text(json.dumps(info, indent=2) + "\n")
        converted = {**manifest, "image_size": list(output_size),
                     "storage": {"kind": "ffv1_matroska", "video_filename": "video.mkv",
                                 "lossless": True, "frame_access": "sequential"},
                     "calibration": {"available": True, "status": "generated_pinhole"},
                     "undistortion": info}
        converted.pop("decode", None)
        (temporary / "manifest.json").write_text(json.dumps(converted, indent=2) + "\n")
        temporary.rename(output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fov-deg", type=float, default=120)
    parser.add_argument("--width", type=int, default=0)
    parser.add_argument("--height", type=int, default=0)
    parser.add_argument("--reuse", action="store_true")
    parser.add_argument("--from-environment", action="store_true")
    args = parser.parse_args()
    options = options_from_environment() if args.from_environment else undistortion_options(args.fov_deg, args.width, args.height)
    if options is None:
        parser.error("missing undistortion environment options")
    undistort_dataset(args.input, args.output, options, args.reuse)


if __name__ == "__main__":
    main()
