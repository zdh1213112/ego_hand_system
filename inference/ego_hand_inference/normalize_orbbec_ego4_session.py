#!/usr/bin/env python3
"""Normalize a four-MP4 Orbbec EGO Pro export for multiview processing."""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
from pathlib import Path
import re
import shutil
import sys
import tempfile

import cv2
import numpy as np

from ego_hand_inference.ego_data.calibration import cameras_from_ego_yaml
from ego_data.pairing import pair_timestamps_ns
from ego_hand_inference.orbbec_io import read_timestamps, sha256_file, discover_calibration


# Source IDs come from calibration, rather than the order of MP4 filenames.
CAMERA_SOURCES = {
    "camera0": ("left", "cam_0"),
    "camera1": ("right", "cam_1"),
    "camera2": ("right_side", "cam_2"),
    "camera3": ("left_side", "cam_3"),
}


def session_stream_file(session: Path, stream: str, *, timestamps: bool) -> Path:
    """Find a flat export or a single-part EGO4 recording without copying it."""
    suffix = f"_camera_{stream}_pts.csv" if timestamps else f"_camera_{stream}.mp4"
    flat = sorted(session.glob(f"*{suffix}"))
    if len(flat) == 1:
        return flat[0]
    if flat:
        raise RuntimeError(f"expected one '*{suffix}' in {session}, found {len(flat)}")
    ending = "_pts.csv" if timestamps else ".mp4"
    parts = sorted(session.glob(f"*_camera_{stream}_part*{ending}"))
    if not parts:
        parts = sorted(session.glob(f"part*/**/*_camera_{stream}_part*{ending}"))
    if len(parts) != 1:
        raise RuntimeError(
            f"expected one {stream} {'timestamp file' if timestamps else 'video'} "
            f"in {session}; found {len(parts)}. Multiple parts require concatenation."
        )
    return parts[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--calibration", type=Path, help="matching Orbbec YAML or camera0..3.json directory")
    parser.add_argument("--reference-camera", choices=tuple(CAMERA_SOURCES), default="camera0")
    parser.add_argument("--max-delta-us", type=int, default=1500)
    parser.add_argument("--max-frames", type=int, default=0, help="limit synchronized rows; 0 means all")
    return parser.parse_args()


def _write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _video_info(path: Path) -> tuple[int, tuple[int, int]]:
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"cannot open video: {path}")
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        if count <= 0 or min(size) <= 0:
            raise RuntimeError(f"cannot read video frame count/size: {path}")
        return count, size
    finally:
        capture.release()


def synchronize_rows(
    timestamps: dict[str, list[int]], reference_camera: str, max_delta_ns: int,
) -> list[dict]:
    """Intersect one-to-one timestamp pairs against the reference stream."""
    reference = timestamps[reference_camera]
    matched = {
        camera: {pair.left_frame_index: pair.right_frame_index
                 for pair in pair_timestamps_ns(reference, values, max_delta_ns)}
        for camera, values in timestamps.items() if camera != reference_camera
    }
    rows = []
    for reference_index, reference_ns in enumerate(reference):
        if not all(reference_index in pairs for pairs in matched.values()):
            continue
        row = {"sync_index": len(rows), "reference_timestamp_ns": reference_ns}
        for camera, values in timestamps.items():
            index = reference_index if camera == reference_camera else matched[camera][reference_index]
            timestamp_ns = values[index]
            row[f"{camera}_frame_index"] = index
            row[f"{camera}_timestamp_ns"] = timestamp_ns
            row[f"{camera}_delta_us"] = (timestamp_ns - reference_ns) / 1000.0
        rows.append(row)
    return rows


def normalize(args: argparse.Namespace) -> Path:
    session = args.session.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not session.is_dir():
        raise FileNotFoundError(f"session does not exist: {session}")
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    if args.max_delta_us < 0 or args.max_frames < 0:
        raise ValueError("max-delta-us/max-frames must be non-negative")
    if args.reference_camera not in CAMERA_SOURCES:
        raise ValueError(f"unknown reference camera: {args.reference_camera}")

    # EGOViewer stores calibration/metadata at the recording root while the
    # selected part holds only its own video/PTS streams. Never search sibling
    # parts when a specific part directory was requested.
    metadata_root = session
    if (not list(session.glob("*_calibration_camera.yaml"))
            and re.fullmatch(r"part\d+", session.name)):
        metadata_root = session.parent
    calibration_path = getattr(args, "calibration", None) or discover_calibration(session)
    calibration_path = Path(calibration_path).expanduser().resolve()
    if calibration_path.is_dir():
        from ego_hand_inference.ego_data.calibration import load_orbbec_four_calibrations
        loaded = load_orbbec_four_calibrations(calibration_path)
        source_calibrations = {CAMERA_SOURCES[c][1]: value for c, value in loaded.items()}
        calibration_files = [calibration_path / f"{c}.json" for c in CAMERA_SOURCES]
    else:
        source_calibrations = cameras_from_ego_yaml(calibration_path)
        calibration_files = [calibration_path]
    expected_ids = {source_id for _, source_id in CAMERA_SOURCES.values()}
    if set(source_calibrations) != expected_ids:
        raise ValueError(f"expected calibration IDs {sorted(expected_ids)}, found {sorted(source_calibrations)}")
    videos = {}
    pts_files = {}
    timestamps = {}
    calibrations = {}
    for camera, (stream, source_id) in CAMERA_SOURCES.items():
        video = session_stream_file(session, stream, timestamps=False)
        pts = session_stream_file(session, stream, timestamps=True)
        values = read_timestamps(pts)
        count, size = _video_info(video)
        calibration = source_calibrations[source_id]
        if count != len(values):
            raise ValueError(f"{camera} video has {count} frames but PTS has {len(values)} rows")
        if size != calibration.image_size:
            raise ValueError(f"{camera} video size {size} differs from calibration {calibration.image_size}")
        videos[camera], pts_files[camera], timestamps[camera] = video, pts, values
        calibrations[camera] = replace(calibration, camera_id=camera)

    synchronized = synchronize_rows(timestamps, args.reference_camera, args.max_delta_us * 1000)
    if args.max_frames:
        synchronized = synchronized[:args.max_frames]
    if not synchronized:
        raise RuntimeError("no synchronized four-camera frames were found")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        calibration_dir = temporary / "calibration"
        calibration_dir.mkdir()
        for camera in CAMERA_SOURCES:
            directory = temporary / "cameras" / camera
            directory.mkdir(parents=True)
            (directory / "video.mp4").symlink_to(videos[camera])
            _write_csv(
                directory / "timestamps.csv",
                ["frame_index", "timestamp_ns", "timestamp_us"],
                [{"frame_index": index, "timestamp_ns": value, "timestamp_us": value // 1000}
                 for index, value in enumerate(timestamps[camera])],
            )
            calibrations[camera].save(calibration_dir / f"{camera}.json")

        columns = ["sync_index", "reference_timestamp_ns"]
        for camera in CAMERA_SOURCES:
            columns.extend((f"{camera}_frame_index", f"{camera}_timestamp_ns", f"{camera}_delta_us"))
        _write_csv(temporary / "multiview_frames.csv", columns, synchronized)
        metadata_files = sorted(metadata_root.glob("meta_*.json"))
        if len(metadata_files) > 1:
            raise ValueError(f"expected at most one metadata JSON in {metadata_root}")
        if metadata_files:
            shutil.copy2(metadata_files[0], temporary / "source_meta.json")
        source_files = [*videos.values(), *pts_files.values(), *calibration_files, *metadata_files]
        deltas = [abs(float(row[f"{camera}_delta_us"])) for row in synchronized
                  for camera in CAMERA_SOURCES if camera != args.reference_camera]
        manifest = {
            "schema_version": 1,
            "dataset_type": "normalized_multiview",
            "source": {
                "kind": "orbbec_ego4_export", "session": str(session),
                "files": [{"path": str(path), "size_bytes": path.stat().st_size,
                           "sha256": sha256_file(path)} for path in source_files],
            },
            "camera_ids": list(CAMERA_SOURCES),
            "reference_camera": args.reference_camera,
            "camera_sources": {camera: {"stream": stream, "calibration_id": source_id}
                               for camera, (stream, source_id) in CAMERA_SOURCES.items()},
            "image_size": list(calibrations[args.reference_camera].image_size),
            "fps": 30.0,
            "storage": {"kind": "original_mp4", "video_filename": "video.mp4",
                        "frame_access": "sequential", "source_video_links": True},
            "synchronization": {
                "frame_count": len(synchronized), "max_delta_us": args.max_delta_us,
                "abs_delta_us_median": float(np.median(deltas)),
                "abs_delta_us_p95": float(np.percentile(deltas, 95)),
                "abs_delta_us_max": float(max(deltas)),
            },
            "calibration": {"available": True, "status": "external_json" if calibration_path.is_dir() else "source_yaml",
                            "sha256": {str(path): sha256_file(path) for path in calibration_files}},
            "coordinate_system": ("external calibration reference frame; distances in metres" if calibration_path.is_dir() else "Orbbec cam_0 optical reference frame; distances in metres"),
        }
        (temporary / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        temporary.rename(output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output


def main() -> int:
    output = normalize(parse_args())
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    print(f"Normalized EGO4 dataset: {output}")
    print(f"Synchronized frames: {manifest['synchronization']['frame_count']}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
