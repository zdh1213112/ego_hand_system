#!/usr/bin/env python3
"""Remux and synchronize an arbitrary set of GEN cameras in one MCAP pass."""

from __future__ import annotations

import argparse
from bisect import bisect_left
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np

from ego_data.genrobot_mcap import remux_stereo_mcap


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--cameras", nargs="+", default=[f"camera{i}" for i in range(6)],
        help="GEN camera IDs to synchronize",
    )
    parser.add_argument("--reference-camera", default="camera2")
    parser.add_argument("--max-delta-us", type=int, default=1500)
    parser.add_argument("--max-frames", type=int, default=0)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def synchronize_rows(
    frame_rows: dict[str, list[dict]], camera_ids: tuple[str, ...],
    reference_camera: str, max_delta_ns: int,
) -> list[dict]:
    """Nearest-neighbour synchronization without reusing a camera frame."""
    timestamps = {
        camera: [int(row["timestamp_ns"]) for row in frame_rows[camera]]
        for camera in camera_ids
    }
    used = {camera: set() for camera in camera_ids if camera != reference_camera}
    synchronized: list[dict] = []
    for reference_index, reference_ns in enumerate(timestamps[reference_camera]):
        selected = {reference_camera: reference_index}
        for camera in camera_ids:
            if camera == reference_camera:
                continue
            values = timestamps[camera]
            position = bisect_left(values, reference_ns)
            candidates = [index for index in (position - 1, position) if 0 <= index < len(values)]
            candidates = [index for index in candidates if index not in used[camera]]
            if not candidates:
                selected = {}
                break
            best = min(candidates, key=lambda index: abs(values[index] - reference_ns))
            if abs(values[best] - reference_ns) > max_delta_ns:
                selected = {}
                break
            selected[camera] = best
        if not selected:
            continue
        row: dict[str, int | float] = {
            "sync_index": len(synchronized),
            "reference_timestamp_ns": reference_ns,
        }
        for camera in camera_ids:
            index = selected[camera]
            timestamp_ns = timestamps[camera][index]
            row[f"{camera}_frame_index"] = index
            row[f"{camera}_timestamp_ns"] = timestamp_ns
            row[f"{camera}_delta_us"] = (timestamp_ns - reference_ns) / 1000.0
            if camera != reference_camera:
                used[camera].add(index)
        synchronized.append(row)
    return synchronized


def normalize(args: argparse.Namespace) -> Path:
    source = args.input.resolve()
    output = args.output.resolve()
    camera_ids = tuple(dict.fromkeys(args.cameras))
    if not source.is_file():
        raise FileNotFoundError(source)
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    if len(camera_ids) < 2 or args.reference_camera not in camera_ids:
        raise ValueError("at least two unique cameras and a listed reference camera are required")
    if args.max_delta_us < 0 or args.max_frames < 0:
        raise ValueError("max-delta-us/max-frames must be non-negative")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        video_paths = {}
        for camera in camera_ids:
            camera_root = temporary / "cameras" / camera
            camera_root.mkdir(parents=True)
            video_paths[camera] = camera_root / "video.mkv"
        calibrations, frame_rows, decode_stats, topics = remux_stereo_mcap(
            source, camera_ids, video_paths, args.max_frames
        )
        for camera in camera_ids:
            _write_csv(
                temporary / "cameras" / camera / "timestamps.csv",
                ["frame_index", "timestamp_ns", "timestamp_us", "source_message_index", "keyframe"],
                frame_rows[camera],
            )
        synchronized = synchronize_rows(
            frame_rows, camera_ids, args.reference_camera, args.max_delta_us * 1000
        )
        if not synchronized:
            raise RuntimeError("no synchronized multiview frames were found")
        fields = ["sync_index", "reference_timestamp_ns"]
        for camera in camera_ids:
            fields.extend([
                f"{camera}_frame_index", f"{camera}_timestamp_ns", f"{camera}_delta_us",
            ])
        _write_csv(temporary / "multiview_frames.csv", fields, synchronized)
        calibration_root = temporary / "calibration"
        calibration_root.mkdir()
        for camera in camera_ids:
            calibrations[camera].save(calibration_root / f"{camera}.json")
        absolute_deltas = [
            abs(float(row[f"{camera}_delta_us"]))
            for row in synchronized for camera in camera_ids
            if camera != args.reference_camera
        ]
        manifest = {
            "schema_version": 1,
            "dataset_type": "normalized_multiview",
            "source": {
                "path": str(source), "size_bytes": source.stat().st_size,
                "sha256": _sha256(source),
            },
            "camera_ids": list(camera_ids),
            "reference_camera": args.reference_camera,
            "image_size": list(calibrations[args.reference_camera].image_size),
            "storage": {"kind": "h264_matroska", "video_filename": "video.mkv"},
            "topics": topics,
            "decode": decode_stats,
            "synchronization": {
                "frame_count": len(synchronized),
                "max_delta_us": args.max_delta_us,
                "abs_delta_us_median": float(np.median(absolute_deltas)),
                "abs_delta_us_p95": float(np.percentile(absolute_deltas, 95)),
                "abs_delta_us_max": float(max(absolute_deltas)),
            },
            "coordinate_system": "GEN base/rig frame from camera_info T_b_c",
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        temporary.rename(output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output


def main() -> int:
    args = parse_args()
    output = normalize(args)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    sync = manifest["synchronization"]
    print(f"Normalized multiview dataset: {output}")
    print(f"Cameras: {', '.join(manifest['camera_ids'])}")
    print(f"Synchronized frames: {sync['frame_count']}")
    print(f"Synchronization |delta| p95/max: {sync['abs_delta_us_p95']:.1f}/{sync['abs_delta_us_max']:.1f} us")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
