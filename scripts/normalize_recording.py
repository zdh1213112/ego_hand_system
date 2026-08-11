#!/usr/bin/env python3
"""Convert a GEN DAS EGO MCAP into a model-independent raw stereo dataset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile

from ego_data.calibration import StereoCalibration
from ego_data.genrobot_mcap import remux_stereo_mcap
from ego_data.pairing import pair_timestamps_ns, pairing_statistics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="GEN DAS EGO .mcap")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--left-camera", default="camera2")
    parser.add_argument("--right-camera", default="camera3")
    parser.add_argument("--max-delta-us", type=int, default=1500)
    parser.add_argument("--max-frames", type=int, default=0, help="debug limit per camera; 0 means all")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def normalize(args: argparse.Namespace) -> Path:
    source = args.input.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    if args.left_camera == args.right_camera:
        raise ValueError("left and right camera must differ")
    if args.max_delta_us < 0 or args.max_frames < 0:
        raise ValueError("max-delta-us/max-frames must be non-negative")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    camera_ids = (args.left_camera, args.right_camera)
    try:
        for camera_id in camera_ids:
            (temporary / "cameras" / camera_id).mkdir(parents=True)
        video_paths = {
            camera_id: temporary / "cameras" / camera_id / "video.mkv"
            for camera_id in camera_ids
        }
        calibrations, frame_rows, decode_stats, topics = remux_stereo_mcap(
            source, camera_ids, video_paths, args.max_frames
        )
        for camera_id in camera_ids:
            write_csv(
                temporary / "cameras" / camera_id / "timestamps.csv",
                ["frame_index", "timestamp_ns", "timestamp_us", "source_message_index", "keyframe"],
                frame_rows[camera_id],
            )
        stereo = StereoCalibration.from_cameras(
            calibrations[args.left_camera], calibrations[args.right_camera]
        )
        left_timestamps = [int(row["timestamp_ns"]) for row in frame_rows[args.left_camera]]
        right_timestamps = [int(row["timestamp_ns"]) for row in frame_rows[args.right_camera]]
        pairs = pair_timestamps_ns(left_timestamps, right_timestamps, args.max_delta_us * 1000)
        if not pairs:
            raise RuntimeError("no synchronized stereo pairs were found")
        pair_rows = [{
            "pair_index": pair.pair_index,
            "left_frame_index": pair.left_frame_index,
            "right_frame_index": pair.right_frame_index,
            "left_timestamp_ns": pair.left_timestamp_ns,
            "right_timestamp_ns": pair.right_timestamp_ns,
            "delta_ns": pair.delta_ns,
            "delta_us": pair.delta_ns / 1000.0,
        } for pair in pairs]
        write_csv(
            temporary / "stereo_pairs.csv",
            ["pair_index", "left_frame_index", "right_frame_index", "left_timestamp_ns",
             "right_timestamp_ns", "delta_ns", "delta_us"],
            pair_rows,
        )
        calibration_dir = temporary / "calibration"
        calibration_dir.mkdir()
        for camera_id in camera_ids:
            calibrations[camera_id].save(calibration_dir / f"{camera_id}.json")
        (calibration_dir / "stereo.json").write_text(
            json.dumps(stereo.to_dict(), indent=2) + "\n", encoding="utf-8"
        )
        manifest = {
            "schema_version": 1,
            "dataset_type": "normalized_stereo",
            "source": {
                "path": str(source),
                "size_bytes": source.stat().st_size,
                "sha256": sha256_file(source),
            },
            "left_camera": args.left_camera,
            "right_camera": args.right_camera,
            "topics": topics,
            "image_size": list(stereo.left.image_size),
            "storage": {
                "kind": "h264_matroska",
                "lossless_remux": True,
                "video_filename": "video.mkv",
                "frame_access": "sequential",
            },
            "camera_model": stereo.left.model if stereo.left.model == stereo.right.model else "mixed",
            "baseline_m": stereo.baseline_m,
            "decode": decode_stats,
            "pairing": pairing_statistics(len(left_timestamps), len(right_timestamps), pairs),
            "parameters": {"max_delta_us": args.max_delta_us, "max_frames": args.max_frames},
            "coordinate_system": "raw camera frames; stereo 3D reference is the left OpenCV optical frame",
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
    result = normalize(args)
    manifest = json.loads((result / "manifest.json").read_text(encoding="utf-8"))
    print(f"Normalized dataset: {result}")
    print(f"Stereo pairs: {manifest['pairing']['pair_count']}")
    print(f"Baseline: {manifest['baseline_m']:.6f} m")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(1)
