#!/usr/bin/env python3
"""Convert an Orbbec EGO session into the common normalized stereo dataset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile

from ego_data.calibration import StereoCalibration, stereo_from_ego_yaml
from ego_data.pairing import pair_timestamps_ns, pairing_statistics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-delta-us", type=int, default=1500)
    return parser.parse_args()


def unique_file(session: Path, suffix: str) -> Path:
    matches = sorted(session.glob(f"*{suffix}"))
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one '*{suffix}' in {session}, found {len(matches)}")
    return matches[0]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_timestamps(path: Path) -> list[int]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise RuntimeError(f"timestamp CSV is empty: {path}")
    result = []
    for row in rows:
        if row.get("timestamp_ns") not in (None, ""):
            value = int(row["timestamp_ns"])
        elif row.get("timestamp_us") not in (None, ""):
            value = int(row["timestamp_us"]) * 1000
        elif row.get("timestamp") not in (None, ""):
            # Orbbec exports timestamp in microseconds when no unit is named.
            value = int(float(row["timestamp"]) * 1000.0)
        else:
            raise RuntimeError(f"timestamp CSV has no timestamp_ns/timestamp_us column: {path}")
        result.append(value)
    if any(current <= previous for previous, current in zip(result, result[1:])):
        raise RuntimeError(f"timestamps are not strictly increasing: {path}")
    return result


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def normalize(args: argparse.Namespace) -> Path:
    session = args.session.resolve()
    output = args.output.resolve()
    if not session.is_dir():
        raise FileNotFoundError(f"session does not exist: {session}")
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    if args.max_delta_us < 0:
        raise ValueError("max-delta-us must be non-negative")

    left_video = unique_file(session, "_camera_left.mp4")
    right_video = unique_file(session, "_camera_right.mp4")
    left_pts = unique_file(session, "_camera_left_pts.csv")
    right_pts = unique_file(session, "_camera_right_pts.csv")
    calibration_path = unique_file(session, "_calibration_camera.yaml")
    stereo = stereo_from_ego_yaml(calibration_path)
    left_timestamps = read_timestamps(left_pts)
    right_timestamps = read_timestamps(right_pts)
    pairs = pair_timestamps_ns(left_timestamps, right_timestamps, args.max_delta_us * 1000)
    if not pairs:
        raise RuntimeError("no synchronized stereo pairs were found")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        camera_dirs = {"left": temporary / "cameras" / stereo.left.camera_id,
                       "right": temporary / "cameras" / stereo.right.camera_id}
        for directory in camera_dirs.values():
            directory.mkdir(parents=True, exist_ok=True)
        # Keep the normalized dataset self-contained. This is intentionally a copy,
        # so a dataset remains usable after the original recording is moved.
        shutil.copy2(left_video, camera_dirs["left"] / "video.mp4")
        shutil.copy2(right_video, camera_dirs["right"] / "video.mp4")
        write_csv(
            camera_dirs["left"] / "timestamps.csv",
            ["frame_index", "timestamp_ns", "timestamp_us"],
            [{"frame_index": i, "timestamp_ns": t, "timestamp_us": t // 1000}
             for i, t in enumerate(left_timestamps)],
        )
        write_csv(
            camera_dirs["right"] / "timestamps.csv",
            ["frame_index", "timestamp_ns", "timestamp_us"],
            [{"frame_index": i, "timestamp_ns": t, "timestamp_us": t // 1000}
             for i, t in enumerate(right_timestamps)],
        )
        write_csv(
            temporary / "stereo_pairs.csv",
            ["pair_index", "left_frame_index", "right_frame_index", "left_timestamp_ns",
             "right_timestamp_ns", "delta_ns", "delta_us"],
            [{"pair_index": p.pair_index, "left_frame_index": p.left_frame_index,
              "right_frame_index": p.right_frame_index, "left_timestamp_ns": p.left_timestamp_ns,
              "right_timestamp_ns": p.right_timestamp_ns, "delta_ns": p.delta_ns,
              "delta_us": p.delta_ns / 1000.0} for p in pairs],
        )
        calibration_dir = temporary / "calibration"
        calibration_dir.mkdir()
        stereo.left.save(calibration_dir / f"{stereo.left.camera_id}.json")
        stereo.right.save(calibration_dir / f"{stereo.right.camera_id}.json")
        (calibration_dir / "stereo.json").write_text(
            json.dumps(stereo.to_dict(), indent=2) + "\n", encoding="utf-8"
        )
        files = [left_video, right_video, left_pts, right_pts, calibration_path]
        manifest = {
            "schema_version": 1,
            "dataset_type": "normalized_stereo",
            "source": {
                "kind": "orbbec_session", "session": str(session),
                "files": [{"path": str(path), "size_bytes": path.stat().st_size,
                            "sha256": sha256_file(path)} for path in files],
            },
            "left_camera": stereo.left.camera_id,
            "right_camera": stereo.right.camera_id,
            "image_size": list(stereo.left.image_size),
            "storage": {"kind": "original_mp4", "video_filename": "video.mp4",
                         "frame_access": "sequential", "lossless_copy": True},
            "camera_model": stereo.left.model if stereo.left.model == stereo.right.model else "mixed",
            "baseline_m": stereo.baseline_m,
            "pairing": pairing_statistics(len(left_timestamps), len(right_timestamps), pairs),
            "parameters": {"max_delta_us": args.max_delta_us},
            "coordinate_system": "raw camera frames; stereo 3D reference is the left OpenCV optical frame",
        }
        (temporary / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        temporary.rename(output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output


def main() -> int:
    result = normalize(parse_args())
    manifest = json.loads((result / "manifest.json").read_text(encoding="utf-8"))
    print(f"Normalized dataset: {result}")
    print(f"Stereo pairs: {manifest['pairing']['pair_count']}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(1)
