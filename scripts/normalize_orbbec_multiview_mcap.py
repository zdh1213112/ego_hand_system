#!/usr/bin/env python3
"""Normalize an Orbbec EGO Pro four-camera MCAP for raw WiLoR review.

The newer EGO Pro recorder stores four H.264 streams directly in one MCAP:
``/camera/left``, ``/camera/left_side``, ``/camera/right`` and
``/camera/right_side``.  Unlike the older exported Orbbec session, this file
does not contain camera-info/calibration messages.  This adapter therefore
remuxes the streams losslessly, synchronizes their device timestamps, and
marks calibration as unavailable instead of inventing extrinsics.  Raw 2-D
WiLoR inference and the native 2-D renderer do not need calibration.

If a separately measured calibration is available, ``--calibration-dir`` can
point to a directory containing ``camera0.json`` ... ``camera3.json``; the
files are copied into the normalized dataset and the manifest records them.
"""

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
from typing import Any

import numpy as np
from mcap.exceptions import EndOfFile
from mcap.reader import NonSeekingReader
from mcap_protobuf.decoder import DecoderFactory


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from ego_data.genrobot_mcap import H264MatroskaWriter  # noqa: E402


DEFAULT_TOPICS = {
    "camera0": "/camera/left",
    "camera1": "/camera/left_side",
    "camera2": "/camera/right",
    "camera3": "/camera/right_side",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", "--session", dest="input", required=True, type=Path,
        help="the .mcap file or a directory containing exactly one .mcap",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--cameras", nargs="+", default=list(DEFAULT_TOPICS),
        help="normalized camera IDs (default: camera0 camera1 camera2 camera3)",
    )
    parser.add_argument(
        "--camera-topic", action="append", default=[], metavar="CAMERA=TOPIC",
        help="override a source topic; may be repeated",
    )
    parser.add_argument("--reference-camera", default="camera0")
    parser.add_argument("--max-delta-us", type=int, default=1500)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--width", type=int, default=0)
    parser.add_argument("--height", type=int, default=0)
    parser.add_argument(
        "--calibration-dir", type=Path, default=None,
        help="optional directory with camera*.json calibration files",
    )
    parser.add_argument(
        "--allow-truncated", action="store_true",
        help="keep complete messages if the MCAP ends in a partial record",
    )
    return parser.parse_args()


def resolve_mcap(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.is_file():
        if path.suffix.lower() != ".mcap":
            raise ValueError(f"input file is not an MCAP: {path}")
        return path
    if not path.is_dir():
        raise FileNotFoundError(path)
    matches = sorted(path.glob("*.mcap"))
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one .mcap in {path}, found {len(matches)}"
        )
    return matches[0].resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _topic_map(cameras: tuple[str, ...], overrides: list[str]) -> dict[str, str]:
    topics = {camera: DEFAULT_TOPICS.get(camera, f"/camera/{camera}") for camera in cameras}
    for value in overrides:
        if "=" not in value:
            raise ValueError(f"invalid --camera-topic {value!r}; expected CAMERA=TOPIC")
        camera, topic = value.split("=", 1)
        if camera not in topics:
            raise ValueError(f"camera in --camera-topic is not selected: {camera}")
        if not topic.startswith("/"):
            raise ValueError(f"camera topic must start with '/': {topic!r}")
        topics[camera] = topic
    if len(set(topics.values())) != len(topics):
        raise ValueError(f"camera topics must be unique: {topics}")
    return topics


def _image_size(args: argparse.Namespace, source: Path) -> tuple[int, int]:
    if (args.width, args.height) != (0, 0):
        if args.width <= 0 or args.height <= 0:
            raise ValueError("--width and --height must both be positive")
        return int(args.width), int(args.height)
    # The recorder writes this metadata beside the MCAP.  Keep the adapter
    # usable without metadata by falling back to the EGO Pro documented mode.
    candidates = [
        source.with_name(f"meta_{source.stem}.json"),
        source.parent / f"meta_{source.stem}.json",
    ]
    for candidate in candidates:
        if not candidate.is_file():
            continue
        try:
            metadata = json.loads(candidate.read_text(encoding="utf-8"))
            streams = metadata.get("streams", {})
            for value in streams.values():
                if isinstance(value, dict) and value.get("width") and value.get("height"):
                    return int(value["width"]), int(value["height"])
        except (OSError, ValueError, TypeError):
            pass
    return 1600, 1200


def synchronize_rows(
    frame_rows: dict[str, list[dict[str, Any]]],
    cameras: tuple[str, ...],
    reference_camera: str,
    max_delta_ns: int,
) -> list[dict[str, Any]]:
    """Nearest-neighbor synchronization without reusing a frame."""
    timestamps = {
        camera: [int(row["timestamp_ns"]) for row in frame_rows[camera]]
        for camera in cameras
    }
    used = {camera: set() for camera in cameras if camera != reference_camera}
    synchronized: list[dict[str, Any]] = []
    for reference_index, reference_ns in enumerate(timestamps[reference_camera]):
        selected: dict[str, int] = {reference_camera: reference_index}
        for camera in cameras:
            if camera == reference_camera:
                continue
            values = timestamps[camera]
            position = bisect_left(values, reference_ns)
            candidates = [
                index for index in (position - 1, position)
                if 0 <= index < len(values) and index not in used[camera]
            ]
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
        row: dict[str, Any] = {
            "sync_index": len(synchronized),
            "reference_timestamp_ns": int(reference_ns),
        }
        for camera in cameras:
            index = selected[camera]
            timestamp_ns = timestamps[camera][index]
            row[f"{camera}_frame_index"] = int(index)
            row[f"{camera}_timestamp_ns"] = int(timestamp_ns)
            row[f"{camera}_delta_us"] = (timestamp_ns - reference_ns) / 1000.0
            if camera != reference_camera:
                used[camera].add(index)
        synchronized.append(row)
    return synchronized


def _copy_calibration(
    source: Path | None, temporary: Path, cameras: tuple[str, ...],
) -> tuple[bool, dict[str, str]]:
    if source is None:
        return False, {}
    source = source.expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"calibration directory does not exist: {source}")
    destination = temporary / "calibration"
    destination.mkdir()
    hashes: dict[str, str] = {}
    for camera in cameras:
        path = source / f"{camera}.json"
        if not path.is_file():
            raise FileNotFoundError(f"missing calibration file: {path}")
        shutil.copy2(path, destination / path.name)
        hashes[camera] = sha256_file(path)
    return True, hashes


def normalize(args: argparse.Namespace) -> Path:
    source = resolve_mcap(args.input)
    output = args.output.expanduser().resolve()
    cameras = tuple(dict.fromkeys(args.cameras))
    if len(cameras) < 2:
        raise ValueError("at least two cameras are required")
    if args.reference_camera not in cameras:
        raise ValueError("reference camera must be included in --cameras")
    if args.max_delta_us < 0 or args.max_frames < 0:
        raise ValueError("max-delta-us/max-frames must be non-negative")
    topics = _topic_map(cameras, args.camera_topic)
    image_size = _image_size(args, source)
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    writers: dict[str, H264MatroskaWriter] = {}
    try:
        video_paths: dict[str, Path] = {}
        for camera in cameras:
            root = temporary / "cameras" / camera
            root.mkdir(parents=True)
            video_paths[camera] = root / "video.mkv"
            writers[camera] = H264MatroskaWriter(camera, video_paths[camera], image_size)
        topic_to_camera = {topic: camera for camera, topic in topics.items()}
        factory = DecoderFactory()
        try:
            with source.open("rb") as stream:
                reader = NonSeekingReader(stream, decoder_factories=[factory])
                for schema, channel, message in reader.iter_messages(
                    topics=topic_to_camera.keys(), log_time_order=False
                ):
                    camera = topic_to_camera[channel.topic]
                    decoder = factory.decoder_for(channel.message_encoding, schema)
                    if decoder is None:
                        raise RuntimeError(f"no protobuf decoder for {channel.topic}")
                    proto = decoder(message.data)
                    if str(getattr(proto, "format", "")).strip().lower() != "h264":
                        raise ValueError(
                            f"{camera} topic {channel.topic} is not H264: "
                            f"{getattr(proto, 'format', None)!r}"
                        )
                    writer = writers[camera]
                    if args.max_frames and writer.packet_count >= args.max_frames:
                        if all(item.packet_count >= args.max_frames for item in writers.values()):
                            break
                        continue
                    writer.write(bytes(proto.data), int(message.log_time))
        except EndOfFile:
            if not args.allow_truncated:
                raise
        finally:
            for writer in writers.values():
                writer.close()
        frame_rows = {camera: writers[camera].rows for camera in cameras}
        for camera in cameras:
            if not frame_rows[camera]:
                raise RuntimeError(f"{camera} contains no H264 keyframe sequence")
            decoded = writers[camera].validate(image_size)
            if decoded != len(frame_rows[camera]):
                raise RuntimeError(
                    f"{camera} decoded {decoded} frames but remuxed "
                    f"{len(frame_rows[camera])} packets"
                )
            _write_csv(
                temporary / "cameras" / camera / "timestamps.csv",
                ["frame_index", "timestamp_ns", "timestamp_us", "source_message_index", "keyframe"],
                frame_rows[camera],
            )
        synchronized = synchronize_rows(
            frame_rows, cameras, args.reference_camera, args.max_delta_us * 1000
        )
        if not synchronized:
            raise RuntimeError("no synchronized four-camera frames were found")
        fields = ["sync_index", "reference_timestamp_ns"]
        for camera in cameras:
            fields.extend([
                f"{camera}_frame_index", f"{camera}_timestamp_ns", f"{camera}_delta_us",
            ])
        _write_csv(temporary / "multiview_frames.csv", fields, synchronized)

        calibration_available, calibration_hashes = _copy_calibration(
            args.calibration_dir, temporary, cameras
        )
        deltas = [
            abs(float(row[f"{camera}_delta_us"]))
            for row in synchronized for camera in cameras if camera != args.reference_camera
        ]
        source_meta = source.with_name(f"meta_{source.stem}.json")
        if source_meta.is_file():
            shutil.copy2(source_meta, temporary / "source_meta.json")
        manifest = {
            "schema_version": 1,
            "dataset_type": "normalized_multiview",
            "source": {
                "path": str(source), "size_bytes": source.stat().st_size,
                "sha256": sha256_file(source), "kind": "orbbec_ego_pro_mcap",
                "allow_truncated": bool(args.allow_truncated),
            },
            "camera_ids": list(cameras),
            "reference_camera": args.reference_camera,
            "image_size": list(image_size),
            "fps": 30.0,
            "storage": {"kind": "h264_matroska", "video_filename": "video.mkv"},
            "topics": {f"{camera}_image": topics[camera] for camera in cameras},
            "decode": {
                camera: {
                    **writers[camera].statistics(len(frame_rows[camera])),
                    "source_topic": topics[camera],
                }
                for camera in cameras
            },
            "synchronization": {
                "frame_count": len(synchronized),
                "max_delta_us": args.max_delta_us,
                "abs_delta_us_median": float(np.median(deltas)) if deltas else 0.0,
                "abs_delta_us_p95": float(np.percentile(deltas, 95)) if deltas else 0.0,
                "abs_delta_us_max": float(max(deltas)) if deltas else 0.0,
            },
            "calibration": {
                "available": calibration_available,
                "status": "external_json" if calibration_available else "not_in_mcap",
                "sha256": calibration_hashes,
            },
            "coordinate_system": (
                "external calibration coordinate system"
                if calibration_available else
                "unknown; raw camera pixel coordinates only (no extrinsics in MCAP)"
            ),
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        temporary.rename(output)
    except Exception:
        for writer in writers.values():
            try:
                writer.close()
            except Exception:
                pass
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output


def main() -> int:
    args = parse_args()
    output = normalize(args)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    sync = manifest["synchronization"]
    print(f"Normalized Orbbec four-view dataset: {output}")
    print(f"Cameras: {', '.join(manifest['camera_ids'])}")
    print(f"Synchronized frames: {sync['frame_count']}")
    print(
        "Synchronization |delta| p95/max: "
        f"{sync['abs_delta_us_p95']:.1f}/{sync['abs_delta_us_max']:.1f} us"
    )
    print(f"Calibration available: {manifest['calibration']['available']}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
