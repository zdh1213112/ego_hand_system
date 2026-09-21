#!/usr/bin/env python3
"""Normalize an EGO Pro four-camera Foxglove MCAP into the common dataset format."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np
import yaml
from mcap.reader import NonSeekingReader
from mcap_protobuf.decoder import DecoderFactory

from ego_data.calibration import CameraCalibration
from ego_data.genrobot_mcap import H264MatroskaWriter
from normalize_multiview_recording import synchronize_rows


CAMERA_TOPICS = {
    "camera0": "/camera/left",
    "camera1": "/camera/right",
    "camera2": "/camera/right_side",
    "camera3": "/camera/left_side",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--calibration", required=True, type=Path)
    parser.add_argument("--max-delta-us", type=int, default=1500)
    parser.add_argument("--max-frames", type=int, default=0)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _camera_from_yaml(camera_id: str, value: dict) -> CameraCalibration:
    if str(value["distortion_model"]).upper() != "KB":
        raise ValueError(f"{camera_id} calibration must use KB distortion")
    intr = value["intrinsics"]
    dist = value["distortion"]
    rotation = np.asarray(value["extrinsics"]["rotation"], dtype=np.float64)
    translation = np.asarray(value["extrinsics"]["translation"], dtype=np.float64) * 1e-3
    if rotation.shape != (3, 3) or translation.shape != (3,):
        raise ValueError(f"invalid extrinsics for {camera_id}")
    # EGO Pro stores reference->camera; common datasets store base->camera.
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation.T
    transform[:3, 3] = -rotation.T @ translation
    return CameraCalibration(
        camera_id=camera_id,
        frame_id=str(value.get("name", camera_id)),
        model="KB",
        image_size=(int(value["image_width"]), int(value["image_height"])),
        K=np.asarray(
            [[intr["fx"], 0.0, intr["cx"]],
             [0.0, intr["fy"], intr["cy"]],
             [0.0, 0.0, 1.0]], dtype=np.float64,
        ),
        distortion=np.asarray(
            [dist["k1"], dist["k2"], dist["k3"], dist["k4"]], dtype=np.float64
        ),
        T_base_camera=transform,
    )


def load_calibrations(path: Path) -> dict[str, CameraCalibration]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    cameras = {str(item.get("name")): item for item in value.get("cameras", [])}
    names = ("Color_Left", "Color_Right", "Side_Right", "Side_left")
    if any(name not in cameras for name in names):
        raise ValueError(f"calibration is missing EGO Pro cameras: {names}")
    return {
        camera_id: _camera_from_yaml(camera_id, cameras[name])
        for camera_id, name in zip(CAMERA_TOPICS, names)
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields = ["frame_index", "timestamp_ns", "timestamp_us", "source_message_index", "keyframe"]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def normalize(args: argparse.Namespace) -> Path:
    source = args.input.resolve()
    output = args.output.resolve()
    calibration_path = args.calibration.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if not calibration_path.is_file():
        raise FileNotFoundError(calibration_path)
    if output.exists():
        raise FileExistsError(output)
    if args.max_delta_us < 0 or args.max_frames < 0:
        raise ValueError("max-delta-us/max-frames must be non-negative")

    cameras = tuple(CAMERA_TOPICS)
    calibrations = load_calibrations(calibration_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    writers = {}
    try:
        for camera in cameras:
            directory = temporary / "cameras" / camera
            directory.mkdir(parents=True)
            writers[camera] = H264MatroskaWriter(
                camera, directory / "video.mkv", calibrations[camera].image_size
            )
        topic_to_camera = {topic: camera for camera, topic in CAMERA_TOPICS.items()}
        factory = DecoderFactory()
        with source.open("rb") as stream:
            reader = NonSeekingReader(stream, decoder_factories=[factory])
            for schema, channel, message in reader.iter_messages(
                topics=topic_to_camera, log_time_order=False
            ):
                camera = topic_to_camera[channel.topic]
                writer = writers[camera]
                if args.max_frames and writer.packet_count >= args.max_frames:
                    if all(item.packet_count >= args.max_frames for item in writers.values()):
                        break
                    continue
                decoder = factory.decoder_for(channel.message_encoding, schema)
                if decoder is None:
                    raise RuntimeError(f"no protobuf decoder for {channel.topic}")
                proto = decoder(message.data)
                if str(proto.format).strip().lower() != "h264":
                    raise ValueError(f"{channel.topic} is not h264: {proto.format!r}")
                writer.write(bytes(proto.data), int(message.log_time))
        frame_rows = {}
        decode_stats = {}
        for camera, writer in writers.items():
            writer.close()
            decoded = writer.validate(calibrations[camera].image_size)
            _write_csv(temporary / "cameras" / camera / "timestamps.csv", writer.rows)
            frame_rows[camera] = writer.rows
            decode_stats[camera] = writer.statistics(decoded)
        synchronized = synchronize_rows(
            frame_rows, cameras, "camera0", args.max_delta_us * 1000
        )
        if not synchronized:
            raise RuntimeError("no synchronized four-camera frames were found")
        fields = ["sync_index", "reference_timestamp_ns"]
        for camera in cameras:
            fields.extend([
                f"{camera}_frame_index", f"{camera}_timestamp_ns", f"{camera}_delta_us"
            ])
        with (temporary / "multiview_frames.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(synchronized)
        calibration_dir = temporary / "calibration"
        calibration_dir.mkdir()
        for camera in cameras:
            calibrations[camera].save(calibration_dir / f"{camera}.json")
        deltas = [
            abs(float(row[f"{camera}_delta_us"]))
            for row in synchronized for camera in cameras if camera != "camera0"
        ]
        manifest = {
            "schema_version": 1,
            "dataset_type": "normalized_multiview",
            "source": {"path": str(source), "size_bytes": source.stat().st_size,
                        "sha256": sha256_file(source)},
            "calibration_source": {"path": str(calibration_path),
                                    "sha256": sha256_file(calibration_path)},
            "camera_ids": list(cameras), "reference_camera": "camera0",
            "image_size": list(calibrations["camera0"].image_size),
            "storage": {"kind": "h264_matroska", "video_filename": "video.mkv",
                        "lossless_remux": True, "frame_access": "sequential"},
            "topics": {camera: topic for camera, topic in CAMERA_TOPICS.items()},
            "decode": decode_stats,
            "synchronization": {
                "frame_count": len(synchronized), "max_delta_us": args.max_delta_us,
                "abs_delta_us_median": float(np.median(deltas)),
                "abs_delta_us_p95": float(np.percentile(deltas, 95)),
                "abs_delta_us_max": float(max(deltas)),
            },
            "coordinate_system": "EGO Pro reference/base frame from KB calibration",
        }
        (temporary / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        temporary.rename(output)
    except Exception:
        for writer in writers.values():
            writer.close()
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output


def main() -> int:
    result = normalize(parse_args())
    manifest = json.loads((result / "manifest.json").read_text(encoding="utf-8"))
    sync = manifest["synchronization"]
    print(f"Normalized EGO Pro multiview dataset: {result}")
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
