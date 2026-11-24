#!/usr/bin/env python3
"""Decode OpenTouch HDF5 recordings into files and deterministic manifests.

The decoder deliberately preserves the raw coordinate values.  It does not
claim that ``right_hand_landmarks`` are in the RGB camera coordinate system and
does not create projected 2D labels.  JPEG payloads are copied byte-for-byte,
while synchronized numeric arrays are stored once per clip in an NPZ archive.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    import h5py
except ImportError as exc:  # pragma: no cover - exercised by the target env
    raise SystemExit(
        "h5py is required. Install pre_opentouch_dataset/requirements.txt "
        "or run this script in an environment that already provides h5py."
    ) from exc

import numpy as np


SCHEMA_VERSION = 1
DEFAULT_INPUT_DIR = Path("datasets/opentouch_raw")
DEFAULT_OUTPUT_DIR = Path("datasets/opentouch_decoded")
REQUIRED_CLIP_DATASETS = {
    "timestamps",
    "rgb_images_jpeg",
    "right_hand_landmarks",
    "right_wrist_pos",
    "right_palm_pos",
    "right_pressure",
    "camera_poses",
}
NUMERIC_CLIP_DATASETS = (
    "timestamps",
    "right_hand_landmarks",
    "right_wrist_pos",
    "right_palm_pos",
    "right_pressure",
    "camera_poses",
)


@dataclass(frozen=True)
class DecodeConfig:
    input_dir: str
    output_dir: str
    pattern: str
    frame_step: int
    max_files: int | None
    max_clips_per_file: int | None
    max_frames_per_clip: int | None
    compress_arrays: bool
    verify_jpeg_markers: bool


@dataclass
class DecodeStats:
    source_files: int = 0
    source_clips: int = 0
    source_frames: int = 0
    decoded_clips: int = 0
    decoded_frames: int = 0
    image_bytes: int = 0
    clip_array_bytes: int = 0
    invalid_keypoint_frames: int = 0
    skipped_existing_images: int = 0
    rewritten_images: int = 0
    skipped_existing_clip_arrays: int = 0
    rewritten_clip_arrays: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Decode OpenTouch HDF5 files into raw JPEGs, per-clip numeric "
            "arrays, and JSONL manifests."
        )
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--pattern", default="*.hdf5")
    parser.add_argument(
        "--frame-step",
        type=int,
        default=1,
        help="Decode every Nth source frame; default 1 keeps every frame.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Limit source files for a smoke test.",
    )
    parser.add_argument(
        "--max-clips-per-file",
        type=int,
        default=None,
        help="Limit clips within each source file for a smoke test.",
    )
    parser.add_argument(
        "--max-frames-per-clip",
        type=int,
        default=None,
        help="Limit selected frames within each clip for a smoke test.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, os.cpu_count() or 1),
        help="Concurrent JPEG writers. HDF5 reading remains single-threaded.",
    )
    parser.add_argument(
        "--compress-arrays",
        action="store_true",
        help="Use compressed NPZ files. This saves space but is much slower.",
    )
    parser.add_argument(
        "--no-verify-jpeg-markers",
        dest="verify_jpeg_markers",
        action="store_false",
        help="Do not check JPEG SOI/EOI markers before writing.",
    )
    parser.set_defaults(verify_jpeg_markers=True)
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume an interrupted decode using exactly the same options. "
            "Existing files with matching shapes/sizes are reused."
        ),
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate source schemas and print totals without writing output.",
    )
    parser.add_argument(
        "--log-every-clips",
        type=int,
        default=50,
        help="Print progress every N decoded clips.",
    )
    return parser.parse_args()


def positive_or_none(name: str, value: int | None) -> None:
    if value is not None and value < 1:
        raise ValueError(f"{name} must be at least 1, got {value}")


def validate_args(args: argparse.Namespace) -> None:
    positive_or_none("frame_step", args.frame_step)
    positive_or_none("max_files", args.max_files)
    positive_or_none("max_clips_per_file", args.max_clips_per_file)
    positive_or_none("max_frames_per_clip", args.max_frames_per_clip)
    positive_or_none("workers", args.workers)
    positive_or_none("log_every_clips", args.log_every_clips)


def jsonable(value: Any) -> Any:
    """Convert HDF5/NumPy values to JSON-compatible Python values."""
    if isinstance(value, np.ndarray):
        if value.dtype.names:
            return [
                {name: jsonable(row[name]) for name in value.dtype.names}
                for row in value
            ]
        return value.tolist()
    if isinstance(value, np.void) and value.dtype.names:
        return {name: jsonable(value[name]) for name in value.dtype.names}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
            temporary = Path(stream.name)
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def atomic_write_text(path: Path, payload: str) -> None:
    atomic_write_bytes(path, payload.encode("utf-8"))


def atomic_save_npz(
    path: Path,
    arrays: Mapping[str, np.ndarray],
    *,
    compressed: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    save = np.savez_compressed if compressed else np.savez
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            save(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
            temporary = Path(stream.name)
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_json_line(stream: Any, value: Mapping[str, Any]) -> None:
    stream.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")


def source_files(input_dir: Path, pattern: str, max_files: int | None) -> list[Path]:
    files = sorted(path for path in input_dir.glob(pattern) if path.is_file())
    if not files:
        raise FileNotFoundError(
            f"No source files matching {pattern!r} under {input_dir}"
        )
    return files if max_files is None else files[:max_files]


def selected_frame_indices(
    frame_count: int,
    frame_step: int,
    max_frames_per_clip: int | None,
) -> np.ndarray:
    values = np.arange(0, frame_count, frame_step, dtype=np.int64)
    if max_frames_per_clip is not None:
        values = values[:max_frames_per_clip]
    if values.size == 0:
        raise ValueError("Frame selection unexpectedly produced an empty clip")
    return values


def jpeg_bytes(value: Any) -> bytes:
    if isinstance(value, bytes):
        return value
    return np.asarray(value, dtype=np.uint8).tobytes()


def verify_jpeg(payload: bytes, context: str) -> None:
    if len(payload) < 4 or payload[:2] != b"\xff\xd8" or payload[-2:] != b"\xff\xd9":
        raise ValueError(f"Stored image is not a complete JPEG: {context}")


def write_one_image(
    destination: Path,
    payload: bytes,
    *,
    resume: bool,
) -> tuple[int, bool]:
    if resume and destination.is_file() and destination.stat().st_size == len(payload):
        return len(payload), True
    atomic_write_bytes(destination, payload)
    return len(payload), False


def validate_clip_schema(
    source_path: Path,
    clip_name: str,
    clip: h5py.Group,
) -> int:
    missing = sorted(REQUIRED_CLIP_DATASETS.difference(clip.keys()))
    if missing:
        raise ValueError(f"{source_path.name}:{clip_name} missing datasets {missing}")
    frame_count = int(clip["timestamps"].shape[0])
    expected_shapes = {
        "rgb_images_jpeg": (frame_count,),
        "right_hand_landmarks": (frame_count, 21, 3),
        "right_wrist_pos": (frame_count, 3),
        "right_palm_pos": (frame_count, 3),
        "right_pressure": (frame_count, 16, 16),
        "camera_poses": (frame_count, 4, 4),
    }
    for name, expected in expected_shapes.items():
        actual = tuple(int(size) for size in clip[name].shape)
        if actual != expected:
            raise ValueError(
                f"{source_path.name}:{clip_name}:{name} has shape {actual}, "
                f"expected {expected}"
            )
    if frame_count < 1:
        raise ValueError(f"{source_path.name}:{clip_name} is empty")
    return frame_count


def calibration_record(source_path: Path, hdf: h5py.File) -> dict[str, Any]:
    if "calibration/rgb" not in hdf:
        raise ValueError(f"{source_path.name} has no calibration/rgb group")
    rgb = hdf["calibration/rgb"]
    required = ("T_device_camera", "focal_length", "image_size", "principal_point")
    missing = [name for name in required if name not in rgb]
    if missing:
        raise ValueError(f"{source_path.name} missing RGB calibration {missing}")
    return {
        "schema_version": SCHEMA_VERSION,
        "source_file": source_path.name,
        "source_attributes": jsonable(dict(hdf.attrs)),
        "data_attributes": jsonable(dict(hdf["data"].attrs)),
        "rgb": {name: jsonable(np.asarray(rgb[name][()])) for name in required},
        "transform_slam_to_rgb": (
            jsonable(np.asarray(hdf["transform_slam_to_rgb"][()]))
            if "transform_slam_to_rgb" in hdf
            else None
        ),
        "coordinate_note": (
            "right_hand_landmarks are preserved in their source coordinate "
            "frame; no Rokoko-to-RGB transform is applied"
        ),
    }


def clip_labels(clip: h5py.Group) -> Any:
    return jsonable(np.asarray(clip["labels"][()])) if "labels" in clip else None


def clip_arrays(
    clip: h5py.Group,
    indices: np.ndarray,
) -> dict[str, np.ndarray]:
    result = {name: np.asarray(clip[name][indices]) for name in NUMERIC_CLIP_DATASETS}
    result["source_frame_indices"] = indices
    return result


def valid_existing_npz(path: Path, expected_indices: np.ndarray) -> bool:
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != set(NUMERIC_CLIP_DATASETS) | {
                "source_frame_indices"
            }:
                return False
            actual_indices = np.asarray(archive["source_frame_indices"])
            if not np.array_equal(actual_indices, expected_indices):
                return False
            expected_count = int(expected_indices.size)
            return all(
                int(archive[name].shape[0]) == expected_count
                for name in NUMERIC_CLIP_DATASETS
            )
    except (OSError, ValueError, KeyError):
        return False


def relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def configuration(
    args: argparse.Namespace, input_dir: Path, output_dir: Path
) -> DecodeConfig:
    return DecodeConfig(
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        pattern=args.pattern,
        frame_step=args.frame_step,
        max_files=args.max_files,
        max_clips_per_file=args.max_clips_per_file,
        max_frames_per_clip=args.max_frames_per_clip,
        compress_arrays=args.compress_arrays,
        verify_jpeg_markers=args.verify_jpeg_markers,
    )


def prepare_output(
    output_dir: Path,
    config: DecodeConfig,
    *,
    resume: bool,
) -> None:
    config_path = output_dir / "decode_config.json"
    expected = asdict(config)
    if output_dir.exists() and any(output_dir.iterdir()):
        if not resume:
            raise FileExistsError(
                f"Output directory is not empty: {output_dir}. "
                "Use --resume only for an interrupted run with identical options."
            )
        if not config_path.is_file():
            raise FileNotFoundError(f"Cannot resume because {config_path} is missing")
        actual = json.loads(config_path.read_text(encoding="utf-8"))
        if actual != expected:
            raise ValueError(
                "Resume options differ from decode_config.json. "
                f"Existing={actual}, requested={expected}"
            )
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_text(
        config_path,
        json.dumps(expected, ensure_ascii=False, indent=2) + "\n",
    )


def check_disk_space(files: Sequence[Path], output_dir: Path) -> dict[str, int]:
    source_bytes = sum(path.stat().st_size for path in files)
    usage = shutil.disk_usage(output_dir.parent)
    # HDF5 size is a useful conservative approximation for copied JPEGs plus
    # uncompressed numeric arrays.  Warn rather than reject because smoke-test
    # limits and filesystem compression make the exact requirement variable.
    recommended_free = int(source_bytes * 1.35)
    if usage.free < recommended_free:
        print(
            "WARNING: free disk space is below the conservative estimate: "
            f"free={usage.free / 2**30:.1f} GiB, "
            f"recommended={recommended_free / 2**30:.1f} GiB",
            file=sys.stderr,
        )
    return {
        "selected_source_bytes": source_bytes,
        "free_bytes_before_decode": usage.free,
        "recommended_free_bytes": recommended_free,
    }


def audit_sources(
    files: Sequence[Path],
    args: argparse.Namespace,
) -> dict[str, Any]:
    stats = DecodeStats(source_files=len(files))
    modes: Counter[str] = Counter()
    root_count_mismatches: list[dict[str, Any]] = []
    for source_path in files:
        with h5py.File(source_path, "r") as hdf:
            if "data" not in hdf:
                raise ValueError(f"{source_path.name} has no data group")
            calibration_record(source_path, hdf)
            modes[
                str(
                    hdf.attrs.get(
                        "hand_mode", hdf["data"].attrs.get("hand_mode", "unknown")
                    )
                )
            ] += 1
            names = sorted(hdf["data"].keys())
            selected_names = (
                names
                if args.max_clips_per_file is None
                else names[: args.max_clips_per_file]
            )
            actual_frames = 0
            for clip_name in names:
                frame_count = validate_clip_schema(
                    source_path, clip_name, hdf["data"][clip_name]
                )
                actual_frames += frame_count
                stats.source_clips += 1
                stats.source_frames += frame_count
                if clip_name in selected_names:
                    stats.decoded_clips += 1
                    stats.decoded_frames += int(
                        selected_frame_indices(
                            frame_count,
                            args.frame_step,
                            args.max_frames_per_clip,
                        ).size
                    )
            declared_clips = int(hdf.attrs.get("num_clips", len(names)))
            declared_frames = int(hdf.attrs.get("total_frames", actual_frames))
            if declared_clips != len(names) or declared_frames != actual_frames:
                root_count_mismatches.append(
                    {
                        "source_file": source_path.name,
                        "declared_clips": declared_clips,
                        "actual_clips": len(names),
                        "declared_frames": declared_frames,
                        "actual_frames": actual_frames,
                    }
                )
    return {
        "stats": asdict(stats),
        "hand_modes_by_file": dict(modes),
        "root_count_mismatches": root_count_mismatches,
    }


def decode(
    files: Sequence[Path],
    input_dir: Path,
    output_dir: Path,
    args: argparse.Namespace,
    disk: Mapping[str, int],
) -> dict[str, Any]:
    stats = DecodeStats(source_files=len(files))
    modes: Counter[str] = Counter()
    source_summaries: list[dict[str, Any]] = []
    root_count_mismatches: list[dict[str, Any]] = []
    frames_tmp = output_dir / ".frames.jsonl.tmp"
    clips_tmp = output_dir / ".clips.jsonl.tmp"

    with (
        frames_tmp.open("w", encoding="utf-8") as frame_stream,
        clips_tmp.open("w", encoding="utf-8") as clip_stream,
        ThreadPoolExecutor(max_workers=args.workers) as pool,
    ):
        for file_index, source_path in enumerate(files, start=1):
            print(f"[{file_index}/{len(files)}] {source_path.name}")
            with h5py.File(source_path, "r") as hdf:
                if "data" not in hdf:
                    raise ValueError(f"{source_path.name} has no data group")
                calibration = calibration_record(source_path, hdf)
                calibration_path = (
                    output_dir / "calibration" / f"{source_path.stem}.json"
                )
                atomic_write_text(
                    calibration_path,
                    json.dumps(calibration, ensure_ascii=False, indent=2) + "\n",
                )
                mode = str(
                    hdf.attrs.get(
                        "hand_mode", hdf["data"].attrs.get("hand_mode", "unknown")
                    )
                )
                modes[mode] += 1
                clip_names = sorted(hdf["data"].keys())
                selected_names = (
                    clip_names
                    if args.max_clips_per_file is None
                    else clip_names[: args.max_clips_per_file]
                )
                actual_source_frames = 0
                for clip_name in clip_names:
                    actual_source_frames += validate_clip_schema(
                        source_path, clip_name, hdf["data"][clip_name]
                    )
                stats.source_clips += len(clip_names)
                stats.source_frames += actual_source_frames
                declared_clips = int(hdf.attrs.get("num_clips", len(clip_names)))
                declared_frames = int(
                    hdf.attrs.get("total_frames", actual_source_frames)
                )
                if (
                    declared_clips != len(clip_names)
                    or declared_frames != actual_source_frames
                ):
                    root_count_mismatches.append(
                        {
                            "source_file": source_path.name,
                            "declared_clips": declared_clips,
                            "actual_clips": len(clip_names),
                            "declared_frames": declared_frames,
                            "actual_frames": actual_source_frames,
                        }
                    )

                for clip_name in selected_names:
                    clip = hdf["data"][clip_name]
                    frame_count = int(clip["timestamps"].shape[0])
                    indices = selected_frame_indices(
                        frame_count,
                        args.frame_step,
                        args.max_frames_per_clip,
                    )
                    arrays_path = (
                        output_dir / "clips" / source_path.stem / f"{clip_name}.npz"
                    )
                    if args.resume and valid_existing_npz(arrays_path, indices):
                        stats.skipped_existing_clip_arrays += 1
                    else:
                        arrays = clip_arrays(clip, indices)
                        finite = np.isfinite(arrays["right_hand_landmarks"]).all(
                            axis=(1, 2)
                        )
                        stats.invalid_keypoint_frames += int((~finite).sum())
                        atomic_save_npz(
                            arrays_path,
                            arrays,
                            compressed=args.compress_arrays,
                        )
                        stats.rewritten_clip_arrays += 1
                    stats.clip_array_bytes += arrays_path.stat().st_size

                    image_dir = output_dir / "images" / source_path.stem / clip_name
                    image_jobs = []
                    timestamps = np.asarray(clip["timestamps"][indices], dtype=np.int64)
                    for decoded_index, source_frame_index in enumerate(
                        indices.tolist()
                    ):
                        payload = jpeg_bytes(
                            clip["rgb_images_jpeg"][source_frame_index]
                        )
                        context = f"{source_path.name}:{clip_name}:{source_frame_index}"
                        if args.verify_jpeg_markers:
                            verify_jpeg(payload, context)
                        image_path = image_dir / f"{source_frame_index:06d}.jpg"
                        future = pool.submit(
                            write_one_image,
                            image_path,
                            payload,
                            resume=args.resume,
                        )
                        image_jobs.append(
                            (
                                future,
                                image_path,
                                decoded_index,
                                source_frame_index,
                                int(timestamps[decoded_index]),
                            )
                        )

                    for (
                        future,
                        image_path,
                        decoded_index,
                        source_frame_index,
                        timestamp,
                    ) in image_jobs:
                        image_size, skipped = future.result()
                        stats.image_bytes += image_size
                        if skipped:
                            stats.skipped_existing_images += 1
                        else:
                            stats.rewritten_images += 1
                        write_json_line(
                            frame_stream,
                            {
                                "sample_id": (
                                    f"{source_path.stem}::{clip_name}::"
                                    f"{source_frame_index:06d}"
                                ),
                                "source_file": source_path.name,
                                "source_stem": source_path.stem,
                                "clip_id": clip_name,
                                "source_frame_index": source_frame_index,
                                "clip_data_index": decoded_index,
                                "timestamp_ns": timestamp,
                                "image": relative(image_path, output_dir),
                                "clip_data": relative(arrays_path, output_dir),
                                "handedness": mode,
                            },
                        )

                    clip_timestamps = np.asarray(clip["timestamps"])
                    write_json_line(
                        clip_stream,
                        {
                            "source_file": source_path.name,
                            "source_stem": source_path.stem,
                            "clip_id": clip_name,
                            "source_frame_count": frame_count,
                            "decoded_frame_count": int(indices.size),
                            "start_timestamp_ns": int(clip_timestamps[0]),
                            "end_timestamp_ns": int(clip_timestamps[-1]),
                            "handedness": mode,
                            "image_dir": relative(image_dir, output_dir),
                            "clip_data": relative(arrays_path, output_dir),
                            "calibration": relative(calibration_path, output_dir),
                            "attributes": jsonable(dict(clip.attrs)),
                            "quality_labels": clip_labels(clip),
                        },
                    )
                    stats.decoded_clips += 1
                    stats.decoded_frames += int(indices.size)
                    if stats.decoded_clips % args.log_every_clips == 0:
                        print(
                            f"  decoded clips={stats.decoded_clips:,}, "
                            f"frames={stats.decoded_frames:,}"
                        )

                source_summaries.append(
                    {
                        "source_file": source_path.name,
                        "source_size_bytes": source_path.stat().st_size,
                        "actual_clip_count": len(clip_names),
                        "actual_frame_count": actual_source_frames,
                        "decoded_clip_count": len(selected_names),
                        "handedness": mode,
                        "calibration": relative(calibration_path, output_dir),
                    }
                )

        frame_stream.flush()
        os.fsync(frame_stream.fileno())
        clip_stream.flush()
        os.fsync(clip_stream.fileno())

    os.replace(frames_tmp, output_dir / "frames.jsonl")
    os.replace(clips_tmp, output_dir / "clips.jsonl")
    summary = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "stats": asdict(stats),
        "hand_modes_by_file": dict(modes),
        "source_files": source_summaries,
        "root_count_mismatches": root_count_mismatches,
        "disk_preflight": dict(disk),
        "image_encoding": "original JPEG bytes copied without re-encoding",
        "coordinate_status": {
            "right_hand_landmarks": "source coordinate frame, unchanged",
            "projected_2d_keypoints": "not generated",
            "rokoko_to_rgb_transform": "not available in decoded output",
        },
    }
    atomic_write_text(
        output_dir / "summary.json",
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
    )
    return summary


def main() -> int:
    args = parse_args()
    validate_args(args)
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    files = source_files(input_dir, args.pattern, args.max_files)
    config = configuration(args, input_dir, output_dir)

    if args.check_only:
        result = audit_sources(files, args)
        result["config"] = asdict(config)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    prepare_output(output_dir, config, resume=args.resume)
    disk = check_disk_space(files, output_dir)
    summary = decode(files, input_dir, output_dir, args, disk)
    print(json.dumps(summary["stats"], ensure_ascii=False, indent=2))
    print(f"Decoded dataset: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
