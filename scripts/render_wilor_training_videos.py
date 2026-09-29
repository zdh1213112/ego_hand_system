#!/usr/bin/env python3
"""Render chronological per-recording videos from an exported WiLoR dataset."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
import json
import os
from pathlib import Path
import re
import tempfile
import time
from typing import Any

import cv2
import numpy as np

from render_wilor_training_dataset import _draw_sample, _load_rows


VIDEO_VISUALIZATION_SCHEMA_VERSION = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cameras", nargs="+", help="camera subset; default uses summary.json")
    parser.add_argument(
        "--sources", nargs="+",
        help="source experiment names to render; default renders every source",
    )
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument(
        "--workers", type=int, default=2,
        help="source/camera videos rendered concurrently",
    )
    parser.add_argument(
        "--video-codec", choices=("auto", "h264_nvenc", "mp4v"), default="auto",
        help="auto prefers NVIDIA NVENC through PyAV and falls back to OpenCV mp4v",
    )
    parser.add_argument(
        "--max-frames", type=int, default=0,
        help="maximum exported frames per source/camera video; 0 means all",
    )
    parser.add_argument("--progress-interval", type=int, default=200)
    return parser.parse_args()


def _source_key(row: dict[str, Any], dataset: Path) -> tuple[int | None, str]:
    source_id = row.get("source_dataset_id")
    source_name = str(row.get("source_experiment") or dataset.parent.name)
    return (None if source_id is None else int(source_id), source_name)


def _group_rows(
    rows: list[dict[str, Any]], dataset: Path,
) -> dict[tuple[int | None, str, str], dict[int, list[dict[str, Any]]]]:
    grouped: dict[
        tuple[int | None, str, str], dict[int, list[dict[str, Any]]]
    ] = {}
    for row in rows:
        source_id, source_name = _source_key(row, dataset)
        camera = str(row["camera"])
        sync_index = int(row["sync_index"])
        grouped.setdefault((source_id, source_name, camera), {}).setdefault(
            sync_index, []
        ).append(row)
    return grouped


def _safe_token(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return token or "source"


def _video_name(source_id: int | None, source_name: str, camera: str) -> str:
    prefix = "" if source_id is None else f"source{source_id:04d}_"
    return f"{prefix}{_safe_token(source_name)}_{_safe_token(camera)}_labels.mp4"


class _OpenCvWriter:
    def __init__(self, path: Path, fps: float, frame_size: tuple[int, int]) -> None:
        self.writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, frame_size
        )
        if not self.writer.isOpened():
            raise RuntimeError(f"cannot create video: {path}")

    def write(self, frame: np.ndarray) -> None:
        self.writer.write(frame)

    def release(self) -> None:
        self.writer.release()


class _NvencWriter:
    def __init__(self, path: Path, fps: float, frame_size: tuple[int, int]) -> None:
        import av

        self.av = av
        self.container = av.open(str(path), mode="w")
        self.stream = self.container.add_stream(
            "h264_nvenc", rate=Fraction(fps).limit_denominator(100_000)
        )
        self.stream.width, self.stream.height = frame_size
        self.stream.pix_fmt = "yuv420p"
        self.stream.options = {
            "preset": "p2", "tune": "hq", "rc": "vbr", "cq": "19",
        }
        self.stream.codec_context.open()
        self.closed = False

    def write(self, frame: np.ndarray) -> None:
        video_frame = self.av.VideoFrame.from_ndarray(frame, format="bgr24")
        for packet in self.stream.encode(video_frame):
            self.container.mux(packet)

    def release(self) -> None:
        if self.closed:
            return
        for packet in self.stream.encode():
            self.container.mux(packet)
        self.container.close()
        self.closed = True


def _create_writer(
    path: Path, fps: float, frame_size: tuple[int, int], codec: str,
):
    if codec in ("auto", "h264_nvenc"):
        try:
            return _NvencWriter(path, fps, frame_size), "h264_nvenc"
        except Exception as error:
            path.unlink(missing_ok=True)
            if codec == "h264_nvenc":
                raise RuntimeError(f"cannot initialize h264_nvenc: {error}") from error
            print(f"[label-video] NVENC unavailable ({error}); using mp4v", flush=True)
    return _OpenCvWriter(path, fps, frame_size), "mp4v"


def _draw_header(
    frame: np.ndarray, source_name: str, camera: str, sync_index: int,
    frame_number: int, frame_count: int, hand_count: int,
) -> None:
    width = frame.shape[1]
    cv2.rectangle(frame, (0, 0), (width, 62), (16, 24, 32), -1)
    text = (
        f"{source_name} | {camera} | sync {sync_index} | "
        f"frame {frame_number}/{frame_count} | hands {hand_count}"
    )
    cv2.putText(
        frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
        0.78, (245, 245, 245), 2, cv2.LINE_AA,
    )


def _render_video(
    dataset: Path, output: Path, source_name: str, camera: str,
    frames: dict[int, list[dict[str, Any]]], frame_size: tuple[int, int],
    fps: float, codec: str, max_frames: int, progress_interval: int,
) -> dict[str, Any]:
    sync_indices = sorted(frames)
    if max_frames > 0:
        sync_indices = sync_indices[:max_frames]
    if not sync_indices:
        raise RuntimeError(f"no frames for {source_name}/{camera}")

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.stem}.tmp-", suffix=".mp4", dir=output.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    temporary.unlink()
    writer = None
    resolved_codec = ""
    start = time.perf_counter()
    hand_samples = 0
    try:
        writer, resolved_codec = _create_writer(temporary, fps, frame_size, codec)
        for frame_number, sync_index in enumerate(sync_indices, start=1):
            group = sorted(frames[sync_index], key=lambda row: int(row["side"]))
            frame = cv2.imread(str(dataset / group[0]["image"]), cv2.IMREAD_COLOR)
            if frame is None or frame.shape[1::-1] != frame_size:
                raise RuntimeError(
                    f"invalid image for {source_name}/{camera}/sync {sync_index}"
                )
            for row in group:
                sample = np.load(dataset / row["label"], allow_pickle=True).item()
                _draw_sample(frame, sample, row)
            _draw_header(
                frame, source_name, camera, sync_index,
                frame_number, len(sync_indices), len(group),
            )
            writer.write(frame)
            hand_samples += len(group)
            if progress_interval and (
                frame_number % progress_interval == 0
                or frame_number == len(sync_indices)
            ):
                print(
                    f"[label-video] {source_name}/{camera}: "
                    f"{frame_number}/{len(sync_indices)}",
                    flush=True,
                )
        writer.release()
        writer = None
        temporary.replace(output)
    except Exception:
        if writer is not None:
            writer.release()
        temporary.unlink(missing_ok=True)
        raise
    elapsed = time.perf_counter() - start
    return {
        "source_experiment": source_name,
        "camera": camera,
        "video": output.name,
        "codec": resolved_codec,
        "fps": fps,
        "frame_count": len(sync_indices),
        "hand_sample_count": hand_samples,
        "sync_index_range": [sync_indices[0], sync_indices[-1]],
        "duration_seconds": len(sync_indices) / fps,
        "render_elapsed_seconds": elapsed,
    }


def main() -> int:
    args = parse_args()
    if (
        args.fps <= 0.0 or args.workers <= 0
        or args.max_frames < 0 or args.progress_interval < 0
    ):
        raise ValueError(
            "fps and workers must be positive; frame limits must be non-negative"
        )
    dataset = args.dataset.resolve()
    output = args.output.resolve()
    summary = json.loads((dataset / "summary.json").read_text(encoding="utf-8"))
    rows = _load_rows(dataset / "index.jsonl")
    cameras = tuple(dict.fromkeys(args.cameras or summary.get("cameras", [])))
    if not cameras:
        raise ValueError("no visualization cameras selected")
    unknown_cameras = sorted(set(cameras) - set(summary.get("cameras", [])))
    if unknown_cameras:
        raise ValueError(f"cameras are not in the dataset: {unknown_cameras}")
    requested_sources = set(args.sources or ())
    grouped = _group_rows(rows, dataset)
    available_sources = {key[1] for key in grouped}
    unknown_sources = sorted(requested_sources - available_sources)
    if unknown_sources:
        raise ValueError(f"sources are not in the dataset: {unknown_sources}")
    selected = [
        (key, frames) for key, frames in sorted(
            grouped.items(),
            key=lambda item: (
                -1 if item[0][0] is None else item[0][0], item[0][1], item[0][2]
            ),
        )
        if key[2] in cameras and (not requested_sources or key[1] in requested_sources)
    ]
    if not selected:
        raise RuntimeError("no source/camera groups selected")

    output.mkdir(parents=True, exist_ok=True)
    frame_size = tuple(int(value) for value in summary["image_size"])
    tasks = []
    for (source_id, source_name, camera), frames in selected:
        video_path = output / _video_name(source_id, source_name, camera)
        if video_path.exists():
            raise FileExistsError(
                f"video already exists: {video_path}; use a new output directory"
            )
        tasks.append((
            dataset, video_path, source_name, camera, frames, frame_size,
            args.fps, args.video_codec, args.max_frames, args.progress_interval,
        ))
    if args.workers == 1:
        results = [_render_video(*task) for task in tasks]
    else:
        with ThreadPoolExecutor(max_workers=min(args.workers, len(tasks))) as executor:
            results = list(executor.map(lambda task: _render_video(*task), tasks))

    video_summary = {
        "schema_version": VIDEO_VISUALIZATION_SCHEMA_VERSION,
        "stage": "wilor_training_label_video_visualization",
        "dataset": str(dataset),
        "frame_timing": "compact_exported_sample_sequence",
        "results": results,
    }
    (output / "summary.json").write_text(
        json.dumps(video_summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[label-video] finished {len(results)} videos -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
