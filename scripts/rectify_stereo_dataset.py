#!/usr/bin/env python3
"""Rectify a normalized KB/DS stereo dataset to a common pinhole geometry."""

from __future__ import annotations

import argparse
import csv
from fractions import Fraction
import json
from pathlib import Path
import shutil
import sys
import tempfile

import cv2
import numpy as np
import av

from camera_models import RectificationOptions, create_stereo_rectification
from camera_models.base import rectification_hash
from ego_data.dataset import NormalizedStereoDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="normalized stereo dataset")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--camera-model", choices=("auto", "kb", "ds"), default="auto")
    parser.add_argument("--balance", type=float, default=0.0)
    parser.add_argument("--focal-scale", type=float, default=1.0)
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument(
        "--video-codec", choices=("h264", "ffv1"), default="h264",
        help="rectified-video codec; H264 is compact, FFV1 is lossless",
    )
    parser.add_argument("--crf", type=int, default=18, help="H264 quality [0 lossless, 51 worst]")
    return parser.parse_args()


class RectifiedVideoWriter:
    def __init__(self, path: Path, image_size: tuple[int, int], fps: float, codec: str, crf: int):
        self.codec = codec
        self.frame_index = 0
        self.container = None
        self.cv_writer = None
        if codec == "ffv1":
            self.cv_writer = cv2.VideoWriter(
                str(path), cv2.VideoWriter_fourcc(*"FFV1"), fps, image_size
            )
            if not self.cv_writer.isOpened():
                raise RuntimeError(f"cannot create FFV1 video: {path}")
        else:
            self.container = av.open(str(path), "w", format="matroska")
            self.stream = self.container.add_stream("libx264", rate=max(1, round(fps)))
            self.stream.width, self.stream.height = image_size
            self.stream.pix_fmt = "yuv420p"
            self.stream.options = {"crf": str(crf), "preset": "fast"}

    def write(self, image: np.ndarray) -> None:
        if self.cv_writer is not None:
            self.cv_writer.write(image)
        else:
            frame = av.VideoFrame.from_ndarray(image, format="bgr24")
            frame.pts = self.frame_index
            frame.time_base = Fraction(1, int(self.stream.average_rate))
            for packet in self.stream.encode(frame):
                self.container.mux(packet)
        self.frame_index += 1

    def close(self) -> None:
        if self.cv_writer is not None:
            self.cv_writer.release()
            self.cv_writer = None
        if self.container is not None:
            for packet in self.stream.encode(None):
                self.container.mux(packet)
            self.container.close()
            self.container = None


def rectify_dataset(args: argparse.Namespace) -> Path:
    source = NormalizedStereoDataset(args.input)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    if (args.width is None) != (args.height is None):
        raise ValueError("--width and --height must be supplied together")
    if args.max_pairs < 0:
        raise ValueError("max-pairs must be non-negative")
    if not 0 <= args.crf <= 51:
        raise ValueError("crf must be in [0, 51]")
    output_size = (args.width, args.height) if args.width is not None else None
    options = RectificationOptions(output_size, args.balance, args.focal_scale)
    rectification = create_stereo_rectification(source.stereo, options, args.camera_model)
    digest = rectification_hash(source.stereo, options)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        selected_rows = source.pairs[:args.max_pairs or None]
        output_rows = []
        timestamps = [int(row["left_timestamp_ns"]) for row in selected_rows]
        fps = 1e9 / float(np.median(np.diff(timestamps))) if len(timestamps) > 1 else 30.0
        writers = [
            RectifiedVideoWriter(
                temporary / name, rectification.image_size, fps,
                args.video_codec, args.crf,
            )
            for name in ("left.mkv", "right.mkv")
        ]
        try:
            for output_index, (row, (left, right)) in enumerate(source):
                if output_index >= len(selected_rows):
                    break
                if int(row["pair_index"]) != output_index:
                    raise RuntimeError("normalized pair indices must be contiguous from zero")
                left_rectified = cv2.remap(
                    left, rectification.map_left_x, rectification.map_left_y,
                    cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                )
                right_rectified = cv2.remap(
                    right, rectification.map_right_x, rectification.map_right_y,
                    cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                )
                writers[0].write(left_rectified)
                writers[1].write(right_rectified)
                output_rows.append(dict(row))
        finally:
            for writer in writers:
                writer.close()
        if not output_rows:
            raise RuntimeError("no rectified pairs were written")
        fields = list(output_rows[0].keys())
        with (temporary / "stereo_pairs.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(output_rows)
        rectification.save_npz(temporary / "rectification.npz")
        metadata = {
            "schema_version": 1,
            "model": rectification.model,
            "left_camera": source.left_id,
            "right_camera": source.right_id,
            "image_size": list(rectification.image_size),
            "baseline_m": source.stereo.baseline_m,
            "balance": args.balance,
            "focal_scale": args.focal_scale,
            "focal_px": float(rectification.P1[0, 0]),
            "principal_point_px": [float(rectification.P1[0, 2]), float(rectification.P1[1, 2])],
            "valid_left_fraction": float(rectification.valid_left.mean()),
            "valid_right_fraction": float(rectification.valid_right.mean()),
            "storage": {
                "kind": f"{args.video_codec}_matroska",
                "lossless": args.video_codec == "ffv1" or (args.video_codec == "h264" and args.crf == 0),
                "left_video": "left.mkv",
                "right_video": "right.mkv",
                "fps": fps,
                "crf": args.crf if args.video_codec == "h264" else None,
                "frame_access": "sequential",
            },
            "calibration_hash": digest,
            "coordinate_system": "rectified OpenCV pinhole; 3D output converts back to the original left optical frame via R1.T",
        }
        (temporary / "rectification.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        manifest = {
            **metadata,
            "dataset_type": "rectified_stereo",
            "source_dataset": str(source.root),
            "source_mcap_sha256": source.manifest.get("source", {}).get("sha256"),
            "pair_count": len(output_rows),
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
    result = rectify_dataset(parse_args())
    metadata = json.loads((result / "rectification.json").read_text(encoding="utf-8"))
    print(f"Rectified dataset: {result}")
    print(f"Camera model: {metadata['model']}")
    print(f"Valid pixels L/R: {metadata['valid_left_fraction']:.1%}/{metadata['valid_right_fraction']:.1%}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(1)
