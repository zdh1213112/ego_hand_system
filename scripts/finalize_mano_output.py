#!/usr/bin/env python3
"""Finalize cached MANO tracks without rerunning optimization."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys

import cv2
import numpy as np


def load_fit_module():
    path = Path(__file__).with_name("fit_mano_sequence.py")
    spec = importlib.util.spec_from_file_location("fit_mano_sequence", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def compose_video(paths: list[Path], output: Path) -> None:
    captures = [cv2.VideoCapture(str(path)) for path in paths]
    if not all(capture.isOpened() for capture in captures):
        raise RuntimeError("cannot open one or more fitted track videos")
    frame_counts = [int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) for capture in captures]
    fps = captures[0].get(cv2.CAP_PROP_FPS) or 30.0
    output_size = (960, 1080)
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, output_size)
    if not writer.isOpened():
        raise RuntimeError(f"cannot create {output}")
    try:
        for frame_index in range(max(frame_counts)):
            rows = []
            for track, capture in enumerate(captures):
                ok, frame = capture.read() if frame_index < frame_counts[track] else (False, None)
                if not ok:
                    frame = np.full((540, 960, 3), 20, dtype=np.uint8)
                    cv2.putText(frame, f"Track {track} inactive", (30, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (180, 180, 180), 2, cv2.LINE_AA)
                else:
                    frame = cv2.resize(frame, (960, 540), interpolation=cv2.INTER_AREA)
                rows.append(frame)
            writer.write(np.vstack(rows))
    finally:
        writer.release()
        for capture in captures:
            capture.release()


def montage_from_video(video: Path, output: Path) -> None:
    capture = cv2.VideoCapture(str(video))
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    samples = []
    try:
        for frame_index in np.linspace(0, max(count - 1, 0), 6, dtype=int):
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            ok, frame = capture.read()
            if ok:
                samples.append(cv2.resize(frame, (320, 360), interpolation=cv2.INTER_AREA))
    finally:
        capture.release()
    if len(samples) != 6:
        raise RuntimeError("could not decode six montage frames")
    if not cv2.imwrite(str(output), np.hstack(samples)):
        raise RuntimeError(f"cannot write {output}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    fit = load_fit_module()
    video_paths = []
    for path in sorted(output.glob("track_*.npz")):
        with np.load(path) as archive:
            result = {name: archive[name].copy() for name in archive.files
                      if name not in ("pair_indices", "track_id", "handedness", "faces")}
            track_id = int(archive["track_id"])
            handedness = str(archive["handedness"])
            pairs = archive["pair_indices"].copy()
        fit.write_parameter_csv(
            output / f"track_{track_id}_parameters.csv", pairs, track_id, handedness, result
        )
        video_path = output / f"track_{track_id}_fit.mp4"
        if not video_path.is_file():
            raise FileNotFoundError(video_path)
        video_paths.append(video_path)
    if len(video_paths) != 2:
        raise RuntimeError(f"expected two fitted tracks, found {len(video_paths)}")
    combined = output / "mano_fit_both_hands.mp4"
    compose_video(video_paths, combined)
    montage_from_video(combined, output / "preview_montage.jpg")
    print(f"Combined video: {combined}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(1)
