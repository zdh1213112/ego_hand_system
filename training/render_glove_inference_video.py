#!/usr/bin/env python3
"""Render saved glove inference on paired rectified validation frames."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


HAND_CONNECTIONS = tuple(
    (a, b) for base in (1, 5, 9, 13, 17)
    for a, b in ((0, base), (base, base + 1),
                 (base + 1, base + 2), (base + 2, base + 3))
)
TILE_WIDTH = 800
TILE_HEIGHT = 500
RED = (35, 35, 245)
MAGENTA = (225, 30, 225)
CYAN = (240, 220, 25)


def draw_tile(image: np.ndarray, marker: np.ndarray, valid: np.ndarray,
              prediction: dict | None, camera: str, sync: int,
              mode: str) -> np.ndarray:
    height, width = image.shape[:2]
    tile = cv2.resize(image, (TILE_WIDTH, TILE_HEIGHT), interpolation=cv2.INTER_AREA)
    scale = np.array([TILE_WIDTH / width, TILE_HEIGHT / height], dtype=np.float32)
    for point in marker[valid]:
        if np.isfinite(point).all():
            xy = np.rint(point * scale).astype(int)
            cv2.circle(tile, tuple(xy), 3, MAGENTA, -1, cv2.LINE_AA)
    if prediction is not None:
        points = np.asarray(prediction["pred_ego_legacy_2d"], dtype=np.float32)
        if points.shape != (21, 2):
            raise ValueError(f"Expected 21 joints, got {points.shape}")
        points *= scale
        for first, second in HAND_CONNECTIONS:
            if np.isfinite(points[[first, second]]).all():
                p1, p2 = np.rint(points[[first, second]]).astype(int)
                cv2.line(tile, tuple(p1), tuple(p2), RED, 2, cv2.LINE_AA)
        for point in points:
            if np.isfinite(point).all():
                cv2.circle(tile, tuple(np.rint(point).astype(int)), 3, RED, -1,
                           cv2.LINE_AA)
        box = np.asarray(prediction["bbox"], dtype=np.float32).reshape(2, 2) * scale
        left, right = np.rint(box).astype(int)
        cv2.rectangle(tile, tuple(left), tuple(right), CYAN, 2, cv2.LINE_AA)
        error = prediction.get("ego_legacy_error_px")
        status = f"marker mean {error:.1f}px" if error is not None else "predicted"
        if mode == "detector" and prediction.get("detector_iou") is not None:
            status += f"  IoU {prediction['detector_iou']:.2f}"
    else:
        status = "NO DETECTION"
    cv2.rectangle(tile, (0, 0), (TILE_WIDTH, 65), (0, 0, 0), -1)
    mode_label = "detector0 (GT-picked)" if mode == "detector" else "GT box"
    cv2.putText(tile, f"{camera}  sync {sync}  {mode_label}", (15, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(tile, status, (15, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(tile, "RED: LoRA  MAGENTA: Nokov marker", (15, TILE_HEIGHT - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    return tile


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--marker-input", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("detector", "label"), default="detector")
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest = defaultdict(dict)
    for line in (args.dataset_root / "val.jsonl").open():
        row = json.loads(line)
        manifest[int(row["sync_index"])][row["camera"]] = row
    predictions = {}
    for line in args.predictions.open():
        row = json.loads(line)
        if row["mode"] == args.mode:
            predictions[(int(row["sync_index"]), row["camera"])] = row
    with np.load(args.marker_input) as archive:
        hand_index = next(i for i, side in enumerate(archive["handedness"])
                          if str(side).lower() == "left")
        indices = {int(value): i for i, value in enumerate(archive["pair_indices"])}
        markers = {
            "camera0": archive["left_rectified_px"][hand_index].copy(),
            "camera1": archive["right_rectified_px"][hand_index].copy(),
        }
        valids = {
            "camera0": archive["left_rectified_valid"][hand_index].copy(),
            "camera1": archive["right_rectified_valid"][hand_index].copy(),
        }
    syncs = sorted(sync for sync, cameras in manifest.items()
                   if all(camera in cameras for camera in ("camera0", "camera1"))
                   and sync in indices)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
               "-pix_fmt", "bgr24", "-s", f"{2*TILE_WIDTH}x{TILE_HEIGHT}",
               "-r", str(args.fps), "-i", "-", "-an", "-c:v", "libx264",
               "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
               "-movflags", "+faststart", str(args.output)]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    try:
        for sync in syncs:
            tiles = []
            for camera in ("camera0", "camera1"):
                image = cv2.imread(str(args.dataset_root / manifest[sync][camera]["image"]))
                if image is None:
                    raise FileNotFoundError(manifest[sync][camera]["image"])
                idx = indices[sync]
                tiles.append(draw_tile(image, markers[camera][idx], valids[camera][idx],
                                       predictions.get((sync, camera)), camera, sync,
                                       args.mode))
            frame = np.concatenate(tiles, axis=1)
            assert process.stdin is not None
            process.stdin.write(frame.tobytes())
    finally:
        assert process.stdin is not None
        process.stdin.close()
        if process.wait() != 0:
            raise RuntimeError("ffmpeg failed")
    print(json.dumps({"output": str(args.output), "frames": len(syncs),
                      "fps": args.fps, "mode": args.mode,
                      "predictions": len(predictions)}, indent=2))


if __name__ == "__main__":
    main()
