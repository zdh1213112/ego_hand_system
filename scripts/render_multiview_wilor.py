#!/usr/bin/env python3
"""Render a tiled diagnostic video for multiview WiLoR fusion."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from ego_data.dataset import SequentialVideoReader  # noqa: E402
from render_wilor_predictions import HAND_CONNECTIONS, LEFT_COLOR, RIGHT_COLOR  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--fusion", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--cameras", nargs="+",
        help="camera subset and display order (default: fusion summary or dataset order)",
    )
    parser.add_argument(
        "--columns", type=int, default=0,
        help="tile columns; 0 selects 2 for four views, 3 for six views, otherwise auto",
    )
    parser.add_argument("--tile-width", type=int, default=480)
    parser.add_argument("--max-frames", type=int, default=0)
    return parser.parse_args()


def _load_jsonl(path: Path) -> dict[int, dict]:
    if not path.is_file():
        return {}
    return {
        int(row["sync_index"]): row
        for row in (
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }


def _draw_hand(frame: np.ndarray, hand: dict, side: int) -> None:
    color = RIGHT_COLOR if side else LEFT_COLOR
    box = np.rint(hand["bbox_xyxy"]).astype(np.int32)
    inlier_count = int(hand.get("inlier_joint_count", 21))
    if inlier_count <= 0:
        cv2.rectangle(frame, tuple(box[:2]), tuple(box[2:]), (40, 40, 255), 4, cv2.LINE_AA)
        cv2.putText(
            frame, "OUTLIER", (int(box[0]), max(30, int(box[1]) - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (40, 40, 255), 2, cv2.LINE_AA,
        )
        return
    cv2.rectangle(frame, tuple(box[:2]), tuple(box[2:]), color, 4, cv2.LINE_AA)
    points = np.rint(hand["joints_2d"]).astype(np.int32)
    if points.shape != (21, 2):
        return
    for start, end in HAND_CONNECTIONS:
        cv2.line(frame, tuple(points[start]), tuple(points[end]), color, 4, cv2.LINE_AA)
    for point in points:
        cv2.circle(frame, tuple(point), 5, color, -1, cv2.LINE_AA)


def main() -> int:
    args = parse_args()
    dataset = args.dataset.resolve()
    fusion = args.fusion.resolve()
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    available_cameras = tuple(manifest["camera_ids"])
    fusion_summary_path = fusion / "summary.json"
    fusion_summary = (
        json.loads(fusion_summary_path.read_text(encoding="utf-8"))
        if fusion_summary_path.is_file() else {}
    )
    default_cameras = fusion_summary.get("camera_ids", available_cameras)
    cameras = tuple(dict.fromkeys(args.cameras or default_cameras))
    if not cameras:
        raise ValueError("at least one render camera is required")
    unknown_cameras = [camera for camera in cameras if camera not in available_cameras]
    if unknown_cameras:
        raise ValueError(f"selected cameras are not present in the dataset: {unknown_cameras}")
    if args.columns < 0:
        raise ValueError("columns must be non-negative")
    if args.columns:
        columns = args.columns
    elif len(cameras) == 4:
        columns = 2
    elif len(cameras) == 6:
        columns = 3
    else:
        columns = max(1, int(math.ceil(math.sqrt(len(cameras)))))
    row_count = int(math.ceil(len(cameras) / columns))
    with (dataset / "multiview_frames.csv").open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    rows = rows[: args.max_frames or None]
    accepted = _load_jsonl(fusion / "accepted.jsonl")
    rejected = _load_jsonl(fusion / "rejected.jsonl")
    image_size = tuple(manifest["image_size"])
    tile_width = args.tile_width
    tile_height = int(round(tile_width * image_size[1] / image_size[0]))
    timestamps = np.asarray([int(row["reference_timestamp_ns"]) for row in rows], dtype=np.int64)
    delta = np.diff(timestamps).astype(np.float64) / 1e9
    fps = float(1.0 / np.median(delta[delta > 0])) if np.any(delta > 0) else 30.0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), fps,
        (tile_width * columns, tile_height * row_count),
    )
    if not writer.isOpened():
        raise RuntimeError(f"cannot create video: {args.output}")
    readers = {
        camera: SequentialVideoReader(
            dataset / "cameras" / camera / manifest["storage"]["video_filename"], image_size
        )
        for camera in cameras
    }
    try:
        for ordinal, row in enumerate(rows):
            sync_index = int(row["sync_index"])
            result = accepted.get(sync_index)
            tiles = []
            for camera in cameras:
                frame = readers[camera].read(int(row[f"{camera}_frame_index"]))
                if result is not None:
                    for hand in result["hands"]:
                        view = hand["views"].get(camera)
                        if view is not None:
                            _draw_hand(frame, view, int(hand["side"]))
                camera_views = [
                    hand["views"][camera] for hand in result["hands"]
                    if camera in hand["views"]
                ] if result is not None else []
                inlier_joints = sum(int(view.get("inlier_joint_count", 21)) for view in camera_views)
                if result is None:
                    status, status_color = "REJECTED", (40, 80, 255)
                elif not camera_views:
                    status, status_color = "INACTIVE", (180, 180, 180)
                elif inlier_joints == 0:
                    status, status_color = "OUTLIER", (40, 80, 255)
                else:
                    status, status_color = f"USED {inlier_joints}/42", (70, 220, 70)
                cv2.putText(
                    frame, f"{camera} | {status} | frame {sync_index}", (25, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 3, cv2.LINE_AA,
                )
                if result is None and sync_index in rejected:
                    reason = rejected[sync_index].get("reason") or ",".join(rejected[sync_index].get("reasons", []))
                    cv2.putText(
                        frame, reason[:55], (25, 82), cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, status_color, 2, cv2.LINE_AA,
                    )
                tiles.append(cv2.resize(frame, (tile_width, tile_height), interpolation=cv2.INTER_AREA))
            while len(tiles) < row_count * columns:
                tiles.append(np.zeros((tile_height, tile_width, 3), dtype=np.uint8))
            canvas = np.vstack([
                np.hstack(tiles[start:start + columns])
                for start in range(0, len(tiles), columns)
            ])
            writer.write(canvas)
            if (ordinal + 1) % 50 == 0:
                print(f"rendered {ordinal + 1}/{len(rows)}", flush=True)
    finally:
        for reader in readers.values():
            reader.close()
        writer.release()
    print(f"Multiview diagnostic video: {args.output} ({len(rows)} frames at {fps:.3f} fps)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
