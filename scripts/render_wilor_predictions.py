#!/usr/bin/env python3
"""Render WiLoR JSONL predictions on a rectified camera video."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
)
LEFT_COLOR = (154, 165, 53)
RIGHT_COLOR = (60, 199, 255)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rectified-dataset", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--camera", required=True, choices=("left", "right"))
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def draw_hand(frame: np.ndarray, hand: dict) -> None:
    color = RIGHT_COLOR if hand["is_right"] else LEFT_COLOR
    box = np.rint(hand["bbox_xyxy"]).astype(np.int32)
    cv2.rectangle(frame, tuple(box[:2]), tuple(box[2:]), color, 3, cv2.LINE_AA)
    label = f"{hand['handedness']} {hand['confidence']:.2f}"
    cv2.putText(frame, label, (int(box[0]), max(25, int(box[1]) - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA)
    points = np.rint(hand["joints_2d"]).astype(np.int32)
    if points.shape != (21, 2):
        return
    for start, end in HAND_CONNECTIONS:
        cv2.line(frame, tuple(points[start]), tuple(points[end]), color, 3, cv2.LINE_AA)
    for index, point in enumerate(points):
        radius = 6 if index == 0 else 4
        cv2.circle(frame, tuple(point), radius, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, tuple(point), max(2, radius - 2), color, -1, cv2.LINE_AA)


def main() -> None:
    args = parse_args()
    video = args.rectified_dataset / f"{args.camera}.mkv"
    rows = [json.loads(line) for line in args.predictions.read_text(encoding="utf-8").splitlines()]
    by_pair = {int(row["pair_index"]): row for row in rows}
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {video}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"cannot create video: {args.output}")
    pair_index = 0
    written = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            row = by_pair.get(pair_index)
            if row is not None:
                for hand in row["hands"]:
                    draw_hand(frame, hand)
                cv2.putText(frame, f"WiLoR {args.camera} | pair {pair_index}", (25, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255),
                            2, cv2.LINE_AA)
                writer.write(frame)
                written += 1
            pair_index += 1
    finally:
        capture.release()
        writer.release()
    if written != len(rows):
        raise RuntimeError(f"rendered {written} prediction frames, expected {len(rows)}")
    print(f"WiLoR annotated video: {args.output}")


if __name__ == "__main__":
    main()
