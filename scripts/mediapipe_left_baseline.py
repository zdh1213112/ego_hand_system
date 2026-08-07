#!/usr/bin/env python3
"""Run the first EGO hand-landmark baseline on the rectified left video."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
import time

# MediaPipe imports matplotlib even when no plot is requested. Keep its cache in a
# writable, disposable directory instead of depending on the user's home config.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/ego-hand-matplotlib")

import cv2
import mediapipe as mp
import numpy as np
import yaml


HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rectify an EGO left-camera recording and run MediaPipe Hand Landmarker."
    )
    parser.add_argument("--session", required=True, type=Path, help="EgoViewer recording directory")
    parser.add_argument("--model", required=True, type=Path, help="hand_landmarker.task path")
    parser.add_argument("--output", required=True, type=Path, help="output directory")
    parser.add_argument("--balance", type=float, default=0.0, help="fisheye rectify balance [0, 1]")
    parser.add_argument("--num-hands", type=int, default=2)
    parser.add_argument("--min-detection", type=float, default=0.35)
    parser.add_argument("--min-presence", type=float, default=0.35)
    parser.add_argument("--min-tracking", type=float, default=0.35)
    parser.add_argument("--stride", type=int, default=1, help="process every Nth decoded frame")
    parser.add_argument("--max-frames", type=int, default=0, help="0 processes the complete video")
    parser.add_argument("--no-video", action="store_true", help="do not write annotated MP4")
    return parser.parse_args()


def unique_file(session: Path, suffix: str) -> Path:
    matches = sorted(session.glob(f"*{suffix}"))
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one '*{suffix}' in {session}, found {len(matches)}")
    return matches[0]


def camera_matrices(camera: dict) -> tuple[np.ndarray, np.ndarray]:
    intr = camera["intrinsics"]
    dist = camera["distortion"]
    k = np.array(
        [[intr["fx"], 0.0, intr["cx"]], [0.0, intr["fy"], intr["cy"]], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    d = np.array([dist["k1"], dist["k2"], dist["k3"], dist["k4"]], dtype=np.float64)
    return k, d


def create_stereo_rectification(calibration_path: Path, balance: float):
    if not 0.0 <= balance <= 1.0:
        raise ValueError("--balance must be in [0, 1]")
    with calibration_path.open("r", encoding="utf-8") as stream:
        calibration = yaml.safe_load(stream)
    from camera_models import RectificationOptions, create_stereo_rectification as create_unified
    from ego_data.calibration import stereo_from_ego_yaml
    stereo = stereo_from_ego_yaml(calibration_path)
    result = create_unified(stereo, RectificationOptions(balance=balance), "kb")
    return result.legacy_dict(calibration["calibration_info"]["serial_number"])


def create_left_rectification(calibration_path: Path, balance: float):
    rectification = create_stereo_rectification(calibration_path, balance)
    return (
        rectification["image_size"],
        rectification["map_left_x"],
        rectification["map_left_y"],
        rectification["p1"],
        rectification["calibration_serial"],
    )


def read_timestamps(path: Path) -> list[int]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = csv.DictReader(stream)
        timestamps = [int(row["timestamp_us"]) for row in rows]
    if not timestamps or any(current <= previous for previous, current in zip(timestamps, timestamps[1:])):
        raise RuntimeError(f"timestamps are empty or not strictly increasing: {path}")
    return timestamps


def handedness_for(result, hand_index: int) -> tuple[str, float]:
    if hand_index >= len(result.handedness) or not result.handedness[hand_index]:
        return "Unknown", 0.0
    category = result.handedness[hand_index][0]
    return category.category_name or "Unknown", float(category.score or 0.0)


def draw_hand(image: np.ndarray, landmarks, label: str, score: float, hand_index: int) -> None:
    height, width = image.shape[:2]
    points = [
        (int(round(np.clip(point.x, 0.0, 1.0) * (width - 1))),
         int(round(np.clip(point.y, 0.0, 1.0) * (height - 1))))
        for point in landmarks
    ]
    color = (40, 220, 40) if label == "Left" else (40, 170, 255)
    for start, end in HAND_CONNECTIONS:
        cv2.line(image, points[start], points[end], color, 3, cv2.LINE_AA)
    for point in points:
        cv2.circle(image, point, 4, (245, 245, 245), -1, cv2.LINE_AA)
        cv2.circle(image, point, 5, color, 1, cv2.LINE_AA)
    anchor = points[0]
    cv2.putText(
        image,
        f"#{hand_index} {label} {score:.2f}",
        (max(0, anchor[0] - 35), max(25, anchor[1] - 18)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
        cv2.LINE_AA,
    )


def main() -> int:
    args = parse_args()
    if args.stride < 1 or args.num_hands < 1:
        raise ValueError("--stride and --num-hands must be positive")
    session = args.session.resolve()
    model = args.model.resolve()
    output = args.output.resolve()
    if not session.is_dir():
        raise FileNotFoundError(f"session directory not found: {session}")
    if not model.is_file():
        raise FileNotFoundError(f"model not found: {model}")
    output.mkdir(parents=True, exist_ok=True)

    video_path = unique_file(session, "_camera_left.mp4")
    pts_path = unique_file(session, "_camera_left_pts.csv")
    calibration_path = unique_file(session, "_calibration_camera.yaml")
    timestamps_us = read_timestamps(pts_path)
    image_size, map_x, map_y, p1, calibration_serial = create_left_rectification(
        calibration_path, args.balance
    )

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    decoded_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    decoded_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if (decoded_width, decoded_height) != image_size:
        raise RuntimeError(
            f"video is {decoded_width}x{decoded_height}, calibration is {image_size[0]}x{image_size[1]}"
        )
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    output_fps = source_fps / args.stride if source_fps > 0 else 30.0 / args.stride

    writer = None
    annotated_path = output / "left_annotated.mp4"
    if not args.no_video:
        writer = cv2.VideoWriter(
            str(annotated_path), cv2.VideoWriter_fourcc(*"mp4v"), output_fps, image_size
        )
        if not writer.isOpened():
            raise RuntimeError(f"cannot create annotated video: {annotated_path}")

    frames_path = output / "left_frames.csv"
    landmarks_path = output / "left_landmarks.csv"
    frame_fields = ["frame_index", "timestamp_us", "timestamp_relative_ms", "detected_hands"]
    landmark_fields = [
        "frame_index", "timestamp_us", "hand_index", "handedness", "handedness_score",
        "landmark_index", "x_normalized", "y_normalized", "z_normalized",
        "x_rectified_px", "y_rectified_px", "world_x_m", "world_y_m", "world_z_m",
    ]

    base_options = mp.tasks.BaseOptions(model_asset_path=str(model))
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=args.num_hands,
        min_hand_detection_confidence=args.min_detection,
        min_hand_presence_confidence=args.min_presence,
        min_tracking_confidence=args.min_tracking,
    )

    decoded_frames = 0
    processed_frames = 0
    frames_with_hands = 0
    hand_instances = 0
    landmark_rows = 0
    detected_hands_histogram = {hand_count: 0 for hand_count in range(args.num_hands + 1)}
    first_timestamp_us = timestamps_us[0]
    last_timestamp_ms = -1
    start_time = time.perf_counter()

    with frames_path.open("w", encoding="utf-8", newline="") as frame_stream, \
         landmarks_path.open("w", encoding="utf-8", newline="") as landmark_stream, \
         mp.tasks.vision.HandLandmarker.create_from_options(options) as landmarker:
        frame_writer = csv.DictWriter(frame_stream, fieldnames=frame_fields)
        landmark_writer = csv.DictWriter(landmark_stream, fieldnames=landmark_fields)
        frame_writer.writeheader()
        landmark_writer.writeheader()

        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame_index = decoded_frames
            decoded_frames += 1
            if frame_index >= len(timestamps_us):
                raise RuntimeError("decoded video has more frames than the PTS CSV")
            if frame_index % args.stride != 0:
                continue
            if args.max_frames > 0 and processed_frames >= args.max_frames:
                break

            timestamp_us = timestamps_us[frame_index]
            timestamp_ms = int((timestamp_us - first_timestamp_us) // 1000)
            timestamp_ms = max(timestamp_ms, last_timestamp_ms + 1)
            last_timestamp_ms = timestamp_ms
            rectified = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            rgb = cv2.cvtColor(rectified, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
            result = landmarker.detect_for_video(mp_image, timestamp_ms)
            detected_hands = len(result.hand_landmarks)
            detected_hands_histogram[detected_hands] = detected_hands_histogram.get(detected_hands, 0) + 1
            frame_writer.writerow(
                {
                    "frame_index": frame_index,
                    "timestamp_us": timestamp_us,
                    "timestamp_relative_ms": timestamp_ms,
                    "detected_hands": detected_hands,
                }
            )
            processed_frames += 1
            if detected_hands:
                frames_with_hands += 1
            hand_instances += detected_hands

            annotated = rectified.copy() if writer is not None else None
            for hand_index, landmarks in enumerate(result.hand_landmarks):
                label, score = handedness_for(result, hand_index)
                world = result.hand_world_landmarks[hand_index]
                if annotated is not None:
                    draw_hand(annotated, landmarks, label, score, hand_index)
                for landmark_index, (point, world_point) in enumerate(zip(landmarks, world)):
                    landmark_writer.writerow(
                        {
                            "frame_index": frame_index,
                            "timestamp_us": timestamp_us,
                            "hand_index": hand_index,
                            "handedness": label,
                            "handedness_score": f"{score:.8f}",
                            "landmark_index": landmark_index,
                            "x_normalized": f"{point.x:.9f}",
                            "y_normalized": f"{point.y:.9f}",
                            "z_normalized": f"{point.z:.9f}",
                            "x_rectified_px": f"{point.x * (image_size[0] - 1):.6f}",
                            "y_rectified_px": f"{point.y * (image_size[1] - 1):.6f}",
                            "world_x_m": f"{world_point.x:.9f}",
                            "world_y_m": f"{world_point.y:.9f}",
                            "world_z_m": f"{world_point.z:.9f}",
                        }
                    )
                    landmark_rows += 1
            if writer is not None:
                cv2.putText(
                    annotated,
                    f"frame={frame_index} hands={detected_hands}",
                    (25, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                writer.write(annotated)

    capture.release()
    if writer is not None:
        writer.release()
    elapsed_s = time.perf_counter() - start_time
    summary = {
        "stage": "left_camera_mediapipe_baseline",
        "session": str(session),
        "video": str(video_path),
        "calibration": str(calibration_path),
        "calibration_serial": calibration_serial,
        "model": str(model),
        "mediapipe_version": mp.__version__,
        "opencv_version": cv2.__version__,
        "rectified_size": list(image_size),
        "rectified_projection_matrix": np.asarray(p1).tolist(),
        "balance": args.balance,
        "source_fps": source_fps,
        "output_fps": output_fps,
        "decoded_frames": decoded_frames,
        "processed_frames": processed_frames,
        "frames_with_hands": frames_with_hands,
        "hand_detection_frame_rate": frames_with_hands / processed_frames if processed_frames else 0.0,
        "detected_hands_histogram": {
            str(hand_count): frame_count
            for hand_count, frame_count in sorted(detected_hands_histogram.items())
        },
        "two_hand_frame_rate": (
            detected_hands_histogram.get(2, 0) / processed_frames if processed_frames else 0.0
        ),
        "hand_instances": hand_instances,
        "landmark_rows": landmark_rows,
        "elapsed_seconds": elapsed_s,
        "processing_fps": processed_frames / elapsed_s if elapsed_s > 0 else 0.0,
        "note": "world_* values are MediaPipe model-relative hand coordinates, not stereo camera metric coordinates",
    }
    summary_path = output / "summary.json"
    with summary_path.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
        stream.write("\n")

    print("EGO MediaPipe left-camera baseline")
    print(f"Processed frames: {processed_frames}")
    print(f"Frames with hands: {frames_with_hands} ({summary['hand_detection_frame_rate']:.1%})")
    print(f"Detected hand instances: {hand_instances}")
    print(f"Processing speed: {summary['processing_fps']:.2f} fps")
    print(f"Frames CSV: {frames_path}")
    print(f"Landmarks CSV: {landmarks_path}")
    if writer is not None:
        print(f"Annotated video: {annotated_path}")
    print(f"Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:  # Keep command-line failures concise and actionable.
        print(f"error: {error}", file=sys.stderr)
        sys.exit(1)
