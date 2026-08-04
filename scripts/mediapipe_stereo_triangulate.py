#!/usr/bin/env python3
"""Detect EGO stereo hand landmarks, associate hands, and triangulate 3D joints."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("MPLCONFIGDIR", "/tmp/ego-hand-matplotlib")

import cv2
import mediapipe as mp
import numpy as np

from mediapipe_left_baseline import (
    HAND_CONNECTIONS,
    create_stereo_rectification,
    handedness_for,
    read_timestamps,
    unique_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MediaPipe on paired EGO stereo frames and triangulate 21 hand landmarks."
    )
    parser.add_argument("--session", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--balance", type=float, default=0.0)
    parser.add_argument("--num-hands", type=int, default=2)
    parser.add_argument("--max-delta-us", type=int, default=1500)
    parser.add_argument("--min-detection", type=float, default=0.35)
    parser.add_argument("--min-presence", type=float, default=0.35)
    parser.add_argument("--min-tracking", type=float, default=0.35)
    parser.add_argument("--max-epipolar-px", type=float, default=12.0)
    parser.add_argument("--min-depth-m", type=float, default=0.15)
    parser.add_argument("--max-depth-m", type=float, default=3.0)
    parser.add_argument("--max-reprojection-px", type=float, default=8.0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument("--no-video", action="store_true")
    return parser.parse_args()


def pair_timestamps(left: list[int], right: list[int], max_delta_us: int) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    left_index = 0
    right_index = 0
    while left_index < len(left) and right_index < len(right):
        delta = right[right_index] - left[left_index]
        if abs(delta) <= max_delta_us:
            pairs.append((left_index, right_index))
            left_index += 1
            right_index += 1
        elif delta > 0:
            left_index += 1
        else:
            right_index += 1
    return pairs


def read_frame_at(capture: cv2.VideoCapture, target: int, state: list[int]) -> np.ndarray:
    frame = None
    while state[0] <= target:
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError(f"video ended before frame {target}")
        state[0] += 1
    if frame is None:
        raise RuntimeError(f"failed to decode frame {target}")
    return frame


def detect(landmarker, image_bgr: np.ndarray, timestamp_ms: int, image_size: tuple[int, int]):
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
    result = landmarker.detect_for_video(image, timestamp_ms)
    width, height = image_size
    observations = []
    for index, landmarks in enumerate(result.hand_landmarks):
        label, score = handedness_for(result, index)
        pixels = np.asarray(
            [[point.x * (width - 1), point.y * (height - 1)] for point in landmarks],
            dtype=np.float64,
        )
        observations.append({"index": index, "label": label, "score": score, "pixels": pixels})
    return observations


def stereo_candidate(left: dict, right: dict) -> dict:
    delta_y = np.abs(left["pixels"][:, 1] - right["pixels"][:, 1])
    disparity = left["pixels"][:, 0] - right["pixels"][:, 0]
    plausible = (delta_y < 60.0) & (disparity > 2.0) & (disparity < 300.0)
    median_y = float(np.median(delta_y))
    median_disparity = float(np.median(disparity))
    plausible_count = int(np.count_nonzero(plausible))
    label_penalty = 0.0
    if left["label"] != right["label"] and min(left["score"], right["score"]) > 0.75:
        label_penalty = 8.0
    invalid_penalty = float(21 - plausible_count) * 2.0
    cost = median_y + label_penalty + invalid_penalty
    accepted = plausible_count >= 12 and median_y <= 35.0 and 2.0 < median_disparity < 250.0
    return {
        "left": left,
        "right": right,
        "cost": cost,
        "accepted": accepted,
        "median_y": median_y,
        "median_disparity": median_disparity,
        "plausible_count": plausible_count,
    }


def associate_hands(left_hands: list[dict], right_hands: list[dict]) -> list[dict]:
    if not left_hands or not right_hands:
        return []
    candidates = {
        (left_index, right_index): stereo_candidate(left, right)
        for left_index, left in enumerate(left_hands)
        for right_index, right in enumerate(right_hands)
    }
    best: tuple[int, float, list[dict]] = (-1, float("inf"), [])
    maximum = min(len(left_hands), len(right_hands))
    for match_count in range(1, maximum + 1):
        for left_subset in itertools.combinations(range(len(left_hands)), match_count):
            for right_subset in itertools.combinations(range(len(right_hands)), match_count):
                for right_order in itertools.permutations(right_subset):
                    selected = [candidates[pair] for pair in zip(left_subset, right_order)]
                    if not all(candidate["accepted"] for candidate in selected):
                        continue
                    total_cost = sum(candidate["cost"] for candidate in selected)
                    score = (match_count, -total_cost)
                    if score > (best[0], -best[1]):
                        best = (match_count, total_cost, selected)
    return best[2]


def triangulate(candidate: dict, rectification: dict, args: argparse.Namespace) -> dict:
    left_points = candidate["left"]["pixels"]
    right_points = candidate["right"]["pixels"]
    homogeneous = cv2.triangulatePoints(
        rectification["p1"], rectification["p2"], left_points.T, right_points.T
    )
    points_rectified = (homogeneous[:3] / homogeneous[3]).T
    points_left = (rectification["r1"].T @ points_rectified.T).T

    left_projection = (rectification["p1"] @ np.column_stack((points_rectified, np.ones(21))).T).T
    right_projection = (rectification["p2"] @ np.column_stack((points_rectified, np.ones(21))).T).T
    left_reprojected = left_projection[:, :2] / left_projection[:, 2:3]
    right_reprojected = right_projection[:, :2] / right_projection[:, 2:3]
    reprojection = np.sqrt(
        (
            np.sum((left_reprojected - left_points) ** 2, axis=1)
            + np.sum((right_reprojected - right_points) ** 2, axis=1)
        ) / 2.0
    )
    disparity = left_points[:, 0] - right_points[:, 0]
    epipolar = np.abs(left_points[:, 1] - right_points[:, 1])
    finite = np.all(np.isfinite(points_rectified), axis=1) & np.isfinite(reprojection)
    valid = (
        finite
        & (disparity > 2.0)
        & (epipolar <= args.max_epipolar_px)
        & (points_rectified[:, 2] >= args.min_depth_m)
        & (points_rectified[:, 2] <= args.max_depth_m)
        & (reprojection <= args.max_reprojection_px)
    )
    palm_indices = np.asarray([0, 5, 9, 13, 17])
    valid_palm = palm_indices[valid[palm_indices]]
    if len(valid_palm) >= 2:
        center = np.median(points_left[valid_palm], axis=0)
    elif np.count_nonzero(valid) >= 3:
        center = np.median(points_left[valid], axis=0)
    else:
        center = np.array([np.nan, np.nan, np.nan])
    return {
        **candidate,
        "points_rectified": points_rectified,
        "points_left": points_left,
        "disparity": disparity,
        "epipolar": epipolar,
        "reprojection": reprojection,
        "valid": valid,
        "center": center,
        "center_left_px": np.median(left_points[[0, 5, 9, 13, 17]], axis=0),
    }


class TrackManager:
    def __init__(self, max_missed: int = 30, max_distance_px: float = 280.0):
        self.next_id = 0
        self.max_missed = max_missed
        self.max_distance_px = max_distance_px
        self.tracks: dict[int, dict] = {}

    def assign(self, matches: list[dict]) -> None:
        for track in self.tracks.values():
            track["missed"] += 1
        usable = [
            index for index, match in enumerate(matches)
            if np.all(np.isfinite(match["center_left_px"]))
        ]
        track_ids = list(self.tracks)
        assignment: dict[int, int] = {}
        if usable and track_ids:
            count = min(len(usable), len(track_ids))
            best_cost = float("inf")
            best_pairs = []
            for match_subset in itertools.combinations(usable, count):
                for track_subset in itertools.combinations(track_ids, count):
                    for track_order in itertools.permutations(track_subset):
                        pairs = list(zip(match_subset, track_order))
                        distances = []
                        cost = 0.0
                        for match_index, track_id in pairs:
                            track = self.tracks[track_id]
                            prediction = track["position"] + track["velocity"] * min(track["missed"], 3)
                            distance = float(np.linalg.norm(matches[match_index]["center_left_px"] - prediction))
                            distances.append(distance)
                            label_penalty = 0.0
                            match_label = matches[match_index]["left"]["label"]
                            if match_label != track["label"] and matches[match_index]["left"]["score"] > 0.80:
                                label_penalty = 80.0
                            cost += distance + label_penalty + track["missed"] * 2.0
                        if all(distance < self.max_distance_px for distance in distances) and cost < best_cost:
                            best_cost = cost
                            best_pairs = pairs
            assignment.update(best_pairs)
        for match_index, match in enumerate(matches):
            if match_index not in assignment:
                assignment[match_index] = self.next_id
                self.tracks[self.next_id] = {
                    "position": match["center_left_px"].copy(),
                    "velocity": np.zeros(2, dtype=np.float64),
                    "missed": 0,
                    "label": match["left"]["label"],
                }
                self.next_id += 1
            track_id = assignment[match_index]
            match["track_id"] = track_id
            track = self.tracks[track_id]
            new_position = match["center_left_px"]
            observed_velocity = new_position - track["position"]
            track["velocity"] = 0.65 * track["velocity"] + 0.35 * observed_velocity
            track["position"] = new_position.copy()
            track["missed"] = 0
            if match["left"]["score"] > 0.90:
                track["label"] = match["left"]["label"]
        self.tracks = {
            track_id: track for track_id, track in self.tracks.items()
            if track["missed"] <= self.max_missed
        }


def draw_observation(image: np.ndarray, observation: dict, color: tuple[int, int, int], text: str) -> None:
    points = np.rint(observation["pixels"]).astype(int)
    for start, end in HAND_CONNECTIONS:
        cv2.line(image, tuple(points[start]), tuple(points[end]), color, 2, cv2.LINE_AA)
    for point in points:
        cv2.circle(image, tuple(point), 3, color, -1, cv2.LINE_AA)
    anchor = points[0]
    cv2.putText(image, text, (max(0, anchor[0] - 30), max(25, anchor[1] - 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)


def make_options(args: argparse.Namespace, model: Path):
    return mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=args.num_hands,
        min_hand_detection_confidence=args.min_detection,
        min_hand_presence_confidence=args.min_presence,
        min_tracking_confidence=args.min_tracking,
    )


def main() -> int:
    args = parse_args()
    if args.stride < 1 or args.num_hands < 1 or args.max_delta_us < 0:
        raise ValueError("stride/num-hands must be positive and max-delta-us non-negative")
    session = args.session.resolve()
    model = args.model.resolve()
    output = args.output.resolve()
    if not session.is_dir() or not model.is_file():
        raise FileNotFoundError("session or model does not exist")
    output.mkdir(parents=True, exist_ok=True)

    calibration_path = unique_file(session, "_calibration_camera.yaml")
    left_video = unique_file(session, "_camera_left.mp4")
    right_video = unique_file(session, "_camera_right.mp4")
    left_pts_path = unique_file(session, "_camera_left_pts.csv")
    right_pts_path = unique_file(session, "_camera_right_pts.csv")
    left_timestamps = read_timestamps(left_pts_path)
    right_timestamps = read_timestamps(right_pts_path)
    timestamp_pairs = pair_timestamps(left_timestamps, right_timestamps, args.max_delta_us)
    rectification = create_stereo_rectification(calibration_path, args.balance)
    image_size = rectification["image_size"]

    left_capture = cv2.VideoCapture(str(left_video))
    right_capture = cv2.VideoCapture(str(right_video))
    if not left_capture.isOpened() or not right_capture.isOpened():
        raise RuntimeError("cannot open one or both stereo videos")
    left_state = [0]
    right_state = [0]

    video_path = output / "stereo_annotated.mp4"
    writer = None
    preview_size = (image_size[0], image_size[1] // 2)
    if not args.no_video:
        source_fps = float(left_capture.get(cv2.CAP_PROP_FPS))
        fps = (source_fps if source_fps > 0 else 30.0) / args.stride
        writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, preview_size)
        if not writer.isOpened():
            raise RuntimeError(f"cannot create {video_path}")

    frames_path = output / "stereo_frames.csv"
    landmarks_path = output / "stereo_landmarks_3d.csv"
    frame_fields = [
        "pair_index", "left_index", "right_index", "left_timestamp_us", "right_timestamp_us",
        "timestamp_delta_us", "left_hands", "right_hands", "matched_hands", "valid_3d_points",
        "median_epipolar_px", "median_reprojection_px",
    ]
    landmark_fields = [
        "pair_index", "left_index", "right_index", "track_id", "match_index",
        "left_hand_index", "right_hand_index", "left_handedness", "right_handedness",
        "left_handedness_score", "right_handedness_score", "landmark_index",
        "left_x_rectified_px", "left_y_rectified_px", "right_x_rectified_px", "right_y_rectified_px",
        "disparity_px", "epipolar_error_px", "valid_3d", "reprojection_error_px",
        "x_rectified_m", "y_rectified_m", "z_rectified_m", "x_left_camera_m", "y_left_camera_m", "z_left_camera_m",
    ]

    tracker = TrackManager()
    processed_pairs = 0
    pairs_with_matches = 0
    matched_hand_instances = 0
    valid_points_total = 0
    observed_track_ids = set()
    valid_points_by_landmark = np.zeros(21, dtype=np.int64)
    observations_by_landmark = np.zeros(21, dtype=np.int64)
    all_valid_epipolar = []
    all_valid_reprojection = []
    all_valid_depth = []
    start_time = time.perf_counter()
    first_timestamp_us = min(left_timestamps[0], right_timestamps[0])
    last_left_ms = -1
    last_right_ms = -1

    with frames_path.open("w", encoding="utf-8", newline="") as frame_stream, \
         landmarks_path.open("w", encoding="utf-8", newline="") as landmark_stream, \
         mp.tasks.vision.HandLandmarker.create_from_options(make_options(args, model)) as left_landmarker, \
         mp.tasks.vision.HandLandmarker.create_from_options(make_options(args, model)) as right_landmarker:
        frame_writer = csv.DictWriter(frame_stream, fieldnames=frame_fields)
        landmark_writer = csv.DictWriter(landmark_stream, fieldnames=landmark_fields)
        frame_writer.writeheader()
        landmark_writer.writeheader()

        for pair_index, (left_index, right_index) in enumerate(timestamp_pairs):
            left_frame = read_frame_at(left_capture, left_index, left_state)
            right_frame = read_frame_at(right_capture, right_index, right_state)
            if pair_index % args.stride != 0:
                continue
            if args.max_pairs > 0 and processed_pairs >= args.max_pairs:
                break
            if left_frame.shape[1::-1] != image_size or right_frame.shape[1::-1] != image_size:
                raise RuntimeError("decoded resolution differs from calibration")

            left_rectified = cv2.remap(left_frame, rectification["map_left_x"], rectification["map_left_y"], cv2.INTER_LINEAR)
            right_rectified = cv2.remap(right_frame, rectification["map_right_x"], rectification["map_right_y"], cv2.INTER_LINEAR)
            left_ms = max(int((left_timestamps[left_index] - first_timestamp_us) // 1000), last_left_ms + 1)
            right_ms = max(int((right_timestamps[right_index] - first_timestamp_us) // 1000), last_right_ms + 1)
            last_left_ms = left_ms
            last_right_ms = right_ms
            left_hands = detect(left_landmarker, left_rectified, left_ms, image_size)
            right_hands = detect(right_landmarker, right_rectified, right_ms, image_size)
            matches = [triangulate(candidate, rectification, args) for candidate in associate_hands(left_hands, right_hands)]
            tracker.assign(matches)

            frame_valid = int(sum(np.count_nonzero(match["valid"]) for match in matches))
            valid_epipolar = np.concatenate([match["epipolar"][match["valid"]] for match in matches]) if matches else np.array([])
            valid_reprojection = np.concatenate([match["reprojection"][match["valid"]] for match in matches]) if matches else np.array([])
            frame_writer.writerow({
                "pair_index": pair_index,
                "left_index": left_index,
                "right_index": right_index,
                "left_timestamp_us": left_timestamps[left_index],
                "right_timestamp_us": right_timestamps[right_index],
                "timestamp_delta_us": right_timestamps[right_index] - left_timestamps[left_index],
                "left_hands": len(left_hands),
                "right_hands": len(right_hands),
                "matched_hands": len(matches),
                "valid_3d_points": frame_valid,
                "median_epipolar_px": f"{np.median(valid_epipolar):.6f}" if len(valid_epipolar) else "nan",
                "median_reprojection_px": f"{np.median(valid_reprojection):.6f}" if len(valid_reprojection) else "nan",
            })

            for match_index, match in enumerate(matches):
                observed_track_ids.add(match["track_id"])
                for landmark_index in range(21):
                    valid = bool(match["valid"][landmark_index])
                    observations_by_landmark[landmark_index] += 1
                    if valid:
                        valid_points_by_landmark[landmark_index] += 1
                    rectified_point = match["points_rectified"][landmark_index]
                    left_point_3d = match["points_left"][landmark_index]
                    landmark_writer.writerow({
                        "pair_index": pair_index, "left_index": left_index, "right_index": right_index,
                        "track_id": match["track_id"], "match_index": match_index,
                        "left_hand_index": match["left"]["index"], "right_hand_index": match["right"]["index"],
                        "left_handedness": match["left"]["label"], "right_handedness": match["right"]["label"],
                        "left_handedness_score": f"{match['left']['score']:.8f}",
                        "right_handedness_score": f"{match['right']['score']:.8f}",
                        "landmark_index": landmark_index,
                        "left_x_rectified_px": f"{match['left']['pixels'][landmark_index, 0]:.6f}",
                        "left_y_rectified_px": f"{match['left']['pixels'][landmark_index, 1]:.6f}",
                        "right_x_rectified_px": f"{match['right']['pixels'][landmark_index, 0]:.6f}",
                        "right_y_rectified_px": f"{match['right']['pixels'][landmark_index, 1]:.6f}",
                        "disparity_px": f"{match['disparity'][landmark_index]:.6f}",
                        "epipolar_error_px": f"{match['epipolar'][landmark_index]:.6f}",
                        "valid_3d": int(valid),
                        "reprojection_error_px": f"{match['reprojection'][landmark_index]:.6f}",
                        "x_rectified_m": f"{rectified_point[0]:.9f}" if valid else "nan",
                        "y_rectified_m": f"{rectified_point[1]:.9f}" if valid else "nan",
                        "z_rectified_m": f"{rectified_point[2]:.9f}" if valid else "nan",
                        "x_left_camera_m": f"{left_point_3d[0]:.9f}" if valid else "nan",
                        "y_left_camera_m": f"{left_point_3d[1]:.9f}" if valid else "nan",
                        "z_left_camera_m": f"{left_point_3d[2]:.9f}" if valid else "nan",
                    })

            if matches:
                pairs_with_matches += 1
            processed_pairs += 1
            matched_hand_instances += len(matches)
            valid_points_total += frame_valid
            all_valid_epipolar.extend(valid_epipolar.tolist())
            all_valid_reprojection.extend(valid_reprojection.tolist())
            for match in matches:
                all_valid_depth.extend(match["points_left"][match["valid"], 2].tolist())

            if writer is not None:
                annotated_left = left_rectified.copy()
                annotated_right = right_rectified.copy()
                colors = [(30, 220, 30), (20, 170, 255), (255, 80, 180), (255, 220, 40)]
                matched_left = set()
                matched_right = set()
                for match in matches:
                    color = colors[match["track_id"] % len(colors)]
                    valid_count = int(np.count_nonzero(match["valid"]))
                    draw_observation(annotated_left, match["left"], color, f"T{match['track_id']} {valid_count}/21")
                    draw_observation(annotated_right, match["right"], color, f"T{match['track_id']} {valid_count}/21")
                    matched_left.add(match["left"]["index"])
                    matched_right.add(match["right"]["index"])
                for observation in left_hands:
                    if observation["index"] not in matched_left:
                        draw_observation(annotated_left, observation, (120, 120, 120), "unmatched")
                for observation in right_hands:
                    if observation["index"] not in matched_right:
                        draw_observation(annotated_right, observation, (120, 120, 120), "unmatched")
                pair_image = cv2.hconcat([annotated_left, annotated_right])
                cv2.putText(pair_image, f"pair={pair_index} dt={right_timestamps[right_index]-left_timestamps[left_index]}us matches={len(matches)}",
                            (25, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
                pair_image = cv2.resize(pair_image, preview_size, interpolation=cv2.INTER_AREA)
                writer.write(pair_image)

    left_capture.release()
    right_capture.release()
    if writer is not None:
        writer.release()
    elapsed = time.perf_counter() - start_time

    def percentile(values, level):
        return float(np.percentile(values, level)) if values else None

    summary = {
        "stage": "stereo_mediapipe_triangulation_baseline",
        "session": str(session),
        "model": str(model),
        "mediapipe_version": mp.__version__,
        "opencv_version": cv2.__version__,
        "calibration_serial": rectification["calibration_serial"],
        "rectified_size": list(image_size),
        "p1": np.asarray(rectification["p1"]).tolist(),
        "p2": np.asarray(rectification["p2"]).tolist(),
        "timestamp_pairs_available": len(timestamp_pairs),
        "processed_pairs": processed_pairs,
        "pairs_with_stereo_matches": pairs_with_matches,
        "stereo_match_pair_rate": pairs_with_matches / processed_pairs if processed_pairs else 0.0,
        "matched_hand_instances": matched_hand_instances,
        "track_ids": sorted(observed_track_ids),
        "track_count": len(observed_track_ids),
        "valid_3d_points": valid_points_total,
        "valid_3d_rate_of_matched": valid_points_total / (matched_hand_instances * 21) if matched_hand_instances else 0.0,
        "epipolar_abs_px_median": percentile(all_valid_epipolar, 50),
        "epipolar_abs_px_p95": percentile(all_valid_epipolar, 95),
        "reprojection_px_median": percentile(all_valid_reprojection, 50),
        "reprojection_px_p95": percentile(all_valid_reprojection, 95),
        "depth_m_median": percentile(all_valid_depth, 50),
        "depth_m_p05": percentile(all_valid_depth, 5),
        "depth_m_p95": percentile(all_valid_depth, 95),
        "valid_3d_rate_by_landmark": {
            str(index): (
                float(valid_points_by_landmark[index] / observations_by_landmark[index])
                if observations_by_landmark[index] else 0.0
            )
            for index in range(21)
        },
        "elapsed_seconds": elapsed,
        "processing_fps": processed_pairs / elapsed if elapsed > 0 else 0.0,
        "filters": {
            "max_delta_us": args.max_delta_us,
            "max_epipolar_px": args.max_epipolar_px,
            "min_depth_m": args.min_depth_m,
            "max_depth_m": args.max_depth_m,
            "max_reprojection_px": args.max_reprojection_px,
        },
        "coordinate_note": "x_left_camera/y_left_camera/z_left_camera are metric coordinates in the original left optical frame",
    }
    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("EGO MediaPipe stereo triangulation baseline")
    print(f"Processed stereo pairs: {processed_pairs}")
    print(f"Pairs with stereo hand matches: {pairs_with_matches} ({summary['stereo_match_pair_rate']:.1%})")
    print(f"Matched hand instances: {matched_hand_instances}")
    print(f"Valid 3D landmarks: {valid_points_total} ({summary['valid_3d_rate_of_matched']:.1%})")
    print(f"Epipolar median/P95: {summary['epipolar_abs_px_median']:.3f}/{summary['epipolar_abs_px_p95']:.3f} px")
    print(f"Reprojection median/P95: {summary['reprojection_px_median']:.3f}/{summary['reprojection_px_p95']:.3f} px")
    print(f"Processing speed: {summary['processing_fps']:.2f} stereo fps")
    print(f"3D landmarks CSV: {landmarks_path}")
    if writer is not None:
        print(f"Annotated stereo video: {video_path}")
    print(f"Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(1)
