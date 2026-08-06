#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import time

import cv2
import numpy as np
import yaml


TRACK_COLORS = {"Left": (20, 145, 255), "Right": (255, 125, 35)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fuse Basalt camera VIO with camera-frame hand 6D poses."
    )
    parser.add_argument("--session", required=True, type=Path)
    parser.add_argument("--basalt-dataset", required=True, type=Path)
    parser.add_argument("--basalt-trajectory", type=Path)
    parser.add_argument("--hand-pose-csv", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--trajectory-length", type=int, default=300)
    parser.add_argument("--max-world-jump-m", type=float, default=0.15)
    parser.add_argument("--axis-length-m", type=float, default=0.055)
    parser.add_argument("--no-video", action="store_true")
    return parser.parse_args()


def unique_file(directory: Path, suffix: str) -> Path:
    matches = sorted(directory.glob(f"*{suffix}"))
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one '*{suffix}' in {directory}, found {len(matches)}"
        )
    return matches[0]


def make_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.asarray(rotation, dtype=np.float64)
    transform[:3, 3] = np.asarray(translation, dtype=np.float64)
    return transform


def invert_transform(transform: np.ndarray) -> np.ndarray:
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rotation.T
    result[:3, 3] = -(rotation.T @ translation)
    return result


def quaternion_xyzw_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float64)
    quaternion /= np.linalg.norm(quaternion)
    x, y, z, w = quaternion
    return np.asarray([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def quaternion_xyzw_from_matrix(rotation: np.ndarray) -> np.ndarray:
    rotation = np.asarray(rotation, dtype=np.float64)
    quaternion = np.empty(4, dtype=np.float64)
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quaternion[:] = (
            (rotation[2, 1] - rotation[1, 2]) / scale,
            (rotation[0, 2] - rotation[2, 0]) / scale,
            (rotation[1, 0] - rotation[0, 1]) / scale,
            0.25 * scale,
        )
    else:
        index = int(np.argmax(np.diag(rotation)))
        if index == 0:
            scale = math.sqrt(1 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2
            quaternion[:] = (0.25 * scale, (rotation[0, 1] + rotation[1, 0]) / scale,
                             (rotation[0, 2] + rotation[2, 0]) / scale,
                             (rotation[2, 1] - rotation[1, 2]) / scale)
        elif index == 1:
            scale = math.sqrt(1 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2
            quaternion[:] = ((rotation[0, 1] + rotation[1, 0]) / scale, 0.25 * scale,
                             (rotation[1, 2] + rotation[2, 1]) / scale,
                             (rotation[0, 2] - rotation[2, 0]) / scale)
        else:
            scale = math.sqrt(1 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2
            quaternion[:] = ((rotation[0, 2] + rotation[2, 0]) / scale,
                             (rotation[1, 2] + rotation[2, 1]) / scale, 0.25 * scale,
                             (rotation[1, 0] - rotation[0, 1]) / scale)
    quaternion /= np.linalg.norm(quaternion)
    if quaternion[3] < 0:
        quaternion *= -1
    return quaternion


def rpy_zyx_from_matrix(rotation: np.ndarray) -> np.ndarray:
    pitch = math.asin(float(np.clip(-rotation[2, 0], -1.0, 1.0)))
    if abs(math.cos(pitch)) > 1e-8:
        roll = math.atan2(rotation[2, 1], rotation[2, 2])
        yaw = math.atan2(rotation[1, 0], rotation[0, 0])
    else:
        roll = 0.0
        yaw = math.atan2(-rotation[0, 1], rotation[1, 1])
    return np.asarray((roll, pitch, yaw), dtype=np.float64)


def pose_from_basalt_json(entry: dict) -> np.ndarray:
    quaternion = np.asarray((entry["qx"], entry["qy"], entry["qz"], entry["qw"]))
    return make_transform(
        quaternion_xyzw_to_matrix(quaternion),
        (entry["px"], entry["py"], entry["pz"]),
    )


def load_basalt_trajectory(path: Path) -> dict[int, np.ndarray]:
    poses = {}
    with path.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.reader(line for line in stream if not line.startswith("#")):
            if len(row) != 8:
                raise RuntimeError(f"invalid Basalt trajectory row: {row}")
            timestamp_ns = int(row[0])
            translation = np.asarray([float(value) for value in row[1:4]])
            quaternion_wxyz = np.asarray([float(value) for value in row[4:8]])
            quaternion_xyzw = quaternion_wxyz[[1, 2, 3, 0]]
            poses[timestamp_ns] = make_transform(
                quaternion_xyzw_to_matrix(quaternion_xyzw), translation
            )
    if not poses:
        raise RuntimeError(f"empty Basalt trajectory: {path}")
    return poses


def load_timestamp_map(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = [
            {
                "pair_index": int(row["pair_index"]),
                "left_index": int(row["left_index"]),
                "right_index": int(row["right_index"]),
                "left_timestamp_us": int(row["left_timestamp_us"]),
                "right_timestamp_us": int(row["right_timestamp_us"]),
                "basalt_timestamp_ns": int(row["basalt_timestamp_ns"]),
            }
            for row in csv.DictReader(stream)
        ]
    if not rows:
        raise RuntimeError(f"empty timestamp map: {path}")
    return rows


def hand_transform_from_row(row: dict[str, str]) -> np.ndarray:
    rotation = np.asarray([
        [float(row[f"r{r}{c}"]) for c in range(3)] for r in range(3)
    ])
    translation = np.asarray((float(row["x_m"]), float(row["y_m"]), float(row["z_m"])))
    return make_transform(rotation, translation)


def load_hand_poses(path: Path) -> dict[int, list[dict]]:
    poses: dict[int, list[dict]] = {}
    with path.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            pair_index = int(row["pair_index"])
            poses.setdefault(pair_index, []).append({
                "track_id": int(row["track_id"]),
                "handedness": row["handedness"],
                "T_c_h": hand_transform_from_row(row),
            })
    if not poses:
        raise RuntimeError(f"empty hand pose file: {path}")
    return poses


def camera_matrices(calibration_path: Path) -> tuple[np.ndarray, np.ndarray]:
    with calibration_path.open("r", encoding="utf-8") as stream:
        calibration = yaml.safe_load(stream)
    camera = calibration["cameras"][0]
    intrinsics = camera["intrinsics"]
    distortion = camera["distortion"]
    matrix = np.asarray([
        [intrinsics["fx"], 0.0, intrinsics["cx"]],
        [0.0, intrinsics["fy"], intrinsics["cy"]],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    coefficients = np.asarray(
        [distortion[f"k{index}"] for index in range(1, 5)], dtype=np.float64
    )
    return matrix, coefficients


def project_fisheye(
    points_camera: np.ndarray, matrix: np.ndarray, distortion: np.ndarray
) -> np.ndarray:
    projected, _ = cv2.fisheye.projectPoints(
        np.asarray(points_camera, dtype=np.float64).reshape(-1, 1, 3),
        np.zeros(3), np.zeros(3), matrix, distortion,
    )
    return projected[:, 0]


def transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    return (transform[:3, :3] @ points.T).T + transform[:3, 3]


def shade_color(color: tuple[int, int, int], scale: float) -> tuple[int, int, int]:
    return tuple(int(np.clip(channel * scale, 0, 255)) for channel in color)


def draw_world_trail(
    image: np.ndarray,
    points_world: list[np.ndarray],
    t_camera_world: np.ndarray,
    matrix: np.ndarray,
    distortion: np.ndarray,
    color: tuple[int, int, int],
    maximum_frames: int,
    maximum_jump_m: float,
) -> None:
    if not points_world:
        return
    points_world_array = np.asarray(points_world[-maximum_frames:], dtype=np.float64)
    points_camera = transform_points(t_camera_world, points_world_array)
    finite = np.isfinite(points_camera).all(axis=1) & (points_camera[:, 2] > 0.03)
    safe = points_camera.copy()
    safe[~finite] = (0.0, 0.0, 1.0)
    pixels = project_fisheye(safe, matrix, distortion)
    height, width = image.shape[:2]
    inside = finite & (pixels[:, 0] >= 0) & (pixels[:, 0] < width) & (pixels[:, 1] >= 0) & (pixels[:, 1] < height)
    for index in range(1, len(points_world_array)):
        if not (inside[index - 1] and inside[index]):
            continue
        if np.linalg.norm(points_world_array[index] - points_world_array[index - 1]) > maximum_jump_m:
            continue
        progress = index / max(len(points_world_array) - 1, 1)
        cv2.line(
            image,
            tuple(np.rint(pixels[index - 1]).astype(int)),
            tuple(np.rint(pixels[index]).astype(int)),
            shade_color(color, 0.25 + 0.75 * progress),
            max(1, int(round(1 + 3 * progress))), cv2.LINE_AA,
        )


def draw_hand_axes(
    image: np.ndarray,
    t_camera_hand: np.ndarray,
    matrix: np.ndarray,
    distortion: np.ndarray,
    length_m: float,
) -> tuple[int, int] | None:
    points = np.vstack((
        t_camera_hand[:3, 3],
        transform_points(t_camera_hand, np.asarray(((length_m, 0, 0), (0, length_m, 0), (0, 0, length_m)))),
    ))
    if np.any(points[:, 2] <= 0.03):
        return None
    pixels = project_fisheye(points, matrix, distortion)
    origin = tuple(np.rint(pixels[0]).astype(int))
    for endpoint, color, label in zip(
        pixels[1:], ((40, 40, 245), (40, 220, 40), (245, 120, 40)), ("X", "Y", "Z")
    ):
        end = tuple(np.rint(endpoint).astype(int))
        cv2.arrowedLine(image, origin, end, color, 3, cv2.LINE_AA, tipLength=0.18)
        cv2.putText(image, label, end, cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    return origin


def draw_metric_label(
    image: np.ndarray, origin: tuple[int, int], position_world: np.ndarray,
    color: tuple[int, int, int], label: str,
) -> None:
    millimetres = position_world * 1000.0
    text = f"{label} W x {millimetres[0]:+.0f} y {millimetres[1]:+.0f} z {millimetres[2]:+.0f} mm"
    (width, height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    x = int(np.clip(origin[0] + 12, 5, max(image.shape[1] - width - 12, 5)))
    y = int(np.clip(origin[1] - 12, height + 10, image.shape[0] - 8))
    overlay = image.copy()
    cv2.rectangle(overlay, (x - 5, y - height - 6), (x + width + 6, y + 5), (10, 13, 18), -1)
    cv2.addWeighted(overlay, 0.78, image, 0.22, 0, image)
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


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


def serialize_pose(prefix: str, transform: np.ndarray) -> dict[str, str]:
    position = transform[:3, 3]
    rotation = transform[:3, :3]
    quaternion = quaternion_xyzw_from_matrix(rotation)
    rpy = rpy_zyx_from_matrix(rotation)
    values = {
        f"{prefix}_x_m": f"{position[0]:.9f}",
        f"{prefix}_y_m": f"{position[1]:.9f}",
        f"{prefix}_z_m": f"{position[2]:.9f}",
        f"{prefix}_roll_rad": f"{rpy[0]:.9f}",
        f"{prefix}_pitch_rad": f"{rpy[1]:.9f}",
        f"{prefix}_yaw_rad": f"{rpy[2]:.9f}",
        f"{prefix}_qx": f"{quaternion[0]:.9f}",
        f"{prefix}_qy": f"{quaternion[1]:.9f}",
        f"{prefix}_qz": f"{quaternion[2]:.9f}",
        f"{prefix}_qw": f"{quaternion[3]:.9f}",
    }
    values.update({
        f"{prefix}_r{row}{column}": f"{rotation[row, column]:.9f}"
        for row in range(3) for column in range(3)
    })
    return values


def main() -> int:
    args = parse_args()
    if args.trajectory_length < 1 or args.max_world_jump_m <= 0 or args.axis_length_m <= 0:
        raise ValueError("invalid trajectory visualization parameters")
    session = args.session.resolve()
    basalt_dataset = args.basalt_dataset.resolve()
    trajectory_path = (
        args.basalt_trajectory.resolve()
        if args.basalt_trajectory else basalt_dataset / "trajectory.csv"
    )
    hand_pose_path = args.hand_pose_csv.resolve()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    timestamp_rows = load_timestamp_map(basalt_dataset / "ego_timestamp_map.csv")
    imu_poses = load_basalt_trajectory(trajectory_path)
    hand_poses = load_hand_poses(hand_pose_path)
    with (basalt_dataset / "calibration.json").open("r", encoding="utf-8") as stream:
        basalt_calibration = json.load(stream)["value0"]
    t_imu_camera = pose_from_basalt_json(basalt_calibration["T_imu_cam"][0])

    missing_timestamps = [
        row["basalt_timestamp_ns"] for row in timestamp_rows
        if row["basalt_timestamp_ns"] not in imu_poses
    ]
    if missing_timestamps:
        raise RuntimeError(f"Basalt trajectory is missing {len(missing_timestamps)} image timestamps")

    t_world_camera_raw = {
        row["pair_index"]: imu_poses[row["basalt_timestamp_ns"]] @ t_imu_camera
        for row in timestamp_rows
    }
    first_pair = timestamp_rows[0]["pair_index"]
    first_camera_raw = t_world_camera_raw[first_pair]
    # Gravity-aligned Basalt world axes are retained; only the origin is shifted
    # to the first left-camera optical centre.
    t_gravity_world = make_transform(np.eye(3), -first_camera_raw[:3, 3])
    first_camera_inverse = invert_transform(first_camera_raw)
    camera_poses = {}
    fused_hands: dict[int, list[dict]] = {}
    track_origins: dict[int, np.ndarray] = {}
    for row in timestamp_rows:
        pair_index = row["pair_index"]
        raw_pose = t_world_camera_raw[pair_index]
        t_gravity_camera = t_gravity_world @ raw_pose
        t_first_camera = first_camera_inverse @ raw_pose
        camera_poses[pair_index] = {
            "T_g_c": t_gravity_camera,
            "T_c0_c": t_first_camera,
            **row,
        }
        for hand in hand_poses.get(pair_index, []):
            t_gravity_hand = t_gravity_camera @ hand["T_c_h"]
            t_first_hand = t_first_camera @ hand["T_c_h"]
            track_id = hand["track_id"]
            track_origins.setdefault(track_id, t_gravity_hand[:3, 3].copy())
            fused_hands.setdefault(pair_index, []).append({
                **hand,
                "T_g_h": t_gravity_hand,
                "T_c0_h": t_first_hand,
                "delta_g_m": t_gravity_hand[:3, 3] - track_origins[track_id],
            })

    camera_fields = [
        "pair_index", "left_index", "right_index", "left_timestamp_us",
        "right_timestamp_us", "basalt_timestamp_ns",
    ]
    pose_components = [
        "x_m", "y_m", "z_m", "roll_rad", "pitch_rad", "yaw_rad",
        "qx", "qy", "qz", "qw",
    ] + [f"r{row}{column}" for row in range(3) for column in range(3)]
    camera_fields += [f"world_camera_{component}" for component in pose_components]
    camera_fields += [f"first_camera_{component}" for component in pose_components]
    with (output / "camera_trajectory_world.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=camera_fields)
        writer.writeheader()
        for row in timestamp_rows:
            pose = camera_poses[row["pair_index"]]
            output_row = {key: row[key] for key in camera_fields[:6]}
            output_row.update(serialize_pose("world_camera", pose["T_g_c"]))
            output_row.update(serialize_pose("first_camera", pose["T_c0_c"]))
            writer.writerow(output_row)

    hand_fields = [
        "pair_index", "left_index", "left_timestamp_us", "basalt_timestamp_ns",
        "track_id", "handedness", "delta_world_x_m", "delta_world_y_m", "delta_world_z_m",
    ]
    hand_fields += [f"world_hand_{component}" for component in pose_components]
    hand_fields += [f"first_camera_hand_{component}" for component in pose_components]
    with (output / "hand_trajectory_world.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=hand_fields)
        writer.writeheader()
        for row in timestamp_rows:
            for hand in fused_hands.get(row["pair_index"], []):
                output_row = {
                    "pair_index": row["pair_index"], "left_index": row["left_index"],
                    "left_timestamp_us": row["left_timestamp_us"],
                    "basalt_timestamp_ns": row["basalt_timestamp_ns"],
                    "track_id": hand["track_id"], "handedness": hand["handedness"],
                    "delta_world_x_m": f"{hand['delta_g_m'][0]:.9f}",
                    "delta_world_y_m": f"{hand['delta_g_m'][1]:.9f}",
                    "delta_world_z_m": f"{hand['delta_g_m'][2]:.9f}",
                }
                output_row.update(serialize_pose("world_hand", hand["T_g_h"]))
                output_row.update(serialize_pose("first_camera_hand", hand["T_c0_h"]))
                writer.writerow(output_row)

    video_path = output / "world_hand_trajectory_overlay.mp4"
    preview_path = output / "preview_montage.jpg"
    processed = 0
    previews = []
    started = time.perf_counter()
    if not args.no_video:
        source_video = unique_file(session, "_camera_left.mp4")
        calibration_path = unique_file(session, "_calibration_camera.yaml")
        matrix, distortion = camera_matrices(calibration_path)
        capture = cv2.VideoCapture(str(source_video))
        if not capture.isOpened():
            raise RuntimeError(f"cannot open {source_video}")
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
        writer = cv2.VideoWriter(
            str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )
        if not writer.isOpened():
            raise RuntimeError(f"cannot create {video_path}")
        state = [0]
        histories: dict[int, list[np.ndarray]] = {}
        preview_indices = set(np.linspace(0, len(timestamp_rows) - 1, 6).round().astype(int))
        for ordinal, row in enumerate(timestamp_rows):
            frame = read_frame_at(capture, row["left_index"], state)
            pose = camera_poses[row["pair_index"]]
            t_camera_gravity = invert_transform(pose["T_g_c"])
            for hand in fused_hands.get(row["pair_index"], []):
                histories.setdefault(hand["track_id"], []).append(hand["T_g_h"][:3, 3].copy())
            for hand in fused_hands.get(row["pair_index"], []):
                color = TRACK_COLORS.get(hand["handedness"], (80, 220, 100))
                draw_world_trail(
                    frame, histories[hand["track_id"]], t_camera_gravity,
                    matrix, distortion, color, args.trajectory_length,
                    args.max_world_jump_m,
                )
                origin = draw_hand_axes(
                    frame, hand["T_c_h"], matrix, distortion, args.axis_length_m
                )
                if origin is not None:
                    cv2.circle(frame, origin, 7, color, -1, cv2.LINE_AA)
                    draw_metric_label(
                        frame, origin, hand["T_g_h"][:3, 3], color,
                        f"{hand['handedness']} T{hand['track_id']}",
                    )
            camera_mm = pose["T_g_c"][:3, 3] * 1000.0
            cv2.rectangle(frame, (0, 0), (width, 72), (10, 13, 18), -1)
            cv2.putText(
                frame, "EGO BASALT WORLD-FRAME HAND TRAJECTORY", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.72, (245, 245, 245), 2, cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"pair {row['pair_index']} | camera W x {camera_mm[0]:+.0f} "
                f"y {camera_mm[1]:+.0f} z {camera_mm[2]:+.0f} mm | Z is gravity up",
                (20, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (190, 210, 230), 1, cv2.LINE_AA,
            )
            writer.write(frame)
            if ordinal in preview_indices:
                previews.append(cv2.resize(frame, (640, 520), interpolation=cv2.INTER_AREA))
            processed += 1
        capture.release()
        writer.release()
        if previews:
            while len(previews) < 6:
                previews.append(previews[-1])
            montage = np.vstack((np.hstack(previews[:3]), np.hstack(previews[3:6])))
            cv2.imwrite(str(preview_path), montage)

    camera_positions = np.asarray([
        camera_poses[row["pair_index"]]["T_g_c"][:3, 3] for row in timestamp_rows
    ])
    camera_steps = np.linalg.norm(np.diff(camera_positions, axis=0), axis=1)
    track_summary = {}
    for track_id in sorted(track_origins):
        positions = np.asarray([
            hand["T_g_h"][:3, 3]
            for row in timestamp_rows
            for hand in fused_hands.get(row["pair_index"], [])
            if hand["track_id"] == track_id
        ])
        steps = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        track_summary[str(track_id)] = {
            "frames": len(positions),
            "handedness": next(
                hand["handedness"] for hands in fused_hands.values() for hand in hands
                if hand["track_id"] == track_id
            ),
            "world_displacement_m": float(np.linalg.norm(positions[-1] - positions[0])),
            "world_path_length_m": float(np.sum(steps)),
            "world_step_p95_m": float(np.percentile(steps, 95)) if len(steps) else 0.0,
        }
    summary = {
        "stage": "basalt_vio_world_hand_trajectory",
        "session": str(session),
        "basalt_dataset": str(basalt_dataset),
        "basalt_trajectory": str(trajectory_path),
        "hand_pose_csv": str(hand_pose_path),
        "frames": len(timestamp_rows),
        "world_frame": {
            "origin": "first left-camera optical centre",
            "orientation": "Basalt gravity-aligned world axes; +Z is opposite measured gravity",
            "metric_scale": "stereo baseline",
        },
        "camera_path_length_m": float(np.sum(camera_steps)),
        "camera_displacement_m": float(np.linalg.norm(camera_positions[-1] - camera_positions[0])),
        "camera_step_p95_m": float(np.percentile(camera_steps, 95)),
        "tracks": track_summary,
        "trajectory_length_frames": args.trajectory_length,
        "elapsed_seconds": time.perf_counter() - started,
        "outputs": {
            "camera_csv": "camera_trajectory_world.csv",
            "hand_csv": "hand_trajectory_world.csv",
            "video": None if args.no_video else video_path.name,
            "preview": None if args.no_video else preview_path.name,
        },
    }
    with (output / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
        stream.write("\n")
    print(f"Camera world trajectory: {output / 'camera_trajectory_world.csv'}")
    print(f"Hand world trajectory: {output / 'hand_trajectory_world.csv'}")
    if not args.no_video:
        print(f"World trajectory overlay: {video_path}")
    print(f"Camera path/displacement: {summary['camera_path_length_m']:.3f} / {summary['camera_displacement_m']:.3f} m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
