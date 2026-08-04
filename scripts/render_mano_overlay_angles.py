#!/usr/bin/env python3
"""Overlay fitted MANO meshes and geometric finger angles on EGO video."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from mediapipe_left_baseline import (  # noqa: E402
    HAND_CONNECTIONS,
    camera_matrices,
    create_stereo_rectification,
    unique_file,
)
from fit_mano_sequence import MEDIAPIPE_NAMES  # noqa: E402


FINGER_CHAINS = {
    "thumb": ((0, 1, 2, "cmc"), (1, 2, 3, "mcp"), (2, 3, 4, "ip")),
    "index": ((0, 5, 6, "mcp"), (5, 6, 7, "pip"), (6, 7, 8, "dip")),
    "middle": ((0, 9, 10, "mcp"), (9, 10, 11, "pip"), (10, 11, 12, "dip")),
    "ring": ((0, 13, 14, "mcp"), (13, 14, 15, "pip"), (14, 15, 16, "dip")),
    "pinky": ((0, 17, 18, "mcp"), (17, 18, 19, "pip"), (18, 19, 20, "dip")),
}

ANGLE_KEYS = tuple(
    [f"{finger}_{joint}_bend_deg" for finger, chains in FINGER_CHAINS.items() for *_, joint in chains]
    + [f"{finger}_spread_deg" for finger in FINGER_CHAINS]
)

DISPLAY_JOINTS = {
    "thumb": ("cmc", "mcp", "ip"),
    "index": ("mcp", "pip", "dip"),
    "middle": ("mcp", "pip", "dip"),
    "ring": ("mcp", "pip", "dip"),
    "pinky": ("mcp", "pip", "dip"),
}

TRACK_COLORS = {
    "Right": (255, 115, 35),
    "Left": (35, 135, 255),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render fitted MANO meshes on the original EGO left camera with angle gauges."
    )
    parser.add_argument("--session", required=True, type=Path)
    parser.add_argument("--mano-fit", required=True, type=Path, help="directory containing track_*.npz")
    parser.add_argument("--stereo-frames", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--balance", type=float, default=0.0)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--panel-width", type=int, default=590)
    parser.add_argument("--mesh-alpha", type=float, default=0.38)
    parser.add_argument("--angle-radius", type=int, default=2)
    parser.add_argument("--start-pair", type=int, default=0)
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument("--no-video", action="store_true")
    return parser.parse_args()


def unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm < 1e-9:
        return np.full(3, np.nan, dtype=np.float64)
    return np.asarray(vector, dtype=np.float64) / norm


def bend_angle(parent: np.ndarray, joint: np.ndarray, child: np.ndarray) -> float:
    """Return geometric bend: 0 degrees straight, increasing while flexed."""
    incoming = unit(joint - parent)
    outgoing = unit(child - joint)
    if not np.isfinite(incoming).all() or not np.isfinite(outgoing).all():
        return float("nan")
    cosine = float(np.clip(np.dot(incoming, outgoing), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def signed_angle_on_plane(reference: np.ndarray, vector: np.ndarray, normal: np.ndarray) -> float:
    normal = unit(normal)
    if not np.isfinite(normal).all():
        return float("nan")
    reference = unit(reference - normal * np.dot(reference, normal))
    vector = unit(vector - normal * np.dot(vector, normal))
    if not np.isfinite(reference).all() or not np.isfinite(vector).all():
        return float("nan")
    sine = float(np.dot(np.cross(reference, vector), normal))
    cosine = float(np.clip(np.dot(reference, vector), -1.0, 1.0))
    return float(np.degrees(np.arctan2(sine, cosine)))


def spread_angle_on_plane(reference: np.ndarray, vector: np.ndarray, normal: np.ndarray) -> float:
    """Return finger spread in [-90, 90], treating projected bone direction as an axis."""
    angle = signed_angle_on_plane(reference, vector, normal)
    if not np.isfinite(angle):
        return angle
    if angle > 90.0:
        angle -= 180.0
    elif angle < -90.0:
        angle += 180.0
    return angle


def compute_joint_angles(joints: np.ndarray) -> dict[str, float]:
    joints = np.asarray(joints, dtype=np.float64)
    if joints.shape != (21, 3):
        raise ValueError(f"expected (21, 3) joints, got {joints.shape}")
    result: dict[str, float] = {}
    for finger, chains in FINGER_CHAINS.items():
        for parent, joint, child, joint_name in chains:
            result[f"{finger}_{joint_name}_bend_deg"] = bend_angle(
                joints[parent], joints[joint], joints[child]
            )

    palm_normal = np.cross(joints[5] - joints[0], joints[17] - joints[0])
    reference = joints[10] - joints[9]
    spread_vectors = {
        "thumb": joints[2] - joints[1],
        "index": joints[6] - joints[5],
        "middle": joints[10] - joints[9],
        "ring": joints[14] - joints[13],
        "pinky": joints[18] - joints[17],
    }
    for finger, vector in spread_vectors.items():
        result[f"{finger}_spread_deg"] = spread_angle_on_plane(reference, vector, palm_normal)
    return result


def median_filter(values: np.ndarray, radius: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or radius < 0:
        raise ValueError("values must be 2D and radius non-negative")
    output = np.full_like(values, np.nan)
    for frame in range(len(values)):
        window = values[max(0, frame - radius):min(len(values), frame + radius + 1)]
        for column in range(values.shape[1]):
            finite = window[:, column][np.isfinite(window[:, column])]
            if len(finite):
                output[frame, column] = np.median(finite)
    return output


def load_tracks(directory: Path, angle_radius: int) -> list[dict]:
    tracks = []
    for path in sorted(directory.glob("track_*.npz")):
        with np.load(path) as archive:
            required = {"pair_indices", "track_id", "handedness", "faces", "vertices", "joints"}
            missing = required - set(archive.files)
            if missing:
                raise RuntimeError(f"{path} missing arrays: {sorted(missing)}")
            track = {name: archive[name].copy() for name in required}
        pairs = track["pair_indices"].astype(np.int64)
        if len(np.unique(pairs)) != len(pairs):
            raise RuntimeError(f"duplicate pair index in {path}")
        raw = np.asarray([
            [angles[key] for key in ANGLE_KEYS]
            for angles in (compute_joint_angles(joints) for joints in track["joints"])
        ])
        track["angles_raw"] = raw
        track["angles"] = median_filter(raw, angle_radius)
        track["lookup"] = {int(pair): index for index, pair in enumerate(pairs)}
        track["track_id"] = int(track["track_id"])
        track["handedness"] = str(track["handedness"])
        tracks.append(track)
    if not tracks:
        raise FileNotFoundError(f"no track_*.npz files in {directory}")
    return tracks


def load_frame_rows(path: Path, start_pair: int, max_pairs: int) -> list[dict[str, int]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = [
            {"pair_index": int(row["pair_index"]), "left_index": int(row["left_index"])}
            for row in csv.DictReader(stream)
        ]
    rows = [row for row in rows if row["pair_index"] >= start_pair]
    if max_pairs > 0:
        rows = rows[:max_pairs]
    if not rows:
        raise RuntimeError("no stereo frame rows selected")
    if any(b["left_index"] <= a["left_index"] for a, b in zip(rows, rows[1:])):
        raise RuntimeError("left frame indices must be strictly increasing")
    return rows


def read_frame_at(capture: cv2.VideoCapture, target: int, state: list[int]) -> np.ndarray:
    frame = None
    while state[0] <= target:
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError(f"video ended before left frame {target}")
        state[0] += 1
    if frame is None:
        raise RuntimeError(f"failed to decode left frame {target}")
    return frame


def project_fisheye(points: np.ndarray, camera_matrix: np.ndarray, distortion: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    projected, _ = cv2.fisheye.projectPoints(
        points.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), camera_matrix, distortion
    )
    return projected[:, 0]


def raw_rectified_projection_residual(
    points: np.ndarray,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    rotation: np.ndarray,
    projection: np.ndarray,
) -> np.ndarray:
    raw = project_fisheye(points, camera_matrix, distortion)
    via_undistort = cv2.fisheye.undistortPoints(
        raw.reshape(-1, 1, 2), camera_matrix, distortion, R=rotation, P=projection
    )[:, 0]
    rectified = (rotation @ np.asarray(points, dtype=np.float64).T).T
    homogeneous = np.column_stack((rectified, np.ones(len(rectified)))) @ projection.T
    direct = homogeneous[:, :2] / homogeneous[:, 2:]
    return np.linalg.norm(via_undistort - direct, axis=1)


def shade_color(color: tuple[int, int, int], scale: float) -> tuple[int, int, int]:
    return tuple(int(np.clip(channel * scale, 0, 255)) for channel in color)


def draw_mesh(
    image: np.ndarray,
    vertices: np.ndarray,
    joints: np.ndarray,
    faces: np.ndarray,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    color: tuple[int, int, int],
    alpha: float,
    label: str,
) -> None:
    height, width = image.shape[:2]
    vertex_px = project_fisheye(vertices, camera_matrix, distortion)
    joint_px = project_fisheye(joints, camera_matrix, distortion)
    face_vertices = vertices[faces]
    normals = np.cross(face_vertices[:, 1] - face_vertices[:, 0], face_vertices[:, 2] - face_vertices[:, 0])
    normal_norm = np.linalg.norm(normals, axis=1)
    lighting = np.divide(
        np.abs(normals[:, 2]), normal_norm, out=np.zeros_like(normal_norm), where=normal_norm > 1e-9
    )
    buckets = np.clip((lighting * 4.999).astype(np.int32), 0, 4)
    on_camera = np.all(face_vertices[:, :, 2] > 0.03, axis=1)
    polygons = np.rint(vertex_px[faces]).astype(np.int32)
    intersects = (
        (polygons[:, :, 0].max(axis=1) >= 0)
        & (polygons[:, :, 0].min(axis=1) < width)
        & (polygons[:, :, 1].max(axis=1) >= 0)
        & (polygons[:, :, 1].min(axis=1) < height)
    )
    keep = on_camera & intersects
    overlay = image.copy()
    visible_polygons = []
    for bucket in range(5):
        selected = polygons[keep & (buckets == bucket)]
        if len(selected):
            cv2.fillPoly(overlay, list(selected), shade_color(color, 0.66 + bucket * 0.105), cv2.LINE_AA)
            visible_polygons.extend(selected[::2])
    cv2.addWeighted(overlay, alpha, image, 1.0 - alpha, 0.0, image)
    if visible_polygons:
        cv2.polylines(image, visible_polygons, True, shade_color(color, 0.60), 1, cv2.LINE_AA)
    for start, end in HAND_CONNECTIONS:
        a = tuple(np.rint(joint_px[start]).astype(int))
        b = tuple(np.rint(joint_px[end]).astype(int))
        cv2.line(image, a, b, shade_color(color, 1.15), 2, cv2.LINE_AA)
    for point in joint_px:
        cv2.circle(image, tuple(np.rint(point).astype(int)), 3, (245, 245, 245), -1, cv2.LINE_AA)
    wrist = np.rint(joint_px[0]).astype(int)
    cv2.putText(
        image, label, (int(wrist[0]) + 12, int(wrist[1]) - 12),
        cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2, cv2.LINE_AA,
    )


def draw_bar(
    panel: np.ndarray,
    origin: tuple[int, int],
    width: int,
    value: float,
    color: tuple[int, int, int],
    maximum: float = 120.0,
) -> None:
    x, y = origin
    cv2.rectangle(panel, (x, y), (x + width, y + 8), (57, 65, 78), -1, cv2.LINE_AA)
    if np.isfinite(value):
        filled = int(round(width * np.clip(value / maximum, 0.0, 1.0)))
        cv2.rectangle(panel, (x, y), (x + filled, y + 8), color, -1, cv2.LINE_AA)


def draw_hand_card(
    panel: np.ndarray,
    rect: tuple[int, int, int, int],
    track: dict,
    frame_index: int | None,
) -> None:
    x, y, width, height = rect
    handedness = track["handedness"]
    color = TRACK_COLORS.get(handedness, (120, 220, 120))
    cv2.rectangle(panel, (x, y), (x + width, y + height), (29, 35, 45), -1, cv2.LINE_AA)
    cv2.rectangle(panel, (x, y), (x + width, y + height), (70, 80, 95), 1, cv2.LINE_AA)
    title = f"{handedness.upper()} HAND   TRACK {track['track_id']}"
    cv2.putText(panel, title, (x + 18, y + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.70, color, 2, cv2.LINE_AA)
    cv2.putText(
        panel, "geometric bend (deg):  MCP / PIP / DIP", (x + 18, y + 53),
        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (170, 178, 190), 1, cv2.LINE_AA,
    )
    if frame_index is None:
        cv2.putText(
            panel, "NOT VISIBLE", (x + 155, y + height // 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.90, (105, 112, 123), 2, cv2.LINE_AA,
        )
        return

    angles = {key: track["angles"][frame_index, index] for index, key in enumerate(ANGLE_KEYS)}
    row_height = 68
    row_top = y + 66
    label_width = 75
    column_gap = 8
    usable = width - 36 - label_width
    column_width = (usable - 2 * column_gap) // 3
    for row, (finger, joint_names) in enumerate(DISPLAY_JOINTS.items()):
        top = row_top + row * row_height
        cv2.putText(
            panel, finger.upper(), (x + 18, top + 27),
            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (224, 227, 232), 1, cv2.LINE_AA,
        )
        for column, joint_name in enumerate(joint_names):
            key = f"{finger}_{joint_name}_bend_deg"
            value = angles[key]
            bar_x = x + 18 + label_width + column * (column_width + column_gap)
            value_text = f"{value:5.1f}" if np.isfinite(value) else "  nan"
            cv2.putText(
                panel, value_text, (bar_x, top + 17),
                cv2.FONT_HERSHEY_SIMPLEX, 0.43, (220, 224, 230), 1, cv2.LINE_AA,
            )
            draw_bar(panel, (bar_x, top + 27), column_width, value, color)
    spreads = [angles[f"{finger}_spread_deg"] for finger in FINGER_CHAINS]
    spread_text = "spread T/I/M/R/P: " + " / ".join(f"{value:+.0f}" for value in spreads)
    cv2.putText(
        panel, spread_text, (x + 18, y + height - 17),
        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (164, 173, 187), 1, cv2.LINE_AA,
    )


def compose_canvas(
    frame: np.ndarray,
    tracks: list[dict],
    track_frame_indices: dict[int, int],
    output_size: tuple[int, int],
    panel_width: int,
    pair_index: int,
    left_index: int,
) -> np.ndarray:
    width, height = output_size
    view_width = width - panel_width
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    scale = min(view_width / frame.shape[1], height / frame.shape[0])
    resized_size = (int(round(frame.shape[1] * scale)), int(round(frame.shape[0] * scale)))
    resized = cv2.resize(frame, resized_size, interpolation=cv2.INTER_AREA)
    offset_x = (view_width - resized_size[0]) // 2
    offset_y = (height - resized_size[1]) // 2
    canvas[offset_y:offset_y + resized_size[1], offset_x:offset_x + resized_size[0]] = resized
    cv2.rectangle(canvas, (0, 0), (view_width - 1, height - 1), (75, 82, 92), 1)

    title_layer = canvas.copy()
    cv2.rectangle(title_layer, (18, 16), (580, 72), (9, 13, 18), -1, cv2.LINE_AA)
    cv2.addWeighted(title_layer, 0.72, canvas, 0.28, 0.0, canvas)
    cv2.putText(
        canvas, "EGO MANO LIVE OVERLAY", (35, 47),
        cv2.FONT_HERSHEY_SIMPLEX, 0.80, (242, 244, 247), 2, cv2.LINE_AA,
    )
    cv2.putText(
        canvas, f"pair {pair_index:03d}   left frame {left_index:03d}", (35, 66),
        cv2.FONT_HERSHEY_SIMPLEX, 0.43, (174, 182, 194), 1, cv2.LINE_AA,
    )

    panel = canvas[:, view_width:]
    panel[:] = (16, 20, 27)
    cv2.putText(
        panel, "HAND JOINT ANGLES", (20, 38),
        cv2.FONT_HERSHEY_SIMPLEX, 0.78, (238, 241, 246), 2, cv2.LINE_AA,
    )
    cv2.putText(
        panel, "MANO: ON   |   metric 3D + fisheye projection", (20, 61),
        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (147, 158, 174), 1, cv2.LINE_AA,
    )
    card_gap = 12
    margin = 13
    top = 78
    card_height = (height - top - margin - card_gap) // 2
    for card_index, track in enumerate(tracks[:2]):
        frame_index = track_frame_indices.get(track["track_id"])
        draw_hand_card(
            panel,
            (margin, top + card_index * (card_height + card_gap), panel_width - 2 * margin, card_height),
            track,
            frame_index,
        )
    return canvas


def write_montage(frames: list[np.ndarray], path: Path) -> None:
    if not frames:
        return
    thumbs = [cv2.resize(frame, (640, 360), interpolation=cv2.INTER_AREA) for frame in frames]
    while len(thumbs) < 6:
        thumbs.append(thumbs[-1].copy())
    montage = cv2.vconcat([cv2.hconcat(thumbs[:3]), cv2.hconcat(thumbs[3:6])])
    if not cv2.imwrite(str(path), montage):
        raise RuntimeError(f"cannot write {path}")


def main() -> int:
    args = parse_args()
    if args.width <= 0 or args.height <= 0 or not 250 <= args.panel_width < args.width:
        raise ValueError("invalid output dimensions/panel width")
    if not 0.0 < args.mesh_alpha <= 1.0 or args.angle_radius < 0:
        raise ValueError("mesh alpha must be in (0,1] and angle radius non-negative")
    session = args.session.resolve()
    fit_dir = args.mano_fit.resolve()
    frames_path = args.stereo_frames.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    calibration_path = unique_file(session, "_calibration_camera.yaml")
    left_video_path = unique_file(session, "_camera_left.mp4")
    with calibration_path.open("r", encoding="utf-8") as stream:
        calibration = yaml.safe_load(stream)
    camera = calibration["cameras"][0]
    camera_matrix, distortion = camera_matrices(camera)
    expected_size = (int(camera["image_width"]), int(camera["image_height"]))
    rectification = create_stereo_rectification(calibration_path, args.balance)
    tracks = load_tracks(fit_dir, args.angle_radius)
    frame_rows = load_frame_rows(frames_path, args.start_pair, args.max_pairs)

    validation_points = tracks[0]["vertices"][0].astype(np.float64)
    projection_residual = raw_rectified_projection_residual(
        validation_points, camera_matrix, distortion, rectification["r1"], rectification["p1"]
    )
    if float(np.max(projection_residual)) > 1e-6:
        raise RuntimeError("raw fisheye projection disagrees with rectified projection")

    capture = cv2.VideoCapture(str(left_video_path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open {left_video_path}")
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    fps = source_fps if source_fps > 0 else 30.0
    video_path = output / "mano_overlay_angles.mp4"
    writer = None
    if not args.no_video:
        writer = cv2.VideoWriter(
            str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (args.width, args.height)
        )
        if not writer.isOpened():
            raise RuntimeError(f"cannot create {video_path}")

    angle_csv_path = output / "mano_joint_angles.csv"
    angle_fields = ["pair_index", "left_frame_index", "track_id", "handedness"]
    angle_fields += [f"{key}_raw" for key in ANGLE_KEYS] + list(ANGLE_KEYS)
    preview_ordinals = set(np.linspace(0, len(frame_rows) - 1, min(6, len(frame_rows)), dtype=int).tolist())
    previews: list[np.ndarray] = []
    frame_state = [0]
    processed = 0
    visible_instances = 0
    start_time = time.perf_counter()

    with angle_csv_path.open("w", encoding="utf-8", newline="") as angle_stream:
        angle_writer = csv.DictWriter(angle_stream, fieldnames=angle_fields)
        angle_writer.writeheader()
        for ordinal, row in enumerate(frame_rows):
            pair_index = row["pair_index"]
            left_index = row["left_index"]
            frame = read_frame_at(capture, left_index, frame_state)
            if frame.shape[1::-1] != expected_size:
                raise RuntimeError(f"decoded size {frame.shape[1::-1]} differs from calibration {expected_size}")

            visible = []
            frame_indices: dict[int, int] = {}
            for track in tracks:
                track_frame = track["lookup"].get(pair_index)
                if track_frame is None:
                    continue
                frame_indices[track["track_id"]] = track_frame
                visible.append((float(np.median(track["vertices"][track_frame, :, 2])), track, track_frame))
                angle_row = {
                    "pair_index": pair_index,
                    "left_frame_index": left_index,
                    "track_id": track["track_id"],
                    "handedness": track["handedness"],
                }
                for index, key in enumerate(ANGLE_KEYS):
                    angle_row[f"{key}_raw"] = f"{track['angles_raw'][track_frame, index]:.6f}"
                    angle_row[key] = f"{track['angles'][track_frame, index]:.6f}"
                angle_writer.writerow(angle_row)

            for _, track, track_frame in sorted(visible, reverse=True, key=lambda item: item[0]):
                color = TRACK_COLORS.get(track["handedness"], (120, 220, 120))
                draw_mesh(
                    frame,
                    track["vertices"][track_frame],
                    track["joints"][track_frame],
                    track["faces"].astype(np.int32),
                    camera_matrix,
                    distortion,
                    color,
                    args.mesh_alpha,
                    f"{track['handedness']} T{track['track_id']}",
                )
            canvas = compose_canvas(
                frame, tracks, frame_indices, (args.width, args.height), args.panel_width,
                pair_index, left_index,
            )
            if writer is not None:
                writer.write(canvas)
            if ordinal in preview_ordinals:
                previews.append(canvas.copy())
            visible_instances += len(visible)
            processed += 1

    capture.release()
    if writer is not None:
        writer.release()
    elapsed = time.perf_counter() - start_time
    preview_path = output / "preview_montage.jpg"
    write_montage(previews, preview_path)

    angle_summary = {}
    for track in tracks:
        values = track["angles"]
        bend_values = values[:, :15]
        spread_values = values[:, 15:]
        angle_summary[str(track["track_id"])] = {
            "handedness": track["handedness"],
            "frames": int(len(track["pair_indices"])),
            "bend_observations": int(np.count_nonzero(np.isfinite(bend_values))),
            "bend_over_130_deg": int(np.count_nonzero(bend_values > 130.0)),
            "bend_over_130_rate": float(np.nanmean(bend_values > 130.0)),
            "bend_deg_median": float(np.nanmedian(bend_values)),
            "bend_deg_p95": float(np.nanpercentile(bend_values, 95)),
            "spread_abs_deg_median": float(np.nanmedian(np.abs(spread_values))),
            "spread_deg_min": float(np.nanmin(spread_values)),
            "spread_deg_max": float(np.nanmax(spread_values)),
        }
    summary = {
        "stage": "mano_camera_overlay_and_geometric_joint_angles",
        "session": str(session),
        "mano_fit": str(fit_dir),
        "source_video": str(left_video_path),
        "source_fps": fps,
        "source_size": list(expected_size),
        "output_size": [args.width, args.height],
        "processed_pairs": processed,
        "visible_hand_instances": visible_instances,
        "track_count": len(tracks),
        "projection_validation_max_px": float(np.max(projection_residual)),
        "projection_validation_median_px": float(np.median(projection_residual)),
        "mesh_alpha": args.mesh_alpha,
        "angle_smoothing_radius_frames": args.angle_radius,
        "processing_fps": processed / elapsed if elapsed > 0 else 0.0,
        "angle_definition": (
            "Geometric bend is the angle between consecutive 3D bone directions: 0 degrees is straight. "
            "Spread is a signed proximal-bone angle in the fitted palm plane. These are not raw MANO axis-angle components."
        ),
        "tracks": angle_summary,
        "outputs": {
            "video": video_path.name if writer is not None else None,
            "joint_angles_csv": angle_csv_path.name,
            "preview_montage": preview_path.name,
        },
    }
    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("EGO MANO camera overlay and joint-angle dashboard")
    print(f"Processed pairs: {processed}")
    print(f"Visible hand instances: {visible_instances}")
    print(f"Raw/rectified projection agreement max: {summary['projection_validation_max_px']:.3e} px")
    print(f"Rendering speed: {summary['processing_fps']:.2f} fps")
    if writer is not None:
        print(f"Overlay video: {video_path}")
    print(f"Joint angles CSV: {angle_csv_path}")
    print(f"Preview montage: {preview_path}")
    print(f"Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
