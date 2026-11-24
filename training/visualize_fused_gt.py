"""Visualize fused multiview GT on camera RGB and in wrist-relative 3D."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
import json
import logging
from pathlib import Path

import av
import cv2
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import numpy as np

from visualize_hand_3d import (
    COLORS, CONNECTIONS, H264VideoWriter, HandScene, display_points, letterbox, put_text,
)
from wilor.utils.visualization import LEFT_COLOR, RIGHT_COLOR


LOG = logging.getLogger("fused_gt")


def resolve_run(path: Path) -> Path:
    path = path.expanduser().resolve()
    if (path / "fusion_multiview/accepted.jsonl").is_file():
        return path
    candidates = sorted(
        item.parent.parent
        for item in path.glob("*/fusion_multiview/accepted.jsonl")
        if item.is_file()
    )
    if len(candidates) != 1:
        raise ValueError(f"Expected exactly one fused run under {path}, found {len(candidates)}")
    return candidates[0]


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def base_to_camera(points_base: np.ndarray, calibration: dict) -> np.ndarray:
    points = np.asarray(points_base, dtype=np.float64)
    if points.shape != (21, 3):
        raise ValueError(f"Expected fused joints with shape (21, 3), got {points.shape}")
    transform = np.asarray(calibration["T_base_camera"], dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError("T_base_camera must have shape (4, 4)")
    rotation = transform[:3, :3]
    center = transform[:3, 3]
    return (rotation.T @ (points - center).T).T


def project_double_sphere(points_camera: np.ndarray, calibration: dict) -> tuple[np.ndarray, np.ndarray]:
    """Project camera-frame points using GEN's published DS parameter order."""
    points = np.asarray(points_camera, dtype=np.float64)
    flat = points.reshape(-1, 3)
    x, y, z = flat.T
    fx, fy, cx, cy, xi, alpha = np.asarray(calibration["distortion"], dtype=np.float64)
    d1 = np.linalg.norm(flat, axis=1)
    z1 = alpha * d1 + (1.0 - alpha) * z
    d2 = np.sqrt(x * x + y * y + z1 * z1)
    denominator = xi * d2 + z1
    valid = np.all(np.isfinite(flat), axis=1) & (d1 > 1e-12) & (denominator > 1e-12)
    pixels = np.full((len(flat), 2), np.nan, dtype=np.float64)
    pixels[valid, 0] = fx * x[valid] / denominator[valid] + cx
    pixels[valid, 1] = fy * y[valid] / denominator[valid] + cy
    return pixels.reshape(points.shape[:-1] + (2,)), valid.reshape(points.shape[:-1])


def root_relative_hands(result: dict | None, calibration: dict) -> list[tuple[int, np.ndarray]]:
    hands = []
    if result is None:
        return hands
    for hand in result["hands"]:
        side = int(hand["side"])
        points = base_to_camera(hand["joints_base_m"], calibration)
        finite = np.isfinite(points).all(axis=1)
        if side not in (0, 1) or not finite[0]:
            continue
        points = points - points[:1]
        points += np.array([0.14 if side else -0.14, 0.0, 0.0])
        hands.append((side, points))
    return hands


def partial_scene_bounds(frames_hands: list[list[tuple[int, np.ndarray]]]) -> tuple[np.ndarray, float]:
    clouds = []
    for hands in frames_hands:
        for _, points in hands:
            displayed = display_points(points)
            clouds.append(displayed[np.isfinite(displayed).all(axis=1)])
    clouds = [points for points in clouds if len(points)]
    if not clouds:
        return np.zeros(3), 0.25
    all_points = np.concatenate(clouds)
    low, high = all_points.min(axis=0), all_points.max(axis=0)
    return (low + high) / 2, max(float((high - low).max()) * 0.60, 0.22)


class PartialHandScene(HandScene):
    def render_partial(self, hands: list[tuple[int, np.ndarray]]) -> np.ndarray:
        for artist in self.artists:
            artist.remove()
        self.artists.clear()
        valid_count = 0
        for side, coordinates in hands:
            points = display_points(coordinates)
            finite = np.isfinite(points).all(axis=1)
            valid_count += int(finite.sum())
            edges = [points[[start, end]] for start, end in CONNECTIONS if finite[start] and finite[end]]
            if edges:
                lines = Line3DCollection(edges, colors=COLORS[side], linewidths=2.3)
                self.axes.add_collection3d(lines)
                self.artists.append(lines)
            if finite.any():
                dots = self.axes.scatter(*points[finite].T, color=COLORS[side], s=22,
                                         edgecolors="white", linewidths=0.4, depthshade=False)
                self.artists.append(dots)
        self.empty_text.set_text("Fusion rejected / no valid GT")
        self.empty_text.set_visible(valid_count == 0)
        self.canvas.draw()
        return cv2.cvtColor(np.asarray(self.canvas.buffer_rgba()), cv2.COLOR_RGBA2BGR)


def draw_gt_overlay(image: np.ndarray, result: dict | None, calibration: dict) -> np.ndarray:
    canvas = image.copy()
    if result is None:
        return canvas
    height, width = canvas.shape[:2]
    for hand in result["hands"]:
        side = int(hand["side"])
        color = RIGHT_COLOR if side else LEFT_COLOR
        points_camera = base_to_camera(hand["joints_base_m"], calibration)
        pixels, model_valid = project_double_sphere(points_camera, calibration)
        finite = model_valid & np.isfinite(pixels).all(axis=1)
        inside = finite & (pixels[:, 0] >= 0) & (pixels[:, 0] < width) & (pixels[:, 1] >= 0) & (pixels[:, 1] < height)
        rounded = np.zeros((21, 2), dtype=np.int32)
        rounded[finite] = np.rint(pixels[finite]).astype(np.int32)
        for start, end in CONNECTIONS:
            if finite[start] and finite[end]:
                cv2.line(canvas, tuple(rounded[start]), tuple(rounded[end]), color, 3, cv2.LINE_AA)
        for point in rounded[inside]:
            cv2.circle(canvas, tuple(point), 5, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(canvas, tuple(point), 3, color, -1, cv2.LINE_AA)
        view = hand.get("views", {}).get(calibration["camera_id"])
        box = None if view is None else np.asarray(view.get("bbox_xyxy"), dtype=np.float64)
        if box is None or box.shape != (4,) or not np.isfinite(box).all():
            if inside.any():
                low, high = pixels[inside].min(axis=0), pixels[inside].max(axis=0)
                box = np.r_[low - 18, high + 18]
        if box is not None and np.asarray(box).shape == (4,):
            box = np.rint(box).astype(np.int32)
            cv2.rectangle(canvas, tuple(box[:2]), tuple(box[2:]), color, 3, cv2.LINE_AA)
            cv2.putText(canvas, f"GT {'R' if side else 'L'} {int(finite.sum())}/21",
                        (int(box[0]), max(25, int(box[1]) - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.62, color, 2, cv2.LINE_AA)
    return canvas


def load_sync_rows(path: Path, camera: str) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["sync_index"] = int(row["sync_index"])
        row["camera_frame_index"] = int(row[f"{camera}_frame_index"])
        row["timestamp_ns"] = int(row[f"{camera}_timestamp_ns"])
    rows.sort(key=lambda row: row["sync_index"])
    if [row["sync_index"] for row in rows] != list(range(len(rows))):
        raise ValueError("Synchronized frame indices must be contiguous from zero")
    return rows


def estimate_fps(rows: list[dict]) -> float:
    deltas = np.diff(np.asarray([row["timestamp_ns"] for row in rows], dtype=np.int64))
    deltas = deltas[deltas > 0]
    return 1e9 / float(np.median(deltas)) if len(deltas) else 30.0


def process(args) -> dict:
    run = resolve_run(args.input)
    normalized = run / "normalized_multiview"
    fusion = run / "fusion_multiview"
    calibration = json.loads((normalized / "calibration" / f"{args.camera}.json").read_text())
    if str(calibration["model"]).upper() != "DS":
        raise ValueError("This visualizer currently expects a Double-Sphere camera")
    rows = load_sync_rows(normalized / "multiview_frames.csv", args.camera)
    if args.max_frames is not None:
        rows = rows[:args.max_frames]
    accepted_list = load_jsonl(fusion / "accepted.jsonl")
    rejected_list = load_jsonl(fusion / "rejected.jsonl")
    accepted = {int(item["sync_index"]): item for item in accepted_list}
    rejected = {int(item["sync_index"]): item for item in rejected_list}
    if set(accepted) & set(rejected):
        raise ValueError("A sync frame cannot be both accepted and rejected")
    missing = {row["sync_index"] for row in rows} - set(accepted) - set(rejected)
    if missing:
        raise ValueError(f"Fusion status missing for sync frames: {sorted(missing)[:10]}")

    frames_hands = [root_relative_hands(accepted.get(row["sync_index"]), calibration) for row in rows]
    output_dir = args.output_dir or fusion / "gt_visualization"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{args.camera}_rgb_fused_gt_3d.mp4"
    temporary = output.with_name(f"{output.stem}.partial.mp4")
    fps = args.fps or estimate_fps(rows)
    writer = H264VideoWriter(temporary, fps, args.width, args.height)
    half, header, footer = args.width // 2, 58, 78
    panel_height = args.height - header - footer
    scene = PartialHandScene(args.width - half, panel_height)
    scene.set_bounds(*partial_scene_bounds(frames_hands))
    row_by_camera_frame = {row["camera_frame_index"]: row for row in rows}
    wanted = set(row_by_camera_frame)
    written = 0
    success = False
    video_path = normalized / "cameras" / args.camera / "video.mkv"
    base_timestamp = rows[0]["timestamp_ns"]
    try:
        with av.open(str(video_path)) as container:
            stream = container.streams.video[0]
            for camera_frame_index, av_frame in enumerate(container.decode(stream)):
                if camera_frame_index not in wanted:
                    continue
                row = row_by_camera_frame[camera_frame_index]
                sync_index = row["sync_index"]
                result = accepted.get(sync_index)
                image = av_frame.to_ndarray(format="bgr24")
                overlay = draw_gt_overlay(image, result, calibration)
                canvas = np.full((args.height, args.width, 3), 24, dtype=np.uint8)
                canvas[header:header + panel_height, :half] = letterbox(overlay, half, panel_height)
                canvas[header:header + panel_height, half:] = scene.render_partial(
                    root_relative_hands(result, calibration)
                )
                put_text(canvas, f"{args.camera} | source boxes + fused GT 2D projection",
                         (16, 35), half - 30, 0.58)
                put_text(canvas, "Fused multiview GT | wrist-relative camera axes | fixed L/R anchors",
                         (half + 16, 35), half - 30, 0.54)
                if result is None:
                    status = f"REJECTED | {rejected[sync_index].get('reason', 'unknown')}"
                else:
                    valid = [int(np.isfinite(np.asarray(hand["joints_base_m"])).all(axis=1).sum())
                             for hand in result["hands"]]
                    status = (f"ACCEPTED | fusion={result.get('hands', [{}])[0].get('fusion_mode', '-')}"
                              f" | GT joints L/R: {valid[0]}/{valid[1]}")
                put_text(canvas, status, (16, args.height - 48), args.width - 32, 0.58)
                put_text(canvas,
                         f"sync {sync_index} | camera frame {camera_frame_index}"
                         f" | {(row['timestamp_ns'] - base_timestamp) / 1e9:.3f}s",
                         (16, args.height - 18), args.width - 32, 0.55)
                writer.write(canvas)
                written += 1
                if written % 100 == 0:
                    LOG.info("Rendered %d/%d frames", written, len(rows))
        if written != len(rows):
            raise RuntimeError(f"Rendered {written} frames, expected {len(rows)}")
        success = True
    finally:
        try:
            writer.release()
        except BaseException:
            success = False
            raise
        finally:
            scene.close()
            if not success:
                temporary.unlink(missing_ok=True)
    temporary.replace(output)
    selected_accepted = [accepted[row["sync_index"]] for row in rows if row["sync_index"] in accepted]
    valid_counts = [
        int(np.isfinite(np.asarray(hand["joints_base_m"])).all(axis=1).sum())
        for result in selected_accepted for hand in result["hands"]
    ]
    report = {
        "source_run": str(run), "camera": args.camera,
        "coordinate_source": "fusion_multiview joints_base_m",
        "three_d_display": "camera-axis wrist-relative with fixed left/right anchors",
        "two_d_projection": "GEN Double-Sphere calibration",
        "frames": written, "accepted_frames": len(selected_accepted),
        "rejected_frames": written - len(selected_accepted),
        "hands": sum(len(result["hands"]) for result in selected_accepted),
        "complete_21_joint_hands": sum(count == 21 for count in valid_counts),
        "valid_joint_count_distribution": dict(sorted(Counter(valid_counts).items())),
        "fps": fps, "video": str(output),
    }
    (output_dir / "gt_visualization_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    LOG.info("Saved %s", output)
    return report


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True,
                        help="Fused run directory, or a batch directory containing one run.")
    parser.add_argument("--camera", default="camera2")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--fps", type=float)
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--max-frames", type=int)
    args = parser.parse_args(argv)
    args.input = args.input.expanduser().resolve()
    if not args.input.is_dir():
        parser.error(f"--input is not a directory: {args.input}")
    if args.output_dir is not None:
        args.output_dir = args.output_dir.expanduser().resolve()
    if not args.camera.startswith("camera") or not args.camera[6:].isdigit():
        parser.error("--camera must look like camera2")
    if args.fps is not None and (not np.isfinite(args.fps) or args.fps <= 0):
        parser.error("--fps must be finite and positive")
    if args.width < 960 or args.height < 600 or args.width % 2 or args.height % 2:
        parser.error("video dimensions must be even and at least 960x600")
    if args.max_frames is not None and args.max_frames < 1:
        parser.error("--max-frames must be positive")
    return args


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    process(parse_args())


if __name__ == "__main__":
    main()
