"""Write RGB / camera-space hand annotation previews without exporting images."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass, field
from fractions import Fraction
import json
import logging
from pathlib import Path
import pickle
import re

import av
import cv2
import matplotlib

matplotlib.use("Agg")
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_DATASETS = (
    ROOT / "datasets/0826_400_has_out_scene_2D/dataset",
    ROOT / "datasets/0827_sl",
    ROOT / "datasets/0905_jhn/dataset",
)
# Wrist, then four joints per finger, in the stored WiLoR joint order.
CONNECTIONS = tuple(
    edge
    for base in (1, 5, 9, 13, 17)
    for edge in ((0, base), (base, base + 1), (base + 1, base + 2), (base + 2, base + 3))
)
COLORS = {0: "#35a5cf", 1: "#ed9b24"}
BGR_COLORS = {0: (207, 165, 53), 1: (36, 155, 237)}
LOG = logging.getLogger("hand_3d")


class H264VideoWriter:
    """Encode browser-compatible MP4 directly, without intermediate frame images."""

    def __init__(self, path: Path, fps: float, width: int, height: int):
        # Fail immediately if the installed PyAV build lacks the encoder.
        av.Codec("libx264", "w")
        self.container = av.open(str(path), mode="w", format="mp4",
                                 options={"movflags": "+faststart"})
        try:
            rate = Fraction(str(fps)).limit_denominator(1_000_000)
            self.stream = self.container.add_stream("libx264", rate=rate)
            self.stream.width = width
            self.stream.height = height
            self.stream.pix_fmt = "yuv420p"
            self.stream.options = {"crf": "18", "preset": "fast"}
            self.stream.codec_context.thread_count = 4
            self.time_base = 1 / rate
            self.frame_count = 0
        except BaseException:
            self.container.close()
            path.unlink(missing_ok=True)
            raise

    def write(self, image: np.ndarray) -> None:
        if image.dtype != np.uint8 or image.shape != (self.stream.height, self.stream.width, 3):
            raise ValueError("Expected a uint8 BGR frame matching the video dimensions")
        frame = av.VideoFrame.from_ndarray(image, format="bgr24")
        frame.pts = self.frame_count
        frame.time_base = self.time_base
        for packet in self.stream.encode(frame):
            self.container.mux(packet)
        self.frame_count += 1

    def release(self) -> None:
        try:
            for packet in self.stream.encode(None):
                self.container.mux(packet)
        finally:
            self.container.close()


@dataclass
class Frame:
    source: str
    camera: str
    index: int
    rows: list[dict] = field(default_factory=list)


def source_key(row: dict) -> str:
    for key in ("source_group", "source_dataset_id", "source_dataset", "source_experiment"):
        value = row.get(key)
        if value is not None and str(value):
            return f"{key}:{value}"
    return "recording"


def natural_key(value: str) -> tuple:
    return tuple((1, int(part)) if part.isdigit() else (0, part)
                 for part in re.split(r"(\d+)", value))


def read_frames(root: Path) -> list[Frame]:
    groups = {}
    with (root / "index.jsonl").open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                source, camera = source_key(row), str(row.get("camera", "unknown"))
                sync = int(row.get("sync_index", -1))
                index = int(row.get("source_frame_index", sync))
                identity = ("sync", sync) if sync >= 0 else (
                    ("frame", index) if index >= 0 else ("image", str(row["image"]))
                )
                key = source, camera, identity
                if key not in groups:
                    groups[key] = Frame(source, camera, index)
                groups[key].rows.append(row)
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"{root / 'index.jsonl'}:{line_number}: {exc}") from exc
    return sorted(groups.values(), key=lambda f: (
        natural_key(f.source), natural_key(f.camera), f.index,
        natural_key(str(f.rows[0]["image"])),
    ))


def resolve(root: Path, path: str) -> Path:
    path = Path(path).expanduser()
    return path if path.is_absolute() else root / path


def camera_joints(label: dict) -> np.ndarray:
    joints = np.asarray(label["joints_3d"], dtype=np.float64)
    translation = np.asarray(label["trans"], dtype=np.float64)
    if joints.shape != (21, 3) or translation.shape != (3,):
        raise ValueError(f"expected joints (21, 3), trans (3,), got {joints.shape}, {translation.shape}")
    points = joints + translation
    if not np.isfinite(points).all():
        raise ValueError("non-finite 3D coordinates")
    return points


def project_camera_joints(points: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    """Project camera-space joints with the intrinsics stored in a WiLoR label."""
    points = np.asarray(points, dtype=np.float64)
    intrinsics = np.asarray(intrinsics, dtype=np.float64)
    if points.shape != (21, 3) or intrinsics.shape != (3, 3):
        raise ValueError("expected camera points (21, 3) and intrinsics (3, 3)")
    homogeneous = (intrinsics @ points.T).T
    if not np.isfinite(homogeneous).all() or np.any(homogeneous[:, 2] <= 1e-8):
        raise ValueError("cannot project camera-space joints")
    return homogeneous[:, :2] / homogeneous[:, 2:3]


def draw_label_overlay(image: np.ndarray, root: Path, frame: Frame) -> np.ndarray:
    """Draw each exported label's box and projected 21-joint skeleton."""
    canvas = image.copy()
    height, width = canvas.shape[:2]
    for row in frame.rows:
        try:
            label = np.load(resolve(root, row["label"]), allow_pickle=True).item()
            side = int(label.get("side", row.get("side", -1)))
            if side not in (0, 1):
                raise ValueError(f"invalid hand side {side}")
            pixels = project_camera_joints(camera_joints(label), label["K"])
            rounded = np.rint(pixels).astype(np.int32)
            color = BGR_COLORS[side]
            for start, end in CONNECTIONS:
                cv2.line(canvas, tuple(rounded[start]), tuple(rounded[end]),
                         color, 4, cv2.LINE_AA)
            inside = ((pixels[:, 0] >= 0) & (pixels[:, 0] < width)
                      & (pixels[:, 1] >= 0) & (pixels[:, 1] < height))
            for point in rounded[inside]:
                cv2.circle(canvas, tuple(point), 6, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(canvas, tuple(point), 4, color, -1, cv2.LINE_AA)

            box = np.asarray(label.get("bbox"), dtype=np.float64)
            if box.shape == (4,) and np.isfinite(box).all():
                box = np.rint(box).astype(np.int32)
                cv2.rectangle(canvas, tuple(box[:2]), tuple(box[2:]), color, 4, cv2.LINE_AA)
                name = "Right" if side else "Left"
                inliers = row.get("camera_inlier_joint_count")
                suffix = "" if inliers is None else f" | inliers {int(inliers)}/21"
                cv2.putText(canvas, f"GT {name}{suffix}",
                            (int(box[0]), max(30, int(box[1]) - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.72, color, 2, cv2.LINE_AA)
        except (OSError, KeyError, TypeError, ValueError, EOFError,
                pickle.UnpicklingError) as exc:
            LOG.warning("Cannot draw label %s: %s", row.get("label"), exc)
    return canvas


def load_hands(root: Path, frame: Frame, stats: dict) -> list[tuple[int, np.ndarray]]:
    hands = []
    seen = set()
    for row in frame.rows:
        try:
            # These are trusted, local dataset labels; some also contain torch tensors.
            label = np.load(resolve(root, row["label"]), allow_pickle=True).item()
            points = camera_joints(label)
            side = int(label.get("side", row.get("side", -1)))
            if side not in (0, 1):
                raise ValueError(f"invalid hand side {side}")
            if side in seen:
                raise ValueError(f"duplicate side {side} in the same frame")
            seen.add(side)
            hands.append((side, points))
        except (OSError, KeyError, TypeError, ValueError, EOFError, pickle.UnpicklingError) as exc:
            stats["skipped_hands"] += 1
            LOG.warning("Skipping label %s: %s", row.get("label"), exc)
    return hands


def display_points(points: np.ndarray) -> np.ndarray:
    """View camera coordinates with X right, Z depth, and -Y up."""
    return points[:, (0, 2, 1)] * np.array([1, 1, -1])


def root_relative_hands(hands: list[tuple[int, np.ndarray]]) -> list[tuple[int, np.ndarray]]:
    """Place wrist-relative poses at stable left/right anchors for comparison."""
    result = []
    for side, points in hands:
        anchor = np.array([0.14 if side else -0.14, 0.0, 0.0])
        result.append((side, points - points[:1] + anchor))
    return result


def scene_bounds(frames_hands: list) -> tuple[np.ndarray, float]:
    clouds = [display_points(points) for hands in frames_hands for _, points in hands]
    if not clouds:
        return np.zeros(3), 0.25
    points = np.concatenate(clouds)
    low, high = points.min(axis=0), points.max(axis=0)
    return (low + high) / 2, max(float((high - low).max()) * 0.60, 0.10)


class HandScene:
    def __init__(self, width: int, height: int):
        self.figure = Figure(figsize=(width / 100, height / 100), dpi=100,
                             facecolor="#f5f6f8")
        self.canvas = FigureCanvasAgg(self.figure)
        self.axes = self.figure.add_subplot(111, projection="3d")
        self.figure.subplots_adjust(left=0.06, right=0.91, bottom=0.13, top=0.92)
        self.axes.set_facecolor("#f5f6f8")
        self.axes.set_box_aspect((1, 1, 1))
        self.axes.view_init(elev=22, azim=-65)
        self.axes.set_xlabel("X right (m)", labelpad=9)
        self.axes.set_ylabel("Z forward (m)", labelpad=9)
        self.axes.set_zlabel("-Y up (m)", labelpad=9)
        self.axes.tick_params(labelsize=8)
        self.artists = []
        self.empty_text = self.axes.text2D(
            0.5, 0.5, "No valid 3D annotation", transform=self.axes.transAxes,
            ha="center", color="#666666", visible=False,
        )
        for side, name in ((0, "Left"), (1, "Right")):
            self.figure.text(0.30 + side * 0.28, 0.045, f"● {name}",
                             ha="center", color=COLORS[side], fontsize=12)

    def set_bounds(self, center: np.ndarray, radius: float) -> None:
        for setter, c in zip((self.axes.set_xlim, self.axes.set_ylim, self.axes.set_zlim), center):
            setter(c - radius, c + radius)

    def render(self, hands: list[tuple[int, np.ndarray]]) -> np.ndarray:
        for artist in self.artists:
            artist.remove()
        self.artists.clear()
        self.empty_text.set_visible(not hands)
        for side, camera_points in hands:
            points = display_points(camera_points)
            lines = Line3DCollection(points[np.asarray(CONNECTIONS)],
                                     colors=COLORS[side], linewidths=2.3)
            self.axes.add_collection3d(lines)
            dots = self.axes.scatter(*points.T, color=COLORS[side], s=22,
                                     edgecolors="white", linewidths=0.4, depthshade=False)
            self.artists.extend((lines, dots))
        self.canvas.draw()
        return cv2.cvtColor(np.asarray(self.canvas.buffer_rgba()), cv2.COLOR_RGBA2BGR)

    def close(self) -> None:
        self.figure.clear()


def letterbox(image: np.ndarray, width: int, height: int) -> np.ndarray:
    scale = min(width / image.shape[1], height / image.shape[0])
    w, h = max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale))
    resized = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    canvas = np.full((height, width, 3), 24, dtype=np.uint8)
    x, y = (width - w) // 2, (height - h) // 2
    canvas[y:y + h, x:x + w] = resized
    return canvas


def put_text(canvas: np.ndarray, text: str, xy: tuple, max_width: int,
             scale: float = 0.55) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    while len(text) > 4 and cv2.getTextSize(text, font, scale, 1)[0][0] > max_width:
        text = text[:-4] + "..."
    cv2.putText(canvas, text, xy, font, scale, (240, 240, 240), 1, cv2.LINE_AA)


def dataset_name(root: Path) -> str:
    return root.parent.name if root.name == "dataset" else root.name


def process_dataset(root: Path, args: argparse.Namespace) -> dict:
    root = root.expanduser().resolve()
    if not (root / "index.jsonl").is_file() and (root / "dataset/index.jsonl").is_file():
        root = root / "dataset"
    frames = read_frames(root)
    if args.max_frames is not None:
        frames = frames[:args.max_frames]
    if not frames:
        raise ValueError(f"No frames in {root / 'index.jsonl'}")
    segments = defaultdict(list)
    for frame in frames:
        segments[frame.source, frame.camera].append(frame)
    name = dataset_name(root)
    output = args.output_dir / f"{name}_rgb_3d.mp4"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f"{output.stem}.partial.mp4")
    stats = {"written_frames": 0, "skipped_frames": 0, "skipped_hands": 0,
             "empty_frames": 0}
    half = args.width // 2
    header, footer = 42, 64
    panel_height = args.height - header - footer
    writer = H264VideoWriter(temporary, args.fps, args.width, args.height)
    scene = None
    success = False
    try:
        scene = HandScene(args.width - half, panel_height)
        for segment_index, ((source, camera), segment) in enumerate(segments.items(), 1):
            LOG.info("%s: segment %d/%d, %s, %s, %d frames", name, segment_index,
                     len(segments), source, camera, len(segment))
            absolute_hands_by_frame = [load_hands(root, frame, stats) for frame in segment]
            hands_by_frame = [
                root_relative_hands(hands) if args.coordinate_mode == "root-relative" else hands
                for hands in absolute_hands_by_frame
            ]
            scene.set_bounds(*scene_bounds(hands_by_frame))
            for frame, hands in zip(segment, hands_by_frame):
                rgb = None
                # Per-hand manifest rows may reference separate copies of the same RGB.
                for image_path in dict.fromkeys(row["image"] for row in frame.rows):
                    rgb = cv2.imread(str(resolve(root, image_path)))
                    if rgb is not None:
                        break
                if rgb is None:
                    stats["skipped_frames"] += 1
                    LOG.warning("Skipping unreadable RGB: %s, %s, frame %s", source, camera, frame.index)
                    continue
                if args.overlay_2d:
                    rgb = draw_label_overlay(rgb, root, frame)
                canvas = np.full((args.height, args.width, 3), 24, dtype=np.uint8)
                canvas[header:header + panel_height, :half] = letterbox(rgb, half, panel_height)
                canvas[header:header + panel_height, half:] = scene.render(hands)
                left_title = "RGB + GT box and 2D projection" if args.overlay_2d else "Original RGB"
                coordinate_title = ("wrist-relative | fixed L/R anchors"
                                    if args.coordinate_mode == "root-relative" else "camera coordinates")
                put_text(canvas, left_title, (16, 28), half - 30, 0.65)
                put_text(canvas, f"3D annotation | {coordinate_title}",
                         (half + 16, 28), half - 30, 0.6)
                source_title = str(frame.rows[0].get("source_experiment", source))
                put_text(canvas, f"{name} | {source_title}", (16, args.height - 39), args.width - 32)
                put_text(canvas, f"{camera} | source frame {frame.index} | segment {segment_index}/{len(segments)}"
                         f" | annotation preview {args.fps:g} FPS", (16, args.height - 15), args.width - 32)
                writer.write(canvas)
                stats["written_frames"] += 1
                stats["empty_frames"] += int(not hands)
                if stats["written_frames"] % 100 == 0:
                    LOG.info("%s: %d/%d frames written", name, stats["written_frames"], len(frames))
        if not stats["written_frames"]:
            raise ValueError(f"No readable RGB frames in {root}")
        success = True
    finally:
        try:
            writer.release()
        except BaseException:
            success = False
            raise
        finally:
            if scene is not None:
                scene.close()
            if not success:
                temporary.unlink(missing_ok=True)
    temporary.replace(output)
    LOG.info("Saved %s | %s", output, stats)
    return {"output": str(output), **stats}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, help="Dataset containing index.jsonl (or its parent).")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output/hand_3d_videos")
    parser.add_argument("--fps", type=float, default=10.0, help="Annotation preview FPS, not source-video FPS.")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--max-frames", type=int, help="Limit selected RGB frame groups per dataset.")
    parser.add_argument("--overlay-2d", action="store_true",
                        help="Draw the stored GT boxes and projected 21-joint skeletons on RGB.")
    parser.add_argument("--coordinate-mode", choices=("camera", "root-relative"),
                        default="camera", help="Coordinate system used by the right 3D panel.")
    args = parser.parse_args(argv)
    if not np.isfinite(args.fps) or args.fps <= 0:
        parser.error("--fps must be finite and positive")
    if args.width < 640 or args.height < 480 or args.width % 2 or args.height % 2:
        parser.error("video size must be even and at least 640x480")
    if args.max_frames is not None and args.max_frames < 1:
        parser.error("--max-frames must be positive")
    args.output_dir = args.output_dir.expanduser().resolve()
    return args


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    roots = [args.dataset_root] if args.dataset_root else DEFAULT_DATASETS
    for root in roots:
        process_dataset(root, args)


if __name__ == "__main__":
    main()
