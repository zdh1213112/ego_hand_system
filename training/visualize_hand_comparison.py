"""Compare saved validation predictions with GT in an RGB / 3D H.264 video."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import json
import logging
from pathlib import Path

import cv2
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import numpy as np
import yaml

from visualize_hand_3d import (
    CONNECTIONS, Frame, H264VideoWriter, HandScene, ROOT, camera_joints,
    display_points, letterbox, put_text, read_frames, resolve, scene_bounds,
)

LOG = logging.getLogger("hand_comparison")
GT_COLOR = "#219653"
PRED_COLOR = "#e13d54"


@dataclass
class Pair:
    side: int
    gt: np.ndarray
    pred: np.ndarray
    mpjpe_mm: float
    sample_id: str


def matched_pair(root: Path, row: dict, record: dict) -> Pair:
    """Undo left canonicalization and anchor BOTH skeletons at the GT wrist.

    Evaluation only stores wrist-relative predictions. Its pred_cam_t belongs
    to the crop camera and must not be used as a full-image camera translation.
    """
    sample_id = str(row["index"])
    for key in ("side", "sync_index", "source_frame_index", "camera"):
        if str(row.get(key, "")) != str(record.get(key, "")):
            raise ValueError(f"Sample {sample_id}: mismatched {key}")
    side = int(record["side"])
    if side not in (0, 1):
        raise ValueError(f"Sample {sample_id}: invalid hand side {side}")
    label = np.load(resolve(root, row["label"]), allow_pickle=True).item()
    if int(label["side"]) != side:
        raise ValueError(f"Sample {sample_id}: label side mismatch")
    gt_camera = camera_joints(label)
    relative = []
    for key in ("gt_joints_3d_root_relative", "pred_joints_3d_root_relative"):
        points = np.array(record[key], dtype=np.float64, copy=True)
        if points.shape != (21, 3) or not np.isfinite(points).all():
            raise ValueError(f"Sample {sample_id}: invalid {key}")
        if not np.allclose(points[0], 0, atol=1e-6):
            raise ValueError(f"Sample {sample_id}: {key} is not wrist-relative")
        if side == 0:
            points[:, 0] *= -1
        relative.append(points)
    gt_root, pred_root = relative
    if not np.allclose(gt_root, gt_camera - gt_camera[:1], atol=1e-5, rtol=0):
        raise ValueError(f"Sample {sample_id}: saved GT does not match dataset label")
    error = float(np.linalg.norm(pred_root - gt_root, axis=1).mean() * 1000)
    return Pair(side, gt_camera, pred_root + gt_camera[:1], error, sample_id)


def load_comparisons(root: Path, predictions: Path) -> list[tuple[Frame, list[Pair]]]:
    records = {}
    with predictions.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            sample_id = str(record["sample_id"])
            if sample_id in records:
                raise ValueError(f"{predictions}:{line_number}: duplicate sample_id {sample_id}")
            records[sample_id] = record
    result = []
    matched = set()
    for frame in read_frames(root):
        pairs, matched_rows = [], []
        for row in frame.rows:
            sample_id = str(row["index"])
            if sample_id not in records:
                continue
            record = records[sample_id]
            expected_source = str(row.get("source_group") or frame.source)
            if expected_source == "recording":
                expected_source = ""
            if record.get("source_group", "") != expected_source:
                raise ValueError(f"Sample {sample_id}: mismatched recording source")
            if sample_id in matched:
                raise ValueError(f"Duplicate manifest sample_id {sample_id}")
            pairs.append(matched_pair(root, row, record))
            matched_rows.append(row)
            matched.add(sample_id)
        if pairs:
            if len({p.side for p in pairs}) != len(pairs):
                raise ValueError(f"Duplicate hand side: {frame.source}, {frame.camera}, {frame.index}")
            # Keep only evaluated hands; do not imply predictions for other GT hands.
            result.append((Frame(frame.source, frame.camera, frame.index, matched_rows), pairs))
    missing = set(records) - matched
    if missing:
        raise ValueError(f"Predictions have no matching labels: {sorted(missing)[:10]}")
    if not result:
        raise ValueError("No matched GT/prediction frames")
    return result


class ComparisonScene(HandScene):
    def __init__(self, width: int, height: int):
        super().__init__(width, height)
        for text in list(self.figure.texts):
            text.remove()
        self.figure.text(0.32, 0.045, "GT (solid)", color=GT_COLOR, fontsize=12, ha="center")
        self.figure.text(0.68, 0.045, "Prediction (dashed)", color=PRED_COLOR, fontsize=12, ha="center")

    def render_pairs(self, pairs: list[Pair]) -> np.ndarray:
        for artist in self.artists:
            artist.remove()
        self.artists.clear()
        self.empty_text.set_visible(not pairs)
        for pair in pairs:
            for coordinates, color, style in ((pair.gt, GT_COLOR, "solid"), (pair.pred, PRED_COLOR, "dashed")):
                points = display_points(coordinates)
                lines = Line3DCollection(points[np.asarray(CONNECTIONS)], colors=color,
                                         linewidths=2.2, linestyles=style)
                self.axes.add_collection3d(lines)
                dots = self.axes.scatter(*points.T, c=color, s=18, depthshade=False)
                self.artists.extend((lines, dots))
            wrist = display_points(pair.gt[:1])[0]
            text = self.axes.text(*wrist, " L" if pair.side == 0 else " R", fontsize=12,
                                  color="#333333", weight="bold")
            self.artists.append(text)
        self.canvas.draw()
        return cv2.cvtColor(np.asarray(self.canvas.buffer_rgba()), cv2.COLOR_RGBA2BGR)


def choose_example(comparisons: list[tuple[Frame, list[Pair]]]) -> int:
    """Prefer both hands, with frame MPJPE nearest the overall frame median."""
    errors = np.array([np.mean([p.mpjpe_mm for p in pairs]) for _, pairs in comparisons])
    candidates = [i for i, (_, pairs) in enumerate(comparisons) if len(pairs) == 2]
    if not candidates:
        candidates = list(range(len(comparisons)))
    return min(candidates, key=lambda i: abs(errors[i] - np.median(errors)))


def render_comparison(image: np.ndarray, frame: Frame, pairs: list[Pair], scene: ComparisonScene,
                      args: argparse.Namespace, segment: str) -> np.ndarray:
    width, height = args.width, args.height
    half, header, footer = width // 2, 66, 88
    panel_height = height - header - footer
    canvas = np.full((height, width, 3), 24, dtype=np.uint8)
    canvas[header:header + panel_height, :half] = letterbox(image, half, panel_height)
    canvas[header:header + panel_height, half:] = scene.render_pairs(pairs)
    put_text(canvas, "Original RGB", (16, 28), half - 30, 0.65)
    put_text(canvas, "GT + trained prediction", (half + 16, 28), half - 30, 0.65)
    put_text(canvas, "GT wrist aligned | pose comparison only", (half + 16, 51), half - 30, 0.52)
    errors = " | ".join(f"{'L' if p.side == 0 else 'R'} MPJPE: {p.mpjpe_mm:.1f} mm" for p in pairs)
    put_text(canvas, errors, (16, height - 63), width - 32, 0.60)
    put_text(canvas, f"{args.run_dir.name} | {frame.source} | {frame.camera} | frame {frame.index}",
             (16, height - 39), width - 32)
    put_text(canvas, f"Saved validation predictions | segment {segment} | preview {args.fps:g} FPS",
             (16, height - 15), width - 32)
    return canvas


def process_comparison(args: argparse.Namespace) -> dict:
    comparisons = load_comparisons(args.dataset_root, args.predictions)
    if args.max_frames is not None:
        comparisons = comparisons[:args.max_frames]
    LOG.info("Matched %d RGB frames, %d hands", len(comparisons), sum(len(p) for _, p in comparisons))
    example_index = choose_example(comparisons)
    example_frame = comparisons[example_index][0]
    segments = defaultdict(list)
    for frame, pairs in comparisons:
        segments[frame.source, frame.camera].append((frame, pairs))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "rgb_gt_pred_3d.mp4"
    temporary = output.with_name("rgb_gt_pred_3d.partial.mp4")
    example = args.output_dir / "rgb_gt_pred_example.png"
    temporary_example = example.with_name("rgb_gt_pred_example.partial.png")
    writer = H264VideoWriter(temporary, args.fps, args.width, args.height)
    scene = None
    success = False
    written = 0
    try:
        scene = ComparisonScene(args.width - args.width // 2, args.height - 154)
        for segment_index, ((source, camera), values) in enumerate(segments.items(), 1):
            LOG.info("Segment %d/%d: %s, %s, %d frames", segment_index, len(segments), source, camera, len(values))
            clouds = [[(p.side, points) for p in pairs for points in (p.gt, p.pred)] for _, pairs in values]
            scene.set_bounds(*scene_bounds(clouds))
            for frame, pairs in values:
                image = None
                for image_path in dict.fromkeys(row["image"] for row in frame.rows):
                    image = cv2.imread(str(resolve(args.dataset_root, image_path)))
                    if image is not None:
                        break
                if image is None:
                    raise OSError(f"Cannot read RGB for {source}, {camera}, frame {frame.index}")
                canvas = render_comparison(image, frame, pairs, scene, args, f"{segment_index}/{len(segments)}")
                writer.write(canvas)
                if frame is example_frame and not args.no_example_image:
                    if not cv2.imwrite(str(temporary_example), canvas):
                        raise OSError(f"Cannot write {temporary_example}")
                written += 1
                if written % 100 == 0:
                    LOG.info("Written %d/%d frames", written, len(comparisons))
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
                temporary_example.unlink(missing_ok=True)
    temporary.replace(output)
    if not args.no_example_image:
        temporary_example.replace(example)
    report = {
        "predictions": str(args.predictions), "dataset_root": str(args.dataset_root),
        "coordinate_mode": "physical camera axes, prediction anchored at GT wrist; pose only",
        "prediction_camera_translation_used": False,
        "frames": written, "hands": sum(len(p) for _, p in comparisons),
        "mean_mpjpe_mm": float(np.mean([p.mpjpe_mm for _, pairs in comparisons for p in pairs])),
        "fps": args.fps, "output": str(output),
        "example": None if args.no_example_image else {
            "path": str(example), "selection": "prefer two hands, nearest overall frame median MPJPE",
            "source": example_frame.source, "camera": example_frame.camera,
            "source_frame_index": example_frame.index,
            "sample_ids": [p.sample_id for p in comparisons[example_index][1]],
        },
    }
    (args.output_dir / "comparison_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    LOG.info("Saved %s (%d frames); mean wrist-relative MPJPE %.2f mm", output, written, report["mean_mpjpe_mm"])
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--predictions", type=Path, help="Saved evaluation predictions.jsonl; defaults to run-dir/val_visualization.")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--fps", type=float, default=10)
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--no-example-image", action="store_true")
    args = parser.parse_args(argv)
    args.run_dir = args.run_dir.expanduser().resolve()
    if args.dataset_root is None:
        config = yaml.safe_load((args.run_dir / "resolved_config.yaml").read_text())
        args.dataset_root = Path(config["paths"]["dataset_root"])
        if not args.dataset_root.is_absolute():
            args.dataset_root = ROOT / args.dataset_root
    args.dataset_root = args.dataset_root.expanduser().resolve()
    if not (args.dataset_root / "index.jsonl").is_file() and (args.dataset_root / "dataset/index.jsonl").is_file():
        args.dataset_root = args.dataset_root / "dataset"
    args.predictions = (args.predictions or args.run_dir / "val_visualization/predictions.jsonl").expanduser().resolve()
    args.output_dir = (args.output_dir or args.run_dir / "comparison_visualization").expanduser().resolve()
    if not np.isfinite(args.fps) or args.fps <= 0:
        parser.error("--fps must be finite and positive")
    if args.width < 960 or args.height < 600 or args.width % 2 or args.height % 2:
        parser.error("video dimensions must be even and at least 960x600")
    if args.max_frames is not None and args.max_frames < 1:
        parser.error("--max-frames must be positive")
    return args


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    process_comparison(parse_args())


if __name__ == "__main__":
    main()
