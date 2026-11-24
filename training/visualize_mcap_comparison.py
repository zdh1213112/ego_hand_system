"""Render synchronized predictions from two WiLoR runs on one MCAP."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from mcap_video_demo import (
    camera_topic, estimated_fps, iter_mcap_frames, overlay_predictions,
    visualization_hands,
)
from visualize_hand_3d import H264VideoWriter, HandScene, letterbox, put_text, scene_bounds


LOG = logging.getLogger("mcap_comparison")


def load_run(directory: Path) -> tuple[dict, list[dict]]:
    report = json.loads((directory / "run_report.json").read_text(encoding="utf-8"))
    recorded_path = Path(report["predictions"])
    local_path = directory / recorded_path.name
    predictions = (local_path if local_path.is_file() else
                   recorded_path if recorded_path.is_absolute() else directory / recorded_path)
    report["predictions"] = str(predictions.resolve())
    records = [json.loads(line) for line in predictions.read_text(encoding="utf-8").splitlines()
               if line.strip()]
    if not records:
        raise ValueError(f"No predictions in {predictions}")
    return report, records


def validate_alignment(base: list[dict], finetuned: list[dict]) -> None:
    """Reject frame, timestamp or detector differences before comparing poses."""
    if len(base) != len(finetuned):
        raise ValueError("Base and fine-tuned frame counts differ")
    seen = set()
    for first, second in zip(base, finetuned):
        index = first["frame_index"]
        if index in seen:
            raise ValueError(f"Duplicate frame index {index}")
        seen.add(index)
        if index != second["frame_index"] or first["timestamp_ns"] != second["timestamp_ns"]:
            raise ValueError(f"Frame / timestamp mismatch at frame {index}")
        if len(first["hands"]) != len(second["hands"]):
            raise ValueError(f"Detection count mismatch at frame {index}")
        for left, right in zip(first["hands"], second["hands"]):
            if int(left["is_right"]) != int(right["is_right"]):
                raise ValueError(f"Detection class mismatch at frame {index}")
            for key in ("bbox_xyxy", "confidence"):
                if not np.allclose(left[key], right[key], rtol=0, atol=1e-5):
                    raise ValueError(f"Detection {key} mismatch at frame {index}")


def validate_run_settings(base: dict, finetuned: dict) -> None:
    for key in ("detector", "checkpoint"):
        if base.get(key) != finetuned.get(key):
            raise ValueError(f"Prediction run settings differ: {key}")
    # Older reports did not record these options. Their detector outputs are
    # still checked frame by frame by validate_alignment.
    for key in ("confidence", "iou", "max_detections_per_class", "rescale_factor"):
        if key in base and key in finetuned and base[key] != finetuned[key]:
            raise ValueError(f"Prediction run settings differ: {key}")


def process(args: argparse.Namespace) -> dict:
    base_report, base = load_run(args.base_dir)
    fine_report, fine = load_run(args.finetuned_dir)
    for report in (base_report, fine_report):
        if Path(report["input"]).resolve() != args.input or report["camera"] != args.camera:
            raise ValueError("Prediction run does not match the input MCAP / camera")
    validate_run_settings(base_report, fine_report)
    validate_alignment(base, fine)
    if args.max_frames is not None:
        base, fine = base[:args.max_frames], fine[:args.max_frames]
    fps = args.fps or estimated_fps(base, 30.0)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / (args.output_name or
                               f"{args.input.stem}_{args.camera}_base_vs_finetuned.mp4")
    temporary = output.with_name(f"{output.stem}.partial.mp4")
    half = args.width // 2
    header, middle, footer = 50, 36, 54
    panel_height = (args.height - header - middle - footer) // 2
    three_d_top = header + panel_height + middle
    scenes = [HandScene(half, panel_height), HandScene(args.width - half, panel_height)]
    poses = [
        [visualization_hands(record["hands"], "root-relative") for record in records]
        for records in (base, fine)
    ]
    bounds = scene_bounds(poses[0] + poses[1])
    for scene in scenes:
        scene.set_bounds(*bounds)
    indexed = {a["frame_index"]: (a, b, pa, pb)
               for a, b, pa, pb in zip(base, fine, poses[0], poses[1])}
    first_timestamp = int(base[0]["timestamp_ns"])
    base_label = (args.base_label or base_report.get("model_label") or
                  ("Base WiLoR" if base_report.get("adapter") is None else
                   f"WiLoR ({Path(base_report['adapter']).parent.name})"))
    fine_label = (args.finetuned_label or fine_report.get("model_label")
                  or ("Base WiLoR" if fine_report.get("adapter") is None else
                      f"WiLoR ({Path(fine_report['adapter']).parent.name})"))
    writer = H264VideoWriter(temporary, fps, args.width, args.height)
    written = 0
    success = False
    try:
        for index, timestamp, rgb in iter_mcap_frames(args.input, args.camera, args.max_frames):
            if index not in indexed:
                raise RuntimeError(f"Unexpected decoded frame {index}")
            first, second, base_pose, fine_pose = indexed[index]
            if timestamp != first["timestamp_ns"]:
                raise RuntimeError(f"Decoded timestamp mismatch at frame {index}")
            canvas = np.full((args.height, args.width, 3), 24, dtype=np.uint8)
            for column, record, pose, scene in zip(
                    (0, 1), (first, second), (base_pose, fine_pose), scenes):
                start = column * half
                canvas[header:header + panel_height, start:start + half] = letterbox(
                    overlay_predictions(rgb, record["hands"]), half, panel_height
                )
                canvas[three_d_top:three_d_top + panel_height, start:start + half] = scene.render(pose)
                name = base_label if column == 0 else fine_label
                put_text(canvas, f"{name} | {args.camera} | YOLO + 2D joints",
                         (start + 16, 33), half - 32, 0.58)
                put_text(canvas, f"{name} | wrist-relative 3D | shared axes",
                         (start + 16, three_d_top - 12), half - 32, 0.56)
            labels = ", ".join(f"{'R' if h['is_right'] else 'L'} {h['confidence']:.2f}"
                               for h in first["hands"]) or "No hand detected"
            put_text(canvas, f"frame {index} | {(timestamp - first_timestamp) / 1e9:.3f}s | {labels}",
                     (16, args.height - 30), args.width - 32, 0.58)
            put_text(canvas, f"Same YOLO boxes / classes | {Path(base_report['detector']).name}"
                     " | separate display wrists for repeated classes",
                     (16, args.height - 9), args.width - 32, 0.48)
            writer.write(canvas)
            written += 1
            if written % 100 == 0:
                LOG.info("Rendered %d/%d comparison frames", written, len(base))
        if written != len(base):
            raise RuntimeError(f"Rendered {written} frames, expected {len(base)}")
        success = True
    finally:
        try:
            writer.release()
        except BaseException:
            success = False
            raise
        finally:
            for scene in scenes:
                scene.close()
            if not success:
                temporary.unlink(missing_ok=True)
    temporary.replace(output)
    report = {
        "input": str(args.input), "camera": args.camera,
        "base_run": str(args.base_dir), "finetuned_run": str(args.finetuned_dir),
        "left_run": str(args.base_dir), "right_run": str(args.finetuned_dir),
        "left_adapter": base_report.get("adapter"), "right_adapter": fine_report.get("adapter"),
        "left_label": base_label, "right_label": fine_label,
        "detector": base_report["detector"], "checkpoint": base_report["checkpoint"],
        "adapter": fine_report["adapter"], "detections_verified_identical": True,
        "finetuned_label": fine_label,
        "frames": written, "fps": fps,
        "frames_with_hands": sum(bool(record["hands"]) for record in base),
        "frames_with_multiple_hands": sum(len(record["hands"]) > 1 for record in base),
        "detector_class_counts": {
            str(side): sum(int(hand["is_right"]) == side for record in base for hand in record["hands"])
            for side in (0, 1)
        },
        "three_d_display": "wrist-relative camera axes, shared bounds across both models",
        "axis_center": bounds[0].tolist(), "axis_radius": bounds[1],
        "video": str(output),
    }
    report_path = args.output_dir / (f"{output.stem}_report.json" if args.output_name
                                   else "comparison_report.json")
    report_path.write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    LOG.info("Saved %s", output)
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--base-dir", "--left-dir", dest="base_dir", type=Path, required=True,
                        help="Run directory shown in the left column; may be base or trained.")
    parser.add_argument("--finetuned-dir", "--right-dir", dest="finetuned_dir", type=Path, required=True,
                        help="Run directory shown in the right column; may be base or trained.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-name", help="MP4 filename; a matching report filename is also used.")
    parser.add_argument("--camera", default="camera2")
    parser.add_argument("--base-label", "--left-label", dest="base_label",
                        help="Override the model name shown in the left column.")
    parser.add_argument("--finetuned-label", "--right-label", dest="finetuned_label",
                        help="Override the model name shown in the right column.")
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=1200)
    parser.add_argument("--fps", type=float)
    parser.add_argument("--max-frames", type=int)
    args = parser.parse_args(argv)
    for key in ("input", "base_dir", "finetuned_dir", "output_dir"):
        setattr(args, key, getattr(args, key).expanduser().resolve())
    if not args.input.is_file():
        parser.error("--input must be an existing MCAP file")
    if args.output_name is not None and (Path(args.output_name).name != args.output_name
                                        or Path(args.output_name).suffix.lower() != ".mp4"):
        parser.error("--output-name must be an MP4 filename without directory components")
    camera_topic(args.camera)
    if args.width < 1280 or args.height < 900 or args.width % 2 or args.height % 2:
        parser.error("comparison dimensions must be even and at least 1280x900")
    if args.fps is not None and (not np.isfinite(args.fps) or args.fps <= 0):
        parser.error("--fps must be finite and positive")
    if args.max_frames is not None and args.max_frames < 1:
        parser.error("--max-frames must be positive")
    return args


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    process(parse_args())


if __name__ == "__main__":
    main()
