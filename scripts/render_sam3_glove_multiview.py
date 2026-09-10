#!/usr/bin/env python3
"""Render a six-view video with the SAM3 glove masks over the camera images.

The regular SAM3 assist stage intentionally stores compact mask diagnostics,
not full-resolution masks.  This renderer therefore re-runs the same
YOLO-box + text prompted SAM3 pass and writes a tiled visualization without
changing any WiLoR, SAM3-assisted, or fusion outputs.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from ego_data.dataset import SequentialVideoReader  # noqa: E402
from render_wilor_predictions import HAND_CONNECTIONS  # noqa: E402
from sam3_glove_assist import (  # noqa: E402
    _detect_with_sam3,
    _match_masks_to_detections,
    _resize_mask,
    _unique_detections,
)


DEFAULT_TOOL_ROOT = Path("/home/p3/data_sda1/tool/auto-obb-annotator")
DEFAULT_PROMPT = "hand wearing a gray and black protective work glove"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument(
        "--predictions", required=True, type=Path,
        help="WiLoR prediction root (camera*/predictions.jsonl) or one predictions.jsonl",
    )
    parser.add_argument(
        "--assisted-predictions", type=Path,
        help="SAM3-assisted prediction root used only for corrected 2D skeletons/labels",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tool-root", type=Path, default=DEFAULT_TOOL_ROOT)
    parser.add_argument("--model", type=Path, default=DEFAULT_TOOL_ROOT / "models/sam3.pt")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument(
        "--cameras", nargs="+",
        default=["camera0", "camera1", "camera2", "camera3", "camera4", "camera5"],
    )
    parser.add_argument("--columns", type=int, default=3)
    parser.add_argument("--tile-width", type=int, default=480)
    parser.add_argument("--alpha", type=float, default=0.40)
    parser.add_argument("--progress-interval", type=int, default=10)
    parser.add_argument(
        "--text-only", action="store_true",
        help="use global text-only SAM3 prompts instead of YOLO-box + text prompts",
    )
    return parser.parse_args()


def _load_jsonl(path: Path) -> dict[int, dict]:
    records: dict[int, dict] = {}
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                record = json.loads(line)
                records[int(record["sync_index"])] = record
    return records


def _load_prediction_roots(
    path: Path, cameras: list[str], *, optional: bool = False,
) -> dict[str, dict[int, dict]]:
    """Load per-camera prediction records from a stage directory."""
    if path.is_file():
        shared = _load_jsonl(path)
        return {camera: shared for camera in cameras}
    records: dict[str, dict[int, dict]] = {}
    for camera in cameras:
        prediction_path = path / camera / "predictions.jsonl"
        if not prediction_path.is_file():
            if optional:
                records[camera] = {}
                continue
            raise FileNotFoundError(prediction_path)
        records[camera] = _load_jsonl(prediction_path)
    return records


def _load_dataset(dataset: Path) -> tuple[dict, list[dict[str, str]]]:
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    with (dataset / "multiview_frames.csv").open(
        "r", encoding="utf-8", newline=""
    ) as stream:
        rows = list(csv.DictReader(stream))
    return manifest, rows


def _color_for_detection(detection: dict, detection_index: int) -> tuple[int, int, int]:
    side = detection.get("detector_is_right")
    if side == 0:
        return (255, 205, 40)  # left: cyan/blue
    if side == 1:
        return (40, 220, 255)  # right: yellow
    palette = ((80, 220, 80), (220, 80, 220), (80, 180, 255), (255, 150, 60))
    return palette[detection_index % len(palette)]


def _draw_skeleton(
    image: np.ndarray, joints: object, color: tuple[int, int, int],
    thickness: int = 3, radius: int = 4,
) -> None:
    points = np.asarray(joints, dtype=np.float64)
    if points.shape != (21, 2):
        return
    finite = np.isfinite(points).all(axis=1)
    rounded = np.zeros((21, 2), dtype=np.int32)
    rounded[finite] = np.rint(points[finite]).astype(np.int32)
    for start, end in HAND_CONNECTIONS:
        if finite[start] and finite[end]:
            cv2.line(
                image, tuple(rounded[start]), tuple(rounded[end]),
                color, thickness, cv2.LINE_AA,
            )
    for point in rounded[finite]:
        cv2.circle(image, tuple(point), radius, color, -1, cv2.LINE_AA)


def _assisted_by_detection(record: dict | None) -> dict[int, list[dict]]:
    result: dict[int, list[dict]] = {}
    if not record:
        return result
    for hand in record.get("hands", []):
        try:
            index = int(hand["detection_index"])
        except (KeyError, TypeError, ValueError):
            continue
        result.setdefault(index, []).append(hand)
    return result


def _draw_sam_overlay(
    frame: np.ndarray,
    detections: list[dict],
    masks: list[np.ndarray],
    sam_boxes: list[list[float]],
    scores: list[float],
    association: dict[int, int],
    assisted_record: dict | None,
    alpha: float,
    camera: str,
    sync_index: int,
    prompt: str,
) -> np.ndarray:
    assisted = _assisted_by_detection(assisted_record)
    drawn = 0
    for detection_index, sam_index in sorted(association.items()):
        if detection_index < 0 or detection_index >= len(detections):
            continue
        if sam_index < 0 or sam_index >= len(masks):
            continue
        detection = detections[detection_index]
        color = _color_for_detection(detection, detection_index)
        mask = np.asarray(masks[sam_index], dtype=np.uint8)
        mask = _resize_mask(mask, frame.shape)
        if not np.any(mask):
            continue
        frame[mask] = (
            (1.0 - alpha) * frame[mask] + alpha * np.asarray(color, dtype=np.float32)
        ).astype(np.uint8)
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(frame, contours, -1, color, 2, cv2.LINE_AA)
        box = np.rint(np.asarray(detection.get("bbox_xyxy", []), dtype=np.float64)).astype(np.int32)
        if box.shape == (4,):
            cv2.rectangle(frame, tuple(box[:2]), tuple(box[2:]), color, 2, cv2.LINE_AA)
        sam_box = np.rint(np.asarray(sam_boxes[sam_index], dtype=np.float64)).astype(np.int32)
        if sam_box.shape == (4,):
            cv2.rectangle(frame, tuple(sam_box[:2]), tuple(sam_box[2:]), (70, 255, 70), 2, cv2.LINE_AA)

        # Draw the selected SAM3-constrained 2D hypothesis when available.
        hypotheses = assisted.get(int(detection.get("detection_index", detection_index)), [])
        chosen = [
            item for item in hypotheses
            if item.get("sam3_assist", {}).get("accepted")
        ] or hypotheses
        if chosen:
            def _support_value(item: dict) -> float:
                value = item.get("sam3_assist", {}).get(
                    "tolerant_inside_ratio_visible"
                )
                try:
                    return float(value) if value is not None else 0.0
                except (TypeError, ValueError):
                    return 0.0

            chosen = sorted(
                chosen,
                key=_support_value,
                reverse=True,
            )[0]
            _draw_skeleton(frame, chosen.get("joints_2d", []), color, 3, 4)
            info = chosen.get("sam3_assist", {})
            support = info.get("tolerant_inside_ratio_visible")
            support_text = f" J{float(support):.2f}" if support is not None else ""
        else:
            info = {}
            support_text = ""
        score = float(scores[sam_index]) if sam_index < len(scores) else 1.0
        label = f"SAM3 d{int(detection.get('detection_index', detection_index))} S{score:.2f}{support_text}"
        text_x = int(box[0]) if box.shape == (4,) else 8
        text_y = max(24, int(box[1]) - 8) if box.shape == (4,) else 24
        cv2.putText(
            frame, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
            0.55, (255, 255, 255), 2, cv2.LINE_AA,
        )
        drawn += 1
    # The title makes it unambiguous that this is a mask view, not the final
    # 3-D overlay.  Keep the prompt out of the title after a readable prefix.
    title = f"{camera} | SAM3 MASK | sync {sync_index} | masks {drawn}"
    cv2.rectangle(frame, (5, 5), (min(frame.shape[1] - 5, 610), 35), (18, 18, 18), -1)
    cv2.putText(
        frame, title, (12, 27), cv2.FONT_HERSHEY_SIMPLEX,
        0.60, (240, 240, 240), 2, cv2.LINE_AA,
    )
    return frame


def main() -> None:
    args = parse_args()
    if not args.cameras:
        raise ValueError("at least one camera is required")
    if args.columns < 1 or args.tile_width < 64:
        raise ValueError("columns must be positive and tile width must be at least 64")
    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    if args.max_frames < 0:
        raise ValueError("max-frames must be non-negative")
    for path in (args.dataset, args.predictions, args.tool_root, args.model):
        if not path.exists():
            raise FileNotFoundError(path)

    manifest, rows = _load_dataset(args.dataset)
    rows = rows[: args.max_frames or None]
    missing = [camera for camera in args.cameras if camera not in manifest.get("cameras", args.cameras)]
    if missing:
        raise ValueError(f"cameras are not present in dataset: {missing}")
    predictions = _load_prediction_roots(args.predictions, args.cameras)
    assisted = (
        _load_prediction_roots(args.assisted_predictions, args.cameras, optional=True)
        if args.assisted_predictions else {camera: {} for camera in args.cameras}
    )
    image_size = tuple(manifest["image_size"])
    tile_width = int(args.tile_width)
    tile_height = int(round(tile_width * image_size[1] / image_size[0]))
    columns = min(int(args.columns), len(args.cameras))
    rows_in_tile = int(np.ceil(len(args.cameras) / columns))
    frame_size = (tile_width * columns, tile_height * rows_in_tile)
    timestamps = np.asarray(
        [int(row["reference_timestamp_ns"]) for row in rows], dtype=np.int64
    )
    delta = np.diff(timestamps).astype(np.float64) / 1e9
    fps = float(1.0 / np.median(delta[delta > 0])) if np.any(delta > 0) else 30.0

    # Delay external imports/model loading until all local input checks pass.
    sys.path.insert(0, str(args.tool_root))
    from auto_obb_annotator.app_protective_gloves import (  # noqa: E402
        ProtectiveGloveSAM3TextSegmenter,
    )

    print(f"loading SAM3: {args.model}", flush=True)
    load_started = time.perf_counter()
    segmenter = ProtectiveGloveSAM3TextSegmenter(
        model_path=str(args.model), device=args.device, conf=args.conf,
    )
    print(f"SAM3 loaded in {time.perf_counter() - load_started:.1f}s", flush=True)

    readers = {
        camera: SequentialVideoReader(
            args.dataset / "cameras" / camera / manifest["storage"]["video_filename"],
            image_size,
        )
        for camera in args.cameras
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), fps, frame_size,
    )
    if not writer.isOpened():
        for reader in readers.values():
            reader.close()
        raise RuntimeError(f"failed to create video writer: {args.output}")

    started = time.perf_counter()
    processed = 0
    matched = 0
    total_masks = 0
    try:
        for row in rows:
            sync_index = int(row["sync_index"])
            tiles: list[np.ndarray] = []
            for camera in args.cameras:
                frame_index = int(row[f"{camera}_frame_index"])
                frame = readers[camera].read(frame_index)
                record = predictions[camera].get(sync_index, {"hands": []})
                detections = _unique_detections(record)
                masks, sam_boxes, scores = _detect_with_sam3(
                    segmenter, frame, args.prompt, detections,
                    args.device, args.conf, args.text_only,
                )
                association = _match_masks_to_detections(sam_boxes, detections)
                matched += len(association)
                total_masks += len(masks)
                frame = _draw_sam_overlay(
                    frame, detections, masks, sam_boxes, scores, association,
                    assisted[camera].get(sync_index), args.alpha,
                    camera, sync_index, args.prompt,
                )
                tiles.append(cv2.resize(frame, (tile_width, tile_height), interpolation=cv2.INTER_AREA))

            canvas = np.zeros((frame_size[1], frame_size[0], 3), dtype=np.uint8)
            for index, tile in enumerate(tiles):
                y = (index // columns) * tile_height
                x = (index % columns) * tile_width
                canvas[y:y + tile_height, x:x + tile_width] = tile
            writer.write(canvas)
            processed += 1
            if processed % max(1, args.progress_interval) == 0 or processed == len(rows):
                elapsed = time.perf_counter() - started
                print(
                    f"render SAM3 masks: {processed}/{len(rows)} | "
                    f"matched {matched} | {processed / max(elapsed, 1e-9):.2f} sync frames/s",
                    flush=True,
                )
    finally:
        writer.release()
        for reader in readers.values():
            reader.close()

    elapsed = time.perf_counter() - started
    summary = {
        "stage": "render_sam3_glove_mask_multiview",
        "dataset": str(args.dataset),
        "predictions": str(args.predictions),
        "assisted_predictions": str(args.assisted_predictions) if args.assisted_predictions else None,
        "model": str(args.model),
        "prompt": args.prompt,
        "prompt_mode": "text_only" if args.text_only else "yolo_box_plus_text",
        "camera_ids": args.cameras,
        "frame_count": processed,
        "fps": fps,
        "matched_detection_count": matched,
        "raw_sam_mask_count": total_masks,
        "elapsed_seconds": elapsed,
        "output": str(args.output),
    }
    summary_path = args.output.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
