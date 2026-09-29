#!/usr/bin/env python3
"""Create SAM3 text-only hand proposals for the WiLoR front end.

The normal WiLoR path asks YOLO for a hand box first.  With gloves, YOLO can
return one merged/partial box or miss the second hand entirely.  This helper
uses the existing protective-glove SAM3 segmenter in *text-only* mode and
merges its boxes with the saved YOLO boxes.  The resulting proposal tree is
consumed by :mod:`wilor_multiview_inference` through ``--external-boxes``.

This is deliberately a proposal stage, not a pose stage: SAM3 never changes
WiLoR joints.  A SAM proposal is only added when it is not already represented
by a good YOLO box; WiLoR remains responsible for the 2D/3D hand estimate.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time
from typing import Any

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from ego_data.dataset import SequentialVideoReader  # noqa: E402


DEFAULT_TOOL_ROOT = Path("/home/p3/data_sda1/tool/auto-obb-annotator")
DEFAULT_MODEL = DEFAULT_TOOL_ROOT / "models/sam3.pt"
DEFAULT_PROMPT = "hand wearing a black protective work glove"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument(
        "--predictions", required=True, type=Path,
        help="saved WiLoR/YOLO prediction tree used as the base proposals",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cameras", nargs="+")
    parser.add_argument("--tool-root", type=Path, default=DEFAULT_TOOL_ROOT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument(
        "--min-score", type=float, default=0.20,
        help="discard low-score text-only masks before adding proposals",
    )
    parser.add_argument(
        "--pad-ratio", type=float, default=0.12,
        help="expand a SAM box on each side so the WiLoR crop includes the wrist",
    )
    parser.add_argument(
        "--merge-iou", type=float, default=0.35,
        help="SAM/YOLO overlap at which a similarly sized box is considered covered",
    )
    parser.add_argument(
        "--duplicate-iou", type=float, default=0.85,
        help="suppress near-identical proposals after merging",
    )
    parser.add_argument(
        "--duplicate-containment", type=float, default=0.90,
        help="suppress nested duplicate proposals after merging",
    )
    parser.add_argument(
        "--max-proposals", type=int, default=4,
        help="maximum physical hand boxes per camera frame",
    )
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--progress-interval", type=int, default=10)
    return parser.parse_args()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _bbox_area(box: Any) -> float:
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return 0.0
    x1, y1, x2, y2 = (float(value) for value in box)
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _bbox_iou(a: Any, b: Any) -> float:
    if not isinstance(a, (list, tuple)) or not isinstance(b, (list, tuple)):
        return 0.0
    if len(a) != 4 or len(b) != 4:
        return 0.0
    ax1, ay1, ax2, ay2 = (float(value) for value in a)
    bx1, by1, bx2, by2 = (float(value) for value in b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = _bbox_area(a) + _bbox_area(b) - intersection
    return intersection / union if union > 0.0 else 0.0


def _containment(inner: Any, outer: Any) -> float:
    """Area(inner ∩ outer) / area(inner)."""
    if not isinstance(inner, (list, tuple)) or not isinstance(outer, (list, tuple)):
        return 0.0
    if len(inner) != 4 or len(outer) != 4:
        return 0.0
    ix1, iy1, ix2, iy2 = (float(value) for value in inner)
    ox1, oy1, ox2, oy2 = (float(value) for value in outer)
    area = _bbox_area(inner)
    if area <= 0.0:
        return 0.0
    overlap = max(0.0, min(ix2, ox2) - max(ix1, ox1)) * max(
        0.0, min(iy2, oy2) - max(iy1, oy1)
    )
    return overlap / area


def _clip_box(box: Any, width: int, height: int) -> list[float] | None:
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    try:
        x1, y1, x2, y2 = (float(value) for value in box)
    except (TypeError, ValueError):
        return None
    x1 = max(0.0, min(float(width - 1), x1))
    y1 = max(0.0, min(float(height - 1), y1))
    x2 = max(x1 + 1.0, min(float(width), x2))
    y2 = max(y1 + 1.0, min(float(height), y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def _pad_box(box: Any, width: int, height: int, ratio: float) -> list[float] | None:
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    x1, y1, x2, y2 = (float(value) for value in box)
    dx = max(2.0, (x2 - x1) * float(ratio))
    dy = max(2.0, (y2 - y1) * float(ratio))
    return _clip_box([x1 - dx, y1 - dy, x2 + dx, y2 + dy], width, height)


def _unique_yolo_boxes(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Collapse WiLoR's two handedness hypotheses to physical detector boxes."""
    by_detection: dict[int, dict[str, Any]] = {}
    for hand in record.get("hands", []):
        if not isinstance(hand, dict):
            continue
        try:
            index = int(hand["detection_index"])
            box = [float(value) for value in hand.get("bbox_xyxy", [])]
            confidence = float(hand.get("confidence", 0.0))
        except (KeyError, TypeError, ValueError):
            continue
        if len(box) != 4 or _bbox_area(box) <= 0.0:
            continue
        old = by_detection.get(index)
        if old is None or confidence > old["confidence"]:
            by_detection[index] = {
                "bbox_xyxy": box,
                "confidence": confidence,
                "detector_is_right": hand.get("detector_is_right"),
                "source": "yolo",
            }
    return list(by_detection.values())


def _deduplicate(
    proposals: list[dict[str, Any]], iou_threshold: float,
    containment_threshold: float,
) -> list[dict[str, Any]]:
    """Keep the most trustworthy member of near-identical/nested boxes."""
    ranked = sorted(
        proposals,
        key=lambda item: (
            1 if item.get("source") == "yolo" else 0,
            float(item.get("confidence", 0.0)),
            _bbox_area(item.get("bbox_xyxy", [])),
        ),
        reverse=True,
    )
    kept: list[dict[str, Any]] = []
    for candidate in ranked:
        box = candidate.get("bbox_xyxy", [])
        duplicate = False
        for previous in kept:
            other = previous.get("bbox_xyxy", [])
            if _bbox_iou(box, other) >= iou_threshold:
                duplicate = True
                break
            # A much smaller SAM proposal inside a giant YOLO merge is useful,
            # so only suppress nested boxes when their scales are comparable.
            area = _bbox_area(box)
            other_area = _bbox_area(other)
            if area > 0.0 and other_area > 0.0:
                if _containment(box, other) >= containment_threshold and area >= 0.55 * other_area:
                    duplicate = True
                    break
                if _containment(other, box) >= containment_threshold and other_area >= 0.55 * area:
                    duplicate = True
                    break
        if not duplicate:
            kept.append(candidate)
    return kept


def _merge_proposals(
    yolo_boxes: list[dict[str, Any]], sam_boxes: list[list[float]],
    sam_scores: list[float], width: int, height: int, args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    proposals = list(yolo_boxes)
    added = 0
    skipped_covered = 0
    for raw_box, raw_score in zip(sam_boxes, sam_scores):
        try:
            score = float(raw_score)
        except (TypeError, ValueError):
            continue
        if score < args.min_score:
            continue
        box = _pad_box(raw_box, width, height, args.pad_ratio)
        if box is None:
            continue
        sam_area = _bbox_area(box)
        covered = False
        for yolo in yolo_boxes:
            yolo_box = yolo.get("bbox_xyxy", [])
            iou = _bbox_iou(box, yolo_box)
            ratio = sam_area / max(_bbox_area(yolo_box), 1e-6)
            # Same physical box, allowing for SAM's modest padding.
            if iou >= args.merge_iou and ratio >= 0.55:
                covered = True
                break
            # If a detector box is almost entirely inside SAM and comparable
            # in scale, SAM adds no new physical hand.
            if _containment(yolo_box, box) >= 0.85 and ratio <= 1.8:
                covered = True
                break
        if covered:
            skipped_covered += 1
            continue
        proposals.append({
            "bbox_xyxy": box,
            "confidence": score,
            "detector_is_right": None,
            "source": "sam3_text",
            "sam3_score": score,
        })
        added += 1
    proposals = _deduplicate(
        proposals, args.duplicate_iou, args.duplicate_containment
    )
    proposals.sort(key=lambda item: float(item.get("confidence", 0.0)), reverse=True)
    trimmed = max(0, len(proposals) - args.max_proposals)
    if args.max_proposals > 0:
        proposals = proposals[: args.max_proposals]
    return proposals, {
        "yolo_boxes": len(yolo_boxes),
        "sam_masks": len(sam_boxes),
        "sam_added": added,
        "sam_skipped_covered": skipped_covered,
        "proposals": len(proposals),
        "proposals_trimmed": trimmed,
    }


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"output already exists: {args.output}")
    if args.max_frames < 0 or args.max_proposals < 1:
        raise ValueError("max-frames must be non-negative and max-proposals positive")
    if not 0.0 <= args.min_score <= 1.0 or not 0.0 <= args.conf <= 1.0:
        raise ValueError("conf/min-score must be in [0, 1]")
    dataset = args.dataset.resolve()
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("dataset_type") != "normalized_multiview":
        raise ValueError(f"not a normalized multiview dataset: {dataset}")
    with (dataset / "multiview_frames.csv").open(
        "r", encoding="utf-8", newline=""
    ) as stream:
        sync_rows = list(csv.DictReader(stream))
    sync_rows = sync_rows[: args.max_frames or None]
    cameras = tuple(args.cameras or manifest["camera_ids"])
    unknown = sorted(set(cameras) - set(manifest["camera_ids"]))
    if unknown:
        raise ValueError(f"unknown cameras: {unknown}")
    image_size = tuple(manifest["image_size"])
    width, height = int(image_size[0]), int(image_size[1])

    sys.path.insert(0, str(args.tool_root.resolve()))
    from auto_obb_annotator.app_protective_gloves import (  # noqa: E402
        ProtectiveGloveSAM3TextSegmenter,
    )

    print(f"loading SAM3 text proposal model: {args.model}", flush=True)
    segmenter = ProtectiveGloveSAM3TextSegmenter(
        model_path=str(args.model), device=args.device, conf=args.conf,
    )
    output = args.output.resolve()
    output.mkdir(parents=True)
    summary_by_camera: dict[str, Any] = {}
    total_started = time.perf_counter()
    for camera in cameras:
        source_path = args.predictions.resolve() / camera / "predictions.jsonl"
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        source_rows = _load_jsonl(source_path)
        source_by_sync = {
            int(row["sync_index"]): row for row in source_rows
            if "sync_index" in row
        }
        camera_output = output / camera
        camera_output.mkdir()
        reader = SequentialVideoReader(
            dataset / "cameras" / camera
            / manifest["storage"]["video_filename"], image_size,
        )
        counters = {
            "frame_count": 0,
            "frames_with_yolo": 0,
            "frames_with_two_proposals": 0,
            "yolo_boxes": 0,
            "sam_masks": 0,
            "sam_added": 0,
            "sam_skipped_covered": 0,
            "proposals": 0,
            "proposals_trimmed": 0,
        }
        started = time.perf_counter()
        try:
            with (camera_output / "predictions.jsonl").open(
                "w", encoding="utf-8"
            ) as stream:
                for ordinal, sync_row in enumerate(sync_rows, start=1):
                    sync_index = int(sync_row["sync_index"])
                    source = source_by_sync.get(sync_index, {})
                    frame_index = int(sync_row[f"{camera}_frame_index"])
                    frame = reader.read(frame_index)
                    masks, sam_boxes, sam_scores = segmenter.detect(
                        frame, args.prompt
                    )
                    del masks  # proposals use boxes; masks are not serialized here
                    proposals, stats = _merge_proposals(
                        _unique_yolo_boxes(source), sam_boxes, sam_scores,
                        width, height, args,
                    )
                    result = {
                        "sync_index": sync_index,
                        "source_frame_index": frame_index,
                        "timestamp_ns": int(sync_row[f"{camera}_timestamp_ns"]),
                        "boxes": proposals,
                        "sam3": {
                            "prompt": args.prompt,
                            "text_only": True,
                            "raw_mask_count": len(sam_boxes),
                            "accepted_min_score": args.min_score,
                        },
                    }
                    stream.write(json.dumps(result, separators=(",", ":")) + "\n")
                    counters["frame_count"] += 1
                    counters["frames_with_yolo"] += int(stats["yolo_boxes"] > 0)
                    counters["frames_with_two_proposals"] += int(stats["proposals"] >= 2)
                    for key in (
                        "yolo_boxes", "sam_masks", "sam_added", "sam_skipped_covered",
                        "proposals", "proposals_trimmed",
                    ):
                        counters[key] += stats[key]
                    if ordinal % args.progress_interval == 0 or ordinal == len(sync_rows):
                        elapsed = time.perf_counter() - started
                        print(
                            f"{camera}: {ordinal}/{len(sync_rows)} | "
                            f"2+ proposals {counters['frames_with_two_proposals']} | "
                            f"SAM added {counters['sam_added']} | "
                            f"{ordinal / max(elapsed, 1e-9):.2f} frames/s",
                            flush=True,
                        )
        finally:
            reader.close()
        elapsed = time.perf_counter() - started
        camera_summary = {
            **counters,
            "two_proposal_rate": counters["frames_with_two_proposals"] / max(counters["frame_count"], 1),
            "elapsed_seconds": elapsed,
            "frames_per_second": counters["frame_count"] / max(elapsed, 1e-9),
        }
        (camera_output / "summary.json").write_text(
            json.dumps(camera_summary, indent=2) + "\n", encoding="utf-8"
        )
        summary_by_camera[camera] = camera_summary
    summary = {
        "schema_version": 1,
        "stage": "sam3_text_only_hand_proposals",
        "dataset": str(dataset),
        "source_predictions": str(args.predictions.resolve()),
        "camera_ids": list(cameras),
        "parameters": {
            "model": str(args.model.resolve()),
            "prompt": args.prompt,
            "device": args.device,
            "conf": args.conf,
            "min_score": args.min_score,
            "pad_ratio": args.pad_ratio,
            "merge_iou": args.merge_iou,
            "duplicate_iou": args.duplicate_iou,
            "duplicate_containment": args.duplicate_containment,
            "max_proposals": args.max_proposals,
        },
        "frame_count": len(sync_rows),
        "cameras": summary_by_camera,
        "elapsed_seconds": time.perf_counter() - total_started,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
