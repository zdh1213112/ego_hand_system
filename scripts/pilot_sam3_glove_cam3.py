#!/usr/bin/env python3
"""Small SAM3 glove-assist pilot on an existing normalized camera3 run.

The pilot deliberately does not change WiLoR predictions.  It runs the
protective-glove SAM3 segmenter on a sparse set of frames, associates masks
with the existing YOLO boxes, and measures how well raw WiLoR joints and the
existing fused 3-D projection agree with the visible glove mask.

Run this from the ``annotator`` conda environment.  The external annotator
tree is added through ``--tool-root`` so the ego-hand repository stays
independent of its Qt application.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import csv
import json
import math
import sys
import time

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from camera_models import project_points  # noqa: E402
from ego_data.calibration import CameraCalibration  # noqa: E402
from ego_data.dataset import SequentialVideoReader  # noqa: E402
from render_wilor_predictions import HAND_CONNECTIONS  # noqa: E402


DEFAULT_VIDEO = (
    PROJECT_ROOT
    / "output/gen6_batch_0907/"
    "DAS-Ego_20260907180222_none_none_689985_7e8568fa/"
    "normalized_multiview/cameras/camera3/video.mkv"
)
DEFAULT_PREDICTIONS = (
    PROJECT_ROOT
    / "output/gen6_batch_0907/"
    "DAS-Ego_20260907180222_none_none_689985_7e8568fa/"
    "wilor_multiview/camera3/predictions.jsonl"
)
DEFAULT_FUSION = (
    PROJECT_ROOT
    / "output/gen6_batch_0907/"
    "DAS-Ego_20260907180222_none_none_689985_7e8568fa/"
    "fusion_multiview_optimized_v6/accepted.jsonl"
)
DEFAULT_DATASET = (
    PROJECT_ROOT
    / "output/gen6_batch_0907/"
    "DAS-Ego_20260907180222_none_none_689985_7e8568fa/normalized_multiview"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "output/gen6_batch_0907/"
    "DAS-Ego_20260907180222_none_none_689985_7e8568fa/"
    "sam3_pilot_cam3"
)
DEFAULT_TOOL_ROOT = Path("/home/p3/data_sda1/tool/auto-obb-annotator")
DEFAULT_PROMPT = "hand wearing a gray and black protective work glove"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--fusion", type=Path, default=DEFAULT_FUSION)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--tool-root", type=Path, default=DEFAULT_TOOL_ROOT)
    parser.add_argument(
        "--model", type=Path,
        default=DEFAULT_TOOL_ROOT / "models/sam3.pt",
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument(
        "--geometric-prompts", action="store_true",
        help="pass the existing YOLO boxes to SAM3 together with the text prompt",
    )
    parser.add_argument("--num-frames", type=int, default=300)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument(
        "--mask-margin-px", type=int, default=8,
        help="dilation margin used for a tolerant joint-in-mask score",
    )
    parser.add_argument(
        "--preview-fps", type=float, default=10.0,
        help="fps for the sparse-frame preview video",
    )
    return parser.parse_args()


def _read_jsonl(path: Path) -> dict[int, dict]:
    records: dict[int, dict] = {}
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                record = json.loads(line)
                records[int(record["sync_index"])] = record
    return records


def _read_sync_rows(dataset: Path) -> dict[int, dict[str, int]]:
    path = dataset / "multiview_frames.csv"
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = {}
        for row in csv.DictReader(stream):
            sync = int(row["sync_index"])
            rows[sync] = {
                "camera3_frame_index": int(row["camera3_frame_index"]),
            }
    return rows


def _box_iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return float(intersection / union) if union > 0.0 else 0.0


def _mask_box(mask: np.ndarray) -> list[float] | None:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]


def _mask_metrics(mask: np.ndarray, yolo_box, image_shape) -> dict:
    height, width = image_shape[:2]
    mask_bool = np.asarray(mask).astype(bool)
    box = np.asarray(yolo_box, dtype=np.float64)
    x1, y1, x2, y2 = box
    x1i = max(0, min(width, int(math.floor(x1))))
    y1i = max(0, min(height, int(math.floor(y1))))
    x2i = max(0, min(width, int(math.ceil(x2))))
    y2i = max(0, min(height, int(math.ceil(y2))))
    box_area = max(1, x2i - x1i) * max(1, y2i - y1i)
    mask_pixels = int(mask_bool.sum())
    inside_pixels = int(mask_bool[y1i:y2i, x1i:x2i].sum()) if x2i > x1i and y2i > y1i else 0
    mask_box = _mask_box(mask_bool)
    touches_border = bool(
        mask_box is not None
        and (mask_box[0] <= 1 or mask_box[1] <= 1
             or mask_box[2] >= width - 1 or mask_box[3] >= height - 1)
    )
    return {
        "mask_area_px": mask_pixels,
        "mask_area_ratio_image": float(mask_pixels / max(1, width * height)),
        "mask_area_ratio_yolo_box": float(inside_pixels / box_area),
        "mask_box": mask_box,
        "mask_box_iou_yolo": float(_box_iou(mask_box, yolo_box)) if mask_box else 0.0,
        "mask_touches_image_border": touches_border,
    }


def _joint_mask_metrics(mask: np.ndarray, joints, margin_px: int) -> dict:
    mask_bool = np.asarray(mask).astype(np.uint8)
    height, width = mask_bool.shape[:2]
    margin_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * max(0, margin_px) + 1, 2 * max(0, margin_px) + 1)
    )
    tolerant = cv2.dilate(mask_bool, margin_kernel, iterations=1) if margin_px > 0 else mask_bool
    points = np.asarray(joints, dtype=np.float64).reshape(-1, 2)
    visible = np.isfinite(points).all(axis=1)
    visible &= points[:, 0] >= 0.0
    visible &= points[:, 0] < width
    visible &= points[:, 1] >= 0.0
    visible &= points[:, 1] < height
    strict_inside = np.zeros(len(points), dtype=bool)
    tolerant_inside = np.zeros(len(points), dtype=bool)
    ys = np.clip(np.rint(points[visible, 1]), 0, height - 1).astype(np.int32)
    xs = np.clip(np.rint(points[visible, 0]), 0, width - 1).astype(np.int32)
    strict_inside[visible] = mask_bool[ys, xs] > 0
    tolerant_inside[visible] = tolerant[ys, xs] > 0
    visible_count = int(visible.sum())
    return {
        "visible_joint_count": visible_count,
        "strict_inside_count": int(strict_inside.sum()),
        "tolerant_inside_count": int(tolerant_inside.sum()),
        "strict_inside_ratio_visible": float(strict_inside.sum() / visible_count) if visible_count else None,
        "tolerant_inside_ratio_visible": float(tolerant_inside.sum() / visible_count) if visible_count else None,
        "strict_inside_ratio_all21": float(strict_inside.mean()) if len(points) else None,
        "tolerant_inside_ratio_all21": float(tolerant_inside.mean()) if len(points) else None,
    }


def _match_sam_to_yolo(sam_boxes, yolo_detections, image_shape) -> dict[int, int]:
    """Greedy one-to-one association; YOLO owns identity and box semantics."""
    height, width = image_shape[:2]
    pairs = []
    for sam_index, sam_box in enumerate(sam_boxes):
        sx = (float(sam_box[0]) + float(sam_box[2])) * 0.5
        sy = (float(sam_box[1]) + float(sam_box[3])) * 0.5
        for det_index, det in enumerate(yolo_detections):
            box = det["bbox_xyxy"]
            x1, y1, x2, y2 = [float(v) for v in box]
            center_in = x1 <= sx <= x2 and y1 <= sy <= y2
            iou = _box_iou(sam_box, box)
            # A center-contained mask is useful even when SAM tightens the box.
            score = 0.80 * iou + 0.20 * float(center_in)
            if center_in or iou >= 0.10:
                pairs.append((score, iou, sam_index, det_index))
    pairs.sort(reverse=True)
    result: dict[int, int] = {}
    used_sam: set[int] = set()
    for _, _, sam_index, det_index in pairs:
        if det_index in result or sam_index in used_sam:
            continue
        result[det_index] = sam_index
        used_sam.add(sam_index)
    return result


def _detect_sam(segmenter, image, prompt, detections, device, conf, geometric_prompts):
    """Return raw SAM3 masks/boxes/scores, optionally guided by YOLO boxes."""
    if not geometric_prompts:
        return segmenter.detect(image, prompt)
    boxes = [det["bbox_xyxy"] for det in detections if len(det.get("bbox_xyxy", [])) == 4]
    if not boxes:
        return [], [], []
    # SAM3's geometric prompt path may return more than one mask per input
    # box (for example a whole glove plus a small fragment).  We keep all
    # masks here and associate the best one to each YOLO box downstream.
    from ultralytics.engine.model import Model as UltralyticsModel
    from ultralytics.models.sam.predict import SAM3SemanticPredictor

    results = UltralyticsModel.predict(
        segmenter.model,
        source=image,
        predictor=SAM3SemanticPredictor,
        prompts={
            "bboxes": boxes,
            "labels": [1] * len(boxes),
            "text": [str(prompt)],
        },
        device=device,
        conf=conf,
        verbose=False,
    )
    if not results:
        return [], [], []
    result = results[0]
    masks = (
        result.masks.data.detach().cpu().numpy().astype(np.uint8).tolist()
        if result.masks is not None else []
    )
    boxes_out = (
        result.boxes.xyxy.detach().cpu().numpy().tolist()
        if result.boxes is not None and len(result.boxes) else []
    )
    scores = (
        result.boxes.conf.detach().cpu().numpy().tolist()
        if result.boxes is not None and result.boxes.conf is not None else []
    )
    return masks, boxes_out, scores


def _unique_detections(record: dict) -> list[dict]:
    detections = {}
    for hand in record.get("hands", []):
        index = int(hand.get("detection_index", len(detections)))
        item = detections.setdefault(index, {
            "detection_index": index,
            "bbox_xyxy": hand.get("bbox_xyxy", []),
            "confidence": float(hand.get("confidence", 0.0)),
            "detector_is_right": hand.get("detector_is_right"),
            "hypotheses": [],
        })
        item["hypotheses"].append(hand)
    return [detections[key] for key in sorted(detections)]


def _draw_skeleton(frame, joints, color, thickness=2, radius=3):
    points = np.asarray(joints, dtype=np.float64).reshape(-1, 2)
    if points.shape != (21, 2):
        return
    for start, end in HAND_CONNECTIONS:
        if not np.isfinite(points[[start, end]]).all():
            continue
        p1 = tuple(np.rint(points[start]).astype(int))
        p2 = tuple(np.rint(points[end]).astype(int))
        cv2.line(frame, p1, p2, color, thickness, cv2.LINE_AA)
    for point in points:
        if np.isfinite(point).all():
            cv2.circle(frame, tuple(np.rint(point).astype(int)), radius, color, -1, cv2.LINE_AA)


def _draw_overlay(frame, detections, sam_results, fused_results, sync_index, prompt):
    overlay = frame.copy()
    for result in sam_results.values():
        mask = result["mask"].astype(bool)
        color = np.asarray((45, 210, 80), dtype=np.uint8)
        overlay[mask] = (0.60 * overlay[mask] + 0.40 * color).astype(np.uint8)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0.0, frame)

    for det in detections:
        box = np.rint(det["bbox_xyxy"]).astype(int)
        if len(box) == 4:
            cv2.rectangle(frame, tuple(box[:2]), tuple(box[2:]), (255, 120, 0), 2, cv2.LINE_AA)
        det_index = det["detection_index"]
        sam = sam_results[det_index] if det_index in sam_results else None
        if sam is not None:
            sbox = np.rint(sam["sam_box"]).astype(int)
            cv2.rectangle(frame, tuple(sbox[:2]), tuple(sbox[2:]), (0, 220, 80), 2, cv2.LINE_AA)
        for hypothesis in det["hypotheses"]:
            side = int(hypothesis.get("is_right", 0))
            color = (40, 220, 255) if side else (255, 190, 40)
            _draw_skeleton(frame, hypothesis.get("joints_2d", []), color, 2, 3)
        label = f"YOLO d{det_index}"
        if sam is not None:
            ratio = sam["hypotheses"][0].get("tolerant_inside_ratio_visible")
            label += f" SAM {sam['score']:.2f}"
            if ratio is not None:
                label += f" J {ratio:.2f}"
        cv2.putText(frame, label, (max(4, int(box[0])), max(24, int(box[1]) - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

    for item in fused_results:
        _draw_skeleton(frame, item["projected_joints"], (255, 40, 220), 3, 4)
    cv2.rectangle(frame, (6, 6), (760, 54), (18, 18, 18), -1)
    cv2.putText(
        frame,
        f"SAM3 pilot cam3 | sync {sync_index} | {prompt[:75]}",
        (16, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (240, 240, 240), 2, cv2.LINE_AA,
    )
    return frame


def main() -> None:
    args = parse_args()
    for path in (args.video, args.predictions, args.dataset, args.model):
        if not path.exists():
            raise FileNotFoundError(path)
    args.output.mkdir(parents=True, exist_ok=True)
    sample_dir = args.output / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)

    predictions = _read_jsonl(args.predictions)
    sync_rows = _read_sync_rows(args.dataset)
    fusion = _read_jsonl(args.fusion) if args.fusion.is_file() else {}
    calibration = CameraCalibration.load(args.dataset / "calibration/camera3.json")

    available = sorted(set(predictions) & set(sync_rows))
    if not available:
        raise RuntimeError("no overlapping prediction and normalized sync indices")
    sample_count = min(max(1, int(args.num_frames)), len(available))
    selected = sorted(set(np.linspace(0, len(available) - 1, sample_count, dtype=int).tolist()))
    selected_sync = [available[index] for index in selected]
    selected_by_video_index = {
        int(sync_rows[sync]["camera3_frame_index"]): sync
        for sync in selected_sync
    }

    # Importing the specialized segmenter is intentionally delayed until after
    # input validation; this makes path/configuration errors cheap to diagnose.
    sys.path.insert(0, str(args.tool_root))
    from auto_obb_annotator.app_protective_gloves import (  # noqa: E402
        ProtectiveGloveSAM3TextSegmenter,
    )

    print(f"loading SAM3: {args.model}", flush=True)
    started = time.time()
    segmenter = ProtectiveGloveSAM3TextSegmenter(
        model_path=str(args.model), device=args.device, conf=args.conf,
    )
    print(f"SAM3 loaded in {time.time() - started:.1f}s", flush=True)

    reader = SequentialVideoReader(args.video, tuple(calibration.image_size))
    writer = None
    frame_records = []
    sample_images = []
    all_mask_metrics = []
    all_hypothesis_metrics = []
    fused_raw_metrics = []
    fused_projected_metrics = []
    matched_detections = 0
    total_detections = 0
    processed = 0
    target_indices = sorted(selected_by_video_index)
    try:
        for target_frame_index in target_indices:
            frame = reader.read(target_frame_index)
            sync_index = selected_by_video_index[target_frame_index]
            prediction = predictions[sync_index]
            detections = _unique_detections(prediction)
            total_detections += len(detections)

            infer_started = time.time()
            masks, sam_boxes, sam_scores = _detect_sam(
                segmenter,
                frame,
                args.prompt,
                detections,
                args.device,
                args.conf,
                args.geometric_prompts,
            )
            infer_seconds = time.time() - infer_started
            association = _match_sam_to_yolo(sam_boxes, detections, frame.shape)
            sam_results: dict[int, dict] = {}
            for det_index, sam_index in association.items():
                mask = np.asarray(masks[sam_index]).astype(np.uint8)
                det = detections[det_index]
                mask_info = _mask_metrics(mask, det["bbox_xyxy"], frame.shape)
                hypothesis_metrics = []
                for hypothesis in det["hypotheses"]:
                    joint_info = _joint_mask_metrics(
                        mask, hypothesis.get("joints_2d", []), args.mask_margin_px,
                    )
                    joint_info["is_right"] = hypothesis.get("is_right")
                    joint_info["hypothesis_index"] = hypothesis.get("hypothesis_index")
                    hypothesis_metrics.append(joint_info)
                    all_hypothesis_metrics.append(joint_info)
                sam_results[det_index] = {
                    "mask": mask,
                    "sam_box": [float(v) for v in sam_boxes[sam_index]],
                    "score": float(sam_scores[sam_index]) if sam_index < len(sam_scores) else 1.0,
                    "mask_metrics": mask_info,
                    "hypotheses": hypothesis_metrics,
                }
                all_mask_metrics.append({**mask_info, "score": sam_results[det_index]["score"]})
            matched_detections += len(sam_results)

            # Compare the existing refined fusion projection with raw cam3
            # only when that fused hand explicitly uses a camera3 observation.
            fused_items = []
            fused_record = fusion.get(sync_index)
            if fused_record:
                for fused_hand in fused_record.get("hands", []):
                    view = fused_hand.get("views", {}).get("camera3")
                    if not view or "joints_base_m" not in fused_hand:
                        continue
                    det_index = int(view.get("detection_index", -1))
                    if det_index not in sam_results:
                        continue
                    points_base = np.asarray(fused_hand["joints_base_m"], dtype=np.float64)
                    rotation = calibration.T_base_camera[:3, :3]
                    center = calibration.T_base_camera[:3, 3]
                    points_camera = (rotation.T @ (points_base - center).T).T
                    projected, valid = project_points(calibration, points_camera)
                    projected[~valid] = np.nan
                    raw_info = _joint_mask_metrics(
                        sam_results[det_index]["mask"], view.get("joints_2d", []), args.mask_margin_px,
                    )
                    projected_info = _joint_mask_metrics(
                        sam_results[det_index]["mask"], projected, args.mask_margin_px,
                    )
                    fused_raw_metrics.append(raw_info)
                    fused_projected_metrics.append(projected_info)
                    fused_items.append({
                        "detection_index": det_index,
                        "projected_joints": projected,
                        "raw": raw_info,
                        "projected": projected_info,
                    })

            compact_sam = {}
            for det_index, value in sam_results.items():
                compact_sam[str(det_index)] = {
                    "sam_box": value["sam_box"],
                    "score": value["score"],
                    **value["mask_metrics"],
                    "hypotheses": value["hypotheses"],
                }
            record = {
                "sync_index": sync_index,
                "video_frame_index": target_frame_index,
                "wilor_detection_count": len(detections),
                "sam_detection_count": len(sam_results),
                "sam_inference_seconds": infer_seconds,
                "sam_prompt_mode": "yolo_box_plus_text" if args.geometric_prompts else "text_only",
                "sam": compact_sam,
                "fused_camera3": [
                    {
                        "detection_index": item["detection_index"],
                        "raw": item["raw"],
                        "projected": item["projected"],
                    }
                    for item in fused_items
                ],
            }
            frame_records.append(record)

            if writer is None:
                writer = cv2.VideoWriter(
                    str(args.output / "sam3_preview_cam3.mp4"),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    float(args.preview_fps),
                    (frame.shape[1], frame.shape[0]),
                )
                if not writer.isOpened():
                    raise RuntimeError("failed to open pilot preview video writer")
            sam_for_draw = {int(key): value for key, value in sam_results.items()}
            overlay = _draw_overlay(
                frame.copy(), detections, sam_for_draw, fused_items, sync_index, args.prompt,
            )
            writer.write(overlay)
            if processed % max(1, len(selected_sync) // 24) == 0 or processed == len(selected_sync) - 1:
                image_path = sample_dir / f"sync_{sync_index:06d}.jpg"
                cv2.imwrite(str(image_path), overlay, [cv2.IMWRITE_JPEG_QUALITY, 90])
                sample_images.append(image_path)
            processed += 1
            if processed % 10 == 0 or processed == len(selected_sync):
                print(
                    f"processed {processed}/{len(selected_sync)} | "
                    f"SAM matched {matched_detections}/{total_detections} | "
                    f"last {infer_seconds:.2f}s",
                    flush=True,
                )
    finally:
        reader.close()
        if writer is not None:
            writer.release()

    (args.output / "metrics.jsonl").write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in frame_records),
        encoding="utf-8",
    )

    def median(values):
        values = [float(v) for v in values if v is not None and np.isfinite(v)]
        return float(np.median(values)) if values else None

    def percentile(values, q):
        values = [float(v) for v in values if v is not None and np.isfinite(v)]
        return float(np.percentile(values, q)) if values else None

    def summarize(items):
        return {
            "count": len(items),
            "strict_inside_ratio_visible_median": median(
                [x.get("strict_inside_ratio_visible") for x in items]
            ),
            "tolerant_inside_ratio_visible_median": median(
                [x.get("tolerant_inside_ratio_visible") for x in items]
            ),
            "tolerant_inside_ratio_visible_p10": percentile(
                [x.get("tolerant_inside_ratio_visible") for x in items], 10
            ),
            "tolerant_inside_ratio_visible_p90": percentile(
                [x.get("tolerant_inside_ratio_visible") for x in items], 90
            ),
        }

    detector_side_better = 0
    alternate_better = 0
    paired_hypotheses = 0
    for record in frame_records:
        original = predictions[record["sync_index"]]
        det_by_index = {det["detection_index"]: det for det in _unique_detections(original)}
        for det_index, sam in record["sam"].items():
            det = det_by_index[int(det_index)]
            hypotheses = sam.get("hypotheses", [])
            by_side = {
                int(item["is_right"]): item
                for item in hypotheses if item.get("is_right") is not None
            }
            detector_side = det.get("detector_is_right")
            if len(by_side) >= 2 and detector_side in by_side:
                paired_hypotheses += 1
                detector_score = by_side[int(detector_side)].get("tolerant_inside_ratio_visible")
                alternate_score = by_side[1 - int(detector_side)].get("tolerant_inside_ratio_visible")
                if detector_score is not None and alternate_score is not None:
                    if detector_score >= alternate_score:
                        detector_side_better += 1
                    else:
                        alternate_better += 1

    summary = {
        "schema_version": 1,
        "stage": "sam3_glove_assist_pilot_cam3",
        "source": {
            "video": str(args.video),
            "predictions": str(args.predictions),
            "fusion": str(args.fusion),
            "dataset": str(args.dataset),
            "model": str(args.model),
            "prompt": args.prompt,
            "prompt_mode": "yolo_box_plus_text" if args.geometric_prompts else "text_only",
        },
        "sampling": {
            "requested_frames": int(args.num_frames),
            "processed_frames": processed,
            "first_sync_index": selected_sync[0],
            "last_sync_index": selected_sync[-1],
            "mask_margin_px": int(args.mask_margin_px),
        },
        "sam": {
            "wilor_detections": total_detections,
            "matched_detections": matched_detections,
            "match_rate": float(matched_detections / total_detections) if total_detections else 0.0,
            "mask_quality": {
                "count": len(all_mask_metrics),
                "score_median": median([x["score"] for x in all_mask_metrics]),
                "mask_box_iou_yolo_median": median([x["mask_box_iou_yolo"] for x in all_mask_metrics]),
                "mask_area_ratio_yolo_box_median": median([x["mask_area_ratio_yolo_box"] for x in all_mask_metrics]),
                "border_touch_rate": float(
                    np.mean([x["mask_touches_image_border"] for x in all_mask_metrics])
                ) if all_mask_metrics else None,
            },
            "hypothesis_support": summarize(all_hypothesis_metrics),
            "paired_hypotheses": paired_hypotheses,
            "detector_side_at_least_as_supported": detector_side_better,
            "alternate_side_more_supported": alternate_better,
        },
        "fused_projection_support": {
            "raw_cam3": summarize(fused_raw_metrics),
            "refined_fused_projected_to_cam3": summarize(fused_projected_metrics),
            "paired_hands": min(len(fused_raw_metrics), len(fused_projected_metrics)),
        },
        "outputs": {
            "metrics_jsonl": str(args.output / "metrics.jsonl"),
            "preview_video": str(args.output / "sam3_preview_cam3.mp4"),
            "samples": str(sample_dir),
        },
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
