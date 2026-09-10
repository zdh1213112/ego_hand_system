#!/usr/bin/env python3
"""Attach conservative SAM3 glove evidence to saved WiLoR predictions.

This stage is deliberately separate from WiLoR inference.  SAM3 does not
predict joints; it segments the protective glove and writes a small
``sam3_assist`` record next to every WiLoR handedness hypothesis.  The guided
multiview fusion can then use that record as a soft view/hypothesis gate.

The source predictions are never modified.  A new prediction tree is written
so that the experiment can be compared with the original WiLoR run.
The SAM3 model and its Ultralytics dependencies live in the external
``auto-obb-annotator`` environment; ``--tool-root`` is added at run time.
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
DEFAULT_PROMPT = "hand wearing a gray and black protective work glove"
HAND_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cameras", nargs="+")
    parser.add_argument("--tool-root", type=Path, default=DEFAULT_TOOL_ROOT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument(
        "--text-only", action="store_true",
        help="use the text-only SAM3 query instead of YOLO-box + text prompts",
    )
    parser.add_argument(
        "--mask-margin-px", type=int, default=8,
        help="dilation margin for tolerant joint-in-mask support",
    )
    parser.add_argument(
        "--refine-2d", type=int, choices=(0, 1), default=1,
        help=(
            "pull non-wrist WiLoR joints a small distance toward the matched "
            "SAM3 glove mask; raw joints are retained in wilor_joints_2d"
        ),
    )
    parser.add_argument(
        "--max-2d-correction-px", type=float, default=24.0,
        help="maximum mask-guided correction distance for one 2D joint",
    )
    parser.add_argument(
        "--2d-correction-blend", dest="two_d_correction_blend",
        type=float, default=0.75,
        help="fraction of the safe mask-boundary correction to apply",
    )
    parser.add_argument(
        "--max-2d-bone-change-ratio", type=float, default=0.55,
        help=(
            "reject a mask correction if any adjacent 2D bone changes by "
            "more than this relative ratio"
        ),
    )
    parser.add_argument(
        "--min-score", type=float, default=0.25,
        help="minimum SAM3 confidence for a positive assist gate",
    )
    parser.add_argument(
        "--min-box-iou", type=float, default=0.20,
        help="minimum SAM3/YOLO box IoU for a positive assist gate",
    )
    parser.add_argument(
        "--min-joint-support", type=float, default=0.55,
        help="minimum tolerant fraction of visible WiLoR joints inside mask",
    )
    parser.add_argument(
        "--max-frames", type=int, default=0,
        help="process only the first N synchronized frames; 0 means all",
    )
    parser.add_argument("--progress-interval", type=int, default=25)
    return parser.parse_args()


def _load_jsonl(path: Path, max_frames: int = 0) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    return rows[: max_frames or None]


def _load_manifest(dataset: Path) -> tuple[dict[str, Any], list[dict[str, str]]]:
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    with (dataset / "multiview_frames.csv").open(
        "r", encoding="utf-8", newline=""
    ) as stream:
        rows = list(csv.DictReader(stream))
    return manifest, rows


def _unique_detections(record: dict[str, Any]) -> list[dict[str, Any]]:
    detections: dict[int, dict[str, Any]] = {}
    for hand in record.get("hands", []):
        try:
            detection_index = int(hand["detection_index"])
        except (KeyError, TypeError, ValueError):
            continue
        item = detections.setdefault(
            detection_index,
            {
                "detection_index": detection_index,
                "bbox_xyxy": hand.get("bbox_xyxy", []),
                "confidence": float(hand.get("confidence", 0.0)),
                "detector_is_right": hand.get("detector_is_right"),
                "hypotheses": [],
            },
        )
        item["hypotheses"].append(hand)
    return [detections[key] for key in sorted(detections)]


def _box_iou(a: Any, b: Any) -> float:
    if len(a) != 4 or len(b) != 4:
        return 0.0
    ax1, ay1, ax2, ay2 = (float(value) for value in a)
    bx1, by1, bx2, by2 = (float(value) for value in b)
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


def _resize_mask(mask: Any, image_shape: tuple[int, ...]) -> np.ndarray:
    image_height, image_width = image_shape[:2]
    value = np.asarray(mask, dtype=np.uint8)
    value = np.squeeze(value)
    if value.ndim != 2:
        raise ValueError(f"SAM3 mask is not two-dimensional: {value.shape}")
    if value.shape != (image_height, image_width):
        value = cv2.resize(
            value, (image_width, image_height), interpolation=cv2.INTER_NEAREST
        )
    return value > 0


def _joint_support(
    mask: np.ndarray, joints: Any, margin_px: int,
) -> dict[str, float | int | None]:
    points = np.asarray(joints, dtype=np.float64)
    if points.shape != (21, 2):
        return {
            "visible_joint_count": 0,
            "strict_inside_count": 0,
            "tolerant_inside_count": 0,
            "strict_inside_ratio_visible": None,
            "tolerant_inside_ratio_visible": None,
            "visible_flags": [],
            "strict_inside_flags": [],
            "tolerant_inside_flags": [],
            "hard_outlier_flags": [],
        }
    height, width = mask.shape[:2]
    visible = np.isfinite(points).all(axis=1)
    visible &= points[:, 0] >= 0.0
    visible &= points[:, 0] < width
    visible &= points[:, 1] >= 0.0
    visible &= points[:, 1] < height
    strict = np.zeros(21, dtype=bool)
    tolerant = np.zeros(21, dtype=bool)
    ys = np.clip(np.rint(points[visible, 1]), 0, height - 1).astype(np.int32)
    xs = np.clip(np.rint(points[visible, 0]), 0, width - 1).astype(np.int32)
    mask_u8 = mask.astype(np.uint8)
    if margin_px > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * margin_px + 1, 2 * margin_px + 1)
        )
        tolerant_mask = cv2.dilate(mask_u8, kernel, iterations=1)
    else:
        tolerant_mask = mask_u8
    strict[visible] = mask_u8[ys, xs] > 0
    tolerant[visible] = tolerant_mask[ys, xs] > 0
    count = int(visible.sum())
    return {
        "visible_joint_count": count,
        "strict_inside_count": int(strict.sum()),
        "tolerant_inside_count": int(tolerant.sum()),
        "strict_inside_ratio_visible": (
            float(strict.sum() / count) if count else None
        ),
        "tolerant_inside_ratio_visible": (
            float(tolerant.sum() / count) if count else None
        ),
        "visible_flags": visible.tolist(),
        "strict_inside_flags": strict.tolist(),
        "tolerant_inside_flags": tolerant.tolist(),
        # A point outside the dilated mask is the only per-joint condition
        # strong enough to gate a ray.  Invisible/NaN points remain neutral.
        "hard_outlier_flags": [
            bool(is_visible and not is_inside)
            for is_visible, is_inside in zip(visible, tolerant)
        ],
    }


def _nearest_mask_point(
    mask: np.ndarray, point: np.ndarray, max_distance_px: float,
) -> tuple[np.ndarray | None, float]:
    """Find a nearby foreground pixel without a full-image distance map.

    There are only 21 joints per hypothesis.  A small local search is both
    deterministic and considerably cheaper than constructing a distance map
    for every hand.  The returned point is on the observed glove boundary or
    inside the mask, never an extrapolated point.
    """
    if max_distance_px <= 0.0 or not np.isfinite(point).all():
        return None, float("inf")
    height, width = mask.shape[:2]
    x, y = float(point[0]), float(point[1])
    if x < 0.0 or x >= width or y < 0.0 or y >= height:
        return None, float("inf")
    radius = int(np.ceil(max_distance_px))
    x0 = max(0, int(np.floor(x)) - radius)
    y0 = max(0, int(np.floor(y)) - radius)
    x1 = min(width, int(np.floor(x)) + radius + 1)
    y1 = min(height, int(np.floor(y)) + radius + 1)
    local = np.asarray(mask[y0:y1, x0:x1], dtype=np.uint8)
    foreground = cv2.findNonZero(local)
    if foreground is None:
        return None, float("inf")
    candidates = np.asarray(foreground, dtype=np.float64).reshape(-1, 2)
    candidates[:, 0] += x0
    candidates[:, 1] += y0
    distances = np.linalg.norm(candidates - np.asarray(point)[None, :], axis=1)
    index = int(np.argmin(distances))
    distance = float(distances[index])
    if distance > max_distance_px:
        return None, distance
    return candidates[index], distance


def _mask_guided_2d_refinement(
    mask: np.ndarray,
    joints: Any,
    margin_px: int,
    max_correction_px: float,
    blend: float,
    max_bone_change_ratio: float = 0.55,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Conservatively correct only non-wrist points outside a glove mask.

    A SAM3 glove mask does not necessarily contain the wrist/cuff or every
    skin-facing fingertip.  Consequently joint 0 is protected and points are
    changed only when a foreground pixel is close enough to provide a real
    geometric target.  This is a 2D assist, not a replacement for WiLoR.
    """
    points = np.asarray(joints, dtype=np.float64)
    output = points.copy()
    flags = [False] * 21
    rejected_flags = [False] * 21
    shifts = np.zeros(21, dtype=np.float64)
    if points.shape != (21, 2) or max_correction_px <= 0.0 or blend <= 0.0:
        return output, {
            "enabled": False,
            "corrected_joint_count": 0,
            "corrected_joint_flags": flags,
            "rejected_joint_flags": rejected_flags,
            "median_shift_px": 0.0,
            "max_shift_px": 0.0,
        }
    height, width = mask.shape[:2]
    mask_u8 = np.asarray(mask, dtype=np.uint8)
    kernel_size = 2 * max(0, int(margin_px)) + 1
    if kernel_size > 1:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )
        tolerant = cv2.dilate(mask_u8, kernel, iterations=1)
    else:
        tolerant = mask_u8
    # The wrist is often outside a glove-only segmentation (cuff/skin seam).
    # Leave it to WiLoR and the calibrated multiview vote.
    proposals: dict[int, np.ndarray] = {}
    for joint in range(1, 21):
        point = points[joint]
        if not np.isfinite(point).all():
            continue
        x = int(np.rint(point[0]))
        y = int(np.rint(point[1]))
        if x < 0 or x >= width or y < 0 or y >= height:
            continue
        if tolerant[y, x] > 0:
            continue
        target, distance = _nearest_mask_point(
            mask_u8 > 0, point, max_correction_px
        )
        if target is None:
            continue
        alpha = float(np.clip(blend, 0.0, 1.0))
        proposals[joint] = point + alpha * (target - point)

    # Validate all proposals against the original local topology.  Revert a
    # proposal if it would stretch/compress an adjacent bone excessively;
    # this prevents an isolated segmentation fragment from bending a finger
    # even though the point itself lies closer to the mask.
    trial = output.copy()
    for joint, proposal in proposals.items():
        trial[joint] = proposal
    edge_by_joint: dict[int, list[tuple[int, int]]] = {
        joint: [] for joint in range(21)
    }
    for start, end in HAND_EDGES:
        edge_by_joint[start].append((start, end))
        edge_by_joint[end].append((start, end))
    for joint in list(proposals):
        valid = True
        for start, end in edge_by_joint[joint]:
            raw_start, raw_end = points[start], points[end]
            trial_start, trial_end = trial[start], trial[end]
            if not (
                np.isfinite(raw_start).all()
                and np.isfinite(raw_end).all()
                and np.isfinite(trial_start).all()
                and np.isfinite(trial_end).all()
            ):
                continue
            raw_length = float(np.linalg.norm(raw_start - raw_end))
            trial_length = float(np.linalg.norm(trial_start - trial_end))
            if raw_length <= 2.0:
                continue
            relative_change = abs(trial_length - raw_length) / raw_length
            if relative_change > max(0.0, max_bone_change_ratio):
                valid = False
                break
        if not valid:
            trial[joint] = output[joint]
            rejected_flags[joint] = True
        else:
            output[joint] = trial[joint]
            shifts[joint] = float(np.linalg.norm(output[joint] - points[joint]))
            flags[joint] = bool(shifts[joint] > 1e-6)
    changed = np.flatnonzero(np.asarray(flags, dtype=bool))
    return output, {
        "enabled": True,
        "corrected_joint_count": int(len(changed)),
        "corrected_joint_flags": flags,
        "rejected_joint_flags": rejected_flags,
        "median_shift_px": (
            float(np.median(shifts[changed])) if len(changed) else 0.0
        ),
        "max_shift_px": float(np.max(shifts[changed])) if len(changed) else 0.0,
    }


def _match_masks_to_detections(
    sam_boxes: list[list[float]], detections: list[dict[str, Any]],
) -> dict[int, int]:
    """Greedy one-to-one association while keeping YOLO identity authoritative."""
    candidates = []
    for sam_index, sam_box in enumerate(sam_boxes):
        center_x = (float(sam_box[0]) + float(sam_box[2])) * 0.5
        center_y = (float(sam_box[1]) + float(sam_box[3])) * 0.5
        for det_index, detection in enumerate(detections):
            box = detection.get("bbox_xyxy", [])
            if len(box) != 4:
                continue
            center_inside = (
                float(box[0]) <= center_x <= float(box[2])
                and float(box[1]) <= center_y <= float(box[3])
            )
            iou = _box_iou(sam_box, box)
            if center_inside or iou >= 0.10:
                candidates.append(
                    (0.80 * iou + 0.20 * float(center_inside), iou, sam_index, det_index)
                )
    candidates.sort(reverse=True)
    result: dict[int, int] = {}
    used_masks: set[int] = set()
    for _, _, sam_index, det_index in candidates:
        if det_index in result or sam_index in used_masks:
            continue
        result[det_index] = sam_index
        used_masks.add(sam_index)
    return result


def _detect_with_sam3(
    segmenter: Any,
    image: np.ndarray,
    prompt: str,
    detections: list[dict[str, Any]],
    device: str,
    conf: float,
    text_only: bool,
) -> tuple[list[np.ndarray], list[list[float]], list[float]]:
    if text_only:
        masks, boxes, scores = segmenter.detect(image, prompt)
        return [np.asarray(mask) for mask in masks], boxes, scores
    boxes = [detection.get("bbox_xyxy", []) for detection in detections]
    boxes = [box for box in boxes if len(box) == 4]
    if not boxes:
        return [], [], []
    # Box + text is more reliable for this egocentric sequence than a global
    # text-only search, while SAM3 still supplies the object mask itself.
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
    masks = []
    if result.masks is not None:
        masks = [mask for mask in result.masks.data.detach().cpu().numpy()]
    output_boxes = []
    if result.boxes is not None and len(result.boxes):
        output_boxes = result.boxes.xyxy.detach().cpu().numpy().tolist()
        scores = (
            result.boxes.conf.detach().cpu().numpy().tolist()
            if result.boxes.conf is not None else [1.0] * len(output_boxes)
        )
    else:
        scores = []
    return masks, output_boxes, scores


def _base_assist(enabled: bool = True) -> dict[str, Any]:
    return {
        "enabled": bool(enabled),
        "matched": False,
        "accepted": False,
        "reason": "no_mask",
        "mask_score": 0.0,
        "mask_box_iou_yolo": 0.0,
        "mask_area_ratio_yolo_box": 0.0,
        "strict_inside_ratio_visible": None,
        "tolerant_inside_ratio_visible": None,
        "visible_joint_count": 0,
        "visible_flags": [],
        "strict_inside_flags": [],
        "tolerant_inside_flags": [],
        "hard_outlier_flags": [],
        "two_d_correction": {
            "enabled": False,
            "corrected_joint_count": 0,
            "corrected_joint_flags": [],
            "rejected_joint_flags": [],
            "median_shift_px": 0.0,
            "max_shift_px": 0.0,
        },
    }


def _annotate_record(
    record: dict[str, Any],
    masks: list[np.ndarray],
    sam_boxes: list[list[float]],
    sam_scores: list[float],
    image_shape: tuple[int, ...],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, int]]:
    detections = _unique_detections(record)
    association = _match_masks_to_detections(sam_boxes, detections)
    by_detection: dict[int, dict[str, Any]] = {}
    stats = {
        "detections": len(detections),
        "matched_detections": 0,
        "accepted_hypotheses": 0,
        "rejected_hypotheses": 0,
        "corrected_hypotheses": 0,
        "corrected_joints": 0,
    }
    for det_index, detection in enumerate(detections):
        mask_index = association.get(det_index)
        if mask_index is None or mask_index >= len(masks):
            by_detection[int(detection["detection_index"])] = {
                "base": _base_assist(), "hypotheses": {}
            }
            continue
        mask = _resize_mask(masks[mask_index], image_shape)
        mask_box = _mask_box(mask)
        yolo_box = detection.get("bbox_xyxy", [])
        x1, y1, x2, y2 = [float(value) for value in yolo_box]
        height, width = image_shape[:2]
        x1i = max(0, min(width, int(np.floor(x1))))
        y1i = max(0, min(height, int(np.floor(y1))))
        x2i = max(0, min(width, int(np.ceil(x2))))
        y2i = max(0, min(height, int(np.ceil(y2))))
        box_area = max(1, x2i - x1i) * max(1, y2i - y1i)
        mask_inside = int(mask[y1i:y2i, x1i:x2i].sum()) if x2i > x1i and y2i > y1i else 0
        score = float(sam_scores[mask_index]) if mask_index < len(sam_scores) else 1.0
        iou = _box_iou(mask_box, yolo_box) if mask_box else 0.0
        common = _base_assist()
        common.update({
            "matched": True,
            "mask_score": score,
            "sam_box": [float(value) for value in sam_boxes[mask_index]],
            "mask_box": mask_box,
            "mask_box_iou_yolo": float(iou),
            "mask_area_ratio_yolo_box": float(mask_inside / box_area),
        })
        hypotheses: dict[int, dict[str, Any]] = {}
        for hypothesis in detection["hypotheses"]:
            try:
                side = int(hypothesis.get("is_right"))
            except (TypeError, ValueError):
                continue
            support = _joint_support(
                mask, hypothesis.get("joints_2d", []), args.mask_margin_px
            )
            corrected_points, correction = _mask_guided_2d_refinement(
                mask,
                hypothesis.get("joints_2d", []),
                args.mask_margin_px,
                args.max_2d_correction_px,
                args.two_d_correction_blend,
                args.max_2d_bone_change_ratio,
            ) if args.refine_2d else (
                np.asarray(hypothesis.get("joints_2d", []), dtype=np.float64),
                {
                    "enabled": False,
                    "corrected_joint_count": 0,
                    "corrected_joint_flags": [False] * 21,
                    "rejected_joint_flags": [False] * 21,
                    "median_shift_px": 0.0,
                    "max_shift_px": 0.0,
                },
            )
            tolerant = support["tolerant_inside_ratio_visible"]
            accepted = (
                score >= args.min_score
                and iou >= args.min_box_iou
                and tolerant is not None
                and float(tolerant) >= args.min_joint_support
            )
            info = dict(common)
            info.update(support)
            if correction.get("corrected_joint_count", 0):
                correction = dict(correction)
                correction["corrected_joints_2d"] = corrected_points.tolist()
            info["two_d_correction"] = correction
            info["accepted"] = bool(accepted)
            info["reason"] = "accepted" if accepted else (
                "low_joint_support" if tolerant is not None and float(tolerant) < args.min_joint_support
                else "low_mask_quality"
            )
            hypotheses[side] = info
        by_detection[int(detection["detection_index"])] = {
            "base": common, "hypotheses": hypotheses
        }
        stats["matched_detections"] += 1

    output = dict(record)
    hands = []
    for hand in record.get("hands", []):
        copied = dict(hand)
        detection_index = int(hand.get("detection_index", -1))
        entry = by_detection.get(detection_index)
        if entry is None:
            copied["sam3_assist"] = _base_assist()
        else:
            try:
                side = int(hand.get("is_right"))
            except (TypeError, ValueError):
                side = -1
            copied["sam3_assist"] = dict(
                entry["hypotheses"].get(side, entry["base"])
            )
        # Keep a stable raw-WiLoR channel for A/B visualization.  Downstream
        # fusion consumes joints_2d, which may contain the conservative SAM3
        # correction; wilor_joints_2d always remains the pre-SAM3 points.
        if "wilor_joints_2d" not in copied:
            copied["wilor_joints_2d"] = copied.get("joints_2d", [])
        correction = copied["sam3_assist"].get("two_d_correction", {})
        corrected_count = int(correction.get("corrected_joint_count", 0))
        if corrected_count and np.asarray(copied.get("joints_2d", [])).shape == (21, 2):
            # Recompute the corrected points from the persisted target only
            # when the current hypothesis was annotated with a mask.  The
            # actual array is attached below by _annotate_record.
            corrected = correction.get("corrected_joints_2d")
            if corrected is not None:
                copied["joints_2d"] = corrected
        stats["corrected_hypotheses"] += int(corrected_count > 0)
        stats["corrected_joints"] += corrected_count
        stats["accepted_hypotheses"] += int(copied["sam3_assist"].get("accepted", False))
        stats["rejected_hypotheses"] += int(not copied["sam3_assist"].get("accepted", False))
        hands.append(copied)
    output["hands"] = hands
    output["sam3"] = {
        "enabled": True,
        "matched_detection_count": stats["matched_detections"],
        "detection_count": stats["detections"],
        "corrected_hypothesis_count": stats["corrected_hypotheses"],
        "corrected_joint_count": stats["corrected_joints"],
    }
    return output, stats


def main() -> int:
    args = parse_args()
    if args.max_frames < 0 or args.progress_interval < 1:
        raise ValueError("max-frames must be non-negative; progress interval must be positive")
    if not 0.0 < args.conf <= 1.0:
        raise ValueError("--conf must be in (0, 1]")
    if not 0.0 <= args.min_score <= 1.0:
        raise ValueError("--min-score must be in [0, 1]")
    if not 0.0 <= args.min_box_iou <= 1.0:
        raise ValueError("--min-box-iou must be in [0, 1]")
    if not 0.0 <= args.min_joint_support <= 1.0:
        raise ValueError("--min-joint-support must be in [0, 1]")
    if args.mask_margin_px < 0:
        raise ValueError("--mask-margin-px must be non-negative")
    if args.max_2d_correction_px < 0.0:
        raise ValueError("--max-2d-correction-px must be non-negative")
    if not 0.0 <= args.two_d_correction_blend <= 1.0:
        raise ValueError("--2d-correction-blend must be in [0, 1]")
    if args.max_2d_bone_change_ratio < 0.0:
        raise ValueError("--max-2d-bone-change-ratio must be non-negative")
    dataset = args.dataset.resolve()
    source_root = args.predictions.resolve()
    output = args.output.resolve()
    for path in (dataset / "manifest.json", dataset / "multiview_frames.csv", args.model.resolve()):
        if not path.exists():
            raise FileNotFoundError(path)
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    output.mkdir(parents=True)
    manifest, sync_rows = _load_manifest(dataset)
    available_cameras = tuple(manifest["camera_ids"])
    cameras = tuple(dict.fromkeys(args.cameras or available_cameras))
    unknown = [camera for camera in cameras if camera not in available_cameras]
    if unknown:
        raise ValueError(f"selected cameras are not in the dataset: {unknown}")
    sync_rows = sync_rows[: args.max_frames or None]
    if not sync_rows:
        raise ValueError("normalized dataset contains no selected frames")

    sys.path.insert(0, str(args.tool_root.resolve()))
    from auto_obb_annotator.app_protective_gloves import (  # noqa: E402
        ProtectiveGloveSAM3TextSegmenter,
    )

    print(f"loading SAM3: {args.model}", flush=True)
    started = time.perf_counter()
    segmenter = ProtectiveGloveSAM3TextSegmenter(
        model_path=str(args.model), device=args.device, conf=args.conf,
    )
    print(f"SAM3 loaded in {time.perf_counter() - started:.1f}s", flush=True)
    image_size = tuple(manifest["image_size"])
    summary_by_camera: dict[str, dict[str, Any]] = {}
    total_started = time.perf_counter()
    for camera in cameras:
        source_path = source_root / camera / "predictions.jsonl"
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        rows = _load_jsonl(source_path, args.max_frames)
        if len(rows) != len(sync_rows):
            # Predictions should normally have one row per normalized frame.
            # Match by sync index so a partially saved source remains safe.
            by_sync = {int(row["sync_index"]): row for row in rows}
            rows = [by_sync[int(row["sync_index"])] for row in sync_rows if int(row["sync_index"]) in by_sync]
        reader = SequentialVideoReader(
            dataset / "cameras" / camera / manifest["storage"]["video_filename"],
            image_size,
        )
        camera_output = output / camera
        camera_output.mkdir()
        camera_started = time.perf_counter()
        counters = {
            "frame_count": 0,
            "detections": 0,
            "matched_detections": 0,
            "accepted_hypotheses": 0,
            "rejected_hypotheses": 0,
            "corrected_hypotheses": 0,
            "corrected_joints": 0,
        }
        try:
            with (camera_output / "predictions.jsonl").open("w", encoding="utf-8") as stream:
                for ordinal, row in enumerate(rows, start=1):
                    frame = reader.read(int(row["source_frame_index"]))
                    masks, boxes, scores = _detect_with_sam3(
                        segmenter, frame, args.prompt, _unique_detections(row),
                        args.device, args.conf, args.text_only,
                    )
                    annotated, stats = _annotate_record(
                        row, masks, boxes, scores, frame.shape, args,
                    )
                    stream.write(json.dumps(annotated, separators=(",", ":")) + "\n")
                    counters["frame_count"] += 1
                    for key in counters:
                        if key != "frame_count":
                            counters[key] += int(stats.get(key, 0))
                    if ordinal % args.progress_interval == 0 or ordinal == len(rows):
                        elapsed = time.perf_counter() - camera_started
                        print(
                            f"{camera}: {ordinal}/{len(rows)} | "
                            f"SAM matched {counters['matched_detections']}/"
                            f"{counters['detections']} | "
                            f"{ordinal / max(elapsed, 1e-9):.2f} frames/s",
                            flush=True,
                        )
        finally:
            reader.close()
        elapsed = time.perf_counter() - camera_started
        camera_summary = {
            **counters,
            "match_rate": counters["matched_detections"] / max(counters["detections"], 1),
            "accepted_hypothesis_rate": counters["accepted_hypotheses"] / max(
                counters["accepted_hypotheses"] + counters["rejected_hypotheses"], 1
            ),
            "elapsed_seconds": elapsed,
            "frames_per_second": counters["frame_count"] / max(elapsed, 1e-9),
        }
        (camera_output / "summary.json").write_text(
            json.dumps(camera_summary, indent=2) + "\n", encoding="utf-8"
        )
        summary_by_camera[camera] = camera_summary

    total_frames = sum(item["frame_count"] for item in summary_by_camera.values())
    total_detections = sum(item["detections"] for item in summary_by_camera.values())
    total_matched = sum(item["matched_detections"] for item in summary_by_camera.values())
    total_accepted = sum(item["accepted_hypotheses"] for item in summary_by_camera.values())
    total_rejected = sum(item["rejected_hypotheses"] for item in summary_by_camera.values())
    total_corrected_hypotheses = sum(
        item["corrected_hypotheses"] for item in summary_by_camera.values()
    )
    total_corrected_joints = sum(
        item["corrected_joints"] for item in summary_by_camera.values()
    )
    summary = {
        "schema_version": 1,
        "stage": "wilor_sam3_glove_mask_assist",
        "dataset": str(dataset),
        "source_predictions": str(source_root),
        "camera_ids": list(cameras),
        "parameters": {
            "model": str(args.model.resolve()),
            "prompt": args.prompt,
            "prompt_mode": "text_only" if args.text_only else "yolo_box_plus_text",
            "device": args.device,
            "conf": args.conf,
            "mask_margin_px": args.mask_margin_px,
            "min_score": args.min_score,
            "min_box_iou": args.min_box_iou,
            "min_joint_support": args.min_joint_support,
            "refine_2d": bool(args.refine_2d),
            "max_2d_correction_px": args.max_2d_correction_px,
            "two_d_correction_blend": args.two_d_correction_blend,
            "max_2d_bone_change_ratio": args.max_2d_bone_change_ratio,
        },
        "frame_count": len(sync_rows),
        "processed_camera_frames": total_frames,
        "detections": total_detections,
        "matched_detections": total_matched,
        "match_rate": total_matched / max(total_detections, 1),
        "accepted_hypotheses": total_accepted,
        "rejected_hypotheses": total_rejected,
        "accepted_hypothesis_rate": total_accepted / max(total_accepted + total_rejected, 1),
        "corrected_hypotheses": total_corrected_hypotheses,
        "corrected_joints": total_corrected_joints,
        "elapsed_seconds": time.perf_counter() - total_started,
        "cameras": summary_by_camera,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
