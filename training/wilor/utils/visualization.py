"""Reusable OpenCV drawing helpers for full-frame hand visualizations."""

from __future__ import annotations

from typing import Any, Iterable

import cv2
import numpy as np


# WiLoR/MANO order: wrist, then thumb/index/middle/ring/little fingers.
HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
)

LEFT_COLOR = (154, 165, 53)       # BGR
RIGHT_COLOR = (60, 199, 255)      # BGR
GT_COLOR = (40, 220, 40)          # BGR
PRED_COLOR = (30, 30, 235)        # BGR


def _points(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    points = np.asarray(value, dtype=np.float32)
    if points.shape != (21, 2) or not np.isfinite(points).all():
        return None
    return np.rint(points).astype(np.int32)


def _color(value: Any, default: tuple[int, int, int]) -> tuple[int, int, int]:
    if value is None:
        return default
    color = tuple(int(channel) for channel in value)
    if len(color) != 3:
        raise ValueError(f"A BGR color must have three channels, got {value!r}")
    return color


def _draw_skeleton(canvas: np.ndarray, points: np.ndarray,
                   color: tuple[int, int, int]) -> None:
    for start, end in HAND_CONNECTIONS:
        cv2.line(canvas, tuple(points[start]), tuple(points[end]), color, 2, cv2.LINE_AA)
    for point in points:
        cv2.circle(canvas, tuple(point), 5, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(canvas, tuple(point), 3, color, -1, cv2.LINE_AA)


def draw_hand_overlay(
    image: np.ndarray,
    hands: Iterable[dict[str, Any]],
    footer: str | None = None,
    show_legend: bool = True,
) -> np.ndarray:
    """Draw any number of hand results on one BGR frame.

    Each item may contain ``bbox`` (or ``bbox_xyxy``), ``gt_joints_2d`` and/or
    ``pred_joints_2d``.  ``joints_2d`` is accepted as an alias for prediction
    points so the same helper can be used by normal inference and evaluation.
    ``is_right`` selects the bbox color; ``label`` is rendered above the box.
    Per-hand ``bbox_color``, ``gt_color`` and ``pred_color`` values can override
    the defaults without duplicating any OpenCV drawing code at the call site.
    """
    canvas = image.copy()
    has_gt = False
    has_pred = False
    has_left = False
    has_right = False

    for hand in hands:
        is_right = bool(hand.get("is_right", hand.get("side", 1)))
        has_right |= is_right
        has_left |= not is_right
        box_color = _color(
            hand.get("bbox_color"), RIGHT_COLOR if is_right else LEFT_COLOR)
        bbox = hand.get("bbox", hand.get("bbox_xyxy"))
        if bbox is not None:
            box = np.rint(np.asarray(bbox, dtype=np.float32)).astype(np.int32).reshape(4)
            cv2.rectangle(canvas, tuple(box[:2]), tuple(box[2:]), box_color, 3)
            label = hand.get("label")
            if label:
                text = str(label)
                (tw, th), baseline = cv2.getTextSize(
                    text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
                )
                text_top = max(0, int(box[1]) - th - baseline - 4)
                x1 = max(0, int(box[0]))
                cv2.rectangle(
                    canvas, (x1, text_top), (x1 + tw + 6, text_top + th + baseline + 4),
                    box_color, -1,
                )
                cv2.putText(
                    canvas, text, (x1 + 3, text_top + th + 1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA,
                )

        gt = _points(hand.get("gt_joints_2d"))
        pred = _points(hand.get("pred_joints_2d", hand.get("joints_2d")))
        if gt is not None:
            has_gt = True
            _draw_skeleton(canvas, gt, _color(hand.get("gt_color"), GT_COLOR))
        if pred is not None:
            has_pred = True
            _draw_skeleton(canvas, pred, _color(hand.get("pred_color"), PRED_COLOR))

    legend_items: list[tuple[str, tuple[int, int, int]]] = []
    if has_gt:
        legend_items.append(("GT joints", GT_COLOR))
    if has_pred:
        legend_items.append(("Pred joints", PRED_COLOR))
    if has_left:
        legend_items.append(("L bbox", LEFT_COLOR))
    if has_right:
        legend_items.append(("R bbox", RIGHT_COLOR))
    if show_legend and legend_items:
        x = 35
        width = max(220, 25 + sum(cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2
        )[0][0] + 35 for text, _ in legend_items))
        cv2.rectangle(canvas, (20, 20), (width, 76), (20, 20, 20), -1)
        for text, color in legend_items:
            cv2.putText(canvas, text, (x, 55), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, color, 2, cv2.LINE_AA)
            x += cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)[0][0] + 35

    if footer:
        cv2.putText(canvas, str(footer), (20, canvas.shape[0] - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    return canvas
