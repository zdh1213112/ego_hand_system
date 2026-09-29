#!/usr/bin/env python3
"""Fuse WiLoR with a reliable stereo anchor and independently matched side views."""

from __future__ import annotations

import argparse
from bisect import bisect_left
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import itertools
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time
from typing import Any

# Fusion launches several processes and all of its linear algebra matrices are
# tiny.  A BLAS thread pool inside every worker only oversubscribes the CPU.
for _thread_variable in (
    "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_thread_variable, "1")

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from camera_models import project_points  # noqa: E402
from ego_data.calibration import CameraCalibration  # noqa: E402
from fuse_multiview_wilor import (  # noqa: E402
    _groups, _load_jsonl, _serialize_hand,
    _triangulate_hand as _triangulate_hand_base,
)
from fusion_anatomy_refinement import (  # noqa: E402
    FusionAnatomyConfig,
    _apply_reprojection_guard,
    refine_accepted_rows,
)
from fusion_temporal_guard import guard_temporal_rows  # noqa: E402
from fusion_joint_temporal import (  # noqa: E402
    stabilize_hand_identities,
    stabilize_joint_temporal_rows,
)


HAND_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
)


def _triangulate_hand(
    side: int,
    selected: dict[str, dict[str, Any]],
    calibrations: dict[str, CameraCalibration],
    threshold_px: float,
    min_views: int,
    sam3_gate: bool = False,
    reject_ill_conditioned: bool = True,
    condition_hard_threshold: float = 1000.0,
) -> dict[str, Any]:
    """Guided fusion uses angularly weighted per-joint ray intersections."""
    result = _triangulate_hand_base(
        side, selected, calibrations, threshold_px, min_views, sam3_gate,
        ray_angle_weighting=True,
    )
    # A tiny ray angle can make a two-view intersection numerically finite but
    # physically meaningless (the point may be tens of metres away while all
    # image residuals still look small).  Do not pass that branch to the
    # temporal smoother: retain a missing joint and let the bidirectional
    # tracker recover it from neighbouring, well-conditioned observations.
    # The threshold is intentionally conservative and matches the diagnostic
    # ``condition_ge_1000`` bucket used in the reports.
    condition = np.asarray(result.get("ray_condition_numbers", []), dtype=float)
    points = np.asarray(result.get("points", []), dtype=float)
    # Keep the two-view anchor alive for camera association.  A pair with a
    # weak baseline cannot certify depth yet, but rejecting it during anchor
    # scoring would prevent the remaining cameras from being matched.  Final
    # fusion calls apply the stricter configured minimum-view policy.
    if reject_ill_conditioned and condition.shape == (21,) and points.shape == (21, 3):
        hard_threshold = float(condition_hard_threshold)
        invalid = ~np.isfinite(condition) | (condition >= hard_threshold)
        if np.any(invalid):
            points[invalid] = np.nan
            result["points"] = points
            result["valid_joints"] = int(np.count_nonzero(np.isfinite(points).all(axis=1)))
            result["condition_rejected_joint_count"] = int(np.count_nonzero(invalid))
        else:
            result["condition_rejected_joint_count"] = 0
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--cameras", nargs="+",
        help="camera subset used for fusion (default: all cameras in the dataset)",
    )
    parser.add_argument("--anchor-cameras", nargs=2, default=("camera2", "camera3"))
    parser.add_argument(
        "--preferred-anchor-camera", default=None,
        help="prefer any valid two-camera anchor pair containing this camera",
    )
    parser.add_argument("--max-anchor-detections", type=int, default=3)
    parser.add_argument("--max-side-detections", type=int, default=4)
    parser.add_argument(
        "--suppress-duplicate-detector-boxes", type=int, choices=(0, 1), default=1,
        help=(
            "suppress nearly identical detector boxes before anchor matching"
        ),
    )
    parser.add_argument(
        "--duplicate-detector-iou", type=float, default=0.95,
        help="IoU threshold for cross-class duplicate detector suppression",
    )
    parser.add_argument(
        "--duplicate-detector-containment", type=float, default=0.95,
        help=(
            "intersection/min-area threshold for suppressing nested "
            "cross-class boxes (merged-hand guard)"
        ),
    )
    parser.add_argument(
        "--detector-handedness", choices=("strict", "ignore", "adaptive"), default="strict",
        help=(
            "strict keeps detector.pt identity; ignore lets geometry choose; "
            "adaptive tries strict first and falls back to ignore for a rejected frame"
        ),
    )
    parser.add_argument(
        "--allow-single-hand-anchors", type=int, choices=(0, 1), default=0,
        help="permit a calibrated two-view anchor for one detected hand",
    )
    parser.add_argument("--association-threshold-px", type=float, default=55.0)
    parser.add_argument("--anchor-threshold-px", type=float, default=60.0)
    parser.add_argument("--ransac-threshold-px", type=float, default=20.0)
    parser.add_argument(
        "--condition-hard-threshold", type=float, default=1000.0,
        help=(
            "reject final 3D joints whose multiview ray condition number is "
            "at or above this value; lower values trade coverage for safer "
            "depth (default: 1000)"
        ),
    )
    parser.add_argument(
        "--min-joint-inlier-views", type=int, default=3,
        help=(
            "minimum RANSAC-supported cameras for a final 3D joint when "
            "three or more views are selected (default: 3)"
        ),
    )
    parser.add_argument(
        "--allow-stereo-final", type=int, choices=(0, 1), default=0,
        help=(
            "allow two-view stereo points in the final 3D output when fewer "
            "than --min-joint-inlier-views cameras survive; disabled by "
            "default because such depth branches can jump"
        ),
    )
    parser.add_argument("--min-valid-joints", type=int, default=12)
    parser.add_argument("--max-reprojection-median-px", type=float, default=15.0)
    parser.add_argument("--max-reprojection-p95-px", type=float, default=40.0)
    parser.add_argument(
        "--allow-partial-hands", type=int, choices=(0, 1), default=0,
        help=(
            "keep a frame when at least one physical hand passes the full "
            "multiview quality gate; failed hands remain absent (default: 0)"
        ),
    )
    parser.add_argument("--temporal-recovery-gap", type=int, default=3)
    parser.add_argument("--temporal-association-threshold-px", type=float, default=90.0)
    parser.add_argument("--max-temporal-wrist-step-m", type=float, default=0.12)
    parser.add_argument(
        "--robust-view-consensus", type=int, choices=(0, 1), default=1,
        help=(
            "reject a per-hand camera view when at least two other views "
            "strongly disagree with it"
        ),
    )
    parser.add_argument("--consensus-min-views", type=int, default=3)
    parser.add_argument("--consensus-min-inlier-joints", type=int, default=8)
    parser.add_argument("--consensus-outlier-median-px", type=float, default=35.0)
    parser.add_argument("--consensus-outlier-ratio", type=float, default=2.5)
    parser.add_argument(
        "--sam3-gate", type=int, choices=(0, 1), default=0,
        help=(
            "use per-view sam3_assist records as a conservative hypothesis/"
            "view gate (0 keeps the original fusion behavior)"
        ),
    )
    parser.add_argument("--sam3-min-score", type=float, default=0.25)
    parser.add_argument("--sam3-min-box-iou", type=float, default=0.20)
    parser.add_argument("--sam3-min-joint-support", type=float, default=0.55)
    parser.add_argument(
        "--sam3-hard-joint-support", type=float, default=0.35,
        help="support below this value can reject a view when other views agree",
    )
    parser.add_argument("--anatomy-refinement", type=int, choices=(0, 1), default=1)
    parser.add_argument("--anatomy-outlier-window", type=int, default=4)
    parser.add_argument("--anatomy-outlier-distance-m", type=float, default=0.10)
    parser.add_argument("--anatomy-bone-relative", type=float, default=0.45)
    parser.add_argument("--anatomy-smoothing-radius", type=int, default=2)
    parser.add_argument("--anatomy-shape-strength", type=float, default=0.55)
    parser.add_argument("--anatomy-reliable-adjustment-blend", type=float, default=0.35)
    parser.add_argument("--anatomy-max-reliable-adjustment-m", type=float, default=0.020)
    parser.add_argument("--anatomy-max-reprojection-regression-px", type=float, default=15.0)
    parser.add_argument("--anatomy-max-reprojection-shift-px", type=float, default=35.0)
    parser.add_argument("--anatomy-temporal-radius", type=int, default=4)
    parser.add_argument("--anatomy-temporal-palm-residual-m", type=float, default=0.070)
    parser.add_argument(
        "--anatomy-temporal-palm-strong-residual-m", type=float, default=0.140,
    )
    parser.add_argument(
        "--anatomy-temporal-global-min-rejected-joints", type=int, default=6,
    )
    parser.add_argument("--anatomy-temporal-local-residual", type=float, default=0.90)
    parser.add_argument("--anatomy-temporal-blend", type=float, default=0.50)
    parser.add_argument(
        "--anatomy-temporal-max-global-adjustment-m", type=float, default=0.10,
    )
    parser.add_argument(
        "--anatomy-temporal-max-local-adjustment-m", type=float, default=0.05,
    )
    parser.add_argument(
        "--anatomy-temporal-spike-joint-step-m", type=float, default=0.12,
    )
    parser.add_argument(
        "--anatomy-temporal-spike-palm-step-m", type=float, default=0.20,
    )
    parser.add_argument(
        "--anatomy-temporal-spike-max-spread-m", type=float, default=0.08,
    )
    parser.add_argument(
        "--anatomy-protected-cameras", nargs="*", default=(),
        help=(
            "camera views whose calibrated 2-D observations must remain "
            "consistent during anatomy/temporal correction"
        ),
    )
    parser.add_argument(
        "--anatomy-protected-camera-shift-px", type=float, default=35.0,
        help=(
            "maximum protected-camera pixel shift from the pre-refinement "
            "pose (default: 35 px)"
        ),
    )
    parser.add_argument(
        "--post-temporal-branch-threshold-m", type=float, default=0.030,
        help=(
            "3-D displacement above which the post-target guard treats a "
            "depth branch as an outlier when camera motion does not support "
            "it (default: 0.030 m)"
        ),
    )
    parser.add_argument(
        "--post-temporal-joint-repass", type=int, choices=(0, 1), default=1,
        help=(
            "run one additional bidirectional per-joint temporal pass after "
            "protected-camera reprojection (default: 1)"
        ),
    )
    parser.add_argument(
        "--temporal-3d-guard", type=int, choices=(0, 1), default=0,
        help=(
            "guard ambiguous 3D depth with bidirectional high-support anchors; "
            "2D detector/WiLoR observations are left unchanged. Disabled by "
            "default because clearing a whole hand creates visible dropouts; "
            "use the per-joint guard for ordinary branch rejection."
        ),
    )
    parser.add_argument("--temporal-3d-guard-radius", type=int, default=24)
    parser.add_argument("--temporal-3d-guard-threshold-m", type=float, default=0.020)
    parser.add_argument("--temporal-3d-guard-min-anchor-cameras", type=int, default=5)
    parser.add_argument("--temporal-3d-guard-min-palm-inliers", type=int, default=3)
    parser.add_argument("--temporal-3d-guard-max-anchor-quality-px", type=float, default=12.0)
    parser.add_argument("--temporal-3d-guard-max-anchor-speed-m", type=float, default=0.050)
    parser.add_argument("--temporal-3d-guard-duplicate-iou", type=float, default=0.90)
    parser.add_argument(
        "--temporal-3d-guard-max-reprojection-median-px", type=float, default=35.0,
    )
    parser.add_argument(
        "--temporal-3d-guard-max-reprojection-p95-px", type=float, default=100.0,
    )
    parser.add_argument(
        "--temporal-3d-guard-max-local-reprojection-ratio",
        type=float,
        default=1.5,
        help=(
            "allow a local temporal translation only when its median 2D "
            "residual is no more than this multiple of the current residual"
        ),
    )
    parser.add_argument(
        "--temporal-3d-guard-clear-endpoint", type=int, choices=(0, 1), default=1,
    )
    parser.add_argument(
        "--joint-temporal-guard", type=int, choices=(0, 1), default=1,
        help=(
            "stabilize each finite 3D joint with bidirectional ray/depth "
            "tracking; camera 2D observations are left unchanged"
        ),
    )
    parser.add_argument("--joint-temporal-ray-prior-weight", type=float, default=3.0)
    parser.add_argument("--joint-temporal-ray-blend", type=float, default=0.80)
    parser.add_argument("--joint-temporal-max-step-m", type=float, default=0.025)
    parser.add_argument("--joint-temporal-max-radius-m", type=float, default=0.55)
    parser.add_argument("--joint-temporal-max-missing-gap", type=int, default=2)
    parser.add_argument("--joint-temporal-local-radius", type=int, default=3)
    parser.add_argument(
        "--joint-temporal-local-outlier-threshold-m", type=float, default=0.025,
    )
    parser.add_argument("--joint-temporal-local-candidate-ratio", type=float, default=0.82)
    parser.add_argument("--joint-temporal-smoothing-sigma", type=float, default=1.30)
    parser.add_argument("--joint-temporal-bone-iterations", type=int, default=15)
    parser.add_argument("--joint-temporal-bone-strength", type=float, default=0.80)
    parser.add_argument(
        "--joint-temporal-min-hand-joints", type=int, default=18,
        help=(
            "clear a hand-level 3D output when fewer than this many joints "
            "are finite after ray gating (default: 18)"
        ),
    )
    parser.add_argument(
        "--joint-temporal-excessive-jump-threshold-m", type=float, default=0.030,
        help=(
            "clear an unresolved consecutive-joint branch above this motion "
            "instead of clipping it (default: 0.030 m)"
        ),
    )
    parser.add_argument(
        "--joint-temporal-view-wrist-jump-px", type=float, default=180.0,
        help=(
            "invalidate a hand when a common camera's corrected 2D wrist "
            "jumps this far in one frame (default: 180 px)"
        ),
    )
    parser.add_argument("--joint-temporal-view-expanded-box-ratio", type=float, default=1.50)
    parser.add_argument("--joint-temporal-view-expanded-box-wrist-px", type=float, default=130.0)
    parser.add_argument(
        "--temporal-hand-identity", type=int, choices=(0, 1), default=1,
        help=(
            "keep physical left/right identities across detector label swaps "
            "using multi-camera box and palm tracks"
        ),
    )
    parser.add_argument("--temporal-hand-identity-max-gap", type=int, default=60)
    parser.add_argument("--temporal-hand-identity-swap-margin", type=float, default=0.25)
    parser.add_argument("--temporal-hand-identity-swap-min-cost", type=float, default=0.90)
    parser.add_argument(
        "--workers", type=int, default=1,
        help="independent frame-fusion worker processes",
    )
    parser.add_argument("--progress-interval", type=int, default=25)
    parser.add_argument("--max-frames", type=int, default=0)
    return parser.parse_args()


def _bbox_iou(first: Any, second: Any) -> float:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    if first.shape != (4,) or second.shape != (4,):
        return 0.0
    left = max(float(first[0]), float(second[0]))
    top = max(float(first[1]), float(second[1]))
    right = min(float(first[2]), float(second[2]))
    bottom = min(float(first[3]), float(second[3]))
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, float(first[2] - first[0])) * max(
        0.0, float(first[3] - first[1])
    )
    second_area = max(0.0, float(second[2] - second[0])) * max(
        0.0, float(second[3] - second[1])
    )
    union = first_area + second_area - intersection
    return intersection / union if union > 1e-9 else 0.0


def _bbox_containment(first: Any, second: Any) -> float:
    """Intersection divided by the smaller box area.

    A detector box that accidentally contains both hands can have only
    moderate IoU with the correct, smaller box.  Containment exposes this
    nested duplicate while leaving two adjacent boxes untouched.
    """
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    if first.shape != (4,) or second.shape != (4,):
        return 0.0
    left = max(float(first[0]), float(second[0]))
    top = max(float(first[1]), float(second[1]))
    right = min(float(first[2]), float(second[2]))
    bottom = min(float(first[3]), float(second[3]))
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, float(first[2] - first[0])) * max(
        0.0, float(first[3] - first[1])
    )
    second_area = max(0.0, float(second[2] - second[0])) * max(
        0.0, float(second[3] - second[1])
    )
    smaller = min(first_area, second_area)
    return intersection / smaller if smaller > 1e-9 else 0.0


def _bbox_area(box: Any) -> float:
    value = np.asarray(box, dtype=np.float64)
    if value.shape != (4,) or not np.isfinite(value).all():
        return 0.0
    return max(float(value[2] - value[0]), 0.0) * max(
        float(value[3] - value[1]), 0.0
    )


def _complete_detections(
    groups: dict[int, dict[int, dict[str, Any]]], limit: int,
    suppress_duplicate_boxes: bool = False,
    duplicate_iou: float = 0.95,
    duplicate_containment: float = 0.95,
) -> list[int]:
    detections = [index for index, sides in groups.items() if 0 in sides and 1 in sides]
    detections.sort(
        key=lambda index: float(groups[index][0].get("confidence", 0.0)), reverse=True
    )
    if suppress_duplicate_boxes:
        # Ultralytics/top-k inference can retain two boxes at exactly the same
        # location, including when both boxes have the same handedness class.
        # Treat the higher-confidence box as the physical detection and discard
        # the duplicate.  A very high IoU/containment threshold keeps genuinely
        # adjacent hands intact while preventing one physical hand from being
        # assigned to both fused sides through duplicate detector slots.
        removed: set[int] = set()
        for first_index, first_detection in enumerate(detections):
            if first_detection in removed:
                continue
            first_hand = groups[first_detection][0]
            for second_detection in detections[first_index + 1:]:
                if second_detection in removed:
                    continue
                second_hand = groups[second_detection][0]
                first_box = first_hand["bbox_xyxy"]
                second_box = second_hand["bbox_xyxy"]
                iou = _bbox_iou(first_box, second_box)
                containment = _bbox_containment(first_box, second_box)
                if iou >= float(duplicate_iou):
                    # Same-size duplicate boxes: keep the higher-confidence
                    # representative because the list is confidence sorted.
                    removed.add(second_detection)
                elif containment >= float(duplicate_containment):
                    # Nested duplicate boxes: the larger box is usually the
                    # accidental two-hand merge, so prefer the smaller one.
                    first_area = _bbox_area(first_hand["bbox_xyxy"])
                    second_area = _bbox_area(second_hand["bbox_xyxy"])
                    if first_area >= second_area:
                        removed.add(first_detection)
                        break
                    removed.add(second_detection)
        detections = [index for index in detections if index not in removed]
    return detections[:limit]


def _detector_side(hand: dict[str, Any]) -> int | None:
    value = hand.get("detector_is_right")
    if value is None:
        return None
    try:
        value = int(value)
    except (TypeError, ValueError):
        return None
    return value if value in (0, 1) else None


def _ordered_hand_pairs(
    detections: list[int], groups: dict[int, dict[int, dict[str, Any]]],
    handedness_mode: str,
) -> list[tuple[int, int]]:
    pairs = list(itertools.permutations(detections, 2))
    if handedness_mode == "strict":
        pairs = [
            pair for pair in pairs
            if _detector_side(groups[pair[0]][0]) in (None, 0)
            and _detector_side(groups[pair[1]][1]) in (None, 1)
        ]
    return pairs


def _project_base_points(
    camera: CameraCalibration, points_base: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    rotation = camera.T_base_camera[:3, :3]
    center = camera.T_base_camera[:3, 3]
    points_camera = (rotation.T @ (points_base - center).T).T
    return project_points(camera, points_camera)


def _candidate_error(
    anchor_points: np.ndarray, camera: CameraCalibration, hand: dict[str, Any],
) -> float:
    projected, valid = _project_base_points(camera, anchor_points)
    observed = np.asarray(hand["joints_2d"], dtype=np.float64)
    valid &= np.isfinite(anchor_points).all(axis=1) & np.isfinite(observed).all(axis=1)
    if np.count_nonzero(valid) < 12:
        return float("inf")
    errors = np.linalg.norm(projected[valid] - observed[valid], axis=1)
    return float(np.median(errors))


def _match_camera(
    camera_id: str,
    groups: dict[int, dict[int, dict[str, Any]]],
    anchors: dict[int, dict[str, Any]],
    calibration: CameraCalibration,
    threshold_px: float,
    max_detections: int,
    handedness_mode: str,
    suppress_duplicate_boxes: bool = False,
    duplicate_iou: float = 0.95,
    duplicate_containment: float = 0.95,
) -> tuple[dict[int, dict[str, Any]], dict[int, float]]:
    detections = _complete_detections(
        groups, max_detections, suppress_duplicate_boxes, duplicate_iou,
        duplicate_containment,
    )
    candidates: dict[int, list[tuple[int | None, float, dict[str, Any] | None]]] = {}
    for side in (0, 1):
        values: list[tuple[int | None, float, dict[str, Any] | None]] = [
            (None, threshold_px, None)
        ]
        for detection in detections:
            hand = groups[detection][side]
            if handedness_mode == "strict" and _detector_side(hand) not in (None, side):
                continue
            error = _candidate_error(anchors[side]["points"], calibration, hand)
            if error <= threshold_px:
                values.append((detection, error, hand))
        candidates[side] = values
    options = []
    for left, right in itertools.product(candidates[0], candidates[1]):
        if left[0] is not None and left[0] == right[0]:
            continue
        matched = int(left[0] is not None) + int(right[0] is not None)
        options.append((left[1] + right[1] - 0.25 * threshold_px * matched, left, right))
    if not options:
        return {}, {}
    _, left, right = min(options, key=lambda item: item[0])
    selected = {}
    errors = {}
    for side, value in ((0, left), (1, right)):
        if value[0] is not None and value[2] is not None:
            selected[side] = value[2]
            errors[side] = float(value[1])
    return selected, errors


def _anatomy_penalty(points: np.ndarray) -> float:
    lengths = []
    for start, end in HAND_EDGES:
        if np.all(np.isfinite(points[[start, end]])):
            lengths.append(float(np.linalg.norm(points[start] - points[end])))
    if len(lengths) < 12:
        return 500.0
    values = np.asarray(lengths)
    return float(
        1000.0 * np.maximum(0.006 - values, 0.0).sum()
        + 1000.0 * np.maximum(values - 0.09, 0.0).sum()
    )


def _view_consensus_metrics(
    result: dict[str, Any],
) -> dict[str, dict[str, float | int]]:
    """Summarize each camera's agreement with one triangulated hand."""
    metrics: dict[str, dict[str, float | int]] = {}
    errors = np.asarray(result["errors"], dtype=np.float64)
    inlier_mask = np.asarray(result["inlier_mask"], dtype=bool)
    for index, camera in enumerate(result["cameras"]):
        values = errors[:, index]
        finite = np.isfinite(values)
        metrics[camera] = {
            "median_px": (
                float(np.median(values[finite])) if np.any(finite) else float("inf")
            ),
            "p95_px": (
                float(np.percentile(values[finite], 95))
                if np.any(finite) else float("inf")
            ),
            "inlier_joint_count": int(np.count_nonzero(inlier_mask[:, index])),
            "valid_joint_count": int(np.count_nonzero(finite)),
        }
    return metrics


def _robust_view_consensus(
    selected_by_side: dict[int, dict[str, dict[str, Any]]],
    calibrations: dict[str, CameraCalibration],
    args: argparse.Namespace,
) -> tuple[
    list[dict[str, Any]],
    dict[int, list[str]],
    dict[int, dict[str, dict[str, float | int]]],
]:
    """Remove a per-hand view only when other views provide strong support.

    The old path always trusted the selected anchor pair.  This pass keeps the
    pair as an initial seed, but tests every selected view against the robust
    multi-view result.  A camera is removed for that hand only when at least
    two other views have usable inliers and the camera is an absolute and
    relative outlier.  This makes camera5 and isolated camera3 failures unable
    to move an otherwise consistent 3D estimate, while retaining two-view
    fallback when the sequence genuinely has only two valid observations.
    """
    rejected_by_side: dict[int, list[str]] = {0: [], 1: []}
    diagnostics_by_side: dict[int, dict[str, dict[str, float | int]]] = {
        0: {}, 1: {},
    }
    enabled = bool(getattr(args, "robust_view_consensus", 0))
    min_views = max(3, int(getattr(args, "consensus_min_views", 3)))
    min_inliers = max(2, int(getattr(args, "consensus_min_inlier_joints", 8)))
    absolute_threshold = float(
        getattr(args, "consensus_outlier_median_px", 35.0)
    )
    ratio_threshold = float(getattr(args, "consensus_outlier_ratio", 2.5))

    for _iteration in range(2):
        results = [
            _triangulate_hand(
                side, selected_by_side[side], calibrations,
                args.ransac_threshold_px,
                _final_joint_min_views(selected_by_side[side], args),
                bool(getattr(args, "sam3_gate", 0)),
                condition_hard_threshold=float(
                    getattr(args, "condition_hard_threshold", 1000.0)
                ),
            )
            for side in (0, 1)
        ]
        if not enabled:
            return results, rejected_by_side, diagnostics_by_side
        changed = False
        for side, result in enumerate(results):
            metrics = _view_consensus_metrics(result)
            # SAM3 is an auxiliary image-space check.  Keep its diagnostics
            # beside the geometric diagnostics so a rejected view is
            # explainable in accepted.jsonl/summary.json.
            for camera, values in metrics.items():
                selected_hand = selected_by_side[side].get(camera, {})
                values["sam3_enabled"] = int(
                    bool(selected_hand.get("sam3_assist", {}).get("enabled"))
                )
                values["sam3_matched"] = int(
                    bool(selected_hand.get("sam3_assist", {}).get("matched"))
                )
                support_value = selected_hand.get("sam3_assist", {}).get(
                    "tolerant_inside_ratio_visible"
                )
                values["sam3_joint_support"] = (
                    float(support_value) if support_value is not None else -1.0
                )
                values["sam3_bad"] = int(
                    _sam3_view_is_bad(selected_hand, args)
                )
            for camera, values in metrics.items():
                diagnostics_by_side[side][camera] = dict(values)
            selected = selected_by_side[side]
            if len(selected) < min_views:
                continue
            to_remove: list[str] = []
            for camera, values in metrics.items():
                others = [item for key, item in metrics.items() if key != camera]
                supporting = [
                    item for item in others
                    if int(item["inlier_joint_count"]) >= min_inliers
                    and np.isfinite(float(item["median_px"]))
                    and not bool(item.get("sam3_bad", 0))
                ]
                if len(supporting) < 2:
                    continue
                reference_median = float(np.median([
                    float(item["median_px"]) for item in supporting
                ]))
                camera_median = float(values["median_px"])
                high_error = (
                    camera_median > absolute_threshold
                    and camera_median > ratio_threshold * max(reference_median, 1.0)
                )
                low_support = (
                    int(values["inlier_joint_count"]) < min_inliers
                    and len(supporting) >= 2
                )
                sam3_bad = bool(values.get("sam3_bad", 0))
                # A SAM3 rejection is only actionable when at least two other
                # views support the same hand.  This preserves the old
                # two-view fallback and makes SAM3 a conservative gate.
                if (high_error or low_support or (sam3_bad and len(supporting) >= 2)) and camera in selected:
                    to_remove.append(camera)
            # Never reduce a hand below a valid stereo pair.  Removing several
            # clear outliers at once is useful when two cameras fail together.
            max_removable = max(0, len(selected) - 2)
            for camera in to_remove[:max_removable]:
                selected.pop(camera, None)
                if camera not in rejected_by_side[side]:
                    rejected_by_side[side].append(camera)
                changed = True
        if not changed:
            return results, rejected_by_side, diagnostics_by_side

    results = [
        _triangulate_hand(
            side, selected_by_side[side], calibrations,
            args.ransac_threshold_px,
            _final_joint_min_views(selected_by_side[side], args),
            bool(getattr(args, "sam3_gate", 0)),
            condition_hard_threshold=float(
                getattr(args, "condition_hard_threshold", 1000.0)
            ),
        )
        for side in (0, 1)
    ]
    for side, result in enumerate(results):
        for camera, values in _view_consensus_metrics(result).items():
            diagnostics_by_side[side][camera] = dict(values)
    return results, rejected_by_side, diagnostics_by_side


def _marker_assist_cost(
    selected_by_side: dict[int, dict[str, dict[str, Any]]],
) -> float:
    """Use marker evidence as a mild hypothesis-selection term.

    Multiview geometry remains dominant. Strong marker corrections get the
    original mild preference; topology-valid partial evidence gets only one
    quarter of that weight and can therefore break close choices without
    moving image joints.
    """
    cost = 0.0
    for selected in selected_by_side.values():
        for hand in selected.values():
            marker = hand.get("marker_assist", {})
            if marker.get("applied"):
                weight = 1.0
            elif marker.get("evidence_only"):
                weight = 0.25
            else:
                continue
            matches = float(marker.get("matched_marker_count", 0))
            groups = float(marker.get("finger_group_count", 0))
            residual = float(marker.get("match_residual_median_px", 0.0))
            cost += weight * (
                0.15 * residual - 0.30 * matches - 0.20 * groups
            )
    return cost


def _sam3_assist_cost(
    selected_by_side: dict[int, dict[str, dict[str, Any]]],
    args: argparse.Namespace,
) -> float:
    """Add a small SAM3 preference without overriding multiview geometry.

    SAM3 is evidence about the visible glove region, not a replacement for a
    calibrated ray.  Accepted hypotheses receive a mild bonus; a matched mask
    with poor joint support receives a penalty.  Missing masks are neutral so
    an occluded/low-contrast frame can still use the original WiLoR path.
    """
    if not bool(getattr(args, "sam3_gate", 0)):
        return 0.0
    cost = 0.0
    min_support = float(getattr(args, "sam3_min_joint_support", 0.55))
    min_score = float(getattr(args, "sam3_min_score", 0.25))
    min_iou = float(getattr(args, "sam3_min_box_iou", 0.20))
    for selected in selected_by_side.values():
        for hand in selected.values():
            info = hand.get("sam3_assist", {})
            if not info.get("enabled"):
                continue
            if not info.get("matched"):
                # Keep missing masks neutral for hard rejection (occlusion is
                # common), but give a small preference to detections that did
                # produce a glove mask when competing geometric hypotheses
                # are otherwise nearly tied.
                cost += 0.15
                continue
            score = float(info.get("mask_score", 0.0))
            iou = float(info.get("mask_box_iou_yolo", 0.0))
            support_value = info.get("tolerant_inside_ratio_visible")
            support = (
                float(support_value) if support_value is not None else 0.0
            )
            quality = min(
                score / max(min_score, 1e-6),
                iou / max(min_iou, 1e-6),
                support / max(min_support, 1e-6),
            )
            cost += 0.8 * max(0.0, 1.0 - min(quality, 1.0))
            if quality >= 1.0:
                cost -= 0.35
    return float(cost)


def _sam3_view_is_bad(hand: dict[str, Any], args: argparse.Namespace) -> bool:
    """Return true only for a matched mask that strongly rejects this pose."""
    if not bool(getattr(args, "sam3_gate", 0)):
        return False
    info = hand.get("sam3_assist", {})
    if not info.get("enabled") or not info.get("matched"):
        # A missing mask is intentionally neutral.  It can be caused by
        # occlusion and must not erase an otherwise geometrically valid view.
        return False
    score = float(info.get("mask_score", 0.0))
    iou = float(info.get("mask_box_iou_yolo", 0.0))
    support_value = info.get("tolerant_inside_ratio_visible")
    support = float(support_value) if support_value is not None else 0.0
    return (
        score < float(getattr(args, "sam3_min_score", 0.25))
        or iou < float(getattr(args, "sam3_min_box_iou", 0.20))
        or support < float(getattr(args, "sam3_hard_joint_support", 0.35))
    )


def _final_joint_min_views(
    selected: Mapping[str, dict[str, Any]], args: argparse.Namespace,
) -> int:
    """Require multi-view support for final depth, optionally retaining stereo.

    A hand may be associated in four or more cameras while an individual
    fingertip has only two RANSAC inliers.  Those two rays are enough to make a
    finite point, but not enough to disambiguate a noisy depth branch.  Keep
    such points missing when a third camera is available.  The final output
    is strict by default even when only two views survived association;
    ``--allow-stereo-final 1`` restores the legacy fallback for experiments.
    """
    # Older callers/tests construct a minimal ``argparse.Namespace`` without
    # the new option.  Preserve the legacy two-view behavior for those
    # programmatic calls; the CLI parser explicitly defaults to three views
    # for the safer production path.
    requested = max(2, int(getattr(args, "min_joint_inlier_views", 2)))
    if bool(getattr(args, "allow_stereo_final", 0)) and len(selected) < requested:
        return 2
    return requested


def _evaluate_anchor_assignment(
    anchor_cameras: tuple[str, str],
    anchor_pairs: tuple[tuple[int, int], tuple[int, int]],
    groups: dict[str, dict[int, dict[int, dict[str, Any]]]],
    calibrations: dict[str, CameraCalibration],
    all_cameras: tuple[str, ...],
    args: argparse.Namespace,
) -> dict[str, Any]:
    selected_by_side: dict[int, dict[str, dict[str, Any]]] = {0: {}, 1: {}}
    for camera, pair in zip(anchor_cameras, anchor_pairs):
        selected_by_side[0][camera] = groups[camera][pair[0]][0]
        selected_by_side[1][camera] = groups[camera][pair[1]][1]
    anchor_results = {
        side: _triangulate_hand(
            side, selected_by_side[side], calibrations,
            args.anchor_threshold_px, 2, bool(getattr(args, "sam3_gate", 0)),
            # The anchor pair is used only to associate the remaining views.
            # Keep its finite (even if weakly conditioned) points available
            # for image-space matching; the condition gate is applied to the
            # final all-view result below.
            reject_ill_conditioned=False,
        )
        for side in (0, 1)
    }
    if any(result["valid_joints"] < args.min_valid_joints for result in anchor_results.values()):
        return {"cost": float("inf")}
    association_errors: dict[int, dict[str, float]] = {0: {}, 1: {}}
    for camera in all_cameras:
        if camera in anchor_cameras:
            continue
        matched, errors = _match_camera(
            camera, groups[camera], anchor_results, calibrations[camera],
            args.association_threshold_px, args.max_side_detections,
            args.detector_handedness,
            bool(getattr(args, "suppress_duplicate_detector_boxes", 0)),
            float(getattr(args, "duplicate_detector_iou", 0.95)),
            float(getattr(args, "duplicate_detector_containment", 0.95)),
        )
        for side, hand in matched.items():
            selected_by_side[side][camera] = hand
            association_errors[side][camera] = errors[side]
    final_results = [
        _triangulate_hand(
            side, selected_by_side[side], calibrations,
            args.ransac_threshold_px,
            _final_joint_min_views(selected_by_side[side], args),
            bool(getattr(args, "sam3_gate", 0)),
            condition_hard_threshold=float(
                getattr(args, "condition_hard_threshold", 1000.0)
            ),
        )
        for side in (0, 1)
    ]
    consensus_rejected, consensus_diagnostics = ({0: [], 1: []}, {0: {}, 1: {}})
    if bool(getattr(args, "robust_view_consensus", 0)):
        (
            final_results,
            consensus_rejected,
            consensus_diagnostics,
        ) = _robust_view_consensus(selected_by_side, calibrations, args)
    missing = sum(21 - result["valid_joints"] for result in final_results)
    extra_joint_support = sum(
        int(np.maximum(result["inlier_counts"] - 2, 0).sum())
        for result in final_results
    )
    matched_side_views = sum(
        max(len(selected_by_side[side]) - 2, 0) for side in (0, 1)
    )
    rejected_view_count = sum(
        len(consensus_rejected[side]) for side in (0, 1)
    )
    cost = (
        100.0 * missing
        + sum(result["median_px"] + 0.2 * result["p95_px"] for result in final_results)
        + sum(_anatomy_penalty(result["points"]) for result in final_results)
        + _marker_assist_cost(selected_by_side)
        + _sam3_assist_cost(selected_by_side, args)
        - 1.5 * extra_joint_support
        - 4.0 * matched_side_views
        + 1.0 * rejected_view_count
    )
    return {
        "cost": float(cost),
        "anchor_cameras": list(anchor_cameras),
        "selected": selected_by_side,
        "results": final_results,
        "association_errors": association_errors,
        "consensus_rejected": consensus_rejected,
        "consensus_diagnostics": consensus_diagnostics,
    }


def _quality_reasons(results: list[dict[str, Any]], args: argparse.Namespace) -> list[str]:
    reasons = []
    for result in results:
        if result["valid_joints"] < args.min_valid_joints:
            reasons.append("too_few_valid_joints")
        if result["median_px"] > args.max_reprojection_median_px:
            reasons.append("reprojection_median_too_large")
        if result["p95_px"] > args.max_reprojection_p95_px:
            reasons.append("reprojection_p95_too_large")
    return sorted(set(reasons))


def _restore_condition_rejected_joints(
    rows: list[dict[str, Any]], args: argparse.Namespace,
) -> int:
    """Keep the final ray-quality gate intact after anatomy interpolation.

    The anatomy pass is deliberately allowed to fill ordinary missing joints,
    but it must not resurrect a joint that the multiview ray gate marked as
    geometrically unobservable.  Otherwise a high-condition depth branch can
    re-enter immediately before temporal filtering and appear as a plausible
    finite point again.
    """
    threshold = float(getattr(args, "condition_hard_threshold", 1000.0))
    restored = 0
    for row in rows:
        for hand in row.get("hands", []):
            quality = hand.get("quality", {})
            conditions = np.asarray(
                quality.get("ray_condition_numbers", []), dtype=np.float64,
            )
            points = np.asarray(hand.get("joints_base_m", []), dtype=np.float64)
            if conditions.shape != (21,) or points.shape != (21, 3):
                continue
            invalid = ~np.isfinite(conditions) | (conditions >= threshold)
            newly_invalid = invalid & np.isfinite(points).all(axis=1)
            if not np.any(newly_invalid):
                continue
            points[newly_invalid] = np.nan
            hand["joints_base_m"] = points.tolist()
            restored += int(np.count_nonzero(newly_invalid))
    return restored


def _post_temporal_target_camera_guard(
    rows: list[dict[str, Any]],
    calibrations: dict[str, CameraCalibration],
    config: FusionAnatomyConfig,
    branch_threshold_m: float = 0.030,
) -> dict[str, Any]:
    """Re-apply the protected-camera reprojection guard after 3-D tracking.

    The per-joint temporal tracker runs after anatomy refinement and can move
    a point onto a smooth but wrong depth branch.  The anatomy guard has
    already protected its own output, but that later pass used to bypass the
    target-camera constraint.  Compare the final tracked pose with the
    original fused pose saved as ``unrefined_joints_base_m`` and limit only
    the offending joints along that correction path.  Camera-native 2-D
    observations are never modified.
    """
    protected = tuple(config.protected_camera_ids)
    if not rows or not protected or not calibrations:
        return {
            "enabled": bool(protected),
            "applied": False,
            "protected_camera_ids": list(protected),
            "limited_joint_count": 0,
            "reverted_joint_count": 0,
        }
    frame_count = max(
        (int(row.get("sync_index", -1)) for row in rows), default=-1
    ) + 1
    if frame_count <= 0:
        return {
            "enabled": True,
            "applied": False,
            "protected_camera_ids": list(protected),
            "limited_joint_count": 0,
            "reverted_joint_count": 0,
        }
    raw = np.full((2, frame_count, 21, 3), np.nan, dtype=np.float64)
    current = np.full_like(raw, np.nan)
    hands: dict[tuple[int, int], dict[str, Any]] = {}
    for row in rows:
        frame = int(row.get("sync_index", -1))
        if frame < 0 or frame >= frame_count:
            continue
        for hand in row.get("hands", []):
            side = int(hand.get("side", -1))
            if side not in (0, 1):
                continue
            points = np.asarray(hand.get("joints_base_m", []), dtype=np.float64)
            if points.shape != (21, 3):
                continue
            original = np.asarray(
                hand.get("unrefined_joints_base_m", hand.get("joints_base_m", [])),
                dtype=np.float64,
            )
            if original.shape != (21, 3):
                original = points.copy()
            raw[side, frame] = original
            current[side, frame] = points
            hands[(frame, side)] = hand
    if not hands:
        return {
            "enabled": True,
            "applied": False,
            "protected_camera_ids": list(protected),
            "limited_joint_count": 0,
            "reverted_joint_count": 0,
        }
    guarded, limited, reverted, diagnostics = _apply_reprojection_guard(
        current, raw, hands, calibrations, config,
    )
    for (frame, side), hand in hands.items():
        hand["joints_base_m"] = guarded[side, frame].tolist()
    branch_guard = _clear_post_guard_temporal_branches(
        rows,
        threshold_m=max(float(branch_threshold_m), 1e-6),
        min_hand_joints=18,
    )
    return {
        "enabled": True,
        "applied": bool(limited or reverted),
        "protected_camera_ids": list(protected),
        "protected_camera_shift_px": float(config.protected_camera_shift_px),
        "branch_threshold_m": float(branch_threshold_m),
        "limited_joint_count": int(limited),
        "reverted_joint_count": int(reverted),
        "temporal_branch_guard": branch_guard,
        "reprojection_diagnostics": diagnostics,
    }


def _clear_post_guard_temporal_branches(
    rows: list[dict[str, Any]],
    *,
    threshold_m: float,
    min_hand_joints: int,
) -> dict[str, Any]:
    """Remove residual 3-D branch runs introduced by the target-camera pass.

    A reprojection limit moves each frame toward its own raw ray solution.  If
    neighbouring raw solutions select different depth branches, that local
    operation can create a new 3-D step even though every individual image
    residual is acceptable.  Scan each joint between neighbouring trusted
    accepted samples;
    when the camera-native wrist is not moving by a comparable amount, discard
    the branch until it returns to that sample.  This is a visibility guard,
    not a fixed-size motion clamp: a genuinely fast hand motion with matching
    2-D wrist evidence is kept.
    """
    threshold = max(float(threshold_m), 1e-6)
    ordered = sorted(rows, key=lambda row: int(row.get("sync_index", 0)))
    cleared_joint_count = 0
    cleared_hand_count = 0
    repaired_joint_count = 0
    held_hand_count = 0
    event_count = 0

    def _points(hand: dict[str, Any]) -> np.ndarray:
        value = np.asarray(hand.get("joints_base_m", []), dtype=np.float64)
        return value if value.shape == (21, 3) else np.full((21, 3), np.nan)

    palm_indices = np.asarray((0, 1, 2, 5, 9, 13, 17), dtype=np.int64)

    def _wrist_motion_px(previous: dict[str, Any], current: dict[str, Any]) -> float | None:
        values: list[float] = []
        for camera in set(previous.get("views", {})).intersection(current.get("views", {})):
            first = np.asarray(previous["views"][camera].get("joints_2d", []), dtype=np.float64)
            second = np.asarray(current["views"][camera].get("joints_2d", []), dtype=np.float64)
            if first.shape == (21, 2) and second.shape == (21, 2):
                if np.isfinite(first[0]).all() and np.isfinite(second[0]).all():
                    values.append(float(np.linalg.norm(second[0] - first[0])))
        return float(np.median(values)) if values else None

    def _view_motion_px(
        previous: dict[str, Any], current: dict[str, Any],
    ) -> tuple[float | None, int]:
        """Measure robust palm 2-D motion across common cameras.

        A single wrist can be missing or mislabeled, so use the median motion
        of core palm joints in each common view, then take the median across
        views.  Distal fingers are intentionally excluded: a left/right
        hypothesis flip can move several fingertips by 80--150 px while the
        physical palm is stationary.  Requiring two agreeing cameras prevents
        a newly acquired stereo pair from certifying a spurious 3-D
        translation.
        """
        values: list[float] = []
        for camera in set(previous.get("views", {})).intersection(current.get("views", {})):
            first = np.asarray(previous["views"][camera].get("joints_2d", []), dtype=np.float64)
            second = np.asarray(current["views"][camera].get("joints_2d", []), dtype=np.float64)
            if first.shape != (21, 2) or second.shape != (21, 2):
                continue
            finite = np.isfinite(first[palm_indices]).all(axis=1) & np.isfinite(
                second[palm_indices]
            ).all(axis=1)
            if np.count_nonzero(finite) >= 3:
                first_palm = first[palm_indices][finite]
                second_palm = second[palm_indices][finite]
                values.append(float(np.median(np.linalg.norm(
                    second_palm - first_palm, axis=1
                ))))
        return (float(np.median(values)), len(values)) if values else (None, 0)

    for side in (0, 1):
        records = [
            (
                int(row.get("sync_index", 0)),
                next((hand for hand in row.get("hands", []) if int(hand.get("side", -1)) == side), None),
            )
            for row in ordered
        ]
        records = [(frame, hand) for frame, hand in records if hand is not None]
        for record_index, ((previous_frame, previous_hand), (current_frame, current_hand)) in enumerate(
            zip(records, records[1:])
        ):
            frame_gap = current_frame - previous_frame
            # A rejected/occluded frame must not hide a branch switch. Check
            # short accepted-track gaps as well as strictly consecutive rows;
            # longer gaps are left to the gap-aware temporal tracker.
            if frame_gap < 1 or frame_gap > 3:
                continue
            previous = _points(previous_hand)
            current = _points(current_hand)
            # This guard is intentionally offline and zero-phase.  A
            # one-sided ``hold previous`` repair can lock a hand on a wrong
            # depth branch forever when the branch persists for several
            # frames (the old all-gap video made this especially obvious).
            # Require a following accepted sample and only repair a point
            # when that sample returns close to the preceding trajectory.
            # Sustained motion/branches are left to the bidirectional joint
            # tracker instead of being frozen here.
            # Search a few accepted samples ahead for the return endpoint.
            # Glove WiLoR branches often last two frames, so looking only at
            # the immediate next row misses exactly the jump we need to
            # remove.  Per-joint endpoints allow a fingertip to be missing
            # while the rest of the hand still gets a valid return target.
            return_points = np.full((21, 3), np.nan, dtype=np.float64)
            return_span = np.zeros(21, dtype=np.int64)
            return_gap = 0
            for lookahead in range(1, 4):
                candidate_record = (
                    records[record_index + 1 + lookahead]
                    if record_index + 1 + lookahead < len(records)
                    else None
                )
                if candidate_record is None:
                    break
                candidate_frame, candidate_hand = candidate_record
                candidate_gap = int(candidate_frame) - current_frame
                if candidate_gap < 1 or candidate_gap > 3 * lookahead:
                    continue
                candidate_points = _points(candidate_hand)
                candidate_finite = np.isfinite(candidate_points).all(axis=1)
                previous_finite = np.isfinite(previous).all(axis=1)
                endpoint = np.linalg.norm(candidate_points - previous, axis=1)
                endpoint_limit = max(
                    threshold * (frame_gap + candidate_gap) * 1.25,
                    0.045,
                )
                usable = (
                    candidate_finite
                    & previous_finite
                    & ~np.isfinite(return_points).all(axis=1)
                    & (endpoint <= endpoint_limit)
                )
                if np.any(usable):
                    return_points[usable] = candidate_points[usable]
                    return_span[usable] = candidate_gap + frame_gap
                    return_gap = max(return_gap, candidate_gap)
            following = return_points
            next_gap = return_gap
            # Compare only core palm joints that are finite in *both* frames.
            # Taking two independent medians makes a missing thumb/pinky look
            # like a large palm translation and incorrectly preserves a distal
            # depth branch.  A common-joint median is invariant to that
            # visibility change.
            common_palm = (
                np.isfinite(previous[palm_indices]).all(axis=1)
                & np.isfinite(current[palm_indices]).all(axis=1)
            )
            previous_palm = previous[palm_indices][common_palm]
            current_palm = current[palm_indices][common_palm]
            palm_step = (
                float(np.linalg.norm(
                    np.median(current_palm, axis=0)
                    - np.median(previous_palm, axis=0)
                ))
                if previous_palm.shape[0] >= 3
                else float("inf")
            )
            wrist_px = _wrist_motion_px(previous_hand, current_hand)
            view_motion_px, view_motion_camera_count = _view_motion_px(
                previous_hand, current_hand,
            )
            # A large 3-D step is retained only when at least two common
            # cameras observe comparable 2-D motion.  This catches both
            # depth-only branches and a hand that reappears from a different
            # two-view association; a lone camera/wrist jump is not enough.
            motion_supported = (
                view_motion_camera_count >= 2
                and view_motion_px is not None
                and view_motion_px > 60.0 * frame_gap
                # A large distal-joint displacement is not evidence of a
                # real hand translation.  Require the wrist to move as well;
                # this prevents a simultaneous finger-hypothesis flip from
                # certifying a depth branch as fast physical motion.
                and wrist_px is not None
                and wrist_px > 60.0 * frame_gap
            )
            finite = np.isfinite(previous).all(axis=1) & np.isfinite(current).all(axis=1)
            distances = np.linalg.norm(current - previous, axis=1)
            # A depth branch can make the same fingertip jump in 3-D while
            # the whole-hand 2-D median also moves (for example, when one
            # WiLoR hypothesis flips several distal joints together).  In
            # that case the aggregate motion test below is too permissive.
            # If at least half of the common SAM3-supported views explicitly
            # mark this joint outside the glove mask, treat the joint as a
            # visibility outlier and clear it independently of whole-hand
            # motion.  SAM3 remains only a veto for an already-large 3-D
            # step; it never creates a 3-D point.
            sam3_hard_conflict = np.zeros(21, dtype=bool)
            sam3_observation_count = np.zeros(21, dtype=np.int32)
            sam3_outlier_count = np.zeros(21, dtype=np.int32)
            for camera in set(previous_hand.get("views", {})).intersection(
                current_hand.get("views", {})
            ):
                previous_sam3 = previous_hand["views"][camera].get("sam3_assist", {})
                current_sam3 = current_hand["views"][camera].get("sam3_assist", {})
                if not (
                    isinstance(previous_sam3, dict)
                    and isinstance(current_sam3, dict)
                    and current_sam3.get("enabled")
                    and current_sam3.get("matched")
                ):
                    continue
                flags = current_sam3.get("hard_outlier_flags", [])
                if not isinstance(flags, (list, tuple)) or len(flags) != 21:
                    continue
                for joint in range(21):
                    sam3_observation_count[joint] += 1
                    sam3_outlier_count[joint] += int(bool(flags[joint]))
            valid_sam3 = sam3_observation_count >= 2
            sam3_hard_conflict[valid_sam3] = (
                sam3_outlier_count[valid_sam3] * 2
                >= sam3_observation_count[valid_sam3]
            )
            # ``motion_supported`` is a scalar Python bool.  Using the
            # bitwise ``~`` operator here turns ``False`` into ``-1`` and
            # silently converts the boolean mask to integer indices (which
            # only clears joints 0/1).  Use logical negation so the mask
            # remains boolean and the actual distal branch points are
            # removed.
            suspicious = finite & (
                distances > threshold * frame_gap
            ) & ((not motion_supported) | sam3_hard_conflict)
            if not np.isfinite(following).any() or next_gap < 1:
                # No trusted right endpoint: do not manufacture a pose.
                suspicious[:] = False
            else:
                following_finite = np.isfinite(following).all(axis=1)
                # ``following`` contains only endpoints that passed the
                # return-distance test above; its finite mask is therefore
                # the per-joint definition of an isolated branch.
                suspicious &= following_finite
            if not np.any(suspicious):
                continue
            # Interpolate between the two trusted endpoints rather than
            # holding the left endpoint.  This removes a single-frame branch
            # without introducing a fixed pose or a fixed-size motion step.
            previous_finite = np.isfinite(previous).all(axis=1)
            following_finite = np.isfinite(following).all(axis=1)
            repairable = suspicious & previous_finite & following_finite
            span = np.maximum(return_span.astype(np.float64), 1.0)
            alpha = float(frame_gap) / span
            target = previous + alpha[:, None] * (following - previous)
            current[repairable] = target[repairable]
            current[suspicious & ~repairable] = np.nan
            current_hand["joints_base_m"] = current.tolist()
            count = int(np.count_nonzero(suspicious & ~repairable))
            repaired = int(np.count_nonzero(repairable))
            repaired_joint_count += repaired
            cleared_joint_count += count
            event_count += 1
            current_hand["post_temporal_branch_guard"] = {
                "applied": True,
                "cleared_joint_count": count,
                "repaired_joint_count": repaired,
                "repair_mode": "interpolate_returning_endpoint",
                "threshold_m": threshold,
                "frame_gap": frame_gap,
                "next_gap": next_gap,
                "palm_step_m": None if not np.isfinite(palm_step) else palm_step,
                "wrist_motion_px": wrist_px,
                "view_motion_px": view_motion_px,
                "view_motion_camera_count": view_motion_camera_count,
            }

    # Keep hand visibility atomic after deleting branch points.  For a short
    # isolated partial pose, hold the nearest complete trusted pose instead of
    # turning a valid hand into a blank tile.  A long gap has no nearby stable
    # endpoint and remains an explicit dropout.
    minimum = max(int(min_hand_joints), 0)
    stable_by_side: dict[int, list[tuple[int, np.ndarray]]] = {0: [], 1: []}
    for side in (0, 1):
        for frame, hand in [
            (int(row.get("sync_index", 0)), hand)
            for row in ordered
            for hand in row.get("hands", [])
            if int(hand.get("side", -1)) == side
        ]:
            points = _points(hand)
            if int(np.isfinite(points).all(axis=1).sum()) >= minimum:
                stable_by_side[side].append((frame, points.copy()))
    if minimum:
        for row in ordered:
            for hand in row.get("hands", []):
                points = _points(hand)
                finite_count = int(np.isfinite(points).all(axis=1).sum())
                if finite_count == 0 or finite_count >= minimum:
                    continue
                side = int(hand.get("side", -1))
                frame = int(row.get("sync_index", 0))
                candidates = stable_by_side.get(side, [])
                nearest = min(
                    candidates, key=lambda item: abs(item[0] - frame),
                    default=None,
                )
                if nearest is not None and abs(nearest[0] - frame) <= 3:
                    hand["joints_base_m"] = nearest[1].tolist()
                    hand["post_temporal_branch_visibility_guard"] = {
                        "applied": True,
                        "minimum_finite_joints": minimum,
                        "finite_joint_count_before": finite_count,
                        "repair_mode": "hold_nearest_complete_pose",
                        "held_from_sync_index": int(nearest[0]),
                    }
                    held_hand_count += 1
                    repaired_joint_count += finite_count
                    continue
                hand["joints_base_m"] = np.full((21, 3), np.nan).tolist()
                hand["post_temporal_branch_visibility_guard"] = {
                    "applied": True,
                    "minimum_finite_joints": minimum,
                    "finite_joint_count_before": finite_count,
                }
                cleared_hand_count += 1
                cleared_joint_count += finite_count
    return {
        "enabled": True,
        "threshold_m": threshold,
        "view_motion_threshold_px": 60.0,
        "cleared_joint_count": int(cleared_joint_count),
        "cleared_hand_count": int(cleared_hand_count),
        "repaired_joint_count": int(repaired_joint_count),
        "held_hand_count": int(held_hand_count),
        "event_count": int(event_count),
    }


def _recover_from_reference(
    sync_index: int,
    reference: dict[str, Any],
    groups: dict[str, dict[int, dict[int, dict[str, Any]]]],
    calibrations: dict[str, CameraCalibration],
    cameras: tuple[str, ...],
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    reference_points = {
        int(hand["side"]): np.asarray(hand["joints_base_m"], dtype=np.float64)
        for hand in reference["hands"]
    }
    if set(reference_points) != {0, 1}:
        return None
    seeds = {side: {"points": reference_points[side]} for side in (0, 1)}
    selected_by_side: dict[int, dict[str, dict[str, Any]]] = {0: {}, 1: {}}
    association_errors: dict[int, dict[str, float]] = {0: {}, 1: {}}
    for camera in cameras:
        matched, errors = _match_camera(
            camera, groups[camera], seeds, calibrations[camera],
            args.temporal_association_threshold_px, args.max_side_detections,
            args.detector_handedness,
            bool(getattr(args, "suppress_duplicate_detector_boxes", 0)),
            float(getattr(args, "duplicate_detector_iou", 0.95)),
            float(getattr(args, "duplicate_detector_containment", 0.95)),
        )
        for side, hand in matched.items():
            selected_by_side[side][camera] = hand
            association_errors[side][camera] = errors[side]
    if any(len(selected_by_side[side]) < 2 for side in (0, 1)):
        return None
    results = [
        _triangulate_hand(
            side, selected_by_side[side], calibrations,
            args.ransac_threshold_px,
            _final_joint_min_views(selected_by_side[side], args),
            bool(getattr(args, "sam3_gate", 0)),
            condition_hard_threshold=float(
                getattr(args, "condition_hard_threshold", 1000.0)
            ),
        )
        for side in (0, 1)
    ]
    consensus_rejected, consensus_diagnostics = ({0: [], 1: []}, {0: {}, 1: {}})
    if bool(getattr(args, "robust_view_consensus", 0)):
        (
            results,
            consensus_rejected,
            consensus_diagnostics,
        ) = _robust_view_consensus(selected_by_side, calibrations, args)
    if _quality_reasons(results, args):
        return None
    frame_gap = abs(sync_index - int(reference["sync_index"]))
    max_wrist_step = args.max_temporal_wrist_step_m * max(frame_gap, 1)
    for side, result in enumerate(results):
        if not np.all(np.isfinite(result["points"][0])):
            return None
        if np.linalg.norm(result["points"][0] - reference_points[side][0]) > max_wrist_step:
            return None
    common = set(selected_by_side[0]) & set(selected_by_side[1])
    preferred = tuple(args.anchor_cameras)
    if all(camera in common for camera in preferred):
        comparison_pair = preferred
    elif len(common) >= 2:
        comparison_pair = tuple(sorted(common)[:2])
    else:
        comparison_pair = preferred
    hands = []
    for side, result in enumerate(results):
        hand = _serialize_hand(
            result, selected_by_side[side], calibrations, comparison_pair
        )
        hand["association_error_px"] = association_errors[side]
        if bool(getattr(args, "robust_view_consensus", 0)):
            hand["consensus_rejected_camera_ids"] = sorted(
                consensus_rejected.get(side, [])
            )
            hand["consensus_view_diagnostics"] = consensus_diagnostics.get(
                side, {}
            )
        hand["fusion_mode"] = "temporal_guided_multiview"
        hands.append(hand)
    active = sorted(set().union(*(hand["camera_ids"] for hand in hands)))
    return {
        "sync_index": sync_index,
        "anchor_cameras": list(comparison_pair),
        "recovery": {
            "method": "nearest_confirmed_multiview",
            "reference_sync_index": int(reference["sync_index"]),
            "frame_gap": frame_gap,
        },
        "active_cameras": active,
        "active_camera_count": len(active),
        "hands": hands,
    }


def _anchor_pair_tiers(
    cameras: tuple[str, ...],
    complete_by_camera: dict[str, list[int]],
    preferred_anchors: tuple[str, str],
    preferred_anchor_camera: str | None,
    minimum_detections: int,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    available = [
        pair for pair in itertools.combinations(cameras, 2)
        if all(len(complete_by_camera[camera]) >= minimum_detections for camera in pair)
    ]
    if preferred_anchor_camera is not None:
        return (
            [pair for pair in available if preferred_anchor_camera in pair],
            [pair for pair in available if preferred_anchor_camera not in pair],
        )
    preferred_available = all(
        len(complete_by_camera[camera]) >= minimum_detections
        for camera in preferred_anchors
    )
    return (
        [preferred_anchors] if preferred_available else [],
        [pair for pair in available if set(pair) != set(preferred_anchors)],
    )


def _anchor_preference_penalty(
    pair: tuple[str, str], preferred_anchors: tuple[str, str],
    args: argparse.Namespace,
) -> float:
    preferred_camera = getattr(args, "preferred_anchor_camera", None)
    if preferred_camera is not None:
        if set(pair) == set(preferred_anchors):
            return 0.0
        return 5.0 if preferred_camera in pair else 10.0
    return 5.0 * len(set(preferred_anchors) - set(pair))


def _prefer_usable_anchor_pair(
    usable: list[dict[str, Any]], preferred_anchors: tuple[str, str],
    preferred_anchor_camera: str | None,
) -> list[dict[str, Any]]:
    if preferred_anchor_camera is None:
        return usable
    preferred_pair_usable = [
        candidate for candidate in usable
        if set(candidate["anchor_cameras"]) == set(preferred_anchors)
    ]
    return preferred_pair_usable or usable


def _evaluate_anchor_pairs(
    anchor_pairs: list[tuple[str, str]],
    complete_by_camera: dict[str, list[int]],
    groups: dict[str, dict[int, dict[int, dict[str, Any]]]],
    calibrations: dict[str, CameraCalibration],
    cameras: tuple[str, ...],
    preferred_anchors: tuple[str, str],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    results = []
    for anchor_pair in anchor_pairs:
        pair_options = [
            _ordered_hand_pairs(
                complete_by_camera[camera], groups[camera], args.detector_handedness
            )
            for camera in anchor_pair
        ]
        for detection_pairs in itertools.product(*pair_options):
            candidate = _evaluate_anchor_assignment(
                anchor_pair, detection_pairs, groups, calibrations, cameras, args
            )
            if np.isfinite(candidate["cost"]):
                candidate["cost"] += _anchor_preference_penalty(
                    anchor_pair, preferred_anchors, args
                )
                results.append(candidate)
    return results


def _fuse_single_hand_frame(
    sync_index: int,
    groups: dict[str, dict[int, dict[int, dict[str, Any]]]],
    complete_by_camera: dict[str, list[int]],
    calibrations: dict[str, CameraCalibration],
    cameras: tuple[str, ...],
    preferred_anchors: tuple[str, str],
    args: argparse.Namespace,
) -> tuple[bool, dict[str, Any]]:
    """Recover one hand only when two independent calibrated views agree."""
    preferred_pairs, fallback_pairs = _anchor_pair_tiers(
        cameras, complete_by_camera, preferred_anchors,
        getattr(args, "preferred_anchor_camera", None), 1,
    )
    candidates: list[tuple[int, float, dict[str, Any]]] = []
    pairs = preferred_pairs + fallback_pairs
    for pair in pairs:
        if any(not complete_by_camera[camera] for camera in pair):
            continue
        for side in (0, 1):
            for first_index, second_index in itertools.product(
                complete_by_camera[pair[0]], complete_by_camera[pair[1]]
            ):
                anchor_hands = {
                    pair[0]: groups[pair[0]][first_index][side],
                    pair[1]: groups[pair[1]][second_index][side],
                }
                if args.detector_handedness == "strict" and any(
                    _detector_side(hand) not in (None, side)
                    for hand in anchor_hands.values()
                ):
                    continue
                anchor = _triangulate_hand(
                    side, anchor_hands, calibrations,
                    args.anchor_threshold_px, 2,
                    bool(getattr(args, "sam3_gate", 0)),
                    reject_ill_conditioned=False,
                )
                if anchor["valid_joints"] < args.min_valid_joints:
                    continue
                selected = dict(anchor_hands)
                association_errors: dict[str, float] = {}
                for camera in cameras:
                    if camera in pair:
                        continue
                    matches = []
                    for detection in complete_by_camera[camera]:
                        hand = groups[camera][detection][side]
                        if (args.detector_handedness == "strict"
                                and _detector_side(hand) not in (None, side)):
                            continue
                        error = _candidate_error(anchor["points"], calibrations[camera], hand)
                        if error <= args.association_threshold_px:
                            matches.append((error, hand))
                    if matches:
                        error, hand = min(matches, key=lambda item: item[0])
                        selected[camera] = hand
                        association_errors[camera] = float(error)
                result = _triangulate_hand(
                    side, selected, calibrations,
                    args.ransac_threshold_px,
                    _final_joint_min_views(selected, args),
                    bool(getattr(args, "sam3_gate", 0)),
                    condition_hard_threshold=float(
                        getattr(args, "condition_hard_threshold", 1000.0)
                    ),
                )
                if _quality_reasons([result], args):
                    continue
                disagreement = sum(
                    _detector_side(hand) not in (None, side)
                    for hand in selected.values()
                )
                missing = 21 - int(result["valid_joints"])
                cost = (
                    100.0 * missing + result["median_px"]
                    + 0.2 * result["p95_px"]
                    + _anatomy_penalty(result["points"])
                    + 15.0 * disagreement - 4.0 * (len(selected) - 2)
                    + _anchor_preference_penalty(pair, preferred_anchors, args)
                )
                hand = _serialize_hand(result, selected, calibrations, pair)
                hand["association_error_px"] = association_errors
                hand["fusion_mode"] = (
                    "multiview" if len(selected) >= 3 else "stereo_anchor"
                )
                preference_rank = 0
                if getattr(args, "preferred_anchor_camera", None) is not None:
                    if set(pair) == set(preferred_anchors):
                        preference_rank = 0
                    elif pair in preferred_pairs:
                        preference_rank = 1
                    else:
                        preference_rank = 2
                candidates.append((preference_rank, float(cost), {
                    "sync_index": sync_index,
                    "anchor_cameras": list(pair),
                    "assignment_cost": float(cost),
                    "assignment_margin": None,
                    "active_cameras": sorted(selected),
                    "active_camera_count": len(selected),
                    "hands": [hand],
                    "partial_hands": True,
                    "single_hand_anchor": True,
                }))
    if not candidates:
        return False, {"sync_index": sync_index,
                       "reason": "no_valid_single_hand_multiview_anchor"}
    candidates.sort(key=lambda item: (item[0], item[1]))
    _, best_cost, best = candidates[0]
    if len(candidates) > 1 and candidates[1][0] == candidates[0][0]:
        best["assignment_margin"] = float(candidates[1][1] - best_cost)
    return True, best


def _fuse_primary_frame_mode(
    sync_index: int,
    prediction_rows: dict[str, dict[int, dict[str, Any]]],
    calibrations: dict[str, CameraCalibration],
    cameras: tuple[str, ...],
    preferred_anchors: tuple[str, str],
    args: argparse.Namespace,
) -> tuple[bool, dict[str, Any]]:
    groups = {
        camera: _groups(prediction_rows[camera][sync_index]) for camera in cameras
    }
    complete_by_camera = {
        camera: _complete_detections(
            groups[camera], args.max_anchor_detections,
            bool(getattr(args, "suppress_duplicate_detector_boxes", 0)),
            float(getattr(args, "duplicate_detector_iou", 0.95)),
            float(getattr(args, "duplicate_detector_containment", 0.95)),
        )
        for camera in cameras
    }
    preferred_pairs, fallback_pairs = _anchor_pair_tiers(
        cameras, complete_by_camera, preferred_anchors,
        getattr(args, "preferred_anchor_camera", None), 2,
    )
    if not preferred_pairs and not fallback_pairs:
        if bool(getattr(args, "allow_single_hand_anchors", 0)):
            return _fuse_single_hand_frame(
                sync_index, groups, complete_by_camera, calibrations,
                cameras, preferred_anchors, args,
            )
        return False, {
            "sync_index": sync_index,
            "reason": "no_two_camera_anchor_with_two_hands",
        }

    candidates = (
        _evaluate_anchor_pairs(
            preferred_pairs, complete_by_camera, groups, calibrations,
            cameras, preferred_anchors, args,
        )
        if preferred_pairs else []
    )
    usable = [
        candidate for candidate in candidates
        if not _quality_reasons(candidate["results"], args)
    ]
    usable = _prefer_usable_anchor_pair(
        usable, preferred_anchors,
        getattr(args, "preferred_anchor_camera", None),
    )
    if not usable and fallback_pairs:
        fallback_candidates = _evaluate_anchor_pairs(
            fallback_pairs, complete_by_camera, groups, calibrations,
            cameras, preferred_anchors, args,
        )
        candidates.extend(fallback_candidates)
        usable = [
            candidate for candidate in fallback_candidates
            if not _quality_reasons(candidate["results"], args)
        ]
    if (not usable and bool(getattr(args, "allow_single_hand_anchors", 0))
            and not bool(getattr(args, "allow_partial_hands", 0))):
        single_accepted, single_row = _fuse_single_hand_frame(
            sync_index, groups, complete_by_camera, calibrations,
            cameras, preferred_anchors, args,
        )
        if single_accepted:
            return True, single_row
    if not candidates:
        return False, {"sync_index": sync_index, "reason": "no_valid_anchor_assignment"}

    candidates.sort(key=lambda item: item["cost"])
    usable.sort(key=lambda item: item["cost"])
    best = usable[0] if usable else candidates[0]
    chosen_anchors = tuple(best["anchor_cameras"])
    quality_reasons = _quality_reasons(best["results"], args)
    valid_sides = [
        side for side, result in enumerate(best["results"])
        if int(result.get("valid_joints", 0)) >= int(args.min_valid_joints)
    ]
    if quality_reasons and not bool(getattr(args, "allow_partial_hands", 0)):
        return False, {
            "sync_index": sync_index,
            "reasons": sorted(set(quality_reasons)),
        }
    if not valid_sides:
        return False, {
            "sync_index": sync_index,
            "reasons": sorted(set(quality_reasons or ["too_few_valid_joints"])),
        }
    margin = usable[1]["cost"] - best["cost"] if len(usable) > 1 else None
    hands = []
    for side, result in enumerate(best["results"]):
        if side not in valid_sides:
            continue
        hand = _serialize_hand(
            result, best["selected"][side], calibrations, chosen_anchors
        )
        hand["association_error_px"] = best["association_errors"][side]
        if bool(getattr(args, "robust_view_consensus", 0)):
            hand["consensus_rejected_camera_ids"] = sorted(
                best.get("consensus_rejected", {}).get(side, [])
            )
            hand["consensus_view_diagnostics"] = best.get(
                "consensus_diagnostics", {}
            ).get(side, {})
        hand["fusion_mode"] = (
            "multiview" if len(best["selected"][side]) >= 3 else "stereo_anchor"
        )
        hands.append(hand)
    if not hands:
        return False, {
            "sync_index": sync_index,
            "reasons": sorted(set(quality_reasons or ["too_few_valid_joints"])),
        }
    active_cameras = sorted(set().union(*(hand["camera_ids"] for hand in hands)))
    return True, {
        "sync_index": sync_index,
        "anchor_cameras": list(chosen_anchors),
        "assignment_cost": best["cost"],
        "assignment_margin": margin,
        "active_cameras": active_cameras,
        "active_camera_count": len(active_cameras),
        "hands": hands,
        "partial_hands": len(hands) < 2,
    }


def _fuse_primary_frame(
    sync_index: int,
    prediction_rows: dict[str, dict[int, dict[str, Any]]],
    calibrations: dict[str, CameraCalibration],
    cameras: tuple[str, ...],
    preferred_anchors: tuple[str, str],
    args: argparse.Namespace,
) -> tuple[bool, dict[str, Any]]:
    """Fuse one frame, retrying only rejected strict frames when adaptive."""
    if args.detector_handedness != "adaptive":
        return _fuse_primary_frame_mode(
            sync_index, prediction_rows, calibrations, cameras,
            preferred_anchors, args,
        )
    strict_args = argparse.Namespace(**vars(args))
    strict_args.detector_handedness = "strict"
    accepted, strict_row = _fuse_primary_frame_mode(
        sync_index, prediction_rows, calibrations, cameras,
        preferred_anchors, strict_args,
    )
    if accepted:
        strict_row["handedness_mode"] = "strict"
        return True, strict_row
    ignore_args = argparse.Namespace(**vars(args))
    ignore_args.detector_handedness = "ignore"
    rescued, rescued_row = _fuse_primary_frame_mode(
        sync_index, prediction_rows, calibrations, cameras,
        preferred_anchors, ignore_args,
    )
    if rescued:
        rescued_row["handedness_mode"] = "ignore_fallback"
        rescued_row["strict_rejection"] = strict_row
        return True, rescued_row
    strict_row["handedness_mode"] = "adaptive_rejected"
    strict_row["fallback_attempted"] = True
    return False, strict_row


_FUSION_WORKER_CONTEXT: tuple[Any, ...] | None = None


def _initialize_fusion_worker(
    prediction_rows: dict[str, dict[int, dict[str, Any]]],
    calibrations: dict[str, CameraCalibration],
    cameras: tuple[str, ...],
    preferred_anchors: tuple[str, str],
    args: argparse.Namespace,
) -> None:
    global _FUSION_WORKER_CONTEXT
    _FUSION_WORKER_CONTEXT = (
        prediction_rows, calibrations, cameras, preferred_anchors, args,
    )


def _fuse_primary_frame_worker(sync_index: int) -> tuple[bool, dict[str, Any]]:
    if _FUSION_WORKER_CONTEXT is None:
        raise RuntimeError("fusion worker was not initialized")
    return _fuse_primary_frame(sync_index, *_FUSION_WORKER_CONTEXT)


def _primary_results(
    frame_ids: list[int],
    prediction_rows: dict[str, dict[int, dict[str, Any]]],
    calibrations: dict[str, CameraCalibration],
    cameras: tuple[str, ...],
    preferred_anchors: tuple[str, str],
    args: argparse.Namespace,
):
    if args.workers == 1:
        for sync_index in frame_ids:
            yield _fuse_primary_frame(
                sync_index, prediction_rows, calibrations, cameras,
                preferred_anchors, args,
            )
        return
    start_methods = multiprocessing.get_all_start_methods()
    method = "fork" if "fork" in start_methods else start_methods[0]
    with ProcessPoolExecutor(
        max_workers=args.workers,
        mp_context=multiprocessing.get_context(method),
        initializer=_initialize_fusion_worker,
        initargs=(prediction_rows, calibrations, cameras, preferred_anchors, args),
    ) as executor:
        chunksize = max(
            1, min(4, len(frame_ids) // max(args.workers * 4, 1))
        )
        yield from executor.map(
            _fuse_primary_frame_worker, frame_ids, chunksize=chunksize
        )


def _nearest_reference_row(
    references: list[dict[str, Any]],
    reference_ids: list[int],
    sync_index: int,
) -> tuple[dict[str, Any], int]:
    """Find the closest accepted frame without scanning the full sequence."""
    position = bisect_left(reference_ids, sync_index)
    candidates = []
    if position > 0:
        candidates.append(position - 1)
    if position < len(reference_ids):
        candidates.append(position)
    nearest_index = min(
        candidates,
        key=lambda index: (abs(reference_ids[index] - sync_index), index),
    )
    return references[nearest_index], abs(
        reference_ids[nearest_index] - sync_index
    )


def _stabilize_coverage_partitioned(
    rows: list[dict[str, Any]],
    calibrations: dict[str, CameraCalibration],
    *,
    preserve_full_hands: bool,
    **kwargs: Any,
) -> dict[str, Any]:
    """Keep weak single-hand additions from changing confirmed dual-hand tracks.

    The temporal solver estimates each side from its surrounding accepted
    rows.  Adding single-hand rows fills previously rejected frame gaps and
    can change the solver's branch for a good two-hand result.  Solve the two
    coverage classes separately while retaining their original frame IDs.
    """
    full = [row for row in rows if len(row.get("hands", [])) == 2]
    partial = [row for row in rows if len(row.get("hands", [])) == 1]
    if not preserve_full_hands or not full or not partial:
        return stabilize_joint_temporal_rows(rows, calibrations, **kwargs)
    full_summary = stabilize_joint_temporal_rows(full, calibrations, **kwargs)
    partial_summary = stabilize_joint_temporal_rows(partial, calibrations, **kwargs)
    summary = dict(full_summary)
    summary["coverage_partitioned"] = True
    summary["full_hand_row_count"] = len(full)
    summary["partial_hand_row_count"] = len(partial)
    summary["partial_hand_summary"] = partial_summary
    for key, value in partial_summary.items():
        if key.endswith("_count") and isinstance(value, (int, float)):
            if isinstance(summary.get(key), (int, float)):
                summary[key] += value
    return summary


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"output already exists: {args.output}")
    if args.workers < 1 or args.progress_interval < 1 or args.max_frames < 0:
        raise ValueError(
            "workers/progress-interval must be positive; max-frames must be non-negative"
        )
    if not 0.0 <= args.sam3_min_score <= 1.0:
        raise ValueError("--sam3-min-score must be in [0, 1]")
    if not 0.0 <= args.sam3_min_box_iou <= 1.0:
        raise ValueError("--sam3-min-box-iou must be in [0, 1]")
    if not 0.0 <= args.sam3_hard_joint_support <= 1.0:
        raise ValueError("--sam3-hard-joint-support must be in [0, 1]")
    if not 0.0 <= args.sam3_min_joint_support <= 1.0:
        raise ValueError("--sam3-min-joint-support must be in [0, 1]")
    if args.sam3_hard_joint_support > args.sam3_min_joint_support:
        raise ValueError(
            "--sam3-hard-joint-support cannot exceed --sam3-min-joint-support"
        )
    if not 0.0 <= args.duplicate_detector_iou <= 1.0:
        raise ValueError("--duplicate-detector-iou must be in [0, 1]")
    if not 0.0 <= args.duplicate_detector_containment <= 1.0:
        raise ValueError("--duplicate-detector-containment must be in [0, 1]")
    if args.temporal_3d_guard_radius < 1:
        raise ValueError("--temporal-3d-guard-radius must be positive")
    if args.temporal_3d_guard_threshold_m <= 0.0:
        raise ValueError("--temporal-3d-guard-threshold-m must be positive")
    if args.temporal_3d_guard_min_anchor_cameras < 2:
        raise ValueError("--temporal-3d-guard-min-anchor-cameras must be at least 2")
    if args.temporal_3d_guard_min_palm_inliers < 2:
        raise ValueError("--temporal-3d-guard-min-palm-inliers must be at least 2")
    if args.temporal_3d_guard_max_anchor_quality_px <= 0.0:
        raise ValueError("--temporal-3d-guard-max-anchor-quality-px must be positive")
    if args.temporal_3d_guard_max_anchor_speed_m <= 0.0:
        raise ValueError("--temporal-3d-guard-max-anchor-speed-m must be positive")
    if not 0.0 <= args.temporal_3d_guard_duplicate_iou <= 1.0:
        raise ValueError("--temporal-3d-guard-duplicate-iou must be in [0, 1]")
    if args.temporal_3d_guard_max_reprojection_median_px <= 0.0:
        raise ValueError(
            "--temporal-3d-guard-max-reprojection-median-px must be positive"
        )
    if args.temporal_3d_guard_max_reprojection_p95_px <= 0.0:
        raise ValueError(
            "--temporal-3d-guard-max-reprojection-p95-px must be positive"
        )
    if args.temporal_3d_guard_max_local_reprojection_ratio <= 0.0:
        raise ValueError(
            "--temporal-3d-guard-max-local-reprojection-ratio must be positive"
        )
    if args.joint_temporal_ray_prior_weight <= 0.0:
        raise ValueError("--joint-temporal-ray-prior-weight must be positive")
    if args.condition_hard_threshold <= 0.0:
        raise ValueError("--condition-hard-threshold must be positive")
    if not 0.0 <= args.joint_temporal_ray_blend <= 1.0:
        raise ValueError("--joint-temporal-ray-blend must be in [0, 1]")
    if args.joint_temporal_max_step_m <= 0.0:
        raise ValueError("--joint-temporal-max-step-m must be positive")
    if args.joint_temporal_max_radius_m <= 0.0:
        raise ValueError("--joint-temporal-max-radius-m must be positive")
    if args.joint_temporal_max_missing_gap < 0:
        raise ValueError("--joint-temporal-max-missing-gap must be non-negative")
    if args.joint_temporal_local_radius < 1:
        raise ValueError("--joint-temporal-local-radius must be positive")
    if args.joint_temporal_local_outlier_threshold_m <= 0.0:
        raise ValueError(
            "--joint-temporal-local-outlier-threshold-m must be positive"
        )
    if not 0.0 < args.joint_temporal_local_candidate_ratio <= 1.0:
        raise ValueError(
            "--joint-temporal-local-candidate-ratio must be in (0, 1]"
        )
    if args.joint_temporal_smoothing_sigma < 0.0:
        raise ValueError("--joint-temporal-smoothing-sigma must be non-negative")
    if args.joint_temporal_bone_iterations < 0:
        raise ValueError("--joint-temporal-bone-iterations must be non-negative")
    if not 0.0 <= args.joint_temporal_bone_strength <= 1.0:
        raise ValueError("--joint-temporal-bone-strength must be in [0, 1]")
    if not 0 <= args.joint_temporal_min_hand_joints <= 21:
        raise ValueError("--joint-temporal-min-hand-joints must be in [0, 21]")
    if args.joint_temporal_excessive_jump_threshold_m <= 0.0:
        raise ValueError(
            "--joint-temporal-excessive-jump-threshold-m must be positive"
        )
    if args.joint_temporal_view_wrist_jump_px <= 0.0:
        raise ValueError("--joint-temporal-view-wrist-jump-px must be positive")
    if args.joint_temporal_view_expanded_box_ratio < 1.0:
        raise ValueError("--joint-temporal-view-expanded-box-ratio must be at least 1")
    if args.joint_temporal_view_expanded_box_wrist_px <= 0.0:
        raise ValueError("--joint-temporal-view-expanded-box-wrist-px must be positive")
    if args.anatomy_protected_camera_shift_px <= 0.0:
        raise ValueError("--anatomy-protected-camera-shift-px must be positive")
    if args.post_temporal_branch_threshold_m <= 0.0:
        raise ValueError("--post-temporal-branch-threshold-m must be positive")
    if args.min_joint_inlier_views < 2:
        raise ValueError("--min-joint-inlier-views must be at least 2")
    anatomy_config = FusionAnatomyConfig(
        outlier_window=args.anatomy_outlier_window,
        outlier_distance_m=args.anatomy_outlier_distance_m,
        bone_outlier_relative=args.anatomy_bone_relative,
        smoothing_radius=args.anatomy_smoothing_radius,
        local_shape_strength=args.anatomy_shape_strength,
        reliable_adjustment_blend=args.anatomy_reliable_adjustment_blend,
        max_reliable_adjustment_m=args.anatomy_max_reliable_adjustment_m,
        max_reprojection_regression_px=(
            args.anatomy_max_reprojection_regression_px
        ),
        max_reprojection_shift_px=args.anatomy_max_reprojection_shift_px,
        temporal_radius=args.anatomy_temporal_radius,
        temporal_palm_residual_m=args.anatomy_temporal_palm_residual_m,
        temporal_palm_strong_residual_m=(
            args.anatomy_temporal_palm_strong_residual_m
        ),
        temporal_global_min_rejected_joints=(
            args.anatomy_temporal_global_min_rejected_joints
        ),
        temporal_local_residual=args.anatomy_temporal_local_residual,
        temporal_blend=args.anatomy_temporal_blend,
        temporal_max_global_adjustment_m=(
            args.anatomy_temporal_max_global_adjustment_m
        ),
        temporal_max_local_adjustment_m=(
            args.anatomy_temporal_max_local_adjustment_m
        ),
        temporal_spike_joint_step_m=args.anatomy_temporal_spike_joint_step_m,
        temporal_spike_palm_step_m=args.anatomy_temporal_spike_palm_step_m,
        temporal_spike_max_spread_m=args.anatomy_temporal_spike_max_spread_m,
        protected_camera_ids=tuple(args.anatomy_protected_cameras),
        protected_camera_shift_px=args.anatomy_protected_camera_shift_px,
    )
    anatomy_config.validate()
    fusion_started = time.perf_counter()
    dataset = args.dataset.resolve()
    prediction_root = args.predictions.resolve()
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    available_cameras = tuple(manifest["camera_ids"])
    cameras = tuple(dict.fromkeys(args.cameras or available_cameras))
    if len(cameras) < 2:
        raise ValueError("fusion requires at least two cameras")
    unknown_cameras = [camera for camera in cameras if camera not in available_cameras]
    if unknown_cameras:
        raise ValueError(f"selected cameras are not present in the dataset: {unknown_cameras}")
    unknown_protected = [
        camera for camera in args.anatomy_protected_cameras
        if camera not in cameras
    ]
    if unknown_protected:
        raise ValueError(
            "protected anatomy cameras must be selected fusion cameras: "
            f"{unknown_protected}"
        )
    preferred_anchors = tuple(args.anchor_cameras)
    if len(set(preferred_anchors)) != 2 or any(camera not in cameras for camera in preferred_anchors):
        raise ValueError(f"invalid anchor cameras: {preferred_anchors}")
    if args.preferred_anchor_camera is not None and args.preferred_anchor_camera not in cameras:
        raise ValueError(
            f"preferred anchor camera is not selected: {args.preferred_anchor_camera}"
        )
    calibrations = {
        camera: CameraCalibration.load(dataset / "calibration" / f"{camera}.json")
        for camera in cameras
    }
    prediction_rows = {
        camera: _load_jsonl(prediction_root / camera / "predictions.jsonl")
        for camera in cameras
    }
    frame_ids = sorted(set.intersection(*(set(rows) for rows in prediction_rows.values())))
    frame_ids = frame_ids[: args.max_frames or None]
    accepted = []
    rejected = []
    primary_started = time.perf_counter()
    results = _primary_results(
        frame_ids, prediction_rows, calibrations, cameras, preferred_anchors, args
    )
    for completed, (is_accepted, row) in enumerate(results, start=1):
        (accepted if is_accepted else rejected).append(row)
        if completed % args.progress_interval == 0 or completed == len(frame_ids):
            elapsed = time.perf_counter() - primary_started
            print(
                f"fusion: {completed}/{len(frame_ids)} | "
                f"{completed / max(elapsed, 1e-9):.2f} frames/s | "
                f"workers={args.workers}",
                flush=True,
            )
    primary_seconds = time.perf_counter() - primary_started
    primary_accepted_count = len(accepted)
    temporal_recovered = []
    if args.temporal_recovery_gap > 0 and accepted:
        references = sorted(accepted, key=lambda row: int(row["sync_index"]))
        reference_ids = [int(row["sync_index"]) for row in references]
        remaining_rejected = []
        for rejection in rejected:
            sync_index = int(rejection["sync_index"])
            reference, reference_distance = _nearest_reference_row(
                references, reference_ids, sync_index
            )
            if reference_distance > args.temporal_recovery_gap:
                remaining_rejected.append(rejection)
                continue
            groups = {
                camera: _groups(prediction_rows[camera][sync_index]) for camera in cameras
            }
            recovered = _recover_from_reference(
                sync_index, reference, groups, calibrations, cameras, args
            )
            if recovered is None:
                remaining_rejected.append(rejection)
            else:
                temporal_recovered.append(recovered)
        accepted.extend(temporal_recovered)
        accepted.sort(key=lambda row: int(row["sync_index"]))
        rejected = remaining_rejected
    identity_summary = None
    if args.temporal_hand_identity:
        # Correct whole-hand detector label exchanges before anatomy and
        # per-joint depth refinement; otherwise the temporal filter blends two
        # different physical hands into one apparently jumping skeleton.
        identity_rows = accepted
        if (args.allow_single_hand_anchors or args.allow_partial_hands) and args.detector_handedness == "strict":
            # A partial observation can attach to the opposite prior track
            # and then trigger a false whole-hand swap on the next complete
            # frame.  Detector handedness is strict for the single-hand
            # fallback; preserve it and track confirmed dual-hand rows on
            # their own timeline.
            identity_rows = [row for row in accepted if len(row.get("hands", [])) == 2]
        identity_summary = stabilize_hand_identities(
            identity_rows,
            max_frame_gap=args.temporal_hand_identity_max_gap,
            swap_margin=args.temporal_hand_identity_swap_margin,
            swap_min_cost=args.temporal_hand_identity_swap_min_cost,
        )
        if identity_rows is not accepted:
            identity_summary["coverage_partitioned"] = True
            identity_summary["partial_hand_row_count"] = len(accepted) - len(identity_rows)
    anatomy_summary = None
    if args.anatomy_refinement:
        anatomy_summary = refine_accepted_rows(
            accepted, len(cameras), anatomy_config, calibrations
        )
    # Anatomy interpolation is useful for ordinary brief occlusions, but it
    # must not revive joints that the calibrated-ray gate explicitly rejected.
    condition_gate_restored_count = _restore_condition_rejected_joints(
        accepted, args
    )
    temporal_3d_summary = None
    if args.temporal_3d_guard:
        temporal_3d_summary = guard_temporal_rows(
            accepted,
            radius=args.temporal_3d_guard_radius,
            replacement_threshold_m=args.temporal_3d_guard_threshold_m,
            min_anchor_cameras=args.temporal_3d_guard_min_anchor_cameras,
            min_anchor_palm_inliers=args.temporal_3d_guard_min_palm_inliers,
            max_anchor_reprojection_px=args.temporal_3d_guard_max_anchor_quality_px,
            max_anchor_speed_m_per_frame=args.temporal_3d_guard_max_anchor_speed_m,
            duplicate_iou=args.temporal_3d_guard_duplicate_iou,
            clear_unrecoverable_endpoint=bool(args.temporal_3d_guard_clear_endpoint),
            calibrations=calibrations,
            max_reprojection_median_px=(
                args.temporal_3d_guard_max_reprojection_median_px
            ),
            max_reprojection_p95_px=args.temporal_3d_guard_max_reprojection_p95_px,
            max_local_reprojection_ratio=(
                args.temporal_3d_guard_max_local_reprojection_ratio
            ),
        )
    joint_temporal_summary = None
    if args.joint_temporal_guard and args.post_temporal_joint_repass:
        joint_temporal_summary = _stabilize_coverage_partitioned(
            accepted,
            calibrations,
            preserve_full_hands=bool(args.allow_single_hand_anchors or args.allow_partial_hands),
            ray_prior_weight=args.joint_temporal_ray_prior_weight,
            ray_blend=args.joint_temporal_ray_blend,
            max_step_m=args.joint_temporal_max_step_m,
            max_radius_m=args.joint_temporal_max_radius_m,
            max_missing_gap=args.joint_temporal_max_missing_gap,
            local_radius=args.joint_temporal_local_radius,
            local_outlier_threshold_m=args.joint_temporal_local_outlier_threshold_m,
            local_candidate_ratio=args.joint_temporal_local_candidate_ratio,
            smoothing_sigma=args.joint_temporal_smoothing_sigma,
            bone_iterations=args.joint_temporal_bone_iterations,
            bone_strength=args.joint_temporal_bone_strength,
            max_reprojection_median_px=(
                args.temporal_3d_guard_max_reprojection_median_px
            ),
            max_reprojection_p95_px=args.temporal_3d_guard_max_reprojection_p95_px,
            min_hand_joints=args.joint_temporal_min_hand_joints,
            excessive_jump_threshold_m=args.joint_temporal_excessive_jump_threshold_m,
            view_wrist_jump_px=args.joint_temporal_view_wrist_jump_px,
            view_expanded_box_ratio=args.joint_temporal_view_expanded_box_ratio,
            view_expanded_box_wrist_px=args.joint_temporal_view_expanded_box_wrist_px,
        )
    post_temporal_target_guard = _post_temporal_target_camera_guard(
        accepted, calibrations, anatomy_config,
        branch_threshold_m=args.post_temporal_branch_threshold_m,
    )
    # The protected-camera pass intentionally runs after the first temporal
    # filter, because anatomy/refinement can otherwise move the target view.
    # That reprojection-limited update can still expose a short depth branch;
    # run the same zero-phase per-joint tracker once more so the final output
    # is continuous without holding a whole hand at one historical pose.
    post_temporal_joint_repass = None
    if args.joint_temporal_guard:
        post_temporal_joint_repass = _stabilize_coverage_partitioned(
            accepted,
            calibrations,
            preserve_full_hands=bool(args.allow_single_hand_anchors or args.allow_partial_hands),
            ray_prior_weight=args.joint_temporal_ray_prior_weight,
            ray_blend=args.joint_temporal_ray_blend,
            max_step_m=args.joint_temporal_max_step_m,
            max_radius_m=args.joint_temporal_max_radius_m,
            max_missing_gap=args.joint_temporal_max_missing_gap,
            local_radius=args.joint_temporal_local_radius,
            local_outlier_threshold_m=args.joint_temporal_local_outlier_threshold_m,
            local_candidate_ratio=args.joint_temporal_local_candidate_ratio,
            smoothing_sigma=args.joint_temporal_smoothing_sigma,
            bone_iterations=args.joint_temporal_bone_iterations,
            bone_strength=args.joint_temporal_bone_strength,
            max_reprojection_median_px=(
                args.temporal_3d_guard_max_reprojection_median_px
            ),
            max_reprojection_p95_px=args.temporal_3d_guard_max_reprojection_p95_px,
            min_hand_joints=args.joint_temporal_min_hand_joints,
            excessive_jump_threshold_m=args.joint_temporal_excessive_jump_threshold_m,
            view_wrist_jump_px=args.joint_temporal_view_wrist_jump_px,
            view_expanded_box_ratio=args.joint_temporal_view_expanded_box_ratio,
            view_expanded_box_wrist_px=args.joint_temporal_view_expanded_box_wrist_px,
        )
    reasons = Counter()
    for rejection in rejected:
        if "reason" in rejection:
            reasons[rejection["reason"]] += 1
        else:
            reasons.update(rejection.get("reasons", []))
    output = args.output.resolve()
    output.mkdir(parents=True)
    (output / "accepted.jsonl").write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in accepted),
        encoding="utf-8",
    )
    (output / "rejected.jsonl").write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rejected),
        encoding="utf-8",
    )
    hands = [hand for row in accepted for hand in row["hands"]]
    view_counts = [len(hand["camera_ids"]) for hand in hands]
    joint_support = [
        count for hand in hands for count in hand["inlier_view_counts"] if count >= 2
    ]
    multiview_hands = sum(hand["fusion_mode"] != "stereo_anchor" for hand in hands)
    anchor_pair_counts = Counter(
        "+".join(row["anchor_cameras"]) for row in accepted
    )
    camera_contribution_counts = Counter(
        camera
        for hand in hands for camera, view in hand["views"].items()
        if int(view.get("inlier_joint_count", 0)) > 0
    )
    consensus_rejected_counts = Counter(
        camera
        for hand in hands
        for camera in hand.get("consensus_rejected_camera_ids", [])
    )
    detector_handedness_mismatches = sum(
        1
        for hand in hands for view in hand["views"].values()
        if int(view.get("inlier_joint_count", 0)) > 0
        and view.get("detector_is_right") is not None
        and int(view["detector_is_right"]) != int(hand["side"])
    )
    handedness_fallback_count = sum(
        row.get("handedness_mode") == "ignore_fallback" for row in accepted
    )
    marker_assisted_views = [
        view
        for hand in hands for view in hand["views"].values()
        if view.get("marker_assist", {}).get("applied")
    ]
    marker_match_counts = [
        int(view["marker_assist"]["matched_marker_count"])
        for view in marker_assisted_views
    ]
    marker_evidence_views = [
        view
        for hand in hands for view in hand["views"].values()
        if view.get("marker_assist", {}).get("applied")
        or view.get("marker_assist", {}).get("evidence_only")
    ]
    marker_evidence_only_views = [
        view
        for view in marker_evidence_views
        if view.get("marker_assist", {}).get("evidence_only")
    ]
    sam3_views = [
        view
        for hand in hands for view in hand["views"].values()
        if view.get("sam3_assist", {}).get("enabled")
    ]
    sam3_matched_views = [
        view for view in sam3_views
        if view.get("sam3_assist", {}).get("matched")
    ]
    sam3_accepted_views = [
        view for view in sam3_views
        if view.get("sam3_assist", {}).get("accepted")
    ]
    sam3_rejected_views = [
        view for view in sam3_views
        if (
            view.get("sam3_assist", {}).get("matched")
            and not view.get("sam3_assist", {}).get("accepted")
        )
    ]
    sam3_corrected_views = [
        view for view in sam3_views
        if int(view.get("sam3_assist", {}).get("two_d_correction", {})
               .get("corrected_joint_count", 0)) > 0
    ]
    sam3_corrected_joints = sum(
        int(view.get("sam3_assist", {}).get("two_d_correction", {})
            .get("corrected_joint_count", 0))
        for view in sam3_views
    )
    condition_rejected_joints = sum(
        int(hand.get("quality", {}).get("condition_rejected_joint_count", 0))
        for hand in hands
    )
    stereo_cross_view = [
        hand["stereo_baseline_comparison"]["cross_view_reprojection_median_px"]
        for hand in hands
        if hand["stereo_baseline_comparison"].get("cross_view_reprojection_median_px") is not None
    ]
    multiview_cross_view = [
        hand["quality"]["all_view_reprojection_median_px"]
        for hand in hands
        if hand["stereo_baseline_comparison"].get("cross_view_reprojection_median_px") is not None
    ]
    differences_mm = [
        hand["stereo_baseline_comparison"]["multiview_3d_difference_median_mm"]
        for hand in hands
        if hand["stereo_baseline_comparison"].get("multiview_3d_difference_median_mm") is not None
    ]
    stereo_cross_median = float(np.median(stereo_cross_view)) if stereo_cross_view else None
    multiview_cross_median = float(np.median(multiview_cross_view)) if multiview_cross_view else None
    fusion_seconds = time.perf_counter() - fusion_started
    summary = {
        "schema_version": 1,
        "stage": (
            "wilor_anchor_guided_multiview_fusion_with_sam3_and_glove_marker_assist"
            if sam3_views and marker_assisted_views
            else "wilor_anchor_guided_multiview_fusion_with_sam3_assist"
            if sam3_views
            else "wilor_anchor_guided_multiview_fusion_with_glove_marker_assist"
            if marker_assisted_views
            else "wilor_anchor_guided_multiview_fusion"
        ),
        "camera_ids": list(cameras),
        "preferred_anchor_cameras": list(preferred_anchors),
        "preferred_anchor_camera": args.preferred_anchor_camera,
        "selected_anchor_pair_counts": dict(anchor_pair_counts),
        "camera_contribution_hand_counts": {
            camera: camera_contribution_counts.get(camera, 0) for camera in cameras
        },
        "consensus_rejected_view_count": int(sum(consensus_rejected_counts.values())),
        "consensus_rejected_camera_counts": {
            camera: consensus_rejected_counts.get(camera, 0) for camera in cameras
        },
        "detector_handedness_mismatch_observation_count": detector_handedness_mismatches,
        "marker_assisted_view_count": len(marker_assisted_views),
        "marker_evidence_view_count": len(marker_evidence_views),
        "marker_evidence_only_view_count": len(marker_evidence_only_views),
        "marker_matched_count_median": (
            float(np.median(marker_match_counts)) if marker_match_counts else None
        ),
        "sam3_gate_enabled": bool(args.sam3_gate),
        "sam3_view_count": len(sam3_views),
        "sam3_matched_view_count": len(sam3_matched_views),
        "sam3_accepted_view_count": len(sam3_accepted_views),
        "sam3_rejected_matched_view_count": len(sam3_rejected_views),
        "sam3_corrected_view_count": len(sam3_corrected_views),
        "sam3_corrected_joint_count": sam3_corrected_joints,
        "condition_rejected_joint_count": condition_rejected_joints,
        "condition_gate_restored_joint_count": int(condition_gate_restored_count),
        "processed_frame_count": len(frame_ids),
        "accepted_frame_count": len(accepted),
        "primary_accepted_frame_count": primary_accepted_count,
        "temporal_recovered_frame_count": len(temporal_recovered),
        "rejected_frame_count": len(rejected),
        "accepted_rate": len(accepted) / max(len(frame_ids), 1),
        "rejected_reason_counts": dict(reasons),
        "hand_count": len(hands),
        "multiview_hand_count": multiview_hands,
        "multiview_hand_rate": multiview_hands / max(len(hands), 1),
        "selected_view_count_median": float(np.median(view_counts)) if view_counts else None,
        "valid_joint_inlier_view_count_median": (
            float(np.median(joint_support)) if joint_support else None
        ),
        "valid_joint_inlier_view_count_distribution": {
            str(count): joint_support.count(count) for count in sorted(set(joint_support))
        },
        "reprojection_median_px": (
            float(np.median([hand["quality"]["multiview_reprojection_median_px"] for hand in hands]))
            if hands else None
        ),
        "fusion_workers": args.workers,
        "primary_fusion_seconds": primary_seconds,
        "fusion_seconds": fusion_seconds,
        "fusion_frames_per_second": len(frame_ids) / max(fusion_seconds, 1e-9),
        "stereo_anchor_comparison": {
            "comparable_hand_count": len(stereo_cross_view),
            "stereo_cross_view_median_px": stereo_cross_median,
            "multiview_cross_view_median_px": multiview_cross_median,
            "cross_view_median_improvement_percent": (
                100.0 * (1.0 - multiview_cross_median / stereo_cross_median)
                if stereo_cross_median and multiview_cross_median is not None else None
            ),
            "stereo_multiview_3d_difference_median_mm": (
                float(np.median(differences_mm)) if differences_mm else None
            ),
        },
        "parameters": {
            "association_threshold_px": args.association_threshold_px,
            "anchor_threshold_px": args.anchor_threshold_px,
            "ransac_threshold_px": args.ransac_threshold_px,
            "condition_hard_threshold": float(args.condition_hard_threshold),
            "min_joint_inlier_views": int(args.min_joint_inlier_views),
            "allow_stereo_final": bool(getattr(args, "allow_stereo_final", 0)),
            "allow_single_hand_anchors": bool(getattr(args, "allow_single_hand_anchors", 0)),
            "min_valid_joints": args.min_valid_joints,
            "temporal_recovery_gap": args.temporal_recovery_gap,
            "temporal_association_threshold_px": args.temporal_association_threshold_px,
            "anatomy_protected_cameras": list(args.anatomy_protected_cameras),
            "anatomy_protected_camera_shift_px": float(
                args.anatomy_protected_camera_shift_px
            ),
            "post_temporal_branch_threshold_m": float(
                args.post_temporal_branch_threshold_m
            ),
            "post_temporal_joint_repass": bool(args.post_temporal_joint_repass),
            "detector_handedness": args.detector_handedness,
            "suppress_duplicate_detector_boxes": bool(
                getattr(args, "suppress_duplicate_detector_boxes", 0)
            ),
            "duplicate_detector_iou": float(
                getattr(args, "duplicate_detector_iou", 0.95)
            ),
            "duplicate_detector_containment": float(
                getattr(args, "duplicate_detector_containment", 0.95)
            ),
            "temporal_3d_guard": bool(args.temporal_3d_guard),
            "temporal_3d_guard_radius": int(args.temporal_3d_guard_radius),
            "temporal_3d_guard_threshold_m": float(args.temporal_3d_guard_threshold_m),
            "temporal_3d_guard_min_anchor_cameras": int(
                args.temporal_3d_guard_min_anchor_cameras
            ),
            "temporal_3d_guard_min_palm_inliers": int(
                args.temporal_3d_guard_min_palm_inliers
            ),
            "temporal_3d_guard_max_anchor_quality_px": float(
                args.temporal_3d_guard_max_anchor_quality_px
            ),
            "temporal_3d_guard_max_anchor_speed_m": float(
                args.temporal_3d_guard_max_anchor_speed_m
            ),
            "temporal_3d_guard_duplicate_iou": float(
                args.temporal_3d_guard_duplicate_iou
            ),
            "temporal_3d_guard_max_reprojection_median_px": float(
                args.temporal_3d_guard_max_reprojection_median_px
            ),
            "temporal_3d_guard_max_reprojection_p95_px": float(
                args.temporal_3d_guard_max_reprojection_p95_px
            ),
            "temporal_3d_guard_max_local_reprojection_ratio": float(
                args.temporal_3d_guard_max_local_reprojection_ratio
            ),
            "temporal_3d_guard_clear_endpoint": bool(
                args.temporal_3d_guard_clear_endpoint
            ),
            "joint_temporal_guard": bool(args.joint_temporal_guard),
            "joint_temporal_ray_prior_weight": float(
                args.joint_temporal_ray_prior_weight
            ),
            "joint_temporal_ray_blend": float(args.joint_temporal_ray_blend),
            "joint_temporal_max_step_m": float(args.joint_temporal_max_step_m),
            "joint_temporal_max_radius_m": float(args.joint_temporal_max_radius_m),
            "joint_temporal_max_missing_gap": int(args.joint_temporal_max_missing_gap),
            "joint_temporal_local_radius": int(args.joint_temporal_local_radius),
            "joint_temporal_local_outlier_threshold_m": float(
                args.joint_temporal_local_outlier_threshold_m
            ),
            "joint_temporal_local_candidate_ratio": float(
                args.joint_temporal_local_candidate_ratio
            ),
            "joint_temporal_smoothing_sigma": float(
                args.joint_temporal_smoothing_sigma
            ),
            "joint_temporal_bone_iterations": int(
                args.joint_temporal_bone_iterations
            ),
            "joint_temporal_bone_strength": float(
                args.joint_temporal_bone_strength
            ),
            "joint_temporal_min_hand_joints": int(
                args.joint_temporal_min_hand_joints
            ),
            "joint_temporal_excessive_jump_threshold_m": float(
                args.joint_temporal_excessive_jump_threshold_m
            ),
            "joint_temporal_view_wrist_jump_px": float(
                args.joint_temporal_view_wrist_jump_px
            ),
            "joint_temporal_view_expanded_box_ratio": float(
                args.joint_temporal_view_expanded_box_ratio
            ),
            "joint_temporal_view_expanded_box_wrist_px": float(
                args.joint_temporal_view_expanded_box_wrist_px
            ),
            "temporal_hand_identity": bool(args.temporal_hand_identity),
            "temporal_hand_identity_max_gap": int(args.temporal_hand_identity_max_gap),
            "temporal_hand_identity_swap_margin": float(
                args.temporal_hand_identity_swap_margin
            ),
            "temporal_hand_identity_swap_min_cost": float(
                args.temporal_hand_identity_swap_min_cost
            ),
            "robust_view_consensus": bool(
                getattr(args, "robust_view_consensus", 0)
            ),
            "consensus_min_views": int(
                getattr(args, "consensus_min_views", 3)
            ),
            "consensus_min_inlier_joints": int(
                getattr(args, "consensus_min_inlier_joints", 8)
            ),
            "consensus_outlier_median_px": float(
                getattr(args, "consensus_outlier_median_px", 35.0)
            ),
            "consensus_outlier_ratio": float(
                getattr(args, "consensus_outlier_ratio", 2.5)
            ),
            "sam3_gate": bool(args.sam3_gate),
            "sam3_min_score": float(args.sam3_min_score),
            "sam3_min_box_iou": float(args.sam3_min_box_iou),
            "sam3_min_joint_support": float(args.sam3_min_joint_support),
            "sam3_hard_joint_support": float(args.sam3_hard_joint_support),
            "workers": args.workers,
        },
        "anatomy_refinement": anatomy_summary or {
            "enabled": False,
            "parameters": anatomy_config.to_dict(),
        },
        "temporal_3d_guard": temporal_3d_summary or {
            "enabled": False,
        },
        "joint_temporal_guard": joint_temporal_summary or {
            "enabled": False,
        },
        "post_temporal_target_camera_guard": post_temporal_target_guard,
        "post_temporal_joint_repass": post_temporal_joint_repass or {
            "enabled": False,
            "requested": bool(args.post_temporal_joint_repass),
        },
        "temporal_hand_identity": identity_summary or {
            "enabled": False,
        },
    }
    if args.detector_handedness == "adaptive":
        summary["handedness_fallback_frame_count"] = handedness_fallback_count
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
