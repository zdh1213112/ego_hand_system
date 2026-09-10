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
    _groups, _load_jsonl, _serialize_hand, _triangulate_hand,
)
from fusion_anatomy_refinement import (  # noqa: E402
    FusionAnatomyConfig,
    refine_accepted_rows,
)


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
    parser.add_argument(
        "--cameras", nargs="+",
        help="camera subset used for fusion (default: all cameras in the dataset)",
    )
    parser.add_argument("--anchor-cameras", nargs=2, default=("camera2", "camera3"))
    parser.add_argument("--max-anchor-detections", type=int, default=3)
    parser.add_argument("--max-side-detections", type=int, default=4)
    parser.add_argument(
        "--detector-handedness", choices=("strict", "ignore", "adaptive"), default="strict",
        help=(
            "strict keeps detector.pt identity; ignore lets geometry choose; "
            "adaptive tries strict first and falls back to ignore for a rejected frame"
        ),
    )
    parser.add_argument("--association-threshold-px", type=float, default=55.0)
    parser.add_argument("--anchor-threshold-px", type=float, default=60.0)
    parser.add_argument("--ransac-threshold-px", type=float, default=20.0)
    parser.add_argument("--min-valid-joints", type=int, default=12)
    parser.add_argument("--max-reprojection-median-px", type=float, default=15.0)
    parser.add_argument("--max-reprojection-p95-px", type=float, default=40.0)
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
        "--workers", type=int, default=1,
        help="independent frame-fusion worker processes",
    )
    parser.add_argument("--progress-interval", type=int, default=25)
    parser.add_argument("--max-frames", type=int, default=0)
    return parser.parse_args()


def _complete_detections(
    groups: dict[int, dict[int, dict[str, Any]]], limit: int,
) -> list[int]:
    detections = [index for index, sides in groups.items() if 0 in sides and 1 in sides]
    detections.sort(
        key=lambda index: float(groups[index][0].get("confidence", 0.0)), reverse=True
    )
    return detections[:limit]


def _detector_side(hand: dict[str, Any]) -> int | None:
    value = hand.get("detector_is_right")
    return None if value is None else int(value)


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
) -> tuple[dict[int, dict[str, Any]], dict[int, float]]:
    detections = _complete_detections(groups, max_detections)
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
                args.ransac_threshold_px, 2, bool(getattr(args, "sam3_gate", 0)),
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
            args.ransac_threshold_px, 2, bool(getattr(args, "sam3_gate", 0)),
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
        )
        for side, hand in matched.items():
            selected_by_side[side][camera] = hand
            association_errors[side][camera] = errors[side]
    final_results = [
        _triangulate_hand(
            side, selected_by_side[side], calibrations,
            args.ransac_threshold_px, 2, bool(getattr(args, "sam3_gate", 0)),
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
        )
        for side, hand in matched.items():
            selected_by_side[side][camera] = hand
            association_errors[side][camera] = errors[side]
    if any(len(selected_by_side[side]) < 2 for side in (0, 1)):
        return None
    results = [
        _triangulate_hand(
            side, selected_by_side[side], calibrations,
            args.ransac_threshold_px, 2, bool(getattr(args, "sam3_gate", 0)),
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
                missing_preferred = len(set(preferred_anchors) - set(anchor_pair))
                candidate["cost"] += 5.0 * missing_preferred
                results.append(candidate)
    return results


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
        camera: _complete_detections(groups[camera], args.max_anchor_detections)
        for camera in cameras
    }
    preferred_available = all(
        len(complete_by_camera[camera]) >= 2 for camera in preferred_anchors
    )
    fallback_pairs = [
        pair for pair in itertools.combinations(cameras, 2)
        if pair != preferred_anchors
        and any(camera in preferred_anchors for camera in pair)
        and all(len(complete_by_camera[camera]) >= 2 for camera in pair)
    ]
    if not preferred_available and not fallback_pairs:
        return False, {
            "sync_index": sync_index,
            "reason": "no_two_camera_anchor_with_two_hands",
        }

    candidates = (
        _evaluate_anchor_pairs(
            [preferred_anchors], complete_by_camera, groups, calibrations,
            cameras, preferred_anchors, args,
        )
        if preferred_available else []
    )
    usable = [
        candidate for candidate in candidates
        if not _quality_reasons(candidate["results"], args)
    ]
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
    if not candidates:
        return False, {"sync_index": sync_index, "reason": "no_valid_anchor_assignment"}

    candidates.sort(key=lambda item: item["cost"])
    usable.sort(key=lambda item: item["cost"])
    best = usable[0] if usable else candidates[0]
    chosen_anchors = tuple(best["anchor_cameras"])
    quality_reasons = _quality_reasons(best["results"], args)
    if quality_reasons:
        return False, {
            "sync_index": sync_index,
            "reasons": sorted(set(quality_reasons)),
        }
    margin = usable[1]["cost"] - best["cost"] if len(usable) > 1 else None
    hands = []
    for side, result in enumerate(best["results"]):
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
    active_cameras = sorted(set().union(*(hand["camera_ids"] for hand in hands)))
    return True, {
        "sync_index": sync_index,
        "anchor_cameras": list(chosen_anchors),
        "assignment_cost": best["cost"],
        "assignment_margin": margin,
        "active_cameras": active_cameras,
        "active_camera_count": len(active_cameras),
        "hands": hands,
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
    preferred_anchors = tuple(args.anchor_cameras)
    if len(set(preferred_anchors)) != 2 or any(camera not in cameras for camera in preferred_anchors):
        raise ValueError(f"invalid anchor cameras: {preferred_anchors}")
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
    anatomy_summary = None
    if args.anatomy_refinement:
        anatomy_summary = refine_accepted_rows(
            accepted, len(cameras), anatomy_config, calibrations
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
            "min_valid_joints": args.min_valid_joints,
            "temporal_recovery_gap": args.temporal_recovery_gap,
            "temporal_association_threshold_px": args.temporal_association_threshold_px,
            "detector_handedness": args.detector_handedness,
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
