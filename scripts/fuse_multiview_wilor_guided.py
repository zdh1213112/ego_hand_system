#!/usr/bin/env python3
"""Fuse WiLoR with a reliable stereo anchor and independently matched side views."""

from __future__ import annotations

import argparse
from collections import Counter
import itertools
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from camera_models import project_points  # noqa: E402
from ego_data.calibration import CameraCalibration  # noqa: E402
from fuse_multiview_wilor import (  # noqa: E402
    _groups, _load_jsonl, _serialize_hand, _triangulate_hand,
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
        "--detector-handedness", choices=("strict", "ignore"), default="strict",
        help="strict keeps detector.pt left/right identity through all fusion stages",
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
            args.anchor_threshold_px, 2,
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
            args.ransac_threshold_px, 2,
        )
        for side in (0, 1)
    ]
    missing = sum(21 - result["valid_joints"] for result in final_results)
    extra_joint_support = sum(
        int(np.maximum(result["inlier_counts"] - 2, 0).sum())
        for result in final_results
    )
    matched_side_views = sum(len(selected_by_side[side]) - 2 for side in (0, 1))
    cost = (
        100.0 * missing
        + sum(result["median_px"] + 0.2 * result["p95_px"] for result in final_results)
        + sum(_anatomy_penalty(result["points"]) for result in final_results)
        - 1.5 * extra_joint_support
        - 4.0 * matched_side_views
    )
    return {
        "cost": float(cost),
        "anchor_cameras": list(anchor_cameras),
        "selected": selected_by_side,
        "results": final_results,
        "association_errors": association_errors,
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
            args.ransac_threshold_px, 2,
        )
        for side in (0, 1)
    ]
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
def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"output already exists: {args.output}")
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
    reasons: Counter[str] = Counter()
    for sync_index in frame_ids:
        groups = {camera: _groups(prediction_rows[camera][sync_index]) for camera in cameras}
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
            reason = "no_two_camera_anchor_with_two_hands"
            rejected.append({"sync_index": sync_index, "reason": reason})
            reasons[reason] += 1
            continue
        def evaluate_pairs(anchor_pairs: list[tuple[str, str]]) -> list[dict[str, Any]]:
            result = []
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
                        result.append(candidate)
            return result

        candidates = evaluate_pairs([preferred_anchors]) if preferred_available else []
        usable = [candidate for candidate in candidates if not _quality_reasons(candidate["results"], args)]
        if not usable and fallback_pairs:
            fallback_candidates = evaluate_pairs(fallback_pairs)
            candidates.extend(fallback_candidates)
            usable = [
                candidate for candidate in fallback_candidates
                if not _quality_reasons(candidate["results"], args)
            ]
        if not candidates:
            reason = "no_valid_anchor_assignment"
            rejected.append({"sync_index": sync_index, "reason": reason})
            reasons[reason] += 1
            continue
        candidates.sort(key=lambda item: item["cost"])
        usable.sort(key=lambda item: item["cost"])
        best = usable[0] if usable else candidates[0]
        chosen_anchors = tuple(best["anchor_cameras"])
        quality_reasons = _quality_reasons(best["results"], args)
        if quality_reasons:
            quality_reasons = sorted(set(quality_reasons))
            rejected.append({"sync_index": sync_index, "reasons": quality_reasons})
            reasons.update(quality_reasons)
            continue
        margin = usable[1]["cost"] - best["cost"] if len(usable) > 1 else None
        hands = []
        for side, result in enumerate(best["results"]):
            hand = _serialize_hand(
                result, best["selected"][side], calibrations, chosen_anchors
            )
            hand["association_error_px"] = best["association_errors"][side]
            hand["fusion_mode"] = "multiview" if len(best["selected"][side]) >= 3 else "stereo_anchor"
            hands.append(hand)
        accepted.append({
            "sync_index": sync_index,
            "anchor_cameras": list(chosen_anchors),
            "assignment_cost": best["cost"],
            "assignment_margin": margin,
            "active_cameras": sorted(set().union(*(hand["camera_ids"] for hand in hands))),
            "active_camera_count": len(set().union(*(hand["camera_ids"] for hand in hands))),
            "hands": hands,
        })
    primary_accepted_count = len(accepted)
    temporal_recovered = []
    if args.temporal_recovery_gap > 0 and accepted:
        references = sorted(accepted, key=lambda row: int(row["sync_index"]))
        remaining_rejected = []
        for rejection in rejected:
            sync_index = int(rejection["sync_index"])
            reference = min(
                references, key=lambda row: abs(int(row["sync_index"]) - sync_index)
            )
            if abs(int(reference["sync_index"]) - sync_index) > args.temporal_recovery_gap:
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
    detector_handedness_mismatches = sum(
        1
        for hand in hands for view in hand["views"].values()
        if int(view.get("inlier_joint_count", 0)) > 0
        and view.get("detector_is_right") is not None
        and int(view["detector_is_right"]) != int(hand["side"])
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
    summary = {
        "schema_version": 1,
        "stage": "wilor_anchor_guided_multiview_fusion",
        "camera_ids": list(cameras),
        "preferred_anchor_cameras": list(preferred_anchors),
        "selected_anchor_pair_counts": dict(anchor_pair_counts),
        "camera_contribution_hand_counts": {
            camera: camera_contribution_counts.get(camera, 0) for camera in cameras
        },
        "detector_handedness_mismatch_observation_count": detector_handedness_mismatches,
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
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
