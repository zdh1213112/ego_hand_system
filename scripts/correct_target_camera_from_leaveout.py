#!/usr/bin/env python3
"""Correct one camera's 2-D WiLoR joints from a leave-one-view-out fusion.

This is an offline A/B tool.  It keeps every source prediction untouched,
uses a fusion made without ``--target-camera`` as the reference, projects the
five-view 3-D hand back into the target camera, and changes only target-camera
joints whose residual is large enough and whose leave-out result has support
from several other views.
"""

from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import shutil
import sys
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from camera_models import project_points  # noqa: E402
from ego_data.calibration import CameraCalibration  # noqa: E402
from fuse_multiview_wilor import _load_jsonl  # noqa: E402


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
    parser.add_argument("--leaveout-fusion", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--target-camera", default="camera3")
    parser.add_argument(
        "--source-cameras", nargs="+",
        default=["camera0", "camera1", "camera2", "camera4", "camera5"],
    )
    parser.add_argument("--min-source-views", type=int, default=3)
    parser.add_argument("--min-valid-joints", type=int, default=12)
    parser.add_argument("--association-threshold-px", type=float, default=70.0)
    parser.add_argument("--correction-threshold-px", type=float, default=18.0)
    parser.add_argument("--wrist-correction-threshold-px", type=float, default=30.0)
    parser.add_argument("--replacement-threshold-px", type=float, default=70.0)
    parser.add_argument("--blend", type=float, default=0.80)
    parser.add_argument("--max-shift-px", type=float, default=80.0)
    parser.add_argument("--max-bone-change-ratio", type=float, default=0.55)
    parser.add_argument(
        "--allow-handedness-relabel", type=int, choices=(0, 1), default=0,
        help=(
            "allow the target-camera detection's alternate WiLoR hand hypothesis to "
            "replace the detector-labelled hypothesis when leave-out geometry "
            "is substantially better"
        ),
    )
    parser.add_argument(
        "--handedness-relabel-margin-px", type=float, default=20.0,
        help="minimum leave-out residual improvement required for relabeling",
    )
    parser.add_argument(
        "--temporal-guard", type=int, choices=(0, 1), default=0,
        help="reject a cross-view correction that conflicts with a stable local target track",
    )
    parser.add_argument("--temporal-radius", type=int, default=2)
    parser.add_argument("--temporal-gate-px", type=float, default=35.0)
    parser.add_argument("--temporal-margin-px", type=float, default=15.0)
    parser.add_argument("--min-raw-temporal-residual-px", type=float, default=20.0)
    parser.add_argument("--max-frames", type=int, default=0)
    return parser.parse_args()


def _project(
    calibration: CameraCalibration, points_base: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points_base, dtype=np.float64)
    rotation = calibration.T_base_camera[:3, :3]
    center = calibration.T_base_camera[:3, 3]
    points_camera = (rotation.T @ (points - center).T).T
    pixels, valid = project_points(calibration, points_camera)
    pixels = np.asarray(pixels, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    pixels[~valid] = np.nan
    return pixels, valid


def _finite_points(points: Any) -> np.ndarray:
    value = np.asarray(points, dtype=np.float64)
    return value if value.shape == (21, 2) else np.full((21, 2), np.nan)


def _finite_base_points(points: Any) -> np.ndarray:
    value = np.asarray(points, dtype=np.float64)
    return value if value.shape == (21, 3) else np.full((21, 3), np.nan)


def _candidate_error(
    observed: np.ndarray, projected: np.ndarray, valid: np.ndarray,
) -> tuple[float, int]:
    usable = valid & np.isfinite(observed).all(axis=1) & np.isfinite(projected).all(axis=1)
    count = int(np.count_nonzero(usable))
    if count == 0:
        return float("inf"), 0
    errors = np.linalg.norm(observed[usable] - projected[usable], axis=1)
    return float(np.median(errors)), count


def _select_target_hand(
    target_record: dict[str, Any], side: int, projected: np.ndarray, valid: np.ndarray,
    association_threshold: float, min_valid_joints: int,
) -> tuple[dict[str, Any] | None, float, int]:
    candidates = [
        hand for hand in target_record.get("hands", [])
        if hand.get("is_right") is not None and int(hand.get("is_right")) == side
    ]
    best: tuple[float, float, int, dict[str, Any]] | None = None
    for hand in candidates:
        observed = _finite_points(hand.get("joints_2d", []))
        median, count = _candidate_error(observed, projected, valid)
        if count < min_valid_joints or median > association_threshold:
            continue
        # Compare only numeric quality fields.  Keeping the hand dictionary
        # out of the tuple is important because dictionaries are not ordered
        # and cannot be used as a tie-breaker by Python.
        key = (median, -float(hand.get("confidence", 0.0)))
        if best is None or key < best[:2]:
            best = (median, key[1], count, hand)
    if best is None:
        return None, float("inf"), 0
    return best[3], best[0], best[2]


def _maybe_relabel_target_hypothesis(
    target_record: dict[str, Any], side: int, target_hand: dict[str, Any],
    projected: np.ndarray, valid: np.ndarray, association_threshold: float,
    min_valid_joints: int, margin_px: float,
) -> tuple[bool, float, int, dict[str, Any] | None]:
    """Use an alternate WiLoR pose only when calibrated geometry supports it.

    WiLoR emits two left/right hypotheses for one detector box.  The detector
    side is normally a useful prior, but a mislabeled box can leave the
    geometrically correct hypothesis inaccessible to strict fusion.  This
    helper keeps the target dictionary's side key stable and copies only the
    alternate 2-D pose (and its SAM3 support) into that slot.  The original
    ``wilor_joints_2d`` remains untouched for an A/B comparison.
    """
    detection_index = target_hand.get("detection_index")
    if detection_index is None:
        return False, float("inf"), 0, None
    current_observed = _finite_points(target_hand.get("joints_2d", []))
    current_error, current_count = _candidate_error(
        current_observed, projected, valid
    )
    alternates = [
        hand for hand in target_record.get("hands", [])
        if hand is not target_hand
        and hand.get("detection_index") == detection_index
        and hand.get("is_right") is not None
        and int(hand.get("is_right")) != side
    ]
    best_alternate: dict[str, Any] | None = None
    best_error = float("inf")
    best_count = 0
    for alternate in alternates:
        error, count = _candidate_error(
            _finite_points(alternate.get("joints_2d", [])), projected, valid
        )
        if count >= min_valid_joints and error < best_error:
            best_alternate = alternate
            best_error = error
            best_count = count
    if best_alternate is None:
        return False, current_error, current_count, None
    if best_error > association_threshold:
        return False, current_error, current_count, None
    if not np.isfinite(current_error) or current_error - best_error < margin_px:
        return False, current_error, current_count, None

    target_hand["joints_2d"] = copy.deepcopy(best_alternate.get("joints_2d", []))
    if "sam3_assist" in best_alternate:
        target_hand["sam3_assist"] = copy.deepcopy(best_alternate["sam3_assist"])
    # Keep ``is_right`` equal to the leave-out physical side.  Strict fusion
    # then accepts this corrected slot, while the original detector label is
    # retained in the provenance record below.
    target_hand["detector_is_right"] = int(side)
    target_hand["target_camera_hypothesis_source"] = {
        "source_hypothesis_index": best_alternate.get("hypothesis_index"),
        "source_is_right": int(best_alternate.get("is_right")),
        "source_detector_is_right": best_alternate.get("detector_is_right"),
        "residual_before_px": float(current_error),
        "residual_alternate_px": float(best_error),
    }
    return True, best_error, best_count, best_alternate


def _temporal_residual(
    points: np.ndarray, temporal_median: np.ndarray, joint: int,
) -> float:
    if joint == 0:
        return float(np.linalg.norm(points[joint] - temporal_median[joint]))
    return float(
        np.linalg.norm(
            (points[joint] - points[0])
            - (temporal_median[joint] - temporal_median[0])
        )
    )


def _apply_temporal_guard(
    observed: np.ndarray, updated: np.ndarray,
    temporal_median: np.ndarray | None, temporal_gate_px: float,
    temporal_margin_px: float, min_raw_temporal_residual_px: float,
) -> tuple[np.ndarray, int]:
    """Keep a correction only when it does not contradict a stable local track."""
    if temporal_median is None:
        return updated, 0
    output = updated.copy()
    rejected = 0
    for joint in range(21):
        if not (
            np.isfinite(observed[joint]).all()
            and np.isfinite(output[joint]).all()
            and np.isfinite(temporal_median[joint]).all()
            and np.isfinite(temporal_median[0]).all()
        ):
            continue
        if np.linalg.norm(output[joint] - observed[joint]) <= 1e-6:
            continue
        raw_residual = _temporal_residual(observed, temporal_median, joint)
        candidate_residual = _temporal_residual(output, temporal_median, joint)
        # If raw WiLoR already follows nearby frames, do not replace it with
        # an isolated leave-out vote.  Otherwise reject only a clear temporal
        # discontinuity, leaving legitimate fast motion eligible.
        reject = raw_residual < min_raw_temporal_residual_px
        reject |= (
            raw_residual < temporal_gate_px
            and candidate_residual
            > max(temporal_gate_px, raw_residual + temporal_margin_px)
        )
        if reject:
            output[joint] = observed[joint]
            rejected += 1
    return output, rejected


def _propose_correction(
    observed: np.ndarray,
    projected: np.ndarray,
    valid: np.ndarray,
    inlier_counts: np.ndarray,
    min_source_views: int,
    correction_threshold: float,
    wrist_threshold: float,
    replacement_threshold: float,
    blend: float,
    max_shift: float,
    max_bone_change_ratio: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    output = observed.copy()
    finite = (
        valid
        & np.isfinite(observed).all(axis=1)
        & np.isfinite(projected).all(axis=1)
        & (inlier_counts >= min_source_views)
    )
    errors = np.full(21, np.nan, dtype=np.float64)
    errors[finite] = np.linalg.norm(observed[finite] - projected[finite], axis=1)
    proposals: dict[int, np.ndarray] = {}
    for joint in np.flatnonzero(finite):
        threshold = wrist_threshold if int(joint) == 0 else correction_threshold
        error = float(errors[joint])
        if error < threshold:
            continue
        delta = projected[joint] - observed[joint]
        shift = float(np.linalg.norm(delta))
        if shift <= 1e-6:
            continue
        if shift > max_shift:
            delta = delta * (max_shift / shift)
        # Apply the cap to both the blended and the strong-replacement path.
        # The previous implementation used the uncapped projected point for
        # replacement, allowing a single bad 3-D reference to move a 2-D
        # joint by hundreds of pixels.
        capped_projection = observed[joint] + delta
        proposal = (
            capped_projection
            if error >= replacement_threshold
            else observed[joint] + float(np.clip(blend, 0.0, 1.0)) * delta
        )
        proposals[int(joint)] = proposal

    trial = output.copy()
    for joint, proposal in proposals.items():
        trial[joint] = proposal

    rejected: list[int] = []
    for joint in list(proposals):
        okay = True
        for start, end in HAND_EDGES:
            if joint not in (start, end):
                continue
            raw_start, raw_end = observed[start], observed[end]
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
                okay = False
                break
        if okay:
            output[joint] = trial[joint]
        else:
            rejected.append(joint)
            trial[joint] = output[joint]

    changed = [joint for joint in proposals if joint not in rejected and np.linalg.norm(output[joint] - observed[joint]) > 1e-6]
    after_errors = np.full(21, np.nan, dtype=np.float64)
    after_errors[finite] = np.linalg.norm(output[finite] - projected[finite], axis=1)
    return output, {
        "enabled": True,
        "corrected_joint_count": len(changed),
        "corrected_joint_flags": [index in changed for index in range(21)],
        "rejected_joint_flags": [index in rejected for index in range(21)],
        "residual_before_median_px": float(np.nanmedian(errors)) if np.any(np.isfinite(errors)) else None,
        "residual_after_median_px": float(np.nanmedian(after_errors)) if np.any(np.isfinite(after_errors)) else None,
        "max_shift_px": float(max((np.linalg.norm(output[j] - observed[j]) for j in changed), default=0.0)),
        "source_inlier_joint_count": int(np.count_nonzero(finite)),
    }


def main() -> int:
    args = parse_args()
    if (
        args.max_frames < 0
        or args.min_source_views < 2
        or args.handedness_relabel_margin_px < 0
        or args.temporal_radius < 0
        or args.temporal_gate_px <= 0
        or args.temporal_margin_px < 0
        or args.min_raw_temporal_residual_px < 0
    ):
        raise ValueError("invalid max-frames/min-source-views")
    for path in (args.dataset, args.predictions, args.leaveout_fusion):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {args.output}")

    manifest = json.loads((args.dataset / "manifest.json").read_text(encoding="utf-8"))
    cameras = tuple(manifest["camera_ids"])
    if args.target_camera not in cameras:
        raise ValueError(f"target camera not in dataset: {args.target_camera}")
    source_cameras = tuple(camera for camera in args.source_cameras if camera != args.target_camera)
    if len(source_cameras) < args.min_source_views:
        raise ValueError("not enough source cameras")

    prediction_rows = {
        camera: _load_jsonl(args.predictions / camera / "predictions.jsonl")
        for camera in cameras
    }
    target_rows = prediction_rows[args.target_camera]
    leaveout_rows = _load_jsonl(args.leaveout_fusion / "accepted.jsonl")
    frame_ids = sorted(set(target_rows) & set(leaveout_rows))
    frame_ids = frame_ids[: args.max_frames or None]
    calibration = CameraCalibration.load(
        args.dataset / "calibration" / f"{args.target_camera}.json"
    )

    corrected_rows: dict[int, dict[str, Any]] = {}
    stats = Counter()
    residual_before: list[float] = []
    residual_after: list[float] = []
    corrected_joint_counts: Counter[int] = Counter()
    correction_contexts: list[dict[str, Any]] = []
    for frame_ordinal, sync_index in enumerate(frame_ids):
        record = copy.deepcopy(target_rows[sync_index])
        reference = leaveout_rows[sync_index]
        stats["frames_with_reference"] += 1
        for fused_hand in reference.get("hands", []):
            side = int(fused_hand.get("side", -1))
            if side not in (0, 1):
                continue
            source_hand_count = len(fused_hand.get("camera_ids", []))
            if source_hand_count < args.min_source_views:
                stats["hands_skipped_few_source_views"] += 1
                continue
            base_points = _finite_base_points(fused_hand.get("joints_base_m", []))
            projected, valid = _project(calibration, base_points)
            inlier_counts = np.asarray(
                fused_hand.get("inlier_view_counts", []), dtype=np.int32
            )
            if inlier_counts.shape != (21,):
                inlier_counts = np.zeros(21, dtype=np.int32)
            selection_threshold = (
                float("inf")
                if args.allow_handedness_relabel
                else args.association_threshold_px
            )
            target_hand, association_error, association_count = _select_target_hand(
                record, side, projected, valid,
                selection_threshold, args.min_valid_joints,
            )
            if target_hand is None:
                stats["hands_without_target_match"] += 1
                continue
            relabelled = False
            if args.allow_handedness_relabel:
                (
                    relabelled,
                    association_error,
                    association_count,
                    _,
                ) = _maybe_relabel_target_hypothesis(
                    record, side, target_hand, projected, valid,
                    args.association_threshold_px, args.min_valid_joints,
                    args.handedness_relabel_margin_px,
                )
                if not relabelled and association_error > args.association_threshold_px:
                    stats["hands_without_target_match"] += 1
                    continue
                if relabelled:
                    stats["hands_relabelled"] += 1
            stats["hands_matched"] += 1
            observed = _finite_points(target_hand.get("joints_2d", []))
            correction_contexts.append({
                "sync_index": int(sync_index),
                "ordinal": int(frame_ordinal),
                "side": side,
                "record": record,
                "target_hand": target_hand,
                "observed": observed,
                "projected": projected,
                "valid": valid,
                "inlier_counts": inlier_counts,
                "association_error": association_error,
                "association_count": association_count,
                "source_fusion_cameras": list(fused_hand.get("camera_ids", [])),
                "relabelled": relabelled,
            })
        corrected_rows[int(sync_index)] = record

    # Build local raw tracks before changing any 2-D points.  The temporal
    # guard is intentionally side-based: the leave-out association above has
    # already selected the physical hand, while detection indices can change
    # during occlusion.
    raw_tracks: dict[int, list[tuple[int, np.ndarray]]] = {0: [], 1: []}
    for context in correction_contexts:
        raw_tracks[int(context["side"])].append(
            (int(context["ordinal"]), context["observed"])
        )
    for context in correction_contexts:
        observed = context["observed"]
        updated, correction = _propose_correction(
            observed, context["projected"], context["valid"],
            context["inlier_counts"], args.min_source_views,
            args.correction_threshold_px, args.wrist_correction_threshold_px,
            args.replacement_threshold_px, args.blend, args.max_shift_px,
            args.max_bone_change_ratio,
        )
        temporal_median = None
        if args.temporal_guard and args.temporal_radius > 0:
            nearby = [
                points for ordinal, points in raw_tracks[int(context["side"])]
                if ordinal != int(context["ordinal"])
                and abs(ordinal - int(context["ordinal"])) <= args.temporal_radius
            ]
            if len(nearby) >= 2:
                temporal_median = np.median(np.stack(nearby), axis=0)
        proposed = updated.copy()
        updated, temporal_rejected = _apply_temporal_guard(
            observed, updated, temporal_median, args.temporal_gate_px,
            args.temporal_margin_px, args.min_raw_temporal_residual_px,
        )
        if temporal_rejected:
            stats["temporal_guard_rejected_joints"] += temporal_rejected
            stats["temporal_guard_rejected_hands"] += 1
        finite = (
            context["valid"]
            & np.isfinite(observed).all(axis=1)
            & np.isfinite(context["projected"]).all(axis=1)
            & (context["inlier_counts"] >= args.min_source_views)
        )
        after_errors = np.full(21, np.nan, dtype=np.float64)
        after_errors[finite] = np.linalg.norm(
            updated[finite] - context["projected"][finite], axis=1
        )
        changed = [
            joint for joint in range(21)
            if np.isfinite(updated[joint]).all()
            and np.isfinite(observed[joint]).all()
            and np.linalg.norm(updated[joint] - observed[joint]) > 1e-6
        ]
        rejected_temporal_flags = [
            bool(
                np.isfinite(proposed[joint]).all()
                and np.isfinite(observed[joint]).all()
                and np.linalg.norm(proposed[joint] - observed[joint]) > 1e-6
                and np.linalg.norm(updated[joint] - observed[joint]) <= 1e-6
            )
            for joint in range(21)
        ]
        correction["corrected_joint_count"] = len(changed)
        correction["corrected_joint_flags"] = [joint in changed for joint in range(21)]
        correction["temporal_rejected_joint_flags"] = rejected_temporal_flags
        correction["temporal_rejected_joint_count"] = int(temporal_rejected)
        correction["max_shift_px"] = float(
            max(
                (np.linalg.norm(updated[joint] - observed[joint]) for joint in changed),
                default=0.0,
            )
        )
        correction["residual_after_median_px"] = (
            float(np.nanmedian(after_errors)) if np.any(np.isfinite(after_errors)) else None
        )
        correction.update({
            "target_camera": args.target_camera,
            "side": int(context["side"]),
            "source_cameras": list(source_cameras),
            "source_fusion_cameras": context["source_fusion_cameras"],
            "association_error_median_px": context["association_error"],
            "association_joint_count": context["association_count"],
            "reference_sync_index": int(context["sync_index"]),
            "handedness_relabelled": bool(context["relabelled"]),
            "temporal_guard_enabled": bool(args.temporal_guard),
        })
        target_hand = context["target_hand"]
        target_hand["target_camera_leaveout_correction"] = correction
        before = correction.get("residual_before_median_px")
        after = correction.get("residual_after_median_px")
        if before is not None:
            residual_before.append(float(before))
        if after is not None:
            residual_after.append(float(after))
        count = len(changed)
        corrected_joint_counts[count] += 1
        if count:
            target_hand["joints_2d"] = updated.tolist()
            stats["hands_corrected"] += 1
            stats["corrected_joints"] += count
        else:
            stats["hands_unchanged"] += 1

    # Preserve all non-target cameras byte-for-byte and write the corrected
    # target camera as a normal prediction stage consumable by the fusion code.
    args.output.mkdir(parents=True)
    for camera in cameras:
        destination = args.output / camera
        destination.mkdir()
        source_path = args.predictions / camera / "predictions.jsonl"
        output_path = destination / "predictions.jsonl"
        if camera != args.target_camera:
            shutil.copyfile(source_path, output_path)
            summary_source = args.predictions / camera / "summary.json"
            if summary_source.is_file():
                shutil.copyfile(summary_source, destination / "summary.json")
            continue
        source_records = [
            corrected_rows.get(sync, target_rows[sync])
            for sync in sorted(target_rows)
        ]
        output_path.write_text(
            "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in source_records),
            encoding="utf-8",
        )
        target_summary = {
            "schema_version": 1,
            "stage": "wilor_target_camera_leaveout_2d_correction",
            "source_predictions": str(args.predictions),
            "leaveout_fusion": str(args.leaveout_fusion),
            "target_camera": args.target_camera,
            "source_cameras": list(source_cameras),
            "frame_count": len(target_rows),
            "tested_frame_count": len(frame_ids),
            "parameters": {
                "min_source_views": args.min_source_views,
                "min_valid_joints": args.min_valid_joints,
                "association_threshold_px": args.association_threshold_px,
                "correction_threshold_px": args.correction_threshold_px,
                "wrist_correction_threshold_px": args.wrist_correction_threshold_px,
                "replacement_threshold_px": args.replacement_threshold_px,
                "blend": args.blend,
                "max_shift_px": args.max_shift_px,
                "max_bone_change_ratio": args.max_bone_change_ratio,
                "allow_handedness_relabel": bool(args.allow_handedness_relabel),
                "handedness_relabel_margin_px": args.handedness_relabel_margin_px,
                "temporal_guard": bool(args.temporal_guard),
                "temporal_radius": args.temporal_radius,
                "temporal_gate_px": args.temporal_gate_px,
                "temporal_margin_px": args.temporal_margin_px,
                "min_raw_temporal_residual_px": args.min_raw_temporal_residual_px,
            },
            "stats": dict(stats),
            "corrected_joint_count_distribution": {
                str(key): value for key, value in sorted(corrected_joint_counts.items())
            },
            "residual_before_median_px": (
                float(np.median(residual_before)) if residual_before else None
            ),
            "residual_after_median_px": (
                float(np.median(residual_after)) if residual_after else None
            ),
        }
        (destination / "summary.json").write_text(
            json.dumps(target_summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    summary = {
        "schema_version": 1,
        "stage": "wilor_target_camera_leaveout_2d_correction",
        "target_camera": args.target_camera,
        "source_cameras": list(source_cameras),
        "frame_count": len(target_rows),
        "tested_frame_count": len(frame_ids),
        "stats": dict(stats),
        "corrected_joint_count_distribution": {
            str(key): value for key, value in sorted(corrected_joint_counts.items())
        },
        "parameters": {
            "min_source_views": args.min_source_views,
            "min_valid_joints": args.min_valid_joints,
            "association_threshold_px": args.association_threshold_px,
            "correction_threshold_px": args.correction_threshold_px,
            "wrist_correction_threshold_px": args.wrist_correction_threshold_px,
            "replacement_threshold_px": args.replacement_threshold_px,
            "blend": args.blend,
            "max_shift_px": args.max_shift_px,
            "max_bone_change_ratio": args.max_bone_change_ratio,
            "allow_handedness_relabel": bool(args.allow_handedness_relabel),
            "handedness_relabel_margin_px": args.handedness_relabel_margin_px,
            "temporal_guard": bool(args.temporal_guard),
            "temporal_radius": args.temporal_radius,
            "temporal_gate_px": args.temporal_gate_px,
            "temporal_margin_px": args.temporal_margin_px,
            "min_raw_temporal_residual_px": args.min_raw_temporal_residual_px,
        },
        "residual_before_median_px": float(np.median(residual_before)) if residual_before else None,
        "residual_after_median_px": float(np.median(residual_after)) if residual_after else None,
        "output": str(args.output),
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
