#!/usr/bin/env python3
"""Compare an identical WiLoR/cam2 run with and without SAM3 assistance."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from ego_data.calibration import CameraCalibration
from evaluate_target_camera_optimization import (
    _load_jsonl,
    _motion_report,
    _points_2d,
    _target_2d_report,
)


SUMMARY_KEYS = (
    "accepted_frame_count", "rejected_frame_count", "accepted_rate",
    "reprojection_median_px", "selected_view_count_median",
    "valid_joint_inlier_view_count_median", "sam3_gate_enabled",
    "sam3_view_count", "sam3_matched_view_count", "sam3_accepted_view_count",
    "sam3_rejected_matched_view_count", "sam3_corrected_view_count",
    "sam3_corrected_joint_count", "consensus_rejected_view_count",
    "consensus_rejected_camera_counts", "fusion_seconds",
    "fusion_frames_per_second",
)


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _stats(values: list[float], scale: float = 1.0) -> dict[str, Any]:
    data = np.asarray(values, dtype=np.float64)
    data = data[np.isfinite(data)] * scale
    if not len(data):
        return {"sample_count": 0}
    return {
        "sample_count": int(len(data)),
        "p50": float(np.percentile(data, 50)),
        "p95": float(np.percentile(data, 95)),
        "p99": float(np.percentile(data, 99)),
        "max": float(np.max(data)),
        "mean": float(np.mean(data)),
        "over_50": int(np.count_nonzero(data > 50.0)),
        "over_80": int(np.count_nonzero(data > 80.0)),
        "over_100": int(np.count_nonzero(data > 100.0)),
    }


def _summary(root: Path) -> dict[str, Any]:
    value = _read(root / "summary.json")
    output = {key: value.get(key) for key in SUMMARY_KEYS}
    for section in ("temporal_3d_guard", "joint_temporal_guard"):
        item = value.get(section, {})
        output[section] = {
            key: item.get(key)
            for key in (
                "corrected_hand_count", "corrected_joint_count",
                "cleared_hand_count", "preserved_missing_joint_count",
            )
            if key in item
        }
        if section == "joint_temporal_guard":
            output[section]["ray_geometry"] = item.get("ray_geometry", {})
    return output


def _assist_report(raw_root: Path, assisted_root: Path) -> dict[str, Any]:
    root = _read(assisted_root / "summary.json")
    per_camera: dict[str, Any] = {}
    for camera, camera_summary in root.get("cameras", {}).items():
        raw_rows = _load_jsonl(raw_root / camera / "predictions.jsonl")
        assisted_rows = _load_jsonl(assisted_root / camera / "predictions.jsonl")
        shifts: list[float] = []
        changed_joints = 0
        paired_hands = 0
        matched = accepted = corrected_hypotheses = corrected_joints = 0
        for sync_index in sorted(set(raw_rows) & set(assisted_rows)):
            raw_hands = {
                (int(hand.get("detection_index", -1)), int(hand.get("is_right", -1))): hand
                for hand in raw_rows[sync_index].get("hands", [])
            }
            for hand in assisted_rows[sync_index].get("hands", []):
                key = (int(hand.get("detection_index", -1)), int(hand.get("is_right", -1)))
                raw_hand = raw_hands.get(key)
                if raw_hand is None:
                    continue
                paired_hands += 1
                old = _points_2d(hand.get("wilor_joints_2d", raw_hand.get("joints_2d", [])))
                new = _points_2d(hand.get("joints_2d", []))
                finite = np.isfinite(old).all(axis=1) & np.isfinite(new).all(axis=1)
                delta = np.linalg.norm(new[finite] - old[finite], axis=1)
                shifts.extend(delta.tolist())
                changed_joints += int(np.count_nonzero(delta > 1e-6))
                info = hand.get("sam3_assist", {})
                matched += int(info.get("matched", False))
                accepted += int(info.get("accepted", False))
                correction = info.get("two_d_correction", {})
                count = int(correction.get("corrected_joint_count", 0))
                corrected_hypotheses += int(count > 0)
                corrected_joints += count
        per_camera[camera] = {
            "summary": camera_summary,
            "paired_hands": paired_hands,
            "shifted_joint_count": changed_joints,
            "joint_shift_px": _stats(shifts),
            "observed_assist_counts": {
                "matched": matched,
                "accepted": accepted,
                "corrected_hypotheses": corrected_hypotheses,
                "corrected_joints": corrected_joints,
            },
        }
    return {
        "summary": {
            key: root.get(key)
            for key in (
                "stage", "frame_count", "processed_camera_frames", "detections",
                "matched_detections", "match_rate", "accepted_hypotheses",
                "rejected_hypotheses", "accepted_hypothesis_rate",
                "corrected_hypotheses", "corrected_joints", "elapsed_seconds",
            )
        },
        "parameters": root.get("parameters", {}),
        "per_camera": per_camera,
    }


def _target_report(
    dataset: Path, raw_root: Path, optimized_root: Path, leaveout_root: Path,
) -> dict[str, Any]:
    calibration = CameraCalibration.load(dataset / "calibration" / "camera2.json")
    return _target_2d_report(
        _load_jsonl(raw_root / "camera2/predictions.jsonl"),
        _load_jsonl(optimized_root / "camera2/predictions.jsonl"),
        _load_jsonl(leaveout_root / "accepted.jsonl"),
        calibration,
    )


def _trajectory_difference(baseline: Path, assisted: Path) -> dict[str, Any]:
    tracks: list[dict[tuple[int, int], np.ndarray]] = []
    for path in (baseline, assisted):
        values: dict[tuple[int, int], np.ndarray] = {}
        for sync_index, row in _load_jsonl(path).items():
            for hand in row.get("hands", []):
                points = np.asarray(hand.get("joints_base_m", []), dtype=np.float64)
                if points.shape == (21, 3):
                    values[(sync_index, int(hand.get("side", -1)))] = points
        tracks.append(values)
    first, second = tracks
    common = set(first) & set(second)
    values: list[float] = []
    for key in common:
        a, b = first[key], second[key]
        finite = np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1)
        values.extend(np.linalg.norm(b[finite] - a[finite], axis=1).tolist())
    return {
        "matching_hand_tracks": len(common),
        "baseline_only_hand_tracks": len(set(first) - common),
        "sam3_only_hand_tracks": len(set(second) - common),
        "matching_joint_count": len(values),
        "joint_difference_mm": _stats(values, scale=1000.0),
    }


def _finite(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _finite(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_finite(item) for item in value]
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--baseline-raw", required=True, type=Path)
    parser.add_argument("--baseline-optimized", required=True, type=Path)
    parser.add_argument("--baseline-leaveout", required=True, type=Path)
    parser.add_argument("--baseline-fusion", required=True, type=Path)
    parser.add_argument("--sam3-raw", required=True, type=Path)
    parser.add_argument("--sam3-optimized", required=True, type=Path)
    parser.add_argument("--sam3-leaveout", required=True, type=Path)
    parser.add_argument("--sam3-fusion", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    baseline_summary = _summary(args.baseline_fusion)
    sam3_summary = _summary(args.sam3_fusion)
    baseline_target = _target_report(
        args.dataset, args.baseline_raw, args.baseline_optimized, args.baseline_leaveout,
    )
    sam3_target = _target_report(
        args.dataset, args.sam3_raw, args.sam3_optimized, args.sam3_leaveout,
    )
    report: dict[str, Any] = {
        "schema_version": 1,
        "target_camera": "camera2",
        "baseline": {
            "raw_predictions": str(args.baseline_raw),
            "optimized_predictions": str(args.baseline_optimized),
            "leaveout_fusion": str(args.baseline_leaveout),
            "final_fusion": str(args.baseline_fusion),
            "final_summary": baseline_summary,
            "motion": _motion_report(args.baseline_fusion / "accepted.jsonl"),
            "target_correction_summary": _read(args.baseline_optimized / "summary.json"),
            "target_2d_calibration": baseline_target,
        },
        "sam3": {
            "raw_predictions": str(args.sam3_raw),
            "optimized_predictions": str(args.sam3_optimized),
            "leaveout_fusion": str(args.sam3_leaveout),
            "final_fusion": str(args.sam3_fusion),
            "assist": _assist_report(args.baseline_raw, args.sam3_raw),
            "leaveout_summary": _summary(args.sam3_leaveout),
            "final_summary": sam3_summary,
            "motion": _motion_report(args.sam3_fusion / "accepted.jsonl"),
            "target_correction_summary": _read(args.sam3_optimized / "summary.json"),
            "target_2d_calibration": sam3_target,
        },
    }
    baseline_geometry = baseline_summary["joint_temporal_guard"].get("ray_geometry", {})
    sam3_geometry = sam3_summary["joint_temporal_guard"].get("ray_geometry", {})
    baseline_motion = report["baseline"]["motion"]
    sam3_motion = report["sam3"]["motion"]
    report["comparison"] = {
        "final_fusion": {
            "accepted_frame_delta": sam3_summary["accepted_frame_count"] - baseline_summary["accepted_frame_count"],
            "accepted_rate_delta": sam3_summary["accepted_rate"] - baseline_summary["accepted_rate"],
            "accepted_rate_delta_percentage_points": 100.0 * (sam3_summary["accepted_rate"] - baseline_summary["accepted_rate"]),
            "rejected_frame_delta": sam3_summary["rejected_frame_count"] - baseline_summary["rejected_frame_count"],
            "reprojection_median_delta_px": sam3_summary["reprojection_median_px"] - baseline_summary["reprojection_median_px"],
            "reprojection_median_change_percent": 100.0 * (sam3_summary["reprojection_median_px"] - baseline_summary["reprojection_median_px"]) / baseline_summary["reprojection_median_px"],
            "selected_view_count_delta": sam3_summary["selected_view_count_median"] - baseline_summary["selected_view_count_median"],
            "valid_joint_inlier_view_count_delta": sam3_summary["valid_joint_inlier_view_count_median"] - baseline_summary["valid_joint_inlier_view_count_median"],
            "consensus_rejected_view_delta": sam3_summary["consensus_rejected_view_count"] - baseline_summary["consensus_rejected_view_count"],
            "consensus_rejected_camera_delta": {
                camera: sam3_summary.get("consensus_rejected_camera_counts", {}).get(camera, 0)
                - baseline_summary.get("consensus_rejected_camera_counts", {}).get(camera, 0)
                for camera in sorted(set(sam3_summary.get("consensus_rejected_camera_counts", {})) | set(baseline_summary.get("consensus_rejected_camera_counts", {})))
            },
            "ray_geometry": {
                key: {"baseline": baseline_geometry.get(key), "sam3": sam3_geometry.get(key)}
                for key in (
                    "sample_count", "condition_number_p50", "condition_number_p95",
                    "condition_number_p99", "condition_number_max", "condition_ge_1000",
                    "min_ray_angle_p50_degrees", "sam3_downweighted_ray_count",
                )
            },
        },
        "target_2d": {
            "baseline": {
                "changed_joint_count": baseline_target["changed_joint_count"],
                "residual_before_p50_px": baseline_target["all_joints_residual_before_px"].get("p50"),
                "residual_after_p50_px": baseline_target["all_joints_residual_after_px"].get("p50"),
                "median_residual_reduction_percent": baseline_target.get("median_residual_reduction_percent"),
            },
            "sam3": {
                "changed_joint_count": sam3_target["changed_joint_count"],
                "residual_before_p50_px": sam3_target["all_joints_residual_before_px"].get("p50"),
                "residual_after_p50_px": sam3_target["all_joints_residual_after_px"].get("p50"),
                "median_residual_reduction_percent": sam3_target.get("median_residual_reduction_percent"),
            },
        },
        "final_3d_motion": {
            "step_p50_delta_mm": {
                side: sam3_motion.get(side, {}).get("step_mm", {}).get("p50", 0.0)
                - baseline_motion.get(side, {}).get("step_mm", {}).get("p50", 0.0)
                for side in ("0", "1")
            },
            "step_p95_delta_mm": {
                side: sam3_motion.get(side, {}).get("step_mm", {}).get("p95", 0.0)
                - baseline_motion.get(side, {}).get("step_mm", {}).get("p95", 0.0)
                for side in ("0", "1")
            },
            "acceleration_p95_delta_mm": {
                side: sam3_motion.get(side, {}).get("acceleration_mm", {}).get("p95", 0.0)
                - baseline_motion.get(side, {}).get("acceleration_mm", {}).get("p95", 0.0)
                for side in ("0", "1")
            },
        },
        "direct_final_3d_difference": _trajectory_difference(
            args.baseline_fusion / "accepted.jsonl", args.sam3_fusion / "accepted.jsonl",
        ),
        "ground_truth_available": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(_finite(report), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(_finite(report["comparison"]), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
