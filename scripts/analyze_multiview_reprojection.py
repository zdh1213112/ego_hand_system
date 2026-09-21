#!/usr/bin/env python3
"""Audit every fused hand frame against its selected 2D observations."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
import sys

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from ego_data.calibration import CameraCalibration  # noqa: E402
from render_multiview_wilor import (  # noqa: E402
    _final_overlay_for_view,
    _project_refined_hand,
)


JOINT_NAMES = (
    "wrist",
    "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
    "index_mcp", "index_pip", "index_dip", "index_tip",
    "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
    "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
    "little_mcp", "little_pip", "little_dip", "little_tip",
)
TIP_IDS = np.asarray((4, 8, 12, 16, 20), dtype=np.int64)
PALM_IDS = np.asarray((0, 5, 9, 13, 17), dtype=np.int64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--fusion", required=True, type=Path)
    parser.add_argument("--comparison-video", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--top-frames", type=int, default=48)
    return parser.parse_args()


def percentile(values: list[float], q: float) -> float | None:
    return float(np.percentile(values, q)) if values else None


def finite_points(value: object) -> np.ndarray | None:
    points = np.asarray(value, dtype=np.float64)
    if points.shape != (21, 2) or not np.isfinite(points).all():
        return None
    return points


def render_contact_sheets(
    video_path: Path, output_dir: Path, ranked: list[dict], limit: int,
) -> list[str]:
    if not video_path or not video_path.is_file() or limit <= 0:
        return []
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return []
    selected = ranked[:limit]
    thumbs: list[np.ndarray] = []
    for row in selected:
        frame_index = int(row["sync_index"])
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok:
            continue
        thumb = cv2.resize(frame, (960, 390), interpolation=cv2.INTER_AREA)
        label = (
            f"frame {frame_index} visible_risk={row['visible_risk_score']:.1f} "
            f"safe_p95={row['safe_residual_p95_px']:.1f}px "
            f"safe_tip_max={row['safe_tip_max_px']:.1f}px "
            f"suppressed={int(row['suppressed_views'])}"
        )
        cv2.rectangle(thumb, (0, 350), (960, 390), (0, 0, 0), -1)
        cv2.putText(
            thumb, label, (12, 378), cv2.FONT_HERSHEY_SIMPLEX,
            0.62, (255, 255, 255), 2, cv2.LINE_AA,
        )
        thumbs.append(thumb)
    capture.release()
    paths: list[str] = []
    for page_start in range(0, len(thumbs), 12):
        page = thumbs[page_start:page_start + 12]
        while len(page) < 12:
            page.append(np.zeros_like(thumbs[0]))
        canvas = np.vstack([np.hstack(page[i:i + 2]) for i in range(0, 12, 2)])
        path = output_dir / f"risk_contact_sheet_{page_start // 12 + 1:02d}.jpg"
        cv2.imwrite(str(path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])
        paths.append(str(path))
    return paths


def main() -> int:
    args = parse_args()
    dataset = args.dataset.resolve()
    fusion = args.fusion.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    summary = json.loads((fusion / "summary.json").read_text(encoding="utf-8"))
    cameras = tuple(summary["camera_ids"])
    calibrations = {
        camera: CameraCalibration.load(dataset / "calibration" / f"{camera}.json")
        for camera in cameras
    }
    rows = [
        json.loads(line) for line in (fusion / "accepted.jsonl").read_text(
            encoding="utf-8"
        ).splitlines() if line.strip()
    ]

    all_residuals: list[float] = []
    unrefined_residuals: list[float] = []
    tip_residuals: list[float] = []
    joint_residuals: dict[int, list[float]] = defaultdict(list)
    camera_residuals: dict[str, list[float]] = defaultdict(list)
    suppression_reasons: Counter[str] = Counter()
    detector_side_mismatches = 0
    observation_count = 0
    cross_hand_negative: list[dict] = []
    tip_confusions: list[dict] = []
    frame_records: list[dict] = []
    sequences: dict[tuple[int, str], dict[int, tuple[np.ndarray, np.ndarray]]] = defaultdict(dict)
    sequences_3d: dict[int, dict[int, np.ndarray]] = defaultdict(dict)

    for row in rows:
        sync_index = int(row["sync_index"])
        frame_residuals: list[float] = []
        frame_tips: list[float] = []
        frame_safe_residuals: list[float] = []
        frame_safe_tips: list[float] = []
        frame_cross_margin: list[float] = []
        frame_tip_confusions = 0
        frame_safe_tip_confusions = 0
        frame_suppressed = 0
        frame_safe_views = 0
        frame_side_mismatches = 0
        frame_corrected_joints = 0
        frame_max_displacement_mm = 0.0
        hands = row.get("hands", [])
        projected: dict[tuple[int, str], np.ndarray] = {}

        for hand in hands:
            side = int(hand["side"])
            points_3d = np.asarray(hand.get("joints_base_m", []), dtype=np.float64)
            unrefined_3d = np.asarray(
                hand.get("unrefined_joints_base_m", []), dtype=np.float64
            )
            if points_3d.shape == (21, 3) and np.isfinite(points_3d).all():
                sequences_3d[side][sync_index] = points_3d
            anatomy = hand.get("anatomy_refinement", {})
            frame_corrected_joints += int(anatomy.get("corrected_joint_count", 0))
            frame_max_displacement_mm = max(
                frame_max_displacement_mm,
                float(anatomy.get("displacement_max_mm", 0.0) or 0.0),
            )
            for camera, view in hand.get("views", {}).items():
                if camera not in calibrations:
                    continue
                observation_count += 1
                safe, reason = _final_overlay_for_view(
                    calibrations[camera], hand, camera
                )
                if safe is None:
                    frame_suppressed += 1
                    suppression_reasons[reason] += 1
                else:
                    frame_safe_views += 1
                if bool(view.get("detector_is_right")) != bool(side):
                    detector_side_mismatches += 1
                    frame_side_mismatches += 1
                observed = finite_points(view.get("joints_2d"))
                if observed is None or points_3d.shape != (21, 3):
                    continue
                final_pixels = _project_refined_hand(calibrations[camera], points_3d)
                if final_pixels.shape != (21, 2):
                    continue
                finite = np.isfinite(final_pixels).all(axis=1)
                if not np.any(finite):
                    continue
                residual = np.linalg.norm(final_pixels[finite] - observed[finite], axis=1)
                all_residuals.extend(residual.tolist())
                frame_residuals.extend(residual.tolist())
                camera_residuals[camera].extend(residual.tolist())
                if safe is not None:
                    frame_safe_residuals.extend(residual.tolist())
                for joint_id in np.flatnonzero(finite):
                    value = np.linalg.norm(final_pixels[joint_id] - observed[joint_id])
                    joint_residuals[int(joint_id)].append(float(value))
                finite_tips = finite[TIP_IDS]
                tip_values = np.linalg.norm(
                    final_pixels[TIP_IDS[finite_tips]] - observed[TIP_IDS[finite_tips]],
                    axis=1,
                )
                tip_residuals.extend(tip_values.tolist())
                frame_tips.extend(tip_values.tolist())
                if safe is not None:
                    frame_safe_tips.extend(tip_values.tolist())
                if np.all(finite):
                    projected[(side, camera)] = final_pixels
                    sequences[(side, camera)][sync_index] = (observed, final_pixels)
                if unrefined_3d.shape == (21, 3) and np.isfinite(unrefined_3d).all():
                    raw_pixels = _project_refined_hand(calibrations[camera], unrefined_3d)
                    if raw_pixels.shape == (21, 2) and np.isfinite(raw_pixels).all():
                        unrefined_residuals.extend(
                            np.linalg.norm(raw_pixels - observed, axis=1).tolist()
                        )

                if np.all(finite[TIP_IDS]):
                    own_tips = observed[TIP_IDS]
                    distances = np.linalg.norm(
                        final_pixels[TIP_IDS, None, :] - own_tips[None, :, :], axis=2
                    )
                    for finger_index in range(5):
                        nearest = int(np.argmin(distances[finger_index]))
                        advantage = float(
                            distances[finger_index, finger_index] - distances[finger_index, nearest]
                        )
                        if nearest != finger_index and advantage > 5.0:
                            frame_tip_confusions += 1
                            if safe is not None:
                                frame_safe_tip_confusions += 1
                            tip_confusions.append({
                                "sync_index": sync_index,
                                "side": side,
                                "camera": camera,
                                "projected_tip": JOINT_NAMES[int(TIP_IDS[finger_index])],
                                "nearest_observed_tip": JOINT_NAMES[int(TIP_IDS[nearest])],
                                "advantage_px": advantage,
                            })

        for hand in hands:
            side = int(hand["side"])
            for camera, own_view in hand.get("views", {}).items():
                final_pixels = projected.get((side, camera))
                own = finite_points(own_view.get("joints_2d"))
                if final_pixels is None or own is None:
                    continue
                own_error = float(np.median(np.linalg.norm(final_pixels - own, axis=1)))
                other_errors = []
                for other_hand in hands:
                    if int(other_hand["side"]) == side:
                        continue
                    other_view = other_hand.get("views", {}).get(camera)
                    if other_view is None:
                        continue
                    other = finite_points(other_view.get("joints_2d"))
                    if other is not None:
                        other_errors.append(
                            float(np.median(np.linalg.norm(final_pixels - other, axis=1)))
                        )
                if other_errors:
                    margin = min(other_errors) - own_error
                    frame_cross_margin.append(margin)
                    if margin < 0.0:
                        cross_hand_negative.append({
                            "sync_index": sync_index,
                            "side": side,
                            "camera": camera,
                            "own_median_px": own_error,
                            "other_median_px": min(other_errors),
                            "margin_px": margin,
                        })

        residual_median = percentile(frame_residuals, 50) or 0.0
        residual_p95 = percentile(frame_residuals, 95) or 0.0
        residual_max = max(frame_residuals, default=0.0)
        tip_p95 = percentile(frame_tips, 95) or 0.0
        tip_max = max(frame_tips, default=0.0)
        safe_residual_p95 = percentile(frame_safe_residuals, 95) or 0.0
        safe_tip_p95 = percentile(frame_safe_tips, 95) or 0.0
        safe_tip_max = max(frame_safe_tips, default=0.0)
        min_cross_margin = min(frame_cross_margin, default=9999.0)
        missing_hands = max(0, 2 - len(hands))
        risk_score = (
            max(0.0, residual_p95 - 20.0)
            + 0.35 * max(0.0, tip_max - 25.0)
            + 25.0 * frame_suppressed
            + 35.0 * len([m for m in frame_cross_margin if m < 0.0])
            + 8.0 * frame_tip_confusions
            + 20.0 * missing_hands
            + 0.15 * frame_max_displacement_mm
        )
        visible_risk_score = (
            max(0.0, safe_residual_p95 - 12.0)
            + 0.45 * max(0.0, safe_tip_max - 20.0)
            + 12.0 * frame_safe_tip_confusions
            + 20.0 * missing_hands
        )
        frame_records.append({
            "sync_index": sync_index,
            "hand_count": len(hands),
            "observation_view_count": sum(len(h.get("views", {})) for h in hands),
            "residual_median_px": residual_median,
            "residual_p95_px": residual_p95,
            "residual_max_px": residual_max,
            "tip_p95_px": tip_p95,
            "tip_max_px": tip_max,
            "safe_view_count": frame_safe_views,
            "safe_residual_p95_px": safe_residual_p95,
            "safe_tip_p95_px": safe_tip_p95,
            "safe_tip_max_px": safe_tip_max,
            "min_cross_hand_margin_px": min_cross_margin,
            "cross_hand_negative_count": len([m for m in frame_cross_margin if m < 0.0]),
            "tip_confusion_count": frame_tip_confusions,
            "safe_tip_confusion_count": frame_safe_tip_confusions,
            "suppressed_views": frame_suppressed,
            "detector_side_mismatches": frame_side_mismatches,
            "corrected_joints": frame_corrected_joints,
            "max_refinement_displacement_mm": frame_max_displacement_mm,
            "risk_score": risk_score,
            "visible_risk_score": visible_risk_score,
        })

    raw_shape_acceleration: list[float] = []
    final_shape_acceleration: list[float] = []
    raw_tip_shape_acceleration: list[float] = []
    final_tip_shape_acceleration: list[float] = []
    for sequence in sequences.values():
        ids = sorted(sequence)
        for frame_id in ids[1:-1]:
            if frame_id - 1 not in sequence or frame_id + 1 not in sequence:
                continue
            raw = [sequence[frame_id + offset][0] for offset in (-1, 0, 1)]
            final = [sequence[frame_id + offset][1] for offset in (-1, 0, 1)]
            raw_shape = [points - points[0] for points in raw]
            final_shape = [points - points[0] for points in final]
            raw_accel = np.linalg.norm(raw_shape[2] - 2 * raw_shape[1] + raw_shape[0], axis=1)
            final_accel = np.linalg.norm(
                final_shape[2] - 2 * final_shape[1] + final_shape[0], axis=1
            )
            raw_shape_acceleration.extend(raw_accel[1:].tolist())
            final_shape_acceleration.extend(final_accel[1:].tolist())
            raw_tip_shape_acceleration.extend(raw_accel[TIP_IDS].tolist())
            final_tip_shape_acceleration.extend(final_accel[TIP_IDS].tolist())

    step_3d: list[float] = []
    tip_step_3d: list[float] = []
    wrist_step_3d: list[float] = []
    frame_3d_risk: dict[int, float] = defaultdict(float)
    for sequence in sequences_3d.values():
        for frame_id in sorted(sequence):
            if frame_id - 1 not in sequence:
                continue
            step = np.linalg.norm(sequence[frame_id] - sequence[frame_id - 1], axis=1) * 1000.0
            step_3d.extend(step.tolist())
            tip_step_3d.extend(step[TIP_IDS].tolist())
            wrist_step_3d.append(float(step[0]))
            frame_3d_risk[frame_id] = max(frame_3d_risk[frame_id], float(np.max(step[TIP_IDS])))
    records_by_id = {int(row["sync_index"]): row for row in frame_records}
    for frame_id, max_tip_step in frame_3d_risk.items():
        records_by_id[frame_id]["max_3d_tip_step_mm"] = max_tip_step
        records_by_id[frame_id]["risk_score"] += 0.3 * max(0.0, max_tip_step - 60.0)
        records_by_id[frame_id]["visible_risk_score"] += 0.3 * max(
            0.0, max_tip_step - 60.0
        )
    for row in frame_records:
        row.setdefault("max_3d_tip_step_mm", 0.0)

    ranked = sorted(frame_records, key=lambda item: item["risk_score"], reverse=True)
    ranked_visible = sorted(
        frame_records, key=lambda item: item["visible_risk_score"], reverse=True
    )
    csv_path = output / "per_frame_audit.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(frame_records[0]))
        writer.writeheader()
        writer.writerows(frame_records)

    detail_path = output / "association_risks.json"
    detail_path.write_text(json.dumps({
        "cross_hand_negative": cross_hand_negative,
        "tip_confusions": tip_confusions,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    contact_sheets = render_contact_sheets(
        args.comparison_video.resolve() if args.comparison_video else None,
        output,
        ranked_visible,
        args.top_frames,
    )
    report = {
        "frame_count": len(rows),
        "hand_observation_view_count": observation_count,
        "all_joint_reprojection_px": {
            "median": percentile(all_residuals, 50),
            "p95": percentile(all_residuals, 95),
            "p99": percentile(all_residuals, 99),
            "max": max(all_residuals, default=None),
        },
        "unrefined_all_joint_reprojection_px": {
            "median": percentile(unrefined_residuals, 50),
            "p95": percentile(unrefined_residuals, 95),
            "p99": percentile(unrefined_residuals, 99),
            "max": max(unrefined_residuals, default=None),
        },
        "fingertip_reprojection_px": {
            "median": percentile(tip_residuals, 50),
            "p95": percentile(tip_residuals, 95),
            "p99": percentile(tip_residuals, 99),
            "max": max(tip_residuals, default=None),
        },
        "per_joint_reprojection_px": {
            JOINT_NAMES[joint_id]: {
                "median": percentile(values, 50),
                "p95": percentile(values, 95),
                "p99": percentile(values, 99),
            }
            for joint_id, values in sorted(joint_residuals.items())
        },
        "per_camera_reprojection_px": {
            camera: {
                "median": percentile(values, 50),
                "p95": percentile(values, 95),
                "p99": percentile(values, 99),
            }
            for camera, values in camera_residuals.items()
        },
        "image_shape_acceleration_px": {
            "raw_2d_median": percentile(raw_shape_acceleration, 50),
            "raw_2d_p95": percentile(raw_shape_acceleration, 95),
            "final_3d_projection_median": percentile(final_shape_acceleration, 50),
            "final_3d_projection_p95": percentile(final_shape_acceleration, 95),
            "raw_2d_tip_median": percentile(raw_tip_shape_acceleration, 50),
            "raw_2d_tip_p95": percentile(raw_tip_shape_acceleration, 95),
            "final_3d_projection_tip_median": percentile(final_tip_shape_acceleration, 50),
            "final_3d_projection_tip_p95": percentile(final_tip_shape_acceleration, 95),
        },
        "motion_3d_mm_per_frame": {
            "all_joint_median": percentile(step_3d, 50),
            "all_joint_p95": percentile(step_3d, 95),
            "tip_median": percentile(tip_step_3d, 50),
            "tip_p95": percentile(tip_step_3d, 95),
            "tip_max": max(tip_step_3d, default=None),
            "wrist_median": percentile(wrist_step_3d, 50),
            "wrist_p95": percentile(wrist_step_3d, 95),
            "wrist_max": max(wrist_step_3d, default=None),
        },
        "association": {
            "detector_side_mismatch_observations": detector_side_mismatches,
            "cross_hand_negative_count": len(cross_hand_negative),
            "tip_confusion_count": len(tip_confusions),
        },
        "final_view_suppression_reasons": dict(suppression_reasons),
        "frame_threshold_counts": {
            "residual_p95_over_25px": sum(r["residual_p95_px"] > 25 for r in frame_records),
            "residual_p95_over_50px": sum(r["residual_p95_px"] > 50 for r in frame_records),
            "tip_max_over_25px": sum(r["tip_max_px"] > 25 for r in frame_records),
            "tip_max_over_50px": sum(r["tip_max_px"] > 50 for r in frame_records),
            "any_view_suppressed": sum(r["suppressed_views"] > 0 for r in frame_records),
            "missing_hand": sum(r["hand_count"] < 2 for r in frame_records),
            "max_3d_tip_step_over_80mm": sum(r["max_3d_tip_step_mm"] > 80 for r in frame_records),
            "visible_tip_max_over_30px": sum(r["safe_tip_max_px"] > 30 for r in frame_records),
            "visible_tip_max_over_50px": sum(r["safe_tip_max_px"] > 50 for r in frame_records),
            "visible_tip_confusion": sum(r["safe_tip_confusion_count"] > 0 for r in frame_records),
        },
        "highest_risk_frames": ranked[:50],
        "highest_visible_risk_frames": ranked_visible[:50],
        "contact_sheets": contact_sheets,
    }
    report_path = output / "audit_summary.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(report_path)
    print(csv_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
