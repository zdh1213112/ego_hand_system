#!/usr/bin/env python3
"""Render a tiled diagnostic video for multiview WiLoR fusion."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from fractions import Fraction
import json
import math
from pathlib import Path
import sys
import time

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from ego_data.dataset import SequentialVideoReader  # noqa: E402
from camera_models import project_points  # noqa: E402
from ego_data.calibration import CameraCalibration  # noqa: E402
from fuse_multiview_wilor import _triangulate_hand  # noqa: E402
from render_wilor_predictions import HAND_CONNECTIONS, LEFT_COLOR, RIGHT_COLOR  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--fusion", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--final-output", type=Path,
        help=(
            "also render final-only overlays in the same decode pass; valid "
            "when --overlay-mode=diagnostic"
        ),
    )
    parser.add_argument(
        "--cameras", nargs="+",
        help="camera subset and display order (default: fusion summary or dataset order)",
    )
    parser.add_argument(
        "--columns", type=int, default=0,
        help="tile columns; 0 selects 2 for four views, 3 for six views, otherwise auto",
    )
    parser.add_argument("--tile-width", type=int, default=480)
    parser.add_argument(
        "--video-codec", choices=("auto", "h264_nvenc", "mp4v"),
        default="auto",
        help="auto prefers NVIDIA NVENC through PyAV and falls back to OpenCV mp4v",
    )
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument(
        "--overlay-mode",
        choices=(
            "diagnostic", "wilor-2d-only", "sam3-2d-only", "wilor-2d-stabilized",
            "cam3-optimized-2d", "final-only",
        ),
        default="diagnostic",
        help=(
            "diagnostic draws camera-space and refined fusion overlays; "
            "wilor-2d-only draws only per-camera detector boxes and 2D WiLoR "
            "joints; sam3-2d-only draws SAM3-constrained 2D joints only; "
            "wilor-2d-stabilized adds 2D-only temporal outlier filtering; "
            "cam3-optimized-2d uses a leave-cam3-out guarded multiview vote; "
            "final-only draws one fused 3D skeleton per physical hand"
        ),
    )
    parser.add_argument(
        "--wilor-2d-smoothing-radius", type=int, default=1,
        help="centered 2D local-shape smoothing radius for wilor-2d-stabilized",
    )
    parser.add_argument(
        "--wilor-2d-smoothing-strength", type=float, default=0.35,
        help="blend toward the local 2D median in wilor-2d-stabilized",
    )
    parser.add_argument(
        "--wilor-2d-outlier-threshold-px", type=float, default=25.0,
        help="minimum local joint residual treated as a 2D outlier",
    )
    parser.add_argument(
        "--wilor-2d-wrist-outlier-threshold-px", type=float, default=100.0,
        help="minimum wrist residual treated as a whole-hand 2D outlier",
    )
    parser.add_argument(
        "--cam3-camera", default="camera3",
        help="target camera for cam3-optimized-2d",
    )
    parser.add_argument(
        "--cam3-ransac-threshold-px", type=float, default=15.0,
        help="non-cam3 reprojection gate for cam3-optimized-2d",
    )
    parser.add_argument(
        "--cam3-min-support", type=int, default=3,
        help=(
            "minimum number of non-cam3 inliers supporting a joint when "
            "four or more non-cam3 views are available; with two or three "
            "views, a two-view majority is used"
        ),
    )
    parser.add_argument(
        "--cam3-correction-threshold-px", type=float, default=18.0,
        help="cam3 residual before a stronger cross-view correction is applied",
    )
    parser.add_argument(
        "--cam3-strong-blend", type=float, default=0.80,
        help="blend toward cross-view projection for cam3 outliers",
    )
    parser.add_argument(
        "--cam3-replacement-threshold-px", type=float, default=80.0,
        help="cam3 residual at which a high-confidence correction replaces the point",
    )
    parser.add_argument(
        "--cam3-smoothing-radius", type=int, default=2,
        help="cam3 local-shape smoothing radius after cross-view correction",
    )
    parser.add_argument(
        "--cam3-smoothing-strength", type=float, default=0.45,
        help="cam3 blend toward the local 2D median after cross-view correction",
    )
    parser.add_argument(
        "--cam3-temporal-radius", type=int, default=2,
        help=(
            "accepted-frame radius used to check whether a proposed cam3 "
            "correction agrees with the raw temporal track"
        ),
    )
    parser.add_argument(
        "--cam3-temporal-gate-px", type=float, default=35.0,
        help=(
            "cam3 temporal residual above which a correction is rejected "
            "when the raw track is stable"
        ),
    )
    parser.add_argument(
        "--cam3-temporal-margin-px", type=float, default=15.0,
        help=(
            "extra residual margin required before rejecting a cross-view "
            "correction against a stable raw cam3 track"
        ),
    )
    parser.add_argument(
        "--cam3-min-raw-temporal-residual-px", type=float, default=20.0,
        help=(
            "do not apply a cross-view correction when the raw cam3 joint "
            "is already this stable relative to nearby raw frames"
        ),
    )
    parser.add_argument(
        "--cam3-raw-fallback", action="store_true",
        help=(
            "on globally rejected frames, draw the available raw camera3 "
            "WiLoR 2D detections instead of leaving the tile empty"
        ),
    )
    parser.add_argument(
        "--joints-source",
        choices=("refined", "unrefined"),
        default="refined",
        help=(
            "3D joints used for projected skeletons; unrefined uses the "
            "fusion result saved before anatomical/temporal refinement"
        ),
    )
    return parser.parse_args()


class _OpenCvVideoWriter:
    def __init__(
        self, path: Path, fps: float, frame_size: tuple[int, int],
    ) -> None:
        self.writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, frame_size,
        )
        if not self.writer.isOpened():
            raise RuntimeError(f"cannot create video: {path}")

    def write(self, frame: np.ndarray) -> None:
        self.writer.write(frame)

    def release(self) -> None:
        self.writer.release()


class _NvencVideoWriter:
    def __init__(
        self, path: Path, fps: float, frame_size: tuple[int, int],
    ) -> None:
        import av

        self.av = av
        self.container = av.open(str(path), mode="w")
        self.stream = self.container.add_stream(
            "h264_nvenc", rate=Fraction(fps).limit_denominator(100_000)
        )
        self.stream.width, self.stream.height = frame_size
        self.stream.pix_fmt = "yuv420p"
        self.stream.options = {
            "preset": "p2", "tune": "hq", "rc": "vbr", "cq": "19",
        }
        # Fail during setup, where auto mode can still safely use its fallback.
        self.stream.codec_context.open()
        self.closed = False

    def write(self, frame: np.ndarray) -> None:
        video_frame = self.av.VideoFrame.from_ndarray(frame, format="bgr24")
        for packet in self.stream.encode(video_frame):
            self.container.mux(packet)

    def release(self) -> None:
        if self.closed:
            return
        for packet in self.stream.encode():
            self.container.mux(packet)
        self.container.close()
        self.closed = True


def _create_video_writer(
    path: Path, fps: float, frame_size: tuple[int, int], codec: str,
):
    if codec in ("auto", "h264_nvenc"):
        try:
            return _NvencVideoWriter(path, fps, frame_size), "h264_nvenc"
        except Exception as error:
            path.unlink(missing_ok=True)
            if codec == "h264_nvenc":
                raise RuntimeError(f"cannot initialize h264_nvenc: {error}") from error
            print(
                f"NVENC unavailable ({error}); falling back to mp4v",
                flush=True,
            )
    return _OpenCvVideoWriter(path, fps, frame_size), "mp4v"


def _load_jsonl(path: Path) -> dict[int, dict]:
    if not path.is_file():
        return {}
    return {
        int(row["sync_index"]): row
        for row in (
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }


def _build_cam3_raw_fallback_results(
    prediction_root: Path, target_camera: str,
) -> dict[int, dict]:
    """Build camera-native fallback rows for frames rejected by fusion.

    WiLoR writes two handedness hypotheses for each detector box.  For a
    fallback we keep the detector's own handedness label, selecting at most
    one hypothesis per physical detection.  This is intentionally raw: no
    cross-view correction is possible on a frame that the global fusion
    rejected.
    """
    prediction_path = prediction_root / target_camera / "predictions.jsonl"
    prediction_rows = _load_jsonl(prediction_path)
    fallback: dict[int, dict] = {}
    for sync_index, row in prediction_rows.items():
        grouped: dict[int, list[dict]] = {}
        for hand in row.get("hands", []):
            try:
                detection_index = int(hand["detection_index"])
            except (KeyError, TypeError, ValueError):
                continue
            grouped.setdefault(detection_index, []).append(hand)
        hands = []
        for detection_index, hypotheses in sorted(grouped.items()):
            if not hypotheses:
                continue
            detector_side = hypotheses[0].get("detector_is_right")
            try:
                detector_side = int(detector_side)
            except (TypeError, ValueError):
                detector_side = None
            selected = next(
                (
                    hand for hand in hypotheses
                    if detector_side in (0, 1)
                    and int(hand.get("is_right", -1)) == detector_side
                ),
                max(
                    hypotheses,
                    key=lambda hand: float(hand.get("confidence", 0.0)),
                ),
            )
            try:
                side = int(selected.get("is_right"))
            except (TypeError, ValueError):
                continue
            points = np.asarray(
                selected.get("wilor_joints_2d", selected.get("joints_2d", [])),
                dtype=np.float64,
            )
            bbox = np.asarray(selected.get("bbox_xyxy", []), dtype=np.float64)
            if side not in (0, 1) or points.shape != (21, 2):
                continue
            if bbox.shape != (4,) or not np.isfinite(bbox).all():
                continue
            view = {
                "detection_index": detection_index,
                "bbox_xyxy": bbox.tolist(),
                "confidence": float(selected.get("confidence", 0.0)),
                "detector_is_right": detector_side,
                "joints_2d": points.tolist(),
                "inlier_joint_count": 21,
            }
            hands.append({
                "side": side,
                "camera_ids": [target_camera],
                "views": {target_camera: view},
                "fusion_mode": "raw_camera_fallback",
            })
        if hands:
            fallback[sync_index] = {
                "sync_index": sync_index,
                "active_cameras": [target_camera],
                "active_camera_count": 1,
                "hands": hands,
                "fusion_mode": "raw_camera_fallback",
            }
    return fallback


def _build_2d_stabilized_views(
    rows: list[dict[str, str]],
    accepted: dict[int, dict],
    cameras: tuple[str, ...],
    radius: int,
    strength: float,
    outlier_threshold_px: float,
    wrist_outlier_threshold_px: float,
    view_overrides: dict[tuple[int, int, str], dict] | None = None,
) -> dict[tuple[int, int, str], dict]:
    """Prepare optional 2D-only stabilization without touching fusion data.

    The wrist remains the per-frame global anchor. Finger joints are smoothed
    in the wrist-local coordinate system, so ordinary hand translation is not
    blurred. A large wrist residual is treated as a one-frame whole-hand
    outlier and replaced by the centered neighbor median.
    """
    samples: dict[tuple[int, str], list[tuple[int, int, dict, np.ndarray]]] = {
        (side, camera): []
        for side in (0, 1) for camera in cameras
    }
    for ordinal, row in enumerate(rows):
        sync_index = int(row["sync_index"])
        result = accepted.get(sync_index)
        if result is None:
            continue
        for hand in result.get("hands", []):
            side = int(hand["side"])
            for camera in cameras:
                view = hand.get("views", {}).get(camera)
                is_override = False
                if view_overrides is not None:
                    override_key = (sync_index, side, camera)
                    if override_key in view_overrides:
                        view = view_overrides[override_key]
                        is_override = True
                if view is None:
                    continue
                points = np.asarray(
                    view.get(
                        "joints_2d" if is_override else "wilor_joints_2d",
                        view.get("joints_2d", []),
                    ),
                    dtype=np.float64,
                )
                if points.shape != (21, 2) or not np.isfinite(points).all():
                    continue
                samples[(side, camera)].append(
                    (ordinal, sync_index, view, points)
                )

    stabilized: dict[tuple[int, int, str], dict] = {}
    for (side, camera), entries in samples.items():
        for position, (ordinal, sync_index, view, points) in enumerate(entries):
            if radius <= 0:
                filtered = points.copy()
            else:
                neighbors = [
                    item[3] for item in entries
                    if abs(item[0] - ordinal) <= radius
                ]
                if len(neighbors) < 2:
                    filtered = points.copy()
                else:
                    median_points = np.median(np.stack(neighbors), axis=0)
                    wrist = points[0]
                    local = points - wrist
                    median_local = median_points - median_points[0]
                    palm_scale = float(np.median(
                        np.linalg.norm(
                            points[[5, 9, 13, 17]] - wrist, axis=1
                        )
                    ))
                    local_threshold = max(
                        outlier_threshold_px, 0.35 * palm_scale
                    )
                    residual = np.linalg.norm(
                        local - median_local, axis=1
                    )
                    filtered_local = (
                        (1.0 - strength) * local
                        + strength * median_local
                    )
                    outlier = residual > local_threshold
                    filtered_local[outlier] = median_local[outlier]
                    filtered = filtered_local + wrist
                    wrist_residual = float(np.linalg.norm(wrist - median_points[0]))
                    if wrist_residual > max(
                        wrist_outlier_threshold_px, 2.0 * palm_scale
                    ):
                        filtered += median_points[0] - wrist

            stabilized_view = dict(view)
            stabilized_view["joints_2d"] = filtered.tolist()
            stabilized[(sync_index, side, camera)] = stabilized_view
    return stabilized


def _build_cam3_optimized_views(
    rows: list[dict[str, str]],
    accepted: dict[int, dict],
    calibrations: dict[str, CameraCalibration],
    all_cameras: tuple[str, ...],
    target_camera: str,
    ransac_threshold_px: float,
    min_support: int,
    correction_threshold_px: float,
    strong_blend: float,
    replacement_threshold_px: float,
    temporal_radius: int = 2,
    temporal_gate_px: float = 35.0,
    temporal_margin_px: float = 15.0,
    min_raw_temporal_residual_px: float = 20.0,
) -> tuple[dict[tuple[int, int, str], dict], dict[str, int]]:
    """Use a leave-cam3-out multiview consensus to correct cam3 joints.

    The target camera is deliberately excluded from the temporary
    triangulation.  This prevents a bad cam3 observation from helping to
    define the very consensus that is used to judge it.  Other cameras first
    form a robust 3D vote, which is projected back to cam3 only when enough
    non-target views agree.  The resulting override contains image-plane
    points only; the temporary triangulation is never rendered as a 3D
    skeleton.
    """
    overrides: dict[tuple[int, int, str], dict] = {}
    # Keep a raw target-camera track for the second, temporal guard.  The
    # leave-cam3-out vote can still be geometrically self-consistent while
    # being wrong because several non-target views share an association or
    # occlusion error.  In that case a stable raw cam3 hand must win.
    raw_target_entries: dict[int, list[tuple[int, np.ndarray]]] = {
        0: [], 1: [],
    }
    if temporal_radius > 0:
        for ordinal, row in enumerate(rows):
            result = accepted.get(int(row["sync_index"]))
            if result is None:
                continue
            for hand in result.get("hands", []):
                try:
                    side = int(hand["side"])
                except (KeyError, TypeError, ValueError):
                    continue
                target_view = hand.get("views", {}).get(target_camera)
                if target_view is None:
                    continue
                raw_points = np.asarray(
                    target_view.get(
                        "wilor_joints_2d",
                        target_view.get("joints_2d", []),
                    ),
                    dtype=np.float64,
                )
                if side in raw_target_entries and (
                    raw_points.shape == (21, 2)
                    and np.isfinite(raw_points).all()
                ):
                    raw_target_entries[side].append((ordinal, raw_points))

    stats = {
        "hands_seen": 0,
        "hands_with_consensus": 0,
        "sparse_two_view_consensus_hands": 0,
        "consensus_joints": 0,
        "target_outlier_joints": 0,
        "corrected_hands": 0,
        "corrected_joints": 0,
        "replaced_joints": 0,
        "geometry_rejected_hands": 0,
        "temporal_guard_rejected_hands": 0,
        "temporal_guard_rejected_joints": 0,
    }
    for ordinal, row in enumerate(rows):
        sync_index = int(row["sync_index"])
        result = accepted.get(sync_index)
        if result is None:
            continue
        for hand in result.get("hands", []):
            target_view = hand.get("views", {}).get(target_camera)
            if target_view is None:
                continue
            stats["hands_seen"] += 1
            hand_views = hand.get("views", {})
            selected = {}
            for camera in all_cameras:
                view = hand_views.get(camera)
                if view is None:
                    continue
                # Keep the vote in the camera-native WiLoR coordinate system.
                # If an earlier fusion stage stored marker-assisted points in
                # joints_2d, do not let that hidden adjustment redefine the
                # raw 2D target that this renderer is meant to improve.
                points = np.asarray(
                    view.get("wilor_joints_2d", view.get("joints_2d", [])),
                    dtype=np.float64,
                )
                if points.shape != (21, 2) or not np.isfinite(points).all():
                    continue
                selected_view = dict(view)
                selected_view["joints_2d"] = points.tolist()
                selected[camera] = selected_view
            # We need at least two non-cam3 voters.  For a three-camera hand,
            # two of the other cameras are already a majority; for denser
            # observations retain the configured support requirement.  With
            # only one other camera there is no independent vote.
            if target_camera not in selected:
                continue
            non_target_count = len(selected) - 1
            if non_target_count < 2:
                continue
            other_views = {
                camera: selected[camera]
                for camera in selected
                if camera != target_camera
            }
            triangulated = _triangulate_hand(
                int(hand["side"]), other_views, calibrations,
                ransac_threshold_px, 2,
            )
            if triangulated["valid_joints"] < 12:
                stats["geometry_rejected_hands"] += 1
                continue
            if not np.isfinite(triangulated["median_px"]):
                stats["geometry_rejected_hands"] += 1
                continue
            stats["hands_with_consensus"] += 1
            if non_target_count == 2:
                stats["sparse_two_view_consensus_hands"] += 1
            projected = _project_refined_hand(
                calibrations[target_camera], triangulated["points"]
            )
            raw = np.asarray(
                selected[target_camera]["joints_2d"], dtype=np.float64
            )
            if raw.shape != (21, 2) or not np.isfinite(raw).all():
                continue
            hand_scale = float(np.median(
                np.linalg.norm(raw[[5, 9, 13, 17]] - raw[0], axis=1)
            ))
            local_gate = max(correction_threshold_px, 0.35 * hand_scale)
            temporal_median = None
            if temporal_radius > 0:
                nearby = [
                    points for previous_ordinal, points in
                    raw_target_entries.get(int(hand["side"]), [])
                    if previous_ordinal != ordinal
                    and abs(previous_ordinal - ordinal) <= temporal_radius
                ]
                if len(nearby) >= 2:
                    temporal_median = np.median(np.stack(nearby), axis=0)
            corrected = raw.copy()
            changed = 0
            replaced = 0
            temporal_rejected = 0
            for joint in range(21):
                if not np.isfinite(projected[joint]).all():
                    continue
                # With three non-target cameras, two agreeing views form a
                # majority.  For four or more, retain the configured support
                # requirement to avoid letting a single pair dominate.
                support_required = (
                    2 if non_target_count <= 3 else min_support
                )
                other_support = int(triangulated["inlier_counts"][joint])
                if other_support < support_required:
                    continue
                stats["consensus_joints"] += 1
                residual = float(np.linalg.norm(projected[joint] - raw[joint]))
                if residual <= local_gate:
                    continue
                stats["target_outlier_joints"] += 1
                raw_temporal_residual = None
                if temporal_median is not None:
                    if joint == 0:
                        raw_temporal_residual = float(
                            np.linalg.norm(raw[joint] - temporal_median[joint])
                        )
                    else:
                        raw_local = raw[joint] - raw[0]
                        median_local = (
                            temporal_median[joint] - temporal_median[0]
                        )
                        raw_temporal_residual = float(
                            np.linalg.norm(raw_local - median_local)
                        )
                    # Multi-view correction is only useful when cam3 itself
                    # is temporally suspect.  This prevents a geometrically
                    # plausible but wrong vote from overriding a stable raw
                    # 2D detector track.
                    if (
                        raw_temporal_residual
                        < min_raw_temporal_residual_px
                    ):
                        continue
                if residual >= replacement_threshold_px:
                    weight = 1.0
                    replaced += 1
                else:
                    weight = strong_blend
                candidate = (
                    (1.0 - weight) * raw[joint] + weight * projected[joint]
                )
                # Reject a proposed correction that introduces a temporal
                # discontinuity while the raw cam3 track is locally stable.
                # Wrist translation is measured in image coordinates; finger
                # joints use wrist-relative coordinates so a moving hand is
                # not mistaken for a pose change.
                if temporal_median is not None:
                    if joint == 0:
                        candidate_temporal_residual = float(
                            np.linalg.norm(
                                candidate[joint] - temporal_median[joint]
                            )
                        )
                    else:
                        # ``candidate`` is one 2D joint here; use the wrist
                        # already accepted for this hand (joint 0 is visited
                        # first and may itself have been corrected).
                        candidate_local = candidate - corrected[0]
                        median_local = temporal_median[joint] - temporal_median[0]
                        candidate_temporal_residual = float(
                            np.linalg.norm(candidate_local - median_local)
                        )
                    if (
                        raw_temporal_residual < temporal_gate_px
                        and candidate_temporal_residual
                        > max(
                            temporal_gate_px,
                            raw_temporal_residual + temporal_margin_px,
                        )
                    ):
                        temporal_rejected += 1
                        continue
                corrected[joint] = candidate
                if residual > 1.0:
                    changed += 1
            if temporal_rejected:
                stats["temporal_guard_rejected_joints"] += temporal_rejected
                stats["temporal_guard_rejected_hands"] += 1
            if changed:
                corrected_view = dict(target_view)
                corrected_view["joints_2d"] = corrected.tolist()
                corrected_view.pop("wilor_joints_2d", None)
                overrides[(sync_index, int(hand["side"]), target_camera)] = (
                    corrected_view
                )
                stats["corrected_hands"] += 1
                stats["corrected_joints"] += changed
                stats["replaced_joints"] += replaced
    return overrides, stats


def _draw_hand(
    frame: np.ndarray,
    hand: dict,
    side: int,
    use_original_wilor: bool = False,
    suppress_marker: bool = False,
) -> None:
    color = RIGHT_COLOR if side else LEFT_COLOR
    box = np.rint(hand["bbox_xyxy"]).astype(np.int32)
    inlier_count = int(hand.get("inlier_joint_count", 21))
    if inlier_count <= 0:
        cv2.rectangle(frame, tuple(box[:2]), tuple(box[2:]), (40, 40, 255), 4, cv2.LINE_AA)
        cv2.putText(
            frame, "OUTLIER", (int(box[0]), max(30, int(box[1]) - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (40, 40, 255), 2, cv2.LINE_AA,
        )
        return
    cv2.rectangle(frame, tuple(box[:2]), tuple(box[2:]), color, 4, cv2.LINE_AA)
    marker = {} if suppress_marker else hand.get("marker_assist", {})
    original = np.asarray(hand.get("wilor_joints_2d", []), dtype=np.float64)
    if marker.get("applied") and original.shape == (21, 2):
        raw_points = np.rint(original).astype(np.int32)
        for start, end in HAND_CONNECTIONS:
            cv2.line(
                frame, tuple(raw_points[start]), tuple(raw_points[end]),
                (0, 180, 255), 2, cv2.LINE_AA,
            )
        for center in np.rint(marker.get("matched_blob_centers", [])).astype(np.int32):
            cv2.circle(frame, tuple(center), 8, (255, 0, 255), 2, cv2.LINE_AA)
            cv2.drawMarker(
                frame, tuple(center), (255, 255, 255), cv2.MARKER_CROSS,
                11, 1, cv2.LINE_AA,
            )
    sam3 = {} if suppress_marker else hand.get("sam3_assist", {})
    sam3_correction = sam3.get("two_d_correction", {})
    if (
        sam3_correction.get("corrected_joint_count", 0)
        and original.shape == (21, 2)
    ):
        raw_points = np.rint(original).astype(np.int32)
        for start, end in HAND_CONNECTIONS:
            cv2.line(
                frame, tuple(raw_points[start]), tuple(raw_points[end]),
                (0, 180, 255), 2, cv2.LINE_AA,
            )
        for point in raw_points:
            cv2.circle(frame, tuple(point), 3, (0, 180, 255), -1, cv2.LINE_AA)
    point_source = (
        original
        if use_original_wilor and original.shape == (21, 2)
        else np.asarray(hand["joints_2d"], dtype=np.float64)
    )
    points = np.rint(point_source).astype(np.int32)
    if points.shape != (21, 2):
        return
    for start, end in HAND_CONNECTIONS:
        cv2.line(frame, tuple(points[start]), tuple(points[end]), color, 4, cv2.LINE_AA)
    for point in points:
        cv2.circle(frame, tuple(point), 5, color, -1, cv2.LINE_AA)
    if marker.get("applied"):
        label = f"MARKER {int(marker.get('matched_marker_count', 0))}"
        cv2.putText(
            frame, label, (int(box[0]), min(frame.shape[0] - 12, int(box[3]) + 28)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 0, 255), 2, cv2.LINE_AA,
        )
    if sam3.get("enabled"):
        support = sam3.get("tolerant_inside_ratio_visible")
        sam3_label = f"SAM3 {float(sam3.get('mask_score', 0.0)):.2f}"
        if support is not None:
            sam3_label += f" J{float(support):.2f}"
        sam3_color = (70, 220, 70) if sam3.get("accepted") else (0, 80, 255)
        cv2.putText(
            frame,
            sam3_label,
            (int(box[0]), min(frame.shape[0] - 12, int(box[3]) + 52)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            sam3_color,
            2,
            cv2.LINE_AA,
        )


def _project_refined_hand(
    camera: CameraCalibration, points_base: np.ndarray,
) -> np.ndarray:
    rotation = camera.T_base_camera[:3, :3]
    center = camera.T_base_camera[:3, 3]
    points_camera = (rotation.T @ (points_base - center).T).T
    pixels, valid = project_points(camera, points_camera)
    pixels[~valid] = np.nan
    return pixels


def _draw_final_detection(
    frame: np.ndarray, view: dict, side: int, joints_source: str,
) -> None:
    box = np.rint(view.get("bbox_xyxy", [])).astype(np.int32)
    if box.shape != (4,):
        return
    color = (255, 210, 40) if side == 0 else (40, 220, 255)
    cv2.rectangle(
        frame, tuple(box[:2]), tuple(box[2:]), (20, 20, 20), 7, cv2.LINE_AA
    )
    cv2.rectangle(
        frame, tuple(box[:2]), tuple(box[2:]), color, 3, cv2.LINE_AA
    )
    cv2.putText(
        frame,
        f"{'FINAL' if joints_source == 'refined' else 'RAW'} {'R' if side else 'L'}",
        (int(box[0]), max(28, int(box[1]) - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        color,
        2,
        cv2.LINE_AA,
    )


def _draw_refined_hand(
    frame: np.ndarray,
    points: np.ndarray,
    side: int,
    corrected_joint_count: int,
    final_only: bool = False,
) -> None:
    if points.shape != (21, 2):
        return
    color = (255, 210, 40) if side == 0 else (40, 220, 255)
    finite = np.isfinite(points).all(axis=1)
    rounded = np.zeros((21, 2), dtype=np.int32)
    rounded[finite] = np.rint(points[finite]).astype(np.int32)
    for start, end in HAND_CONNECTIONS:
        if not (finite[start] and finite[end]):
            continue
        cv2.line(
            frame, tuple(rounded[start]), tuple(rounded[end]),
            (20, 20, 20), 8, cv2.LINE_AA,
        )
        cv2.line(
            frame, tuple(rounded[start]), tuple(rounded[end]),
            color, 4, cv2.LINE_AA,
        )
    for point in rounded[finite]:
        cv2.circle(frame, tuple(point), 6, (20, 20, 20), -1, cv2.LINE_AA)
        cv2.circle(frame, tuple(point), 4, color, -1, cv2.LINE_AA)
    if finite[0] and not final_only:
        label_at = tuple((rounded[0] + np.asarray((10, -12))).tolist())
        label = f"ANATOMY {'R' if side else 'L'} {corrected_joint_count}"
        cv2.putText(
            frame, label, label_at, cv2.FONT_HERSHEY_SIMPLEX,
            0.58, color, 2, cv2.LINE_AA,
        )


def _render_camera_tile(
    frame: np.ndarray,
    result: dict | None,
    rejected: dict[int, dict],
    camera: str,
    calibration: CameraCalibration,
    sync_index: int,
    overlay_mode: str,
    joints_source: str,
    stabilized_views: dict[tuple[int, int, str], dict] | None,
    target_camera: str | None,
    tile_size: tuple[int, int],
    raw_fallback_result: dict | None = None,
) -> np.ndarray:
    using_raw_fallback = (
        result is None
        and raw_fallback_result is not None
        and overlay_mode == "cam3-optimized-2d"
        and camera == target_camera
    )
    display_result = raw_fallback_result if using_raw_fallback else result
    if display_result is not None:
        for hand in display_result["hands"]:
            anatomy = hand.get("anatomy_refinement", {})
            view = hand["views"].get(camera)
            if (
                overlay_mode in ("wilor-2d-stabilized", "cam3-optimized-2d")
                and stabilized_views is not None
            ):
                view = stabilized_views.get(
                    (sync_index, int(hand["side"]), camera), view
                )
            if overlay_mode in (
                "diagnostic", "wilor-2d-only", "sam3-2d-only",
                "wilor-2d-stabilized",
                "cam3-optimized-2d",
            ):
                if view is not None:
                    _draw_hand(
                        frame, view, int(hand["side"]),
                        use_original_wilor=(
                            overlay_mode == "wilor-2d-only"
                            or (
                                overlay_mode == "cam3-optimized-2d"
                                and camera != target_camera
                            )
                        ),
                        suppress_marker=(
                            overlay_mode not in ("diagnostic", "sam3-2d-only")
                        ),
                    )
            elif view is not None:
                _draw_final_detection(
                    frame, view, int(hand["side"]), joints_source
                )
            if (
                overlay_mode == "final-only"
                or (overlay_mode == "diagnostic" and anatomy.get("applied"))
            ):
                joints_key = (
                    "unrefined_joints_base_m"
                    if joints_source == "unrefined" else "joints_base_m"
                )
                if joints_key not in hand:
                    raise ValueError(
                        f"{joints_key} is missing for frame {sync_index}; "
                        f"cannot render --joints-source={joints_source}"
                    )
                refined_pixels = _project_refined_hand(
                    calibration,
                    np.asarray(hand[joints_key], dtype=np.float64),
                )
                _draw_refined_hand(
                    frame,
                    refined_pixels,
                    int(hand["side"]),
                    int(anatomy.get("corrected_joint_count", 0)),
                    final_only=overlay_mode == "final-only",
                )
    camera_views = (
        [
            hand["views"][camera] for hand in display_result["hands"]
            if camera in hand["views"]
        ]
        if display_result is not None else []
    )
    inlier_joints = sum(
        int(view.get("inlier_joint_count", 21)) for view in camera_views
    )
    if using_raw_fallback:
        status, status_color = "RAW FALLBACK", (0, 190, 255)
    elif result is None:
        status, status_color = "REJECTED", (40, 80, 255)
    elif overlay_mode == "final-only":
        status = "FINAL FUSED" if joints_source == "refined" else "RAW FUSED"
        status_color = (70, 220, 70)
    elif not camera_views:
        status, status_color = "INACTIVE", (180, 180, 180)
    elif inlier_joints == 0:
        status, status_color = "OUTLIER", (40, 80, 255)
    else:
        status, status_color = f"USED {inlier_joints}/42", (70, 220, 70)
    cv2.putText(
        frame, f"{camera} | {status} | frame {sync_index}", (25, 45),
        cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 3, cv2.LINE_AA,
    )
    if result is None and sync_index in rejected:
        rejection = rejected[sync_index]
        reason = rejection.get("reason") or ",".join(
            rejection.get("reasons", [])
        )
        cv2.putText(
            frame, reason[:55], (25, 82), cv2.FONT_HERSHEY_SIMPLEX,
            0.75, status_color, 2, cv2.LINE_AA,
        )
    return cv2.resize(frame, tile_size, interpolation=cv2.INTER_AREA)


def main() -> int:
    args = parse_args()
    dataset = args.dataset.resolve()
    fusion = args.fusion.resolve()
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    available_cameras = tuple(manifest["camera_ids"])
    fusion_summary_path = fusion / "summary.json"
    fusion_summary = (
        json.loads(fusion_summary_path.read_text(encoding="utf-8"))
        if fusion_summary_path.is_file() else {}
    )
    default_cameras = fusion_summary.get("camera_ids", available_cameras)
    cameras = tuple(dict.fromkeys(args.cameras or default_cameras))
    if not cameras:
        raise ValueError("at least one render camera is required")
    unknown_cameras = [camera for camera in cameras if camera not in available_cameras]
    if unknown_cameras:
        raise ValueError(f"selected cameras are not present in the dataset: {unknown_cameras}")
    if args.columns < 0:
        raise ValueError("columns must be non-negative")
    if args.wilor_2d_smoothing_radius < 0:
        raise ValueError("--wilor-2d-smoothing-radius must be non-negative")
    if not 0.0 <= args.wilor_2d_smoothing_strength <= 1.0:
        raise ValueError("--wilor-2d-smoothing-strength must be in [0, 1]")
    if args.wilor_2d_outlier_threshold_px <= 0:
        raise ValueError("--wilor-2d-outlier-threshold-px must be positive")
    if args.wilor_2d_wrist_outlier_threshold_px <= 0:
        raise ValueError(
            "--wilor-2d-wrist-outlier-threshold-px must be positive"
        )
    if args.cam3_camera not in available_cameras:
        raise ValueError(f"--cam3-camera is not present: {args.cam3_camera}")
    if args.overlay_mode == "cam3-optimized-2d" and args.cam3_camera not in cameras:
        raise ValueError(
            f"--cameras must include {args.cam3_camera} for cam3-optimized-2d"
        )
    if args.cam3_raw_fallback and args.overlay_mode != "cam3-optimized-2d":
        raise ValueError("--cam3-raw-fallback requires --overlay-mode=cam3-optimized-2d")
    if args.cam3_min_support < 2:
        raise ValueError("--cam3-min-support must be at least 2")
    if args.cam3_ransac_threshold_px <= 0:
        raise ValueError("--cam3-ransac-threshold-px must be positive")
    if args.cam3_correction_threshold_px <= 0:
        raise ValueError("--cam3-correction-threshold-px must be positive")
    if not 0.0 <= args.cam3_strong_blend <= 1.0:
        raise ValueError("--cam3-strong-blend must be in [0, 1]")
    if args.cam3_replacement_threshold_px <= args.cam3_correction_threshold_px:
        raise ValueError(
            "--cam3-replacement-threshold-px must exceed "
            "--cam3-correction-threshold-px"
        )
    if args.cam3_smoothing_radius < 0:
        raise ValueError("--cam3-smoothing-radius must be non-negative")
    if not 0.0 <= args.cam3_smoothing_strength <= 1.0:
        raise ValueError("--cam3-smoothing-strength must be in [0, 1]")
    if args.cam3_temporal_radius < 0:
        raise ValueError("--cam3-temporal-radius must be non-negative")
    if args.cam3_temporal_gate_px <= 0:
        raise ValueError("--cam3-temporal-gate-px must be positive")
    if args.cam3_temporal_margin_px < 0:
        raise ValueError("--cam3-temporal-margin-px must be non-negative")
    if args.cam3_min_raw_temporal_residual_px < 0:
        raise ValueError(
            "--cam3-min-raw-temporal-residual-px must be non-negative"
        )
    if args.columns:
        columns = args.columns
    elif len(cameras) == 4:
        columns = 2
    elif len(cameras) == 6:
        columns = 3
    else:
        columns = max(1, int(math.ceil(math.sqrt(len(cameras)))))
    row_count = int(math.ceil(len(cameras) / columns))
    with (dataset / "multiview_frames.csv").open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    rows = rows[: args.max_frames or None]
    accepted = _load_jsonl(fusion / "accepted.jsonl")
    rejected = _load_jsonl(fusion / "rejected.jsonl")
    raw_fallback_results = {}
    if args.cam3_raw_fallback:
        prediction_root = fusion.parent / "wilor_multiview"
        if not (prediction_root / args.cam3_camera / "predictions.jsonl").is_file():
            assisted_root = fusion.parent / "wilor_multiview_glove_marker_assisted"
            if (assisted_root / args.cam3_camera / "predictions.jsonl").is_file():
                prediction_root = assisted_root
        raw_fallback_results = _build_cam3_raw_fallback_results(
            prediction_root, args.cam3_camera,
        )
        print(
            f"cam3 raw fallback: {len(raw_fallback_results)} frames available",
            flush=True,
        )
    stabilized_views = (
        _build_2d_stabilized_views(
            rows, accepted, cameras,
            args.wilor_2d_smoothing_radius,
            args.wilor_2d_smoothing_strength,
            args.wilor_2d_outlier_threshold_px,
            args.wilor_2d_wrist_outlier_threshold_px,
        )
        if args.overlay_mode == "wilor-2d-stabilized" else None
    )
    image_size = tuple(manifest["image_size"])
    tile_width = args.tile_width
    tile_height = int(round(tile_width * image_size[1] / image_size[0]))
    timestamps = np.asarray([int(row["reference_timestamp_ns"]) for row in rows], dtype=np.int64)
    delta = np.diff(timestamps).astype(np.float64) / 1e9
    fps = float(1.0 / np.median(delta[delta > 0])) if np.any(delta > 0) else 30.0
    jobs = [(args.overlay_mode, args.output.resolve())]
    if args.final_output is not None:
        if args.overlay_mode != "diagnostic":
            raise ValueError(
                "--final-output requires --overlay-mode=diagnostic"
            )
        final_output = args.final_output.resolve()
        if final_output == jobs[0][1]:
            raise ValueError("--output and --final-output must differ")
        jobs.append(("final-only", final_output))
    frame_size = (tile_width * columns, tile_height * row_count)
    writers = {}
    writer_codecs = {}
    try:
        for mode, path in jobs:
            path.parent.mkdir(parents=True, exist_ok=True)
            writers[mode], writer_codecs[mode] = _create_video_writer(
                path, fps, frame_size, args.video_codec
            )
            print(
                f"Render output {mode}: {path} | codec={writer_codecs[mode]}",
                flush=True,
            )
    except Exception:
        for writer in writers.values():
            writer.release()
        raise
    readers = {
        camera: SequentialVideoReader(
            dataset / "cameras" / camera / manifest["storage"]["video_filename"], image_size
        )
        for camera in cameras
    }
    all_calibrations = {
        camera: CameraCalibration.load(
            dataset / "calibration" / f"{camera}.json"
        )
        for camera in available_cameras
    }
    calibrations = {
        camera: all_calibrations[camera]
        for camera in cameras
    }
    if args.overlay_mode == "cam3-optimized-2d":
        optimized_views, cam3_stats = _build_cam3_optimized_views(
            rows, accepted, all_calibrations, available_cameras,
            args.cam3_camera, args.cam3_ransac_threshold_px,
            args.cam3_min_support, args.cam3_correction_threshold_px,
            args.cam3_strong_blend, args.cam3_replacement_threshold_px,
            args.cam3_temporal_radius, args.cam3_temporal_gate_px,
            args.cam3_temporal_margin_px,
            args.cam3_min_raw_temporal_residual_px,
        )
        stabilized_views = _build_2d_stabilized_views(
            rows, accepted, (args.cam3_camera,),
            args.cam3_smoothing_radius,
            args.cam3_smoothing_strength,
            args.wilor_2d_outlier_threshold_px,
            args.wilor_2d_wrist_outlier_threshold_px,
            view_overrides=optimized_views,
        )
        print(
            "cam3 cross-view correction: "
            + ", ".join(f"{key}={value}" for key, value in cam3_stats.items()),
            flush=True,
        )
    cv2.setNumThreads(1)
    render_started = time.perf_counter()
    try:
        worker_count = min(16, max(len(cameras), len(cameras) * len(jobs)))
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            def submit_reads(row: dict[str, str]):
                return {
                    camera: executor.submit(
                        readers[camera].read,
                        int(row[f"{camera}_frame_index"]),
                    )
                    for camera in cameras
                }

            pending_reads = submit_reads(rows[0]) if rows else {}
            for ordinal, row in enumerate(rows):
                source_frames = {
                    camera: pending_reads[camera].result()
                    for camera in cameras
                }
                pending_reads = (
                    submit_reads(rows[ordinal + 1])
                    if ordinal + 1 < len(rows) else {}
                )
                sync_index = int(row["sync_index"])
                result = accepted.get(sync_index)
                tile_futures = {mode: [] for mode, _ in jobs}
                for camera in cameras:
                    source_frame = source_frames[camera]
                    for job_index, (mode, _) in enumerate(jobs):
                        frame = (
                            source_frame
                            if job_index == len(jobs) - 1
                            else source_frame.copy()
                        )
                        tile_futures[mode].append(executor.submit(
                            _render_camera_tile,
                            frame, result, rejected, camera,
                            calibrations[camera], sync_index, mode,
                            args.joints_source,
                            stabilized_views,
                            args.cam3_camera,
                            (tile_width, tile_height),
                            raw_fallback_results.get(sync_index),
                        ))
                for mode, _ in jobs:
                    tiles = [future.result() for future in tile_futures[mode]]
                    while len(tiles) < row_count * columns:
                        tiles.append(np.zeros(
                            (tile_height, tile_width, 3), dtype=np.uint8
                        ))
                    canvas = np.vstack([
                        np.hstack(tiles[start:start + columns])
                        for start in range(0, len(tiles), columns)
                    ])
                    writers[mode].write(canvas)
                if (ordinal + 1) % 50 == 0:
                    elapsed = time.perf_counter() - render_started
                    print(
                        f"rendered {ordinal + 1}/{len(rows)} | "
                        f"{(ordinal + 1) / max(elapsed, 1e-9):.2f} frames/s",
                        flush=True,
                    )
    finally:
        for reader in readers.values():
            reader.close()
        for writer in writers.values():
            writer.release()
    render_seconds = time.perf_counter() - render_started
    for mode, path in jobs:
        print(
            f"Multiview {mode} video: {path} "
            f"({len(rows)} frames at {fps:.3f} fps, "
            f"rendered in {render_seconds:.2f}s, "
            f"{len(rows) / max(render_seconds, 1e-9):.2f} frames/s, "
            f"codec={writer_codecs[mode]})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
