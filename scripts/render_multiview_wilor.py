#!/usr/bin/env python3
"""Render a tiled diagnostic video for multiview WiLoR fusion."""

from __future__ import annotations

import argparse
import copy
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
    parser.add_argument(
        "--final-min-valid-joints", type=int, default=21,
        help=(
            "minimum finite joints required before drawing a final 3-D hand "
            "or using it as a continuous-gap endpoint; the default 21 keeps "
            "a missing finger from being rendered as a distorted skeleton"
        ),
    )
    parser.add_argument(
        "--final-min-joint-inlier-views", type=int, default=3,
        help=(
            "minimum independent camera inliers required for every final 3-D "
            "joint; hands with a sparser joint fall back to native WiLoR 2-D"
        ),
    )
    parser.add_argument(
        "--final-require-local-view", type=int, choices=(0, 1), default=1,
        help=(
            "in final-only mode, draw a fused hand in a camera only when that "
            "camera has a local observation; this prevents an occluded tile "
            "from showing a stale pose projected from another camera"
        ),
    )
    parser.add_argument(
        "--interpolate-short-gaps", type=int, choices=(0, 1), default=0,
        help=(
            "for final/diagnostic review videos only, geodesically fill a "
            "1--2 frame rejected gap when both endpoint poses are well supported; "
            "the fusion JSONL remains unchanged"
        ),
    )
    parser.add_argument(
        "--short-gap-max-frames", type=int, default=2,
        help="maximum rejected gap length eligible for display interpolation",
    )
    parser.add_argument(
        "--short-gap-max-step-m", type=float, default=0.060,
        help="maximum endpoint motion per frame accepted for display interpolation",
    )
    parser.add_argument(
        "--interpolate-all-gaps", type=int, choices=(0, 1), default=0,
        help=(
            "fill every bounded hand-track gap for visualization only; the "
            "synthetic poses are labeled PREDICTED and fusion JSONL is unchanged"
        ),
    )
    parser.add_argument(
        "--all-gap-max-frames", type=int, default=0,
        help=(
            "maximum gap length for --interpolate-all-gaps (0 means no "
            "length limit)"
        ),
    )
    parser.add_argument(
        "--all-gap-max-step-m", type=float, default=0.0,
        help=(
            "optional endpoint displacement per frame bound for all-gap "
            "prediction (0 means no bound)"
        ),
    )
    parser.add_argument(
        "--all-gap-prediction-mode",
        choices=("hold", "hold-previous", "linear", "geodesic"),
        default="hold",
        help=(
            "pose used for display-only long-gap prediction: geodesically "
            "blend rigid finger bones (smooth, fixed-length), or linearly "
            "blend all joints (legacy; may deform fingers), hold the nearest "
            "trusted endpoint, or hold the previous endpoint for the whole "
            "gap; hold is the safe default for delivery review"
        ),
    )
    parser.add_argument(
        "--final-predicted-2d-fallback", type=int, choices=(0, 1), default=0,
        help=(
            "in final-only review, draw the current camera's WiLoR 2-D "
            "observation for a hand whose 3-D pose is synthetic or invalid; "
            "this avoids showing a twisted geodesic pose during a depth gap"
        ),
    )
    parser.add_argument(
        "--final-2d-residual-fallback-px", type=float, default=0.0,
        help=(
            "in final-only review, use the current camera's observed 2-D "
            "hand when the projected 3-D pose has a residual percentile above "
            "this value; 0 disables the per-camera fallback"
        ),
    )
    parser.add_argument(
        "--final-2d-residual-percentile", type=float, default=50.0,
        help=(
            "percentile of per-joint projected-vs-observed 2-D residual used "
            "by --final-2d-residual-fallback-px (50 means median)"
        ),
    )
    parser.add_argument(
        "--final-wrist-residual-fallback-px", type=float, default=0.0,
        help=(
            "fallback to the native camera 2-D pose when the projected "
            "fused wrist differs from that camera by more than this many "
            "pixels; 0 disables the wrist-specific gate"
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


def _interpolate_short_gap_results(
    rows: list[dict[str, str]],
    accepted: dict[int, dict],
    *,
    max_gap_frames: int = 2,
    max_step_m: float = 0.060,
) -> dict[int, dict]:
    """Create conservative display-only poses for very short rejected gaps.

    Fusion rows remain unchanged: a rejected frame is still rejected in the
    JSONL output.  For a review video only, a one- or two-frame gap may be
    filled when both endpoint poses belong to the same physical hand, have at
    least 18 common finite joints, and their endpoint displacement is within a
    plausible per-frame motion bound.  Long gaps and ambiguous reacquisition
    are left blank instead of inventing a trajectory.
    """
    if max_gap_frames <= 0 or max_step_m <= 0.0 or not accepted:
        return {}
    ordered_ids = [int(row["sync_index"]) for row in rows]
    ordered_id_set = set(ordered_ids)
    accepted_ids = sorted(int(sync_index) for sync_index in accepted)
    interpolated: dict[int, dict] = {}

    def _points(hand: dict, key: str) -> np.ndarray:
        value = np.asarray(hand.get(key, []), dtype=np.float64)
        if value.shape != (21, 3):
            return np.full((21, 3), np.nan, dtype=np.float64)
        return value

    def _view_between(
        previous: dict | None, following: dict | None, alpha: float,
    ) -> dict | None:
        if previous is None and following is None:
            return None
        if previous is None:
            return copy.deepcopy(following)
        if following is None:
            return copy.deepcopy(previous)
        view = copy.deepcopy(previous)
        for key in ("bbox_xyxy", "joints_2d", "wilor_joints_2d"):
            first = np.asarray(previous.get(key, []), dtype=np.float64)
            second = np.asarray(following.get(key, []), dtype=np.float64)
            if first.shape != second.shape or first.size == 0:
                continue
            if not np.isfinite(first).all() or not np.isfinite(second).all():
                continue
            view[key] = ((1.0 - alpha) * first + alpha * second).tolist()
        view["confidence"] = float(
            (1.0 - alpha) * float(previous.get("confidence", 0.0))
            + alpha * float(following.get("confidence", 0.0))
        )
        view["interpolated"] = True
        return view

    # Work pair-by-pair so every missing frame in a short block uses the same
    # two trusted endpoint identities.  This avoids chaining a synthetic pose
    # into the next interpolation decision.
    for previous_id, following_id in zip(accepted_ids, accepted_ids[1:]):
        gap = following_id - previous_id - 1
        if gap <= 0 or gap > int(max_gap_frames):
            continue
        previous_result = accepted[previous_id]
        following_result = accepted[following_id]
        previous_hands = {
            int(hand.get("side", -1)): hand
            for hand in previous_result.get("hands", [])
            if int(hand.get("side", -1)) in (0, 1)
        }
        following_hands = {
            int(hand.get("side", -1)): hand
            for hand in following_result.get("hands", [])
            if int(hand.get("side", -1)) in (0, 1)
        }
        for offset in range(1, gap + 1):
            sync_index = previous_id + offset
            if sync_index not in ordered_id_set or sync_index in accepted:
                continue
            alpha = float(offset) / float(gap + 1)
            hands: list[dict] = []
            for side in (0, 1):
                previous_hand = previous_hands.get(side)
                following_hand = following_hands.get(side)
                if previous_hand is None or following_hand is None:
                    continue
                previous_points = _points(previous_hand, "joints_base_m")
                following_points = _points(following_hand, "joints_base_m")
                common = (
                    np.isfinite(previous_points).all(axis=1)
                    & np.isfinite(following_points).all(axis=1)
                )
                if int(np.count_nonzero(common)) < 18:
                    continue
                endpoint_steps = np.linalg.norm(
                    following_points[common] - previous_points[common], axis=1
                )
                # Use the maximum endpoint displacement, not only the wrist:
                # a depth branch in one fingertip must not be hidden by a
                # plausible palm translation.
                if (
                    not np.isfinite(endpoint_steps).all()
                    or float(np.max(endpoint_steps))
                    > float(max_step_m) * float(gap + 1)
                ):
                    continue
                # Interpolate bone directions on the sphere rather than
                # blending each distal joint independently.  The latter can
                # change finger length/shape in the one- or two-frame gap
                # and is the same visible morph that the long-gap geodesic
                # path was introduced to avoid.
                points = _geodesic_hand_pose(
                    previous_points, following_points, alpha
                )
                hand = copy.deepcopy(previous_hand)
                hand["joints_base_m"] = points.tolist()
                for key in ("unrefined_joints_base_m",):
                    first = _points(previous_hand, key)
                    second = _points(following_hand, key)
                    finite = (
                        np.isfinite(first).all(axis=1)
                        & np.isfinite(second).all(axis=1)
                    )
                    if int(np.count_nonzero(finite)) >= 18:
                        value = np.full((21, 3), np.nan, dtype=np.float64)
                        value[finite] = first[finite]
                        for chain in _GEO_CHAINS:
                            for parent, child in zip(chain[:-1], chain[1:]):
                                if not (finite[parent] and finite[child]):
                                    continue
                                first_vector = first[child] - first[parent]
                                second_vector = second[child] - second[parent]
                                first_length = float(np.linalg.norm(first_vector))
                                second_length = float(np.linalg.norm(second_vector))
                                if first_length <= 1e-8 or second_length <= 1e-8:
                                    continue
                                direction = _slerp_direction(
                                    first_vector, second_vector, alpha
                                )
                                value[child] = value[parent] + (
                                    (1.0 - alpha) * first_length
                                    + alpha * second_length
                                ) * direction
                        hand[key] = value.tolist()
                previous_views = previous_hand.get("views", {})
                following_views = following_hand.get("views", {})
                views: dict[str, dict] = {}
                for camera in sorted(set(previous_views) | set(following_views)):
                    view = _view_between(
                        previous_views.get(camera), following_views.get(camera), alpha
                    )
                    if view is not None:
                        views[camera] = view
                hand["views"] = views
                hand["camera_ids"] = [
                    camera for camera in previous_hand.get("camera_ids", [])
                    if camera in views
                ] or list(views)
                hand["fusion_mode"] = "short_gap_interpolation"
                hand["interpolated"] = {
                    "applied": True,
                    "from_sync_index": int(previous_id),
                    "to_sync_index": int(following_id),
                    "alpha": alpha,
                }
                hand["anatomy_refinement"] = {
                    "applied": True,
                    "corrected_joint_count": 0,
                }
                hands.append(hand)
            if hands:
                interpolated[sync_index] = {
                    "sync_index": sync_index,
                    "active_cameras": list(
                        previous_result.get("active_cameras", [])
                    ),
                    "active_camera_count": int(
                        previous_result.get("active_camera_count", 0)
                    ),
                    "hands": hands,
                    "fusion_mode": "short_gap_interpolation",
                    "interpolated": {
                        "applied": True,
                        "from_sync_index": int(previous_id),
                        "to_sync_index": int(following_id),
                    },
                }
    return interpolated


_GEO_CHAINS = (
    (0, 1, 2, 3, 4),
    (5, 6, 7, 8),
    (9, 10, 11, 12),
    (13, 14, 15, 16),
    (17, 18, 19, 20),
)


def _slerp_direction(first: np.ndarray, second: np.ndarray, alpha: float) -> np.ndarray:
    """Spherical interpolation of two bone directions."""
    first_norm = float(np.linalg.norm(first))
    second_norm = float(np.linalg.norm(second))
    if first_norm < 1e-8 or second_norm < 1e-8:
        return (1.0 - alpha) * first + alpha * second
    first = first / first_norm
    second = second / second_norm
    cosine = float(np.clip(np.dot(first, second), -1.0, 1.0))
    angle = float(np.arccos(cosine))
    if angle < 1e-6 or np.sin(angle) < 1e-6:
        value = (1.0 - alpha) * first + alpha * second
    else:
        value = (
            np.sin((1.0 - alpha) * angle) / np.sin(angle) * first
            + np.sin(alpha * angle) / np.sin(angle) * second
        )
    norm = float(np.linalg.norm(value))
    return value / norm if norm > 1e-8 else first


def _geodesic_hand_pose(
    previous: np.ndarray, following: np.ndarray, alpha: float,
) -> np.ndarray:
    """Interpolate a hand by rigid bone directions, never by free joint lines.

    Palm anchors follow a linear rigid translation/orientation proxy, while
    every finger segment follows a spherical direction interpolation and a
    linearly interpolated *length*.  Because children are reconstructed from
    their parent, a gap cannot create the familiar independently-interpolated
    ``twisted finger`` shape.  Missing endpoint joints fall back to the
    available endpoint value.
    """
    first = np.asarray(previous, dtype=np.float64)
    second = np.asarray(following, dtype=np.float64)
    result = np.full((21, 3), np.nan, dtype=np.float64)
    finite_first = np.isfinite(first).all(axis=1)
    finite_second = np.isfinite(second).all(axis=1)
    both = finite_first & finite_second
    result[both] = (1.0 - alpha) * first[both] + alpha * second[both]
    result[finite_first & ~finite_second] = first[finite_first & ~finite_second]
    result[finite_second & ~finite_first] = second[finite_second & ~finite_first]

    for chain in _GEO_CHAINS:
        root = chain[0]
        if not np.isfinite(result[root]).all():
            continue
        for parent, child in zip(chain[:-1], chain[1:]):
            first_vector = first[child] - first[parent]
            second_vector = second[child] - second[parent]
            first_ok = np.isfinite(first_vector).all()
            second_ok = np.isfinite(second_vector).all()
            if first_ok and second_ok and np.isfinite(result[parent]).all():
                first_length = float(np.linalg.norm(first_vector))
                second_length = float(np.linalg.norm(second_vector))
                if first_length > 1e-8 and second_length > 1e-8:
                    direction = _slerp_direction(first_vector, second_vector, alpha)
                    length = (1.0 - alpha) * first_length + alpha * second_length
                    result[child] = result[parent] + length * direction
                    continue
            # A one-sided fingertip or malformed edge is not inventable.  Keep
            # the endpoint value already selected above.
    return result


def _interpolate_all_gap_results(
    rows: list[dict[str, str]],
    accepted: dict[int, dict],
    *,
    max_gap_frames: int = 0,
    max_step_m: float = 0.0,
    min_hand_joints: int = 18,
    prediction_mode: str = "hold",
) -> dict[int, dict]:
    """Build a dense, explicitly predicted display track between observations.

    This is intentionally a rendering aid, not a new measurement: accepted
    and rejected JSONL files are never changed.  Each physical-side track is
    interpolated directly between its two nearest *valid* fused poses, even
    when the other side or the whole frame was rejected.  This removes the
    blank-frame-to-reacquisition teleport that is visually distracting in a
    review video.  The resulting rows are marked ``temporal_gap_prediction``
    and the renderer labels them ``PREDICTED`` so they cannot be mistaken for
    a multi-view measurement.
    """
    if not accepted:
        return {}
    if max_gap_frames < 0 or max_step_m < 0.0:
        return {}
    if prediction_mode not in ("hold", "hold-previous", "linear", "geodesic"):
        raise ValueError(
            "prediction_mode must be 'hold', 'hold-previous', 'linear', or 'geodesic'"
        )
    ordered_ids = [int(row["sync_index"]) for row in rows]
    ordered_id_set = set(ordered_ids)

    def _points(hand: dict, key: str) -> np.ndarray:
        value = np.asarray(hand.get(key, []), dtype=np.float64)
        if value.shape != (21, 3):
            return np.full((21, 3), np.nan, dtype=np.float64)
        return value

    def _valid_hand(hand: dict | None) -> bool:
        if hand is None:
            return False
        points = _points(hand, "joints_base_m")
        finite = np.isfinite(points).all(axis=1)
        # A count-only gate can accept a malformed skeleton with its wrist
        # missing (for example 18 fingertips but no palm root).  Such a pose
        # is not a usable endpoint for temporal display prediction.
        core_indices = (0, 1, 2, 5, 9, 13, 17)
        return (
            int(np.count_nonzero(finite)) >= int(min_hand_joints)
            and bool(finite[list(core_indices)].all())
        )

    def _view_between(
        previous: dict | None, following: dict | None, alpha: float,
    ) -> dict | None:
        if previous is None and following is None:
            return None
        if previous is None:
            view = copy.deepcopy(following)
            view["interpolated"] = True
            return view
        if following is None:
            view = copy.deepcopy(previous)
            view["interpolated"] = True
            return view
        view = copy.deepcopy(previous)
        for key in ("bbox_xyxy", "joints_2d", "wilor_joints_2d"):
            first = np.asarray(previous.get(key, []), dtype=np.float64)
            second = np.asarray(following.get(key, []), dtype=np.float64)
            if first.shape != second.shape or first.size == 0:
                continue
            if not np.isfinite(first).all() or not np.isfinite(second).all():
                continue
            view[key] = ((1.0 - alpha) * first + alpha * second).tolist()
        view["confidence"] = float(
            (1.0 - alpha) * float(previous.get("confidence", 0.0))
            + alpha * float(following.get("confidence", 0.0))
        )
        view["interpolated"] = True
        return view

    # Keep only complete endpoint poses.  Rows that contain a rejected/sparse
    # hand are not allowed to become an interpolation endpoint.
    endpoints: dict[int, list[tuple[int, dict]]] = {0: [], 1: []}
    for sync_index in sorted(int(value) for value in accepted):
        for hand in accepted[sync_index].get("hands", []):
            side = int(hand.get("side", -1))
            if side in endpoints and _valid_hand(hand):
                endpoints[side].append((sync_index, hand))

    predicted: dict[int, dict] = {}
    for side in (0, 1):
        for (previous_id, previous_hand), (following_id, following_hand) in zip(
            endpoints[side], endpoints[side][1:]
        ):
            gap = int(following_id - previous_id - 1)
            if gap <= 0:
                continue
            if max_gap_frames and gap > int(max_gap_frames):
                continue
            previous_points = _points(previous_hand, "joints_base_m")
            following_points = _points(following_hand, "joints_base_m")
            common = (
                np.isfinite(previous_points).all(axis=1)
                & np.isfinite(following_points).all(axis=1)
            )
            previous_finite = np.isfinite(previous_points).all(axis=1)
            following_finite = np.isfinite(following_points).all(axis=1)
            union = previous_finite | following_finite
            # A sparse endpoint can legitimately lose a different fingertip
            # than the other endpoint.  For display-only prediction, accept
            # that case when the overlap is still substantial and the union
            # contains enough joints to draw a complete-looking hand.  The
            # strict fusion gate remains unchanged at ``min_hand_joints``.
            min_common = max(1, int(min_hand_joints) - 2)
            if (
                int(np.count_nonzero(common)) < min_common
                or int(np.count_nonzero(union)) < int(min_hand_joints)
            ):
                continue
            endpoint_steps = np.linalg.norm(
                following_points[common] - previous_points[common], axis=1
            )
            if (
                max_step_m > 0.0
                and (
                    not np.isfinite(endpoint_steps).all()
                    or float(np.max(endpoint_steps))
                    > float(max_step_m) * float(gap + 1)
                )
            ):
                continue
            previous_result = accepted[previous_id]
            following_result = accepted[following_id]
            for offset in range(1, gap + 1):
                sync_index = previous_id + offset
                if sync_index not in ordered_id_set:
                    continue
                alpha = float(offset) / float(gap + 1)
                current = predicted.get(sync_index)
                if current is None:
                    if sync_index in accepted:
                        current = copy.deepcopy(accepted[sync_index])
                    else:
                        current = {
                            "sync_index": sync_index,
                            "anchor_cameras": list(
                                previous_result.get("anchor_cameras", [])
                            ),
                            "active_cameras": list(
                                previous_result.get("active_cameras", [])
                            ),
                            "active_camera_count": int(
                                previous_result.get("active_camera_count", 0)
                            ),
                            "hands": [],
                        }
                # Preserve a real hand measurement if this side is already
                # valid on an accepted row.  Otherwise replace its sparse
                # placeholder with the predicted complete pose.
                kept_hands = []
                for hand in current.get("hands", []):
                    if int(hand.get("side", -1)) != side or _valid_hand(hand):
                        kept_hands.append(hand)
                if any(
                    int(hand.get("side", -1)) == side
                    and _valid_hand(hand)
                    for hand in kept_hands
                ):
                    predicted[sync_index] = current
                    continue
                if prediction_mode in ("hold", "hold-previous"):
                    # A long occlusion is not a measurement.  Blending each
                    # joint independently between potentially different
                    # endpoint articulations creates a visibly twisted hand
                    # (and can make one finger pass through another).  For a
                    # continuous *review* tile, hold the nearest trusted pose
                    # until reacquisition instead.  The row remains explicitly
                    # synthetic and is never written back to fusion JSONL.
                    # ``hold`` switches to the following trusted pose at the
                    # midpoint.  A branch-gated track can request
                    # ``hold-previous`` so one rejected interval keeps one
                    # finger shape throughout instead of changing again in
                    # the middle of a gap.
                    use_previous = prediction_mode == "hold-previous" or alpha < 0.5
                    points = (
                        previous_points.copy()
                        if use_previous else following_points.copy()
                    )
                elif prediction_mode == "geodesic":
                    # Interpolate only the rigid palm anchors and spherical
                    # finger-bone directions.  This is display-only and keeps
                    # each kinematic chain anatomically connected.
                    points = _geodesic_hand_pose(
                        previous_points, following_points, alpha
                    )
                else:
                    points = np.full((21, 3), np.nan, dtype=np.float64)
                    points[common] = (
                        (1.0 - alpha) * previous_points[common]
                        + alpha * following_points[common]
                    )
                    # Keep a one-sided fingertip at its observed endpoint
                    # rather than dropping the entire predicted hand.  Such
                    # points are explicitly synthetic and are never written
                    # to fusion JSONL.
                    previous_only = previous_finite & ~following_finite
                    following_only = following_finite & ~previous_finite
                    points[previous_only] = previous_points[previous_only]
                    points[following_only] = following_points[following_only]
                hand = copy.deepcopy(previous_hand)
                hand["joints_base_m"] = points.tolist()
                for key in ("unrefined_joints_base_m",):
                    first = _points(previous_hand, key)
                    second = _points(following_hand, key)
                    finite = (
                        np.isfinite(first).all(axis=1)
                        & np.isfinite(second).all(axis=1)
                    )
                    if (
                        prediction_mode == "linear"
                        and int(np.count_nonzero(finite)) >= int(min_hand_joints)
                    ):
                        value = np.full((21, 3), np.nan, dtype=np.float64)
                        value[finite] = (
                            (1.0 - alpha) * first[finite]
                            + alpha * second[finite]
                        )
                        first_finite = np.isfinite(first).all(axis=1)
                        second_finite = np.isfinite(second).all(axis=1)
                        value[first_finite & ~second_finite] = first[
                            first_finite & ~second_finite
                        ]
                        value[second_finite & ~first_finite] = second[
                            second_finite & ~first_finite
                        ]
                        hand[key] = value.tolist()
                previous_views = previous_hand.get("views", {})
                following_views = following_hand.get("views", {})
                views: dict[str, dict] = {}
                for camera in sorted(set(previous_views) | set(following_views)):
                    view = _view_between(
                        previous_views.get(camera), following_views.get(camera), alpha
                    )
                    if view is not None:
                        views[camera] = view
                hand["views"] = views
                hand["camera_ids"] = [
                    camera for camera in previous_hand.get("camera_ids", [])
                    if camera in views
                ] or list(views)
                hand["fusion_mode"] = "temporal_gap_prediction"
                hand["temporal_gap_prediction"] = {
                    "applied": True,
                    "from_sync_index": int(previous_id),
                    "to_sync_index": int(following_id),
                    "gap_frames": gap,
                    "alpha": alpha,
                    "prediction_mode": prediction_mode,
                }
                hand["anatomy_refinement"] = {
                    "applied": True,
                    "corrected_joint_count": 0,
                }
                kept_hands.append(hand)
                current["hands"] = sorted(
                    kept_hands, key=lambda value: int(value.get("side", 0))
                )
                current["fusion_mode"] = "temporal_gap_prediction"
                current["interpolated"] = {
                    "applied": True,
                    "from_sync_index": int(previous_id),
                    "to_sync_index": int(following_id),
                }
                predicted[sync_index] = current
    return predicted


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
    point_key: str = "wilor_joints_2d",
    temporal_view_guard: bool = False,
    hide_sparse_3d: bool = False,
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
    stabilized: dict[tuple[int, int, str], dict] = {}
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
                if hide_sparse_3d:
                    base_points = np.asarray(
                        hand.get("joints_base_m", []), dtype=np.float64
                    )
                    if (
                        base_points.shape == (21, 3)
                        and int(np.isfinite(base_points).all(axis=1).sum()) < 18
                    ):
                        # The 3-D hand-level guard intentionally marked this
                        # sample as a dropout.  Do not show a misleading
                        # partial 2-D skeleton in the diagnostic tile.
                        hidden_view = dict(view)
                        hidden_view["inlier_joint_count"] = 0
                        hidden_view["joints_2d"] = []
                        stabilized[(sync_index, side, camera)] = hidden_view
                        continue
                points = np.asarray(
                    view.get(
                        "joints_2d" if is_override else point_key,
                        view.get("joints_2d", []),
                    ),
                    dtype=np.float64,
                )
                if points.shape != (21, 2) or not np.isfinite(points).all():
                    continue
                samples[(side, camera)].append(
                    (ordinal, sync_index, view, points)
                )

    for (side, camera), entries in samples.items():
        for position, (ordinal, sync_index, view, points) in enumerate(entries):
            temporal_jump = False
            previous_points = None
            previous_view = None
            if temporal_view_guard and position > 0:
                previous_ordinal, previous_sync, previous_view, previous_points = entries[position - 1]
                if sync_index - previous_sync != 1:
                    previous_points = None
                    previous_view = None
            if temporal_view_guard and previous_points is not None:
                wrist_jump = float(np.linalg.norm(points[0] - previous_points[0]))
                current_box = np.asarray(view.get("bbox_xyxy", []), dtype=np.float64)
                previous_box = np.asarray(previous_view.get("bbox_xyxy", []), dtype=np.float64)
                current_area = (
                    max(float(current_box[2] - current_box[0]), 0.0)
                    * max(float(current_box[3] - current_box[1]), 0.0)
                    if current_box.shape == (4,) and np.isfinite(current_box).all()
                    else 0.0
                )
                previous_area = (
                    max(float(previous_box[2] - previous_box[0]), 0.0)
                    * max(float(previous_box[3] - previous_box[1]), 0.0)
                    if previous_box.shape == (4,) and np.isfinite(previous_box).all()
                    else 0.0
                )
                area_ratio = current_area / max(previous_area, 1.0)
                temporal_jump = (
                    wrist_jump >= 180.0
                    or (wrist_jump >= 130.0 and area_ratio >= 1.50)
                )
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

            if temporal_jump and previous_points is not None:
                # Keep one unmistakable merged-box sample on the last trusted
                # 2-D pose.  The source 3-D guard marks the same hand missing;
                # this prevents diagnostic/SAM3 tiles from showing a false
                # image-plane jump during that dropout.
                filtered = previous_points.copy()
                stabilized_view = dict(previous_view)
                stabilized_view["temporal_2d_guard"] = {
                    "applied": True,
                    "held_from_sync_index": int(sync_index - 1),
                }
            else:
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
    # Final delivery convention: physical left is cyan (FINAL L), right is
    # yellow (FINAL R).  Keep this independent of the diagnostic detector
    # palette so a label swap cannot make the final-only view ambiguous.
    # OpenCV uses BGR.  Physical left is cyan and physical right is yellow
    # in the delivery convention (the old tuples were accidentally swapped).
    color = (255, 255, 0) if side == 0 else (0, 255, 255)
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


def _draw_final_2d_fallback(
    frame: np.ndarray, view: dict, side: int,
) -> None:
    """Draw an observed camera-space hand when its 3-D pose is unavailable.

    A display-only all-gap prediction can have a perfectly good 2-D detector
    observation while its rays do not determine a trustworthy depth branch.
    Projecting an invented 3-D hand in that case is what creates the visibly
    twisted fingers in the delivery video.  Keep the evidence in the image
    plane instead.  This helper intentionally uses the final physical-side
    palette, but labels the overlay ``2D`` so it cannot be mistaken for a
    measured fused 3-D pose.
    """
    # The native WiLoR image-plane track is the responsive fallback.  SAM3
    # corrections remain available when an older fusion record has no raw
    # copy, but they must not silently replace the A/B native branch.
    points = np.asarray(view.get("wilor_joints_2d", []), dtype=np.float64)
    if points.shape != (21, 2):
        points = np.asarray(view.get("joints_2d", []), dtype=np.float64)
    if points.shape != (21, 2):
        return
    finite = np.isfinite(points).all(axis=1)
    if int(np.count_nonzero(finite)) < 5:
        return
    color = (255, 255, 0) if side == 0 else (0, 255, 255)
    box = np.asarray(view.get("bbox_xyxy", []), dtype=np.float64)
    if box.shape != (4,) or not np.isfinite(box).all():
        visible = points[finite]
        x0, y0 = np.min(visible, axis=0)
        x1, y1 = np.max(visible, axis=0)
        padding = max(12.0, 0.18 * max(float(x1 - x0), float(y1 - y0), 12.0))
        box = np.asarray([x0 - padding, y0 - padding, x1 + padding, y1 + padding])
    rounded = np.zeros((21, 2), dtype=np.int32)
    rounded[finite] = np.rint(points[finite]).astype(np.int32)
    box_i = np.rint(box).astype(np.int32)
    cv2.rectangle(frame, tuple(box_i[:2]), tuple(box_i[2:]), (20, 20, 20), 7, cv2.LINE_AA)
    cv2.rectangle(frame, tuple(box_i[:2]), tuple(box_i[2:]), color, 3, cv2.LINE_AA)
    cv2.putText(
        frame,
        f"2D {'L' if side == 0 else 'R'}",
        (int(box_i[0]), max(28, int(box_i[1]) - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        color,
        2,
        cv2.LINE_AA,
    )
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


def _final_bbox_from_projected_points(
    points: np.ndarray,
    image_shape: tuple[int, ...],
    padding_ratio: float = 0.18,
    min_joints: int = 21,
) -> np.ndarray | None:
    """Return a stable display box derived from the final projected pose.

    Detector boxes are useful evidence in diagnostic mode, but they can jump
    or disappear when one camera changes its detection.  The final-only view
    should use the same fused pose that it draws, so derive its box from the
    finite projected joints and add a modest hand-sized margin.
    """
    values = np.asarray(points, dtype=np.float64)
    if values.shape != (21, 2):
        return None
    finite = np.isfinite(values).all(axis=1)
    if int(np.count_nonzero(finite)) < int(min_joints):
        return None
    height, width = int(image_shape[0]), int(image_shape[1])
    visible = values[finite]
    x0, y0 = np.min(visible, axis=0)
    x1, y1 = np.max(visible, axis=0)
    span = max(float(x1 - x0), float(y1 - y0), 12.0)
    padding = max(12.0, float(padding_ratio) * span)
    box = np.asarray(
        [x0 - padding, y0 - padding, x1 + padding, y1 + padding],
        dtype=np.float64,
    )
    box[[0, 2]] = np.clip(box[[0, 2]], 0.0, max(width - 1, 0))
    box[[1, 3]] = np.clip(box[[1, 3]], 0.0, max(height - 1, 0))
    if box[2] <= box[0] or box[3] <= box[1]:
        return None
    return box


def _has_renderable_final_pose(
    points: np.ndarray, min_joints: int = 21,
) -> bool:
    values = np.asarray(points, dtype=np.float64)
    if values.shape != (21, 3):
        return False
    finite = np.isfinite(values).all(axis=1)
    core_indices = (0, 1, 2, 5, 9, 13, 17)
    return (
        int(np.count_nonzero(finite)) >= int(min_joints)
        and bool(finite[list(core_indices)].all())
    )


def _has_renderable_joint_support(
    hand: dict,
    min_inlier_views: int = 3,
) -> bool:
    """Reject a visually complete pose whose fingers came from sparse rays.

    Temporal/anatomical stages can make a joint finite after its original
    triangulation had only two (or fewer) supporting views.  Such a point can
    look smooth in 3-D while being on the wrong depth branch.  Keep the raw
    image-plane WiLoR observation for display instead of drawing that point as
    a measured final pose.  Older fusion records without support metadata are
    left compatible and pass this check.
    """
    if min_inlier_views <= 0:
        return True
    counts = np.asarray(hand.get("inlier_view_counts", []), dtype=np.float64)
    points = np.asarray(hand.get("joints_base_m", []), dtype=np.float64)
    if counts.shape != (21,) or points.shape != (21, 3):
        return True
    finite = np.isfinite(points).all(axis=1)
    if not finite.any():
        return False
    return bool(np.all(counts[finite] >= float(min_inlier_views)))


def _draw_refined_hand(
    frame: np.ndarray,
    points: np.ndarray,
    side: int,
    corrected_joint_count: int,
    final_only: bool = False,
) -> None:
    if points.shape != (21, 2):
        return
    # Match _draw_final_detection: left/cyan, right/yellow.
    color = (255, 255, 0) if side == 0 else (0, 255, 255)
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
    final_min_valid_joints: int = 21,
    observed_result: dict | None = None,
    final_predicted_2d_fallback: bool = False,
    final_2d_residual_fallback_px: float = 0.0,
    final_2d_residual_percentile: float = 50.0,
    final_wrist_residual_fallback_px: float = 0.0,
    final_min_joint_inlier_views: int = 3,
    final_require_local_view: bool = True,
) -> np.ndarray:
    using_raw_fallback = (
        result is None
        and raw_fallback_result is not None
        and overlay_mode == "cam3-optimized-2d"
        and camera == target_camera
    )
    display_result = raw_fallback_result if using_raw_fallback else result
    final_2d_fallback_count = 0
    observed_hands = {
        int(hand.get("side", -1)): hand
        for hand in (observed_result or {}).get("hands", [])
        if int(hand.get("side", -1)) in (0, 1)
    }
    if display_result is not None:
        for hand in display_result["hands"]:
            side = int(hand.get("side", -1))
            observed_hand = observed_hands.get(side)
            prediction_metadata = hand.get("temporal_gap_prediction", {})
            is_predicted_hand = bool(
                isinstance(prediction_metadata, dict)
                and prediction_metadata.get("applied")
            )
            if overlay_mode == "final-only":
                # A fused hand may remain valid in world coordinates while
                # this particular camera is occluded.  Projecting it anyway
                # makes an old pose appear on a different hand (especially
                # during a laptop/arm occlusion).  For delivery rendering,
                # require local camera evidence; the diagnostic mode still
                # shows all available cross-view information.
                if final_require_local_view and camera not in hand.get("views", {}):
                    continue
                final_points = np.asarray(
                    hand.get("joints_base_m", []), dtype=np.float64
                )
                renderable_final_pose = _has_renderable_final_pose(
                    final_points, min_joints=final_min_valid_joints
                )
                renderable_final_pose = (
                    renderable_final_pose
                    and _has_renderable_joint_support(
                        hand, final_min_joint_inlier_views
                    )
                )
                if final_predicted_2d_fallback and (
                    is_predicted_hand or not renderable_final_pose
                ):
                    # A depth branch can be invalid while the current camera
                    # still has a usable 2-D observation.  Keep that image
                    # evidence instead of projecting an invented 3-D pose.
                    fallback_view = None
                    if observed_hand is not None:
                        fallback_view = observed_hand.get("views", {}).get(camera)
                    if fallback_view is None:
                        fallback_view = hand.get("views", {}).get(camera)
                    if stabilized_views is not None:
                        fallback_view = stabilized_views.get(
                            (sync_index, side, camera), fallback_view
                        )
                    if fallback_view is not None:
                        _draw_final_2d_fallback(frame, fallback_view, side)
                        continue
                if not renderable_final_pose:
                    # A detector box without a trustworthy fused pose is not
                    # a final result.  Hiding the orphan box prevents a
                    # one-frame 3-D dropout from looking like a hand that
                    # teleported while keeping diagnostic/SAM3 2-D evidence
                    # available in their dedicated modes.
                    continue
            anatomy = hand.get("anatomy_refinement", {})
            view = hand["views"].get(camera)
            if (
                overlay_mode in (
                    "diagnostic", "sam3-2d-only", "wilor-2d-stabilized",
                    "cam3-optimized-2d",
                )
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
                if (
                    overlay_mode == "final-only"
                    and final_2d_residual_fallback_px > 0.0
                    and view is not None
                ):
                    observed_points = np.asarray(
                        view.get("wilor_joints_2d", view.get("joints_2d", [])),
                        dtype=np.float64,
                    )
                    if observed_points.shape != (21, 2):
                        observed_points = np.asarray(
                            view.get("joints_2d", []), dtype=np.float64
                        )
                    if observed_points.shape == (21, 2):
                        finite = (
                            np.isfinite(observed_points).all(axis=1)
                            & np.isfinite(refined_pixels).all(axis=1)
                        )
                    else:
                        finite = np.zeros(21, dtype=bool)
                    if int(np.count_nonzero(finite)) >= 5:
                        residuals = np.linalg.norm(
                            refined_pixels[finite] - observed_points[finite],
                            axis=1,
                        )
                        wrist_residual = None
                        if (
                            final_wrist_residual_fallback_px > 0.0
                            and finite[0]
                        ):
                            wrist_residual = float(
                                np.linalg.norm(
                                    refined_pixels[0] - observed_points[0]
                                )
                            )
                        residual = float(np.percentile(
                            residuals,
                            final_2d_residual_percentile,
                        ))
                        wrist_outlier = (
                            wrist_residual is not None
                            and wrist_residual > final_wrist_residual_fallback_px
                        )
                        if residual > final_2d_residual_fallback_px or wrist_outlier:
                            _draw_final_2d_fallback(frame, view, side)
                            final_2d_fallback_count += 1
                            continue
                if overlay_mode == "final-only":
                    final_view = dict(view or {})
                    fused_box = _final_bbox_from_projected_points(
                        refined_pixels, frame.shape,
                        min_joints=final_min_valid_joints,
                    )
                    if fused_box is not None:
                        final_view["bbox_xyxy"] = fused_box.tolist()
                    if final_view.get("bbox_xyxy") is not None:
                        _draw_final_detection(
                            frame, final_view, side, joints_source
                        )
                _draw_refined_hand(
                    frame,
                    refined_pixels,
                    side,
                    int(anatomy.get("corrected_joint_count", 0)),
                    final_only=overlay_mode == "final-only",
                )
    camera_views = (
        [
            hand["views"][camera] for hand in display_result["hands"]
            if camera in hand["views"]
            and (
                overlay_mode != "final-only"
                or (
                    _has_renderable_final_pose(
                        np.asarray(hand.get("joints_base_m", []), dtype=np.float64),
                        min_joints=final_min_valid_joints,
                    )
                )
            )
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
    elif display_result.get("fusion_mode") == "temporal_gap_prediction":
        status, status_color = "PREDICTED", (0, 190, 255)
    elif display_result.get("fusion_mode") == "short_gap_interpolation":
        status, status_color = "INTERPOLATED", (0, 190, 255)
    elif overlay_mode == "final-only":
        if final_2d_fallback_count and camera_views:
            status = "FUSED + 2D FALLBACK"
            status_color = (0, 190, 255)
        elif final_2d_fallback_count:
            status = "2D FALLBACK"
            status_color = (0, 190, 255)
        elif camera_views:
            status = "FINAL FUSED" if joints_source == "refined" else "RAW FUSED"
            status_color = (70, 220, 70)
        else:
            status, status_color = "NO TRUSTED 3D", (0, 190, 255)
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
    if args.short_gap_max_frames < 0:
        raise ValueError("--short-gap-max-frames must be non-negative")
    if args.short_gap_max_step_m <= 0.0:
        raise ValueError("--short-gap-max-step-m must be positive")
    if args.all_gap_max_frames < 0:
        raise ValueError("--all-gap-max-frames must be non-negative")
    if args.all_gap_max_step_m < 0.0:
        raise ValueError("--all-gap-max-step-m must be non-negative")
    if not 1 <= args.final_min_valid_joints <= 21:
        raise ValueError("--final-min-valid-joints must be in [1, 21]")
    if args.final_min_joint_inlier_views < 0:
        raise ValueError("--final-min-joint-inlier-views must be non-negative")
    if args.final_require_local_view not in (0, 1):
        raise ValueError("--final-require-local-view must be 0 or 1")
    if args.final_2d_residual_fallback_px < 0.0:
        raise ValueError("--final-2d-residual-fallback-px must be non-negative")
    if args.final_wrist_residual_fallback_px < 0.0:
        raise ValueError("--final-wrist-residual-fallback-px must be non-negative")
    if not 0.0 <= args.final_2d_residual_percentile <= 100.0:
        raise ValueError("--final-2d-residual-percentile must be in [0, 100]")
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
    display_accepted = accepted
    interpolated_results: dict[int, dict] = {}
    if args.interpolate_all_gaps and args.overlay_mode in ("diagnostic", "final-only"):
        interpolated_results = _interpolate_all_gap_results(
            rows,
            accepted,
            max_gap_frames=args.all_gap_max_frames,
            max_step_m=args.all_gap_max_step_m,
            min_hand_joints=args.final_min_valid_joints,
            prediction_mode=args.all_gap_prediction_mode,
        )
        if interpolated_results:
            display_accepted = dict(accepted)
            display_accepted.update(interpolated_results)
        print(
            "all-gap display prediction: "
            f"{len(interpolated_results)} frames",
            flush=True,
        )
    elif args.interpolate_short_gaps and args.overlay_mode in ("diagnostic", "final-only"):
        interpolated_results = _interpolate_short_gap_results(
            rows,
            accepted,
            max_gap_frames=args.short_gap_max_frames,
            max_step_m=args.short_gap_max_step_m,
        )
        if interpolated_results:
            display_accepted = dict(accepted)
            display_accepted.update(interpolated_results)
        print(
            "short-gap display interpolation: "
            f"{len(interpolated_results)} frames",
            flush=True,
        )
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
    if args.overlay_mode == "wilor-2d-stabilized":
        stabilized_views = _build_2d_stabilized_views(
            rows, display_accepted, cameras,
            args.wilor_2d_smoothing_radius,
            args.wilor_2d_smoothing_strength,
            args.wilor_2d_outlier_threshold_px,
            args.wilor_2d_wrist_outlier_threshold_px,
        )
    elif args.overlay_mode in ("diagnostic", "sam3-2d-only"):
        stabilized_views = _build_2d_stabilized_views(
            rows, display_accepted, cameras,
            args.wilor_2d_smoothing_radius,
            args.wilor_2d_smoothing_strength,
            args.wilor_2d_outlier_threshold_px,
            args.wilor_2d_wrist_outlier_threshold_px,
            point_key="joints_2d",
            temporal_view_guard=True,
            hide_sparse_3d=True,
        )
    elif args.overlay_mode == "final-only" and args.final_predicted_2d_fallback:
        # Build this track from the real accepted rows.  A hand may fail
        # triangulation while its current image-plane observations remain
        # usable; display interpolation must not replace those observations.
        stabilized_views = _build_2d_stabilized_views(
            rows, accepted, cameras,
            args.wilor_2d_smoothing_radius,
            args.wilor_2d_smoothing_strength,
            args.wilor_2d_outlier_threshold_px,
            args.wilor_2d_wrist_outlier_threshold_px,
            point_key="joints_2d",
            temporal_view_guard=True,
            hide_sparse_3d=False,
        )
    else:
        stabilized_views = None
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
                result = display_accepted.get(sync_index)
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
                            args.final_min_valid_joints,
                            accepted.get(sync_index),
                            bool(args.final_predicted_2d_fallback),
                            float(args.final_2d_residual_fallback_px),
                            float(args.final_2d_residual_percentile),
                            float(args.final_wrist_residual_fallback_px),
                            int(args.final_min_joint_inlier_views),
                            bool(args.final_require_local_view),
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
