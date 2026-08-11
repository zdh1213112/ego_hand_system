#!/usr/bin/env python3
"""Detect EGO stereo hand landmarks, associate hands, and triangulate 3D joints."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("MPLCONFIGDIR", "/tmp/ego-hand-matplotlib")

import cv2
import mediapipe as mp
import numpy as np

from mediapipe_left_baseline import (
    HAND_CONNECTIONS,
    create_stereo_rectification,
    handedness_for,
    read_timestamps,
    unique_file,
)


FINGERTIP_INDICES = np.asarray((4, 8, 12, 16, 20), dtype=np.int64)
NON_FINGERTIP_INDICES = np.asarray(
    [index for index in range(21) if index not in set(FINGERTIP_INDICES.tolist())],
    dtype=np.int64,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MediaPipe on paired EGO stereo frames and triangulate 21 hand landmarks."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--session", type=Path, help="legacy Orbbec EGO KB session")
    source.add_argument(
        "--rectified-dataset", type=Path,
        help="model-independent rectified stereo dataset",
    )
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--balance", type=float, default=0.0)
    parser.add_argument("--num-hands", type=int, default=2)
    parser.add_argument("--max-delta-us", type=int, default=1500)
    parser.add_argument("--min-detection", type=float, default=0.35)
    parser.add_argument("--min-presence", type=float, default=0.35)
    parser.add_argument("--min-tracking", type=float, default=0.35)
    parser.add_argument("--max-epipolar-px", type=float, default=12.0)
    parser.add_argument("--min-depth-m", type=float, default=0.15)
    parser.add_argument("--max-depth-m", type=float, default=3.0)
    parser.add_argument("--max-reprojection-px", type=float, default=8.0)
    parser.add_argument(
        "--track-max-missed", type=int, default=75,
        help="keep a hand identity through this many unmatched paired frames",
    )
    parser.add_argument(
        "--track-max-distance-px", type=float, default=280.0,
        help="normal maximum palm-center distance for track association",
    )
    parser.add_argument(
        "--track-reacquire-distance-px", type=float, default=700.0,
        help="maximum palm-center distance when reacquiring a long-missing hand",
    )
    refinement = parser.add_mutually_exclusive_group()
    refinement.add_argument(
        "--stereo-refine", action="store_true", dest="stereo_refine",
        help="refine stereo landmarks with epipolar-constrained subpixel LK matching",
    )
    refinement.add_argument(
        "--no-stereo-refine", action="store_false", dest="stereo_refine",
        help="triangulate the two independent MediaPipe predictions without local refinement",
    )
    parser.set_defaults(stereo_refine=True)
    parser.add_argument("--refine-window", type=int, default=17)
    parser.add_argument("--refine-tip-window", type=int, default=25)
    parser.add_argument("--refine-max-x-shift-px", type=float, default=18.0)
    parser.add_argument("--refine-max-y-residual-px", type=float, default=3.0)
    parser.add_argument("--refine-max-fb-error-px", type=float, default=2.0)
    parser.add_argument("--refine-max-lk-error", type=float, default=32.0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument("--no-video", action="store_true")
    return parser.parse_args()


def pair_timestamps(left: list[int], right: list[int], max_delta_us: int) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    left_index = 0
    right_index = 0
    while left_index < len(left) and right_index < len(right):
        delta = right[right_index] - left[left_index]
        if abs(delta) <= max_delta_us:
            pairs.append((left_index, right_index))
            left_index += 1
            right_index += 1
        elif delta > 0:
            left_index += 1
        else:
            right_index += 1
    return pairs


def read_frame_at(capture: cv2.VideoCapture, target: int, state: list[int]) -> np.ndarray:
    frame = None
    while state[0] <= target:
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError(f"video ended before frame {target}")
        state[0] += 1
    if frame is None:
        raise RuntimeError(f"failed to decode frame {target}")
    return frame


def detect(landmarker, image_bgr: np.ndarray, timestamp_ms: int, image_size: tuple[int, int]):
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
    result = landmarker.detect_for_video(image, timestamp_ms)
    width, height = image_size
    observations = []
    for index, landmarks in enumerate(result.hand_landmarks):
        label, score = handedness_for(result, index)
        pixels = np.asarray(
            [[point.x * (width - 1), point.y * (height - 1)] for point in landmarks],
            dtype=np.float64,
        )
        observations.append({"index": index, "label": label, "score": score, "pixels": pixels})
    return observations


def stereo_candidate(left: dict, right: dict) -> dict:
    delta_y = np.abs(left["pixels"][:, 1] - right["pixels"][:, 1])
    disparity = left["pixels"][:, 0] - right["pixels"][:, 0]
    plausible = (delta_y < 60.0) & (disparity > 2.0) & (disparity < 300.0)
    median_y = float(np.median(delta_y))
    median_disparity = float(np.median(disparity))
    plausible_count = int(np.count_nonzero(plausible))
    label_penalty = 0.0
    if left["label"] != right["label"] and min(left["score"], right["score"]) > 0.75:
        label_penalty = 8.0
    invalid_penalty = float(21 - plausible_count) * 2.0
    recovery_penalty = 40.0 if left.get("recovered") or right.get("recovered") else 0.0
    cost = median_y + label_penalty + invalid_penalty + recovery_penalty
    accepted = plausible_count >= 12 and median_y <= 35.0 and 2.0 < median_disparity < 250.0
    return {
        "left": left,
        "right": right,
        "cost": cost,
        "accepted": accepted,
        "median_y": median_y,
        "median_disparity": median_disparity,
        "plausible_count": plausible_count,
    }


def associate_hands(left_hands: list[dict], right_hands: list[dict]) -> list[dict]:
    if not left_hands or not right_hands:
        return []
    candidates = {
        (left_index, right_index): stereo_candidate(left, right)
        for left_index, left in enumerate(left_hands)
        for right_index, right in enumerate(right_hands)
    }
    best: tuple[int, float, list[dict]] = (-1, float("inf"), [])
    maximum = min(len(left_hands), len(right_hands))
    for match_count in range(1, maximum + 1):
        for left_subset in itertools.combinations(range(len(left_hands)), match_count):
            for right_subset in itertools.combinations(range(len(right_hands)), match_count):
                for right_order in itertools.permutations(right_subset):
                    selected = [candidates[pair] for pair in zip(left_subset, right_order)]
                    if not all(candidate["accepted"] for candidate in selected):
                        continue
                    total_cost = sum(candidate["cost"] for candidate in selected)
                    score = (match_count, -total_cost)
                    if score > (best[0], -best[1]):
                        best = (match_count, total_cost, selected)
    return best[2]


def predict_landmarks_lk(previous_gray: np.ndarray, current_gray: np.ndarray,
                         pixels: np.ndarray, window: int = 25) -> np.ndarray | None:
    """Track one view's previous hand landmarks into the current image."""
    if previous_gray is None or current_gray is None:
        return None
    points = np.asarray(pixels, dtype=np.float32).reshape(-1, 1, 2)
    if len(points) != 21 or not np.isfinite(points).all():
        return None
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.01)
    tracked, status, error = cv2.calcOpticalFlowPyrLK(
        previous_gray, current_gray, points, None,
        winSize=(max(9, int(window) | 1), max(9, int(window) | 1)),
        maxLevel=2, criteria=criteria, minEigThreshold=1e-4,
    )
    if tracked is None or status is None or error is None:
        return None
    tracked = tracked[:, 0].astype(np.float64)
    status = status[:, 0].astype(bool)
    error = error[:, 0]
    height, width = current_gray.shape[:2]
    usable = (
        status & np.isfinite(tracked).all(axis=1) & np.isfinite(error)
        & (error <= 80.0)
        & (tracked[:, 0] >= 0.0) & (tracked[:, 0] < width)
        & (tracked[:, 1] >= 0.0) & (tracked[:, 1] < height)
    )
    if np.count_nonzero(usable) < 12:
        return None
    # Keep the whole semantic hand so triangulation can use the valid subset.
    # Failed points retain their previous location; the recovered candidate is
    # low-confidence and geometric checks still decide whether each point is used.
    result = tracked.copy()
    result[~usable] = points[:, 0, :].astype(np.float64, copy=False)[~usable]
    return result


def prepare_stereo_matching_images(
    left_rectified: np.ndarray, right_rectified: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Create low-cost grayscale images used only for sparse stereo refinement."""
    if left_rectified.ndim == 3:
        left_gray = cv2.cvtColor(left_rectified, cv2.COLOR_BGR2GRAY)
    else:
        left_gray = left_rectified
    if right_rectified.ndim == 3:
        right_gray = cv2.cvtColor(right_rectified, cv2.COLOR_BGR2GRAY)
    else:
        right_gray = right_rectified
    return np.ascontiguousarray(left_gray), np.ascontiguousarray(right_gray)


def _refine_lk_group(
    left_gray: np.ndarray,
    right_gray: np.ndarray,
    left_points: np.ndarray,
    right_points: np.ndarray,
    indices: np.ndarray,
    window: int,
    args: argparse.Namespace,
) -> dict[str, np.ndarray]:
    count = len(indices)
    empty_float = np.full(count, np.nan, dtype=np.float64)
    empty_bool = np.zeros(count, dtype=bool)
    if count == 0:
        return {
            "points": np.empty((0, 2), dtype=np.float64), "accepted": empty_bool,
            "quality": empty_float, "forward_error": empty_float,
            "fb_error": empty_float, "vertical_residual": empty_float,
            "x_shift": empty_float,
        }

    window = max(7, int(window) | 1)
    start = np.asarray(left_points[indices], dtype=np.float32).reshape(-1, 1, 2)
    initial = np.asarray(right_points[indices], dtype=np.float32).copy()
    # Rectified correspondences should lie on the same row. MediaPipe supplies the
    # semantic x initial value; the left landmark supplies the epipolar y initial value.
    initial[:, 1] = left_points[indices, 1]
    initial = initial.reshape(-1, 1, 2)
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        24,
        0.005,
    )
    refined, status, forward_error = cv2.calcOpticalFlowPyrLK(
        left_gray, right_gray, start, initial,
        winSize=(window, window), maxLevel=0, criteria=criteria,
        flags=cv2.OPTFLOW_USE_INITIAL_FLOW, minEigThreshold=1e-4,
    )
    if refined is None or status is None or forward_error is None:
        return {
            "points": initial[:, 0].astype(np.float64), "accepted": empty_bool,
            "quality": np.zeros(count, dtype=np.float64),
            "forward_error": empty_float, "fb_error": empty_float,
            "vertical_residual": empty_float, "x_shift": empty_float,
        }

    backward_initial = start.copy()
    backward, backward_status, _ = cv2.calcOpticalFlowPyrLK(
        right_gray, left_gray, refined, backward_initial,
        winSize=(window, window), maxLevel=0, criteria=criteria,
        flags=cv2.OPTFLOW_USE_INITIAL_FLOW, minEigThreshold=1e-4,
    )
    refined = refined[:, 0].astype(np.float64)
    status = status[:, 0].astype(bool)
    # OpenCV may leave NaN/sentinel values for points whose LK status is false.
    # Keep the native floating dtype here; the status/finite mask rejects them below.
    forward_error = forward_error[:, 0].copy()
    forward_error = np.where(
        np.isfinite(forward_error),
        np.clip(forward_error, 0.0, 1.0e4),
        np.inf,
    ).astype(np.float64, copy=False)
    if backward is None or backward_status is None:
        backward = np.full_like(refined, np.nan)
        backward_status = np.zeros(count, dtype=bool)
    else:
        backward = backward[:, 0].astype(np.float64)
        backward_status = backward_status[:, 0].astype(bool)

    fb_error = np.linalg.norm(backward - left_points[indices], axis=1)
    vertical_residual = np.abs(refined[:, 1] - left_points[indices, 1])
    x_shift = np.abs(refined[:, 0] - right_points[indices, 0])
    disparity = left_points[indices, 0] - refined[:, 0]
    finite = (
        np.all(np.isfinite(refined), axis=1)
        & np.isfinite(forward_error)
        & np.isfinite(fb_error)
    )
    accepted = (
        status & backward_status & finite
        & (forward_error <= float(getattr(args, "refine_max_lk_error", 32.0)))
        & (fb_error <= float(getattr(args, "refine_max_fb_error_px", 2.0)))
        & (vertical_residual <= float(getattr(args, "refine_max_y_residual_px", 3.0)))
        & (x_shift <= float(getattr(args, "refine_max_x_shift_px", 18.0)))
        & (disparity > 2.0) & (disparity < 300.0)
    )

    photometric = np.exp(-np.square(np.minimum(forward_error, 240.0) / 24.0))
    forward_backward = np.exp(-np.square(fb_error / 1.25))
    epipolar = np.exp(-np.square(vertical_residual / 1.75))
    quality = np.cbrt(np.clip(photometric * forward_backward * epipolar, 0.0, 1.0))
    quality[~accepted] = 0.0
    return {
        "points": refined,
        "accepted": accepted,
        "quality": quality,
        "forward_error": forward_error,
        "fb_error": fb_error,
        "vertical_residual": vertical_residual,
        "x_shift": x_shift,
    }


def refine_stereo_correspondences(
    left_points: np.ndarray,
    right_points: np.ndarray,
    matching_images: tuple[np.ndarray, np.ndarray] | None,
    args: argparse.Namespace,
) -> dict[str, np.ndarray]:
    """Refine right-view pixels and symmetrically enforce the rectified epipolar row."""
    raw_left = np.asarray(left_points, dtype=np.float64)
    raw_right = np.asarray(right_points, dtype=np.float64)
    refined_left = raw_left.copy()
    refined_right = raw_right.copy()
    attempted = np.zeros(21, dtype=bool)
    accepted = np.zeros(21, dtype=bool)
    quality = np.zeros(21, dtype=np.float64)
    forward_error = np.full(21, np.nan, dtype=np.float64)
    fb_error = np.full(21, np.nan, dtype=np.float64)
    vertical_residual = np.full(21, np.nan, dtype=np.float64)
    x_shift = np.full(21, np.nan, dtype=np.float64)

    enabled = bool(getattr(args, "stereo_refine", True))
    if matching_images is None or not enabled:
        return {
            "left_points": refined_left, "right_points": refined_right,
            "attempted": attempted, "accepted": accepted, "quality": quality,
            "forward_error": forward_error, "fb_error": fb_error,
            "vertical_residual": vertical_residual, "x_shift": x_shift,
        }

    left_gray, right_gray = matching_images
    groups = (
        (NON_FINGERTIP_INDICES, int(getattr(args, "refine_window", 17))),
        (FINGERTIP_INDICES, int(getattr(args, "refine_tip_window", 25))),
    )
    for indices, window in groups:
        result = _refine_lk_group(
            left_gray, right_gray, raw_left, raw_right, indices, window, args
        )
        attempted[indices] = True
        accepted[indices] = result["accepted"]
        quality[indices] = result["quality"]
        forward_error[indices] = result["forward_error"]
        fb_error[indices] = result["fb_error"]
        vertical_residual[indices] = result["vertical_residual"]
        x_shift[indices] = result["x_shift"]
        for local_index, landmark_index in enumerate(indices):
            if not result["accepted"][local_index]:
                continue
            # The symmetric row correction is the least-squares adjustment that
            # enforces the horizontal epipolar constraint without moving x in the
            # left reference view.
            shared_y = 0.5 * (
                raw_left[landmark_index, 1] + result["points"][local_index, 1]
            )
            refined_left[landmark_index, 1] = shared_y
            refined_right[landmark_index] = (
                result["points"][local_index, 0], shared_y
            )

    return {
        "left_points": refined_left, "right_points": refined_right,
        "attempted": attempted, "accepted": accepted, "quality": quality,
        "forward_error": forward_error, "fb_error": fb_error,
        "vertical_residual": vertical_residual, "x_shift": x_shift,
    }


def triangulate(
    candidate: dict,
    rectification: dict,
    args: argparse.Namespace,
    matching_images: tuple[np.ndarray, np.ndarray] | None = None,
) -> dict:
    raw_left_points = np.asarray(candidate["left"]["pixels"], dtype=np.float64)
    raw_right_points = np.asarray(candidate["right"]["pixels"], dtype=np.float64)
    refinement = refine_stereo_correspondences(
        raw_left_points, raw_right_points, matching_images, args
    )
    left_points = refinement["left_points"]
    right_points = refinement["right_points"]
    homogeneous = cv2.triangulatePoints(
        rectification["p1"], rectification["p2"], left_points.T, right_points.T
    )
    points_rectified = (homogeneous[:3] / homogeneous[3]).T
    points_left = (rectification["r1"].T @ points_rectified.T).T

    left_projection = (rectification["p1"] @ np.column_stack((points_rectified, np.ones(21))).T).T
    right_projection = (rectification["p2"] @ np.column_stack((points_rectified, np.ones(21))).T).T
    left_reprojected = left_projection[:, :2] / left_projection[:, 2:3]
    right_reprojected = right_projection[:, :2] / right_projection[:, 2:3]
    reprojection = np.sqrt(
        (
            np.sum((left_reprojected - left_points) ** 2, axis=1)
            + np.sum((right_reprojected - right_points) ** 2, axis=1)
        ) / 2.0
    )
    disparity = left_points[:, 0] - right_points[:, 0]
    epipolar = np.abs(left_points[:, 1] - right_points[:, 1])
    raw_epipolar = np.abs(raw_left_points[:, 1] - raw_right_points[:, 1])
    finite = np.all(np.isfinite(points_rectified), axis=1) & np.isfinite(reprojection)
    valid = (
        finite
        & (disparity > 2.0)
        & (epipolar <= args.max_epipolar_px)
        & (points_rectified[:, 2] >= args.min_depth_m)
        & (points_rectified[:, 2] <= args.max_depth_m)
        & (reprojection <= args.max_reprojection_px)
    )
    palm_indices = np.asarray([0, 5, 9, 13, 17])
    valid_palm = palm_indices[valid[palm_indices]]
    if len(valid_palm) >= 2:
        center = np.median(points_left[valid_palm], axis=0)
    elif np.count_nonzero(valid) >= 3:
        center = np.median(points_left[valid], axis=0)
    else:
        center = np.array([np.nan, np.nan, np.nan])
    left_2d_quality = np.full(21, float(candidate["left"]["score"]), dtype=np.float64)
    right_2d_quality = np.full(21, float(candidate["right"]["score"]), dtype=np.float64)
    attempted = refinement["attempted"]
    used = refinement["accepted"]
    refined_quality = refinement["quality"]
    if np.any(attempted):
        left_2d_quality[attempted & used] *= 0.75 + 0.25 * refined_quality[attempted & used]
        right_2d_quality[attempted & used] *= 0.35 + 0.65 * refined_quality[attempted & used]
        left_2d_quality[attempted & ~used] *= 0.60
        right_2d_quality[attempted & ~used] *= 0.25
        failed_tips = attempted & ~used
        failed_tips[NON_FINGERTIP_INDICES] = False
        left_2d_quality[failed_tips] *= 0.75
        right_2d_quality[failed_tips] *= 0.60
    left_2d_quality *= np.exp(-np.square(raw_epipolar / 18.0))
    right_2d_quality *= np.exp(-np.square(raw_epipolar / 14.0))
    return {
        **candidate,
        "left_points": left_points,
        "right_points": right_points,
        "raw_left_points": raw_left_points,
        "raw_right_points": raw_right_points,
        "refinement_attempted": refinement["attempted"],
        "refinement_used": refinement["accepted"],
        "refinement_quality": refinement["quality"],
        "refinement_lk_error": refinement["forward_error"],
        "refinement_fb_error": refinement["fb_error"],
        "refinement_y_residual": refinement["vertical_residual"],
        "refinement_x_shift": refinement["x_shift"],
        "left_2d_quality": np.clip(left_2d_quality, 0.0, 1.0),
        "right_2d_quality": np.clip(right_2d_quality, 0.0, 1.0),
        "points_rectified": points_rectified,
        "points_left": points_left,
        "disparity": disparity,
        "epipolar": epipolar,
        "raw_epipolar": raw_epipolar,
        "reprojection": reprojection,
        "valid": valid,
        "center": center,
        "center_left_px": np.median(left_points[[0, 5, 9, 13, 17]], axis=0),
    }


class TrackManager:
    """Maintain the two hand identities through short out-of-view intervals."""

    def __init__(
        self,
        max_missed: int = 75,
        max_distance_px: float = 280.0,
        reacquire_distance_px: float = 700.0,
        max_tracks: int = 2,
        reacquire_after_missed: int = 12,
    ):
        self.next_id = 0
        self.max_missed = max_missed
        self.max_distance_px = max_distance_px
        self.reacquire_distance_px = max(reacquire_distance_px, max_distance_px)
        self.max_tracks = max_tracks
        self.reacquire_after_missed = reacquire_after_missed
        self.tracks: dict[int, dict] = {}

    def association_cost(self, match: dict, track: dict) -> float | None:
        prediction = track["position"] + track["velocity"] * min(track["missed"], 3)
        distance = float(np.linalg.norm(match["center_left_px"] - prediction))
        reacquiring = track["missed"] > self.reacquire_after_missed
        gate = self.reacquire_distance_px if reacquiring else self.max_distance_px
        if distance > gate:
            return None
        label_penalty = 0.0
        match_label = match["left"]["label"]
        if match_label != track["label"] and match["left"]["score"] > 0.80:
            label_penalty = 80.0
        return distance + label_penalty + min(track["missed"], self.max_missed) * 1.5

    def assign(self, matches: list[dict]) -> None:
        for track in self.tracks.values():
            track["missed"] += 1
        usable = [
            index for index, match in enumerate(matches)
            if np.all(np.isfinite(match["center_left_px"]))
        ]
        track_ids = list(self.tracks)
        assignment: dict[int, int] = {}
        used_track_ids: set[int] = set()
        if usable and track_ids:
            count = min(len(usable), len(track_ids))
            best_cost = float("inf")
            best_pairs = []
            for match_subset in itertools.combinations(usable, count):
                for track_subset in itertools.combinations(track_ids, count):
                    for track_order in itertools.permutations(track_subset):
                        pairs = list(zip(match_subset, track_order))
                        cost = 0.0
                        accepted = True
                        for match_index, track_id in pairs:
                            track = self.tracks[track_id]
                            pair_cost = self.association_cost(matches[match_index], track)
                            if pair_cost is None:
                                accepted = False
                                break
                            cost += pair_cost
                        if accepted and cost < best_cost:
                            best_cost = cost
                            best_pairs = pairs
            assignment.update(best_pairs)
            used_track_ids.update(track_id for _, track_id in best_pairs)
        for match_index, match in enumerate(matches):
            if match_index not in assignment:
                available = [track_id for track_id in self.tracks if track_id not in used_track_ids]
                if len(self.tracks) < self.max_tracks:
                    track_id = self.next_id
                    self.next_id += 1
                    self.tracks[track_id] = {
                        "position": match["center_left_px"].copy(),
                        "velocity": np.zeros(2, dtype=np.float64),
                        "missed": 0,
                        "label": match["left"]["label"],
                        "label_votes": {"Left": 0.0, "Right": 0.0},
                        "left_pixels": np.asarray(match["left"]["pixels"], dtype=np.float64).copy(),
                        "right_pixels": np.asarray(match["right"]["pixels"], dtype=np.float64).copy(),
                    }
                elif available:
                    # The detector is configured for at most two hands. Reuse the
                    # best dormant slot instead of creating fragmented identities.
                    match_label = match["left"]["label"]
                    track_id = min(
                        available,
                        key=lambda candidate: (
                            self.tracks[candidate]["label"] != match_label,
                            -self.tracks[candidate]["missed"],
                            float(np.linalg.norm(
                                match["center_left_px"] - self.tracks[candidate]["position"]
                            )),
                        ),
                    )
                else:
                    continue
                assignment[match_index] = track_id
                used_track_ids.add(track_id)
            track_id = assignment[match_index]
            match["track_id"] = track_id
            track = self.tracks[track_id]
            reacquired = track["missed"] > self.reacquire_after_missed
            new_position = match["center_left_px"]
            observed_velocity = new_position - track["position"]
            if reacquired:
                track["velocity"] = np.zeros(2, dtype=np.float64)
            else:
                track["velocity"] = 0.65 * track["velocity"] + 0.35 * observed_velocity
            track["position"] = new_position.copy()
            track["missed"] = 0
            track["left_pixels"] = np.asarray(match["left"]["pixels"], dtype=np.float64).copy()
            track["right_pixels"] = np.asarray(match["right"]["pixels"], dtype=np.float64).copy()
            votes = track.setdefault("label_votes", {"Left": 0.0, "Right": 0.0})
            votes[match["left"]["label"]] += float(match["left"]["score"])
            track["label"] = max(votes, key=votes.get)
            match["stable_handedness"] = track["label"]
        self.tracks = {
            track_id: track for track_id, track in self.tracks.items()
            if track["missed"] <= self.max_missed
        }

    def recover_missing_views(
        self, left_hands: list[dict], right_hands: list[dict],
        previous_left_gray: np.ndarray | None, previous_right_gray: np.ndarray | None,
        current_left_gray: np.ndarray, current_right_gray: np.ndarray,
    ) -> tuple[list[dict], list[dict]]:
        """Add low-confidence LK candidates when one camera temporarily loses a hand."""
        if not self.tracks:
            return left_hands, right_hands
        need_left = len(left_hands) < len(right_hands)
        need_right = len(right_hands) < len(left_hands)
        if left_hands and right_hands and not (need_left or need_right):
            return left_hands, right_hands
        recovered_left = list(left_hands)
        recovered_right = list(right_hands)
        for track_id, track in self.tracks.items():
            if track["missed"] > 2:
                continue
            if need_left or not left_hands:
                pixels = predict_landmarks_lk(previous_left_gray, current_left_gray, track.get("left_pixels"))
                if pixels is not None:
                    recovered_left.append({
                        "index": -1, "label": track["label"],
                        "score": 0.45, "pixels": pixels,
                        "recovered": True, "recovery_track_id": track_id,
                    })
            if need_right or not right_hands:
                pixels = predict_landmarks_lk(previous_right_gray, current_right_gray, track.get("right_pixels"))
                if pixels is not None:
                    recovered_right.append({
                        "index": -1, "label": track["label"],
                        "score": 0.45, "pixels": pixels,
                        "recovered": True, "recovery_track_id": track_id,
                    })
        return recovered_left, recovered_right


def draw_observation(image: np.ndarray, observation: dict, color: tuple[int, int, int], text: str) -> None:
    points = np.rint(observation["pixels"]).astype(int)
    for start, end in HAND_CONNECTIONS:
        cv2.line(image, tuple(points[start]), tuple(points[end]), color, 2, cv2.LINE_AA)
    for point in points:
        cv2.circle(image, tuple(point), 3, color, -1, cv2.LINE_AA)
    anchor = points[0]
    cv2.putText(image, text, (max(0, anchor[0] - 30), max(25, anchor[1] - 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)


def make_options(args: argparse.Namespace, model: Path):
    return mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=args.num_hands,
        min_hand_detection_confidence=args.min_detection,
        min_hand_presence_confidence=args.min_presence,
        min_tracking_confidence=args.min_tracking,
    )


def main() -> int:
    args = parse_args()
    if args.stride < 1 or args.num_hands < 1 or args.max_delta_us < 0:
        raise ValueError("stride/num-hands must be positive and max-delta-us non-negative")
    model = args.model.resolve()
    output = args.output.resolve()
    if not model.is_file():
        raise FileNotFoundError(f"MediaPipe model does not exist: {model}")
    output.mkdir(parents=True, exist_ok=True)
    left_capture = right_capture = None
    left_state = [0]
    right_state = [0]
    rectified_dataset = None
    rectified_iterator = None
    if args.rectified_dataset is not None:
        from ego_data.dataset import RectifiedStereoDataset
        rectified_dataset = RectifiedStereoDataset(args.rectified_dataset)
        rectification = rectified_dataset.rectification
        image_size = rectification["image_size"]
        source_pairs = [{
            "pair_index": int(row["pair_index"]),
            "left_index": int(row["left_frame_index"]),
            "right_index": int(row["right_frame_index"]),
            "left_timestamp_us": int(row["left_timestamp_ns"]) // 1000,
            "right_timestamp_us": int(row["right_timestamp_ns"]) // 1000,
            "row": row,
        } for row in rectified_dataset.pairs]
        source_description = str(rectified_dataset.root)
        source_kind = "rectified_dataset"
        left_times = [item["left_timestamp_us"] for item in source_pairs]
        source_fps = 1e6 / float(np.median(np.diff(left_times))) if len(left_times) > 1 else 30.0
        rectified_iterator = iter(rectified_dataset)
    else:
        session = args.session.resolve()
        if not session.is_dir():
            raise FileNotFoundError(f"session does not exist: {session}")
        calibration_path = unique_file(session, "_calibration_camera.yaml")
        left_video = unique_file(session, "_camera_left.mp4")
        right_video = unique_file(session, "_camera_right.mp4")
        left_timestamps = read_timestamps(unique_file(session, "_camera_left_pts.csv"))
        right_timestamps = read_timestamps(unique_file(session, "_camera_right_pts.csv"))
        timestamp_pairs = pair_timestamps(left_timestamps, right_timestamps, args.max_delta_us)
        source_pairs = [{
            "pair_index": pair_index,
            "left_index": left_index,
            "right_index": right_index,
            "left_timestamp_us": left_timestamps[left_index],
            "right_timestamp_us": right_timestamps[right_index],
        } for pair_index, (left_index, right_index) in enumerate(timestamp_pairs)]
        rectification = create_stereo_rectification(calibration_path, args.balance)
        image_size = rectification["image_size"]
        left_capture = cv2.VideoCapture(str(left_video))
        right_capture = cv2.VideoCapture(str(right_video))
        if not left_capture.isOpened() or not right_capture.isOpened():
            raise RuntimeError("cannot open one or both stereo videos")
        source_description = str(session)
        source_kind = "legacy_session"
        source_fps = float(left_capture.get(cv2.CAP_PROP_FPS))

    video_path = output / "stereo_annotated.mp4"
    writer = None
    preview_size = (image_size[0], image_size[1] // 2)
    if not args.no_video:
        fps = (source_fps if source_fps > 0 else 30.0) / args.stride
        writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, preview_size)
        if not writer.isOpened():
            raise RuntimeError(f"cannot create {video_path}")

    frames_path = output / "stereo_frames.csv"
    landmarks_path = output / "stereo_landmarks_3d.csv"
    frame_fields = [
        "pair_index", "left_index", "right_index", "left_timestamp_us", "right_timestamp_us",
        "timestamp_delta_us", "left_hands", "right_hands", "matched_hands", "valid_3d_points",
        "refined_points", "median_refinement_quality",
        "median_epipolar_px", "median_reprojection_px",
    ]
    landmark_fields = [
        "pair_index", "left_index", "right_index", "track_id", "match_index",
        "left_hand_index", "right_hand_index", "left_handedness", "right_handedness",
        "left_handedness_score", "right_handedness_score", "landmark_index",
        "raw_left_x_rectified_px", "raw_left_y_rectified_px",
        "raw_right_x_rectified_px", "raw_right_y_rectified_px",
        "left_x_rectified_px", "left_y_rectified_px", "right_x_rectified_px", "right_y_rectified_px",
        "refinement_attempted", "refinement_used", "refinement_quality",
        "refinement_lk_error", "refinement_fb_error_px", "refinement_y_residual_px",
        "refinement_x_shift_px",
        "disparity_px", "epipolar_error_px", "valid_3d", "reprojection_error_px",
        "x_rectified_m", "y_rectified_m", "z_rectified_m", "x_left_camera_m", "y_left_camera_m", "z_left_camera_m",
    ]

    tracker = TrackManager(
        max_missed=args.track_max_missed,
        max_distance_px=args.track_max_distance_px,
        reacquire_distance_px=args.track_reacquire_distance_px,
        max_tracks=args.num_hands,
    )
    processed_pairs = 0
    pairs_with_matches = 0
    matched_hand_instances = 0
    valid_points_total = 0
    observed_track_ids = set()
    valid_points_by_landmark = np.zeros(21, dtype=np.int64)
    observations_by_landmark = np.zeros(21, dtype=np.int64)
    all_valid_epipolar = []
    all_valid_reprojection = []
    all_valid_depth = []
    all_refinement_quality = []
    refinement_attempted_total = 0
    refinement_used_total = 0
    start_time = time.perf_counter()
    first_timestamp_us = min(
        source_pairs[0]["left_timestamp_us"], source_pairs[0]["right_timestamp_us"]
    )
    last_left_ms = -1
    last_right_ms = -1
    previous_left_gray = None
    previous_right_gray = None

    with frames_path.open("w", encoding="utf-8", newline="") as frame_stream, \
         landmarks_path.open("w", encoding="utf-8", newline="") as landmark_stream, \
         mp.tasks.vision.HandLandmarker.create_from_options(make_options(args, model)) as left_landmarker, \
         mp.tasks.vision.HandLandmarker.create_from_options(make_options(args, model)) as right_landmarker:
        frame_writer = csv.DictWriter(frame_stream, fieldnames=frame_fields)
        landmark_writer = csv.DictWriter(landmark_stream, fieldnames=landmark_fields)
        frame_writer.writeheader()
        landmark_writer.writeheader()

        for pair_ordinal, pair in enumerate(source_pairs):
            pair_index = pair["pair_index"]
            left_index = pair["left_index"]
            right_index = pair["right_index"]
            left_timestamp_us = pair["left_timestamp_us"]
            right_timestamp_us = pair["right_timestamp_us"]
            if rectified_dataset is not None:
                decoded_pair = next(rectified_iterator)
                if decoded_pair["pair_index"] != pair_index:
                    raise RuntimeError("rectified video and pair index are inconsistent")
                left_rectified = decoded_pair["left_image"]
                right_rectified = decoded_pair["right_image"]
            else:
                left_frame = read_frame_at(left_capture, left_index, left_state)
                right_frame = read_frame_at(right_capture, right_index, right_state)
                if left_frame.shape[1::-1] != image_size or right_frame.shape[1::-1] != image_size:
                    raise RuntimeError("decoded resolution differs from calibration")
                left_rectified = cv2.remap(
                    left_frame, rectification["map_left_x"], rectification["map_left_y"], cv2.INTER_LINEAR
                )
                right_rectified = cv2.remap(
                    right_frame, rectification["map_right_x"], rectification["map_right_y"], cv2.INTER_LINEAR
                )
            if pair_ordinal % args.stride != 0:
                continue
            if args.max_pairs > 0 and processed_pairs >= args.max_pairs:
                break
            left_ms = max(int((left_timestamp_us - first_timestamp_us) // 1000), last_left_ms + 1)
            right_ms = max(int((right_timestamp_us - first_timestamp_us) // 1000), last_right_ms + 1)
            last_left_ms = left_ms
            last_right_ms = right_ms
            left_hands = detect(left_landmarker, left_rectified, left_ms, image_size)
            right_hands = detect(right_landmarker, right_rectified, right_ms, image_size)
            matching_images = (
                prepare_stereo_matching_images(left_rectified, right_rectified)
                if args.stereo_refine else None
            )
            current_left_gray, current_right_gray = prepare_stereo_matching_images(
                left_rectified, right_rectified
            )
            left_hands, right_hands = tracker.recover_missing_views(
                left_hands, right_hands,
                previous_left_gray, previous_right_gray,
                current_left_gray, current_right_gray,
            )
            matches = [
                triangulate(candidate, rectification, args, matching_images)
                for candidate in associate_hands(left_hands, right_hands)
            ]
            tracker.assign(matches)
            previous_left_gray = current_left_gray
            previous_right_gray = current_right_gray

            frame_valid = int(sum(np.count_nonzero(match["valid"]) for match in matches))
            frame_refined = int(sum(np.count_nonzero(match["refinement_used"]) for match in matches))
            frame_refinement_quality = np.concatenate([
                match["refinement_quality"][match["refinement_used"]] for match in matches
            ]) if matches and frame_refined else np.array([])
            valid_epipolar = np.concatenate([match["epipolar"][match["valid"]] for match in matches]) if matches else np.array([])
            valid_reprojection = np.concatenate([match["reprojection"][match["valid"]] for match in matches]) if matches else np.array([])
            frame_writer.writerow({
                "pair_index": pair_index,
                "left_index": left_index,
                "right_index": right_index,
                "left_timestamp_us": left_timestamp_us,
                "right_timestamp_us": right_timestamp_us,
                "timestamp_delta_us": right_timestamp_us - left_timestamp_us,
                "left_hands": len(left_hands),
                "right_hands": len(right_hands),
                "matched_hands": len(matches),
                "valid_3d_points": frame_valid,
                "refined_points": frame_refined,
                "median_refinement_quality": (
                    f"{np.median(frame_refinement_quality):.6f}"
                    if len(frame_refinement_quality) else "nan"
                ),
                "median_epipolar_px": f"{np.median(valid_epipolar):.6f}" if len(valid_epipolar) else "nan",
                "median_reprojection_px": f"{np.median(valid_reprojection):.6f}" if len(valid_reprojection) else "nan",
            })

            for match_index, match in enumerate(matches):
                observed_track_ids.add(match["track_id"])
                for landmark_index in range(21):
                    valid = bool(match["valid"][landmark_index])
                    observations_by_landmark[landmark_index] += 1
                    if valid:
                        valid_points_by_landmark[landmark_index] += 1
                    rectified_point = match["points_rectified"][landmark_index]
                    left_point_3d = match["points_left"][landmark_index]
                    landmark_writer.writerow({
                        "pair_index": pair_index, "left_index": left_index, "right_index": right_index,
                        "track_id": match["track_id"], "match_index": match_index,
                        "left_hand_index": match["left"]["index"], "right_hand_index": match["right"]["index"],
                        "left_handedness": match["left"]["label"], "right_handedness": match["right"]["label"],
                        "left_handedness_score": f"{match['left']['score']:.8f}",
                        "right_handedness_score": f"{match['right']['score']:.8f}",
                        "landmark_index": landmark_index,
                        "raw_left_x_rectified_px": f"{match['raw_left_points'][landmark_index, 0]:.6f}",
                        "raw_left_y_rectified_px": f"{match['raw_left_points'][landmark_index, 1]:.6f}",
                        "raw_right_x_rectified_px": f"{match['raw_right_points'][landmark_index, 0]:.6f}",
                        "raw_right_y_rectified_px": f"{match['raw_right_points'][landmark_index, 1]:.6f}",
                        "left_x_rectified_px": f"{match['left_points'][landmark_index, 0]:.6f}",
                        "left_y_rectified_px": f"{match['left_points'][landmark_index, 1]:.6f}",
                        "right_x_rectified_px": f"{match['right_points'][landmark_index, 0]:.6f}",
                        "right_y_rectified_px": f"{match['right_points'][landmark_index, 1]:.6f}",
                        "refinement_attempted": int(match["refinement_attempted"][landmark_index]),
                        "refinement_used": int(match["refinement_used"][landmark_index]),
                        "refinement_quality": f"{match['refinement_quality'][landmark_index]:.6f}",
                        "refinement_lk_error": (
                            f"{match['refinement_lk_error'][landmark_index]:.6f}"
                            if np.isfinite(match["refinement_lk_error"][landmark_index]) else "nan"
                        ),
                        "refinement_fb_error_px": (
                            f"{match['refinement_fb_error'][landmark_index]:.6f}"
                            if np.isfinite(match["refinement_fb_error"][landmark_index]) else "nan"
                        ),
                        "refinement_y_residual_px": (
                            f"{match['refinement_y_residual'][landmark_index]:.6f}"
                            if np.isfinite(match["refinement_y_residual"][landmark_index]) else "nan"
                        ),
                        "refinement_x_shift_px": (
                            f"{match['refinement_x_shift'][landmark_index]:.6f}"
                            if np.isfinite(match["refinement_x_shift"][landmark_index]) else "nan"
                        ),
                        "disparity_px": f"{match['disparity'][landmark_index]:.6f}",
                        "epipolar_error_px": f"{match['epipolar'][landmark_index]:.6f}",
                        "valid_3d": int(valid),
                        "reprojection_error_px": f"{match['reprojection'][landmark_index]:.6f}",
                        "x_rectified_m": f"{rectified_point[0]:.9f}" if valid else "nan",
                        "y_rectified_m": f"{rectified_point[1]:.9f}" if valid else "nan",
                        "z_rectified_m": f"{rectified_point[2]:.9f}" if valid else "nan",
                        "x_left_camera_m": f"{left_point_3d[0]:.9f}" if valid else "nan",
                        "y_left_camera_m": f"{left_point_3d[1]:.9f}" if valid else "nan",
                        "z_left_camera_m": f"{left_point_3d[2]:.9f}" if valid else "nan",
                    })

            if matches:
                pairs_with_matches += 1
            processed_pairs += 1
            matched_hand_instances += len(matches)
            valid_points_total += frame_valid
            all_valid_epipolar.extend(valid_epipolar.tolist())
            all_valid_reprojection.extend(valid_reprojection.tolist())
            for match in matches:
                all_valid_depth.extend(match["points_left"][match["valid"], 2].tolist())
                attempted = match["refinement_attempted"]
                used = match["refinement_used"]
                refinement_attempted_total += int(np.count_nonzero(attempted))
                refinement_used_total += int(np.count_nonzero(used))
                all_refinement_quality.extend(match["refinement_quality"][used].tolist())

            if writer is not None:
                annotated_left = left_rectified.copy()
                annotated_right = right_rectified.copy()
                colors = [(30, 220, 30), (20, 170, 255), (255, 80, 180), (255, 220, 40)]
                matched_left = set()
                matched_right = set()
                for match in matches:
                    color = colors[match["track_id"] % len(colors)]
                    valid_count = int(np.count_nonzero(match["valid"]))
                    draw_observation(annotated_left, match["left"], color, f"T{match['track_id']} {valid_count}/21")
                    draw_observation(annotated_right, match["right"], color, f"T{match['track_id']} {valid_count}/21")
                    matched_left.add(match["left"]["index"])
                    matched_right.add(match["right"]["index"])
                for observation in left_hands:
                    if observation["index"] not in matched_left:
                        draw_observation(annotated_left, observation, (120, 120, 120), "unmatched")
                for observation in right_hands:
                    if observation["index"] not in matched_right:
                        draw_observation(annotated_right, observation, (120, 120, 120), "unmatched")
                pair_image = cv2.hconcat([annotated_left, annotated_right])
                cv2.putText(pair_image, f"pair={pair_index} dt={right_timestamp_us-left_timestamp_us}us matches={len(matches)}",
                            (25, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
                pair_image = cv2.resize(pair_image, preview_size, interpolation=cv2.INTER_AREA)
                writer.write(pair_image)

    if left_capture is not None:
        left_capture.release()
    if right_capture is not None:
        right_capture.release()
    if writer is not None:
        writer.release()
    elapsed = time.perf_counter() - start_time

    def percentile(values, level):
        return float(np.percentile(values, level)) if values else None

    summary = {
        "stage": "stereo_mediapipe_triangulation_baseline",
        "source": source_description,
        "source_kind": source_kind,
        "session": source_description if source_kind == "legacy_session" else None,
        "model": str(model),
        "mediapipe_version": mp.__version__,
        "opencv_version": cv2.__version__,
        "calibration_serial": rectification["calibration_serial"],
        "rectified_size": list(image_size),
        "p1": np.asarray(rectification["p1"]).tolist(),
        "p2": np.asarray(rectification["p2"]).tolist(),
        "timestamp_pairs_available": len(source_pairs),
        "processed_pairs": processed_pairs,
        "pairs_with_stereo_matches": pairs_with_matches,
        "stereo_match_pair_rate": pairs_with_matches / processed_pairs if processed_pairs else 0.0,
        "matched_hand_instances": matched_hand_instances,
        "lk_recovery": {
            "enabled": True,
            "description": "low-confidence cross-frame LK candidates are added only when one stereo view loses a hand",
        },
        "track_ids": sorted(observed_track_ids),
        "track_count": len(observed_track_ids),
        "valid_3d_points": valid_points_total,
        "valid_3d_rate_of_matched": valid_points_total / (matched_hand_instances * 21) if matched_hand_instances else 0.0,
        "refinement_attempted_points": refinement_attempted_total,
        "refinement_used_points": refinement_used_total,
        "refinement_acceptance_rate": (
            refinement_used_total / refinement_attempted_total
            if refinement_attempted_total else 0.0
        ),
        "refinement_quality_median": percentile(all_refinement_quality, 50),
        "epipolar_abs_px_median": percentile(all_valid_epipolar, 50),
        "epipolar_abs_px_p95": percentile(all_valid_epipolar, 95),
        "reprojection_px_median": percentile(all_valid_reprojection, 50),
        "reprojection_px_p95": percentile(all_valid_reprojection, 95),
        "depth_m_median": percentile(all_valid_depth, 50),
        "depth_m_p05": percentile(all_valid_depth, 5),
        "depth_m_p95": percentile(all_valid_depth, 95),
        "valid_3d_rate_by_landmark": {
            str(index): (
                float(valid_points_by_landmark[index] / observations_by_landmark[index])
                if observations_by_landmark[index] else 0.0
            )
            for index in range(21)
        },
        "elapsed_seconds": elapsed,
        "processing_fps": processed_pairs / elapsed if elapsed > 0 else 0.0,
        "filters": {
            "max_delta_us": args.max_delta_us,
            "max_epipolar_px": args.max_epipolar_px,
            "min_depth_m": args.min_depth_m,
            "max_depth_m": args.max_depth_m,
            "max_reprojection_px": args.max_reprojection_px,
            "stereo_refine": args.stereo_refine,
            "refine_window": args.refine_window,
            "refine_tip_window": args.refine_tip_window,
            "refine_max_x_shift_px": args.refine_max_x_shift_px,
            "refine_max_y_residual_px": args.refine_max_y_residual_px,
            "refine_max_fb_error_px": args.refine_max_fb_error_px,
            "refine_max_lk_error": args.refine_max_lk_error,
        },
        "coordinate_note": "x_left_camera/y_left_camera/z_left_camera are metric coordinates in the original left optical frame",
    }
    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("EGO MediaPipe stereo triangulation baseline")
    print(f"Processed stereo pairs: {processed_pairs}")
    print(f"Pairs with stereo hand matches: {pairs_with_matches} ({summary['stereo_match_pair_rate']:.1%})")
    print(f"Matched hand instances: {matched_hand_instances}")
    print(f"Valid 3D landmarks: {valid_points_total} ({summary['valid_3d_rate_of_matched']:.1%})")
    print(
        "Stereo refinement: "
        f"{refinement_used_total}/{refinement_attempted_total} "
        f"({summary['refinement_acceptance_rate']:.1%})"
    )
    metric = lambda value: f"{value:.3f}" if value is not None else "n/a"
    print(
        "Epipolar median/P95: "
        f"{metric(summary['epipolar_abs_px_median'])}/{metric(summary['epipolar_abs_px_p95'])} px"
    )
    print(
        "Reprojection median/P95: "
        f"{metric(summary['reprojection_px_median'])}/{metric(summary['reprojection_px_p95'])} px"
    )
    print(f"Processing speed: {summary['processing_fps']:.2f} stereo fps")
    print(f"3D landmarks CSV: {landmarks_path}")
    if writer is not None:
        print(f"Annotated stereo video: {video_path}")
    print(f"Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(1)
