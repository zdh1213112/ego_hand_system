#!/usr/bin/env python3
"""Render a conservative, native WiLoR 2-D six-view review video.

This renderer intentionally does not consume fused 3-D points.  It reads the
saved per-camera WiLoR hypotheses, keeps the detector-side identity, rejects
one-frame association teleports, and applies only a small wrist-local 2-D
filter.  It is therefore useful as the delivery fallback for glove sequences
where the calibrated cameras cannot determine a trustworthy depth branch.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path
import time
from typing import Any

import cv2
import numpy as np

from ego_data.dataset import SequentialVideoReader
from render_multiview_wilor import _create_video_writer


HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
)

# OpenCV BGR: physical left is cyan, physical right is yellow.
LEFT_COLOR = (255, 255, 0)
RIGHT_COLOR = (0, 255, 255)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument(
        "--predictions", required=True, type=Path,
        help="native WiLoR prediction root containing camera*/predictions.jsonl",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cameras", nargs="+", default=None)
    parser.add_argument("--columns", type=int, default=0)
    parser.add_argument("--tile-width", type=int, default=480)
    parser.add_argument(
        "--crop-top-px", type=int, default=0,
        help="remove empty pixels above the action area after drawing the poses",
    )
    parser.add_argument(
        "--overlay-style", choices=("diagnostic", "clean"), default="diagnostic",
        help="clean draws only the hand skeleton for a single-view delivery video",
    )
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument(
        "--physical-side-swap", type=int, choices=(0, 1), default=0,
        help=(
            "swap WiLoR side 0/1 before applying physical L/R colors; "
            "WiLoR's is_right=0/1 already means left/right, so the default "
            "is no swap"
        ),
    )
    parser.add_argument(
        "--smoothing-radius", type=int, default=1,
        help="centered wrist-local 2-D smoothing radius (0 disables it)",
    )
    parser.add_argument(
        "--smoothing-strength", type=float, default=0.25,
        help="blend toward the centered local median",
    )
    parser.add_argument(
        "--outlier-threshold-px", type=float, default=25.0,
        help="minimum wrist-local joint residual treated as a 2-D outlier",
    )
    parser.add_argument(
        "--wrist-outlier-threshold-px", type=float, default=150.0,
        help="one-frame wrist jump threshold; the previous pose is held briefly",
    )
    parser.add_argument(
        "--max-hold-frames", type=int, default=2,
        help="number of consecutive bad/missing samples held for display",
    )
    parser.add_argument(
        "--min-visible-joints", type=int, default=0,
        help=(
            "optional quality gate: require this many joints inside the image "
            "and an in-image wrist before displaying a hand (0 disables it)"
        ),
    )
    parser.add_argument(
        "--video-codec", choices=("auto", "h264_nvenc", "mp4v"), default="auto",
    )
    return parser.parse_args()


def _finite_points(value: Any) -> np.ndarray | None:
    points = np.asarray(value, dtype=np.float64)
    if points.shape != (21, 2) or not np.isfinite(points).all():
        return None
    return points


def _bbox_iou(first: Any, second: Any) -> float:
    a = np.asarray(first, dtype=np.float64)
    b = np.asarray(second, dtype=np.float64)
    if a.shape != (4,) or b.shape != (4,) or not np.isfinite(a).all() or not np.isfinite(b).all():
        return 0.0
    left, top = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return float(intersection / union) if union > 1e-9 else 0.0


def _same_detector_box(first: dict[str, Any], second: dict[str, Any]) -> bool:
    """Return whether two WiLoR hypotheses came from one detector box.

    Older prediction files did not persist ``detection_index``.  In those
    files the two handedness hypotheses are otherwise indistinguishable, and
    selecting one independently for each side makes the renderer draw the
    same physical hand twice.  A very high IoU is intentionally required so
    two nearby hands are not merged during a crossing.
    """
    # ``detection_index`` is assigned afresh for every frame, so an equal
    # index does *not* identify a box across two frames.  The renderer calls
    # this helper both for same-frame assignments and for previous/current
    # track cleanup; using the index here would reset a track whenever YOLO
    # happened to enumerate a new box at the same position in the list.
    # Geometry is therefore the only cross-frame-safe duplicate signal.  It
    # also catches the common case where two different indices describe one
    # merged glove box.
    return _bbox_iou(first.get("bbox_xyxy", []), second.get("bbox_xyxy", [])) >= 0.98


def _candidate_side(hand: dict[str, Any], side: int) -> bool:
    """Keep both hypotheses for tracking, but prefer detector-consistent ones."""
    try:
        return int(hand.get("is_right")) == int(side)
    except (TypeError, ValueError):
        return False


def _detector_matches(hand: dict[str, Any], side: int) -> bool:
    try:
        value = hand.get("detector_is_right")
        return value is not None and int(value) == int(side)
    except (TypeError, ValueError):
        return False


def _candidate_cost(candidate: dict[str, Any], previous: dict[str, Any]) -> float:
    current_points = _finite_points(candidate.get("joints_2d", []))
    previous_points = _finite_points(previous.get("joints_2d", []))
    if current_points is None or previous_points is None:
        return 1e6
    current_box = np.asarray(candidate.get("bbox_xyxy", []), dtype=np.float64)
    previous_box = np.asarray(previous.get("bbox_xyxy", []), dtype=np.float64)
    if current_box.shape == (4,) and previous_box.shape == (4,):
        diagonal = max(
            0.5 * float(np.linalg.norm(current_box[2:] - current_box[:2])
                        + np.linalg.norm(previous_box[2:] - previous_box[:2])),
            1.0,
        )
        box_center_cost = float(
            np.linalg.norm(
                0.5 * (current_box[:2] + current_box[2:])
                - 0.5 * (previous_box[:2] + previous_box[2:])
            ) / diagonal
        )
        box_cost = 0.70 * min(box_center_cost, 5.0) + 0.30 * (
            1.0 - _bbox_iou(current_box, previous_box)
        )
    else:
        box_cost = 1.0
    wrist_cost = float(np.linalg.norm(current_points[0] - previous_points[0])) / max(
        float(np.median(np.linalg.norm(previous_points[[5, 9, 13, 17]] - previous_points[0], axis=1))),
        20.0,
    )
    # The wrist/box terms identify the physical hand.  A small local-shape
    # term breaks ties between the two WiLoR handedness hypotheses.
    local_current = current_points - current_points[0]
    local_previous = previous_points - previous_points[0]
    shape_scale = max(float(np.median(np.linalg.norm(local_previous[[5, 9, 13, 17]], axis=1))), 20.0)
    shape_cost = float(np.median(np.linalg.norm(local_current - local_previous, axis=1))) / shape_scale
    return min(wrist_cost, 8.0) * 0.55 + box_cost * 0.30 + min(shape_cost, 8.0) * 0.15


def _frame_candidates(
    record: dict[str, Any], side: int,
    image_size: tuple[int, int] | None = None,
    min_visible_joints: int = 0,
    require_detector_side: bool = False,
) -> list[dict[str, Any]]:
    candidates = []
    for hand in record.get("hands", []):
        if not _candidate_side(hand, side):
            continue
        # WiLoR predicts both handedness branches for every detector box.
        # Once a detector has labelled the box, accepting its opposite branch
        # lets a track that went briefly out of view steal the other hand.
        # Older predictions without detector labels keep their legacy path.
        if (require_detector_side and hand.get("detector_is_right") is not None
                and not _detector_matches(hand, side)):
            continue
        points = _finite_points(hand.get("joints_2d", []))
        if points is None:
            continue
        if image_size is not None and min_visible_joints > 0:
            width, height = (float(image_size[0]), float(image_size[1]))
            visible = (
                (points[:, 0] >= 0.0) & (points[:, 0] < width)
                & (points[:, 1] >= 0.0) & (points[:, 1] < height)
            )
            # A projected wrist outside the image usually means WiLoR chose a
            # depth/handedness branch that is not drawable in this view.  Do
            # not expose a clipped, self-crossing skeleton in the stabilized
            # review; the raw JSON remains untouched for diagnosis.
            wrist = points[0]
            if int(visible.sum()) < int(min_visible_joints):
                continue
            if not (
                0.0 <= wrist[0] < width and 0.0 <= wrist[1] < height
            ):
                continue
        candidates.append(hand)
    return candidates


def _select_tracks(
    records: dict[int, dict[str, Any]], frame_ids: list[int],
    side: int, max_hold_frames: int, wrist_jump_px: float,
    image_size: tuple[int, int] | None = None,
    min_visible_joints: int = 0,
) -> list[dict[str, Any] | None]:
    selected: list[dict[str, Any] | None] = []
    previous: dict[str, Any] | None = None
    hold_count = 0
    for sync_index in frame_ids:
        candidates = _frame_candidates(
            records.get(sync_index, {}), side, image_size, min_visible_joints
        )
        current: dict[str, Any] | None = None
        if candidates:
            if previous is None:
                detector_consistent = [
                    item for item in candidates if _detector_matches(item, side)
                ]
                current = max(
                    detector_consistent or candidates,
                    key=lambda item: float(item.get("confidence", 0.0)),
                )
            else:
                current = min(candidates, key=lambda item: _candidate_cost(item, previous))
                previous_points = _finite_points(previous.get("joints_2d", []))
                current_points = _finite_points(current.get("joints_2d", []))
                jump = float(np.linalg.norm(current_points[0] - previous_points[0]))
                if jump > float(wrist_jump_px):
                    # A sudden detector reassociation is not repaired by 3-D;
                    # hold the last native 2-D pose for at most two frames.
                    if hold_count < max(int(max_hold_frames), 0):
                        held = copy.deepcopy(previous)
                        held["native_2d_hold"] = True
                        selected.append(held)
                        hold_count += 1
                        continue
                    # Do not expose the first sample after the hold if it is
                    # still attached to the wrong detector box.  A short
                    # 2-D dropout is safer than a teleport; a later sample
                    # close to the previous track will reacquire normally.
                    current = None
                    hold_count = 0
                else:
                    hold_count = 0
        elif previous is not None and hold_count < max(int(max_hold_frames), 0):
            held = copy.deepcopy(previous)
            held["native_2d_hold"] = True
            selected.append(held)
            hold_count += 1
            continue
        selected.append(copy.deepcopy(current) if current is not None else None)
        if current is not None and not current.get("native_2d_outlier"):
            previous = current
            hold_count = 0
    return selected


def _select_camera_tracks(
    records: dict[int, dict[str, Any]], frame_ids: list[int],
    max_hold_frames: int, wrist_jump_px: float,
    image_size: tuple[int, int] | None = None,
    min_visible_joints: int = 0,
) -> dict[int, list[dict[str, Any] | None]]:
    """Select both hands jointly so one detector box cannot become two hands.

    WiLoR emits two handedness hypotheses for every detector box.  Selecting
    the left and right tracks independently can therefore choose the same
    box for both colors after a short occlusion.  Enumerating the tiny 2xN
    assignment at each frame keeps the tracks one-to-one while still allowing
    the detector label to seed the identity.
    """
    tracks: dict[int, list[dict[str, Any] | None]] = {0: [], 1: []}
    previous: dict[int, dict[str, Any] | None] = {0: None, 1: None}
    holds = {0: 0, 1: 0}
    for sync_index in frame_ids:
        candidates = {
            side: _frame_candidates(
                records.get(sync_index, {}), side, image_size, min_visible_joints,
                require_detector_side=True,
            )
            for side in (0, 1)
        }
        options: dict[int, list[dict[str, Any] | None]] = {
            side: list(candidates[side]) + [None] for side in (0, 1)
        }

        def cost(side: int, item: dict[str, Any] | None) -> float:
            if item is None:
                # Missing is preferable to inventing a second hand, but a
                # present previous track is allowed to bridge a short gap.
                # When there is no previous track, however, do not make the
                # all-missing assignment cheaper than a real candidate: old
                # WiLoR files may not contain ``detector_is_right`` and both
                # handedness hypotheses then look only mildly inconsistent.
                # A small missing penalty keeps one physical box visible
                # while the one-to-one constraint still suppresses its
                # duplicate hypothesis on the other side.
                return 1.5 if previous[side] is not None else 0.4
            if previous[side] is None:
                return (
                    -0.01 * float(item.get("confidence", 0.0))
                    + (0.0 if _detector_matches(item, side) else 0.35)
                )
            value = _candidate_cost(item, previous[side])
            if _detector_matches(item, side):
                value -= 0.05
            return value

        assignments = []
        for first in options[0]:
            for second in options[1]:
                if first is not None and second is not None:
                    if _same_detector_box(first, second):
                        continue
                assignments.append((cost(0, first) + cost(1, second), first, second))
        if not assignments:
            assignments = [(0.0, None, None)]
        _, chosen0, chosen1 = min(assignments, key=lambda item: item[0])
        chosen = {0: chosen0, 1: chosen1}
        for side in (0, 1):
            item = chosen[side]
            if item is None:
                # If the other side owns the detector box that previously
                # belonged to this track, a short hold would draw the same
                # hand twice (one current pose plus one stale pose).  This is
                # common when WiLoR emits only the two handedness hypotheses
                # of a single glove during an occlusion.  Drop the stale
                # track instead of holding it.
                other_item = chosen[1 - side]
                if (
                    other_item is not None
                    and previous[side] is not None
                    and _same_detector_box(previous[side], other_item)
                ):
                    tracks[side].append(None)
                    holds[side] = 0
                    previous[side] = None
                    continue
                if previous[side] is not None and holds[side] < max(int(max_hold_frames), 0):
                    held = copy.deepcopy(previous[side])
                    held["native_2d_hold"] = True
                    tracks[side].append(held)
                    holds[side] += 1
                    continue
                tracks[side].append(None)
                holds[side] = 0
                # A lost track must be able to start again from a fresh
                # detector observation.  Keeping the old wrist forever made
                # a hand disappear permanently after a large occluded move.
                previous[side] = None
                continue
            if previous[side] is not None:
                current_points = _finite_points(item.get("joints_2d", []))
                previous_points = _finite_points(previous[side].get("joints_2d", []))
                jump = float(np.linalg.norm(current_points[0] - previous_points[0]))
                if jump > float(wrist_jump_px):
                    if holds[side] < max(int(max_hold_frames), 0):
                        held = copy.deepcopy(previous[side])
                        held["native_2d_hold"] = True
                        tracks[side].append(held)
                        holds[side] += 1
                        continue
                    tracks[side].append(None)
                    holds[side] = 0
                    previous[side] = None
                    continue
            selected = copy.deepcopy(item)
            tracks[side].append(selected)
            previous[side] = item
            holds[side] = 0
    return tracks


def _smooth_track(
    track: list[dict[str, Any] | None], radius: int, strength: float,
    outlier_threshold_px: float, wrist_outlier_threshold_px: float,
) -> list[dict[str, Any] | None]:
    if radius <= 0 or not track:
        return track
    output = [copy.deepcopy(item) if item is not None else None for item in track]
    strength = float(np.clip(strength, 0.0, 1.0))
    for index, item in enumerate(track):
        if item is None:
            continue
        points = _finite_points(item.get("joints_2d", []))
        if points is None:
            continue
        neighbours = []
        for other_index in range(max(0, index - radius), min(len(track), index + radius + 1)):
            other = track[other_index]
            other_points = _finite_points(other.get("joints_2d", [])) if other else None
            if other_points is not None:
                neighbours.append(other_points)
        if len(neighbours) < 2:
            continue
        median = np.median(np.stack(neighbours), axis=0)
        filtered = points.copy()
        wrist_residual = float(np.linalg.norm(points[0] - median[0]))
        if wrist_residual > max(float(wrist_outlier_threshold_px), 1.0):
            filtered += median[0] - points[0]
        wrist = filtered[0]
        local = filtered - wrist
        median_local = median - median[0]
        palm_scale = max(float(np.median(np.linalg.norm(local[[5, 9, 13, 17]], axis=1))), 20.0)
        local_threshold = max(float(outlier_threshold_px), 0.35 * palm_scale)
        residual = np.linalg.norm(local - median_local, axis=1)
        filtered_local = (1.0 - strength) * local + strength * median_local
        filtered_local[residual > local_threshold] = median_local[residual > local_threshold]
        filtered = filtered_local + wrist
        output[index]["joints_2d"] = filtered.tolist()
        output[index]["native_2d_smoothed"] = True
    return output


def _display_box(points: np.ndarray, image_shape: tuple[int, ...]) -> np.ndarray:
    finite = np.isfinite(points).all(axis=1)
    visible = points[finite]
    x0, y0 = np.min(visible, axis=0)
    x1, y1 = np.max(visible, axis=0)
    padding = max(12.0, 0.18 * max(float(x1 - x0), float(y1 - y0), 12.0))
    height, width = image_shape[:2]
    box = np.asarray([x0 - padding, y0 - padding, x1 + padding, y1 + padding])
    box[[0, 2]] = np.clip(box[[0, 2]], 0.0, max(width - 1, 0))
    box[[1, 3]] = np.clip(box[[1, 3]], 0.0, max(height - 1, 0))
    return box


def _draw_hand(
    frame: np.ndarray, hand: dict[str, Any], display_side: int,
    *, clean: bool = False,
) -> None:
    points = _finite_points(hand.get("joints_2d", []))
    if points is None:
        return
    color = LEFT_COLOR if display_side == 0 else RIGHT_COLOR
    if not clean:
        box = _display_box(points, frame.shape).astype(np.int32)
        cv2.rectangle(frame, tuple(box[:2]), tuple(box[2:]), (20, 20, 20), 7, cv2.LINE_AA)
        cv2.rectangle(frame, tuple(box[:2]), tuple(box[2:]), color, 3, cv2.LINE_AA)
        label = "2D HOLD" if hand.get("native_2d_hold") else "TRACKED 2D"
        label += " L" if display_side == 0 else " R"
        cv2.putText(
            frame, label, (int(box[0]), max(28, int(box[1]) - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.68, color, 2, cv2.LINE_AA,
        )
    rounded = np.rint(points).astype(np.int32)
    for start, end in HAND_CONNECTIONS:
        cv2.line(frame, tuple(rounded[start]), tuple(rounded[end]), (20, 20, 20),
                 5 if clean else 8, cv2.LINE_AA)
        cv2.line(frame, tuple(rounded[start]), tuple(rounded[end]), color,
                 3 if clean else 4, cv2.LINE_AA)
    for point in rounded:
        cv2.circle(frame, tuple(point), 4 if clean else 6, (20, 20, 20), -1, cv2.LINE_AA)
        cv2.circle(frame, tuple(point), 3 if clean else 4, color, -1, cv2.LINE_AA)


def main() -> int:
    args = parse_args()
    if args.max_frames < 0 or args.tile_width < 32 or args.columns < 0 or args.crop_top_px < 0:
        raise ValueError("max-frames must be non-negative; tile/columns are invalid")
    if args.smoothing_radius < 0 or args.max_hold_frames < 0 or args.min_visible_joints < 0:
        raise ValueError("smoothing radius and max hold frames must be non-negative")
    if not 0.0 <= args.smoothing_strength <= 1.0:
        raise ValueError("smoothing strength must be in [0, 1]")
    dataset = args.dataset.resolve()
    prediction_root = args.predictions.resolve()
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    cameras = tuple(dict.fromkeys(args.cameras or manifest["camera_ids"]))
    available = tuple(manifest["camera_ids"])
    unknown = sorted(set(cameras) - set(available))
    if unknown:
        raise ValueError(f"unknown cameras: {unknown}")
    with (dataset / "multiview_frames.csv").open("r", encoding="utf-8", newline="") as stream:
        frame_rows = list(csv.DictReader(stream))
    frame_rows = frame_rows[: args.max_frames or None]
    frame_ids = [int(row["sync_index"]) for row in frame_rows]
    if not frame_ids:
        raise ValueError("normalized dataset contains no frames")
    prediction_rows: dict[str, dict[int, dict[str, Any]]] = {}
    for camera in cameras:
        path = prediction_root / camera / "predictions.jsonl"
        if not path.is_file():
            raise FileNotFoundError(path)
        prediction_rows[camera] = {
            int(item["sync_index"]): item
            for item in map(json.loads, path.read_text(encoding="utf-8").splitlines())
            if item
        }

    tracks: dict[tuple[str, int], list[dict[str, Any] | None]] = {}
    track_counts: dict[str, dict[str, dict[str, int]]] = {}
    for camera in cameras:
        selected_tracks = _select_camera_tracks(
            prediction_rows[camera], frame_ids, args.max_hold_frames,
            args.wrist_outlier_threshold_px, tuple(manifest["image_size"]),
            args.min_visible_joints,
        )
        track_counts[camera] = {
            str(side): {
                "observed": sum(item is not None and not item.get("native_2d_hold", False)
                                for item in selected_tracks[side]),
                "held": sum(item is not None and item.get("native_2d_hold", False)
                            for item in selected_tracks[side]),
                "missing": sum(item is None for item in selected_tracks[side]),
            }
            for side in (0, 1)
        }
        for side in (0, 1):
            tracks[(camera, side)] = _smooth_track(
                selected_tracks[side], args.smoothing_radius, args.smoothing_strength,
                args.outlier_threshold_px, args.wrist_outlier_threshold_px,
            )

    if args.columns:
        columns = int(args.columns)
    elif len(cameras) == 6:
        columns = 3
    elif len(cameras) == 4:
        columns = 2
    else:
        columns = max(1, int(np.ceil(np.sqrt(len(cameras)))))
    rows_count = int(np.ceil(len(cameras) / columns))
    image_size = tuple(manifest["image_size"])
    if args.crop_top_px >= image_size[1]:
        raise ValueError("--crop-top-px must be smaller than image height")
    tile_height = int(round(args.tile_width * (image_size[1] - args.crop_top_px) / image_size[0]))
    frame_size = (args.tile_width * columns, tile_height * rows_count)
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer, codec = _create_video_writer(
        args.output, 30.0, frame_size, args.video_codec,
    )
    readers = {
        camera: SequentialVideoReader(
            dataset / "cameras" / camera / manifest["storage"]["video_filename"],
            image_size,
        )
        for camera in cameras
    }
    start = time.perf_counter()
    try:
        for index, row in enumerate(frame_rows):
            tiles = []
            for camera in cameras:
                image = readers[camera].read(int(row[f"{camera}_frame_index"]))
                for source_side in (0, 1):
                    item = tracks[(camera, source_side)][index]
                    if item is None:
                        continue
                    display_side = 1 - source_side if args.physical_side_swap else source_side
                    _draw_hand(image, item, display_side,
                               clean=args.overlay_style == "clean")
                if args.overlay_style == "diagnostic":
                    cv2.putText(
                        image,
                        f"{camera} | NATIVE 2D | frame {frame_ids[index]}",
                        (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (70, 220, 70), 2, cv2.LINE_AA,
                    )
                tiles.append(cv2.resize(image[args.crop_top_px:], (args.tile_width, tile_height)))
            while len(tiles) < rows_count * columns:
                tiles.append(np.zeros((tile_height, args.tile_width, 3), dtype=np.uint8))
            canvas = np.vstack([
                np.hstack(tiles[offset:offset + columns])
                for offset in range(0, len(tiles), columns)
            ])
            writer.write(canvas)
            if (index + 1) % 100 == 0 or index + 1 == len(frame_rows):
                elapsed = time.perf_counter() - start
                print(
                    f"native-2d rendered {index + 1}/{len(frame_rows)} | "
                    f"{(index + 1) / max(elapsed, 1e-9):.2f} frames/s",
                    flush=True,
                )
    finally:
        for reader in readers.values():
            reader.close()
        writer.release()
    summary = {
        "schema_version": 1,
        "stage": "native_wilor_2d_multiview_review",
        "dataset": str(dataset),
        "predictions": str(prediction_root),
        "cameras": list(cameras),
        "frame_count": len(frame_rows),
        "physical_side_swap": bool(args.physical_side_swap),
        "overlay_style": args.overlay_style,
        "crop_top_px": int(args.crop_top_px),
        "smoothing_radius": int(args.smoothing_radius),
        "smoothing_strength": float(args.smoothing_strength),
        "wrist_outlier_threshold_px": float(args.wrist_outlier_threshold_px),
        "max_hold_frames": int(args.max_hold_frames),
        "min_visible_joints": int(args.min_visible_joints),
        "track_counts": track_counts,
        "codec": codec,
        "output": str(args.output),
    }
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Native WiLoR 2D multiview video: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
