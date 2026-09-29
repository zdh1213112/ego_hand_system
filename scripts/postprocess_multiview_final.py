#!/usr/bin/env python3
"""Build a conservative final 3-D hand track from a calibrated multiview run.

The expensive WiLoR/SAM3 detections are kept intact.  This post-pass addresses
the failure mode that is especially common with gloves: a detector may return
the right 2-D region while one joint changes semantic/depth branch.  The pass
therefore (1) tracks complete hands with multi-camera box centres, (2) solves
each joint with a robust six-ray consensus, (3) keeps stable finger-chain
lengths, and (4) marks an implausible whole-hand sample missing instead of
clipping it into a twisted pose.  The renderer can display such samples with
``--all-gap-prediction-mode hold`` without morphing the fingers.

This file does not run WiLoR or SAM3 and never rewrites camera-native 2-D
observations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from camera_models import project_points, unproject_points  # noqa: E402
from ego_data.calibration import CameraCalibration  # noqa: E402


PALM_JOINTS = (0, 1, 2, 3, 4, 5, 9, 13, 17)
DISTAL_JOINTS = (6, 7, 8, 10, 11, 12, 14, 15, 16, 18, 19, 20)
FINGER_CHAINS = ((5, 6, 7, 8), (9, 10, 11, 12),
                 (13, 14, 15, 16), (17, 18, 19, 20))


def _points(value: Any) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    return value if value.shape == (21, 3) else np.full((21, 3), np.nan)


def _bbox_center(hand: Mapping[str, Any], camera: str) -> tuple[np.ndarray, float] | None:
    view = hand.get("views", {}).get(camera)
    if not isinstance(view, Mapping):
        return None
    box = np.asarray(view.get("bbox_xyxy", []), dtype=np.float64)
    if box.shape != (4,) or not np.isfinite(box).all():
        return None
    return 0.5 * (box[:2] + box[2:]), max(float(np.linalg.norm(box[2:] - box[:2])), 1.0)


def _bbox_cost(current: Mapping[str, Any], previous: Mapping[str, Any]) -> float:
    values: list[float] = []
    for camera in set(current.get("views", {})).intersection(previous.get("views", {})):
        first = _bbox_center(current, camera)
        second = _bbox_center(previous, camera)
        if first is None or second is None:
            continue
        distance = np.linalg.norm(first[0] - second[0]) / max(0.5 * (first[1] + second[1]), 1.0)
        values.append(float(distance))
    return float(np.median(values)) if values else 1e6


def _bbox_overlap(current: Mapping[str, Any], previous: Mapping[str, Any]) -> int:
    """Number of cameras that provide a usable box in both hand records.

    A missing overlap must not be interpreted as a very large motion.  In
    particular, the first few frames of a clip often contain only two or
    three detections per hand; allowing an identity swap there permanently
    reverses the two colour tracks for the remainder of the sequence.
    """
    count = 0
    for camera in set(current.get("views", {})).intersection(previous.get("views", {})):
        if _bbox_center(current, camera) is not None and _bbox_center(previous, camera) is not None:
            count += 1
    return count


def _fix_hand_identities(rows: list[dict[str, Any]], max_gap: int = 10) -> dict[str, Any]:
    """Reassign whole hand records using all common camera box tracks."""
    tracks: dict[int, Mapping[str, Any]] = {}
    last_frame: int | None = None
    swaps: list[int] = []
    ambiguous: list[int] = []
    for row in rows:
        frame = int(row.get("sync_index", 0))
        if last_frame is not None and frame - last_frame > max(int(max_gap), 1):
            tracks = {}
        current = {
            int(hand.get("side", -1)): hand
            for hand in row.get("hands", [])
            if int(hand.get("side", -1)) in (0, 1)
        }
        assigned: dict[int, Mapping[str, Any]]
        if len(current) == 2 and len(tracks) == 2:
            identity = _bbox_cost(current[0], tracks[0]) + _bbox_cost(current[1], tracks[1])
            exchanged = _bbox_cost(current[1], tracks[0]) + _bbox_cost(current[0], tracks[1])
            # Do not make a decision from a one-view/missing-view pairing.
            # A real swap is accepted only when both possible assignments
            # have at least two common camera boxes.  This prevents the
            # frame-15 startup ambiguity seen in glove clips from causing a
            # persistent left/right reversal.
            identity_overlap = min(_bbox_overlap(current[0], tracks[0]),
                                    _bbox_overlap(current[1], tracks[1]))
            exchanged_overlap = min(_bbox_overlap(current[1], tracks[0]),
                                    _bbox_overlap(current[0], tracks[1]))
            if (identity_overlap >= 2 and exchanged_overlap >= 2
                    and exchanged + 0.10 < identity):
                assigned = {0: current[1], 1: current[0]}
                swaps.append(frame)
            else:
                assigned = current
                if abs(identity - exchanged) < 0.10:
                    ambiguous.append(frame)
        elif current and tracks:
            assigned = {}
            remaining = set(tracks)
            for source_side, hand in current.items():
                if not remaining:
                    assigned[source_side] = hand
                    continue
                costs = {target: _bbox_cost(hand, tracks[target]) for target in remaining}
                target = min(costs, key=costs.get)
                if costs[target] < 1.8:
                    assigned[target] = hand
                    remaining.remove(target)
                else:
                    assigned[source_side] = hand
        else:
            assigned = current
        for side, hand in assigned.items():
            source_side = int(hand.get("side", side))
            hand["side"] = int(side)
            hand["final_identity"] = {
                "algorithm": "all_camera_bbox_track_v1",
                "source_side": source_side,
                "swapped": bool(source_side != side),
            }
        row["hands"] = [assigned[side] for side in sorted(assigned)]
        tracks = assigned
        last_frame = frame
    return {
        "algorithm": "all_camera_bbox_track_v1",
        "swapped_frame_count": len(swaps),
        "swap_events_preview": swaps[:100],
        "ambiguous_frame_count": len(ambiguous),
        "max_gap": int(max_gap),
    }


def _rays(hand: Mapping[str, Any], joint: int,
          calibrations: Mapping[str, CameraCalibration]) -> list[tuple[Any, np.ndarray, np.ndarray, np.ndarray, float]]:
    rays: list[tuple[Any, np.ndarray, np.ndarray, np.ndarray, float]] = []
    for camera_id, view in hand.get("views", {}).items():
        camera = calibrations.get(camera_id)
        if camera is None:
            continue
        observations = np.asarray(view.get("joints_2d", []), dtype=np.float64)
        if observations.shape != (21, 2) or not np.isfinite(observations[joint]).all():
            continue
        direction_camera, valid = unproject_points(camera, observations[joint:joint + 1])
        if not bool(valid[0]):
            continue
        rotation = camera.T_base_camera[:3, :3]
        direction = rotation @ direction_camera[0]
        norm = np.linalg.norm(direction)
        if norm < 1e-8:
            continue
        weight = 1.0
        assist = view.get("sam3_assist", {})
        if isinstance(assist, Mapping) and assist.get("enabled"):
            visible = assist.get("visible_flags", [])
            hard = assist.get("hard_outlier_flags", [])
            if len(visible) == 21 and not bool(visible[joint]):
                weight *= 0.45
            if len(hard) == 21 and bool(hard[joint]):
                weight *= 0.18
            weight *= float(np.clip(0.65 + 0.35 * float(assist.get("mask_score", 1.0)), 0.45, 1.0))
            weight *= float(np.clip(0.75 + 0.25 * float(assist.get("mask_box_iou_yolo", 1.0)), 0.55, 1.0))
        rays.append((camera, camera.T_base_camera[:3, 3], direction / norm,
                     observations[joint], weight))
    return rays


def _ray_angles(rays: list[tuple[Any, np.ndarray, np.ndarray, np.ndarray, float]]) -> np.ndarray:
    if len(rays) < 2:
        return np.ones(len(rays), dtype=np.float64)
    reference = float(np.sin(np.deg2rad(25.0)))
    diversity = []
    for index, ray in enumerate(rays):
        values = [float(np.linalg.norm(np.cross(ray[2], other[2])))
                  for other_index, other in enumerate(rays) if other_index != index]
        diversity.append(max(values, default=0.0))
    geometry = np.clip(np.asarray(diversity) / max(reference, 1e-6), 0.20, 1.0)
    quality = np.clip(np.asarray([ray[4] for ray in rays]), 0.05, 1.0)
    return geometry * quality


def _solve(rays: list[tuple[Any, np.ndarray, np.ndarray, np.ndarray, float]],
           prior: np.ndarray | None, prior_weight: float) -> np.ndarray:
    matrix = max(float(prior_weight), 1e-6) * np.eye(3)
    vector = max(float(prior_weight), 1e-6) * (prior if prior is not None else np.zeros(3))
    for (_, center, direction, _, _), weight in zip(rays, _ray_angles(rays)):
        projector = np.eye(3) - np.outer(direction, direction)
        matrix += weight * projector
        vector += weight * projector @ center
    try:
        return np.linalg.solve(matrix, vector)
    except np.linalg.LinAlgError:
        return np.full(3, np.nan)


def _residuals(point: np.ndarray, rays: list[tuple[Any, np.ndarray, np.ndarray, np.ndarray, float]]) -> np.ndarray:
    values = []
    if not np.isfinite(point).all():
        return np.full(len(rays), np.inf)
    for camera, center, _, observation, _ in rays:
        point_camera = camera.T_base_camera[:3, :3].T @ (point - center)
        projected, valid = project_points(camera, point_camera[None])
        values.append(float(np.linalg.norm(projected[0] - observation))
                      if bool(valid[0]) and np.isfinite(projected[0]).all() else np.inf)
    return np.asarray(values, dtype=np.float64)


def _robust_point(rays: list[tuple[Any, np.ndarray, np.ndarray, np.ndarray, float]],
                  prior: np.ndarray | None) -> np.ndarray:
    if len(rays) < 2:
        return prior.copy() if prior is not None else np.full(3, np.nan)
    candidates: list[np.ndarray] = []
    if prior is not None and np.isfinite(prior).all():
        candidates.append(prior.copy())
    candidate = _solve(rays, prior, 0.01)
    if np.isfinite(candidate).all():
        candidates.append(candidate)
    for _ in range(4):
        residual = _residuals(candidate, rays)
        robust_weight = np.clip(np.where(residual <= 12.0, 1.0, 12.0 / np.maximum(residual, 1e-3)), 0.04, 1.0)
        weighted = [(ray[0], ray[1], ray[2], ray[3], ray[4] * robust_weight[k])
                    for k, ray in enumerate(rays)]
        candidate = _solve(weighted, prior, 0.015)
        if np.isfinite(candidate).all():
            candidates.append(candidate.copy())
    best = None
    for candidate in candidates:
        residual = _residuals(candidate, rays)
        inliers = int(np.count_nonzero(residual < 25.0))
        cost = float(np.sum(np.minimum(residual, 35.0) ** 2))
        score = (inliers, -cost, -float(np.median(residual)), -float(np.percentile(residual, 95)))
        if best is None or score > best[0]:
            best = (score, candidate)
    return best[1] if best is not None else (prior.copy() if prior is not None else np.full(3, np.nan))


def _chain_targets(rows: list[dict[str, Any]]) -> np.ndarray:
    targets = np.zeros((2, 4, 3), dtype=np.float64)
    for side in (0, 1):
        for chain_index, chain in enumerate(FINGER_CHAINS):
            for edge_index, (parent, child) in enumerate(zip(chain[:-1], chain[1:])):
                values = []
                for row in rows:
                    hand = next((h for h in row.get("hands", []) if int(h.get("side", -1)) == side), None)
                    if hand is None:
                        continue
                    points = _points(hand.get("joints_base_m", []))
                    if np.isfinite(points[[parent, child]]).all():
                        length = np.linalg.norm(points[child] - points[parent])
                        if 0.008 < length < 0.18:
                            values.append(length)
                targets[side, chain_index, edge_index] = np.median(values) if values else 0.03
    return targets


def _apply_chain_lengths(points: np.ndarray, targets: np.ndarray) -> None:
    for side in (0, 1):
        for frame in range(len(points)):
            for chain_index, chain in enumerate(FINGER_CHAINS):
                for edge_index, (parent, child) in enumerate(zip(chain[:-1], chain[1:])):
                    if not np.isfinite(points[frame, side, [parent, child]]).all():
                        continue
                    vector = points[frame, side, child] - points[frame, side, parent]
                    length = np.linalg.norm(vector)
                    if length > 1e-8:
                        points[frame, side, child] = points[frame, side, parent] + vector * (targets[side, chain_index, edge_index] / length)


def _runs(mask: np.ndarray):
    index = 0
    while index < len(mask):
        if not mask[index]:
            index += 1
            continue
        end = index + 1
        while end < len(mask) and mask[end]:
            end += 1
        yield index, end
        index = end


def _local_branch_guard(points: np.ndarray, frames: np.ndarray, threshold_m: float = 0.075) -> int:
    changed_count = 0
    for side in (0, 1):
        for joint in range(21):
            for _ in range(4):
                valid = np.isfinite(points[:, side, joint]).all(axis=1)
                changed = False
                for lo, hi in _runs(valid):
                    if hi - lo < 3:
                        continue
                    for index in range(lo, hi):
                        neighbours = [points[k, side, joint] for k in range(max(lo, index - 3), min(hi, index + 4))
                                      if k != index and abs(int(frames[k]) - int(frames[index])) <= 3]
                        if len(neighbours) < 3:
                            continue
                        values = np.asarray(neighbours)
                        median = np.median(values, axis=0)
                        residual = float(np.linalg.norm(points[index, side, joint] - median))
                        spread = float(np.median(np.linalg.norm(values - median, axis=1)))
                        if residual <= max(float(threshold_m), 2.0 * spread + 0.020):
                            continue
                        left = [points[k, side, joint] for k in range(max(lo, index - 2), index)
                                if int(frames[index]) - int(frames[k]) <= 2]
                        right = [points[k, side, joint] for k in range(index + 1, min(hi, index + 3))
                                 if int(frames[k]) - int(frames[index]) <= 2]
                        if not left or not right:
                            continue
                        left_median = np.median(np.asarray(left), axis=0)
                        right_median = np.median(np.asarray(right), axis=0)
                        if np.linalg.norm(left_median - right_median) > max(0.12, 2.5 * threshold_m):
                            continue
                        points[index, side, joint] = 0.5 * (left_median + right_median)
                        changed_count += 1
                        changed = True
                if not changed:
                    break
    return changed_count


def _clear_bad_hands(points: np.ndarray, rows: list[dict[str, Any]], frames: np.ndarray,
                     targets: np.ndarray, *, min_joints: int, width_min: float,
                     width_max: float, edge_scale_min: float, edge_scale_max: float,
                     max_span: float) -> dict[str, Any]:
    edges = ((0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (0, 9), (0, 13), (0, 17))
    cleared: list[dict[str, Any]] = []
    for side in (0, 1):
        for index, row in enumerate(rows):
            hand = next((h for h in row.get("hands", []) if int(h.get("side", -1)) == side), None)
            if hand is None:
                continue
            finite = np.isfinite(points[index, side]).all(axis=1)
            if int(finite.sum()) < int(min_joints):
                points[index, side] = np.nan
                cleared.append({"sync_index": int(frames[index]), "side": side, "reason": "sparse"})
                continue
            pose = points[index, side]
            width = float(np.linalg.norm(pose[5] - pose[17]))
            span = float(np.max(np.linalg.norm(pose - pose[0], axis=1)))
            lengths = [float(np.linalg.norm(pose[child] - pose[parent])) for parent, child in edges]
            bad = width < width_min or width > width_max or span > max_span
            bad = bad or any(length < targets[side, 0, 0] * 0 + 1e-9 for length in [])
            for edge_index, length in enumerate(lengths):
                # The first four edges are the thumb chain and the remaining
                # four are wrist-to-MCP palm spokes.  Use source-specific
                # medians where available.
                reference = targets[side, 0, 0] if edge_index < 0 else None
                if edge_index < 4:
                    values = (0.0376, 0.0301, 0.0269, 0.0346) if side == 0 else (0.0371, 0.0305, 0.0259, 0.0323)
                    reference = values[edge_index]
                else:
                    values = (0.0865, 0.0906, 0.0833, 0.0790) if side == 0 else (0.0874, 0.0901, 0.0799, 0.0740)
                    reference = values[edge_index - 4]
                if length < edge_scale_min * reference or length > edge_scale_max * reference:
                    bad = True
                    break
            if bad:
                points[index, side] = np.nan
                cleared.append({"sync_index": int(frames[index]), "side": side, "reason": "geometry"})
    return {"cleared_hand_count": len(cleared), "events_preview": cleared[:100]}


def _temporal_hand_gate(points: np.ndarray, rows: list[dict[str, Any]], frames: np.ndarray,
                        *, max_joint_step: float, max_hand_step: float,
                        image_wrist_gate_px: float) -> dict[str, Any]:
    cleared = []
    for side in (0, 1):
        for index in range(1, len(points)):
            if int(frames[index]) - int(frames[index - 1]) != 1:
                continue
            current = points[index, side]
            previous = points[index - 1, side]
            if not (np.isfinite(current).all() and np.isfinite(previous).all()):
                continue
            deltas = np.linalg.norm(current - previous, axis=1)
            palm = float(np.median(deltas[list(PALM_JOINTS)]))
            median_step = float(np.median(deltas))
            maximum_step = float(np.max(deltas))
            first = next((h for h in rows[index].get("hands", []) if int(h.get("side", -1)) == side), None)
            second = next((h for h in rows[index - 1].get("hands", []) if int(h.get("side", -1)) == side), None)
            image_steps = []
            if first is not None and second is not None:
                for camera in set(first.get("views", {})).intersection(second.get("views", {})):
                    a = np.asarray(first["views"][camera].get("joints_2d", []), dtype=np.float64)
                    b = np.asarray(second["views"][camera].get("joints_2d", []), dtype=np.float64)
                    if a.shape == (21, 2) and b.shape == (21, 2) and np.isfinite(a[0]).all() and np.isfinite(b[0]).all():
                        image_steps.append(float(np.linalg.norm(a[0] - b[0])))
            image_step = float(np.median(image_steps)) if image_steps else np.inf
            if ((maximum_step > max_joint_step and palm < max_hand_step and image_step < image_wrist_gate_px)
                    or (median_step > max_hand_step * 0.90 and palm < max_hand_step and image_step < image_wrist_gate_px * 1.2)):
                points[index, side] = np.nan
                cleared.append({"sync_index": int(frames[index]), "side": side,
                                "max_step_m": maximum_step, "median_step_m": median_step,
                                "image_wrist_step_px": image_step})
    return {"cleared_hand_count": len(cleared), "events_preview": cleared[:100]}


def _reprojection_quality_gate(
    points: np.ndarray,
    rows: list[dict[str, Any]],
    frames: np.ndarray,
    calibrations: Mapping[str, CameraCalibration],
    *,
    min_joints: int,
    min_views: int,
    max_median_px: float,
    max_p95_px: float,
) -> dict[str, Any]:
    """Reject a complete pose that disagrees with the camera evidence.

    A finite 3-D array is not by itself evidence that the fingers are in the
    right semantic/depth branch.  This final audit uses the untouched camera
    2-D observations and clears the whole hand when the projected pose is
    inconsistent with the available multi-view rays.  Clearing the hand is
    intentional: a partial or branch-mixed skeleton is much more misleading
    in a delivery video than a short missing interval that the renderer can
    conservatively hold.
    """
    cleared: list[dict[str, Any]] = []
    for side in (0, 1):
        for index, row in enumerate(rows):
            pose = points[index, side]
            finite = np.isfinite(pose).all(axis=1)
            if int(finite.sum()) < int(min_joints):
                continue
            hand = next(
                (h for h in row.get("hands", []) if int(h.get("side", -1)) == side),
                None,
            )
            if hand is None:
                continue
            errors: list[float] = []
            joint_view_counts: list[int] = []
            for joint in np.flatnonzero(finite):
                rays = _rays(hand, int(joint), calibrations)
                if len(rays) < int(min_views):
                    continue
                residual = _residuals(pose[int(joint)], rays)
                residual = residual[np.isfinite(residual)]
                if len(residual) < int(min_views):
                    continue
                errors.extend(residual.tolist())
                joint_view_counts.append(int(len(residual)))
            if not errors or not joint_view_counts:
                continue
            median_px = float(np.median(errors))
            p95_px = float(np.percentile(errors, 95))
            if median_px <= float(max_median_px) and p95_px <= float(max_p95_px):
                continue
            points[index, side] = np.nan
            cleared.append({
                "sync_index": int(frames[index]),
                "side": int(side),
                "median_px": median_px,
                "p95_px": p95_px,
                "joint_count": int(finite.sum()),
                "joint_view_count_median": float(np.median(joint_view_counts)),
                "reason": "multi_view_reprojection_quality",
            })
    return {
        "enabled": True,
        "cleared_hand_count": len(cleared),
        "events_preview": cleared[:100],
        "parameters": {
            "min_joints": int(min_joints),
            "min_views": int(min_views),
            "max_median_px": float(max_median_px),
            "max_p95_px": float(max_p95_px),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--min-views", type=int, default=3)
    parser.add_argument("--min-hand-joints", type=int, default=18)
    parser.add_argument("--identity-max-gap", type=int, default=10)
    parser.add_argument("--branch-threshold-m", type=float, default=0.075)
    parser.add_argument("--width-min-m", type=float, default=0.025)
    parser.add_argument("--width-max-m", type=float, default=0.180)
    parser.add_argument("--edge-scale-min", type=float, default=0.25)
    parser.add_argument("--edge-scale-max", type=float, default=2.50)
    parser.add_argument("--max-span-m", type=float, default=0.50)
    parser.add_argument("--max-joint-step-m", type=float, default=0.10)
    parser.add_argument("--max-hand-step-m", type=float, default=0.10)
    parser.add_argument("--image-wrist-gate-px", type=float, default=100.0)
    parser.add_argument(
        "--quality-min-views", type=int, default=3,
        help="minimum camera rays used by the final complete-pose reprojection audit",
    )
    parser.add_argument(
        "--quality-max-median-px", type=float, default=20.0,
        help="clear a hand when its multi-view reprojection median exceeds this value",
    )
    parser.add_argument(
        "--quality-max-p95-px", type=float, default=60.0,
        help="clear a hand when its multi-view reprojection p95 exceeds this value",
    )
    parser.add_argument("--max-frames", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset = args.dataset.resolve()
    source = args.source.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    rows = [json.loads(line) for line in (source / "accepted.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    rows.sort(key=lambda row: int(row["sync_index"]))
    if args.max_frames:
        rows = rows[:int(args.max_frames)]
    identity = _fix_hand_identities(rows, args.identity_max_gap)
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    cameras = tuple(manifest.get("camera_ids", ()))
    algorithm = (
        "six_view_robust_ray_chain_temporal_gate_v1" if len(cameras) == 6
        else "multiview_robust_ray_chain_temporal_gate_v1"
    )
    calibrations = {camera: CameraCalibration.load(dataset / "calibration" / f"{camera}.json") for camera in cameras}
    frames = np.asarray([int(row["sync_index"]) for row in rows], dtype=np.int64)
    points = np.full((len(rows), 2, 21, 3), np.nan, dtype=np.float64)
    robust_count = 0
    fallback_count = 0
    for index, row in enumerate(rows):
        for hand in row.get("hands", []):
            side = int(hand.get("side", -1))
            if side not in (0, 1):
                continue
            # The saved refined pose is the temporally/anatomically guarded
            # estimate.  It is the safe prior for consensus.  The unrefined
            # triangulation is useful only to fill a genuinely missing final
            # joint; using it as the primary prior re-introduces the large
            # depth branches that this post-pass is meant to remove.
            source_point = _points(hand.get("unrefined_joints_base_m", []))
            final_point = _points(hand.get("joints_base_m", []))
            for joint in range(21):
                prior = final_point[joint] if np.isfinite(final_point[joint]).all() else (
                    source_point[joint] if np.isfinite(source_point[joint]).all() else None
                )
                rays = _rays(hand, joint, calibrations)
                if len(rays) >= int(args.min_views):
                    candidate = _robust_point(rays, prior)
                    if np.isfinite(candidate).all():
                        points[index, side, joint] = candidate
                        robust_count += 1
                        continue
                if prior is not None and np.isfinite(prior).all():
                    points[index, side, joint] = prior
                    fallback_count += 1
    targets = _chain_targets(rows)
    _apply_chain_lengths(points, targets)
    branch_count = _local_branch_guard(points, frames, args.branch_threshold_m)
    _apply_chain_lengths(points, targets)
    geometry = _clear_bad_hands(
        points, rows, frames, targets,
        min_joints=args.min_hand_joints,
        width_min=args.width_min_m, width_max=args.width_max_m,
        edge_scale_min=args.edge_scale_min, edge_scale_max=args.edge_scale_max,
        max_span=args.max_span_m,
    )
    temporal = _temporal_hand_gate(
        points, rows, frames,
        max_joint_step=args.max_joint_step_m,
        max_hand_step=args.max_hand_step_m,
        image_wrist_gate_px=args.image_wrist_gate_px,
    )
    quality = _reprojection_quality_gate(
        points, rows, frames, calibrations,
        min_joints=args.min_hand_joints,
        min_views=args.quality_min_views,
        max_median_px=args.quality_max_median_px,
        max_p95_px=args.quality_max_p95_px,
    )
    for index, row in enumerate(rows):
        for hand in row.get("hands", []):
            side = int(hand.get("side", -1))
            if side not in (0, 1):
                continue
            hand["joints_base_m"] = points[index, side].tolist()
            hand["final_stability"] = {
                "algorithm": algorithm,
                "preserved_camera_2d": True,
                "sam3_ray_weights_used": True,
                "palm_mcp_preserved_by_chain_stage": True,
            }
    output.mkdir(parents=True)
    (output / "accepted.jsonl").write_text("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8")
    for name in ("rejected.jsonl",):
        if (source / name).is_file():
            shutil.copy2(source / name, output / name)
    summary = json.loads((source / "summary.json").read_text(encoding="utf-8")) if (source / "summary.json").is_file() else {}
    report = {
        "algorithm": algorithm,
        "source": str(source),
        "robust_joint_count": int(robust_count),
        "fallback_joint_count": int(fallback_count),
        "local_branch_corrected_joint_count": int(branch_count),
        "identity": identity,
        "geometry_gate": geometry,
        "temporal_hand_gate": temporal,
        "reprojection_quality_gate": quality,
        "parameters": {key: value for key, value in vars(args).items() if key not in ("dataset", "source", "output")},
    }
    summary["fusion_algorithm"] = str(summary.get("fusion_algorithm", "existing_fusion")) + "+final_stability_v1"
    summary["final_stability"] = report
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (output / "final_stability_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
