#!/usr/bin/env python3
"""Make a coherent final hand track from an existing multiview fusion run.

The input camera observations are deliberately left untouched.  Only the
exported 3-D joints are post-processed.  The pass is designed for the common
glove failure mode where a MANO/window solution is anatomically valid but one
frame takes a different depth/semantic branch:

* keep physical hand identities attached to a continuous 3-D palm track;
* smooth articulation in a per-frame palm coordinate system (zero phase);
* repair only a short, two-sided local branch outlier;
* re-project the result onto robust, fixed bone lengths.

Long missing intervals remain missing in ``accepted.jsonl``.  Use the renderer
with ``--all-gap-prediction-mode hold`` for a conservative continuous review
video; the synthetic poses are explicitly labelled ``PREDICTED`` and are not
measurements.  ``geodesic`` remains available for an explicitly requested
motion-transition view.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from postprocess_multiview_final import _bbox_cost  # noqa: E402
from stabilize_hand_3d import (  # noqa: E402
    PALM_FRAME_JOINTS,
    SKELETON_EDGES,
    constrain_bones,
    estimate_bone_lengths,
    palm_normalized_3d_coordinates,
)


_FINGER_CHAINS = (
    (0, 1, 2, 3, 4),
    (0, 5, 6, 7, 8),
    (0, 9, 10, 11, 12),
    (0, 13, 14, 15, 16),
    (0, 17, 18, 19, 20),
)


def _points(value: Any) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    return value if value.shape == (21, 3) else np.full((21, 3), np.nan)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path,
                        help="existing fusion directory containing accepted.jsonl")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--identity-max-gap", type=int, default=10)
    parser.add_argument(
        "--identity-mode", choices=("3d", "preserve"), default="3d",
        help=(
            "identity assignment for the input 3-D track: 3d follows palm "
            "continuity (default), preserve keeps the upstream side labels"
        ),
    )
    parser.add_argument("--smooth-radius", type=int, default=8)
    parser.add_argument("--smooth-strength", type=float, default=0.95)
    parser.add_argument(
        "--minimum-valid-joints", type=int, default=18,
        help=(
            "clear a hand before smoothing when fewer than this many joints "
            "are finite; use 21 for a strict complete-finger track"
        ),
    )
    parser.add_argument("--bone-iterations", type=int, default=8)
    parser.add_argument("--bone-strength", type=float, default=0.45)
    parser.add_argument(
        "--max-bone-angle-step-deg", type=float, default=35.0,
        help=("maximum frame-to-frame change of a finger-bone direction in "
              "the palm frame; set 0 to disable"),
    )
    parser.add_argument("--branch-threshold", type=float, default=0.24,
                        help="palm-normalized median residual for a branch outlier")
    parser.add_argument("--branch-agreement", type=float, default=0.22,
                        help="maximum left/right neighbour disagreement")
    parser.add_argument("--branch-max-joint", type=float, default=0.55,
                        help="minimum palm-normalized maximum residual")
    parser.add_argument("--branch-radius", type=int, default=4)
    parser.add_argument("--max-frame-distance", type=int, default=4,
                        help="do not mix neighbours across a larger sync-index gap")
    return parser.parse_args()


_IDENTITY_ANCHORS = np.asarray((0, 5, 9, 13, 17), dtype=np.int32)


def _hand_points(hand: dict[str, Any]) -> np.ndarray:
    """Return the 3-D points used only for physical-track assignment."""
    value = np.asarray(hand.get("joints_base_m", []), dtype=np.float64)
    return value if value.shape == (21, 3) else np.full((21, 3), np.nan)


def _pose_track_cost(current: dict[str, Any], previous: dict[str, Any]) -> float | None:
    """Robust distance between two observations of the same physical hand.

    The old identity pass used only image boxes.  When two gloves cross in a
    camera, a box can be closer to the *other* hand even though the 3-D fused
    pose is continuous.  Palm anchors are much safer for that decision.  The
    returned value is in metres and deliberately ignores finger tips, whose
    semantic branch is the noisy part we are repairing below.
    """
    first = _hand_points(current)
    second = _hand_points(previous)
    mask = np.isfinite(first[_IDENTITY_ANCHORS]).all(axis=1)
    mask &= np.isfinite(second[_IDENTITY_ANCHORS]).all(axis=1)
    if int(mask.sum()) < 2:
        return None
    distances = np.linalg.norm(
        first[_IDENTITY_ANCHORS][mask] - second[_IDENTITY_ANCHORS][mask], axis=1
    )
    # The wrist/palm anchors carry identity; a median prevents one bad MCP
    # from forcing a whole-hand exchange.
    return float(np.median(distances))


def _fix_hand_identities_3d(
    rows: list[dict[str, Any]], max_gap: int = 10,
) -> dict[str, Any]:
    """Track physical hands with 3-D palm continuity and 2-D only as tie-break.

    A previous implementation reassigned based solely on all-camera bbox
    overlap.  That is unsafe at crossings and when one hand is missing: it can
    move a complete pose from side 1 to side 0 for one frame, producing a
    30-cm jump in the final video.  Here the 3-D palm anchor cost decides when
    available; boxes are used only when the 3-D observation is incomplete.
    """
    tracks: dict[int, dict[str, Any]] = {}
    track_frames: dict[int, int] = {}
    last_frame: int | None = None
    swaps: list[int] = []
    ambiguous: list[int] = []

    for row in rows:
        frame = int(row.get("sync_index", 0))
        if last_frame is not None and frame - last_frame > max(int(max_gap), 1):
            tracks.clear()
            track_frames.clear()

        current_all = {
            int(hand.get("side", -1)): hand
            for hand in row.get("hands", [])
            if int(hand.get("side", -1)) in (0, 1)
        }
        # A detector record can exist even when its 3-D array is entirely
        # invalid (for example, all rays were rejected).  Such a record must
        # not participate in the two-hand assignment: using its bbox as a
        # tie-break can exchange a valid hand with an all-NaN placeholder and
        # create a full-hand jump at the next frame.  Keep the placeholder for
        # output, but match only observations with at least two palm anchors.
        current = {
            source_side: hand
            for source_side, hand in current_all.items()
            if int(np.isfinite(_hand_points(hand)[_IDENTITY_ANCHORS]).all(axis=1).sum()) >= 2
        }
        # Do not compare against a stale hand that has been absent for too
        # long.  Keeping a short-lived track is important for a one-frame
        # detector dropout, though.
        for side in list(tracks):
            if frame - track_frames.get(side, frame) > max(int(max_gap), 1):
                tracks.pop(side, None)
                track_frames.pop(side, None)

        assigned: dict[int, dict[str, Any]] = {}
        if len(current) == 2 and len(tracks) == 2:
            first, second = current[0], current[1]
            c00 = _pose_track_cost(first, tracks[0])
            c11 = _pose_track_cost(second, tracks[1])
            c01 = _pose_track_cost(second, tracks[0])
            c10 = _pose_track_cost(first, tracks[1])
            if None not in (c00, c11, c01, c10):
                identity = float(c00 + c11)
                exchanged = float(c01 + c10)
                if exchanged + 0.010 < identity:
                    assigned = {0: second, 1: first}
                    swaps.append(frame)
                else:
                    assigned = {0: first, 1: second}
                    if abs(identity - exchanged) < 0.010:
                        ambiguous.append(frame)
            else:
                identity = _bbox_cost(first, tracks[0]) + _bbox_cost(second, tracks[1])
                exchanged = _bbox_cost(second, tracks[0]) + _bbox_cost(first, tracks[1])
                assigned = ({0: second, 1: first}
                            if exchanged + 0.10 < identity
                            else {0: first, 1: second})
                if exchanged + 0.10 < identity:
                    swaps.append(frame)
        elif current and tracks:
            remaining = set(tracks)
            # Assign the most geometrically distinctive current hand first.
            # With two observations this also avoids order-dependent choices.
            candidates = []
            for source_side, hand in current.items():
                costs = {
                    target: _pose_track_cost(hand, tracks[target])
                    for target in remaining
                }
                candidates.append((
                    min((v for v in costs.values() if v is not None), default=np.inf),
                    source_side, hand, costs,
                ))
            for _, source_side, hand, costs in sorted(candidates, key=lambda value: value[0]):
                if not remaining:
                    assigned[source_side] = hand
                    continue
                usable = {target: value for target, value in costs.items()
                          if target in remaining and value is not None}
                if usable:
                    target = min(usable, key=usable.get)
                    assigned[target] = hand
                    remaining.remove(target)
                else:
                    # Incomplete 3-D: fall back to image boxes, but do not
                    # relabel a hand merely because its box is ambiguous.
                    box_costs = {
                        target: _bbox_cost(hand, tracks[target])
                        for target in remaining
                    }
                    target = min(box_costs, key=box_costs.get)
                    if box_costs[target] < 1.8:
                        assigned[target] = hand
                        remaining.remove(target)
                    else:
                        assigned[source_side] = hand
        else:
            assigned = current.copy()

        # Preserve untrackable detector records on an unused side without
        # updating its physical track.  This keeps the original sparse record
        # shape while ensuring a NaN placeholder cannot cause a relabel.
        occupied = set(assigned)
        free_targets = [target for target in (0, 1) if target not in occupied]
        for source_side, hand in current_all.items():
            if source_side in current:
                continue
            if source_side in free_targets:
                target = source_side
                free_targets.remove(target)
            elif free_targets:
                target = free_targets.pop(0)
            else:
                # Both physical tracks already have a usable observation;
                # retaining a third placeholder would overwrite a valid hand.
                continue
            assigned[target] = hand

        # Update only tracks observed in this row; retain the other side for a
        # short gap so its next complete observation keeps the same identity.
        for side, hand in assigned.items():
            source_side = int(hand.get("side", side))
            hand["side"] = int(side)
            if int(np.isfinite(_hand_points(hand)[_IDENTITY_ANCHORS]).all(axis=1).sum()) < 2:
                hand["final_identity"] = {
                    "algorithm": "3d_palm_continuity_track_v2",
                    "source_side": source_side,
                    "swapped": bool(source_side != side),
                    "trackable": False,
                }
                continue
            hand["final_identity"] = {
                "algorithm": "3d_palm_continuity_track_v2",
                "source_side": source_side,
                "swapped": bool(source_side != side),
                "trackable": True,
            }
            tracks[side] = hand
            track_frames[side] = frame
        row["hands"] = [assigned[side] for side in sorted(assigned)]
        last_frame = frame

    return {
        "algorithm": "3d_palm_continuity_track_v2",
        "swapped_frame_count": len(swaps),
        "swap_events_preview": swaps[:100],
        "ambiguous_frame_count": len(ambiguous),
        "max_gap": int(max_gap),
    }


def _limited_unit_direction(
    previous: np.ndarray, current: np.ndarray, max_angle_rad: float,
) -> tuple[np.ndarray, bool]:
    """Move ``current`` toward ``previous`` when its turn is implausibly sharp."""
    previous_norm = float(np.linalg.norm(previous))
    current_norm = float(np.linalg.norm(current))
    if previous_norm < 1e-8 or current_norm < 1e-8:
        return current, False
    first = previous / previous_norm
    second = current / current_norm
    cosine = float(np.clip(np.dot(first, second), -1.0, 1.0))
    angle = float(np.arccos(cosine))
    if angle <= max_angle_rad:
        return second, False
    # Spherical interpolation avoids shrinking a bone while turning it back
    # toward the previous direction.  The antipodal case is vanishingly rare
    # for a hand, but linear blending remains finite there.
    if np.sin(angle) < 1e-6:
        blended = (1.0 - max_angle_rad / max(angle, 1e-8)) * first
        blended += (max_angle_rad / max(angle, 1e-8)) * second
    else:
        fraction = float(np.clip(max_angle_rad / angle, 0.0, 1.0))
        blended = (
            np.sin((1.0 - fraction) * angle) / np.sin(angle) * first
            + np.sin(fraction * angle) / np.sin(angle) * second
        )
    norm = float(np.linalg.norm(blended))
    return (blended / norm if norm > 1e-8 else first), True


def _limit_finger_direction_steps(
    points: np.ndarray,
    valid: np.ndarray,
    frame_ids: np.ndarray,
    *,
    max_angle_deg: float,
    passes: int = 2,
) -> tuple[np.ndarray, int]:
    """Rate-limit finger-bone turns while preserving each frame's palm pose.

    MANO can choose a different finger-depth branch for one frame even when
    the palm and 2-D boxes are continuous.  A bone-length projection alone
    does not remove that kink.  This pass works in the moving palm frame and
    clamps only the direction of finger-chain edges; the palm anchors and
    camera observations are untouched.
    """
    if max_angle_deg <= 0.0 or passes <= 0:
        return points.copy(), 0
    local, origins, axes, scales, frame_valid = palm_normalized_3d_coordinates(
        points, valid
    )
    result = local.copy()
    limit = float(np.deg2rad(max_angle_deg))
    changed = 0
    for pass_index in range(int(passes)):
        direction = 1 if pass_index % 2 == 0 else -1
        indices = range(1, len(frame_ids)) if direction > 0 else range(
            len(frame_ids) - 2, -1, -1
        )
        for side in (0, 1):
            for index in indices:
                neighbour = index - direction
                if (
                    not frame_valid[side, index]
                    or not frame_valid[side, neighbour]
                    or abs(int(frame_ids[index]) - int(frame_ids[neighbour])) != 1
                ):
                    continue
                for chain in _FINGER_CHAINS:
                    for parent, child in zip(chain[:-1], chain[1:]):
                        if not (
                            valid[side, index, parent]
                            and valid[side, index, child]
                            and valid[side, neighbour, parent]
                            and valid[side, neighbour, child]
                        ):
                            continue
                        current_vector = result[side, index, child] - result[
                            side, index, parent
                        ]
                        previous_vector = result[side, neighbour, child] - result[
                            side, neighbour, parent
                        ]
                        current_length = float(np.linalg.norm(current_vector))
                        if current_length < 1e-8:
                            continue
                        limited, was_limited = _limited_unit_direction(
                            previous_vector, current_vector, limit
                        )
                        if not was_limited:
                            continue
                        result[side, index, child] = (
                            result[side, index, parent] + current_length * limited
                        )
                        changed += 1
    reconstructed = points.copy()
    for side in (0, 1):
        for index in range(len(frame_ids)):
            if not frame_valid[side, index]:
                continue
            for joint in np.flatnonzero(valid[side, index]):
                reconstructed[side, index, joint] = (
                    origins[side, index]
                    + scales[side, index] * (
                        result[side, index, joint] @ axes[side, index]
                    )
                )
    reconstructed[~valid] = np.nan
    return reconstructed, changed


def _repair_local_branches(
    points: np.ndarray,
    valid: np.ndarray,
    frame_ids: np.ndarray,
    *,
    threshold: float,
    agreement: float,
    max_joint_residual: float,
    radius: int,
    max_frame_distance: int,
) -> tuple[np.ndarray, int, list[dict[str, float | int]]]:
    """Repair isolated articulation branches using two-sided palm-local medians."""
    result = np.asarray(points, dtype=np.float64).copy()
    events: list[dict[str, float | int]] = []
    if radius <= 0:
        return result, 0, events

    for pass_index in range(2):
        changed_this_pass = 0
        for side in (0, 1):
            local, origins, axes, scales, frame_valid = palm_normalized_3d_coordinates(
                result[side:side + 1], valid[side:side + 1]
            )
            local = local[0]
            origins = origins[0]
            axes = axes[0]
            scales = scales[0]
            frame_valid = frame_valid[0]
            for index in range(len(frame_ids)):
                # Replacing a partial skeleton can turn an intended dropout
                # into a false complete hand.  The renderer's hold path is
                # safer for those rows, so only complete articulated samples
                # enter this branch repair.
                if not (frame_valid[index] and valid[side, index].all()):
                    continue
                neighbours = [
                    other for other in range(
                        max(0, index - radius), min(len(frame_ids), index + radius + 1)
                    )
                    if other != index
                    and frame_valid[other]
                    and valid[side, other].all()
                    and abs(int(frame_ids[other]) - int(frame_ids[index]))
                    <= max_frame_distance
                ]
                if len(neighbours) < 4:
                    continue
                median_local = np.median(local[neighbours], axis=0)
                residuals = np.linalg.norm(local[index] - median_local, axis=1)
                median_residual = float(np.median(residuals))
                max_residual = float(np.max(residuals))
                if median_residual < threshold or max_residual < max_joint_residual:
                    continue
                left = [other for other in neighbours if other < index]
                right = [other for other in neighbours if other > index]
                if not left or not right:
                    continue
                left_local = np.median(local[left], axis=0)
                right_local = np.median(local[right], axis=0)
                side_disagreement = float(
                    np.median(np.linalg.norm(left_local - right_local, axis=1))
                )
                if side_disagreement > agreement:
                    continue

                # Palm anchors preserve the per-frame rigid motion; only the
                # finger articulation is replaced by the agreed local branch.
                candidate_local = local[index].copy()
                candidate_local[1:] = median_local[1:]
                candidate_local -= np.median(candidate_local[PALM_FRAME_JOINTS], axis=0)
                candidate = origins[index] + scales[index] * (
                    candidate_local @ axes[index]
                )
                if not np.isfinite(candidate).all():
                    continue
                result[side, index] = candidate
                changed_this_pass += 1
                events.append({
                    "sync_index": int(frame_ids[index]),
                    "side": int(side),
                    "median_residual": median_residual,
                    "max_residual": max_residual,
                    "neighbour_disagreement": side_disagreement,
                    "pass": int(pass_index),
                })
        if changed_this_pass == 0:
            break
    return result, len(events), events


def _finite_joint_counts(points: np.ndarray) -> list[int]:
    return [int(np.isfinite(points[side]).all(axis=-1).sum()) for side in (0, 1)]


def _smooth_fingers_preserve_palm(
    points: np.ndarray,
    valid: np.ndarray,
    frame_ids: np.ndarray,
    radius: int,
    strength: float,
) -> np.ndarray:
    """Smooth articulation in the current palm frame without moving the palm.

    The generic 3-D smoother also averages each frame's palm rotation and
    scale.  That is useful for a free-standing mocap track, but in this
    renderer it can move a perfectly good 2-D hand by tens of pixels.  Keep
    the five palm anchors (and their calibrated camera projection) exactly at
    the current observation, and only filter distal finger coordinates in the
    current frame's local basis.  Runs are processed independently so a
    rejected ``sync_index`` gap cannot leak into the filter.
    """
    points = np.asarray(points, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    output = np.full_like(points, np.nan)
    if len(frame_ids) == 0:
        return output
    if radius <= 0 or strength <= 0.0:
        output[...] = points
        output[~valid] = np.nan
        return output
    local, origins, axes, scales, frame_valid = palm_normalized_3d_coordinates(
        points, valid
    )
    # Exclude the palm anchors from the filtered set.  Finger roots remain
    # attached to the observed palm; all other joints receive a robust,
    # weighted temporal median in the current frame's basis.
    palm_set = set(int(value) for value in PALM_FRAME_JOINTS)
    breaks = np.flatnonzero(np.diff(frame_ids) != 1)
    runs = np.split(np.arange(len(frame_ids)), breaks + 1)
    for side in (0, 1):
        for run in runs:
            if len(run) == 0:
                continue
            for index in run:
                if not frame_valid[side, index]:
                    continue
                lo = max(int(run[0]), int(index) - int(radius))
                hi = min(int(run[-1]) + 1, int(index) + int(radius) + 1)
                candidate_frames = np.arange(lo, hi)
                palm_candidates = candidate_frames[
                    frame_valid[side, candidate_frames]
                ]
                if len(palm_candidates) == 0:
                    continue
                blended_local = local[side, index].copy()
                for joint in range(21):
                    if joint in palm_set or not valid[side, index, joint]:
                        continue
                    candidates = palm_candidates[
                        valid[side, palm_candidates, joint]
                    ]
                    if len(candidates) < 2:
                        continue
                    values = local[side, candidates, joint]
                    median = np.median(values, axis=0)
                    distances = np.linalg.norm(values - median, axis=1)
                    distance_median = float(np.median(distances))
                    distance_mad = 1.4826 * float(
                        np.median(np.abs(distances - distance_median))
                    )
                    robust_limit = max(0.10, distance_median + 3.5 * distance_mad)
                    inliers = distances <= robust_limit
                    if int(np.count_nonzero(inliers)) >= 2:
                        values = values[inliers]
                        candidates = candidates[inliers]
                    weights = 1.0 + float(radius) - np.abs(candidates - index)
                    smoothed = np.average(values, axis=0, weights=weights)
                    blended_local[joint] = (
                        (1.0 - strength) * local[side, index, joint]
                        + strength * smoothed
                    )
                for joint in np.flatnonzero(valid[side, index]):
                    if joint in palm_set:
                        output[side, index, joint] = points[side, index, joint]
                    else:
                        output[side, index, joint] = (
                            origins[side, index]
                            + scales[side, index]
                            * (blended_local[joint] @ axes[side, index])
                        )
    output[~valid] = np.nan
    return output


def _smooth_contiguous_runs(
    points: np.ndarray,
    valid: np.ndarray,
    frame_ids: np.ndarray,
    radius: int,
    strength: float,
) -> np.ndarray:
    """Smooth only within synchronized runs, never across a source gap."""
    output = np.full_like(points, np.nan)
    if len(frame_ids) == 0:
        return output
    breaks = np.flatnonzero(np.diff(frame_ids) != 1)
    runs = np.split(np.arange(len(frame_ids)), breaks + 1)
    for run in runs:
        if len(run) == 0:
            continue
        output[:, run] = _smooth_fingers_preserve_palm(
            points[:, run], valid[:, run], radius=radius, strength=strength,
            frame_ids=frame_ids[run],
        )
    output[~valid] = np.nan
    return output


def main() -> int:
    args = _parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    accepted_path = source / "accepted.jsonl"
    if not accepted_path.is_file():
        raise FileNotFoundError(accepted_path)
    if output.exists():
        raise FileExistsError(output)
    if args.smooth_radius < 0 or args.bone_iterations < 0 or args.branch_radius < 0:
        raise ValueError("radii and iterations must be non-negative")
    if not 1 <= args.minimum_valid_joints <= 21:
        raise ValueError("--minimum-valid-joints must be in [1, 21]")
    if not 0.0 <= args.smooth_strength <= 1.0:
        raise ValueError("--smooth-strength must be in [0, 1]")
    if not 0.0 <= args.bone_strength <= 1.0:
        raise ValueError("--bone-strength must be in [0, 1]")

    rows = [
        json.loads(line) for line in accepted_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows.sort(key=lambda row: int(row.get("sync_index", 0)))
    if args.identity_mode == "3d":
        identity = _fix_hand_identities_3d(rows, max_gap=args.identity_max_gap)
    else:
        identity = {
            "algorithm": "preserve_upstream_side_labels",
            "swapped_frame_count": 0,
            "swap_events_preview": [],
            "ambiguous_frame_count": 0,
            "max_gap": int(args.identity_max_gap),
        }
        for row in rows:
            for hand in row.get("hands", []):
                side = int(hand.get("side", -1))
                if side in (0, 1):
                    hand["final_identity"] = {
                        "algorithm": "preserve_upstream_side_labels",
                        "source_side": side,
                        "swapped": False,
                    }
    frame_ids = np.asarray([int(row.get("sync_index", 0)) for row in rows], dtype=np.int64)

    points = np.full((2, len(rows), 21, 3), np.nan, dtype=np.float64)
    for index, row in enumerate(rows):
        for hand in row.get("hands", []):
            side = int(hand.get("side", -1))
            if side not in (0, 1):
                continue
            value = _points(hand.get("joints_base_m", []))
            if value.shape == (21, 3):
                points[side, index] = value
    valid = np.isfinite(points).all(axis=-1)
    before_counts = _finite_joint_counts(points)
    visibility_cleared = []
    for side in (0, 1):
        for index, row in enumerate(rows):
            finite_count = int(valid[side, index].sum())
            if 0 < finite_count < int(args.minimum_valid_joints):
                points[side, index] = np.nan
                valid[side, index] = False
                visibility_cleared.append({
                    "sync_index": int(frame_ids[index]),
                    "side": int(side),
                    "finite_joint_count_before": finite_count,
                    "minimum_valid_joints": int(args.minimum_valid_joints),
                })

    smoothed = _smooth_contiguous_runs(
        points, valid, frame_ids,
        radius=args.smooth_radius, strength=args.smooth_strength,
    )

    guarded, branch_count, branch_events = _repair_local_branches(
        smoothed, valid, frame_ids,
        threshold=args.branch_threshold,
        agreement=args.branch_agreement,
        max_joint_residual=args.branch_max_joint,
        radius=args.branch_radius,
        max_frame_distance=args.max_frame_distance,
    )
    # Estimate each physical hand's stable lengths from the unmodified source,
    # then use a light projection to remove any numerical drift introduced by
    # palm-local smoothing.  Invalid joints remain invalid.
    for side in (0, 1):
        confidence = np.where(valid[side:side + 1], 1.0, 0.0)
        targets = estimate_bone_lengths(
            points[side:side + 1], valid[side:side + 1], confidence
        )
        guarded[side] = constrain_bones(
            guarded[side:side + 1], valid[side:side + 1], confidence,
            targets, args.bone_iterations, args.bone_strength,
        )[0]
        guarded[side][~valid[side]] = np.nan

    # Bone projection adjusts neighbouring joints sequentially and can
    # re-introduce a sharp distal turn.  Apply the direction limiter *after*
    # that projection so the exported pose is the one that is actually
    # rate-limited; the camera 2-D observations remain untouched.
    guarded, angle_limited_count = _limit_finger_direction_steps(
        guarded, valid, frame_ids,
        max_angle_deg=args.max_bone_angle_step_deg,
    )

    # Bone projection and the direction limiter both operate edge-by-edge.
    # Their small corrections can nevertheless make a *single* distal branch
    # disagree with the two-sided temporal consensus again (most often at a
    # fingertip reacquisition).  Run the branch test once more on the actual
    # exported candidate, after all geometric projections, so the file we
    # render is also the file we have audited.  This is intentionally the
    # same conservative gate as the first pass; it only replaces a complete
    # pose when both temporal neighbours agree.
    guarded, post_branch_count, post_branch_events = _repair_local_branches(
        guarded, valid, frame_ids,
        threshold=args.branch_threshold,
        agreement=args.branch_agreement,
        max_joint_residual=args.branch_max_joint,
        radius=args.branch_radius,
        max_frame_distance=args.max_frame_distance,
    )

    changed_joints = 0
    for index, row in enumerate(rows):
        for hand in row.get("hands", []):
            side = int(hand.get("side", -1))
            if side not in (0, 1):
                continue
            old = points[side, index]
            new = guarded[side, index]
            changed = np.isfinite(old).all(axis=1) & np.isfinite(new).all(axis=1)
            changed &= np.linalg.norm(new - old, axis=1) > 0.001
            changed_joints += int(np.count_nonzero(changed))
            hand["joints_base_m"] = new.tolist()
            hand["final_stability"] = {
                "algorithm": "mano_palm_local_zero_phase_coherent_v2",
                "identity_track": identity.get("algorithm", "unknown"),
                "preserved_camera_2d": True,
                "smooth_radius": int(args.smooth_radius),
                "smooth_strength": float(args.smooth_strength),
                "branch_guard": True,
                "branch_repaired": bool(any(
                    int(event["sync_index"]) == int(row.get("sync_index", -1))
                    and int(event["side"]) == side for event in branch_events
                )),
                "finger_direction_rate_limited": bool(args.max_bone_angle_step_deg > 0),
                "fixed_bone_lengths": True,
            }
            for event in visibility_cleared:
                if (
                    int(event["sync_index"]) == int(row.get("sync_index", -1))
                    and int(event["side"]) == side
                ):
                    hand["visibility_guard"] = dict(event)
                    break

    output.mkdir(parents=True)
    (output / "accepted.jsonl").write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
    for name in ("rejected.jsonl",):
        if (source / name).is_file():
            shutil.copy2(source / name, output / name)
    summary = {}
    if (source / "summary.json").is_file():
        summary = json.loads((source / "summary.json").read_text(encoding="utf-8"))
    report = {
        "algorithm": "mano_palm_local_zero_phase_coherent_v2",
        "source": str(source),
        "identity": identity,
        "input_finite_joint_count_by_side": before_counts,
        "output_finite_joint_count_by_side": _finite_joint_counts(guarded),
        "changed_joint_count": int(changed_joints),
        "branch_repaired_joint_count": int(branch_count),
        "branch_events_preview": branch_events[:100],
        "post_projection_branch_repaired_joint_count": int(post_branch_count),
        "post_projection_branch_events_preview": post_branch_events[:100],
        "visibility_cleared_hand_count": int(len(visibility_cleared)),
        "visibility_cleared_preview": visibility_cleared[:100],
        "finger_direction_rate_limited_count": int(angle_limited_count),
        "parameters": {
            "identity_max_gap": int(args.identity_max_gap),
            "smooth_radius": int(args.smooth_radius),
            "smooth_strength": float(args.smooth_strength),
            "minimum_valid_joints": int(args.minimum_valid_joints),
            "bone_iterations": int(args.bone_iterations),
            "bone_strength": float(args.bone_strength),
            "max_bone_angle_step_deg": float(args.max_bone_angle_step_deg),
            "branch_threshold": float(args.branch_threshold),
            "branch_agreement": float(args.branch_agreement),
            "branch_max_joint": float(args.branch_max_joint),
            "branch_radius": int(args.branch_radius),
            "max_frame_distance": int(args.max_frame_distance),
        },
        "display_policy": {
            "strict_video": "keep rejected gaps empty",
            "continuous_video": "--interpolate-all-gaps 1 --all-gap-prediction-mode hold",
            "linear_gap_interpolation": "disabled because it can morph fingers",
        },
    }
    summary["fusion_algorithm"] = str(summary.get("fusion_algorithm", "existing_fusion")) + "+coherent_final_v2"
    summary["coherent_final"] = report
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output / "coherent_final_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
