#!/usr/bin/env python3
"""Remove depth-branch flips from an already fused hand track.

WiLoR/MANO can return two 3-D finger directions that have almost the same
projection in the cameras.  A per-frame bone-length constraint does not
resolve that ambiguity: the bone remains the right length, but it can turn
through 90--180 degrees for one frame and the rendered finger appears to
twist.  This pass keeps the calibrated/fused pose and camera-native 2-D
observations unchanged.  It only marks a complete hand missing when an
adjacent 3-D finger edge makes an implausibly large turn while the same edge
is stable in at least three camera images.  The renderer's ``hold`` display
policy then keeps the last trusted complete pose instead of drawing a warped
one.

This is deliberately a *rejection* gate, rather than another independent
triangulation.  The source ``joints_base_m`` has already gone through the
anatomy and temporal guards; re-solving every ray here can select a different
depth branch and make the temporal track worse.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
from typing import Any, Mapping

import numpy as np

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from postprocess_multiview_final import _fix_hand_identities  # noqa: E402


PALM_ANCHORS = np.asarray((0, 5, 9, 13, 17), dtype=np.int32)

# The four MCP spokes are palm structure, not finger articulation.  The
# remaining edges are the thumb and four finger chains and are the edges that
# visibly twist when a depth branch changes.
FINGER_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (5, 6), (6, 7), (7, 8),
    (9, 10), (10, 11), (11, 12),
    (13, 14), (14, 15), (15, 16),
    (17, 18), (18, 19), (19, 20),
)


def _points(value: Any) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    return value if value.shape == (21, 3) else np.full((21, 3), np.nan)


def _unit(value: np.ndarray) -> np.ndarray | None:
    value = np.asarray(value, dtype=np.float64)
    norm = float(np.linalg.norm(value))
    return value / norm if norm > 1e-9 and np.isfinite(norm) else None


def _angle_deg(first: np.ndarray, second: np.ndarray) -> float:
    first_unit = _unit(first)
    second_unit = _unit(second)
    if first_unit is None or second_unit is None:
        return float("nan")
    return float(np.degrees(np.arccos(np.clip(first_unit @ second_unit, -1.0, 1.0))))


def _rigid_map(previous: np.ndarray, current: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Return row-vector ``previous @ rotation + translation`` in current frame."""
    if not (
        previous.shape == (21, 3)
        and current.shape == (21, 3)
        and np.isfinite(previous[PALM_ANCHORS]).all()
        and np.isfinite(current[PALM_ANCHORS]).all()
    ):
        return None
    first = previous[PALM_ANCHORS]
    second = current[PALM_ANCHORS]
    first_center = np.mean(first, axis=0)
    second_center = np.mean(second, axis=0)
    first_centered = first - first_center
    second_centered = second - second_center
    left, _, right_t = np.linalg.svd(first_centered.T @ second_centered)
    rotation = left @ right_t
    if np.linalg.det(rotation) < 0.0:
        left[:, -1] *= -1.0
        rotation = left @ right_t
    translation = second_center - first_center @ rotation
    return rotation, translation


def _edge_2d_stats(
    previous: Mapping[str, Any], current: Mapping[str, Any], parent: int, child: int,
) -> tuple[float, float, int] | None:
    """Median image-space edge turn/scale over all common camera views."""
    values: list[float] = []
    ratios: list[float] = []
    previous_views = previous.get("views", {})
    current_views = current.get("views", {})
    for camera in set(previous_views).intersection(current_views):
        first = np.asarray(previous_views[camera].get("joints_2d", []), dtype=np.float64)
        second = np.asarray(current_views[camera].get("joints_2d", []), dtype=np.float64)
        if first.shape != (21, 2) or second.shape != (21, 2):
            continue
        if not (
            np.isfinite(first[[parent, child]]).all()
            and np.isfinite(second[[parent, child]]).all()
        ):
            continue
        first_edge = first[child] - first[parent]
        second_edge = second[child] - second[parent]
        first_norm = float(np.linalg.norm(first_edge))
        second_norm = float(np.linalg.norm(second_edge))
        # A sub-pixel edge has no useful orientation and should not veto a
        # hand.  It is also common when a glove is briefly occluded.
        if first_norm < 2.0 or second_norm < 2.0:
            continue
        values.append(_angle_deg(first_edge, second_edge))
        ratios.append(second_norm / first_norm)
    if not values:
        return None
    return float(np.median(values)), float(np.median(ratios)), len(values)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--identity-max-gap", type=int, default=10)
    parser.add_argument(
        "--min-3d-edge-turn-deg", type=float, default=75.0,
        help="minimum calibrated 3-D edge turn that can be a depth branch",
    )
    parser.add_argument(
        "--max-2d-edge-turn-deg", type=float, default=30.0,
        help="maximum median camera-image edge turn for a branch candidate",
    )
    parser.add_argument("--min-common-views", type=int, default=3)
    parser.add_argument("--min-edge-scale", type=float, default=0.45)
    parser.add_argument("--max-edge-scale", type=float, default=2.20)
    parser.add_argument(
        "--min-palm-width-m", type=float, default=0.025,
        help="reject a complete pose when wrist-to-pinky-MCP width collapses",
    )
    parser.add_argument("--max-palm-width-m", type=float, default=0.180)
    parser.add_argument(
        "--clear-radius", type=int, default=1,
        help="clear this many neighbouring complete poses around a branch pair",
    )
    parser.add_argument("--max-frames", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    if not (source / "accepted.jsonl").is_file():
        raise FileNotFoundError(source / "accepted.jsonl")
    if output.exists():
        raise FileExistsError(output)
    rows = [
        json.loads(line)
        for line in (source / "accepted.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows.sort(key=lambda row: int(row.get("sync_index", 0)))
    if args.max_frames:
        rows = rows[: int(args.max_frames)]
    identity = _fix_hand_identities(rows, max_gap=args.identity_max_gap)

    frame_ids = np.asarray([int(row.get("sync_index", 0)) for row in rows], dtype=np.int64)
    points = np.full((len(rows), 2, 21, 3), np.nan, dtype=np.float64)
    complete = np.zeros((len(rows), 2), dtype=bool)
    hands: dict[tuple[int, int], Mapping[str, Any]] = {}
    for index, row in enumerate(rows):
        for hand in row.get("hands", []):
            side = int(hand.get("side", -1))
            if side not in (0, 1):
                continue
            hands[(index, side)] = hand
            pose = _points(hand.get("joints_base_m", []))
            if pose.shape == (21, 3):
                points[index, side] = pose
                complete[index, side] = bool(np.isfinite(pose).all())

    # First remove the obvious palm-frame degeneracy.  When the 5--17 span
    # collapses to 16--24 mm the local palm basis is ill-conditioned and all
    # fingers can appear to rotate together even though their 2-D projections
    # barely move.
    cleared: set[tuple[int, int]] = set()
    palm_events: list[dict[str, Any]] = []
    for index in range(len(rows)):
        for side in (0, 1):
            if not complete[index, side]:
                continue
            width = float(np.linalg.norm(points[index, side, 5] - points[index, side, 17]))
            if float(args.min_palm_width_m) <= width <= float(args.max_palm_width_m):
                continue
            cleared.add((index, side))
            palm_events.append({
                "sync_index": int(frame_ids[index]), "side": side,
                "palm_width_m": width, "reason": "palm_width_degenerate",
            })

    # Detect an adjacent branch pair.  Clearing both endpoints is intentional:
    # 2-D evidence cannot tell whether the first or second depth solution is
    # the bad one, but keeping either one would expose a one-frame kink.
    branch_events: list[dict[str, Any]] = []
    for index in range(1, len(rows)):
        if int(frame_ids[index]) - int(frame_ids[index - 1]) != 1:
            continue
        for side in (0, 1):
            if not (complete[index, side] and complete[index - 1, side]):
                continue
            previous = points[index - 1, side]
            current = points[index, side]
            mapped = _rigid_map(previous, current)
            if mapped is None:
                continue
            rotation, translation = mapped
            aligned_previous = previous @ rotation + translation
            edge_candidates: list[dict[str, Any]] = []
            for parent, child in FINGER_EDGES:
                turn_3d = _angle_deg(
                    aligned_previous[child] - aligned_previous[parent],
                    current[child] - current[parent],
                )
                if not np.isfinite(turn_3d) or turn_3d < float(args.min_3d_edge_turn_deg):
                    continue
                stats = _edge_2d_stats(
                    hands[(index - 1, side)], hands[(index, side)], parent, child,
                )
                if stats is None:
                    continue
                turn_2d, scale_2d, view_count = stats
                if view_count < int(args.min_common_views):
                    continue
                if turn_2d > float(args.max_2d_edge_turn_deg):
                    continue
                if not float(args.min_edge_scale) <= scale_2d <= float(args.max_edge_scale):
                    continue
                edge_candidates.append({
                    "parent": int(parent), "child": int(child),
                    "turn_3d_deg": float(turn_3d),
                    "turn_2d_deg": float(turn_2d),
                    "edge_scale_2d": float(scale_2d),
                    "common_views": int(view_count),
                })
            if not edge_candidates:
                continue
            radius = max(int(args.clear_radius), 0)
            lo = max(0, index - radius - 1)
            hi = min(len(rows), index + radius + 1)
            for candidate_index in range(lo, hi):
                if complete[candidate_index, side]:
                    cleared.add((candidate_index, side))
            branch_events.append({
                "sync_index_previous": int(frame_ids[index - 1]),
                "sync_index_current": int(frame_ids[index]),
                "side": int(side), "edges": edge_candidates,
                "clear_radius": radius,
            })

    for index, side in cleared:
        pose = points[index, side]
        if not np.isfinite(pose).all():
            continue
        points[index, side] = np.nan
        hand = hands.get((index, side))
        if hand is not None:
            hand["joints_base_m"] = points[index, side].tolist()
            hand["final_stability"] = {
                "algorithm": "six_view_2d_consensus_depth_branch_gate_v1",
                "preserved_camera_2d": True,
                "cleared_for_conservative_hold": True,
            }
    # Keep unchanged source poses and annotate them as having passed the gate.
    for index, side in np.argwhere(complete):
        hand = hands.get((int(index), int(side)))
        if hand is None or (int(index), int(side)) in cleared:
            continue
        hand["final_stability"] = {
            "algorithm": "six_view_2d_consensus_depth_branch_gate_v1",
            "preserved_camera_2d": True,
            "cleared_for_conservative_hold": False,
        }

    output.mkdir(parents=True)
    (output / "accepted.jsonl").write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
    for name in ("rejected.jsonl",):
        if (source / name).is_file():
            shutil.copy2(source / name, output / name)
    summary = json.loads((source / "summary.json").read_text(encoding="utf-8")) if (source / "summary.json").is_file() else {}
    report = {
        "algorithm": "six_view_2d_consensus_depth_branch_gate_v1",
        "source": str(source),
        "identity": identity,
        "input_complete_hand_count": int(np.count_nonzero(complete)),
        "cleared_complete_hand_count": int(len(cleared)),
        "palm_degenerate_hand_count": int(len(palm_events)),
        "branch_pair_count": int(len(branch_events)),
        "palm_events_preview": palm_events[:100],
        "branch_events_preview": branch_events[:100],
        "parameters": {
            key: value for key, value in vars(args).items()
            if key not in ("source", "output")
        },
        "display_policy": {
            "strict_video": "keep branch-gated poses missing",
            "continuous_video": (
                "--interpolate-all-gaps 1 --all-gap-prediction-mode geodesic "
                "(fixed-length bones, follows the endpoint motion)"
            ),
            "hold_previous_video": (
                "--interpolate-all-gaps 1 --all-gap-prediction-mode hold-previous "
                "(zero morphing, but can lag during a gap)"
            ),
            "linear_gap_interpolation": "disabled because it morphs fingers",
        },
    }
    summary["fusion_algorithm"] = str(summary.get("fusion_algorithm", "existing_fusion")) + "+depth_branch_gate_v1"
    summary["depth_branch_gate"] = report
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "depth_branch_gate_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
