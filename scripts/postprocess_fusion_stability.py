#!/usr/bin/env python3
"""Apply conservative visibility/identity guards to an existing fusion JSONL.

This is useful when the expensive WiLoR/SAM3 inference has already finished.
It does not change camera-space observations or fabricate joints: it only
reassigns physical hand identities across short rejected intervals and marks
an ambiguous 3-D hand missing when its evidence is too sparse or its camera
box jumps.  The same guards are part of the online fusion path in
``fusion_joint_temporal.py``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from fusion_joint_temporal import (  # noqa: E402
    _invalidate_excessive_temporal_jumps,
    _invalidate_view_jump_hands,
    stabilize_hand_identities,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--identity-max-gap", type=int, default=60)
    parser.add_argument("--min-hand-joints", type=int, default=18)
    parser.add_argument("--excessive-jump-threshold-m", type=float, default=0.030)
    parser.add_argument("--view-wrist-jump-px", type=float, default=180.0)
    parser.add_argument("--view-expanded-box-ratio", type=float, default=1.50)
    parser.add_argument("--view-expanded-box-wrist-px", type=float, default=130.0)
    return parser.parse_args()


def _points(hand: dict[str, Any]) -> np.ndarray:
    value = np.asarray(hand.get("joints_base_m", []), dtype=np.float64)
    if value.shape != (21, 3):
        return np.full((21, 3), np.nan, dtype=np.float64)
    return value


def main() -> int:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    if not source.is_file():
        raise FileNotFoundError(source)
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows.sort(key=lambda row: int(row["sync_index"]))
    identity = stabilize_hand_identities(
        rows,
        max_frame_gap=args.identity_max_gap,
        swap_margin=0.25,
        swap_min_cost=0.90,
    )
    frame_ids = np.asarray([int(row["sync_index"]) for row in rows], dtype=np.int64)
    report: dict[str, Any] = {
        "source": str(source),
        "parameters": {
            "identity_max_gap": int(args.identity_max_gap),
            "min_hand_joints": int(args.min_hand_joints),
            "excessive_jump_threshold_m": float(args.excessive_jump_threshold_m),
            "view_wrist_jump_px": float(args.view_wrist_jump_px),
            "view_expanded_box_ratio": float(args.view_expanded_box_ratio),
            "view_expanded_box_wrist_px": float(args.view_expanded_box_wrist_px),
        },
        "identity": identity,
        "sparse_hand_cleared_count": 0,
        "sparse_joint_cleared_count": 0,
        "view_jump_cleared_hand_count": 0,
        "view_jump_cleared_joint_count": 0,
        "excessive_jump_cleared_joint_count": 0,
    }

    for side in (0, 1):
        hands: list[dict[str, Any] | None] = [
            next((hand for hand in row.get("hands", []) if int(hand.get("side", -1)) == side), None)
            for row in rows
        ]
        points = np.asarray([_points(hand) if hand is not None else np.full((21, 3), np.nan) for hand in hands])
        valid = np.isfinite(points).all(axis=2)
        finite_count_before = valid.sum(axis=1).copy()
        sparse = valid.sum(axis=1) < max(int(args.min_hand_joints), 0)
        if np.any(sparse):
            points[sparse] = np.nan
            valid[sparse] = False
        report["sparse_hand_cleared_count"] += int(np.count_nonzero(sparse))
        report["sparse_joint_cleared_count"] += int(np.count_nonzero(sparse)) * 21

        previous_hands: list[dict[str, Any] | None] = [None]
        previous_hands.extend(hands[:-1])
        guarded, view_cleared, view_events = _invalidate_view_jump_hands(
            points,
            valid,
            [hand if hand is not None else {"views": {}} for hand in hands],
            previous_hands,
            frame_ids,
            wrist_jump_px=args.view_wrist_jump_px,
            expanded_box_ratio=args.view_expanded_box_ratio,
            expanded_box_wrist_px=args.view_expanded_box_wrist_px,
        )
        if view_events:
            points = guarded
            valid = np.isfinite(points).all(axis=2)
            report["view_jump_cleared_hand_count"] += len(view_events)
            report["view_jump_cleared_joint_count"] += int(np.count_nonzero(view_cleared))
        points, excessive = _invalidate_excessive_temporal_jumps(
            points, valid, frame_ids,
            max_jump_m=args.excessive_jump_threshold_m,
        )
        report["excessive_jump_cleared_joint_count"] += int(np.count_nonzero(excessive))

        for index, hand in enumerate(hands):
            if hand is None:
                continue
            hand["joints_base_m"] = points[index].tolist()
            if sparse[index]:
                hand["sparse_visibility_guard"] = {
                    "applied": True,
                    "minimum_finite_joints": int(args.min_hand_joints),
                    "finite_joint_count_before": int(finite_count_before[index]),
                }
            if int(frame_ids[index]) in view_events:
                hand["temporal_view_guard"] = {
                    "applied": True,
                    "cameras": view_events[int(frame_ids[index])],
                    "wrist_jump_threshold_px": float(args.view_wrist_jump_px),
                }

    output.mkdir(parents=True, exist_ok=False)
    (output / "accepted.jsonl").write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
    rejected = source.parent / "rejected.jsonl"
    if rejected.is_file():
        (output / "rejected.jsonl").write_bytes(rejected.read_bytes())
    summary_path = source.parent / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
    previous_algorithm = summary.get("fusion_algorithm") or "existing_fusion"
    summary["fusion_algorithm"] = str(previous_algorithm) + "+posthoc_stability_v1"
    summary["temporal_hand_identity"] = identity
    jt = summary.setdefault("joint_temporal_guard", {"enabled": True})
    jt.setdefault("parameters", {})
    jt["parameters"].update({
        "min_hand_joints": int(args.min_hand_joints),
        "excessive_jump_threshold_m": float(args.excessive_jump_threshold_m),
        "view_wrist_jump_px": float(args.view_wrist_jump_px),
        "view_expanded_box_ratio": float(args.view_expanded_box_ratio),
        "view_expanded_box_wrist_px": float(args.view_expanded_box_wrist_px),
    })
    jt["posthoc_stability"] = report
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (output / "stability_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
