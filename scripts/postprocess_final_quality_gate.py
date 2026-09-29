#!/usr/bin/env python3
"""Clear weak display-only 3-D poses before continuous rendering.

The expensive multiview fusion output is preserved.  A pose reconstructed
from only two cameras (or with an incomplete 21-joint array) is marked missing
in a copied display directory so the renderer's geodesic gap predictor uses
the next well-supported pose instead of showing an unstable two-view mesh.
Large global corrections made by the temporal depth-branch guard are treated
the same way: their reprojection may look plausible while the 3-D branch is
not physically trustworthy. Camera-native 2-D observations remain untouched.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
from typing import Any

import numpy as np


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--min-views", type=int, default=3)
    parser.add_argument("--min-joints", type=int, default=21)
    parser.add_argument(
        "--clear-temporal-global-correction", type=int, choices=(0, 1), default=1,
        help=(
            "clear a complete pose when the upstream temporal guard had to "
            "replace a global depth branch with a large correction"
        ),
    )
    parser.add_argument(
        "--max-temporal-global-residual-mm", type=float, default=35.0,
        help=(
            "residual threshold for --clear-temporal-global-correction; "
            "set 0 to clear every flagged global correction"
        ),
    )
    return parser.parse_args()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _clear_hand(hand: dict[str, Any], reason: str, support_views: int, valid_joints: int) -> None:
    hand["joints_base_m"] = np.full((21, 3), np.nan).tolist()
    if "unrefined_joints_base_m" in hand:
        hand["unrefined_joints_base_m"] = np.full((21, 3), np.nan).tolist()
    hand["final_quality_gate"] = {
        "applied": True,
        "reason": reason,
        "support_views": int(support_views),
        "valid_joints_before_clear": int(valid_joints),
    }


def main() -> int:
    args = _parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    if source == output:
        raise ValueError("--output must be a new directory")
    if args.min_views < 2:
        raise ValueError("--min-views must be at least 2")
    if not 1 <= args.min_joints <= 21:
        raise ValueError("--min-joints must be in [1, 21]")
    if args.max_temporal_global_residual_mm < 0.0:
        raise ValueError("--max-temporal-global-residual-mm must be non-negative")
    if not (source / "accepted.jsonl").is_file():
        raise FileNotFoundError(source / "accepted.jsonl")
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    shutil.copytree(source, output)

    rows = _read_jsonl(source / "accepted.jsonl")
    cleared = 0
    by_reason: dict[str, int] = {}
    for row in rows:
        for hand in row.get("hands", []):
            points = np.asarray(hand.get("joints_base_m", []), dtype=np.float64)
            valid = int(np.isfinite(points).all(axis=1).sum()) if points.shape == (21, 3) else 0
            support = len(hand.get("camera_ids", []))
            if support < args.min_views:
                reason = "too_few_support_views"
            elif valid < args.min_joints:
                reason = "incomplete_joint_array"
            else:
                anatomy = hand.get("anatomy_refinement", {})
                try:
                    global_residual = float(
                        anatomy.get("temporal_global_residual_mm", 0.0)
                    )
                except (TypeError, ValueError):
                    global_residual = 0.0
                global_correction = bool(
                    anatomy.get("temporal_global_corrected", False)
                )
                if (
                    args.clear_temporal_global_correction
                    and global_correction
                    and (
                        args.max_temporal_global_residual_mm <= 0.0
                        or global_residual >= args.max_temporal_global_residual_mm
                    )
                ):
                    reason = "temporal_global_branch_correction"
                else:
                    continue
            _clear_hand(hand, reason, support, valid)
            cleared += 1
            by_reason[reason] = by_reason.get(reason, 0) + 1
    _write_jsonl(output / "accepted.jsonl", rows)

    summary_path = output / "summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["final_quality_gate"] = {
            "enabled": True,
            "min_support_views": int(args.min_views),
            "min_valid_joints": int(args.min_joints),
            "clear_temporal_global_correction": bool(
                args.clear_temporal_global_correction
            ),
            "max_temporal_global_residual_mm": float(
                args.max_temporal_global_residual_mm
            ),
            "cleared_hand_count": int(cleared),
            "cleared_by_reason": by_reason,
            "source": str(source),
        }
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    report_path = output / "depth_branch_gate_report.json"
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report["final_quality_gate"] = {
            "min_support_views": int(args.min_views),
            "min_valid_joints": int(args.min_joints),
            "clear_temporal_global_correction": bool(
                args.clear_temporal_global_correction
            ),
            "max_temporal_global_residual_mm": float(
                args.max_temporal_global_residual_mm
            ),
            "cleared_hand_count": int(cleared),
            "cleared_by_reason": by_reason,
            "source": str(source),
        }
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(f"Final display quality gate: {source} -> {output}")
    print(f"Cleared hands: {cleared} | {by_reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
