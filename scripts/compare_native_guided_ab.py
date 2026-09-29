#!/usr/bin/env python3
"""Create a compact, same-frame native-vs-guided WiLoR A/B report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _load_jsonl(path: Path) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            rows[int(row["sync_index"])] = row
    return rows


def _stats(values: list[float], scale: float = 1.0) -> dict[str, Any]:
    data = np.asarray(values, dtype=np.float64)
    data = data[np.isfinite(data)] * scale
    if not len(data):
        return {"sample_count": 0}
    return {
        "sample_count": int(len(data)),
        "p50": float(np.percentile(data, 50)),
        "p95": float(np.percentile(data, 95)),
        "p99": float(np.percentile(data, 99)),
        "max": float(np.max(data)),
        "over_30": int(np.count_nonzero(data > 30.0)),
        "over_50": int(np.count_nonzero(data > 50.0)),
        "over_100": int(np.count_nonzero(data > 100.0)),
    }


def _continuity(root: Path) -> dict[str, Any]:
    rows = _load_jsonl(root / "accepted.jsonl")
    previous: dict[int, tuple[int, np.ndarray]] = {}
    palm_steps: list[float] = []
    joint_steps: list[float] = []
    gap_count = 0
    for sync_index in sorted(rows):
        current_sides: set[int] = set()
        for hand in rows[sync_index].get("hands", []):
            side = int(hand.get("side", -1))
            if side not in (0, 1):
                continue
            points = np.asarray(hand.get("joints_base_m", []), dtype=np.float64)
            if points.shape != (21, 3):
                continue
            current_sides.add(side)
            old = previous.get(side)
            if old is not None:
                old_sync, old_points = old
                if sync_index - old_sync == 1:
                    finite = (
                        np.isfinite(points).all(axis=1)
                        & np.isfinite(old_points).all(axis=1)
                    )
                    if finite[0]:
                        palm_steps.append(float(np.linalg.norm(points[0] - old_points[0])))
                    joint_steps.extend(
                        np.linalg.norm(points[finite] - old_points[finite], axis=1).tolist()
                    )
                elif sync_index > old_sync + 1:
                    gap_count += sync_index - old_sync - 1
            previous[side] = (sync_index, points)
        for side in set(previous) - current_sides:
            # Do not count a missing side as a motion sample; the next valid
            # row will be separated by a gap and will not be compared.
            pass
    return {
        "consecutive_palm_step_mm": _stats(palm_steps, scale=1000.0),
        "consecutive_joint_step_mm": _stats(joint_steps, scale=1000.0),
        "accepted_track_gap_frames": int(gap_count),
    }


def _summary(root: Path) -> dict[str, Any]:
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    return {
        "path": str(root.resolve()),
        "processed_frame_count": summary.get("processed_frame_count"),
        "accepted_frame_count": summary.get("accepted_frame_count"),
        "rejected_frame_count": summary.get("rejected_frame_count"),
        "accepted_rate": summary.get("accepted_rate"),
        "rejected_reason_counts": summary.get("rejected_reason_counts", {}),
        "reprojection_median_px": summary.get("reprojection_median_px"),
        "reprojection_p95_px": summary.get("reprojection_p95_px"),
        "all_view_reprojection_median_px": summary.get("all_view_reprojection_median_px"),
        "selected_view_count_median": summary.get("selected_view_count_median"),
        "valid_joint_inlier_view_count_median": summary.get(
            "valid_joint_inlier_view_count_median"
        ),
        "execution_note": summary.get("execution_note"),
        "continuity": _continuity(root),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", required=True, type=Path)
    parser.add_argument("--guided", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--ab-video", type=Path)
    args = parser.parse_args()
    native = _summary(args.native.resolve())
    guided = _summary(args.guided.resolve())
    report: dict[str, Any] = {
        "schema_version": 1,
        "comparison": "same normalized recording and same frame range",
        "native": native,
        "guided": guided,
        "difference": {
            "accepted_frame_delta": (
                guided["accepted_frame_count"] - native["accepted_frame_count"]
            ),
            "accepted_rate_delta_percentage_points": 100.0 * (
                guided["accepted_rate"] - native["accepted_rate"]
            ),
            "reprojection_median_delta_px": (
                guided["reprojection_median_px"] - native["reprojection_median_px"]
            ),
            "native_consecutive_palm_p95_minus_guided_mm": (
                native["continuity"]["consecutive_palm_step_mm"].get("p95", 0.0)
                - guided["continuity"]["consecutive_palm_step_mm"].get("p95", 0.0)
            ),
            "native_consecutive_joint_p95_minus_guided_mm": (
                native["continuity"]["consecutive_joint_step_mm"].get("p95", 0.0)
                - guided["continuity"]["consecutive_joint_step_mm"].get("p95", 0.0)
            ),
        },
        "recommendation": {
            "3d": "guided quality4 temporal output only when complete and locally trusted",
            "2d": "current camera WiLoR 2D fallback when projected 3D residual is high",
            "min_joint_inlier_views": 3,
            "gap_policy": "interpolate only short gaps; leave long gaps missing",
            "note": "No ground-truth 3D is present; this is not an absolute pose accuracy score.",
        },
    }
    if args.ab_video is not None:
        report["ab_video"] = str(args.ab_video.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["difference"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
