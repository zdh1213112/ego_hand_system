#!/usr/bin/env python3
"""Refine an existing multiview fusion with temporally stable 2-D rays.

WiLoR/SAM3 inference is the expensive part of a recording.  This post-pass is
intended for an already completed ``fusion_multiview`` directory: it smooths
the camera-native joint observations in a short, zero-phase window, rebuilds
each 3-D joint from the six calibrated rays, and applies a conservative
temporal step guard.  It never changes the source 2-D predictions and it does
not fill rejected frames.

The post-pass is deliberately separate from the inference JSONL so that an
experiment can be compared with its original ``joints_base_m`` without
rerunning the detector or SAM3.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil
import sys
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from ego_data.calibration import CameraCalibration  # noqa: E402
from fusion_joint_temporal import (  # noqa: E402
    _finite_point,
    _quality,
    _ray_condition_number,
    _rays_for_joint,
    _regularized_ray_point,
    _remove_isolated_step_spikes,
    _limit_temporal_steps,
)
from render_multiview_wilor import _build_2d_stabilized_views  # noqa: E402
from stabilize_hand_3d import smooth_3d_landmarks_in_palm_frame  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", required=True, type=Path,
        help="normalized_multiview directory containing calibration/ and multiview_frames.csv",
    )
    parser.add_argument("--source", required=True, type=Path, help="existing fusion directory")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cameras", nargs="+", default=None)
    parser.add_argument(
        "--input-joints-key",
        choices=("joints_base_m", "unrefined_joints_base_m", "hybrid"),
        default="joints_base_m",
        help=(
            "3-D observations to refine.  ``unrefined_joints_base_m`` "
            "bypasses a strong anatomical projection that can fold fingers; "
            "``hybrid`` replaces only clear 2-D reprojection failures when "
            "the unrefined point is temporally supported."
        ),
    )
    parser.add_argument(
        "--min-input-joints",
        type=int,
        default=18,
        help=(
            "minimum finite joints required when using the unrefined input; "
            "sparser hands are kept missing instead of mixed with refined points"
        ),
    )
    parser.add_argument(
        "--hybrid-min-reprojection-gain-px", type=float, default=4.0,
        help="minimum median 2-D gain required for a hybrid raw-joint replacement",
    )
    parser.add_argument(
        "--hybrid-min-reprojection-p95-gain-px", type=float, default=8.0,
        help="minimum p95 2-D gain required for a hybrid raw-joint replacement",
    )
    parser.add_argument(
        "--hybrid-max-neighbor-step-m", type=float, default=0.080,
        help="maximum immediate raw 3-D neighbor step for a hybrid replacement",
    )
    parser.add_argument("--smoothing-radius", type=int, default=5)
    parser.add_argument("--smoothing-strength", type=float, default=0.50)
    parser.add_argument("--ray-prior-weight", type=float, default=0.60)
    parser.add_argument("--ray-blend", type=float, default=0.70)
    parser.add_argument("--max-ray-shift-m", type=float, default=0.080)
    parser.add_argument("--shape-smoothing-radius", type=int, default=2)
    parser.add_argument("--shape-smoothing-strength", type=float, default=0.30)
    parser.add_argument("--max-step-m", type=float, default=0.040)
    parser.add_argument("--min-views", type=int, default=3)
    parser.add_argument("--max-condition-number", type=float, default=1000.0)
    parser.add_argument("--max-frames", type=int, default=0)
    return parser.parse_args()


def _points(value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (21, 3):
        return np.full((21, 3), np.nan, dtype=np.float64)
    return array


def _load_rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"output already exists: {args.output}")
    if args.smoothing_radius < 0 or args.shape_smoothing_radius < 0:
        raise ValueError("smoothing radii must be non-negative")
    if not 0.0 <= args.smoothing_strength <= 1.0:
        raise ValueError("--smoothing-strength must be in [0, 1]")
    if not 0.0 <= args.shape_smoothing_strength <= 1.0:
        raise ValueError("--shape-smoothing-strength must be in [0, 1]")
    if not 0.0 <= args.ray_blend <= 1.0:
        raise ValueError("--ray-blend must be in [0, 1]")
    if args.ray_prior_weight <= 0.0 or args.max_ray_shift_m <= 0.0:
        raise ValueError("ray prior and shift limits must be positive")
    if args.max_step_m <= 0.0 or args.min_views < 2:
        raise ValueError("--max-step-m must be positive and --min-views at least 2")
    if args.max_condition_number <= 0.0:
        raise ValueError("--max-condition-number must be positive")
    if args.min_input_joints < 1:
        raise ValueError("--min-input-joints must be positive")
    if args.hybrid_min_reprojection_gain_px < 0.0:
        raise ValueError("--hybrid-min-reprojection-gain-px must be non-negative")
    if args.hybrid_min_reprojection_p95_gain_px < 0.0:
        raise ValueError(
            "--hybrid-min-reprojection-p95-gain-px must be non-negative"
        )
    if args.hybrid_max_neighbor_step_m <= 0.0:
        raise ValueError("--hybrid-max-neighbor-step-m must be positive")

    dataset = args.dataset.resolve()
    source = args.source.resolve()
    source_jsonl = source / "accepted.jsonl"
    if not dataset.is_dir() or not source_jsonl.is_file():
        raise FileNotFoundError(f"missing dataset or source accepted.jsonl: {dataset}, {source_jsonl}")
    dataset_manifest = dataset / "manifest.json"
    manifest = json.loads(dataset_manifest.read_text(encoding="utf-8"))
    cameras = tuple(args.cameras or manifest.get("camera_ids", ()))
    if not cameras:
        raise ValueError("no cameras found")
    calibrations = {
        camera: CameraCalibration.load(dataset / "calibration" / f"{camera}.json")
        for camera in cameras
    }
    with (dataset / "multiview_frames.csv").open(
        "r", encoding="utf-8", newline=""
    ) as stream:
        frame_rows = list(csv.DictReader(stream))
    rows = _load_rows(source_jsonl)
    rows.sort(key=lambda row: int(row["sync_index"]))
    if args.max_frames:
        rows = rows[: int(args.max_frames)]
    accepted = {int(row["sync_index"]): row for row in rows}

    smoothed_views = _build_2d_stabilized_views(
        frame_rows,
        accepted,
        cameras,
        args.smoothing_radius,
        args.smoothing_strength,
        22.0,
        45.0,
        point_key="joints_2d",
    )
    frame_ids = np.asarray(
        [int(row["sync_index"]) for row in rows], dtype=np.int64
    )
    frame_count = len(rows)
    points = np.full((2, frame_count, 21, 3), np.nan, dtype=np.float64)
    unrefined_points = np.full_like(points, np.nan)
    hands: dict[tuple[int, int], dict[str, Any]] = {}
    for index, row in enumerate(rows):
        for hand in row.get("hands", []):
            side = int(hand.get("side", -1))
            if side not in (0, 1):
                continue
            hands[(index, side)] = hand
            unrefined = _points(hand.get("unrefined_joints_base_m", []))
            unrefined_points[side, index] = unrefined
            selected = _points(
                hand.get(
                    "joints_base_m"
                    if args.input_joints_key == "hybrid"
                    else args.input_joints_key,
                    [],
                )
            )
            if args.input_joints_key == "unrefined_joints_base_m":
                # A partial unrefined skeleton cannot be safely completed by
                # borrowing the strongly projected result: the two sources
                # can have different depth branches and a mixed hand visibly
                # twists at the missing finger.  Treat it as one visibility
                # unit and let the renderer show a clean dropout.
                finite_count = int(np.isfinite(selected).all(axis=1).sum())
                if finite_count < int(args.min_input_joints):
                    selected[:] = np.nan
                    hand["unrefined_visibility_guard"] = {
                        "applied": True,
                        "minimum_finite_joints": int(args.min_input_joints),
                        "finite_joint_count_before": finite_count,
                    }
            points[side, index] = selected

    hybrid_replacement_count = 0
    if args.input_joints_key == "hybrid":
        # Keep the anatomically projected track as the default because it is
        # temporally coherent.  Use the unrefined multi-view point only when
        # it is demonstrably better supported by the six image observations,
        # has a usable ray baseline, and agrees with at least one neighbouring
        # raw sample.  This prevents a single excellent-looking but depth-
        # ambiguous ray intersection from switching the whole finger branch.
        for (index, side), hand in hands.items():
            raw = unrefined_points[side, index]
            current = points[side, index]
            rays_cache = [_rays_for_joint(hand, joint, calibrations)
                          for joint in range(21)]
            for joint, raw_point in enumerate(raw):
                if not np.isfinite(raw_point).all() or not np.isfinite(current[joint]).all():
                    continue
                rays = rays_cache[joint]
                if len(rays) < int(args.min_views):
                    continue
                condition = _ray_condition_number(rays)
                if not np.isfinite(condition) or condition > float(args.max_condition_number):
                    continue
                raw_neighbours: list[float] = []
                for neighbour in (index - 1, index + 1):
                    if not (0 <= neighbour < frame_count):
                        continue
                    if int(frame_ids[neighbour]) - int(frame_ids[index]) not in (-1, 1):
                        continue
                    value = unrefined_points[side, neighbour, joint]
                    if np.isfinite(value).all():
                        raw_neighbours.append(float(np.linalg.norm(raw_point - value)))
                if not raw_neighbours or min(raw_neighbours) > float(args.hybrid_max_neighbor_step_m):
                    continue
                current_quality = _quality(current[joint], rays)
                raw_quality = _quality(raw_point, rays)
                if (
                    current_quality[0] - raw_quality[0]
                    >= float(args.hybrid_min_reprojection_gain_px)
                    and current_quality[1] - raw_quality[1]
                    >= float(args.hybrid_min_reprojection_p95_gain_px)
                ):
                    points[side, index, joint] = raw_point
                    hybrid_replacement_count += 1

    corrected = np.zeros((2, frame_count, 21), dtype=bool)
    proposed = 0
    accepted_count = 0
    rejected_condition = 0
    rejected_shift = 0
    for (index, side), hand in hands.items():
        original = points[side, index].copy()
        if not np.isfinite(original).any():
            continue
        temporary_views: dict[str, dict[str, Any]] = {}
        for camera, view in hand.get("views", {}).items():
            replacement = smoothed_views.get((int(frame_ids[index]), side, camera))
            temporary_views[camera] = (
                dict(replacement) if replacement is not None else dict(view)
            )
        temporary_hand = dict(hand)
        temporary_hand["views"] = temporary_views
        palm_values = original[[0, 5, 9, 13, 17]]
        palm = (
            np.nanmedian(palm_values, axis=0)
            if np.isfinite(palm_values).all()
            else np.full(3, np.nan)
        )
        for joint in range(21):
            if not np.isfinite(original[joint]).all():
                continue
            rays = _rays_for_joint(temporary_hand, joint, calibrations)
            if len(rays) < args.min_views:
                continue
            condition = _ray_condition_number(rays)
            if not np.isfinite(condition) or condition >= args.max_condition_number:
                rejected_condition += 1
                continue
            candidate = _regularized_ray_point(
                rays, original[joint], args.ray_prior_weight
            )
            if not _finite_point(candidate, palm, 0.55):
                continue
            shift = float(np.linalg.norm(candidate - original[joint]))
            if shift > args.max_ray_shift_m:
                rejected_shift += 1
                continue
            before = _quality(original[joint], rays)
            after = _quality(candidate, rays)
            # The 2-D smoother intentionally removes isolated image spikes;
            # still reject a 3-D proposal that makes the six-view residual
            # clearly worse than the current pose.
            if after[0] > before[0] + 2.0 or after[1] > before[1] + 15.0:
                continue
            points[side, index, joint] = (
                (1.0 - args.ray_blend) * original[joint]
                + args.ray_blend * candidate
            )
            corrected[side, index, joint] = True
            proposed += 1

    # Smooth the local articulation after the ray pass, but do not smooth a
    # missing interval or the hand's global translation.  The final ray guard
    # below still has the authority to cap temporal branches.
    valid = np.isfinite(points).all(axis=3)
    if args.shape_smoothing_radius > 0 and args.shape_smoothing_strength > 0.0:
        points = smooth_3d_landmarks_in_palm_frame(
            points, valid, args.shape_smoothing_radius,
            args.shape_smoothing_strength,
        )
    continuity = np.zeros_like(corrected)
    for side in (0, 1):
        points[side], isolated = _remove_isolated_step_spikes(
            points[side], valid[side], frame_ids, args.max_step_m
        )
        points[side], limited = _limit_temporal_steps(
            points[side], valid[side], frame_ids, args.max_step_m
        )
        continuity[side] = isolated | limited
        corrected[side] |= continuity[side]

    for (index, side), hand in hands.items():
        hand["unrefined_joints_base_m"] = hand.get(
            "unrefined_joints_base_m", hand.get("joints_base_m", [])
        )
        if args.input_joints_key == "unrefined_joints_base_m":
            hand["input_joints_key"] = args.input_joints_key
        hand["joints_base_m"] = points[side, index].tolist()
        count = int(corrected[side, index].sum())
        accepted_count += count
        hand["ray_reprojection_temporal_refinement"] = {
            "applied": bool(count),
            "corrected_joint_count": count,
            "smoothed_2d_radius": int(args.smoothing_radius),
            "smoothed_2d_strength": float(args.smoothing_strength),
            "ray_blend": float(args.ray_blend),
            "shape_smoothing_radius": int(args.shape_smoothing_radius),
            "shape_smoothing_strength": float(args.shape_smoothing_strength),
            "max_step_m": float(args.max_step_m),
            "input_joints_key": args.input_joints_key,
            "min_input_joints": int(args.min_input_joints),
            "hybrid_replacement_count": int(hybrid_replacement_count),
        }

    output = args.output.resolve()
    output.mkdir(parents=True)
    (output / "accepted.jsonl").write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
    for filename in ("rejected.jsonl",):
        path = source / filename
        if path.is_file():
            shutil.copy2(path, output / filename)
    summary_path = source / "summary.json"
    summary = (
        json.loads(summary_path.read_text(encoding="utf-8"))
        if summary_path.is_file() else {}
    )
    summary["fusion_algorithm"] = (
        str(summary.get("fusion_algorithm", "existing_fusion"))
        + "+2d_temporal_ray_reprojection_v1"
    )
    summary["ray_reprojection_temporal_refinement"] = {
        "enabled": True,
        "input_source": str(source),
        "smoothed_view_count": int(len(smoothed_views)),
        "proposed_joint_count": int(proposed),
        "accepted_joint_count": int(accepted_count),
        "condition_rejected_joint_count": int(rejected_condition),
        "shift_rejected_joint_count": int(rejected_shift),
        "parameters": {
            "cameras": list(cameras),
            "smoothing_radius": int(args.smoothing_radius),
            "smoothing_strength": float(args.smoothing_strength),
            "ray_prior_weight": float(args.ray_prior_weight),
            "ray_blend": float(args.ray_blend),
            "max_ray_shift_m": float(args.max_ray_shift_m),
            "shape_smoothing_radius": int(args.shape_smoothing_radius),
            "shape_smoothing_strength": float(args.shape_smoothing_strength),
            "max_step_m": float(args.max_step_m),
            "min_views": int(args.min_views),
            "max_condition_number": float(args.max_condition_number),
            "input_joints_key": args.input_joints_key,
            "min_input_joints": int(args.min_input_joints),
            "hybrid_min_reprojection_gain_px": float(
                args.hybrid_min_reprojection_gain_px
            ),
            "hybrid_min_reprojection_p95_gain_px": float(
                args.hybrid_min_reprojection_p95_gain_px
            ),
            "hybrid_max_neighbor_step_m": float(args.hybrid_max_neighbor_step_m),
            "hybrid_replacement_count": int(hybrid_replacement_count),
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary["ray_reprojection_temporal_refinement"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
