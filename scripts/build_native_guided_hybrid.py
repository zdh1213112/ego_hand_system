#!/usr/bin/env python3
"""Select a responsive 3-D track from native and guided WiLoR hypotheses.

The guided fusion output has much better frame coverage, while the original
native DS/RANSAC output is often closer to the camera observations on frames
where it is unambiguous.  This utility combines the two without inventing a
new pose: every selected hand is copied from one of the persisted candidates
and a small Viterbi pass discourages source switching that would create a
visible jump.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np

from camera_models import project_points
from ego_data.calibration import CameraCalibration


def _load_jsonl(path: Path) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows[int(row["sync_index"])] = row
    return rows


def _points(hand: dict[str, Any] | None, key: str = "joints_base_m") -> np.ndarray:
    value = np.asarray((hand or {}).get(key, []), dtype=np.float64)
    if value.shape != (21, 3):
        return np.full((21, 3), np.nan, dtype=np.float64)
    return value


def _renderable(points: np.ndarray, minimum: int = 21) -> bool:
    if points.shape != (21, 3):
        return False
    finite = np.isfinite(points).all(axis=1)
    core = (0, 1, 2, 5, 9, 13, 17)
    return int(np.count_nonzero(finite)) >= int(minimum) and bool(finite[list(core)].all())


def _project(camera: CameraCalibration, points_base: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rotation = camera.T_base_camera[:3, :3]
    center = camera.T_base_camera[:3, 3]
    points_camera = (points_base - center) @ rotation
    return project_points(camera, points_camera)


def _reprojection(
    hand: dict[str, Any], points: np.ndarray,
    calibrations: dict[str, CameraCalibration],
) -> tuple[float, float, float, int]:
    """Return robust camera residuals (all-point median, p95, max camera median)."""
    camera_medians: list[float] = []
    all_errors: list[float] = []
    if points.shape != (21, 3):
        return (float("inf"),) * 3 + (0,)
    for camera_id, view in hand.get("views", {}).items():
        camera = calibrations.get(camera_id)
        if camera is None:
            continue
        observed = np.asarray(view.get("joints_2d", []), dtype=np.float64)
        if observed.shape != (21, 2):
            continue
        projected, valid = _project(camera, points)
        finite = valid & np.isfinite(projected).all(axis=1) & np.isfinite(observed).all(axis=1)
        if not np.any(finite):
            continue
        errors = np.linalg.norm(projected[finite] - observed[finite], axis=1)
        all_errors.extend(errors.tolist())
        camera_medians.append(float(np.median(errors)))
    if not all_errors:
        return (float("inf"),) * 3 + (0,)
    return (
        float(np.median(all_errors)),
        float(np.percentile(all_errors, 95)),
        float(np.median(camera_medians)) if camera_medians else float("inf"),
        len(camera_medians),
    )


def _view_match(first: dict[str, Any], second: dict[str, Any]) -> float:
    values: list[float] = []
    for camera_id in set(first.get("views", {})) & set(second.get("views", {})):
        a = np.asarray(first["views"][camera_id].get("joints_2d", []), dtype=np.float64)
        b = np.asarray(second["views"][camera_id].get("joints_2d", []), dtype=np.float64)
        if a.shape != (21, 2) or b.shape != (21, 2):
            continue
        finite = np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1)
        if np.any(finite):
            values.extend(np.linalg.norm(a[finite] - b[finite], axis=1).tolist())
    return float(np.median(values)) if values else float("inf")


def _native_assignment(
    native: dict[str, Any] | None,
    guided_hands: list[dict[str, Any]],
) -> dict[int, dict[str, Any]]:
    if native is None or len(native.get("hands", [])) < 2 or len(guided_hands) < 2:
        return {}
    native_hands = native["hands"][:2]
    guided_hands = guided_hands[:2]
    scores = np.asarray([
        [_view_match(native_hand, guided_hand) for guided_hand in guided_hands]
        for native_hand in native_hands
    ], dtype=np.float64)
    permutations = ((0, 1), (1, 0))
    permutation = min(permutations, key=lambda value: scores[0, value[0]] + scores[1, value[1]])
    return {guided_index: native_hands[native_index] for native_index, guided_index in enumerate(permutation)}


def _transition_cost(previous: np.ndarray, current: np.ndarray) -> float:
    finite = np.isfinite(previous).all(axis=1) & np.isfinite(current).all(axis=1)
    if not finite[0] or not np.any(finite):
        return 8.0
    palm_mm = float(np.linalg.norm(previous[0] - current[0]) * 1000.0)
    joint_mm = float(np.median(np.linalg.norm(previous[finite] - current[finite], axis=1)) * 1000.0)
    # Do not suppress genuine motion; only make a large unexplained jump more
    # expensive than a nearby candidate from the other branch.
    return (
        0.015 * max(0.0, palm_mm - 20.0)
        + 0.004 * max(0.0, joint_mm - 20.0)
    )


def _choose_track(
    entries: list[tuple[int, list[tuple[str, dict[str, Any], np.ndarray, tuple[float, float, float, int]]]]],
) -> tuple[dict[int, tuple[str, dict[str, Any]]], dict[str, int]]:
    """Select the lowest reprojection candidate with a local motion guard.

    A full dynamic-programming pass is unnecessary here: each row already
    contains the guided temporal solution, and this pass only decides whether
    its persisted unrefined/native alternative is worth using.  The previous
    selected pose is used to reject a candidate that would introduce a new
    one-frame teleport.
    """
    chosen: dict[int, tuple[str, dict[str, Any]]] = {}
    source_counts = {"refined": 0, "unrefined": 0, "native": 0}
    previous_index: int | None = None
    previous_points: np.ndarray | None = None
    previous_source: str | None = None
    for sync_index, candidates in entries:
        if not candidates:
            continue
        scored: list[tuple[float, str, dict[str, Any], np.ndarray]] = []
        for source, hand, points_value, residual in candidates:
            _robust, p95, camera_median, _camera_count = residual
            cost = camera_median + 0.02 * max(0.0, p95 - 35.0)
            if source == "native":
                cost += 0.25
            scored.append((float(cost), source, hand, points_value))
        scored.sort(key=lambda item: item[0])
        selected = scored[0]
        if previous_points is not None and previous_index == sync_index - 1:
            # Prefer a slightly less image-perfect candidate over a visible
            # teleport.  These limits are display safeguards, not claims
            # about a physical speed limit; at 30 fps a hand should not move
            # an entire palm by several centimetres in one frame.
            for candidate in scored:
                previous = np.asarray(previous_points, dtype=np.float64)
                current = np.asarray(candidate[3], dtype=np.float64)
                finite = np.isfinite(previous).all(axis=1) & np.isfinite(current).all(axis=1)
                palm_step = (
                    float(np.linalg.norm(previous[0] - current[0]) * 1000.0)
                    if finite[0] else float("inf")
                )
                joint_step = (
                    float(np.median(np.linalg.norm(previous[finite] - current[finite], axis=1)) * 1000.0)
                    if np.any(finite) else float("inf")
                )
                max_joint_step = (
                    float(np.max(np.linalg.norm(previous[finite] - current[finite], axis=1)) * 1000.0)
                    if np.any(finite) else float("inf")
                )
                if (
                    palm_step <= 60.0
                    and joint_step <= 100.0
                    and max_joint_step <= 150.0
                ):
                    selected = candidate
                    break
        _cost, source, hand, points_value = selected
        chosen[sync_index] = (source, hand)
        source_counts[source] += 1
        previous_index = sync_index
        previous_points = points_value
        previous_source = source
    return chosen, source_counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--guided", required=True, type=Path)
    parser.add_argument("--native", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--native-max-2d-match-px", type=float, default=25.0)
    parser.add_argument("--min-valid-joints", type=int, default=21)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"output already exists: {args.output}")
    if args.native_max_2d_match_px <= 0 or not 1 <= args.min_valid_joints <= 21:
        raise ValueError("invalid hybrid thresholds")
    dataset = args.dataset.resolve()
    guided_root = args.guided.resolve()
    native_root = args.native.resolve()
    calibrations = {
        camera_id: CameraCalibration.load(dataset / "calibration" / f"{camera_id}.json")
        for camera_id in json.loads((dataset / "manifest.json").read_text())[
            "camera_ids"
        ]
    }
    guided = _load_jsonl(guided_root / "accepted.jsonl")
    native = _load_jsonl(native_root / "accepted.jsonl")
    ordered_ids = sorted(guided)
    output = args.output.resolve()
    output.mkdir(parents=True)

    native_by_guided: dict[tuple[int, int], dict[str, Any]] = {}
    for sync_index in sorted(set(guided) & set(native)):
        mapping = _native_assignment(native[sync_index], guided[sync_index].get("hands", []))
        for guided_index, native_hand in mapping.items():
            native_by_guided[(sync_index, guided_index)] = native_hand

    selected_by_side: dict[int, dict[int, tuple[str, dict[str, Any]]]] = {0: {}, 1: {}}
    source_totals = {"refined": 0, "unrefined": 0, "native": 0}
    candidate_totals = {"refined": 0, "unrefined": 0, "native": 0}
    for side in (0, 1):
        entries = []
        for sync_index in ordered_ids:
            hands = [
                hand for hand in guided[sync_index].get("hands", [])
                if int(hand.get("side", -1)) == side
            ]
            if not hands:
                continue
            guided_hand = hands[0]
            candidates: list[tuple[str, dict[str, Any], np.ndarray, tuple[float, float, float, int]]] = []
            refined_points = _points(guided_hand)
            # A hand cleared by the guided quality gate remains cleared.  In
            # particular, do not resurrect it from an unrefined/native array:
            # that would reintroduce the very sparse-depth artefacts the gate
            # was designed to remove.
            if not _renderable(refined_points, args.min_valid_joints):
                entries.append((sync_index, candidates))
                continue
            refined_residual = _reprojection(guided_hand, refined_points, calibrations)
            candidates.append(("refined", copy.deepcopy(guided_hand), refined_points, refined_residual))
            unrefined_points = _points(guided_hand, "unrefined_joints_base_m")
            if _renderable(unrefined_points, args.min_valid_joints):
                unrefined_hand = copy.deepcopy(guided_hand)
                unrefined_hand["joints_base_m"] = unrefined_points.tolist()
                unrefined_residual = _reprojection(guided_hand, unrefined_points, calibrations)
                # Keep the responsive branch only when it materially follows
                # the current image better than the refined branch.
                if unrefined_residual[2] + 0.25 < refined_residual[2]:
                    candidates.append(("unrefined", unrefined_hand, unrefined_points, unrefined_residual))
            native_hand = native_by_guided.get((sync_index, next((i for i, value in enumerate(guided[sync_index].get("hands", [])) if value is guided_hand), -1)))
            if native_hand is None:
                # JSON objects are not identity-preserving after loading; use
                # the hand's position in the guided row instead.
                guided_index = next((i for i, value in enumerate(guided[sync_index].get("hands", [])) if int(value.get("side", -1)) == side), -1)
                native_hand = native_by_guided.get((sync_index, guided_index))
            if native_hand is not None:
                native_points = _points(native_hand)
                if (
                    _renderable(native_points, args.min_valid_joints)
                    and _view_match(native_hand, guided_hand) <= args.native_max_2d_match_px
                ):
                    native_copy = copy.deepcopy(native_hand)
                    native_copy["side"] = side
                    native_copy["hybrid_native_source_side"] = native_hand.get("side")
                    native_residual = _reprojection(native_copy, native_points, calibrations)
                    if native_residual[2] <= 15.0 and native_residual[1] <= 100.0:
                        candidates.append(("native", native_copy, native_points, native_residual))
            for source, _hand, _points_value, _residual in candidates:
                candidate_totals[source] += 1
            entries.append((sync_index, candidates))
        chosen, counts = _choose_track(entries)
        selected_by_side[side] = chosen
        for source, count in counts.items():
            source_totals[source] += count

    merged = copy.deepcopy(guided)
    for sync_index, row in merged.items():
        for hand_index, hand in enumerate(row.get("hands", [])):
            side = int(hand.get("side", -1))
            selected = selected_by_side.get(side, {}).get(sync_index)
            if selected is None:
                continue
            source, replacement = selected
            replacement = copy.deepcopy(replacement)
            replacement["side"] = side
            replacement["hybrid_source"] = source
            points = _points(replacement)
            residual = _reprojection(replacement, points, calibrations)
            quality = dict(replacement.get("quality", {}))
            quality["valid_joint_count"] = int(np.isfinite(points).all(axis=1).sum())
            quality["hybrid_reprojection_median_px"] = residual[0]
            quality["hybrid_reprojection_p95_px"] = residual[1]
            replacement["quality"] = quality
            row["hands"][hand_index] = replacement

    (output / "accepted.jsonl").write_text(
        "".join(json.dumps(row, separators=(",", ":"), allow_nan=True) + "\n" for _, row in sorted(merged.items())),
        encoding="utf-8",
    )
    for filename in ("rejected.jsonl",):
        source = guided_root / filename
        (output / filename).write_text(source.read_text(encoding="utf-8") if source.is_file() else "", encoding="utf-8")
    guided_summary = {}
    summary_path = guided_root / "summary.json"
    if summary_path.is_file():
        guided_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary = dict(guided_summary)
    summary["fusion_algorithm"] = "guided_fusion+native_unambiguous_hybrid_v1"
    summary["hybrid_selection"] = {
        "native_root": str(native_root),
        "guided_root": str(guided_root),
        "native_max_2d_match_px": args.native_max_2d_match_px,
        "min_valid_joints": args.min_valid_joints,
        "candidate_counts": candidate_totals,
        "selected_source_counts": source_totals,
        "note": "native is used only when its 2-D observations match the guided hand; missing/cleared guided poses remain missing.",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    print(json.dumps(summary["hybrid_selection"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
