#!/usr/bin/env python3
"""Refine saved multiview WiLoR predictions with visible glove markers."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time
from typing import Any

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from ego_data.dataset import SequentialVideoReader  # noqa: E402
from glove_marker_assist import (  # noqa: E402
    MarkerAssistConfig,
    assist_wilor_hypotheses,
)
from render_wilor_predictions import (  # noqa: E402
    HAND_CONNECTIONS, LEFT_COLOR, RIGHT_COLOR,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cameras", nargs="+")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--saturation-max", type=int, default=100)
    parser.add_argument("--value-min", type=int, default=160)
    parser.add_argument("--min-matches", type=int, default=3)
    parser.add_argument("--min-finger-groups", type=int, default=2)
    parser.add_argument("--global-min-matches", type=int, default=8)
    parser.add_argument("--global-min-finger-groups", type=int, default=4)
    parser.add_argument("--search-padding-px", type=float, default=45.0)
    parser.add_argument("--bbox-padding-px", type=float, default=12.0)
    parser.add_argument("--seed-distance-px", type=float, default=35.0)
    parser.add_argument("--match-distance-px", type=float, default=13.0)
    parser.add_argument("--max-shift-px", type=float, default=20.0)
    parser.add_argument("--max-applied-shift-px", type=float, default=10.0)
    parser.add_argument("--max-global-residual-median-px", type=float, default=7.5)
    parser.add_argument("--max-global-residual-p95-px", type=float, default=12.0)
    parser.add_argument("--marker-blend", type=float, default=0.15)
    parser.add_argument("--max-local-adjustment-px", type=float, default=3.0)
    parser.add_argument(
        "--preview-count", type=int, default=6,
        help="successful frames per camera included in marker_assist_preview.jpg",
    )
    parser.add_argument("--progress-interval", type=int, default=50)
    return parser.parse_args()


def _load_rows(path: Path, max_frames: int) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    return rows[: max_frames or None]


def _draw_skeleton(
    image: np.ndarray, points: np.ndarray, color: tuple[int, int, int], thickness: int,
) -> None:
    rounded = np.rint(points).astype(np.int32)
    for start, end in HAND_CONNECTIONS:
        cv2.line(
            image, tuple(rounded[start]), tuple(rounded[end]),
            color, thickness, cv2.LINE_AA,
        )
    for point in rounded:
        cv2.circle(image, tuple(point), max(2, thickness + 1), color, -1, cv2.LINE_AA)


def _preview_frame(
    frame: np.ndarray, hands: list[dict[str, Any]], camera: str, sync_index: int,
) -> np.ndarray | None:
    visible = [
        hand for hand in hands
        if hand.get("marker_assist", {}).get("applied")
        or hand.get("marker_assist", {}).get("evidence_only")
    ]
    if not visible:
        return None
    image = frame.copy()
    matched = 0
    corrected = 0
    evidence_only = 0
    for hand in visible:
        side = int(hand["is_right"])
        color = RIGHT_COLOR if side else LEFT_COLOR
        original = np.asarray(hand["wilor_joints_2d"], dtype=np.float64)
        adjusted = np.asarray(hand["joints_2d"], dtype=np.float64)
        _draw_skeleton(image, original, (0, 180, 255), 2)
        marker = hand["marker_assist"]
        if marker["applied"]:
            _draw_skeleton(image, adjusted, color, 4)
            corrected += 1
        else:
            evidence_only += 1
        centers = np.asarray(marker["matched_blob_centers"], dtype=np.float64)
        for center in np.rint(centers).astype(np.int32):
            cv2.circle(image, tuple(center), 8, (255, 0, 255), 2, cv2.LINE_AA)
            cv2.drawMarker(
                image, tuple(center), (255, 255, 255), cv2.MARKER_CROSS,
                11, 1, cv2.LINE_AA,
            )
        matched += int(marker["matched_marker_count"])
    cv2.rectangle(image, (0, 0), (image.shape[1], 78), (16, 16, 16), -1)
    cv2.putText(
        image,
        f"{camera} frame {sync_index} | corrected {corrected} | "
        f"evidence-only {evidence_only} | markers {matched}",
        (18, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (245, 245, 245), 2,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        "orange=raw | cyan/green=conservative correction | magenta=owned marker",
        (18, 61), cv2.FONT_HERSHEY_SIMPLEX, 0.57, (230, 230, 230), 1,
        cv2.LINE_AA,
    )
    return image


def _retain_preview(
    entries: list[tuple[float, int, np.ndarray]],
    score: float,
    sync_index: int,
    image: np.ndarray,
    count: int,
) -> None:
    if count <= 0:
        return
    thumbnail_width = 900
    thumbnail = cv2.resize(
        image,
        (
            thumbnail_width,
            int(round(thumbnail_width * image.shape[0] / image.shape[1])),
        ),
        interpolation=cv2.INTER_AREA,
    )
    entries.append((score, sync_index, thumbnail))
    entries.sort(key=lambda item: (-item[0], item[1]))
    del entries[count:]


def _write_contact_sheet(images: list[np.ndarray], path: Path) -> None:
    if not images:
        return
    tile_width = 600
    tiles = [
        cv2.resize(
            image,
            (tile_width, int(round(tile_width * image.shape[0] / image.shape[1]))),
            interpolation=cv2.INTER_AREA,
        )
        for image in images
    ]
    columns = 2 if len(tiles) > 1 else 1
    rows = []
    blank = np.zeros_like(tiles[0])
    for start in range(0, len(tiles), columns):
        values = tiles[start:start + columns]
        while len(values) < columns:
            values.append(blank.copy())
        rows.append(np.hstack(values))
    cv2.imwrite(str(path), np.vstack(rows))


def _camera_summary(
    camera: str,
    frame_count: int,
    total_hypotheses: int,
    applied_hypotheses: int,
    evidence_only_hypotheses: int,
    applied_frames: int,
    match_counts: list[int],
    residuals: list[float],
    raw_residuals: list[float],
    assisted_residuals: list[float],
    shifts: list[float],
    applied_shifts: list[float],
    bone_changes: list[float],
    failures: Counter[str],
    elapsed_seconds: float,
) -> dict[str, Any]:
    return {
        "camera": camera,
        "frame_count": frame_count,
        "hypothesis_count": total_hypotheses,
        "assisted_hypothesis_count": applied_hypotheses,
        "assisted_hypothesis_rate": applied_hypotheses / max(total_hypotheses, 1),
        "evidence_only_hypothesis_count": evidence_only_hypotheses,
        "marker_evidence_hypothesis_count": (
            applied_hypotheses + evidence_only_hypotheses
        ),
        "marker_evidence_hypothesis_rate": (
            applied_hypotheses + evidence_only_hypotheses
        ) / max(total_hypotheses, 1),
        "assisted_frame_count": applied_frames,
        "assisted_frame_rate": applied_frames / max(frame_count, 1),
        "matched_marker_count_median": (
            float(np.median(match_counts)) if match_counts else None
        ),
        "match_residual_median_px": (
            float(np.median(residuals)) if residuals else None
        ),
        "raw_marker_residual_median_px": (
            float(np.median(raw_residuals)) if raw_residuals else None
        ),
        "assisted_marker_residual_median_px": (
            float(np.median(assisted_residuals)) if assisted_residuals else None
        ),
        "coarse_shift_norm_median_px": (
            float(np.median(shifts)) if shifts else None
        ),
        "applied_shift_norm_median_px": (
            float(np.median(applied_shifts)) if applied_shifts else None
        ),
        "bone_length_change_p95_ratio_median": (
            float(np.median(bone_changes)) if bone_changes else None
        ),
        "outcome_reason_code_counts": dict(failures),
        "elapsed_seconds": elapsed_seconds,
        "frames_per_second": frame_count / max(elapsed_seconds, 1e-9),
    }


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"output already exists: {args.output}")
    if args.max_frames < 0 or args.preview_count < 0 or args.progress_interval < 1:
        raise ValueError("max-frames/preview-count must be non-negative; progress positive")
    config = MarkerAssistConfig(
        saturation_max=args.saturation_max,
        value_min=args.value_min,
        min_matches=args.min_matches,
        min_finger_groups=args.min_finger_groups,
        global_min_matches=args.global_min_matches,
        global_min_finger_groups=args.global_min_finger_groups,
        search_padding_px=args.search_padding_px,
        bbox_padding_px=args.bbox_padding_px,
        seed_distance_px=args.seed_distance_px,
        match_distance_px=args.match_distance_px,
        max_shift_px=args.max_shift_px,
        max_applied_shift_px=args.max_applied_shift_px,
        max_global_residual_median_px=args.max_global_residual_median_px,
        max_global_residual_p95_px=args.max_global_residual_p95_px,
        marker_blend=args.marker_blend,
        max_local_adjustment_px=args.max_local_adjustment_px,
    )
    config.validate()
    dataset = args.dataset.resolve()
    prediction_root = args.predictions.resolve()
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("dataset_type") != "normalized_multiview":
        raise ValueError(f"not a normalized multiview dataset: {dataset}")
    available_cameras = tuple(manifest["camera_ids"])
    cameras = tuple(dict.fromkeys(args.cameras or available_cameras))
    unknown = [camera for camera in cameras if camera not in available_cameras]
    if unknown:
        raise ValueError(f"selected cameras are not present in the dataset: {unknown}")
    output = args.output.resolve()
    output.mkdir(parents=True)
    image_size = tuple(manifest["image_size"])
    summaries: dict[str, dict[str, Any]] = {}
    started = time.perf_counter()
    for camera in cameras:
        camera_started = time.perf_counter()
        rows = _load_rows(
            prediction_root / camera / "predictions.jsonl", args.max_frames
        )
        camera_output = output / camera
        camera_output.mkdir()
        reader = SequentialVideoReader(
            dataset / "cameras" / camera / manifest["storage"]["video_filename"],
            image_size,
        )
        total_hypotheses = 0
        applied_hypotheses = 0
        evidence_only_hypotheses = 0
        applied_frames = 0
        match_counts: list[int] = []
        residuals: list[float] = []
        raw_residuals: list[float] = []
        assisted_residuals: list[float] = []
        shifts: list[float] = []
        applied_shifts: list[float] = []
        bone_changes: list[float] = []
        failures: Counter[str] = Counter()
        applied_previews: list[tuple[float, int, np.ndarray]] = []
        evidence_previews: list[tuple[float, int, np.ndarray]] = []
        try:
            with (camera_output / "predictions.jsonl").open(
                "w", encoding="utf-8"
            ) as stream:
                for ordinal, row in enumerate(rows, start=1):
                    frame = reader.read(int(row["source_frame_index"]))
                    assisted, _blobs = assist_wilor_hypotheses(
                        frame, row.get("hands", []), config
                    )
                    record = dict(row)
                    record["hands"] = assisted
                    stream.write(json.dumps(record, separators=(",", ":")) + "\n")
                    total_hypotheses += len(assisted)
                    frame_applied = False
                    for hand in assisted:
                        marker = hand["marker_assist"]
                        failures[str(marker.get("reason_code", "unknown"))] += 1
                        if marker["applied"]:
                            frame_applied = True
                            applied_hypotheses += 1
                            match_counts.append(int(marker["matched_marker_count"]))
                            residuals.append(float(marker["match_residual_median_px"]))
                            raw_residuals.append(
                                float(marker["raw_marker_residual_median_px"])
                            )
                            assisted_residuals.append(
                                float(marker["assisted_marker_residual_median_px"])
                            )
                            shifts.append(float(marker["shift_norm_px"]))
                            applied_shifts.append(
                                float(marker["applied_shift_norm_px"])
                            )
                            bone_changes.append(
                                float(marker["bone_length_change_p95_ratio"])
                            )
                        elif marker.get("evidence_only"):
                            evidence_only_hypotheses += 1
                            match_counts.append(int(marker["matched_marker_count"]))
                            residuals.append(float(marker["match_residual_median_px"]))
                            raw_residuals.append(
                                float(marker["raw_marker_residual_median_px"])
                            )
                            shifts.append(float(marker["shift_norm_px"]))
                    applied_frames += int(frame_applied)
                    evidence_frame = any(
                        hand["marker_assist"].get("evidence_only")
                        for hand in assisted
                    )
                    if frame_applied or evidence_frame:
                        sync_index = int(row["sync_index"])
                        preview = _preview_frame(frame, assisted, camera, sync_index)
                        if preview is not None and frame_applied:
                            risk = max(
                                float(hand["marker_assist"].get("shift_norm_px", 0.0))
                                + float(hand["marker_assist"].get(
                                    "match_residual_median_px", 0.0
                                ))
                                + 100.0 * float(hand["marker_assist"].get(
                                    "bone_length_change_p95_ratio", 0.0
                                ))
                                for hand in assisted
                                if hand["marker_assist"].get("applied")
                            )
                            _retain_preview(
                                applied_previews, risk, sync_index, preview,
                                args.preview_count,
                            )
                        if preview is not None and evidence_frame:
                            near_threshold = max(
                                float(hand["marker_assist"].get(
                                    "matched_marker_count", 0
                                ))
                                - 0.05 * float(hand["marker_assist"].get(
                                    "match_residual_median_px", 0.0
                                ))
                                for hand in assisted
                                if hand["marker_assist"].get("evidence_only")
                            )
                            _retain_preview(
                                evidence_previews, near_threshold, sync_index, preview,
                                args.preview_count,
                            )
                    if ordinal % args.progress_interval == 0 or ordinal == len(rows):
                        print(
                            f"{camera}: {ordinal}/{len(rows)} | "
                            f"assisted hypotheses {applied_hypotheses}/{total_hypotheses}",
                            flush=True,
                        )
        finally:
            reader.close()
        preview_path = camera_output / "marker_assist_preview.jpg"
        evidence_preview_path = camera_output / "marker_evidence_preview.jpg"
        _write_contact_sheet([entry[2] for entry in applied_previews], preview_path)
        _write_contact_sheet(
            [entry[2] for entry in evidence_previews], evidence_preview_path
        )
        elapsed = time.perf_counter() - camera_started
        summary = _camera_summary(
            camera, len(rows), total_hypotheses, applied_hypotheses,
            evidence_only_hypotheses,
            applied_frames, match_counts, residuals, raw_residuals,
            assisted_residuals, shifts, applied_shifts, bone_changes,
            failures, elapsed,
        )
        if preview_path.is_file():
            summary["preview"] = str(preview_path)
        if evidence_preview_path.is_file():
            summary["evidence_preview"] = str(evidence_preview_path)
        (camera_output / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        summaries[camera] = summary

    elapsed = time.perf_counter() - started
    total_hypotheses = sum(
        summary["hypothesis_count"] for summary in summaries.values()
    )
    assisted_hypotheses = sum(
        summary["assisted_hypothesis_count"] for summary in summaries.values()
    )
    evidence_only_hypotheses = sum(
        summary["evidence_only_hypothesis_count"] for summary in summaries.values()
    )
    summary = {
        "schema_version": 1,
        "stage": "wilor_glove_marker_image_assist",
        "dataset": str(dataset),
        "source_predictions": str(prediction_root),
        "camera_ids": list(cameras),
        "parameters": config.to_dict(),
        "hypothesis_count": total_hypotheses,
        "assisted_hypothesis_count": assisted_hypotheses,
        "assisted_hypothesis_rate": assisted_hypotheses / max(total_hypotheses, 1),
        "evidence_only_hypothesis_count": evidence_only_hypotheses,
        "marker_evidence_hypothesis_count": (
            assisted_hypotheses + evidence_only_hypotheses
        ),
        "marker_evidence_hypothesis_rate": (
            assisted_hypotheses + evidence_only_hypotheses
        ) / max(total_hypotheses, 1),
        "elapsed_seconds": elapsed,
        "cameras": summaries,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
