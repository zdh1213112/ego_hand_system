#!/usr/bin/env python3
"""Render a safe fused-3D/native-2D multiview delivery video.

The fused JSONL remains the source of 3-D measurements.  A hand is rendered
as 3-D only when every visible joint has enough independent views and the
projection agrees with the native WiLoR 2-D observation.  Rejected frames,
weak-depth hands, and 2-D outliers use the tracked native 2-D pose instead of
an invented/interpolated 3-D skeleton.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time
from typing import Any

import cv2
import numpy as np

from ego_data.calibration import CameraCalibration
from ego_data.dataset import SequentialVideoReader
from multiview_original_projection import OriginalVideoProjection
from render_multiview_wilor import (
    _create_video_writer,
    _draw_final_2d_fallback,
    _draw_final_detection,
    _draw_refined_hand,
    _has_renderable_final_pose,
    _has_renderable_joint_support,
    _project_refined_hand,
)
from render_native_2d_multiview import _select_camera_tracks, _smooth_track


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument(
        "--original-dataset", type=Path,
        help="display on the corresponding raw normalized videos; keep tracking/trust in --dataset coordinates",
    )
    parser.add_argument("--fusion", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument(
        "--gem-mesh-predictions", type=Path,
        help="also overlay the selected per-camera SOMA mesh from camera*/gem_pose.pt",
    )
    parser.add_argument(
        "--shared-mano-mesh", type=Path,
        help="overlay the same shared MANO mesh as the shared video; keep the legacy skeleton unchanged",
    )
    parser.add_argument('--wilor-native-mesh', action='store_true', help='restore each camera original WiLoR mesh from saved MANO parameters')
    parser.add_argument('--wilor-native-predictions', type=Path, help='original WiLoR predictions if skeleton predictions have been assisted')
    parser.add_argument('--native-mano-model-dir', type=Path, default=Path(__file__).resolve().parents[1] / 'models/mano')
    parser.add_argument('--native-model-config', type=Path, default=Path(__file__).resolve().parents[1] / 'models/wilor/model_config.yaml')
    parser.add_argument('--native-mesh-device', default='cuda')
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cameras", nargs="+", default=None)
    parser.add_argument("--columns", type=int, default=0)
    parser.add_argument("--tile-width", type=int, default=480)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument(
        "--physical-side-swap", type=int, choices=(0, 1), default=0,
        help=(
            "swap native WiLoR side labels before physical L/R coloring; "
            "default 0 keeps WiLoR is_right=0/1 as left/right"
        ),
    )
    parser.add_argument("--min-joint-inlier-views", type=int, default=4)
    parser.add_argument("--residual-percentile", type=float, default=75.0)
    parser.add_argument("--residual-threshold-px", type=float, default=15.0)
    parser.add_argument("--wrist-threshold-px", type=float, default=18.0)
    parser.add_argument(
        "--max-joint-residual-px", type=float, default=25.0,
        help="fall back to native 2-D if even one projected joint is this far from WiLoR",
    )
    parser.add_argument(
        "--min-fused-run-frames", type=int, default=2,
        help="minimum consecutive trusted frames before displaying fused 3-D",
    )
    parser.add_argument("--max-condition-number", type=float, default=120.0)
    parser.add_argument(
        "--max-stereo-difference-mm", type=float, default=20.0,
        help=(
            "maximum median 3-D difference between the two anchor views; "
            "larger values indicate an ill-conditioned depth branch and "
            "fall back to native 2-D"
        ),
    )
    parser.add_argument("--smoothing-radius", type=int, default=1)
    parser.add_argument("--smoothing-strength", type=float, default=0.25)
    parser.add_argument("--wrist-outlier-threshold-px", type=float, default=150.0)
    parser.add_argument("--max-hold-frames", type=int, default=2)
    parser.add_argument("--video-codec", choices=("auto", "h264_nvenc", "mp4v"), default="auto")
    return parser.parse_args()


def _load_jsonl(path: Path) -> dict[int, dict[str, Any]]:
    return {
        int(item["sync_index"]): item
        for item in map(json.loads, path.read_text(encoding="utf-8").splitlines())
        if item
    }


def _trusted_3d(
    hand: dict[str, Any], camera: str, calibration: CameraCalibration,
    min_views: int, residual_percentile: float, residual_threshold: float,
    wrist_threshold: float, max_condition: float,
    max_stereo_difference_mm: float,
    max_joint_residual_px: float = 25.0,
) -> tuple[bool, np.ndarray]:
    points = np.asarray(hand.get("joints_base_m", []), dtype=np.float64)
    if not _has_renderable_final_pose(points, min_joints=21):
        return False, np.full((21, 2), np.nan)
    if not _has_renderable_joint_support(hand, min_views):
        return False, np.full((21, 2), np.nan)
    stereo = hand.get("stereo_baseline_comparison", {})
    stereo_difference = float(
        stereo.get("multiview_3d_difference_median_mm", float("nan"))
    )
    # Reprojection can remain small even when depth is almost unobservable:
    # the same 2-D error is explained by many different depths.  The
    # independent anchor-view comparison is therefore a separate hard gate,
    # rather than another soft score.
    if (
        not np.isfinite(stereo_difference)
        or stereo_difference > float(max_stereo_difference_mm)
    ):
        return False, np.full((21, 2), np.nan)
    conditions = np.asarray(hand.get("quality", {}).get("ray_condition_numbers", []), dtype=np.float64)
    finite_points = np.isfinite(points).all(axis=1)
    if conditions.shape == (21,) and np.any(
        finite_points & (~np.isfinite(conditions) | (conditions > float(max_condition)))
    ):
        return False, np.full((21, 2), np.nan)
    projected = _project_refined_hand(calibration, points)
    view = hand.get("views", {}).get(camera)
    if view is None:
        return False, projected
    observed = np.asarray(view.get("wilor_joints_2d", view.get("joints_2d", [])), dtype=np.float64)
    if observed.shape != (21, 2):
        return False, projected
    valid = np.isfinite(projected).all(axis=1) & np.isfinite(observed).all(axis=1)
    if int(valid.sum()) < 18:
        return False, projected
    residuals = np.linalg.norm(projected[valid] - observed[valid], axis=1)
    wrist = float(np.linalg.norm(projected[0] - observed[0])) if valid[0] else float("inf")
    return bool(
        float(np.percentile(residuals, float(residual_percentile))) <= float(residual_threshold)
        and wrist <= float(wrist_threshold)
        and float(np.max(residuals)) <= float(max_joint_residual_px)
    ), projected


def _drop_short_trusted_runs(mask: list[bool], min_run: int) -> list[bool]:
    if min_run <= 1:
        return list(mask)
    output = list(mask)
    start = 0
    while start < len(output):
        end = start + 1
        while end < len(output) and output[end] == output[start]:
            end += 1
        if output[start] and end - start < min_run:
            output[start:end] = [False] * (end - start)
        start = end
    return output


def main() -> int:
    args = parse_args()
    if args.max_frames < 0 or args.tile_width < 32 or args.columns < 0:
        raise ValueError("invalid frame/tile/column argument")
    if args.min_joint_inlier_views < 1 or args.residual_percentile < 0 or args.residual_percentile > 100:
        raise ValueError("invalid 3-D trust thresholds")
    if args.max_stereo_difference_mm <= 0:
        raise ValueError("--max-stereo-difference-mm must be positive")
    if args.max_joint_residual_px <= 0 or args.min_fused_run_frames < 1:
        raise ValueError("joint residual must be positive and minimum fused run at least one")
    if args.smoothing_radius < 0 or args.max_hold_frames < 0:
        raise ValueError("smoothing radius and hold frames must be non-negative")
    if args.gem_mesh_predictions is not None and args.original_dataset is not None:
        raise ValueError("GEM mesh overlays currently require the unchanged inference images")
    if args.shared_mano_mesh is not None and (args.gem_mesh_predictions is not None or args.original_dataset is not None):
        raise ValueError("shared MANO mesh requires its fitting dataset and cannot be combined with another mesh overlay")
    if args.wilor_native_mesh and any(value is not None for value in (args.shared_mano_mesh, args.gem_mesh_predictions, args.original_dataset)):
        raise ValueError('native WiLoR mesh cannot be combined with another mesh or another image coordinate system')
    dataset = args.dataset.resolve()
    fusion = args.fusion.resolve()
    predictions = args.predictions.resolve()
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    available = tuple(manifest["camera_ids"])
    cameras = tuple(dict.fromkeys(args.cameras or available))
    if sorted(set(cameras) - set(available)):
        raise ValueError("selected cameras are not present in the dataset")
    original_projection = (
        OriginalVideoProjection(dataset, args.original_dataset, cameras)
        if args.original_dataset is not None else None
    )
    display_dataset = original_projection.root if original_projection else dataset
    display_manifest = original_projection.manifest if original_projection else manifest
    with (dataset / "multiview_frames.csv").open("r", encoding="utf-8", newline="") as stream:
        frame_rows = list(csv.DictReader(stream))
    frame_rows = frame_rows[: args.max_frames or None]
    frame_ids = [int(row["sync_index"]) for row in frame_rows]
    shared_mesh = None
    if args.shared_mano_mesh is not None:
        from shared_mesh_overlay import SharedManoMeshOverlay
        shared_mesh = SharedManoMeshOverlay(args.shared_mano_mesh, dataset, cameras, frame_ids)
    accepted = _load_jsonl(fusion / "accepted.jsonl")
    prediction_rows = {
        camera: _load_jsonl(predictions / camera / "predictions.jsonl")
        for camera in cameras
    }
    mesh_overlay = None
    if args.gem_mesh_predictions is not None:
        from render_gem_multiview import GemMeshOverlay, MESH_STYLE
        mesh_overlay = GemMeshOverlay(args.gem_mesh_predictions, cameras)
    elif args.wilor_native_mesh:
        from wilor_native_mesh_overlay import WilorNativeMeshOverlay
        native_root = (args.wilor_native_predictions or predictions).resolve()
        native_rows = prediction_rows if native_root == predictions else {
            camera: _load_jsonl(native_root / camera / 'predictions.jsonl') for camera in cameras}
        for row in frame_rows:
            for camera in cameras:
                native = native_rows[camera][int(row['sync_index'])]
                if (int(native['source_frame_index']) != int(row[f'{camera}_frame_index'])
                        or int(native['timestamp_ns']) != int(row[f'{camera}_timestamp_ns'])):
                    raise ValueError('native mesh frame/timestamp differs from displayed video')
        mesh_overlay = WilorNativeMeshOverlay(native_root, native_rows, manifest['image_size'],
            args.native_mano_model_dir, args.native_model_config, args.native_mesh_device)
    if mesh_overlay:
        for records in prediction_rows.values():
            for record in records.values():
                for hand in record.get("hands", []):
                    hand["mesh_source_frame_index"] = int(record["source_frame_index"])
    native_tracks: dict[tuple[str, int], list[dict[str, Any] | None]] = {}
    for camera in cameras:
        selected = _select_camera_tracks(
            prediction_rows[camera], frame_ids,
            args.max_hold_frames, args.wrist_outlier_threshold_px,
        )
        for side in (0, 1):
            native_tracks[(camera, side)] = _smooth_track(
                selected[side], args.smoothing_radius, args.smoothing_strength,
                25.0, args.wrist_outlier_threshold_px,
            )
    calibrations = {
        camera: CameraCalibration.load(dataset / "calibration" / f"{camera}.json")
        for camera in cameras
    }
    trusted_poses: dict[tuple[str, int], list[tuple[bool, np.ndarray]]] = {}
    for camera in cameras:
        for side in (0, 1):
            samples = []
            for sync_index in frame_ids:
                hand = next(
                    (item for item in accepted.get(sync_index, {}).get("hands", [])
                     if int(item.get("side", -1)) == side), None,
                )
                if hand is None:
                    samples.append((False, np.full((21, 2), np.nan)))
                    continue
                samples.append(_trusted_3d(
                    hand, camera, calibrations[camera],
                    args.min_joint_inlier_views, args.residual_percentile,
                    args.residual_threshold_px, args.wrist_threshold_px,
                    args.max_condition_number, args.max_stereo_difference_mm,
                    args.max_joint_residual_px,
                ))
            stable = _drop_short_trusted_runs(
                [ok for ok, _ in samples], args.min_fused_run_frames,
            )
            trusted_poses[camera, side] = [
                (stable[index], points) for index, (_, points) in enumerate(samples)
            ]
    if original_projection:
        # All decisions above were made with the inference calibration/pixels.
        # Convert only the selected display tracks and trusted 3D projections.
        for camera in cameras:
            for side in (0, 1):
                native_tracks[camera, side] = [
                    original_projection.native_view(camera, item)
                    for item in native_tracks[camera, side]
                ]
                converted = []
                for index, (trusted, points) in enumerate(trusted_poses[camera, side]):
                    if trusted:
                        hand = next(item for item in accepted[frame_ids[index]]["hands"]
                                    if int(item.get("side", -1)) == side)
                        points = original_projection.fused_points(camera, hand["joints_base_m"], points)
                        trusted = int(np.isfinite(points).all(axis=1).sum()) >= 5
                    converted.append((trusted, points))
                trusted_poses[camera, side] = converted
    if args.columns:
        columns = int(args.columns)
    elif len(cameras) == 6:
        columns = 3
    elif len(cameras) == 4:
        columns = 2
    else:
        columns = max(1, int(np.ceil(np.sqrt(len(cameras)))))
    row_count = int(np.ceil(len(cameras) / columns))
    image_size = tuple(display_manifest["image_size"])
    tile_height = int(round(args.tile_width * image_size[1] / image_size[0]))
    frame_size = (args.tile_width * columns, tile_height * row_count)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    # Use the synchronized reference timestamps when available.
    timestamps = np.asarray([int(row["reference_timestamp_ns"]) for row in frame_rows], dtype=np.int64)
    delta = np.diff(timestamps).astype(np.float64) / 1e9
    fps = float(1.0 / np.median(delta[delta > 0])) if np.any(delta > 0) else 30.0
    temporary = output.with_name(output.stem + ".partial.mp4")
    writer, codec = _create_video_writer(temporary, fps, frame_size, args.video_codec)
    readers = {
        camera: SequentialVideoReader(
            display_dataset / "cameras" / camera / display_manifest["storage"]["video_filename"], image_size,
        )
        for camera in cameras
    }
    fused_count = 0
    fused_hand_count = 0
    native_count = 0
    start = time.perf_counter()
    audit_path = output.with_suffix('.mesh_projections.jsonl')
    audit_stream = audit_path.open('w') if shared_mesh else None
    try:
        for index, row in enumerate(frame_rows):
            sync_index = frame_ids[index]
            result = accepted.get(sync_index)
            tiles = []
            mesh_audit = {'sync_index': sync_index, 'cameras': {}}
            for camera in cameras:
                image = readers[camera].read(int(row[f"{camera}_frame_index"]))
                if shared_mesh:
                    mesh_audit['cameras'][camera] = shared_mesh.draw(image, sync_index, calibrations[camera])
                used_fused = False
                used_native = False
                drawn_sides: set[int] = set()
                if result is not None:
                    for hand in result.get("hands", []):
                        side = int(hand.get("side", -1))
                        if side not in (0, 1):
                            continue
                        trusted, projected = trusted_poses[camera, side][index]
                        if trusted:
                            final_view = dict(hand.get("views", {}).get(camera, {}))
                            if original_projection:
                                final_view = original_projection.view(camera, final_view)
                            if mesh_overlay:
                                mesh_overlay.draw(image, camera, int(row[f"{camera}_frame_index"]), final_view, side)
                            _draw_final_detection(image, final_view, side, "refined")
                            _draw_refined_hand(image, projected, side, 0, final_only=True)
                            fused_hand_count += 1
                            used_fused = True
                            drawn_sides.add(side)
                        else:
                            native_side = 1 - side if args.physical_side_swap else side
                            item = native_tracks[(camera, native_side)][index]
                            if item is not None and side not in drawn_sides:
                                if mesh_overlay:
                                    mesh_overlay.draw(image, camera, int(row[f"{camera}_frame_index"]), item, side)
                                _draw_final_2d_fallback(image, item, side)
                                native_count += 1
                                used_native = True
                                drawn_sides.add(side)
                    # An accepted fusion row can contain only one hand (or
                    # have a side removed by a quality gate).  Fill the
                    # missing physical side from this camera's native track;
                    # otherwise the final video silently drops a visible
                    # hand whenever the other hand triangulates successfully.
                    for side in (0, 1):
                        if side in drawn_sides:
                            continue
                        native_side = 1 - side if args.physical_side_swap else side
                        item = native_tracks[(camera, native_side)][index]
                        if item is not None:
                            if mesh_overlay:
                                mesh_overlay.draw(image, camera, int(row[f"{camera}_frame_index"]), item, side)
                            _draw_final_2d_fallback(image, item, side)
                            native_count += 1
                            used_native = True
                            drawn_sides.add(side)
                else:
                    # A globally rejected frame has no fusion hand.  Draw the
                    # native tracks directly, preserving the physical palette.
                    for native_side in (0, 1):
                        item = native_tracks[(camera, native_side)][index]
                        if item is not None:
                            display_side = 1 - native_side if args.physical_side_swap else native_side
                            if mesh_overlay:
                                mesh_overlay.draw(image, camera, int(row[f"{camera}_frame_index"]), item, display_side)
                            _draw_final_2d_fallback(image, item, display_side)
                            native_count += 1
                            used_native = True
                fused_count += int(used_fused)
                state = (
                    "MIXED" if used_fused and used_native
                    else "FUSED3D" if used_fused
                    else "NATIVE2D"
                )
                if original_projection:
                    state = "RAW<-UNDISTORTED " + (state if used_fused or used_native else "NO_POSE")
                elif args.wilor_native_mesh:
                    state = 'WILOR LEGACY ' + state + ' | NATIVE MESH'
                elif mesh_overlay:
                    state = "GEM-X | " + (state if used_fused or used_native else "NO_POSE") + " | MESH"
                elif shared_mesh:
                    state = "LEGACY " + state + " | SHARED MANO MESH"
                cv2.putText(
                    image, f"{camera} | {state} | frame {sync_index}",
                    (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (70, 220, 70) if used_fused else (0, 190, 255), 2, cv2.LINE_AA,
                )
                tiles.append(cv2.resize(image, (args.tile_width, tile_height)))
            while len(tiles) < row_count * columns:
                tiles.append(np.zeros((tile_height, args.tile_width, 3), dtype=np.uint8))
            writer.write(np.vstack([
                np.hstack(tiles[offset:offset + columns])
                for offset in range(0, len(tiles), columns)
            ]))
            if audit_stream:
                audit_stream.write(json.dumps(mesh_audit, allow_nan=False) + '\n')
            if (index + 1) % 100 == 0 or index + 1 == len(frame_rows):
                elapsed = time.perf_counter() - start
                print(
                    f"hybrid rendered {index + 1}/{len(frame_rows)} | "
                    f"{(index + 1) / max(elapsed, 1e-9):.2f} frames/s", flush=True,
                )
    finally:
        for reader in readers.values():
            reader.close()
        writer.release()
        if audit_stream:
            audit_stream.close()
    temporary.replace(output)
    summary = {
        "schema_version": 1,
        "stage": "trusted_fused_3d_native_2d_fallback_multiview_review",
        "dataset": str(dataset), "fusion": str(fusion), "predictions": str(predictions),
        "cameras": list(cameras), "frame_count": len(frame_rows), "fps": fps,
        "physical_side_swap": bool(args.physical_side_swap),
        "trust": {
            "min_joint_inlier_views": int(args.min_joint_inlier_views),
            "residual_percentile": float(args.residual_percentile),
            "residual_threshold_px": float(args.residual_threshold_px),
            "wrist_threshold_px": float(args.wrist_threshold_px),
            "max_joint_residual_px": float(args.max_joint_residual_px),
            "min_fused_run_frames": int(args.min_fused_run_frames),
            "max_condition_number": float(args.max_condition_number),
            "max_stereo_difference_mm": float(args.max_stereo_difference_mm),
        },
        "fused_camera_tiles": int(fused_count),
        "fused_camera_hand_tiles": int(fused_hand_count),
        "native_2d_camera_hand_tiles": int(native_count),
        "codec": codec, "output": str(output),
    }
    if original_projection:
        summary["original_display"] = {
            "dataset": str(display_dataset),
            "image_size": list(image_size),
            "projection": "original KB/DS calibration; original optical axes",
            "tracking_and_trust_domain": "inference dataset pixels (before display conversion)",
            "native_2d": "pinhole rays reprojected through original calibration",
            "boxes": "axis-aligned envelope of sampled warped box perimeter",
            "out_of_view_joints": "hidden, including predictions outside the inference image",
        }
    if args.gem_mesh_predictions is not None:
        summary["mesh"] = {
            "model": "GEM-X SOMA", "predictions": str(args.gem_mesh_predictions.resolve()),
            "geometry": "per-camera reconstruction, not multiview-fused geometry",
            "projection": "same pinhole approximation as GEM inference; original images unchanged",
            "selection": "same detection and hypothesis as the displayed final skeleton",
            **MESH_STYLE, **mesh_overlay.stats,
        }
    if args.wilor_native_mesh:
        summary['mesh'] = mesh_overlay.summary()
        summary['skeleton_and_mesh_share_model'] = False
        summary['shared_multiview_mesh'] = False
        summary['skeleton_mode'] = 'original legacy hybrid with native 2D fallback'
        summary['raw_images_without_undistortion'] = 'undistortion' not in manifest
        summary['video_size'] = list(frame_size)
        summary['tile_width'] = args.tile_width
        native_audit = output.with_suffix('.native_mesh.jsonl')
        with native_audit.open('w') as stream:
            for entry in mesh_overlay.audits:
                stream.write(json.dumps(entry) + '\n')
        summary['native_mesh_audit'] = str(native_audit)
    if shared_mesh:
        summary['mesh'] = shared_mesh.summary()
        summary['mesh_projections'] = str(audit_path)
        summary['skeleton_and_mesh_share_model'] = False
        summary['skeleton_mode'] = 'legacy hybrid with native 2D fallback'
        summary['raw_images_without_undistortion'] = 'undistortion' not in manifest
        summary['tile_width'] = args.tile_width
    output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Hybrid multiview video: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
