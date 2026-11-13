#!/usr/bin/env python3
"""Reproject one fitted MANO skeleton and mesh into calibrated Orbbec views."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time

import cv2
import numpy as np
import torch

from ego_data.calibration import CameraCalibration
from camera_models.kb import KB_PROJECTION_VERSION
from ego_data.dataset import SequentialVideoReader
from shared_hand_geometry import project_base
from mesh_rendering import MESH_STYLE
from shared_mesh_overlay import draw_shared_mesh
from video_utils import _create_video_writer, HAND_CONNECTIONS


SKELETON_RENDER_STYLE = "valid_projection_segment_clipping_v2"


def clip_projected_bone(first, second, width, height):
    """Clip a finite segment in floating point before conversion to OpenCV ints."""
    first, second = (
        np.asarray(first, dtype=np.float64),
        np.asarray(second, dtype=np.float64),
    )
    if not (np.isfinite(first).all() and np.isfinite(second).all()):
        return None
    delta = second - first
    enter, leave = 0.0, 1.0
    for axis, upper in enumerate((width - 1.0, height - 1.0)):
        if delta[axis] == 0:
            if not 0 <= first[axis] <= upper:
                return None
            continue
        crossings = sorted(
            (-first[axis] / delta[axis], (upper - first[axis]) / delta[axis])
        )
        enter, leave = max(enter, crossings[0]), min(leave, crossings[1])
        if enter > leave:
            return None
    clipped = np.stack((first + enter * delta, first + leave * delta))
    clipped = np.rint(np.clip(clipped, [0, 0], [width - 1, height - 1])).astype(
        np.int32
    )
    return tuple(map(int, clipped[0])), tuple(map(int, clipped[1]))


def draw_shared_skeleton(image, pixels, valid, side):
    """Keep valid offscreen joints for lines; mark joints only at real image positions."""
    pixels = np.asarray(pixels)
    valid = np.asarray(valid, dtype=bool) & np.isfinite(pixels).all(axis=1)
    height, width = image.shape[:2]
    inside = (
        valid
        & (pixels[:, 0] >= 0)
        & (pixels[:, 0] < width)
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < height)
    )
    color = (255, 255, 0) if side == 0 else (0, 255, 255)
    drawn, clipped = 0, 0
    for first, second in HAND_CONNECTIONS:
        # Points outside the camera model's valid domain cannot define an
        # edge. Being outside the image alone does not invalidate a point.
        if not (valid[first] and valid[second]):
            continue
        segment = clip_projected_bone(pixels[first], pixels[second], width, height)
        if segment is None:
            continue
        cv2.line(image, *segment, (20, 20, 20), 8, cv2.LINE_AA)
        cv2.line(image, *segment, color, 4, cv2.LINE_AA)
        drawn += 1
        clipped += int(not (inside[first] and inside[second]))
    for point in np.rint(pixels[inside]).astype(np.int32):
        cv2.circle(image, tuple(point), 6, (20, 20, 20), -1, cv2.LINE_AA)
        cv2.circle(image, tuple(point), 4, color, -1, cv2.LINE_AA)
    return {
        "visible_joint_count": int(inside.sum()),
        "drawn_bone_count": drawn,
        "clipped_bone_count": clipped,
    }


def shared_joint_points(hand):
    """Read the 21 MANO joints produced together with the mesh."""
    points = np.asarray(hand["joints_base_m"])
    if points.shape != (21, 3):
        raise ValueError("expected 21 shared MANO joints")
    return points


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset", "shared", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--cameras", nargs="+")
    p.add_argument(
        "--detector-predictions",
        type=Path,
        required=True,
        help="original per-camera detector outputs in the native WiLoR predictions (required for shared MANO)",
    )
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--tile-width", type=int, default=960)
    p.add_argument(
        "--video-codec", choices=("auto", "h264_nvenc", "mp4v"), default="auto"
    )
    args = p.parse_args()
    if args.max_frames < 0 or args.tile_width < 32 or args.tile_width % 2:
        p.error("invalid frame count or tile width")
    dataset = args.dataset.resolve()
    manifest = json.loads((dataset / "manifest.json").read_text())
    fit_summary = json.loads((args.shared / "summary.json").read_text())
    cameras = list(dict.fromkeys(args.cameras or fit_summary["camera_ids"]))
    if not cameras or not set(cameras) <= set(fit_summary["camera_ids"]):
        p.error("requested cameras were not used in shared fitting")
    calibrations = {
        c: CameraCalibration.load(dataset / "calibration" / f"{c}.json")
        for c in cameras
    }
    data = torch.load(
        args.shared / "shared_pose.pt",
        weights_only=False,
        map_location="cpu",
        mmap=True,
    )
    model_format = data["format"]
    if model_format != "shared_wilor_mano_v1":
        raise ValueError("expected shared WiLoR MANO geometry")
    model_name = "MANO"
    frames = {int(frame["sync_index"]): frame for frame in data["frames"]}
    with (dataset / "multiview_frames.csv").open() as stream:
        rows = list(csv.DictReader(stream))[: args.max_frames or None]
    if not rows or any(int(row["sync_index"]) not in frames for row in rows):
        raise ValueError(
            "shared results do not cover the requested synchronized frames"
        )
    from raw_detector_overlay import RawDetectorOverlay

    detector_overlay = RawDetectorOverlay(
        args.detector_predictions, dataset, cameras, rows, manifest["image_size"]
    )
    output = args.output.resolve()
    config = {
        "stage": f"shared_{model_name.lower()}_reprojection",
        "shared": str(args.shared.resolve()),
        "shared_mtime_ns": (args.shared / "shared_pose.pt").stat().st_mtime_ns,
        "dataset": str(dataset),
        "cameras": cameras,
        "frame_count": len(rows),
        "tile_width": args.tile_width,
        "mesh_style": MESH_STYLE,
        "video_codec": args.video_codec,
        "skeleton_render_style": SKELETON_RENDER_STYLE,
        "camera_projection_style": KB_PROJECTION_VERSION,
    }
    config["detector_overlay"] = detector_overlay.config
    meta = output.with_suffix(".json")
    if (
        output.is_file()
        and meta.is_file()
        and json.loads(meta.read_text()).get("config") == config
    ):
        print(f"[shared video] reuse {output}", flush=True)
        return
    width, height = manifest["image_size"]
    tile_height = round(args.tile_width * height / width)
    tile_height -= tile_height % 2
    columns = (
        2
        if len(cameras) == 4
        else 3
        if len(cameras) == 6
        else int(np.ceil(np.sqrt(len(cameras))))
    )
    grid_rows = int(np.ceil(len(cameras) / columns))
    timestamps = np.asarray([int(r["reference_timestamp_ns"]) for r in rows])
    dt = np.diff(timestamps) / 1e9
    fps = float(1 / np.median(dt[dt > 0])) if np.any(dt > 0) else float(manifest["fps"])
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.stem + ".partial.mp4")
    writer, codec = _create_video_writer(
        temporary,
        fps,
        (columns * args.tile_width, grid_rows * tile_height),
        args.video_codec,
    )
    readers = {
        c: SequentialVideoReader(
            dataset / "cameras" / c / manifest["storage"]["video_filename"],
            (width, height),
        )
        for c in cameras
    }
    projected_hands = 0
    mesh_draws = 0
    clipped_bones = 0
    started = time.perf_counter()
    projections_path = output.with_suffix(".projections.jsonl")
    projections_temporary = output.with_suffix(".partial.projections.jsonl")
    try:
        with projections_temporary.open("w") as stream:
            for ordinal, row in enumerate(rows):
                sync = int(row["sync_index"])
                hands = frames[sync]["hands"]
                tiles, audit = [], {"sync_index": sync, "cameras": {}}
                for camera in cameras:
                    image = readers[camera].read(int(row[f"{camera}_frame_index"]))
                    audits = []
                    for hand in hands:
                        side = int(hand["side"])
                        # The same immutable world-space arrays enter every
                        # view. No native pose, per-view adjustment or fallback.
                        points = shared_joint_points(hand)
                        pixels, valid, _ = project_base(calibrations[camera], points)
                        mesh_drawn, mesh_fingerprint = draw_shared_mesh(
                            image, hand, data["topology"], calibrations[camera]
                        )
                        mesh_draws += int(mesh_drawn)
                        skeleton = draw_shared_skeleton(image, pixels, valid, side)
                        projected_hands += int(
                            skeleton["visible_joint_count"] > 0
                            or skeleton["drawn_bone_count"] > 0
                        )
                        clipped_bones += skeleton["clipped_bone_count"]
                        audits.append(
                            {
                                "side": side,
                                "shared_model_id": f"{sync}:{side}",
                                "joints_2d": [
                                    xy.tolist() if ok else None
                                    for xy, ok in zip(pixels, valid)
                                ],
                                **skeleton,
                                "supporting_cameras": hand["supporting_cameras"],
                                "mesh_projection_sha256": mesh_fingerprint,
                                "mesh_drawn": bool(mesh_drawn),
                            }
                        )
                    audit["cameras"][camera] = audits
                    # Detector visualization is independent of pose coverage:
                    # also draw boxes for views/hands rejected by shared fitting.
                    audit.setdefault("detector_boxes", {})[camera] = (
                        detector_overlay.draw(image, camera, sync)
                    )
                    state = (
                        f"SHARED {model_name} 3D | hands {len(hands)}"
                        if hands
                        else "NO VALID SHARED 3D"
                    )
                    cv2.putText(
                        image,
                        f"{camera} | {state} | frame {sync}",
                        (8, 24),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (80, 230, 80) if hands else (0, 190, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    tiles.append(cv2.resize(image, (args.tile_width, tile_height)))
                while len(tiles) < columns * grid_rows:
                    tiles.append(np.zeros((tile_height, args.tile_width, 3), np.uint8))
                writer.write(
                    np.vstack(
                        [
                            np.hstack(tiles[n : n + columns])
                            for n in range(0, len(tiles), columns)
                        ]
                    )
                )
                stream.write(json.dumps(audit, allow_nan=False) + "\n")
                if (ordinal + 1) % 100 == 0 or ordinal + 1 == len(rows):
                    print(
                        f"[shared video] {ordinal + 1}/{len(rows)} | {(ordinal + 1) / (time.perf_counter() - started):.1f} fps",
                        flush=True,
                    )
    finally:
        writer.release()
        for reader in readers.values():
            reader.close()
    temporary.replace(output)
    projections_temporary.replace(projections_path)
    summary = {
        "config": config,
        "frame_count": len(rows),
        "fps": fps,
        "codec": codec,
        "video_size": [columns * args.tile_width, grid_rows * tile_height],
        "projected_hand_tiles": projected_hands,
        "mesh_draws": mesh_draws,
        "clipped_bone_count": clipped_bones,
        "skeleton_and_mesh_share_model": True,
        "independent_view_fallback": False,
        "raw_images_without_undistortion": "undistortion" not in manifest,
        "projections": str(projections_path),
        "output": str(output),
        "render_seconds": time.perf_counter() - started,
    }
    summary["detector_boxes"] = {
        "source": "original_hand_detector",
        "independent_of_shared_fit": True,
        "drawn": detector_overlay.drawn,
        "per_camera": detector_overlay.counts,
    }
    meta.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"[shared video] saved {output}", flush=True)


if __name__ == "__main__":
    main()
