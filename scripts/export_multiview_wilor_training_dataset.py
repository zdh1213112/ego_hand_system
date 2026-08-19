#!/usr/bin/env python3
"""Export paired rectified images and 000865-compatible WiLoR training labels."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from ego_data.dataset import SequentialVideoReader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fusion", required=True, type=Path)
    parser.add_argument("--mano-fit", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--rectification", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cameras", nargs="+", default=["camera2", "camera3"])
    parser.add_argument("--min-camera-inlier-joints", type=int, default=12)
    parser.add_argument("--max-reprojection-median-px", type=float, default=25.0)
    parser.add_argument("--max-reprojection-p95-px", type=float, default=60.0)
    parser.add_argument("--bbox-margin", type=float, default=0.10)
    parser.add_argument("--min-visible-vertex-fraction", type=float, default=0.50)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--max-samples", type=int, default=0)
    return parser.parse_args()


def _load_fusion(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
    rows.sort(key=lambda row: int(row["sync_index"]))
    return rows


def _load_sync_rows(path: Path) -> dict[int, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return {int(row["sync_index"]): row for row in csv.DictReader(stream)}


def _load_track(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as archive:
        return {name: archive[name].copy() for name in archive.files}


def _project(vertices: np.ndarray, trans: np.ndarray, K: np.ndarray) -> np.ndarray:
    camera = vertices + trans[None]
    homogeneous = (K @ camera.T).T
    if not np.isfinite(homogeneous).all() or np.any(homogeneous[:, 2] <= 1e-8):
        raise ValueError("fitted mesh contains invalid or behind-camera vertices")
    return (homogeneous[:, :2] / homogeneous[:, 2:3]).astype(np.float32)


def _mesh_bbox(
    projected: np.ndarray, image_size: tuple[int, int], margin: float,
    min_visible_fraction: float,
) -> np.ndarray | None:
    width, height = image_size
    visible = (
        np.isfinite(projected).all(axis=1)
        & (projected[:, 0] >= 0.0) & (projected[:, 0] < width)
        & (projected[:, 1] >= 0.0) & (projected[:, 1] < height)
    )
    if float(visible.mean()) < min_visible_fraction:
        return None
    points = projected[visible]
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    size = maximum - minimum
    minimum -= margin * size
    maximum += margin * size
    minimum = np.maximum(minimum, (0.0, 0.0))
    maximum = np.minimum(maximum, (width - 1.0, height - 1.0))
    if np.any(maximum - minimum < 2.0):
        return None
    return np.asarray([minimum[0], minimum[1], maximum[0], maximum[1]], dtype=np.float64)


def _finite_error(values: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return float("inf"), float("inf")
    return float(np.median(finite)), float(np.percentile(finite, 95))


def main() -> int:
    args = parse_args()
    if args.max_samples < 0:
        raise ValueError("max-samples must be non-negative")
    if not 1 <= args.min_camera_inlier_joints <= 21:
        raise ValueError("min-camera-inlier-joints must be in [1, 21]")
    if not 0.0 <= args.bbox_margin <= 1.0:
        raise ValueError("bbox-margin must be in [0, 1]")
    if not 0.0 < args.min_visible_vertex_fraction <= 1.0:
        raise ValueError("min-visible-vertex-fraction must be in (0, 1]")
    if not 1 <= args.jpeg_quality <= 100:
        raise ValueError("jpeg-quality must be in [1, 100]")

    fusion = args.fusion.resolve()
    fit_root = args.mano_fit.resolve()
    dataset = args.dataset.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("dataset_type") != "normalized_multiview":
        raise ValueError(f"not a normalized multiview dataset: {dataset}")
    camera_ids = tuple(manifest["camera_ids"])
    cameras = tuple(dict.fromkeys(args.cameras))
    if any(camera not in camera_ids for camera in cameras):
        raise ValueError("an export camera is not present in the multiview dataset")

    with np.load(args.rectification.resolve()) as archive:
        left_camera = str(archive["left_camera"].item())
        right_camera = str(archive["right_camera"].item())
        rectification = {
            "R1": archive["R1"].astype(np.float32),
            "P1": archive["P1"].astype(np.float32),
            "P2": archive["P2"].astype(np.float32),
            "map_left_x": archive["map_left_x"].astype(np.float32),
            "map_left_y": archive["map_left_y"].astype(np.float32),
            "map_right_x": archive["map_right_x"].astype(np.float32),
            "map_right_y": archive["map_right_y"].astype(np.float32),
            "image_size": tuple(int(value) for value in archive["image_size"]),
        }
    if any(camera not in (left_camera, right_camera) for camera in cameras):
        raise ValueError(
            f"rectification supports only {left_camera}/{right_camera}, got {cameras}"
        )
    K = rectification["P1"][:, :3].copy().astype(np.float32)
    right_offset = np.linalg.solve(K, rectification["P2"][:, 3]).astype(np.float32)
    sync_rows = _load_sync_rows(dataset / "multiview_frames.csv")
    fusion_rows = _load_fusion(fusion / "accepted.jsonl")
    tracks = {}
    track_lookup = {}
    for side in (0, 1):
        path = fit_root / f"track_{side}.npz"
        if not path.is_file():
            raise FileNotFoundError(path)
        tracks[side] = _load_track(path)
        track_lookup[side] = {
            int(pair): index for index, pair in enumerate(tracks[side]["pair_indices"])
        }

    camera_order = {left_camera: 0, right_camera: 1}
    pending: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for row in fusion_rows:
        sync_index = int(row["sync_index"])
        if sync_index not in sync_rows:
            raise ValueError(f"missing synchronization row {sync_index}")
        for hand in sorted(row["hands"], key=lambda value: int(value["side"])):
            side = int(hand["side"])
            frame_index = track_lookup[side].get(sync_index)
            if frame_index is None:
                rejected.append({
                    "sync_index": sync_index, "side": side, "reason": "missing_mano_fit_frame",
                })
                continue
            track = tracks[side]
            if not bool(track["render_valid"][frame_index]):
                rejected.append({
                    "sync_index": sync_index, "side": side, "reason": "mano_fit_not_render_valid",
                })
                continue
            translation = track["translation"][frame_index].astype(np.float32)
            local_vertices = (
                track["vertices"][frame_index].astype(np.float32) - translation[None]
            )
            local_joints = (
                track["joints"][frame_index].astype(np.float32) - translation[None]
            )
            local_rectified = (rectification["R1"] @ local_vertices.T).T.astype(np.float32)
            joints_rectified = (rectification["R1"] @ local_joints.T).T.astype(np.float32)
            trans_left = (rectification["R1"] @ translation).astype(np.float32)
            global_matrix = Rotation.from_rotvec(
                track["global_orient"][frame_index].astype(np.float64)
            ).as_matrix()
            global_rectified = (
                rectification["R1"].astype(np.float64) @ global_matrix
            ).astype(np.float32)[None]
            hand_pose = Rotation.from_rotvec(
                track["hand_pose_axis_angle"][frame_index].reshape(-1, 3)
            ).as_matrix().astype(np.float32).reshape(15, 3, 3)

            for camera in cameras:
                view = hand.get("views", {}).get(camera)
                if view is None or int(view.get("inlier_joint_count", 0)) < args.min_camera_inlier_joints:
                    rejected.append({
                        "sync_index": sync_index, "side": side, "camera": camera,
                        "reason": "too_few_camera_inlier_joints",
                        "inlier_joint_count": 0 if view is None else int(view["inlier_joint_count"]),
                    })
                    continue
                error_key = (
                    "left_reprojection_error_px" if camera == left_camera
                    else "right_reprojection_error_px"
                )
                median_error, p95_error = _finite_error(track[error_key][frame_index])
                if (median_error > args.max_reprojection_median_px
                        or p95_error > args.max_reprojection_p95_px):
                    rejected.append({
                        "sync_index": sync_index, "side": side, "camera": camera,
                        "reason": "mano_fit_reprojection_error",
                        "median_px": median_error, "p95_px": p95_error,
                    })
                    continue
                trans = trans_left.copy()
                if camera == right_camera:
                    trans += right_offset
                try:
                    projected = _project(local_rectified, trans, K)
                except ValueError as error:
                    rejected.append({
                        "sync_index": sync_index, "side": side, "camera": camera,
                        "reason": "mesh_projection_failed", "detail": str(error),
                    })
                    continue
                bbox = _mesh_bbox(
                    projected, rectification["image_size"], args.bbox_margin,
                    args.min_visible_vertex_fraction,
                )
                if bbox is None:
                    rejected.append({
                        "sync_index": sync_index, "side": side, "camera": camera,
                        "reason": "mesh_not_sufficiently_visible",
                    })
                    continue
                sample = {
                    "bbox": bbox,
                    "vertices": local_rectified.copy(),
                    "joints_3d": joints_rectified.copy(),
                    "joints_2d_array": projected,
                    "side": np.asarray(float(side), dtype=np.float32),
                    "trans": trans,
                    "K": K.copy(),
                    "mano": {
                        "global_orient": global_rectified.copy(),
                        "hand_pose": hand_pose.copy(),
                        "betas": track["betas"].astype(np.float32).copy(),
                    },
                }
                pending.append({
                    "sync_index": sync_index,
                    "side": side,
                    "camera": camera,
                    "sample": sample,
                    "metadata": {
                        "sync_index": sync_index,
                        "camera": camera,
                        "side": side,
                        "handedness": "right" if side else "left",
                        "camera_inlier_joint_count": int(view["inlier_joint_count"]),
                        "fit_reprojection_median_px": median_error,
                        "fit_reprojection_p95_px": p95_error,
                        "fusion_mode": hand.get("fusion_mode"),
                    },
                })

    pending.sort(
        key=lambda item: (item["sync_index"], item["side"], camera_order[item["camera"]])
    )
    pending = pending[: args.max_samples or None]
    if not pending:
        raise RuntimeError("no samples passed training export quality checks")
    for index, item in enumerate(pending):
        stem = f"{index:06d}"
        item["stem"] = stem
        item["metadata"].update({
            "index": index,
            "image": f"images/{stem}.jpg",
            "label": f"labels/{stem}.npy",
        })

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        image_root = temporary / "images"
        label_root = temporary / "labels"
        image_root.mkdir()
        label_root.mkdir()
        import torch

        for item in pending:
            stored = item["sample"]
            sample = {
                "bbox": stored["bbox"],
                "vertices": stored["vertices"],
                "joints_3d": stored["joints_3d"],
                "joints_2d": torch.from_numpy(stored["joints_2d_array"]),
                "side": stored["side"],
                "trans": stored["trans"],
                "K": stored["K"],
                "mano": stored["mano"],
            }
            np.save(label_root / f"{item['stem']}.npy", sample, allow_pickle=True)

        image_size = tuple(manifest["image_size"])
        for camera in cameras:
            selected = [item for item in pending if item["camera"] == camera]
            if not selected:
                continue
            reader = SequentialVideoReader(
                dataset / "cameras" / camera / manifest["storage"]["video_filename"],
                image_size,
            )
            try:
                grouped: dict[int, list[dict[str, Any]]] = {}
                for item in selected:
                    grouped.setdefault(item["sync_index"], []).append(item)
                for sync_index in sorted(grouped):
                    source_index = int(sync_rows[sync_index][f"{camera}_frame_index"])
                    frame = reader.read(source_index)
                    if camera == left_camera:
                        map_x = rectification["map_left_x"]
                        map_y = rectification["map_left_y"]
                    else:
                        map_x = rectification["map_right_x"]
                        map_y = rectification["map_right_y"]
                    rectified = cv2.remap(
                        frame, map_x, map_y, cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT,
                    )
                    for item in grouped[sync_index]:
                        image_path = image_root / f"{item['stem']}.jpg"
                        if not cv2.imwrite(
                            str(image_path), rectified,
                            [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality],
                        ):
                            raise RuntimeError(f"cannot write image: {image_path}")
                        item["metadata"]["source_frame_index"] = source_index
            finally:
                reader.close()

        (temporary / "index.jsonl").write_text(
            "".join(
                json.dumps(item["metadata"], ensure_ascii=False, separators=(",", ":")) + "\n"
                for item in pending
            ),
            encoding="utf-8",
        )
        (temporary / "rejected.jsonl").write_text(
            "".join(
                json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                for row in rejected
            ),
            encoding="utf-8",
        )
        side_counts = {
            name: sum(item["side"] == side for item in pending)
            for side, name in ((0, "left"), (1, "right"))
        }
        camera_counts = {
            camera: sum(item["camera"] == camera for item in pending) for camera in cameras
        }
        summary = {
            "schema_version": 1,
            "stage": "six_view_refined_wilor_training_dataset",
            "schema_reference": "000865.npy",
            "sample_count": len(pending),
            "image_count": len(list(image_root.glob("*.jpg"))),
            "label_count": len(list(label_root.glob("*.npy"))),
            "side_counts": side_counts,
            "camera_counts": camera_counts,
            "rejected_count": len(rejected),
            "image_size": list(rectification["image_size"]),
            "K": K.tolist(),
            "cameras": list(cameras),
            "source": "strict six-view WiLoR fusion + shared MANO sequence fit",
            "quality_thresholds": {
                "min_camera_inlier_joints": args.min_camera_inlier_joints,
                "max_reprojection_median_px": args.max_reprojection_median_px,
                "max_reprojection_p95_px": args.max_reprojection_p95_px,
                "min_visible_vertex_fraction": args.min_visible_vertex_fraction,
            },
        }
        if summary["image_count"] != len(pending) or summary["label_count"] != len(pending):
            raise RuntimeError("paired image/label count mismatch before finalizing output")
        (temporary / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        temporary.rename(output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
