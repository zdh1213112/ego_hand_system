#!/usr/bin/env python3
"""Render random exported WiLoR frames as per-camera 21-joint overlay images."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from render_wilor_predictions import HAND_CONNECTIONS, LEFT_COLOR, RIGHT_COLOR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cameras", nargs="+", help="camera subset; default uses summary.json")
    parser.add_argument("--samples", type=int, default=12, help="random sync frames per camera")
    parser.add_argument("--seed", type=int, default=42, help="reproducible random seed")
    return parser.parse_args()


def _load_rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _project_label_joints(sample: dict[str, Any]) -> np.ndarray:
    joints = np.asarray(sample["joints_3d"], dtype=np.float64)
    translation = np.asarray(sample["trans"], dtype=np.float64)
    intrinsics = np.asarray(sample["K"], dtype=np.float64)
    if joints.shape != (21, 3) or translation.shape != (3,) or intrinsics.shape != (3, 3):
        raise ValueError("invalid joints_3d/trans/K shape in exported label")
    camera = joints + translation[None]
    homogeneous = (intrinsics @ camera.T).T
    if not np.isfinite(homogeneous).all() or np.any(homogeneous[:, 2] <= 1e-8):
        raise ValueError("cannot project exported 21-joint hand")
    return homogeneous[:, :2] / homogeneous[:, 2:3]


def _draw_sample(frame: np.ndarray, sample: dict[str, Any], metadata: dict[str, Any]) -> None:
    side = int(round(float(sample["side"])))
    color = RIGHT_COLOR if side else LEFT_COLOR
    handedness = "Right" if side else "Left"

    mesh = _as_numpy(sample["joints_2d"])
    if mesh.shape == (778, 2):
        mesh_layer = frame.copy()
        for point in np.rint(mesh[::3]).astype(np.int32):
            cv2.circle(mesh_layer, tuple(point), 1, color, -1, cv2.LINE_AA)
        cv2.addWeighted(mesh_layer, 0.35, frame, 0.65, 0.0, frame)

    joints = np.rint(_project_label_joints(sample)).astype(np.int32)
    for start, end in HAND_CONNECTIONS:
        cv2.line(frame, tuple(joints[start]), tuple(joints[end]), color, 4, cv2.LINE_AA)
    for index, point in enumerate(joints):
        radius = 7 if index == 0 else 5
        cv2.circle(frame, tuple(point), radius, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, tuple(point), max(2, radius - 2), color, -1, cv2.LINE_AA)
        cv2.putText(
            frame, str(index), (int(point[0]) + 6, int(point[1]) - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA,
        )

    box = np.rint(sample["bbox"]).astype(np.int32)
    cv2.rectangle(frame, tuple(box[:2]), tuple(box[2:]), color, 3, cv2.LINE_AA)
    inliers = int(metadata.get("camera_inlier_joint_count", 0))
    label = f"{handedness} | camera inliers {inliers}/21"
    cv2.putText(
        frame, label, (int(box[0]), max(32, int(box[1]) - 10)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.72, color, 2, cv2.LINE_AA,
    )


def _random_sync_indices(
    sync_indices: list[int], sample_count: int, rng: np.random.Generator,
) -> list[int]:
    if sample_count <= 0:
        raise ValueError("samples must be positive")
    if len(sync_indices) <= sample_count:
        return list(sync_indices)
    selected = rng.choice(sync_indices, size=sample_count, replace=False)
    return sorted(int(value) for value in selected)


def _reuse_complete(
    path: Path, dataset: Path, cameras: tuple[str, ...], samples: int, seed: int,
) -> bool:
    if not path.is_file():
        return False
    try:
        previous = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if (
        previous.get("dataset") != str(dataset)
        or int(previous.get("seed", -1)) != seed
        or int(previous.get("samples_per_camera", -1)) != samples
        or [result.get("camera") for result in previous.get("results", [])] != list(cameras)
    ):
        return False
    return all(
        (path.parent / image_name).is_file()
        for result in previous.get("results", [])
        for image_name in result.get("images", [])
    )


def _remove_previous_images(path: Path) -> None:
    if not path.is_file():
        return
    try:
        previous = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    for result in previous.get("results", []):
        for image_name in result.get("images", []):
            candidate = path.parent / Path(image_name).name
            if candidate.is_file():
                candidate.unlink()


def main() -> int:
    args = parse_args()
    if args.samples <= 0:
        raise ValueError("samples must be positive")
    dataset = args.dataset.resolve()
    output = args.output.resolve()
    summary = json.loads((dataset / "summary.json").read_text(encoding="utf-8"))
    rows = _load_rows(dataset / "index.jsonl")
    cameras = tuple(dict.fromkeys(args.cameras or summary.get("cameras", [])))
    if not cameras:
        raise ValueError("no visualization cameras selected")
    unknown = sorted(set(cameras) - set(summary.get("cameras", [])))
    if unknown:
        raise ValueError(f"cameras are not present in the exported dataset: {unknown}")
    output.mkdir(parents=True, exist_ok=True)
    visualization_summary_path = output / "summary.json"
    if _reuse_complete(
        visualization_summary_path, dataset, cameras, args.samples, args.seed
    ):
        print(f"[visualization] reuse {visualization_summary_path}")
        return 0
    _remove_previous_images(visualization_summary_path)
    rng = np.random.default_rng(args.seed)

    results = []
    for camera in cameras:
        camera_rows = [row for row in rows if row.get("camera") == camera]
        grouped: dict[int, list[dict[str, Any]]] = {}
        for row in camera_rows:
            grouped.setdefault(int(row["sync_index"]), []).append(row)
        available_sync_indices = sorted(grouped)
        if not available_sync_indices:
            raise RuntimeError(f"no samples available for visualization camera {camera}")
        sync_indices = _random_sync_indices(
            available_sync_indices, args.samples, rng
        )

        view_filter = str(summary.get("view_filter", "legacy"))
        width, height = (int(value) for value in summary["image_size"])
        image_names = []
        for sync_index in sync_indices:
            group = sorted(grouped[sync_index], key=lambda row: int(row["side"]))
            frame = cv2.imread(str(dataset / group[0]["image"]), cv2.IMREAD_COLOR)
            if frame is None or frame.shape[1::-1] != (width, height):
                raise RuntimeError(f"cannot read visualization image for sync {sync_index}")
            for row in group:
                sample = np.load(dataset / row["label"], allow_pickle=True).item()
                _draw_sample(frame, sample, row)
            requirement = (
                "21/21 required" if view_filter == "complete21" else "legacy view filter"
            )
            status = (
                f"{camera} | sync {sync_index} | {view_filter} | "
                f"hands {len(group)} | {requirement}"
            )
            cv2.rectangle(frame, (0, 0), (width, 58), (16, 24, 32), -1)
            cv2.putText(
                frame, status, (24, 39), cv2.FONT_HERSHEY_SIMPLEX,
                0.92, (245, 245, 245), 2, cv2.LINE_AA,
            )
            image_name = f"{camera}_sync{sync_index:06d}_{view_filter}.jpg"
            if not cv2.imwrite(str(output / image_name), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                raise RuntimeError(f"cannot write visualization image: {image_name}")
            image_names.append(image_name)
        results.append({
            "camera": camera,
            "view_filter": view_filter,
            "available_sync_count": len(available_sync_indices),
            "visualized_sync_count": len(sync_indices),
            "selected_sync_indices": sync_indices,
            "images": image_names,
        })
        print(f"[visualization] {camera}: {len(sync_indices)} random frames -> {output}")

    visualization_summary = {
        "stage": "wilor_training_label_visualization",
        "dataset": str(dataset),
        "seed": args.seed,
        "samples_per_camera": args.samples,
        "results": results,
    }
    visualization_summary_path.write_text(
        json.dumps(visualization_summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
