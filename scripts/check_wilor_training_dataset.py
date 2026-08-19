#!/usr/bin/env python3
"""Strictly validate paired images and 000865-compatible WiLoR NPY labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


EXPECTED_KEYS = (
    "bbox", "vertices", "joints_3d", "joints_2d", "side", "trans", "K", "mano"
)
EXPECTED_MANO_KEYS = ("global_orient", "hand_pose", "betas")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--reference", type=Path, help="optional reference NPY such as 000865.npy")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--projection-tolerance-px", type=float, default=1e-3)
    return parser.parse_args()


def _array(value: Any, dtype: np.dtype, shape: tuple[int, ...], name: str) -> None:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name}: expected numpy.ndarray, got {type(value).__name__}")
    if value.dtype != dtype or value.shape != shape:
        raise ValueError(
            f"{name}: expected dtype={dtype}, shape={shape}; "
            f"got dtype={value.dtype}, shape={value.shape}"
        )


def _contract(sample: dict[str, Any], torch) -> None:
    if tuple(sample) != EXPECTED_KEYS:
        raise ValueError(f"keys: expected {EXPECTED_KEYS}, got {tuple(sample)}")
    _array(sample["bbox"], np.dtype("float64"), (4,), "bbox")
    _array(sample["vertices"], np.dtype("float32"), (778, 3), "vertices")
    _array(sample["joints_3d"], np.dtype("float32"), (21, 3), "joints_3d")
    joints_2d = sample["joints_2d"]
    if not isinstance(joints_2d, torch.Tensor):
        raise TypeError(f"joints_2d: expected torch.Tensor, got {type(joints_2d).__name__}")
    if joints_2d.dtype != torch.float32 or tuple(joints_2d.shape) != (778, 2):
        raise ValueError(f"joints_2d: expected torch.float32 (778,2), got {joints_2d.dtype} {tuple(joints_2d.shape)}")
    _array(sample["side"], np.dtype("float32"), (), "side")
    if float(sample["side"]) not in (0.0, 1.0):
        raise ValueError(f"side must be 0 or 1, got {float(sample['side'])}")
    _array(sample["trans"], np.dtype("float32"), (3,), "trans")
    _array(sample["K"], np.dtype("float32"), (3, 3), "K")
    mano = sample["mano"]
    if not isinstance(mano, dict) or tuple(mano) != EXPECTED_MANO_KEYS:
        raise ValueError(f"mano keys: expected {EXPECTED_MANO_KEYS}, got {tuple(mano)}")
    _array(mano["global_orient"], np.dtype("float32"), (1, 3, 3), "mano.global_orient")
    _array(mano["hand_pose"], np.dtype("float32"), (15, 3, 3), "mano.hand_pose")
    _array(mano["betas"], np.dtype("float32"), (10,), "mano.betas")


def _signature(sample: dict[str, Any]) -> dict[str, tuple[str, tuple[int, ...]]]:
    result = {}
    for key in EXPECTED_KEYS:
        value = sample[key]
        if key == "mano":
            for mano_key in EXPECTED_MANO_KEYS:
                item = value[mano_key]
                result[f"mano.{mano_key}"] = (str(item.dtype), tuple(item.shape))
        elif key == "joints_2d":
            result[key] = (str(value.dtype), tuple(value.shape))
        else:
            result[key] = (str(value.dtype), tuple(value.shape))
    return result


def main() -> int:
    args = parse_args()
    if args.max_samples < 0 or args.projection_tolerance_px <= 0:
        raise ValueError("max-samples must be non-negative and tolerance positive")
    root = args.dataset.resolve()
    image_root = root / "images"
    label_root = root / "labels"
    images = sorted(image_root.glob("*.jpg"))
    labels = sorted(label_root.glob("*.npy"))
    if not images or [path.stem for path in images] != [path.stem for path in labels]:
        raise ValueError("images/*.jpg and labels/*.npy must be non-empty one-to-one pairs")
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    if int(summary["sample_count"]) != len(labels):
        raise ValueError("summary sample_count disagrees with paired files")
    index_rows = [
        json.loads(line) for line in (root / "index.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(index_rows) != len(labels):
        raise ValueError("index.jsonl length disagrees with paired files")

    import torch
    reference_signature = None
    if args.reference is not None:
        reference = np.load(args.reference.resolve(), allow_pickle=True).item()
        _contract(reference, torch)
        reference_signature = _signature(reference)
    selected = labels[: args.max_samples or None]
    maximum_projection_error = 0.0
    side_counts = {0: 0, 1: 0}
    for label_path in selected:
        sample = np.load(label_path, allow_pickle=True).item()
        _contract(sample, torch)
        if reference_signature is not None and _signature(sample) != reference_signature:
            raise ValueError(f"{label_path}: schema signature differs from reference")
        image_path = image_root / f"{label_path.stem}.jpg"
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"cannot decode image: {image_path}")
        height, width = image.shape[:2]
        bbox = sample["bbox"]
        if not (
            0 <= bbox[0] < bbox[2] < width
            and 0 <= bbox[1] < bbox[3] < height
        ):
            raise ValueError(f"{label_path}: bbox outside paired image {width}x{height}: {bbox}")
        camera_vertices = sample["vertices"] + sample["trans"][None]
        homogeneous = (sample["K"] @ camera_vertices.T).T
        if np.any(homogeneous[:, 2] <= 0):
            raise ValueError(f"{label_path}: vertices behind camera")
        projected = homogeneous[:, :2] / homogeneous[:, 2:3]
        stored = sample["joints_2d"].detach().cpu().numpy()
        error = float(np.max(np.abs(projected - stored)))
        maximum_projection_error = max(maximum_projection_error, error)
        if error > args.projection_tolerance_px:
            raise ValueError(
                f"{label_path}: vertices/K/trans projection error {error:.6g}px exceeds "
                f"{args.projection_tolerance_px}px"
            )
        side_counts[int(float(sample["side"]))] += 1
    result = {
        "validated_sample_count": len(selected),
        "total_sample_count": len(labels),
        "paired_images": len(images),
        "schema": "000865-compatible",
        "reference_compared": str(args.reference.resolve()) if args.reference else None,
        "maximum_projection_error_px": maximum_projection_error,
        "side_counts_in_validated_subset": {
            "left": side_counts[0], "right": side_counts[1],
        },
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
