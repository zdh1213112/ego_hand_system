"""Geometry, MANO replay, and supervision-profile validation for glove data."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import cv2
import torch

from wilor.models import MANO, resolve_mano_model_path

from .glove_dataset import FrameGroupKey, GloveDataset, frame_group_key


PARAMETER_NAMES = ("global_orient", "hand_pose", "betas")
MANO_SHAPES = {
    "global_orient": (1, 3, 3),
    "hand_pose": (15, 3, 3),
    "betas": (10,),
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _project(points: np.ndarray, trans: np.ndarray, K: np.ndarray, distortion=None) -> np.ndarray:
    camera = points + trans.reshape(1, 3)
    homogeneous = camera @ K.T
    if not np.isfinite(homogeneous).all() or np.any(homogeneous[:, 2] <= 0):
        raise ValueError("geometry contains invalid or non-positive camera depth")
    if distortion is not None:
        return cv2.fisheye.projectPoints(camera.astype(np.float64)[:, None], np.zeros(3), np.zeros(3),
            K.astype(np.float64), np.asarray(distortion, np.float64))[0][:, 0]
    return homogeneous[:, :2] / homogeneous[:, 2:3]


def manifest_camera_model(manifest):
    path = Path(manifest).resolve().parent / 'summary.json'
    summary = json.loads(path.read_text()) if path.is_file() else {}
    return 'physical_kb' if summary.get('image_space') == 'native_raw' else 'physical_perspective'


def validate_manifest(
    manifest: str | Path,
    max_samples: int | None = None,
    projection_tolerance_px: float = 1e-3,
) -> dict[str, Any]:
    """Validate file pairing and the stored geometry/camera projection."""
    if projection_tolerance_px <= 0:
        raise ValueError("projection_tolerance_px must be positive")
    dataset = GloveDataset(manifest, max_samples=max_samples, camera_model=manifest_camera_model(manifest))
    errors: list[str] = []
    side_counts = {0: 0, 1: 0}
    group_sides: set[tuple[FrameGroupKey, int]] = set()
    maximum_projection_error = 0.0
    for index, row in enumerate(dataset.rows):
        try:
            label_path = dataset._resolve(row["label"])
            label = dataset._load_label(label_path, allow_native=dataset.camera_model == "physical_kb")
            image_path = dataset._resolve(row["image"])
            if not image_path.is_file():
                raise FileNotFoundError(f"missing image {image_path}")
            expected_shapes = {
                "bbox": (4,), "vertices": (778, 3), "joints_3d": (21, 3),
                "joints_2d": (778, 2), "trans": (3,), "K": (3, 3),
            }
            arrays: dict[str, np.ndarray] = {}
            for key, shape in expected_shapes.items():
                if key not in label:
                    raise ValueError(f"missing required geometry field {key}")
                value = dataset._as_numpy(label[key])
                if value.shape != shape:
                    raise ValueError(f"{key} shape {value.shape}, expected {shape}")
                if not np.isfinite(value).all():
                    raise ValueError(f"{key} contains NaN/Inf")
                arrays[key] = value.astype(np.float32)
            side = int(np.asarray(label["side"]).item())
            if side not in side_counts:
                raise ValueError(f"side must be 0 or 1, got {side}")
            if int(row.get("side", side)) != side:
                raise ValueError("index/label side mismatch")
            group_side = (frame_group_key(row), side)
            if group_side in group_sides:
                raise ValueError(f"duplicate side {side} in frame group {group_side[0]}")
            group_sides.add(group_side)
            side_counts[side] += 1
            stored_projection = arrays["joints_2d"]
            reconstructed_projection = _project(
                arrays["vertices"], arrays["trans"], arrays["K"], label.get("distortion")
            )
            projection_error = float(np.max(np.abs(
                reconstructed_projection - stored_projection
            )))
            maximum_projection_error = max(maximum_projection_error, projection_error)
            if projection_error > projection_tolerance_px:
                raise ValueError(
                    f"vertices/trans/K projection error {projection_error:.6g}px "
                    f"exceeds {projection_tolerance_px}px"
                )
            _project(arrays["joints_3d"], arrays["trans"], arrays["K"], label.get("distortion"))
        except Exception as exc:
            errors.append(f"row {index}: {exc}")
    return {
        "manifest": str(Path(manifest).expanduser().resolve()),
        "samples": len(dataset),
        "side_counts": side_counts,
        "maximum_vertex_projection_error_px": maximum_projection_error,
        "projection_tolerance_px": projection_tolerance_px,
        "errors": errors,
        "valid": not errors,
    }


def _validate_mano_fields(label: dict[str, Any]) -> dict[str, np.ndarray]:
    raw = label.get("mano")
    if not isinstance(raw, dict):
        raise ValueError("missing mano dictionary")
    result = {}
    for name, shape in MANO_SHAPES.items():
        if name not in raw:
            raise ValueError(f"missing mano.{name}")
        value = GloveDataset._as_numpy(raw[name]).astype(np.float32)
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f"mano.{name} must be finite with shape {shape}")
        result[name] = value
    return result


def replay_mano_labels(
    root: str | Path,
    mano_path: str | Path,
    model_by_side: dict[str, str],
    reconstruction_tolerance_m: float = 1e-5,
    projection_tolerance_px: float = 1e-3,
) -> dict[str, Any]:
    """Replay labels using the model explicitly declared for each physical side."""
    dataset_root = Path(root).expanduser().resolve()
    resolved_model_path = resolve_mano_model_path(mano_path)
    model_dir = (
        resolved_model_path.parent if resolved_model_path.is_file()
        else resolved_model_path
    )
    native = manifest_camera_model(dataset_root / "index.jsonl") == "physical_kb"
    rows = GloveDataset(dataset_root / "index.jsonl", camera_model="physical_kb" if native else "physical_perspective").rows
    grouped: dict[int, list[tuple[int, Path]]] = {0: [], 1: []}
    field_errors: list[str] = []
    for index, row in enumerate(rows):
        label_path = dataset_root / row["label"]
        try:
            label = GloveDataset._load_label(label_path, allow_native=native)
            _validate_mano_fields(dict(label))
            side = int(np.asarray(label["side"]).item())
            grouped[side].append((index, label_path))
        except Exception as exc:
            if len(field_errors) < 20:
                field_errors.append(f"row {index}: {exc}")

    maximum_vertex_error = 0.0
    maximum_joint_error = 0.0
    maximum_projection_error = 0.0
    maximum_joint_projection_error = 0.0
    mismatch_count = 0
    mismatch_examples: list[str] = []
    checked = 0
    for side in (0, 1):
        selected = grouped[side]
        if not selected:
            continue
        side_name = "right" if side else "left"
        model_name = model_by_side[side_name]
        is_right = model_name == "MANO_RIGHT.pkl"
        model = MANO(
            model_path=str(model_dir), is_rhand=is_right, use_pca=False,
            flat_hand_mean=True,
        ).eval()
        for start in range(0, len(selected), 256):
            chunk = selected[start:start + 256]
            labels = [
                GloveDataset._load_label(path, allow_native=native) for _, path in chunk
            ]
            mano_values = [_validate_mano_fields(dict(label)) for label in labels]
            with torch.no_grad():
                output = model(
                    global_orient=torch.from_numpy(np.stack([
                        value["global_orient"] for value in mano_values
                    ])).float(),
                    hand_pose=torch.from_numpy(np.stack([
                        value["hand_pose"] for value in mano_values
                    ])).float(),
                    betas=torch.from_numpy(np.stack([
                        value["betas"] for value in mano_values
                    ])).float(),
                )
            vertices = output.vertices.cpu().numpy()
            joints = output.joints.cpu().numpy()
            for offset, ((row_index, _), label) in enumerate(zip(chunk, labels)):
                stored_vertices = GloveDataset._as_numpy(label["vertices"])
                stored_joints = GloveDataset._as_numpy(label["joints_3d"])
                target_vertices = stored_vertices.copy()
                target_joints = stored_joints.copy()
                model_side_differs = bool(side) != is_right
                if model_side_differs:
                    target_vertices[:, 0] *= -1.0
                    target_joints[:, 0] *= -1.0
                vertex_error = float(np.max(np.abs(
                    vertices[offset] - target_vertices
                )))
                joint_error = float(np.max(np.abs(
                    joints[offset] - target_joints
                )))
                trans = GloveDataset._as_numpy(label["trans"])
                K = GloveDataset._as_numpy(label["K"])
                stored_2d = GloveDataset._as_numpy(label["joints_2d"])
                target_vertex_2d = (
                    _project(target_vertices, trans, K, label.get("distortion"))
                    if model_side_differs else stored_2d
                )
                projection_error = float(np.max(np.abs(
                    _project(vertices[offset], trans, K, label.get("distortion")) - target_vertex_2d
                )))
                joint_projection_error = float(np.max(np.abs(
                    _project(joints[offset], trans, K, label.get("distortion"))
                    - _project(target_joints, trans, K, label.get("distortion"))
                )))
                maximum_vertex_error = max(maximum_vertex_error, vertex_error)
                maximum_joint_error = max(maximum_joint_error, joint_error)
                maximum_projection_error = max(maximum_projection_error, projection_error)
                maximum_joint_projection_error = max(
                    maximum_joint_projection_error, joint_projection_error
                )
                checked += 1
                if (max(vertex_error, joint_error) > reconstruction_tolerance_m
                        or projection_error > projection_tolerance_px
                        or joint_projection_error > projection_tolerance_px):
                    mismatch_count += 1
                    if len(mismatch_examples) < 20:
                        mismatch_examples.append(
                            f"row {row_index}: vertices={vertex_error:.6g}m, "
                            f"joints={joint_error:.6g}m, "
                            f"vertex_projection={projection_error:.6g}px, "
                            f"joint_projection={joint_projection_error:.6g}px"
                        )
    internal_valid = (
        not field_errors and checked == len(rows) and mismatch_count == 0
    )
    return {
        "internally_valid": internal_valid,
        "checked_samples": checked,
        "label_samples": len(rows),
        "field_error_count": len(rows) - sum(len(values) for values in grouped.values()),
        "mismatch_count": mismatch_count,
        "maximum_vertex_error_m": maximum_vertex_error,
        "maximum_joint_error_m": maximum_joint_error,
        "maximum_vertex_projection_error_px": maximum_projection_error,
        "maximum_joint_projection_error_px": maximum_joint_projection_error,
        "reconstruction_tolerance_m": reconstruction_tolerance_m,
        "projection_tolerance_px": projection_tolerance_px,
        "field_error_examples": field_errors,
        "mismatch_examples": mismatch_examples,
    }


def build_dataset_profile(
    root: str | Path,
    mano_path: str | Path,
    projection_tolerance_px: float = 1e-3,
    mano_reconstruction_tolerance_m: float = 1e-5,
) -> dict[str, Any]:
    """Validate a dataset and write the sole source of supervision availability."""
    dataset_root = Path(root).expanduser().resolve()
    geometry = validate_manifest(
        dataset_root / "index.jsonl",
        projection_tolerance_px=projection_tolerance_px,
    )
    summary_path = dataset_root / "summary.json"
    summary = (
        json.loads(summary_path.read_text(encoding="utf-8"))
        if summary_path.is_file() else {}
    )
    declared_model_by_side = summary.get("mano_model_by_side")
    default_model_by_side = {
        "left": "MANO_LEFT.pkl",
        "right": "MANO_RIGHT.pkl",
    }
    mapping_declared = isinstance(declared_model_by_side, dict)
    model_by_side = (
        dict(declared_model_by_side) if mapping_declared
        else default_model_by_side
    )
    mapping_errors = []
    supported_models = {"MANO_LEFT.pkl", "MANO_RIGHT.pkl"}
    if mapping_declared:
        if set(model_by_side) != {"left", "right"}:
            mapping_errors.append(
                "mano_model_by_side must contain exactly left and right"
            )
        elif any(name not in supported_models for name in model_by_side.values()):
            mapping_errors.append(
                f"mano_model_by_side supports only {sorted(supported_models)}"
            )
    resolved_model_path = resolve_mano_model_path(mano_path)
    mano_model_dir = (
        resolved_model_path.parent if resolved_model_path.is_file()
        else resolved_model_path
    )
    required_model_names = set(model_by_side.values()) | {"MANO_RIGHT.pkl"}
    assets = {}
    asset_errors = list(mapping_errors)
    for name in sorted(required_model_names & supported_models):
        asset_path = mano_model_dir / name
        if not asset_path.is_file():
            asset_errors.append(f"missing MANO asset {asset_path}")
            continue
        assets[name] = {
            "path": str(asset_path),
            "sha256": sha256_file(asset_path),
        }
    declared_assets = summary.get("mano_assets", {})
    for name, expected in declared_assets.items():
        if name in assets and expected.get("sha256") != assets[name]["sha256"]:
            asset_errors.append(f"{name} does not match the exporter SHA-256")
    if mapping_declared and not mapping_errors:
        if summary.get("mano_pose_representation") != "rotation_matrix_full_mean":
            asset_errors.append(
                "MANO pose representation must be rotation_matrix_full_mean"
            )
        for name in set(model_by_side.values()):
            if not declared_assets.get(name, {}).get("sha256"):
                asset_errors.append(f"summary does not declare {name} SHA-256")
    if geometry["valid"] and not asset_errors:
        mano_replay = replay_mano_labels(
            dataset_root, mano_model_dir, model_by_side,
            reconstruction_tolerance_m=mano_reconstruction_tolerance_m,
            projection_tolerance_px=projection_tolerance_px,
        )
    else:
        mano_replay = {
            "internally_valid": False,
            "checked_samples": 0,
            "label_samples": geometry["samples"],
            "reason": "geometry or MANO asset validation failed",
        }
    mano_replay["asset_errors"] = asset_errors
    all_right_model = (
        set(model_by_side) == {"left", "right"}
        and all(name == "MANO_RIGHT.pkl" for name in model_by_side.values())
    )
    parameter_available = bool(
        mapping_declared
        and not asset_errors
        and mano_replay["internally_valid"]
        and all_right_model
    )
    if parameter_available:
        parameter_reason = "right-canonical MANO labels replay exactly"
    elif mano_replay["internally_valid"]:
        parameter_reason = (
            "declared MANO models replay exactly, but both sides are not mapped "
            "to MANO_RIGHT.pkl"
        )
    elif not mapping_declared:
        parameter_reason = (
            "dataset has no declared mano_model_by_side and its parameters "
            "do not replay the stored geometry"
        )
    else:
        parameter_reason = "MANO parameters failed asset or geometry replay validation"
    sample_count = int(geometry["samples"])
    geometry_reason = (
        "stored vertices/trans/K and derived 21-joint projections are consistent"
        if geometry["valid"] else "geometry validation failed"
    )
    supervision = {
        "keypoints_2d": {
            "available": bool(geometry["valid"]),
            "count": sample_count if geometry["valid"] else 0,
            "reason": geometry_reason,
        },
        "keypoints_3d": {
            "available": bool(geometry["valid"]),
            "count": sample_count if geometry["valid"] else 0,
            "reason": geometry_reason,
        },
    }
    for name in PARAMETER_NAMES:
        supervision[name] = {
            "available": parameter_available,
            "count": sample_count if parameter_available else 0,
            "reason": parameter_reason,
        }
    profile = {
        "schema_version": 1,
        "dataset_root": str(dataset_root),
        "image_space": summary.get("image_space", "physical_rectified"),
        "camera_model": manifest_camera_model(dataset_root / 'index.jsonl'),
        "sample_count": sample_count,
        "mano_model_by_side": (
            dict(declared_model_by_side) if mapping_declared else None
        ),
        "mano_replay_model_by_side": model_by_side,
        "mano_pose_representation": summary.get("mano_pose_representation"),
        "geometry_validation": geometry,
        "mano_replay": mano_replay,
        "mano_assets": assets,
        "supervision": supervision,
    }
    (dataset_root / "dataset_profile.json").write_text(
        json.dumps(profile, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return profile
