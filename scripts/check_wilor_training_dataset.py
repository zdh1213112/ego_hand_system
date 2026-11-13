#!/usr/bin/env python3
"""Strictly validate paired images and 000865-compatible WiLoR NPY labels."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from mano_conventions import mirror_left_points
from wilor_label_projection import project_label_points


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
    parser.add_argument(
        "--mano-reconstruction-tolerance-m", type=float, default=1e-5,
        help="maximum absolute MANO replay error for vertices and joints",
    )
    parser.add_argument("--mano-source", type=Path, help="licensed MANO Python source")
    parser.add_argument(
        "--mano-model-dir", type=Path,
        help="directory containing the exact MANO_RIGHT.pkl asset",
    )
    parser.add_argument(
        "--device", choices=("cpu", "cuda", "auto"), default="cpu",
        help="device used for batched MANO replay validation",
    )
    parser.add_argument(
        "--batch-size", type=int, default=256,
        help="MANO replay batch size",
    )
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
    expected = EXPECTED_KEYS + (("image_space", "camera_model", "distortion") if sample.get("image_space") == "native_raw" else ())
    if tuple(sample) != expected:
        raise ValueError(f"keys: expected {expected}, got {tuple(sample)}")
    if sample.get("image_space") == "native_raw":
        if sample.get("camera_model") != "KB":
            raise ValueError("native label requires KB camera model")
        _array(sample["distortion"], np.dtype("float64"), (4,), "distortion")
        if not np.isfinite(sample["distortion"]).all():
            raise ValueError("nonfinite distortion")
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _replay_mano(
    records: list[tuple[Path, dict[str, Any]]], source: Path, model_dir: Path,
    model_by_side: dict[str, str], tolerance_m: float,
    device: str = "cpu", batch_size: int = 256,
) -> tuple[float, float]:
    """Replay right-canonical parameters against physical-side label geometry."""
    import torch
    from scipy.spatial.transform import Rotation

    from fit_mano_sequence import MANO_TO_MEDIAPIPE, import_mano

    expected_mapping = {"left": "MANO_RIGHT.pkl", "right": "MANO_RIGHT.pkl"}
    if model_by_side != expected_mapping:
        raise ValueError(
            f"MANO_RIGHT replay requires {expected_mapping}, got {model_by_side}"
        )
    mano = import_mano(source)
    maximum_vertex_error = 0.0
    maximum_joint_error = 0.0
    model = mano.load(
        model_path=str(model_dir), is_rhand=True, use_pca=False,
        num_pca_comps=45, batch_size=1, flat_hand_mean=True,
    ).to(device).eval()
    order = torch.as_tensor(MANO_TO_MEDIAPIPE, dtype=torch.long, device=device)
    for start in range(0, len(records), batch_size):
        chunk = records[start:start + batch_size]
        global_matrices = np.concatenate(
            [sample["mano"]["global_orient"] for _, sample in chunk], axis=0
        )
        hand_matrices = np.stack(
            [sample["mano"]["hand_pose"] for _, sample in chunk], axis=0
        )
        global_axis_angle = Rotation.from_matrix(global_matrices).as_rotvec()
        hand_axis_angle = Rotation.from_matrix(
            hand_matrices.reshape(-1, 3, 3)
        ).as_rotvec().reshape(len(chunk), 45)
        betas = np.stack(
            [sample["mano"]["betas"] for _, sample in chunk], axis=0
        )
        with torch.no_grad():
            output = model(
                betas=torch.from_numpy(betas).float().to(device),
                global_orient=torch.from_numpy(global_axis_angle).float().to(device),
                hand_pose=torch.from_numpy(hand_axis_angle).float().to(device),
                transl=torch.zeros(
                    (len(chunk), 3), dtype=torch.float32, device=device
                ),
                return_verts=True, return_tips=True,
            )
            vertices = output.vertices.cpu().numpy()
            joints = output.joints.index_select(1, order).cpu().numpy()
        for index, (path, sample) in enumerate(chunk):
            side = int(float(sample["side"]))
            canonical_vertices = mirror_left_points(sample["vertices"], side)
            canonical_joints = mirror_left_points(sample["joints_3d"], side)
            vertex_error = float(np.max(np.abs(
                vertices[index] - canonical_vertices
            )))
            joint_error = float(np.max(np.abs(
                joints[index] - canonical_joints
            )))
            maximum_vertex_error = max(maximum_vertex_error, vertex_error)
            maximum_joint_error = max(maximum_joint_error, joint_error)
            if max(vertex_error, joint_error) > tolerance_m:
                raise ValueError(
                    f"{path}: MANO_RIGHT replay error exceeds {tolerance_m}m "
                    f"(vertices={vertex_error:.6g}m, joints={joint_error:.6g}m)"
                )
    return maximum_vertex_error, maximum_joint_error


def main() -> int:
    args = parse_args()
    if (args.max_samples < 0 or args.projection_tolerance_px <= 0
            or args.mano_reconstruction_tolerance_m <= 0
            or args.batch_size <= 0):
        raise ValueError(
            "max-samples must be non-negative; tolerances and batch-size must be positive"
        )
    root = args.dataset.resolve()
    image_root = root / "images"
    label_root = root / "labels"
    images = sorted(image_root.glob("*.jpg"))
    labels = sorted(label_root.glob("*.npy"))
    if not images or not labels:
        raise ValueError("images and labels must be nonempty")
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    schema_version = int(summary.get("schema_version", -1))
    if schema_version not in (4, 5, 6):
        raise ValueError("dataset must use physical-label schema_version 4, 5 or 6")
    if int(summary["sample_count"]) != len(labels):
        raise ValueError("summary sample_count disagrees with paired files")
    index_rows = [
        json.loads(line) for line in (root / "index.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(index_rows) != len(labels):
        raise ValueError("index.jsonl length disagrees with paired files")
    index_by_stem = {
        Path(row["label"]).stem: row for row in index_rows
    }
    if len(index_by_stem) != len(index_rows):
        raise ValueError("index.jsonl contains duplicate label names")

    if {row["image"] for row in index_rows} != {f"images/{p.name}" for p in images}:
        raise ValueError("image files disagree with frame references")
    if {row["label"] for row in index_rows} != {f"labels/{p.name}" for p in labels}:
        raise ValueError("label files disagree with instance references")
    if int(summary["image_count"]) != len(images) or int(summary["label_count"]) != len(labels):
        raise ValueError("summary image/instance counts mismatch")
    frame_sides = {}
    for row in index_rows:
        sides = frame_sides.setdefault(row["image"], set())
        if int(row["side"]) in sides:
            raise ValueError("duplicate physical hand in frame")
        sides.add(int(row["side"]))
    if summary.get("annotation_policy") == "require_both_hands" and any(s != {0, 1} for s in frame_sides.values()):
        raise ValueError("dual-hand dataset contains incomplete frame")
    import torch
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested for MANO validation but unavailable")
    if device == "cuda":
        print(
            f"[dataset-check] CUDA device={torch.cuda.get_device_name()} "
            f"MANO batch_size={args.batch_size}",
            flush=True,
        )
    reference_signature = None
    if args.reference is not None:
        reference = np.load(args.reference.resolve(), allow_pickle=True).item()
        _contract(reference, torch)
        if reference.get("image_space", "physical_rectified") != summary.get("image_space"):
            raise ValueError("reference and export use different image spaces")
        reference_signature = _signature(reference)
    selected = labels[: args.max_samples or None]
    maximum_projection_error = 0.0
    side_counts = {0: 0, 1: 0}
    records: list[tuple[Path, dict[str, Any]]] = []
    for label_path in selected:
        sample = np.load(label_path, allow_pickle=True).item()
        _contract(sample, torch)
        if reference_signature is not None and _signature(sample) != reference_signature:
            raise ValueError(f"{label_path}: schema signature differs from reference")
        image_path = root / index_by_stem[label_path.stem]["image"]
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
        projected = project_label_points(camera_vertices, sample)
        stored = sample["joints_2d"].detach().cpu().numpy()
        error = float(np.max(np.abs(projected - stored)))
        maximum_projection_error = max(maximum_projection_error, error)
        if error > args.projection_tolerance_px:
            raise ValueError(
                f"{label_path}: vertices/K/trans projection error {error:.6g}px exceeds "
                f"{args.projection_tolerance_px}px"
            )
        side_counts[int(float(sample["side"]))] += 1
        index_row = index_by_stem.get(label_path.stem)
        if index_row is None:
            raise ValueError(f"{label_path}: missing index.jsonl record")
        physical_side = int(float(sample["side"]))
        if int(index_row.get("physical_side", -1)) != physical_side:
            raise ValueError(f"{label_path}: physical_side disagrees with label side")
        expected_space = "native_raw" if schema_version == 6 else "physical_rectified"
        if sample.get("image_space", "physical_rectified") != expected_space:
            raise ValueError("label image space disagrees with dataset schema")
        if expected_space == "native_raw":
            if "camera_calibrations_by_source" in summary:
                group = str(index_row["calibration_group"])
                if group != str(index_row["source_dataset_id"]):
                    raise ValueError("calibration group/source dataset mismatch")
                cal = summary["camera_calibrations_by_source"][group][index_row["camera"]]
            else:
                cal = summary["camera_calibrations"][index_row["camera"]]
            if cal["model"] != "KB" or cal["image_size"] != [width, height]:
                raise ValueError("camera calibration model/size mismatch")
            np.testing.assert_allclose(sample["K"], cal["K"], rtol=1e-6)
            np.testing.assert_allclose(sample["distortion"], cal["distortion"], rtol=1e-10)
        if index_row.get("stored_image_space") != expected_space:
            raise ValueError(f"{label_path}: image space disagrees with schema")
        if bool(index_row.get("stored_image_horizontally_flipped")):
            raise ValueError(f"{label_path}: physical image must not be flipped")
        if index_row.get("mano_parameter_space") != "right_canonical":
            raise ValueError(f"{label_path}: MANO parameters are not right_canonical")
        records.append((label_path, sample))

    mano_model_by_side = summary.get("mano_model_by_side")
    maximum_mano_vertex_error = None
    maximum_mano_joint_error = None
    if mano_model_by_side is not None:
        if not isinstance(mano_model_by_side, dict) or set(mano_model_by_side) != {"left", "right"}:
            raise ValueError("mano_model_by_side must contain exactly left and right")
        expected_mapping = {"left": "MANO_RIGHT.pkl", "right": "MANO_RIGHT.pkl"}
        if mano_model_by_side != expected_mapping:
            raise ValueError(
                f"MANO parameter mapping requires {expected_mapping}, got {mano_model_by_side}"
            )
        if summary.get("mano_pose_representation") != "rotation_matrix_full_mean":
            raise ValueError(
                "mano_pose_representation must be rotation_matrix_full_mean"
            )
        if summary.get("image_space") != ("native_raw" if schema_version == 6 else "physical_rectified"):
            raise ValueError("summary image space disagrees with schema")
        if summary.get("geometry_space") != "physical_side":
            raise ValueError("vertices and joints must remain in physical-side space")
        if summary.get("mano_parameter_space") != "right_canonical":
            raise ValueError("MANO parameters must use the right-canonical space")
        if summary.get("side_semantics") != "physical_identity":
            raise ValueError("side must describe physical hand identity")
        if args.mano_source is None or args.mano_model_dir is None:
            raise ValueError(
                "MANO replay requires --mano-source and --mano-model-dir"
            )
        source = args.mano_source.resolve()
        model_dir = args.mano_model_dir.resolve()
        if not (source / "mano" / "model.py").is_file():
            raise FileNotFoundError(f"invalid MANO source: {source}")
        expected_assets = summary.get("mano_assets", {})
        for name in sorted(set(mano_model_by_side.values())):
            asset = model_dir / name
            if not asset.is_file():
                raise FileNotFoundError(asset)
            expected_hash = expected_assets.get(name, {}).get("sha256")
            if not expected_hash:
                raise ValueError(f"summary does not declare the SHA-256 of {name}")
            if _sha256(asset) != expected_hash:
                raise ValueError(f"{asset}: SHA-256 differs from export asset")
        maximum_mano_vertex_error, maximum_mano_joint_error = _replay_mano(
            records, source, model_dir, mano_model_by_side,
            args.mano_reconstruction_tolerance_m, device, args.batch_size,
        )
    result = {
        "validated_sample_count": len(selected),
        "total_sample_count": len(labels),
        "paired_images": len(images),
        "schema": "native-kb-v6" if schema_version == 6 else "000865-compatible",
        "reference_compared": str(args.reference.resolve()) if args.reference else None,
        "maximum_projection_error_px": maximum_projection_error,
        "mano_model_by_side": mano_model_by_side,
        "maximum_mano_vertex_replay_error_m": maximum_mano_vertex_error,
        "maximum_mano_joint_replay_error_m": maximum_mano_joint_error,
        "side_counts_in_validated_subset": {
            "left": side_counts[0], "right": side_counts[1],
        },
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
