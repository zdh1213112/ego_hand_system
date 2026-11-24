"""Dataset reader for glove labels stored in ``datasets/dataset``.

Each row in ``index.jsonl`` describes one monocular hand crop.  The label file
contains the fitted MANO mesh in the source camera frame.  This loader derives
the 21 image-space joints from ``joints_3d + trans`` and ``K`` and applies the
same crop/left-hand canonicalization used by WiLoR inference.

MANO fields always have a fixed tensor structure.  Their masks are controlled
by ``dataset_profile.json``, so a later right-canonical dataset can enable MANO
losses without changing this loader or the training loop.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .utils import expand_to_aspect_ratio, generate_image_patch_cv2


FrameGroupKey = tuple[str, Any, str]


def load_manifest_rows(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Manifest does not exist: {source}")
    return [
        json.loads(line)
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _atomic_write_text(path: str | Path, payload: str) -> None:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        handle.write(payload)
        temporary = Path(handle.name)
    try:
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def write_manifest_rows(path: str | Path, rows: list[dict[str, Any]]) -> None:
    payload = "".join(
        json.dumps(row, separators=(",", ":")) + "\n" for row in rows
    )
    _atomic_write_text(path, payload)


def source_group_key(row: Mapping[str, Any]) -> str:
    """Return a stable recording identity when the manifest provides one."""
    explicit = row.get("source_group")
    if explicit is not None and str(explicit):
        return str(explicit)
    for field in ("source_dataset_id", "source_dataset", "source_experiment"):
        value = row.get(field)
        if value is not None and str(value):
            return f"{field}:{value}"
    return ""


def frame_group_key(row: Mapping[str, Any]) -> FrameGroupKey:
    """Identify all hands that belong to the same recording/camera frame."""
    source_group = source_group_key(row)
    camera = str(row.get("camera", ""))
    sync_index = int(row.get("sync_index", -1))
    if sync_index >= 0:
        return source_group, sync_index, camera
    image = str(row.get("image", row.get("image_path", "")))
    return source_group, image, camera


def _bbox_size_summary(
    dataset_root: Path,
    rows: list[dict[str, Any]],
) -> dict[str, float] | None:
    """Summarize label bbox sizes when the manifest contains label paths."""
    sizes: list[float] = []
    for row in rows:
        label_value = row.get("label")
        if not label_value:
            return None
        label_path = Path(label_value).expanduser()
        if not label_path.is_absolute():
            label_path = dataset_root / label_path
        if not label_path.is_file():
            return None
        try:
            label = np.load(label_path, allow_pickle=True).item()
            bbox = np.asarray(label["bbox"], dtype=np.float32).reshape(4)
        except (KeyError, OSError, TypeError, ValueError):
            return None
        size = float(np.max(bbox[2:] - bbox[:2]))
        if not np.isfinite(size) or size <= 0:
            return None
        sizes.append(size)
    if not sizes:
        return None
    values = np.asarray(sizes, dtype=np.float64)
    return {
        "minimum": float(values.min()),
        "q25": float(np.quantile(values, 0.25)),
        "median": float(np.median(values)),
        "q75": float(np.quantile(values, 0.75)),
        "maximum": float(values.max()),
    }


def make_splits(
    root: str | Path,
    train_ratio: float = 0.90,
    split_seed: int = 42,
    group_by_source: bool = False,
    holdout_by_time: bool = False,
) -> dict[str, Any]:
    """Create deterministic train/validation manifests, optionally by recording."""
    dataset_root = Path(root).expanduser().resolve()
    rows = load_manifest_rows(dataset_root / "index.jsonl")
    if len(rows) < 2:
        raise ValueError("At least two index rows are required for train/val splits")
    if not 0.0 < float(train_ratio) < 1.0:
        raise ValueError(f"train_ratio must be between 0 and 1, got {train_ratio}")

    if holdout_by_time:
        # Hold out the last fraction of each recording's timeline, so every
        # recording contributes to both train and val without frame leakage.
        by_group: dict[str, list[tuple[int, int]]] = {}
        for position, row in enumerate(rows):
            # Merged datasets number each input recording; prefer that over
            # source_group, which is not unique across recordings exported by
            # the one-command pipeline (e.g. "processed__right").
            group = (f"dataset{row['source_dataset_id']}" if "source_dataset_id" in row
                     else source_group_key(row))
            if not group:
                raise ValueError(f"row {position} lacks source_group")
            by_group.setdefault(group, []).append((int(row["sync_index"]), position))
        positions = {"train": [], "val": []}
        for group, items in by_group.items():
            syncs = sorted({sync for sync, _ in items})
            cut = syncs[min(len(syncs) - 1, max(1, int(round(len(syncs) * float(train_ratio)))))]
            for sync, position in items:
                positions["train" if sync < cut else "val"].append(position)
        positions = {name: sorted(values) for name, values in positions.items()}
    elif group_by_source:
        groups: dict[str, list[int]] = {}
        for position, row in enumerate(rows):
            group = source_group_key(row)
            if not group:
                raise ValueError(f"row {position} lacks source_group")
            groups.setdefault(group, []).append(position)
        if len(groups) < 2:
            raise ValueError("source-group split requires at least two recording sessions")
        shuffled = np.random.default_rng(int(split_seed)).permutation(sorted(groups))
        train_groups: set[str] = set()
        train_count = 0
        target_count = len(rows) * float(train_ratio)
        for group in shuffled[:-1]:
            if not train_groups or train_count < target_count:
                train_groups.add(str(group))
                train_count += len(groups[str(group)])
        positions = {
            "train": sorted(position for group in train_groups for position in groups[group]),
            "val": sorted(position for group in groups if group not in train_groups
                          for position in groups[group]),
        }
    else:
        permutation = np.random.default_rng(int(split_seed)).permutation(len(rows))
        train_count = min(
            len(rows) - 1,
            max(1, int(round(len(rows) * float(train_ratio)))),
        )
        positions = {
            "train": sorted(int(value) for value in permutation[:train_count]),
            "val": sorted(int(value) for value in permutation[train_count:]),
        }
    split_rows = {
        name: [rows[position] for position in selected]
        for name, selected in positions.items()
    }

    metadata: dict[str, Any] = {
        "source": str(dataset_root / "index.jsonl"),
        "strategy": "time_holdout" if holdout_by_time else "source_group" if group_by_source else "random_sample",
        "split_seed": int(split_seed),
        "ratios": {
            "train": float(train_ratio),
            "val": float(1.0 - train_ratio),
        },
        "source_positions": positions,
        "sample_ids": {
            name: [str(row.get("index", position)) for position, row in zip(
                positions[name], values
            )]
            for name, values in split_rows.items()
        },
        "samples": {name: len(values) for name, values in split_rows.items()},
        "side_counts": {
            name: dict(Counter(str(int(row["side"])) for row in values))
            for name, values in split_rows.items()
        },
        "camera_counts": {
            name: dict(Counter(str(row.get("camera", "")) for row in values))
            for name, values in split_rows.items()
        },
    }
    bbox_summary = {
        name: _bbox_size_summary(dataset_root, values)
        for name, values in split_rows.items()
    }
    if all(value is not None for value in bbox_summary.values()):
        metadata["bbox_size_px"] = bbox_summary

    assigned = positions["train"] + positions["val"]
    if len(assigned) != len(rows) or len(set(assigned)) != len(rows):
        raise AssertionError("Split does not cover each source row exactly once")

    for name, values in split_rows.items():
        write_manifest_rows(dataset_root / f"{name}.jsonl", values)
    legacy_test_manifest = dataset_root / "test.jsonl"
    removed_legacy_test = legacy_test_manifest.is_file()
    legacy_test_manifest.unlink(missing_ok=True)
    metadata["removed_legacy_test_manifest"] = removed_legacy_test
    splits_path = dataset_root / "splits.json"
    _atomic_write_text(splits_path, json.dumps(metadata, indent=2) + "\n")
    return metadata


class GloveDataset(Dataset):
    """Read one monocular hand sample per JSONL row.

    The persisted annotations stay in original image/camera coordinates.  The
    returned tensors are the coordinates consumed by ``WiLoR.compute_loss``:

    * ``img``: normalized RGB tensor ``[3, 256, 256]``;
    * ``keypoints_2d``: crop-normalized ``[21, 3]`` (x, y, confidence);
    * ``keypoints_3d``: canonical root-relative-ready ``[21, 4]``;
    * empty MANO supervision dictionaries for the first training stage.
    """

    def __init__(
        self,
        manifest: str | Path,
        image_size: int = 256,
        rescale_factor: float = 2.0,
        confidence: float = 1.0,
        max_samples: int | None = None,
        max_frames: int | None = None,
        marker_inputs: Mapping[str, str | Path] | None = None,
        camera_model: str = "physical_perspective",
        use_surface_markers: bool = False,
        crop_jitter: float = 0.,
    ) -> None:
        self.manifest = Path(manifest).expanduser().resolve()
        self.camera_model = camera_model
        if camera_model == 'physical_kb' and marker_inputs:
            raise ValueError('Native marker supervision needs native marker archives; legacy rectified marker inputs are unsupported')
        if not self.manifest.is_file():
            raise FileNotFoundError(f"Manifest does not exist: {self.manifest}")
        self.root = self.manifest.parent
        profile_path = self.root / "dataset_profile.json"
        self.profile = (
            json.loads(profile_path.read_text(encoding="utf-8"))
            if profile_path.is_file() else None
        )
        supervision = (self.profile or {}).get("supervision", {})
        self.mano_available = {
            name: bool(supervision.get(name, {}).get("available", False))
            for name in ("global_orient", "hand_pose", "betas")
        }
        self.rows: List[dict] = load_manifest_rows(self.manifest)
        if max_samples is not None and max_frames is not None:
            raise ValueError("max_samples and max_frames are mutually exclusive")
        if max_samples is not None:
            self.rows = self.rows[: int(max_samples)]
        if max_frames is not None:
            if max_frames < 1:
                raise ValueError("max_frames must be at least 1")
            ordered_groups = list(dict.fromkeys(
                frame_group_key(row) for row in self.rows
            ))
            selected_groups = set(ordered_groups[: int(max_frames)])
            self.rows = [
                row for row in self.rows
                if frame_group_key(row) in selected_groups
            ]
        if not self.rows:
            raise ValueError(f"Manifest is empty: {self.manifest}")
        self.image_size = int(image_size)
        if not 0 <= crop_jitter < 1:
            raise ValueError("crop_jitter must be in [0, 1)")
        self.crop_jitter = crop_jitter
        self.has_surface_only_rows = any(row.get("measured_surface_only", False) for row in self.rows)
        self.surface_targets = None
        if use_surface_markers:
            if camera_model != 'physical_kb':
                raise ValueError('Recorded surface targets require native camera geometry')
            from .nokov_surface import SurfaceMarkerTargets
            self.surface_targets = SurfaceMarkerTargets(self.root, self.profile, {source_group_key(r) for r in self.rows})
        self.rescale_factor = float(rescale_factor)
        self.confidence = float(confidence)
        self.mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
        self.marker_inputs: dict[str, dict[str, np.ndarray]] = {}
        for source, path in (marker_inputs or {}).items():
            with np.load(Path(path).expanduser().resolve()) as archive:
                self.marker_inputs[str(source)] = {
                    name: archive[name] for name in (
                        "left_rectified_px", "right_rectified_px",
                        "left_rectified_valid", "right_rectified_valid",
                        "positions_left_camera_m", "R1", "P1", "P2",
                    ) if name in archive
                }
            item = self.marker_inputs[str(source)]
            if "R1" not in item:
                # The aligned marker archive uses descriptive calibration keys.
                with np.load(Path(path).expanduser().resolve()) as archive:
                    item["R1"] = archive["left_to_rectified_rotation"]
                    item["P1"] = archive["projection_left_rectified"]
                    item["P2"] = archive["projection_right_rectified"]
        self.marker_filtered_count = 0
        if self.marker_inputs:
            missing = {str(row.get("source_experiment", "")) for row in self.rows} - set(self.marker_inputs)
            if missing:
                raise ValueError(f"No marker archive for source experiments: {sorted(missing)}")
            before = len(self.rows)
            self.rows = [row for row in self.rows if self.marker_inputs[
                str(row["source_experiment"])
            ][
                "left_rectified_valid" if str(row["camera"]) == "camera0"
                else "right_rectified_valid"
            ][int(row["side"]), int(row["sync_index"])].all()]
            self.marker_filtered_count = before - len(self.rows)
            if not self.rows:
                raise ValueError("No samples have complete marker targets")

    def __len__(self) -> int:
        return len(self.rows)

    def _resolve(self, value: str | Path) -> Path:
        path = Path(value).expanduser()
        return path if path.is_absolute() else self.root / path

    @staticmethod
    def _as_numpy(value: Any) -> np.ndarray:
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    @staticmethod
    def _load_label(path: Path, allow_native: bool = False) -> Mapping[str, Any]:
        try:
            value = np.load(path, allow_pickle=True).item()
        except Exception as exc:  # pragma: no cover - error context matters
            raise RuntimeError(f"Could not load label {path}: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"Label must contain a dict: {path}")
        if value.get("image_space", "physical_rectified") != "physical_rectified" and not allow_native:
            raise ValueError(
                f"{path}: native camera labels require a camera-aware training "
                "adapter; the current loader uses rectified perspective projection"
            )
        required = {"bbox", "joints_3d", "trans", "K", "side"}
        missing = required.difference(value)
        if missing:
            raise ValueError(f"Label {path} is missing fields: {sorted(missing)}")
        return value

    @staticmethod
    def project_joints(joints_3d: np.ndarray, trans: np.ndarray,
                       K: np.ndarray) -> np.ndarray:
        """Project 21 root-relative/model-space joints to image pixels."""
        joints_camera = joints_3d + trans.reshape(1, 3)
        projected = joints_camera @ K.T
        depth = projected[:, 2:3]
        if np.any(depth <= 0):
            raise ValueError("A label contains a non-positive camera-space depth")
        return projected[:, :2] / depth

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.rows[index]
        image_path = self._resolve(row["image"])
        label_path = self._resolve(row["label"])
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise IOError(f"Could not read image: {image_path}")
        height, width = image.shape[:2]

        label = self._load_label(label_path, allow_native=self.camera_model == 'physical_kb')
        native = label.get('image_space') == 'native_raw'
        if native != (self.camera_model == 'physical_kb'):
            raise ValueError(f'{label_path}: image space and configured camera model disagree')
        distortion = np.asarray(label.get('distortion', np.zeros(4)), dtype=np.float64)
        if native and (label.get('camera_model') != 'KB' or distortion.shape != (4,) or not np.isfinite(distortion).all()):
            raise ValueError(f'{label_path}: require finite KB camera parameters')
        bbox = self._as_numpy(row.get("input_bbox_xyxy", label["bbox"])).astype(np.float32).reshape(4)
        joints_3d = self._as_numpy(label["joints_3d"]).astype(np.float32).reshape(21, 3)
        trans = self._as_numpy(label["trans"]).astype(np.float32).reshape(3)
        K = self._as_numpy(label["K"]).astype(np.float32).reshape(3, 3)
        side = int(np.asarray(label["side"]).item())
        if side not in (0, 1):
            raise ValueError(f"side must be 0 or 1, got {side} in {label_path}")
        if int(row.get("side", side)) != side:
            raise ValueError(f"Index/label side mismatch for {label_path}")

        # Match ViTDetDataset: expand the detector box and make the crop square.
        center = (bbox[:2] + bbox[2:]) * 0.5
        box_wh = bbox[2:] - bbox[:2]
        if not np.isfinite(bbox).all() or np.any(box_wh <= 0):
            raise ValueError(f"Invalid bbox in {label_path}: {bbox.tolist()}")
        if self.crop_jitter:
            shift = np.random.uniform(-self.crop_jitter, self.crop_jitter, size=2)
            scale = np.exp(np.random.uniform(-self.crop_jitter, self.crop_jitter))
            center = center + shift * box_wh
            half = box_wh * scale / 2
            bounds = np.array([width - 1, height - 1], dtype=np.float32)
            low = np.clip(center - half, 0, bounds - 1)
            high = np.clip(center + half, low + 1, bounds)
            bbox = np.concatenate([low, high]).astype(np.float32)
            center = (bbox[:2] + bbox[2:]) / 2
            box_wh = bbox[2:] - bbox[:2]
        scaled_wh = self.rescale_factor * box_wh
        crop_wh = expand_to_aspect_ratio(
            scaled_wh, target_aspect_ratio=[192, 256])
        box_size = float(np.max(crop_wh))

        is_left = side == 0
        crop, transform = generate_image_patch_cv2(
            image,
            float(center[0]),
            float(center[1]),
            box_size,
            box_size,
            self.image_size,
            self.image_size,
            is_left,
            1.0,
            0,
            border_mode=cv2.BORDER_CONSTANT,
        )

        # The label's 778x2 ``joints_2d`` field is actually vertex projection.
        # Derive the required 21 joint pixels from the 3D joints instead.
        if native:
            points = (joints_3d + trans).astype(np.float64)
            if not np.isfinite(points).all() or (points[:, 2] <= 0).any():
                raise ValueError('Invalid native-camera geometry')
            joints_2d = cv2.fisheye.projectPoints(points[:, None], np.zeros(3), np.zeros(3),
                K.astype(np.float64), distortion)[0][:, 0].astype(np.float32)
        else:
            joints_2d = self.project_joints(joints_3d, trans, K)
        source_visible = (
            np.isfinite(joints_2d).all(axis=1)
            & (joints_2d[:, 0] >= 0.0)
            & (joints_2d[:, 0] < float(width))
            & (joints_2d[:, 1] >= 0.0)
            & (joints_2d[:, 1] < float(height))
        )
        if is_left:
            joints_2d[:, 0] = width - joints_2d[:, 0] - 1.0
            joints_3d = joints_3d.copy()
            joints_3d[:, 0] *= -1.0

        joints_2d_h = np.concatenate(
            [joints_2d, np.ones((21, 1), dtype=np.float32)], axis=1)
        joints_2d_crop_pixels = np.asarray(
            [transform @ point for point in joints_2d_h], dtype=np.float32)
        crop_visible = (
            np.isfinite(joints_2d_crop_pixels[:, :2]).all(axis=1)
            & (joints_2d_crop_pixels[:, 0] >= 0.0)
            & (joints_2d_crop_pixels[:, 0] < float(self.image_size))
            & (joints_2d_crop_pixels[:, 1] >= 0.0)
            & (joints_2d_crop_pixels[:, 1] < float(self.image_size))
        )
        visibility = (source_visible & crop_visible).astype(np.float32)
        joints_2d_crop = (
            joints_2d_crop_pixels[:, :2] / float(self.image_size) - 0.5
        )
        keypoints_2d = np.concatenate(
            [joints_2d_crop,
             (visibility * self.confidence).reshape(21, 1)], axis=1)
        keypoints_3d = np.concatenate(
            [joints_3d,
             np.full((21, 1), self.confidence, dtype=np.float32)], axis=1)

        rgb = crop[:, :, ::-1].astype(np.float32) / 255.0
        rgb = (rgb - self.mean[None, None, :]) / self.std[None, None, :]
        image_tensor = torch.from_numpy(rgb.transpose(2, 0, 1).copy()).float()

        raw_mano = label.get("mano") if isinstance(label.get("mano"), dict) else {}
        mano_shapes = {
            "global_orient": (1, 3, 3),
            "hand_pose": (15, 3, 3),
            "betas": (10,),
        }
        mano_params: dict[str, torch.Tensor] = {}
        has_mano_params: dict[str, torch.Tensor] = {}
        for name, shape in mano_shapes.items():
            raw_value = raw_mano.get(name)
            valid = self.mano_available[name] and raw_value is not None
            value = (
                self._as_numpy(raw_value).astype(np.float32).reshape(shape)
                if valid else np.zeros(shape, dtype=np.float32)
            )
            mano_params[name] = torch.from_numpy(value).float()
            has_mano_params[name] = torch.tensor(valid, dtype=torch.bool)

        sample = {
            "img": image_tensor,
            "keypoints_2d": torch.from_numpy(keypoints_2d).float(),
            "keypoints_3d": torch.from_numpy(keypoints_3d).float(),
            # Geometry metadata is ignored by the loss, but lets the training
            # script map predictions back to the original image for test-set
            # visualization without rerunning the crop logic.
            "crop_transform": torch.from_numpy(transform).float(),
            "crop_box_size": torch.tensor(box_size, dtype=torch.float32),
            "camera_K": torch.from_numpy(K).float(),
            "camera_distortion": torch.from_numpy(distortion.astype(np.float32)),
            "camera_root_m": torch.from_numpy(
                self._as_numpy(label["joints_3d"]).astype(np.float32).reshape(21, 3)[0]
                + trans
            ).float(),
            "bbox_xyxy": torch.from_numpy(bbox).float(),
            "image_size": torch.tensor([width, height], dtype=torch.float32),
            "mano_params": mano_params,
            "has_mano_params": has_mano_params,
            "mano_params_is_axis_angle": {
                name: torch.tensor(False, dtype=torch.bool)
                for name in mano_shapes
            },
            "sample_id": str(row.get("index", index)),
            "image_path": str(image_path),
            "source_group": source_group_key(row),
            "sync_index": int(row.get("sync_index", -1)),
            "source_frame_index": int(row.get("source_frame_index", -1)),
            "camera": str(row.get("camera", "")),
            "side": side,
        }
        source_experiment = str(row.get("source_experiment", ""))
        if self.surface_targets is not None:
            sample.update(self.surface_targets.sample(
                source_group_key(row), side, int(row["sync_index"]), row["camera"]
            ))
        if self.has_surface_only_rows:
            trusted = not row.get("measured_surface_only", False)
            sample["geometry_supervised"] = trusted
            if not trusted:
                sample["keypoints_2d"][:, -1] = 0
                sample["keypoints_3d"][:, -1] = 0
                for flag in sample["has_mano_params"].values():
                    flag.fill_(False)
        if source_experiment in self.marker_inputs:
            marker = self.marker_inputs[source_experiment]
            sync = int(row["sync_index"])
            key = "left" if str(row["camera"]) == "camera0" else "right"
            pixels = marker[f"{key}_rectified_px"][side, sync].astype(np.float32).copy()
            valid = marker[f"{key}_rectified_valid"][side, sync].astype(bool).copy()
            if pixels.shape != (21, 2) or valid.shape != (21,) or not valid.all():
                raise ValueError(f"Incomplete marker target at {source_experiment} sync={sync}")
            physical = marker["positions_left_camera_m"][side, sync].astype(np.float32)
            physical = physical @ marker["R1"].astype(np.float32).T
            if key == "right":
                offset = np.linalg.solve(
                    marker["P1"][:, :3], marker["P2"][:, 3]
                ).astype(np.float32)
                physical = physical + offset[None]
            root_marker = physical[0].copy()
            local_marker = physical - root_marker[None]
            if is_left:
                local_marker[:, 0] *= -1.0
                pixels[:, 0] = width - pixels[:, 0] - 1.0
            pixel_h = np.concatenate((pixels, np.ones((21, 1), np.float32)), axis=1)
            crop_pixels = pixel_h @ transform.T
            crop_visible = (
                (crop_pixels[:, 0] >= 0) & (crop_pixels[:, 0] < self.image_size)
                & (crop_pixels[:, 1] >= 0) & (crop_pixels[:, 1] < self.image_size)
            )
            confidence = (valid & crop_visible).astype(np.float32) * self.confidence
            sample["marker_keypoints_2d"] = torch.from_numpy(np.concatenate((
                crop_pixels / self.image_size - 0.5,
                confidence[:, None],
            ), axis=1)).float()
            sample["marker_keypoints_3d"] = torch.from_numpy(np.concatenate((
                local_marker, confidence[:, None],
            ), axis=1)).float()
            sample["marker_root_camera_m"] = torch.from_numpy(root_marker).float()
        return sample


def validate_manifest(manifest: str | Path, max_samples: int | None = None) -> dict:
    """Compatibility entry point for the centralized geometry validator."""
    from .glove_validation import validate_manifest as validate
    return validate(manifest, max_samples=max_samples)
