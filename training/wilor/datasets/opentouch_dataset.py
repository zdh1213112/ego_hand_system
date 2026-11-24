"""Prepared OpenTouch dataset with wrist-relative 3D supervision only.

The decoded RGB frames and Rokoko landmarks are joined through the immutable
``(source_file, clip_id, source_frame_index)`` key.  ``sync_index`` is
deliberately ignored because it restarts in independent recordings.
"""

from __future__ import annotations

import json
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from wilor.utils.hand_geometry import palm_scale_normalize_numpy

from .glove_dataset import load_manifest_rows, write_manifest_rows
from .utils import expand_to_aspect_ratio, generate_image_patch_cv2


FrameKey = tuple[str, str, int]


def opentouch_frame_key(row: Mapping[str, Any]) -> FrameKey:
    """Return the only supported cross-file join key for an OpenTouch frame."""
    try:
        return (
            str(row["source_file"]),
            str(row["clip_id"]),
            int(row["source_frame_index"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "OpenTouch rows require source_file, clip_id and source_frame_index"
        ) from exc


def expected_sample_id(key: FrameKey) -> str:
    source_file, clip_id, source_frame_index = key
    return f"{Path(source_file).stem}::{clip_id}::{source_frame_index:06d}"


def _indexed_unique_rows(
    rows: list[dict[str, Any]], source: Path
) -> dict[FrameKey, dict[str, Any]]:
    by_key: dict[FrameKey, dict[str, Any]] = {}
    sample_ids: set[str] = set()
    for line_number, row in enumerate(rows, start=1):
        key = opentouch_frame_key(row)
        sample_id = str(row.get("sample_id", ""))
        if not sample_id:
            raise ValueError(f"{source}:{line_number}: sample_id is required")
        if sample_id != expected_sample_id(key):
            raise ValueError(
                f"{source}:{line_number}: sample_id {sample_id!r} does not "
                f"match frame key {key!r}"
            )
        if key in by_key:
            raise ValueError(f"{source}:{line_number}: duplicate frame key {key!r}")
        if sample_id in sample_ids:
            raise ValueError(
                f"{source}:{line_number}: duplicate sample_id {sample_id!r}"
            )
        by_key[key] = row
        sample_ids.add(sample_id)
    return by_key


def _validated_bbox(
    value: Any,
    image_path: Path,
    sample_id: str,
) -> list[float] | None:
    if value is None:
        return None
    try:
        bbox = np.asarray(value, dtype=np.float32).reshape(4)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{sample_id}: bbox_xyxy must be null or four numbers") from exc
    if not np.isfinite(bbox).all():
        raise ValueError(f"{sample_id}: bbox_xyxy contains NaN or Inf")
    if bbox[2] - bbox[0] < 2 or bbox[3] - bbox[1] < 2:
        raise ValueError(f"{sample_id}: bbox_xyxy must have positive area")
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise IOError(f"Could not read image for bbox validation: {image_path}")
    height, width = image.shape[:2]
    if bbox[0] < 0 or bbox[1] < 0 or bbox[2] > width or bbox[3] > height:
        raise ValueError(
            f"{sample_id}: bbox_xyxy {bbox.tolist()} leaves the "
            f"{width}x{height} image; only fully visible hands are valid"
        )
    return [float(item) for item in bbox]


def initialize_empty_bbox_annotations(
    decoded_root: str | Path,
    destination: str | Path,
    *,
    overwrite: bool = False,
) -> int:
    """Create the annotation worklist without inventing any usable bbox."""
    decoded = Path(decoded_root).expanduser().resolve()
    target = Path(destination).expanduser().resolve()
    if target.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite bbox annotations: {target}; pass --force to replace"
        )
    frames_path = decoded / "frames.jsonl"
    rows = load_manifest_rows(frames_path)
    _indexed_unique_rows(rows, frames_path)
    annotations = [
        {
            "sample_id": row["sample_id"],
            "source_file": row["source_file"],
            "clip_id": row["clip_id"],
            "source_frame_index": int(row["source_frame_index"]),
            "bbox_xyxy": None,
        }
        for row in rows
    ]
    write_manifest_rows(target, annotations)
    return len(annotations)


def prepare_opentouch_dataset(
    decoded_root: str | Path,
    bbox_annotations: str | Path,
    output_root: str | Path,
    *,
    sample_fps: float = 10.0,
    source_fps: float = 30.0,
    val_ratio: float = 0.1,
    split_seed: int = 42,
) -> dict[str, Any]:
    """Join bbox annotations and decoded data, then make clip-disjoint splits.

    The bbox annotation manifest must cover every decoded frame.  A row with a
    null bbox is an explicit unusable sample; it is validated and omitted from
    train/validation manifests.
    """
    decoded = Path(decoded_root).expanduser().resolve()
    bbox_path = Path(bbox_annotations).expanduser().resolve()
    output = Path(output_root).expanduser().resolve()
    frames_path = decoded / "frames.jsonl"
    frame_rows = load_manifest_rows(frames_path)
    bbox_rows = load_manifest_rows(bbox_path)
    frames = _indexed_unique_rows(frame_rows, frames_path)
    annotations = _indexed_unique_rows(bbox_rows, bbox_path)

    missing = sorted(set(frames).difference(annotations))
    unknown = sorted(set(annotations).difference(frames))
    if missing or unknown:
        raise ValueError(
            "bbox manifest does not exactly cover decoded frames: "
            f"missing={missing[:3]}, unknown={unknown[:3]}"
        )
    if not 0.0 < float(val_ratio) < 1.0:
        raise ValueError(f"val_ratio must be between 0 and 1, got {val_ratio}")
    if sample_fps <= 0 or source_fps <= 0 or sample_fps > source_fps:
        raise ValueError("Require 0 < sample_fps <= source_fps")
    stride = max(1, int(round(float(source_fps) / float(sample_fps))))

    prepared: list[dict[str, Any]] = []
    rejected = 0
    for key, frame in frames.items():
        annotation = annotations[key]
        image_path = decoded / str(frame["image"])
        bbox = _validated_bbox(
            annotation.get("bbox_xyxy"), image_path, str(frame["sample_id"])
        )
        if bbox is None:
            rejected += 1
            continue
        if int(frame["clip_data_index"]) % stride:
            continue
        clip_path = decoded / str(frame["clip_data"])
        if not clip_path.is_file():
            raise FileNotFoundError(f"Missing decoded clip data: {clip_path}")
        row = dict(frame)
        row["bbox_xyxy"] = bbox
        prepared.append(row)

    if not prepared:
        raise ValueError(
            "No usable OpenTouch samples remain; bbox annotations are still empty"
        )
    clip_keys = sorted({(row["source_file"], row["clip_id"]) for row in prepared})
    if len(clip_keys) < 2:
        raise ValueError("At least two usable clips are required for train/val splits")
    permutation = np.random.default_rng(int(split_seed)).permutation(len(clip_keys))
    val_clips_count = min(
        len(clip_keys) - 1,
        max(1, int(round(len(clip_keys) * float(val_ratio)))),
    )
    val_clips = {clip_keys[int(index)] for index in permutation[:val_clips_count]}
    split_rows = {
        "train": [
            row for row in prepared
            if (row["source_file"], row["clip_id"]) not in val_clips
        ],
        "val": [
            row for row in prepared
            if (row["source_file"], row["clip_id"]) in val_clips
        ],
    }
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in split_rows.items():
        write_manifest_rows(output / f"{name}.jsonl", rows)

    supervision = {
        name: {
            "available": name == "local_keypoints_3d",
            "reason": (
                "Rokoko landmarks are normalized in a palm-local frame"
                if name == "local_keypoints_3d"
                else "OpenTouch has no calibrated RGB-aligned supervision"
            ),
        }
        for name in (
            "local_keypoints_3d",
            "keypoints_2d",
            "keypoints_3d",
            "global_orient",
            "hand_pose",
            "betas",
        )
    }
    profile = {
        "schema_version": 1,
        "data_type": "opentouch",
        "decoded_root": str(decoded),
        "bbox_annotations": str(bbox_path),
        "supervision": supervision,
        "samples": {name: len(rows) for name, rows in split_rows.items()},
        "usable_before_sampling": sum(
            annotation.get("bbox_xyxy") is not None
            for annotation in annotations.values()
        ),
        "unusable": rejected,
        "source_fps": float(source_fps),
        "sample_fps": float(sample_fps),
        "sample_stride": stride,
    }
    (output / "dataset_profile.json").write_text(
        json.dumps(profile, indent=2) + "\n", encoding="utf-8"
    )
    split_metadata = {
        "strategy": "clip_disjoint",
        "split_seed": int(split_seed),
        "val_ratio": float(val_ratio),
        "clips": {
            name: sorted({
                f"{row['source_file']}::{row['clip_id']}" for row in rows
            })
            for name, rows in split_rows.items()
        },
        "samples": profile["samples"],
        "source_file_counts": {
            name: dict(Counter(str(row["source_file"]) for row in rows))
            for name, rows in split_rows.items()
        },
    }
    (output / "splits.json").write_text(
        json.dumps(split_metadata, indent=2) + "\n", encoding="utf-8"
    )
    return {"profile": profile, "splits": split_metadata}


class OpenTouchDataset(Dataset):
    """Load bbox-cropped RGB and palm-normalized Rokoko hand landmarks."""

    def __init__(
        self,
        manifest: str | Path,
        image_size: int = 256,
        rescale_factor: float = 2.0,
        max_samples: int | None = None,
        npz_cache_size: int = 8,
    ) -> None:
        self.manifest = Path(manifest).expanduser().resolve()
        self.root = self.manifest.parent
        profile_path = self.root / "dataset_profile.json"
        if not profile_path.is_file():
            raise FileNotFoundError(f"Missing OpenTouch profile: {profile_path}")
        self.profile = json.loads(profile_path.read_text(encoding="utf-8"))
        if self.profile.get("data_type") != "opentouch":
            raise ValueError(f"Not an OpenTouch dataset profile: {profile_path}")
        self.decoded_root = Path(self.profile["decoded_root"]).expanduser().resolve()
        self.rows = load_manifest_rows(self.manifest)
        if max_samples is not None:
            self.rows = self.rows[: int(max_samples)]
        if not self.rows:
            raise ValueError(f"Manifest is empty: {self.manifest}")
        _indexed_unique_rows(self.rows, self.manifest)
        self.image_size = int(image_size)
        self.rescale_factor = float(rescale_factor)
        self.npz_cache_size = max(1, int(npz_cache_size))
        self._landmark_cache: OrderedDict[
            Path, tuple[np.ndarray, np.ndarray]
        ] = OrderedDict()
        self.mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)

    def __len__(self) -> int:
        return len(self.rows)

    def _landmarks(self, row: Mapping[str, Any]) -> np.ndarray:
        clip_path = self.decoded_root / str(row["clip_data"])
        cached = self._landmark_cache.pop(clip_path, None)
        if cached is None:
            with np.load(clip_path, allow_pickle=False) as clip:
                if "right_hand_landmarks" not in clip:
                    raise KeyError(f"{clip_path} has no right_hand_landmarks")
                values = np.asarray(clip["right_hand_landmarks"], dtype=np.float32)
                if "source_frame_indices" not in clip:
                    raise KeyError(f"{clip_path} has no source_frame_indices")
                source_indices = np.asarray(
                    clip["source_frame_indices"], dtype=np.int64
                ).reshape(-1)
            if values.ndim != 3 or values.shape[1:] != (21, 3):
                raise ValueError(
                    f"Invalid right_hand_landmarks shape in {clip_path}: {values.shape}"
                )
            if len(source_indices) != len(values):
                raise ValueError(f"Frame/landmark length mismatch in {clip_path}")
            cached = values, source_indices
        values, source_indices = cached
        self._landmark_cache[clip_path] = cached
        while len(self._landmark_cache) > self.npz_cache_size:
            self._landmark_cache.popitem(last=False)
        index = int(row["clip_data_index"])
        if not 0 <= index < len(values):
            raise IndexError(f"clip_data_index {index} is outside {clip_path}")
        expected_index = int(row["source_frame_index"])
        if int(source_indices[index]) != expected_index:
            raise ValueError(
                f"{row['sample_id']}: clip_data_index maps to source frame "
                f"{int(source_indices[index])}, expected {expected_index}"
            )
        return values[index]

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        if str(row.get("handedness", "right")).lower() != "right":
            raise ValueError(
                f"OpenTouch Stage 1 currently supports right-hand rows only: "
                f"{row['sample_id']}"
            )
        image_path = self.decoded_root / str(row["image"])
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise IOError(f"Could not read image: {image_path}")
        bbox = np.asarray(row["bbox_xyxy"], dtype=np.float32).reshape(4)
        center = (bbox[:2] + bbox[2:]) * 0.5
        box_wh = bbox[2:] - bbox[:2]
        if np.any(box_wh <= 0):
            raise ValueError(f"Invalid bbox for {row['sample_id']}: {bbox.tolist()}")
        crop_wh = expand_to_aspect_ratio(
            self.rescale_factor * box_wh, target_aspect_ratio=[192, 256]
        )
        box_size = float(np.max(crop_wh))
        crop, transform = generate_image_patch_cv2(
            image,
            float(center[0]),
            float(center[1]),
            box_size,
            box_size,
            self.image_size,
            self.image_size,
            False,
            1.0,
            0,
            border_mode=cv2.BORDER_CONSTANT,
        )
        rgb = crop[:, :, ::-1].astype(np.float32) / 255.0
        rgb = (rgb - self.mean[None, None, :]) / self.std[None, None, :]
        landmarks = palm_scale_normalize_numpy(self._landmarks(row)).astype(np.float32)
        return {
            "img": torch.from_numpy(rgb.transpose(2, 0, 1).copy()).float(),
            "local_keypoints_3d": torch.from_numpy(landmarks).float(),
            "has_local_keypoints_3d": torch.tensor(True, dtype=torch.bool),
            "crop_transform": torch.from_numpy(transform).float(),
            "bbox_xyxy": torch.from_numpy(bbox).float(),
            "sample_id": str(row["sample_id"]),
            "source_file": str(row["source_file"]),
            "clip_id": str(row["clip_id"]),
            "source_frame_index": int(row["source_frame_index"]),
            "image_path": str(image_path),
        }
