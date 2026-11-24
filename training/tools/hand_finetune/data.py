"""Training hand crops and frame-level deployment references from native KB exports."""
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .boxes import (FrameKey, apply_offset, box_array, box_iou_matrix, load_proposal_cache,
                    match_box_ious, measure_offset)
from .crop_geometry import PerspectiveCrop, _native_pixels, crop_camera, rgb_chw
from .splits import recording_source_paths


def _recording_id(row, strict):
    if "recording_id" in row:
        return row["recording_id"]
    if strict:
        raise ValueError("Include an explicit recording_id; source_group and export-local IDs can repeat")
    if isinstance(row.get("source_group"), str) and row["source_group"].strip():
        # Historical unversioned FAH and GEM reject overlaps by (source_group, sync_index).
        return row["source_group"]
    if isinstance(row.get("source_recording_id"), str) and row["source_recording_id"].strip():
        return row["source_recording_id"]
    for field, path in recording_source_paths(row):
        return f"{field}:{path}"
    for field in ("source_optimized_fit", "source_dataset_id", "source_dataset"):
        value = row.get(field)
        if value is not None and str(value).strip():
            return f"{field}:{value}"
    raise ValueError("Expected recording identity or legacy source_group")


def _check_geometry_units(source):
    for field in ("units", "length_unit", "geometry_unit", "units_3d"):
        if field not in source:
            continue
        value = source[field]
        units = [value]
        if isinstance(value, dict):
            units = [value[k] for k in ("joints_3d", "vertices", "trans") if k in value]
        if any(str(unit).lower() not in ("m", "meter", "meters", "metre", "metres") for unit in units):
            raise ValueError(f"Expected metres for physical points in {field}")


class FrameManifest:
    """Index both hands by their shared native image without reading any label pixels or targets."""
    def __init__(self, root, split, *, strict=True):
        self.root = Path(root)
        self.rows = []
        self.by_frame = {}
        self.row_keys = []
        samples = set()
        path = self.root / f"{split}.jsonl"
        with path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, 1):
                try:
                    row = json.loads(line)
                    if (row.get("stored_image_horizontally_flipped", False)
                            or row.get("image_space", "native_raw") != "native_raw"
                            or row.get("camera_model", "KB") != "KB"
                            or "rectified" in str(row.get("stored_image_space", ""))):
                        raise ValueError(f"Expected physical native image in {row['image']}")
                    if not isinstance(row.get("label"), str) or not row["label"].strip():
                        raise ValueError("Expected a hand label path")
                    fields = dict(row, recording_id=_recording_id(row, strict))
                    if not strict:
                        fields.setdefault("camera", "legacy")
                        fields["sync_index"] = int(row["sync_index"])
                    key = FrameKey.from_row(fields)
                    if strict:
                        sample = (key, row["label"])
                        if sample in samples:
                            raise ValueError(f"Duplicate hand label for {key}: {row['label']}")
                        samples.add(sample)
                except (KeyError, AttributeError, ValueError, TypeError) as exc:
                    raise ValueError(f"{path}:{line_number}: {exc}") from exc
                self.by_frame.setdefault(key, []).append(len(self.rows))
                self.rows.append(row)
                self.row_keys.append(key)
        if not self.rows:
            raise ValueError(f"Empty {split} split in {root}")
        self.keys = list(self.by_frame)
        self.groups = {(k.recording_id, k.sync_index) for k in self.keys}
        self.recordings = {k.recording_id for k in self.keys}

    def read_image(self, key):
        path = self.root / key.image
        image = cv2.imread(str(path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        if image is None:
            raise OSError(f"Could not read {path}")
        return image

    def read_label(self, index):
        row = self.rows[index]
        path = self.root / row["label"]
        label = np.load(path, allow_pickle=True).item()
        try:
            if (label["camera_model"], label["image_space"]) != ("KB", "native_raw"):
                raise ValueError("Expected native KB label")
            for source in (row, label):
                if source.get("geometry_space", "physical_side") != "physical_side":
                    raise ValueError("Expected physical_side geometry")
                _check_geometry_units(source)
            side = np.asarray(label["side"])
            if side.size != 1 or side.item() not in (0, 1):
                raise ValueError("Expected physical side 0 or 1")
            side = int(side.item())
            if "side" in row and row["side"] != side:
                raise ValueError("Manifest side disagrees with label")
            for source in (row, label):
                declared = source.get("physical_side", side)
                if declared not in (side, ("left", "right")[side]):
                    raise ValueError("physical_side disagrees with side")
            K = np.asarray(label["K"], dtype=np.float64)
            distortion = np.asarray(label["distortion"], dtype=np.float64).reshape(4)
            if (K.shape != (3, 3) or not np.isfinite(K).all() or not np.isfinite(distortion).all()
                    or min(K[0, 0], K[1, 1]) <= 0 or not np.allclose(K[2], [0, 0, 1])
                    or not np.allclose([K[0, 1], K[1, 0]], 0)):
                raise ValueError("Expected finite zero-skew KB4 intrinsics")
            joints = np.asarray(label["joints_3d"], dtype=np.float64)
            vertices = np.asarray(label["vertices"], dtype=np.float64)
            if joints.shape != (21, 3) or vertices.shape != (778, 3):
                raise ValueError("Expected 21 joints and 778 MANO vertices")
            trans = np.asarray(label["trans"], dtype=np.float64).reshape(3)
            joints, vertices = joints + trans, vertices + trans
            if not np.isfinite(joints).all() or not np.isfinite(vertices).all():
                raise ValueError("Nonfinite physical camera-space points")
            hand = dict(bbox=box_array([label["bbox"]])[0], side=side, joints=joints, vertices=vertices, row=row)
        except (KeyError, ValueError, TypeError) as exc:
            raise ValueError(f"Invalid label {path}: {exc}") from exc
        return hand, K, distortion

    def read_frame_labels(self, indices):
        first, frame_K, frame_D = self.read_label(indices[0])
        hands = [first]
        for index in indices[1:]:
            hand, K, distortion = self.read_label(index)
            if not np.array_equal(frame_K, K) or not np.array_equal(frame_D, distortion):
                raise ValueError(f"Hand labels specify conflicting calibration for {self.row_keys[index]}")
            hands.append(hand)
        return hands, frame_K, frame_D


class EvaluationFrames(Dataset):
    """One original BGR image and all its physical-camera hand references per item."""
    def __init__(self, root, split):
        self.manifest = FrameManifest(root, split)
        self.keys = self.manifest.keys
        self.groups, self.recordings = self.manifest.groups, self.manifest.recordings

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        key = self.keys[index]
        hands, K, distortion = self.manifest.read_frame_labels(self.manifest.by_frame[key])
        return dict(key=key, image=self.manifest.read_image(key), K=K, distortion=distortion, hands=hands)


class HandCrops(Dataset):
    def __init__(self, root, split, size, canonical_side, box_augmentation=None):
        self.root, self.size, self.side = Path(root), size, canonical_side
        if box_augmentation is not None and split != "train":
            raise ValueError("Box augmentation and offset fitting are restricted to the train split")
        self.manifest = FrameManifest(root, split, strict=box_augmentation is not None)
        self.rows = [(r["image"], r["label"]) for r in self.manifest.rows]
        self.groups, self.recordings = self.manifest.groups, self.manifest.recordings
        self.augmentation = box_augmentation
        if box_augmentation is not None:
            self._prepare_boxes()

    def _prepare_boxes(self):
        cfg = self.augmentation
        self.matched_boxes = {}
        measured = []
        if cfg.cache is not None:
            cache = load_proposal_cache(cfg.cache, self.manifest.keys)
            for key in sorted(self.manifest.keys):
                indices = sorted(self.manifest.by_frame[key], key=lambda i: self.rows[i][1])
                hands, K, distortion = self.manifest.read_frame_labels(indices)
                gt = [hand["bbox"] for hand in hands]
                proposals = [p.bbox for p in cache[key]]
                iou = box_iou_matrix(gt, proposals)
                for g, hand in enumerate(hands):
                    # Validate GT independently. Only candidate-specific failures remove an edge.
                    self._crop_targets(gt[g], hand, K, distortion)
                    for p in np.flatnonzero(iou[g] >= cfg.min_iou):
                        try:
                            self._crop_targets(proposals[p], hand, K, distortion)
                        except ValueError:
                            iou[g, p] = 0.
                for g, p, _ in match_box_ious(iou, cfg.min_iou):
                    self.matched_boxes[indices[g]] = proposals[p]
                    measured.append(measure_offset(gt[g], proposals[p]))
        self.offset_samples = cfg.offsets(measured)
        # Shared memory lets persistent DataLoader workers see changes before the next iterator.
        self._epoch = torch.zeros((), dtype=torch.int64).share_memory_()
        self._sample_seeds = []
        for key, (_, label_path) in zip(self.manifest.row_keys, self.rows):
            identity = json.dumps([key, label_path], ensure_ascii=False).encode()
            self._sample_seeds.append(int.from_bytes(hashlib.blake2b(identity, digest_size=16).digest(), "big"))

    def set_epoch(self, epoch):
        if self.augmentation is not None:
            if type(epoch) is not int or epoch < 0:
                raise ValueError("epoch must be a non-negative integer")
            self._epoch.fill_(epoch)

    def _crop_targets(self, box, hand, K, distortion):
        crop = PerspectiveCrop.from_box(box, K, distortion, self.size)
        points = crop.view_points(np.concatenate((hand["joints"], hand["vertices"])), hand["side"] != self.side)
        if not np.isfinite(points).all() or np.any(points[:, 2] <= 0):
            raise ValueError(f"Invalid camera-space training target in {hand['row']['label']} for box {box}")
        return crop, points

    def _input_view(self, index, hand, K, distortion, image_shape):
        gt = hand["bbox"]
        gt_view = self._crop_targets(gt, hand, K, distortion)
        cfg = self.augmentation
        if cfg is None:
            return gt_view
        rng = np.random.default_rng([cfg.seed, self._epoch.item(), self._sample_seeds[index]])
        if rng.random() >= cfg.detector_probability:
            return gt_view
        if index in self.matched_boxes:
            return self._crop_targets(self.matched_boxes[index], hand, K, distortion)
        height, width = image_shape[:2]
        pool_size = len(self.offset_samples)
        # A real offset can leave a different camera's invertible KB domain even after clipping.
        for sample in rng.choice(pool_size, size=min(pool_size, 4), replace=False):
            try:
                jitter = apply_offset(gt, self.offset_samples[sample])
                box = np.clip(jitter, 0, [width, height, width, height])
                return self._crop_targets(box, hand, K, distortion)
            except ValueError:
                continue
        return gt_view

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        image = self.manifest.read_image(self.manifest.row_keys[index])
        hand, K, distortion = self.manifest.read_label(index)
        crop, points = self._input_view(index, hand, K, distortion, image.shape)
        mirror = hand["side"] != self.side
        rgb = rgb_chw(crop.rectify(image), mirror)
        return {"rgb": torch.from_numpy(rgb), "points": torch.from_numpy(points.astype(np.float32)),
                "K": torch.from_numpy(crop.K.astype(np.float32))}


def project(points, K):
    projected = points @ K.transpose(1, 2)
    return projected[..., :2] / projected[..., 2:]


def project_native(points, K, distortion):
    """Project physical-camera points to KB pixels, including valid rays beyond 90 degrees."""
    points = np.asarray(points, dtype=np.float64)
    pixels = _native_pixels(points, np.asarray(K), np.asarray(distortion).reshape(4))
    return np.where(np.any(points != 0, axis=-1, keepdims=True), pixels, np.nan)
