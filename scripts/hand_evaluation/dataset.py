"""Read native/rectified WiLoR exports without depending on source video paths."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

from ego_hand_inference.camera_models import project_points
from ego_hand_inference.ego_data.calibration import CameraCalibration


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def array(value, shape, name):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    value = np.asarray(value, dtype=np.float64)
    if value.shape != shape or not np.isfinite(value).all():
        raise ValueError(f'{name}: expected finite {shape}, got {value.shape}')
    return value


@dataclass
class Label:
    row: dict
    bbox: np.ndarray
    side: int
    joints: np.ndarray
    vertices: np.ndarray
    pixels: np.ndarray
    valid: np.ndarray
    calibration: CameraCalibration


class LabeledDataset:
    def __init__(self, root, *, cameras=None, image_stride=1, max_images=0):
        root = Path(root).resolve()
        if not (root / 'index.jsonl').exists() and (root / 'merged/index.jsonl').exists():
            root = root / 'merged'
        if image_stride < 1 or max_images < 0:
            raise ValueError('image_stride must be positive; max_images cannot be negative')
        self.root = root
        self.summary = json.loads((root / 'summary.json').read_text())
        if self.summary.get('geometry_space') != 'physical_side':
            raise ValueError('expected physical-side geometry; canonical geometry needs explicit conversion')
        if self.summary.get('image_space') not in ('native_raw', 'physical_rectified'):
            raise ValueError('unsupported image_space')
        self.rows = [json.loads(line) for line in (root / 'index.jsonl').read_text().splitlines() if line.strip()]
        ids, labels, groups = set(), set(), {}
        for row in self.rows:
            if not isinstance(row['index'], int) or row['index'] in ids or row['label'] in labels:
                raise ValueError('duplicate/invalid sample index or label path')
            ids.add(row['index'])
            labels.add(row['label'])
            if row['side'] not in (0, 1) or row.get('physical_side', row['side']) != row['side']:
                raise ValueError('invalid physical hand side')
            if row.get('stored_image_horizontally_flipped', False):
                raise ValueError('evaluation requires original physical-side images')
            group = groups.setdefault(row['image'], [])
            if group:
                first = group[0]
                for key in ('camera', 'sync_index', 'source_frame_index', 'source_group', 'source_dataset_id'):
                    if first.get(key) != row.get(key):
                        raise ValueError(f'inconsistent {key} for shared image {row["image"]}')
                if row['side'] in {item['side'] for item in group}:
                    raise ValueError('duplicate hand side in one image')
            group.append(row)
        if not self.rows:
            raise ValueError('empty dataset')
        available = {row['camera'] for row in self.rows}
        if cameras and not set(cameras) <= available:
            raise ValueError(f'unknown cameras: {set(cameras) - available}')
        groups = [sorted(group, key=lambda row: row['index']) for group in groups.values()
                  if not cameras or group[0]['camera'] in cameras]
        groups.sort(key=lambda group: min(row['index'] for row in group))
        self.groups = groups[::image_stride][:max_images or None]
        self.selected = [row for group in self.groups for row in group]
        if not self.selected:
            raise ValueError('no selected images')

    def path(self, name):
        # Old absolute source paths in metadata are deliberately never used.
        path = (self.root / name).resolve()
        if not path.is_relative_to(self.root) or not path.is_file():
            raise ValueError(f'missing or out-of-dataset file: {name}')
        return path

    def fingerprint(self):
        digest = hashlib.sha256()
        for name in ('summary.json', 'index.jsonl'):
            digest.update(sha256(self.root / name).encode())
        for name in sorted({row[key] for row in self.selected for key in ('image', 'label')}):
            stat = self.path(name).stat()
            digest.update(json.dumps([name, stat.st_size, stat.st_mtime_ns]).encode())
        digest.update(json.dumps([row['index'] for row in self.selected]).encode())
        return digest.hexdigest()

    def load(self, row):
        # These are the user's trusted local NPY dictionaries, including torch tensors.
        sample = np.load(self.path(row['label']), allow_pickle=True).item()
        side = int(float(sample['side']))
        if side != row['side'] or float(sample['side']) != side:
            raise ValueError(f'{row["label"]}: side disagrees with index')
        bbox = array(sample['bbox'], (4,), 'bbox')
        if np.any(bbox[2:] <= bbox[:2]):
            raise ValueError('empty bbox')
        size = tuple(self.summary['image_size'])
        if np.any(bbox[:2] < 0) or np.any(bbox[2:] >= size):
            raise ValueError('bbox outside source image')
        image_space = sample.get('image_space', self.summary['image_space'])
        if image_space != self.summary['image_space']:
            raise ValueError('label/summary image_space mismatch')
        K = array(sample['K'], (3, 3), 'K')
        if image_space == 'native_raw':
            if 'camera_calibrations_by_source' in self.summary:
                source = str(row.get('calibration_group', row.get('source_dataset_id')))
                cameras = self.summary['camera_calibrations_by_source'][source]
            else:
                cameras = self.summary['camera_calibrations']
            info = dict(cameras[row['camera']])
            if not np.allclose(K, info['K'], rtol=1e-6, atol=1e-4):
                raise ValueError('label/summary intrinsics mismatch')
            if sample.get('camera_model', info['model']) != info['model']:
                raise ValueError('label/summary camera model mismatch')
            if not np.allclose(sample.get('distortion', info['distortion']), info['distortion']):
                raise ValueError('label/summary distortion mismatch')
            calibration = CameraCalibration.from_dict(info)
        else:
            calibration = CameraCalibration(row['camera'], row['camera'], 'PINHOLE', size, K, [], np.eye(4))
        if calibration.image_size != size:
            raise ValueError('calibration/summary image size mismatch')
        translation = array(sample['trans'], (3,), 'trans')
        # trans is MANO's translation, NOT its wrist position. Geometry is already mirrored.
        joints = array(sample['joints_3d'], (21, 3), 'joints_3d') + translation
        vertices = array(sample['vertices'], (778, 3), 'vertices') + translation
        pixels, valid = project_points(calibration, joints)
        stored = array(sample['joints_2d'], (778, 2), 'joints_2d (mesh projections)')
        projected, mesh_valid = project_points(calibration, vertices)
        if not mesh_valid.all() or np.max(np.abs(projected - stored)) > .02:
            raise ValueError(f'{row["label"]}: stored mesh projection disagrees with camera geometry')
        if not valid.all():
            raise ValueError(f'{row["label"]}: invalid labeled joint projection')
        mask = np.asarray(sample.get('joint_valid', np.ones(21, bool)), dtype=bool)
        if mask.shape != (21,) or not mask.any():
            raise ValueError('joint_valid must contain 21 flags and at least one valid joint')
        return Label(row, bbox, side, joints, vertices, pixels, mask, calibration)

    def describe(self):
        return dict(root=str(self.root), total_samples=len(self.rows), selected_samples=len(self.selected),
                    selected_images=len(self.groups), cameras=dict(Counter(r['camera'] for r in self.selected)),
                    sides=dict(Counter('right' if r['side'] else 'left' for r in self.selected)),
                    image_space=self.summary['image_space'],
                    source=self.summary.get('source'), supervision_source=self.summary.get('supervision_source'),
                    annotation_policy=self.summary.get('annotation_policy'),
                    quality_thresholds=self.summary.get('quality_thresholds'),
                    fingerprint=self.fingerprint(), fingerprint_method='index/summary SHA256 + selected file sizes/mtimes + sample IDs')
