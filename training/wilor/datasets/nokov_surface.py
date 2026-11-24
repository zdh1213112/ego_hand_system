"""Observed glove markers and calibrated skin attachments for native-camera RGB supervision."""
from pathlib import Path

import numpy as np
import torch


class SurfaceMarkerTargets:
    def __init__(self, root, profile, groups):
        from .glove_validation import sha256_file
        self.recordings = {}
        for group in groups:
            entry = profile['nokov_surface_archives'][group]
            path = Path(root) / entry['path']
            if sha256_file(path) != entry['sha256']:
                raise ValueError(f'Changed NOKOV surface archive for {group}')
            with np.load(path) as data:
                if data['format'].item() != 'native_surface_camera0_v1':
                    raise ValueError(f'Unexpected surface target format: {path}')
                self.recordings[group] = dict(data)

    def sample(self, group, side, sync, camera):
        rec = self.recordings[group]
        points = rec['points_m'][side, sync]
        valid = np.isfinite(points).all(-1)
        if valid.reshape(5,4).sum(1).min() < 3:
            raise ValueError(f'{group} {side} {sync}: incomplete measured finger')
        cam = rec['cameras'].tolist().index(camera)
        points = points @ rec['rotation'][cam].T + rec['translation'][cam]
        if side == 0:
            points[:,0] *= -1
        return {
            'surface_target_m': torch.from_numpy(np.nan_to_num(points).astype(np.float32)),
            'surface_valid': torch.from_numpy(valid),
            'surface_triangles': torch.from_numpy(rec['triangles'][side]),
            'surface_barycentric': torch.from_numpy(rec['barycentric'][side]),
            'surface_offsets_m': torch.from_numpy(rec['offsets_m'][side]),
        }
