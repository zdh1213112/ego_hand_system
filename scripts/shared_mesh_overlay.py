"""The common calibrated mesh layer for shared and legacy-skeleton videos."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from camera_models import project_points
from render_gem_multiview import MESH_STYLE, draw_mesh_surface


def mesh_camera_vertices(hand, calibration):
    """Keep the exact transform used by the shared skeleton/mesh renderer."""
    return ((np.asarray(hand['vertices_base_m']) - calibration.T_base_camera[:3, 3])
            @ calibration.T_base_camera[:3, :3])


def mesh_projection_fingerprint(vertices, faces, calibration):
    """Audit all vertices and topology, including vertices outside the image."""
    pixels, valid = project_points(calibration, np.asarray(vertices, dtype=np.float64))
    valid &= np.isfinite(pixels).all(axis=1) & (np.abs(pixels) < 1e6).all(axis=1)
    digest = hashlib.sha256()
    for array, dtype in ((vertices, '<f8'), (faces, '<i4'),
                         (np.where(valid[:, None], pixels, 0.), '<f8'), (valid, 'u1')):
        digest.update(np.ascontiguousarray(array, dtype=dtype).tobytes())
    return digest.hexdigest()


def draw_shared_mesh(image, hand, topology, calibration):
    side = int(hand['side'])
    vertices = mesh_camera_vertices(hand, calibration)
    faces = topology['right' if side else 'left']
    drawn = draw_mesh_surface(image, vertices, faces, calibration.K, side, calibration=calibration)
    return drawn, mesh_projection_fingerprint(vertices, faces, calibration)


class SharedManoMeshOverlay:
    """Overlay every shared hand, independently of legacy skeleton fallbacks."""

    def __init__(self, shared, dataset, cameras, frame_ids):
        self.shared = Path(shared).resolve()
        dataset = Path(dataset).resolve()
        config = json.loads((self.shared / 'run_config.json').read_text())
        inputs = {Path(item['path']).resolve(): item for item in config['inputs']}
        # A matching frame number alone is insufficient: verify the actual
        # synchronized images' manifest/table and calibration used in fitting.
        for path in [dataset / 'manifest.json', dataset / 'multiview_frames.csv',
                     *(dataset / 'calibration' / f'{camera}.json' for camera in cameras)]:
            saved, stat = inputs.get(path), path.stat()
            if saved is None or (saved['size'], saved['mtime_ns']) != (stat.st_size, stat.st_mtime_ns):
                raise ValueError(f'shared mesh dataset/calibration differs: {path}')
        self.data = torch.load(self.shared / 'shared_pose.pt', map_location='cpu',
                               weights_only=False, mmap=True)
        if self.data['format'] != 'shared_wilor_mano_v1':
            raise ValueError('expected shared WiLoR MANO mesh')
        self.frames = {int(frame['sync_index']): frame for frame in self.data['frames']}
        if len(self.frames) != len(self.data['frames']) or not set(frame_ids) <= self.frames.keys():
            raise ValueError('shared mesh frames are duplicate or incomplete')
        for frame in self.frames.values():
            sides = [int(hand['side']) for hand in frame['hands']]
            if len(sides) != len(set(sides)) or not set(sides) <= {0, 1}:
                raise ValueError('shared mesh has duplicate/invalid physical hand side')
        self.drawn = self.requested = 0

    def draw(self, image, sync, calibration):
        audits = []
        for hand in self.frames[sync]['hands']:
            drawn, fingerprint = draw_shared_mesh(image, hand, self.data['topology'], calibration)
            self.requested += 1
            self.drawn += int(drawn)
            audits.append({'side': int(hand['side']), 'shared_model_id': f"{sync}:{int(hand['side'])}",
                           'mesh_projection_sha256': fingerprint, 'mesh_drawn': bool(drawn)})
        return audits

    def summary(self):
        path = self.shared / 'shared_pose.pt'
        return {'model': 'WiLoR shared MANO', 'shared': str(self.shared),
                'shared_mtime_ns': path.stat().st_mtime_ns, 'shared_size': path.stat().st_size,
                'geometry': 'same shared vertices, topology and calibrated projection as shared video',
                'independent_view_fallback': False, 'requested': self.requested, 'drawn': self.drawn,
                **MESH_STYLE}
