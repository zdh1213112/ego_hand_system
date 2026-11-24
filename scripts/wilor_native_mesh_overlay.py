"""Replay per-camera WiLoR meshes without multiview fitting or alignment."""
from pathlib import Path
import json

import numpy as np
import torch
import yaml

from fit_shared_wilor_multiview import load_mano, physical_faces, validate_mano_parameters
from ego_data.calibration import CameraCalibration
from render_gem_multiview import MESH_STYLE, draw_mesh_surface, draw_projected_mesh
from wilor_physical_inference import project_physical_points


def select_native_hand(hands, view):
    hypothesis = view.get('hypothesis_index')
    if hypothesis is not None:
        return next((hand for hand in hands if int(hand['hypothesis_index']) == int(hypothesis)), None)
    detection = view.get('detection_index')
    candidates = [hand for hand in hands if detection is not None and int(hand['detection_index']) == int(detection)]
    observed = np.asarray(view.get('wilor_joints_2d', view.get('joints_2d', [])))
    if not candidates or observed.shape != (21, 2) or not np.isfinite(observed).all():
        return None
    return min(candidates, key=lambda hand: np.linalg.norm(np.asarray(hand['joints_2d']) - observed, axis=1).mean())


class NativeManoReplay:
    """Rebuild native geometry in the original inference batches, one camera at a time."""

    def __init__(self, summary, image_size, model_dir, model_config, device):
        self.camera_model = (summary.get('model') or {}).get('camera_model', 'virtual_perspective')
        self.calibrations = {}
        if self.camera_model != 'virtual_perspective':
            expected = {'physical_kb': 'KB', 'physical_perspective': 'PINHOLE'}
            for camera in summary['camera_ids']:
                path = Path(summary['dataset']) / 'calibration' / f'{camera}.json'
                calibration = CameraCalibration.load(path)
                if calibration.model != expected.get(self.camera_model) or calibration.image_size != tuple(image_size):
                    raise ValueError(f'{camera}: calibration does not match {self.camera_model} or image size')
                self.calibrations[camera] = calibration
        config = yaml.safe_load(Path(model_config).read_text())
        self.focal = float(config['EXTRA']['FOCAL_LENGTH']) / float(config['MODEL']['IMAGE_SIZE']) * max(image_size)
        self.K = np.array([[self.focal, 0, image_size[0] / 2], [0, self.focal, image_size[1] / 2], [0, 0, 1]])
        accelerated = summary.get('gpu_profile') == 'rtx5090d'
        if str(device) == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if accelerated and not str(device).startswith('cuda'):
            raise ValueError('exact replay of accelerated WiLoR requires CUDA')
        layer = load_mano(model_dir, device=device)
        if accelerated:
            layer.half()
            torch.set_float32_matmul_precision('high')
            torch.backends.cuda.matmul.allow_tf32 = True
        self.faces = {side: physical_faces(layer, side).numpy() for side in (0, 1)}
        self.batch_size = int(summary.get('effective_hypothesis_batch_size', 1))
        self.frame_batch = int(summary.get('frame_batch_size', 1)) if accelerated else 1
        self.layer = layer
        self.device = device
        self.accelerated = accelerated
        self.stats = {'replayed_hypotheses': 0, 'max_native_joint_replay_difference_m': 0.,
                      'max_native_joint_projection_difference_px': 0.}

    def replay(self, camera, rows):
        """Return relative vertices keyed by (source frame, hypothesis) for one frame batch."""
        vertices_by_hand = {}
        items = [(row, hand) for row in rows for hand in row['hands']]
        for offset in range(0, len(items), self.batch_size):
            group = items[offset:offset + self.batch_size]
            for _, hand in group:
                validate_mano_parameters(hand)
            parameters = {key: torch.tensor(np.stack([
                np.asarray(hand['mano'][key]).reshape(shape) for _, hand in group]),
                dtype=torch.float32, device=self.device) for key, shape in
                [('global_orient', (1, 3, 3)), ('hand_pose', (15, 3, 3)), ('betas', (10,))]}
            with torch.inference_mode(), torch.autocast(device_type='cuda', dtype=torch.float16, enabled=self.accelerated):
                output = self.layer(**parameters, pose2rot=False)
            joints, vertices = output.joints.float().cpu().numpy(), output.vertices.float().cpu().numpy()
            for index, (row, hand) in enumerate(group):
                mirror = 2 * int(hand['is_right']) - 1
                joints[index, :, 0] *= mirror
                vertices[index, :, 0] *= mirror
                difference = float(np.abs(joints[index] - np.asarray(hand['joints_3d_camera_relative'])).max())
                if not np.isfinite(difference) or difference > 1e-6:
                    raise ValueError(f'native MANO replay differs: {camera}/{row["sync_index"]}: {difference} m')
                xyz = joints[index] + np.asarray(hand['camera_translation'], dtype=np.float32)
                if camera in self.calibrations:
                    xy = project_physical_points(xyz, self.calibrations[camera])
                else:
                    xy = self.focal * (xyz / xyz[:, 2:])[:, :2] + self.K[:2, 2]
                pixel_error = float(np.abs(xy - np.asarray(hand['joints_2d'])).max())
                if not np.isfinite(pixel_error) or pixel_error > .002:
                    raise ValueError(f'native projection mismatch: {camera}/{row["sync_index"]}: {pixel_error} px')
                self.stats['max_native_joint_replay_difference_m'] = max(self.stats['max_native_joint_replay_difference_m'], difference)
                self.stats['max_native_joint_projection_difference_px'] = max(self.stats['max_native_joint_projection_difference_px'], pixel_error)
                self.stats['replayed_hypotheses'] += 1
                vertices_by_hand[int(row['source_frame_index']), int(hand['hypothesis_index'])] = vertices[index].copy()
        return vertices_by_hand

    def draw_hand_mesh(self, image, camera, hand, relative_vertices, display_side):
        xyz = relative_vertices + np.asarray(hand['camera_translation'], dtype=np.float32)
        faces = self.faces[int(hand['is_right'])]
        if camera in self.calibrations:
            xy = project_physical_points(xyz, self.calibrations[camera])
            valid = np.isfinite(xy).all(axis=1) & (np.abs(xy) < 1e6).all(axis=1)
            return draw_projected_mesh(image, xyz, faces, xy, valid, display_side)
        return draw_mesh_surface(image, xyz, faces, self.K, display_side)


class WilorNativeMeshOverlay(NativeManoReplay):
    def __init__(self, predictions, records, image_size, model_dir, model_config, device='cuda'):
        self.root = Path(predictions).resolve()
        self.records, self.vertices, self.audits = {}, {}, []
        summary = json.loads((self.root / 'summary.json').read_text())
        super().__init__(summary, image_size, model_dir, model_config, device)
        self.stats.update(requested=0, drawn=0, missing=0, outside_image=0)
        for camera, rows in records.items():
            values = list(rows.values())
            self.records[camera] = {int(row['source_frame_index']): row for row in values}
            self.vertices[camera] = {}
            for start in range(0, len(values), self.frame_batch):
                self.vertices[camera].update(self.replay(camera, values[start:start + self.frame_batch]))
            print(f'[native mesh] {camera}: replayed {len(self.vertices[camera])} original hypotheses', flush=True)
        del self.layer

    def draw(self, image, camera, frame_id, view, side):
        self.stats['requested'] += 1
        source = int(view.get('mesh_source_frame_index', frame_id))
        row = self.records[camera].get(source)
        hand = select_native_hand(row['hands'], view) if row else None
        if hand is None:
            self.stats['missing'] += 1
            self.audits.append({'camera': camera, 'frame_index': frame_id, 'source_frame_index': source,
                                'display_side': side, 'missing': True})
            return
        hypothesis = int(hand['hypothesis_index'])
        drawn = self.draw_hand_mesh(image, camera, hand, self.vertices[camera][source, hypothesis], side)
        self.stats['drawn' if drawn else 'outside_image'] += 1
        self.audits.append({'camera': camera, 'frame_index': frame_id, 'source_frame_index': source,
                            'display_side': side, 'native_side': int(hand['is_right']),
                            'detection_index': int(hand['detection_index']), 'hypothesis_index': hypothesis,
                            'camera_translation': hand['camera_translation'], 'mesh_drawn': bool(drawn)})

    def summary(self):
        return {'model': 'WiLoR native MANO', 'predictions': str(self.root),
                'geometry': 'original per-camera MANO parameters; no shared fitting, alignment or shape changes',
                'projection': self.camera_model if self.calibrations else 'original WiLoR assumed-focal pinhole and saved camera_translation',
                'intrinsics': ({c: cal.K.tolist() for c, cal in self.calibrations.items()} if self.calibrations else self.K.tolist()),
                'shared_multiview_mesh': False,
                **MESH_STYLE, **self.stats}
