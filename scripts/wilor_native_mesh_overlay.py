"""Replay per-camera WiLoR meshes without multiview fitting or alignment."""
from pathlib import Path
import json

import numpy as np
import torch
import yaml

from fit_shared_wilor_multiview import load_mano, physical_faces, validate_mano_parameters
from render_gem_multiview import MESH_STYLE, draw_mesh_surface


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


class WilorNativeMeshOverlay:
    def __init__(self, predictions, records, image_size, model_dir, model_config, device='cuda'):
        self.root = Path(predictions).resolve()
        self.records, self.vertices, self.audits = {}, {}, []
        summary = json.loads((self.root / 'summary.json').read_text())
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
        batch_size = int(summary.get('effective_hypothesis_batch_size', 1))
        frame_batch = int(summary.get('frame_batch_size', 1)) if accelerated else 1
        self.stats = {'requested': 0, 'drawn': 0, 'missing': 0, 'outside_image': 0,
                      'replayed_hypotheses': 0, 'max_native_joint_replay_difference_m': 0.,
                      'max_native_joint_projection_difference_px': 0.}
        for camera, rows in records.items():
            self.records[camera] = {int(row['source_frame_index']): row for row in rows.values()}
            self.vertices[camera] = {}
            values = list(rows.values())
            for start in range(0, len(values), frame_batch):
                items = [(row, hand) for row in values[start:start + frame_batch] for hand in row['hands']]
                for offset in range(0, len(items), batch_size):
                    group = items[offset:offset + batch_size]
                    for _, hand in group:
                        validate_mano_parameters(hand)
                    parameters = {key: torch.tensor(np.stack([
                        np.asarray(hand['mano'][key]).reshape(shape) for _, hand in group]),
                        dtype=torch.float32, device=device) for key, shape in
                        [('global_orient', (1, 3, 3)), ('hand_pose', (15, 3, 3)), ('betas', (10,))]}
                    with torch.inference_mode(), torch.autocast(device_type='cuda', dtype=torch.float16, enabled=accelerated):
                        output = layer(**parameters, pose2rot=False)
                    joints, vertices = output.joints.float().cpu().numpy(), output.vertices.float().cpu().numpy()
                    for index, (row, hand) in enumerate(group):
                        mirror = 2 * int(hand['is_right']) - 1
                        joints[index, :, 0] *= mirror
                        vertices[index, :, 0] *= mirror
                        difference = float(np.abs(joints[index] - np.asarray(hand['joints_3d_camera_relative'])).max())
                        if difference > 1e-6:
                            raise ValueError(f'native MANO replay differs: {camera}/{row["sync_index"]}: {difference} m')
                        # Match WiLoR's original assumed-focal pinhole projection,
                        # including its saved crop-to-full camera translation.
                        xyz = joints[index] + np.asarray(hand['camera_translation'], dtype=np.float32)
                        xy = self.focal * (xyz / xyz[:, 2:])[:, :2] + np.array(image_size) / 2
                        pixel_error = float(np.abs(xy - np.asarray(hand['joints_2d'])).max())
                        if pixel_error > .002:
                            raise ValueError(f'native projection mismatch: {pixel_error} px')
                        self.stats['max_native_joint_replay_difference_m'] = max(self.stats['max_native_joint_replay_difference_m'], difference)
                        self.stats['max_native_joint_projection_difference_px'] = max(self.stats['max_native_joint_projection_difference_px'], pixel_error)
                        self.stats['replayed_hypotheses'] += 1
                        key = (int(row['source_frame_index']), int(hand['hypothesis_index']))
                        self.vertices[camera][key] = vertices[index].copy()
            print(f'[native mesh] {camera}: replayed {sum(len(row["hands"]) for row in values)} original hypotheses', flush=True)
        del layer

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
        xyz = self.vertices[camera][source, hypothesis] + np.asarray(hand['camera_translation'], dtype=np.float32)
        drawn = draw_mesh_surface(image, xyz, self.faces[int(hand['is_right'])], self.K, side)
        self.stats['drawn' if drawn else 'outside_image'] += 1
        self.audits.append({'camera': camera, 'frame_index': frame_id, 'source_frame_index': source,
                            'display_side': side, 'native_side': int(hand['is_right']),
                            'detection_index': int(hand['detection_index']), 'hypothesis_index': hypothesis,
                            'camera_translation': hand['camera_translation'], 'mesh_drawn': bool(drawn)})

    def summary(self):
        return {'model': 'WiLoR native MANO', 'predictions': str(self.root),
                'geometry': 'original per-camera MANO parameters; no shared fitting, alignment or shape changes',
                'projection': 'original WiLoR assumed-focal pinhole and saved camera_translation',
                'intrinsics': self.K.tolist(), 'shared_multiview_mesh': False,
                **MESH_STYLE, **self.stats}
