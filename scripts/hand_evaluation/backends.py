"""Model adapters receive only images, labeled boxes/sides and calibration.

No 3-D labels or MANO label parameters cross this boundary.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np


@dataclass
class FrameInput:
    image: np.ndarray
    boxes: np.ndarray
    sides: np.ndarray
    calibration: object


class WilorBackend:
    def __init__(self, args):
        import torch
        import wilor_inference as base
        self.base, self.torch, self.args = base, torch, args
        self.device = base.choose_device(args.device)
        self.model, self.cfg = base.load_wilor_model(args.wilor_checkpoint, args.wilor_config, args.mano_dir)
        self.model.to(self.device).eval()
        self.description = dict(model='WiLoR', dtype='float32', mesh_topology='MANO778',
                                camera='native WiLoR synthetic focal/principal point; calibrated placement scored separately')

    def predict(self, frames):
        torch, base = self.torch, self.base
        crops, origins = [], []
        results = [[None] * len(frame.boxes) for frame in frames]
        for frame_index, frame in enumerate(frames):
            dataset = base.ViTDetDataset(self.cfg, frame.image, frame.boxes, frame.sides,
                                        rescale_factor=self.args.crop_scale, fp16=False)
            for hand_index in range(len(dataset)):
                crops.append(dataset[hand_index])
                origins.append((frame_index, hand_index))
        from torch.utils.data import default_collate
        for start in range(0, len(crops), self.args.batch_size):
            batch = base.recursive_to(default_collate(crops[start:start + self.args.batch_size]), self.device)
            with torch.inference_mode():
                out = self.model(batch)
            cam = out['pred_cam'].clone()
            cam[:, 1] *= 2 * batch['right'] - 1
            # Per-sample maximum, never the maximum across differently sized images.
            focal = self.cfg.EXTRA.FOCAL_LENGTH / self.cfg.MODEL.IMAGE_SIZE * batch['img_size'].float().max(dim=1).values
            full = base.cam_crop_to_full(cam, batch['box_center'].float(), batch['box_size'].float(),
                                         batch['img_size'].float(), focal).detach().float().cpu().numpy()
            for i, (frame_index, hand_index) in enumerate(origins[start:start + self.args.batch_size]):
                side = int(frames[frame_index].sides[hand_index])
                joints = out['pred_keypoints_3d'][i].detach().float().cpu().numpy().copy()
                vertices = out['pred_vertices'][i].detach().float().cpu().numpy().copy()
                joints[:, 0] *= 2 * side - 1
                vertices[:, 0] *= 2 * side - 1
                pixels = base.project_points(joints, full[i], float(focal[i]),
                                              batch['img_size'][i].detach().cpu().numpy())
                results[frame_index][hand_index] = dict(joints_camera_m=joints + full[i],
                    joints_2d_px=pixels, vertices_camera_m=vertices + full[i])
                if getattr(self.args, 'export_shared_parameters', False):
                    results[frame_index][hand_index]['mano'] = {
                        key: value[i].detach().float().cpu().numpy().copy()
                        for key, value in out['pred_mano_params'].items()}
        return results


class GemBackend:
    def __init__(self, args):
        import torch
        sys.path.insert(0, str(args.gem_root))
        from gem.inference.pose import load_predictor
        self.predictor, release = load_predictor(args.gem_checkpoint, task='hand', release='hand',
                                                  soma_assets=args.soma_assets, device=torch.device(args.device))
        self.topology, self.args = {}, args
        self.description = dict(model='GEM-X EgoHand', dtype='float32', mesh_topology='SOMA (MPVPE unavailable)',
            joint_mapping='SOMA 25 -> 21; approximate anatomical correspondence with MANO',
            camera='pinhole intrinsics approximation on source raster', release=release)

    def predict(self, frames):
        import cv2
        from gem_multiview_inference import SOMA_TO_21, make_hypotheses, predict_in_batches
        results = []
        for index, frame in enumerate(frames):
            hypotheses = make_hypotheses(frame.boxes, np.ones(len(frame.boxes)), frame.sides,
                                         rescale_factor=self.args.crop_scale)
            # Pick the supplied physical side BEFORE inference; never choose using 3-D error.
            instances = [hypotheses[2 * i + int(side)] for i, side in enumerate(frame.sides)]
            native = predict_in_batches(self.predictor, cv2.cvtColor(frame.image, cv2.COLOR_BGR2RGB), index,
                instances, frame.calibration.K, self.topology, self.args.batch_size, amp=False)
            by_id = {int(item['track_id']): item for item in native['instances']}
            hands = []
            for i, side in enumerate(frame.sides):
                item = by_id.get(2 * i + int(side))
                if item is None:
                    hands.append(missing_prediction())
                    continue
                joints = np.asarray(item['joints_camera'], np.float32)[list(SOMA_TO_21)]
                pixels = np.asarray(item['keypoints_2d'], np.float32)[list(SOMA_TO_21)].copy()
                valid = np.asarray(item['projection_valid'], bool)[list(SOMA_TO_21)]
                pixels[~valid] = np.nan
                hands.append(dict(joints_camera_m=joints, joints_2d_px=pixels))
                if getattr(self.args, 'export_shared_parameters', False):
                    hands[-1]['soma_parameters'] = item['parameters']
            results.append(hands)
        return results


class FahBackend:
    def __init__(self, args):
        from types import SimpleNamespace
        from unittest.mock import patch
        sys.path.insert(0, str(args.fah_root))
        import fah
        # GT-box protocol must not construct or execute a detector.
        with patch('bench.methods.fah.build_hand_detector', return_value=SimpleNamespace(names={0: 'left', 1: 'right'})):
            self.model = fah.FAH(args.fah_model, mode='image', weights=str(args.fah_weights),
                mano_dir=str(args.fah_mano_dir), sam3d=str(args.fah_sam3d), device=args.device,
                mesh='mano', tips=False, parts=False, verify=False, fast_det=False,
                compile=False, parts_in_mask=False, dtype='float32', cuda_graph=False,
                tf32=False, gpu_frame=False, fast_batch=False)
        self.model.method.est.max_batch_size = args.batch_size
        self.export_shared_parameters = getattr(args, 'export_shared_parameters', False)
        self.description = dict(model=args.fah_model, dtype='float32', mesh_topology='MANO778',
            camera='KB hand-local perspective on native fisheye; native FAH crop otherwise',
            pose='FAH MANO-convention joints; mesh uses native MANO fit; no video filtering')

    def predict(self, frames):
        from fah_multiview_inference import predict_camera_hands
        results = []
        method = self.model.method
        for frame in frames:
            self.model.reset()
            hands, parameters = predict_camera_hands(method, frame.image, frame.boxes,
                np.ones(len(frame.boxes)), [(bool(s),) for s in frame.sides],
                frame.calibration.K, frame.calibration if frame.calibration.model == 'KB' else None,
                self.export_shared_parameters)
            if len(hands) != len(frame.boxes):
                raise RuntimeError('FAH changed supplied box count; cannot safely associate predictions')
            converted = []
            for side, hand in zip(frame.sides, hands):
                if int(hand['is_right']) != int(side):
                    raise RuntimeError('FAH changed supplied handedness order')
                converted.append(dict(joints_camera_m=np.asarray(hand.get('keypoints_3d_mano', hand['keypoints_3d'])),
                    joints_2d_px=np.asarray(hand.get('keypoints_2d_mano', hand['keypoints_2d'])),
                    vertices_camera_m=np.asarray(hand['vertices_mano'])))
                if self.export_shared_parameters:
                    converted[-1]['mano'] = parameters[len(converted) - 1]
            results.append(converted)
        return results


def missing_prediction():
    return dict(joints_camera_m=np.full((21, 3), np.nan), joints_2d_px=np.full((21, 2), np.nan))


def create_backend(args):
    import torch
    if str(args.device).startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; no automatic CPU fallback')
    torch.set_num_threads(args.workers)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    return {'wilor': WilorBackend, 'gem': GemBackend, 'fah': FahBackend}[args.backend](args)


def asset_paths(args):
    if args.backend == 'wilor':
        return [args.wilor_checkpoint, args.wilor_config, args.mano_dir / 'MANO_RIGHT.pkl']
    if args.backend == 'gem':
        return [args.gem_checkpoint, *sorted(args.soma_assets.glob('*.npz'))]
    return [args.fah_weights / args.fah_model / 'config.json', args.fah_weights / args.fah_model / 'model.pt',
            args.fah_weights / 'shared/mesh_fitter.pt', args.fah_mano_dir / 'MANO_RIGHT.pkl',
            args.fah_sam3d / 'model_config.yaml', args.fah_sam3d / 'assets/mhr_model.pt']
