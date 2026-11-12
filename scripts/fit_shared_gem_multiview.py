#!/usr/bin/env python3
"""Jointly fit one physical SOMA model per hand/frame to calibrated raw views."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

from ego_data.calibration import CameraCalibration
from gem_multiview_inference import SOMA_TO_21
from gem_shared_geometry import associate_hands, project_base, project_base_torch, rigid_initialization, rotation_exp


def signature(path):
    path = Path(path).resolve()
    stat = path.stat()
    return {'path': str(path), 'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}


def fit_batch(items, shape, geometry, calibrations, *, device, steps):
    """Optimize articulation and one SE(3)/scale transform shared by every view."""
    side = items[0]['side']
    assert all(item['side'] == side for item in items)
    count = len(items)
    cameras = list(calibrations)
    dtype = torch.float32
    as_tensor = lambda value: torch.as_tensor(value, dtype=dtype, device=device)
    params = {name: torch.stack([item['seed']['parameters'][name] for item in items]).to(device)
              for name in ('wrist_orient', 'hand_pose')}
    identity = as_tensor(shape['identity_coeffs'])[None].expand(count, -1).clone()
    scales = as_tensor(shape['scale_params'])[None].expand(count, -1).clone()
    poses0 = torch.cat([params['wrist_orient'], params['hand_pose']], dim=1).detach()
    layer = geometry._layer('right' if side else 'left', torch.device(device))
    with torch.no_grad():
        # Match GEM forward_physical(apply_correctives=False). The SOMA
        # low-level preparation otherwise defaults to reposing the rest mesh,
        # which preserves joints but changes vertices on parameter replay.
        layer.prepare_identity(identity, global_scale=scales[:, 0], scale_params=scales[:, 1:],
                               repose_to_bind_pose=False)
        initial_local = layer.pose(poses0, pose2rot=False, fk_only=True).joints
    initial_np = initial_local[:, list(SOMA_TO_21)].cpu().numpy()
    rotations, translations = [], []
    for item, local in zip(items, initial_np):
        rotation, translation = rigid_initialization(local, item['points'], item['valid'])
        rotations.append(rotation)
        translations.append(translation)
    rotation0 = as_tensor(np.stack(rotations))
    translation = torch.nn.Parameter(as_tensor(np.stack(translations)))
    root_delta = torch.nn.Parameter(torch.zeros(count, 3, device=device))
    finger_delta = torch.nn.Parameter(torch.zeros(count, 24, 3, device=device))
    log_scale = torch.nn.Parameter(torch.zeros(count, 1, 1, device=device))
    observed = np.zeros((count, len(cameras), 21, 2), dtype=np.float32)
    weights = np.zeros((count, len(cameras), 21), dtype=np.float32)
    for i, item in enumerate(items):
        for camera, observation in item['selected'].items():
            index = cameras.index(camera)
            observed[i, index] = observation['pixels']
            weights[i, index] = max(0.1, float(observation['hand']['confidence']))
    observed, weights = as_tensor(observed), as_tensor(weights)
    targets = as_tensor(np.stack([item['points'] for item in items]))
    target_mask = as_tensor(np.stack([item['valid'] for item in items]))
    optimizer = torch.optim.Adam([
        {'params': [translation], 'lr': 0.004},
        {'params': [root_delta, finger_delta], 'lr': 0.02},
        {'params': [log_scale], 'lr': 0.006},
    ])

    def decode(fk_only=True):
        fingers = rotation_exp(finger_delta) @ poses0[:, 1:]
        poses = torch.cat([poses0[:, :1], fingers], dim=1)
        result = layer.pose(poses, pose2rot=False, fk_only=fk_only)
        rotation = rotation_exp(root_delta) @ rotation0
        scale = log_scale.exp()
        joints = (result.joints * scale) @ rotation.transpose(-1, -2) + translation[:, None]
        vertices = None if fk_only else (result.vertices * scale) @ rotation.transpose(-1, -2) + translation[:, None]
        return joints, vertices, poses, rotation, scale

    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        joints, _, _, _, _ = decode()
        points = joints[:, list(SOMA_TO_21)]
        projections, depths = [], []
        for camera in cameras:
            xy, _, xyz = project_base_torch(calibrations[camera], points)
            projections.append(xy)
            depths.append(xyz[..., 2])
        residual = (torch.stack(projections, dim=1) - observed) / 10.0
        robust = (torch.sqrt(1 + residual.square()) - 1).sum(dim=-1)
        image_loss = (robust * weights).sum() / weights.sum().clamp_min(1)
        delta3d = (points - targets) / 0.04
        geometry_loss = ((torch.sqrt(1 + delta3d.square()) - 1).sum(-1) * target_mask).sum() / target_mask.sum().clamp_min(1)
        depth_loss = (torch.relu(0.02 - torch.stack(depths, dim=1)).square() * weights).sum() / weights.sum().clamp_min(1)
        loss = (image_loss + 0.15 * geometry_loss + 0.12 * finger_delta.square().mean()
                + 0.10 * log_scale.square().mean() + 100 * depth_loss)
        if not torch.isfinite(loss):
            raise FloatingPointError('non-finite shared SOMA fit loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_([translation, root_delta, finger_delta, log_scale], 10.0)
        optimizer.step()
        with torch.no_grad():
            log_scale.clamp_(np.log(0.65), np.log(1.5))
            finger_delta.clamp_(-0.8, 0.8)
    with torch.no_grad():
        joints, vertices, poses, rotation, scale = decode(fk_only=False)
    output = []
    for i, item in enumerate(items):
        output.append({
            'joints_soma_base_m': joints[i].cpu(), 'vertices_base_m': vertices[i].cpu(),
            'parameters': {'wrist_orient': poses[i, :1].cpu(), 'hand_pose': poses[i, 1:].cpu(),
                           'identity_coeffs': identity[i].cpu(), 'scale_params': scales[i].cpu()},
            'rotation_base_from_model': rotation[i].cpu(), 'translation_base_m': translation[i].detach().cpu(),
            'model_scale': float(scale[i, 0, 0]), 'side': side,
            'seed_camera': item['seed_camera'], 'sync_index': item['sync_index'],
        })
    return output, torch.as_tensor(layer.faces, dtype=torch.int32).cpu()


def fit_quality(model, candidate, calibrations, threshold_px):
    points = np.asarray(model['joints_soma_base_m'])[list(SOMA_TO_21)]
    views, supported = {}, []
    for camera, observation in candidate['selected'].items():
        xy, valid, _ = project_base(calibrations[camera], points)
        errors = np.linalg.norm(xy - observation['pixels'], axis=1)
        good = valid & np.isfinite(errors)
        median = float(np.median(errors[good])) if good.any() else None
        p95 = float(np.percentile(errors[good], 95)) if good.any() else None
        support = int(np.count_nonzero(good & (errors <= threshold_px)))
        trusted = support >= 16 and median is not None and median <= threshold_px * 0.7
        if trusted:
            supported.append(camera)
        hand = observation['hand']
        views[camera] = {'detection_index': int(hand['detection_index']),
                         'hypothesis_index': int(hand['hypothesis_index']),
                         'bbox_xyxy': hand['bbox_xyxy'], 'confidence': float(hand['confidence']),
                         'observed_joints_2d': observation['pixels'].tolist(),
                         'median_px': median, 'p95_px': p95,
                         'inlier_joint_count': support, 'supported': trusted}
    good_shape = np.isfinite(points).all() and 0.04 < np.linalg.norm(points - points[0], axis=1).max() < 0.40
    # Verify geometry using only the views that support the fitted model.
    sufficient_angle = False
    if len(supported) >= 2:
        rays = [points[0] - calibrations[c].T_base_camera[:3, 3] for c in supported]
        rays = [ray / np.linalg.norm(ray) for ray in rays]
        sufficient_angle = any(np.degrees(np.arccos(np.clip(abs(a @ b), 0, 1))) >= 1.0
                               for n, a in enumerate(rays) for b in rays[n + 1:])
    return bool(good_shape and sufficient_angle and len(supported) >= 2), views, supported


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('dataset', 'predictions', 'native-predictions', 'output', 'gem-root', 'soma-assets'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--cameras', nargs='+')
    p.add_argument('--device', default='cuda')
    p.add_argument('--max-frames', type=int, default=0)
    p.add_argument('--steps', type=int, default=100)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--max-detections', type=int, default=3)
    p.add_argument('--reprojection-threshold-px', type=float, default=30.0)
    args = p.parse_args()
    if args.max_frames < 0 or args.steps < 1 or args.batch_size < 1 or args.max_detections < 1 or args.reprojection_threshold_px <= 0:
        p.error('invalid shared fitting options')
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    torch.set_num_threads(4)
    dataset, output = args.dataset.resolve(), args.output.resolve()
    manifest = json.loads((dataset / 'manifest.json').read_text())
    cameras = list(dict.fromkeys(args.cameras or manifest['camera_ids']))
    if len(cameras) < 2 or not set(cameras) <= set(manifest['camera_ids']):
        p.error('shared reconstruction requires at least two available cameras')
    calibrations = {c: CameraCalibration.load(dataset / 'calibration' / f'{c}.json') for c in cameras}
    with (dataset / 'multiview_frames.csv').open() as stream:
        rows = list(csv.DictReader(stream))[:args.max_frames or None]
    if not rows:
        p.error('no synchronized frames')
    inputs = [dataset / 'manifest.json', dataset / 'multiview_frames.csv', args.soma_assets / 'SOMAHand.npz']
    for camera in cameras:
        inputs.extend([dataset / 'calibration' / f'{camera}.json', args.predictions / camera / 'predictions.jsonl',
                       args.native_predictions / camera / 'gem_pose.pt'])
    config = {'algorithm': 'shared_soma_multiview_v2', 'inputs': [signature(path) for path in inputs],
              'cameras': cameras, 'frame_count': len(rows), 'steps': args.steps, 'batch_size': args.batch_size,
              'max_detections': args.max_detections, 'threshold_px': args.reprojection_threshold_px,
              'device': args.device, 'gem_root': str(args.gem_root.resolve()),
              'soma_assets': str(args.soma_assets.resolve()), 'min_supporting_views': 2,
              'no_independent_view_fallback': True}
    if output.exists():
        summary_path = output / 'summary.json'
        if not summary_path.is_file() or not (output / 'shared_pose.pt').is_file():
            raise RuntimeError(f'incomplete shared output exists: {output}; use a new output directory')
        if json.loads((output / 'run_config.json').read_text()) != config:
            raise RuntimeError(f'shared configuration changed: use a new output directory instead of {output}')
        print(f'[shared] reuse completed fit: {output}', flush=True)
        return
    output.mkdir(parents=True)
    (output / 'run_config.json').write_text(json.dumps(config, indent=2) + '\n')
    started = time.perf_counter()
    prediction_rows, native = {}, {}
    for camera in cameras:
        prediction_rows[camera] = {int(r['sync_index']): r for r in map(json.loads,
            (args.predictions / camera / 'predictions.jsonl').read_text().splitlines())}
        data = torch.load(args.native_predictions / camera / 'gem_pose.pt', weights_only=False,
                          map_location='cpu', mmap=True)
        native[camera] = {int(frame['frame_id']): {int(h['track_id']): h for h in frame['instances']}
                          for frame in data['frames']}
    candidates, rejected = [], []
    for row in rows:
        sync = int(row['sync_index'])
        records = {c: prediction_rows[c][sync] for c in cameras}
        hands = associate_hands(records, calibrations, threshold_px=args.reprojection_threshold_px,
                                max_detections=args.max_detections)
        for hand in hands:
            hand['sync_index'] = sync
            # The anchor pose is only an initialization. Its translation is
            # discarded, and all selected raw views constrain the shared fit.
            seed_camera = max(hand['selected'], key=lambda c: float(hand['selected'][c]['hand']['confidence']))
            seed_observation = hand['selected'][seed_camera]
            hand['seed'] = native[seed_camera][seed_observation['source_frame_index']][int(seed_observation['hand']['hypothesis_index'])]
            hand['seed_camera'] = seed_camera
            candidates.append(hand)
        if not hands:
            rejected.append({'sync_index': sync, 'reason': 'insufficient_consistent_multiview_observations'})
    print(f'[shared] associated {len(candidates)} hand/frame models over {len(rows)} frames', flush=True)
    # One identity/shape per side for the recording. Articulation and global
    # transforms vary over time; every camera always sees the same model.
    shapes = {}
    for side in (0, 1):
        group = [c for c in candidates if c['side'] == side]
        if group:
            shapes[side] = {name: np.median(np.stack([np.asarray(c['seed']['parameters'][name]) for c in group]), axis=0)
                            for name in ('identity_coeffs', 'scale_params')}
    sys.path.insert(0, str(args.gem_root.resolve()))
    from gem.pose.geometry.hand import HandGeometry
    geometry = HandGeometry(args.soma_assets, mode='torch').to(args.device)
    model_frames = {int(row['sync_index']): {'sync_index': int(row['sync_index']), 'hands': []} for row in rows}
    accepted = {index: {'sync_index': index, 'hands': []} for index in model_frames}
    topology, fitted = {}, 0
    for side in (0, 1):
        group = [c for c in candidates if c['side'] == side]
        for start in range(0, len(group), args.batch_size):
            batch = group[start:start + args.batch_size]
            models, faces = fit_batch(batch, shapes[side], geometry, calibrations,
                                     device=args.device, steps=args.steps)
            topology['right' if side else 'left'] = faces
            for model, candidate in zip(models, batch):
                valid, views, support = fit_quality(model, candidate, calibrations, args.reprojection_threshold_px)
                if not valid:
                    rejected.append({'sync_index': model['sync_index'], 'side': side,
                                     'reason': 'shared_model_reprojection_or_geometry_failed', 'views': views})
                    continue
                model['views'] = views
                model['supporting_cameras'] = support
                model_frames[model['sync_index']]['hands'].append(model)
                accepted[model['sync_index']]['hands'].append({
                    'side': side, 'joints_base_m': model['joints_soma_base_m'][list(SOMA_TO_21)].tolist(),
                    'joints_soma_base_m': model['joints_soma_base_m'].tolist(),
                    'supporting_cameras': support, 'views': views,
                    'geometry_source': 'shared_soma_model', 'model_scale': model['model_scale'],
                })
            fitted += len(batch)
            print(f'[shared] fitted {fitted}/{len(candidates)} models | {time.perf_counter() - started:.1f}s', flush=True)
    torch.save({'format': 'shared_gem_soma_v1', 'coordinate_system': manifest['coordinate_system'],
                'topology': topology, 'frames': list(model_frames.values()),
                'soma_to_21_indices': list(SOMA_TO_21)}, output / 'shared_pose.pt')
    with (output / 'accepted.jsonl').open('w') as stream:
        for row in accepted.values():
            if row['hands']:
                stream.write(json.dumps(row, allow_nan=False) + '\n')
    with (output / 'rejected.jsonl').open('w') as stream:
        for row in rejected:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
    hands = [h for row in accepted.values() for h in row['hands']]
    errors = [v['median_px'] for h in hands for v in h['views'].values() if v['supported']]
    summary = {'stage': 'shared_soma_multiview', 'frame_count': len(rows), 'candidate_hand_count': len(candidates),
               'accepted_frame_count': sum(bool(row['hands']) for row in accepted.values()),
               'accepted_hand_count': len(hands), 'camera_ids': cameras,
               'supporting_view_counts': dict(Counter(len(h['supporting_cameras']) for h in hands)),
               'supporting_view_median_error_px': float(np.median(errors)) if errors else None,
               'fit_seconds': time.perf_counter() - started,
               'skeleton_and_mesh_share_model': True, 'independent_view_fallback': False,
               'identity_shape_per_side': {str(k): {n: v.tolist() for n, v in s.items()} for k, s in shapes.items()},
               'rejected_hand_or_frame_reasons': dict(Counter(row['reason'] for row in rejected))}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
