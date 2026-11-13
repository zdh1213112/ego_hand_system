#!/usr/bin/env python3
"""Fit one WiLoR-convention MANO model per hand/frame to calibrated views.

The same MANO forward pass creates all 21 joints and all 778 mesh vertices.
Left hands use WiLoR's right-canonical model with one X reflection. Images
remain unchanged; observations are projected with each original calibration.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import importlib.util
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from ego_data.calibration import CameraCalibration
from camera_models.kb import KB_PROJECTION_VERSION
from shared_hand_geometry import (
    associate_hands, camera_domain_penalty_torch, project_base, project_base_torch, rigid_initialization, rotation_exp,
)
from mano_conventions import WILOR_RIGHT_CANONICAL

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def mano_wrapper_path():
    root = Path(os.environ.get('EGO_WILOR_ROOT', PROJECT_ROOT / 'third_party/WiLoR'))
    return root / 'wilor/models/mano_wrapper.py'


def signature(path):
    path = Path(path).resolve()
    stat = path.stat()
    return {'path': str(path), 'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}


def load_mano(model_dir, *, device='cpu'):
    """Load the exact WiLoR MANO wrapper without loading its neural network."""
    from check_python_environment import import_chumpy_compatibly
    import_chumpy_compatibly()
    source = mano_wrapper_path()
    spec = importlib.util.spec_from_file_location('_shared_wilor_mano_wrapper', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # With pose2rot=False, supplied matrices already encode the MANO mean.
    # Never add a hand-pose mean or use the unrelated MANO_LEFT convention.
    return module.MANO(str(Path(model_dir).resolve()), is_rhand=True,
                       use_pca=False, flat_hand_mean=False).to(device).eval()


def physical_geometry(layer, parameters, side):
    result = layer(**parameters, pose2rot=False)
    mirror = result.joints.new_tensor([1. if side else -1., 1., 1.])
    return result.joints * mirror, result.vertices * mirror


def physical_faces(layer, side):
    faces = np.asarray(layer.faces)
    return torch.as_tensor(faces.copy() if side else faces[:, [0, 2, 1]].copy(), dtype=torch.int32)


def proper_rotations(value):
    """Remove small FP16 inference errors before optimizing in SO(3)."""
    u, _, vh = torch.linalg.svd(value)
    correction = torch.eye(3, device=value.device, dtype=value.dtype).expand_as(value).clone()
    correction[..., 2, 2] = torch.linalg.det(u @ vh)
    return u @ correction @ vh


def read_prediction_records(path):
    """Read once without materializing an extra copy of the entire JSON file."""
    records = {}
    with Path(path).open() as stream:
        for line in stream:
            row = json.loads(line)
            sync = int(row['sync_index'])
            if sync in records:
                raise ValueError(f'{path}: duplicate sync_index {sync}')
            ids = [int(h['hypothesis_index']) for h in row['hands']]
            if len(ids) != len(set(ids)):
                raise ValueError(f'{path}: duplicate hypothesis_index in frame {sync}')
            records[sync] = row
    return records


def validate_mano_parameters(hand, context='native prediction'):
    parameters = hand.get('mano')
    if parameters is None:
        raise ValueError(f'{context}: MANO parameters missing; rerun inference with --save-mano-params in a new directory')
    expected = {'global_orient': 9, 'hand_pose': 135, 'betas': 10}
    for name, size in expected.items():
        value = np.asarray(parameters.get(name, []))
        if value.size != size or not np.isfinite(value).all():
            raise ValueError(f'{context}: invalid MANO {name}')


def validate_synchronized_records(rows, records, camera, context):
    for row in rows:
        sync = int(row['sync_index'])
        if sync not in records:
            raise ValueError(f'{context}: missing synchronized frame {sync}')
        record = records[sync]
        if int(record['source_frame_index']) != int(row[f'{camera}_frame_index']):
            raise ValueError(f'{context}: source frame mismatch at sync_index {sync}')
        timestamp = row.get(f'{camera}_timestamp_ns')
        if timestamp is not None and int(record.get('timestamp_ns', -1)) != int(timestamp):
            raise ValueError(f'{context}: timestamp mismatch at sync_index {sync}')


def replay_model(layer, model):
    """Reconstruct saved physical geometry; also used by independent audits."""
    device = next(layer.buffers()).device
    parameters = {key: torch.as_tensor(value, dtype=torch.float32, device=device)[None]
                  for key, value in model['parameters'].items()}
    joints, vertices = physical_geometry(layer, parameters, int(model['side']))
    rotation = torch.as_tensor(model['rotation_base_from_model'], device=device)
    translation = torch.as_tensor(model['translation_base_m'], device=device)
    scale = float(model['model_scale'])
    return joints[0] * scale @ rotation.T + translation, vertices[0] * scale @ rotation.T + translation


def fit_batch(items, shape, layer, calibrations, *, device, steps):
    """Optimize articulation and a single global transform shared by all views."""
    side = items[0]['side']
    assert all(item['side'] == side for item in items)
    count, cameras = len(items), list(calibrations)
    tensor = lambda value: torch.as_tensor(value, dtype=torch.float32, device=device)
    orient0 = proper_rotations(tensor(np.stack([np.asarray(i['seed']['mano']['global_orient']).reshape(1, 3, 3)
                                               for i in items])))
    pose0 = proper_rotations(tensor(np.stack([np.asarray(i['seed']['mano']['hand_pose']).reshape(15, 3, 3)
                                             for i in items])))
    betas = tensor(shape)[None].expand(count, -1).clone()
    with torch.no_grad():
        local, _ = physical_geometry(layer, {'global_orient': orient0, 'hand_pose': pose0, 'betas': betas}, side)
    transforms = [rigid_initialization(points, item['points'], item['valid'])
                  for points, item in zip(local.cpu().numpy(), items)]
    rotation0 = tensor(np.stack([r for r, _ in transforms]))
    translation = torch.nn.Parameter(tensor(np.stack([t for _, t in transforms])))
    root_delta = torch.nn.Parameter(torch.zeros(count, 3, device=device))
    finger_delta = torch.nn.Parameter(torch.zeros(count, 15, 3, device=device))
    log_scale = torch.nn.Parameter(torch.zeros(count, 1, 1, device=device))
    observed = np.zeros((count, len(cameras), 21, 2), np.float32)
    weights = np.zeros((count, len(cameras), 21), np.float32)
    for index, item in enumerate(items):
        for camera, observation in item['selected'].items():
            c = cameras.index(camera)
            observed[index, c] = observation['pixels']
            weights[index, c] = max(.1, float(observation['hand']['confidence']))
    observed, weights = tensor(observed), tensor(weights)
    targets = tensor(np.stack([i['points'] for i in items]))
    target_mask = tensor(np.stack([i['valid'] for i in items]))
    optimizer = torch.optim.Adam([
        {'params': [translation], 'lr': .004},
        {'params': [root_delta, finger_delta], 'lr': .02},
        {'params': [log_scale], 'lr': .006},
    ])

    def decode():
        pose = rotation_exp(finger_delta) @ pose0
        parameters = {'global_orient': orient0, 'hand_pose': pose, 'betas': betas}
        joints, vertices = physical_geometry(layer, parameters, side)
        rotation, scale = rotation_exp(root_delta) @ rotation0, log_scale.exp()
        joints = joints * scale @ rotation.transpose(-1, -2) + translation[:, None]
        vertices = vertices * scale @ rotation.transpose(-1, -2) + translation[:, None]
        return joints, vertices, parameters, rotation, scale

    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        points, _, _, _, _ = decode()
        projections, domain_penalties = [], []
        for camera in cameras:
            xy, _, xyz = project_base_torch(calibrations[camera], points)
            projections.append(xy)
            domain_penalties.append(camera_domain_penalty_torch(calibrations[camera], xyz))
        residual = (torch.stack(projections, dim=1) - observed) / 10.
        robust = (torch.sqrt(1 + residual.square()) - 1).sum(-1)
        image_loss = (robust * weights).sum() / weights.sum().clamp_min(1)
        delta3d = (points - targets) / .04
        geometry_loss = ((torch.sqrt(1 + delta3d.square()) - 1).sum(-1) * target_mask).sum() / target_mask.sum().clamp_min(1)
        depth_loss = (torch.stack(domain_penalties, 1) * weights).sum() / weights.sum().clamp_min(1)
        loss = image_loss + .15 * geometry_loss + .12 * finger_delta.square().mean() + .10 * log_scale.square().mean() + 100 * depth_loss
        if not torch.isfinite(loss):
            raise FloatingPointError('non-finite shared MANO fit loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_([translation, root_delta, finger_delta, log_scale], 10.)
        optimizer.step()
        with torch.no_grad():
            log_scale.clamp_(np.log(.65), np.log(1.5))
            finger_delta.clamp_(-.8, .8)
    with torch.no_grad():
        joints, vertices, parameters, rotation, scale = decode()
    output = []
    for index, item in enumerate(items):
        output.append({
            'joints_base_m': joints[index].cpu(), 'vertices_base_m': vertices[index].cpu(),
            'parameters': {name: value[index].cpu() for name, value in parameters.items()},
            'rotation_base_from_model': rotation[index].cpu(),
            'translation_base_m': translation[index].detach().cpu(),
            'model_scale': float(scale[index, 0, 0]), 'side': side,
            'seed_camera': item['seed_camera'], 'sync_index': item['sync_index'],
            'mano_convention': WILOR_RIGHT_CANONICAL,
        })
    return output, physical_faces(layer, side)


def fit_quality(model, candidate, calibrations, threshold_px):
    points = np.asarray(model['joints_base_m'])
    views, supported = {}, []
    for camera, observation in candidate['selected'].items():
        xy, valid, _ = project_base(calibrations[camera], points)
        errors = np.linalg.norm(xy - observation['pixels'], axis=1)
        good = valid & np.isfinite(errors)
        median = float(np.median(errors[good])) if good.any() else None
        p95 = float(np.percentile(errors[good], 95)) if good.any() else None
        support = int(np.count_nonzero(good & (errors <= threshold_px)))
        trusted = support >= 16 and median is not None and median <= threshold_px * .7
        if trusted:
            supported.append(camera)
        hand = observation['hand']
        views[camera] = {'detection_index': int(hand['detection_index']),
                         'hypothesis_index': int(hand['hypothesis_index']),
                         'bbox_xyxy': hand['bbox_xyxy'], 'confidence': float(hand['confidence']),
                         'observed_joints_2d': observation['pixels'].tolist(),
                         'median_px': median, 'p95_px': p95,
                         'inlier_joint_count': support, 'supported': trusted}
    good_shape = np.isfinite(points).all() and .04 < np.linalg.norm(points - points[0], axis=1).max() < .40
    sufficient_angle = False
    if len(supported) >= 2:
        rays = [points[0] - calibrations[c].T_base_camera[:3, 3] for c in supported]
        rays = [ray / np.linalg.norm(ray) for ray in rays]
        sufficient_angle = any(np.degrees(np.arccos(np.clip(abs(a @ b), 0, 1))) >= 1.
                               for n, a in enumerate(rays) for b in rays[n + 1:])
    return bool(good_shape and sufficient_angle and len(supported) >= 2), views, supported


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('dataset', 'predictions', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--native-predictions', type=Path, help='original predictions with saved MANO parameters; defaults to predictions')
    p.add_argument('--mano-model-dir', type=Path, default=PROJECT_ROOT / 'models/mano')
    p.add_argument('--cameras', nargs='+')
    p.add_argument('--device', default='cuda')
    p.add_argument('--max-frames', type=int, default=0)
    p.add_argument('--steps', type=int, default=100)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--max-detections', type=int, default=3)
    p.add_argument('--reprojection-threshold-px', type=float, default=30.)
    args = p.parse_args()
    if args.max_frames < 0 or min(args.steps, args.batch_size, args.max_detections) < 1 or args.reprojection_threshold_px <= 0:
        p.error('invalid shared fitting options')
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    torch.set_num_threads(4)
    dataset, output = args.dataset.resolve(), args.output.resolve()
    native_root = (args.native_predictions or args.predictions).resolve()
    manifest = json.loads((dataset / 'manifest.json').read_text())
    cameras = list(dict.fromkeys(args.cameras or manifest['camera_ids']))
    if len(cameras) < 2 or not set(cameras) <= set(manifest['camera_ids']):
        p.error('shared reconstruction requires at least two available cameras')
    calibrations = {c: CameraCalibration.load(dataset / 'calibration' / f'{c}.json') for c in cameras}
    with (dataset / 'multiview_frames.csv').open() as stream:
        rows = list(csv.DictReader(stream))[:args.max_frames or None]
    if not rows:
        p.error('no synchronized frames')
    inputs = [dataset / 'manifest.json', dataset / 'multiview_frames.csv', args.mano_model_dir / 'MANO_RIGHT.pkl',
              mano_wrapper_path(), Path(__file__), PROJECT_ROOT / 'scripts/shared_hand_geometry.py']
    for camera in cameras:
        inputs.extend([dataset / 'calibration' / f'{camera}.json', args.predictions / camera / 'predictions.jsonl',
                       native_root / camera / 'predictions.jsonl'])
    config = {'algorithm': 'shared_wilor_mano_multiview_v1', 'inputs': [signature(path) for path in inputs],
              'cameras': cameras, 'frame_count': len(rows), 'steps': args.steps, 'batch_size': args.batch_size,
              'max_detections': args.max_detections, 'threshold_px': args.reprojection_threshold_px,
              'device': args.device, 'mano_model_dir': str(args.mano_model_dir.resolve()),
              'mano_convention': WILOR_RIGHT_CANONICAL, 'min_supporting_views': 2,
              'no_independent_view_fallback': True, 'camera_projection_style': KB_PROJECTION_VERSION}
    if output.exists():
        if not (output / 'summary.json').is_file() or not (output / 'shared_pose.pt').is_file():
            raise RuntimeError(f'incomplete shared output exists: {output}; use a new output directory')
        if json.loads((output / 'run_config.json').read_text()) != config:
            raise RuntimeError(f'shared configuration changed: use a new output directory instead of {output} '
                               'for refitting; to reproject saved geometry only, run render_shared_wilor_multiview.py')
        print(f'[shared MANO] reuse completed fit: {output}', flush=True)
        return
    started = time.perf_counter()
    prediction_rows, native = {}, {}
    for camera in cameras:
        path = args.predictions / camera / 'predictions.jsonl'
        prediction_rows[camera] = read_prediction_records(path)
        raw_path = native_root / camera / 'predictions.jsonl'
        raw_rows = prediction_rows[camera] if raw_path.resolve() == path.resolve() else read_prediction_records(raw_path)
        validate_synchronized_records(rows, prediction_rows[camera], camera, str(path))
        validate_synchronized_records(rows, raw_rows, camera, str(raw_path))
        native[camera] = {sync: {int(h['hypothesis_index']): h for h in row['hands']} for sync, row in raw_rows.items()}
        for frame in native[camera].values():
            for hand in frame.values():
                validate_mano_parameters(hand, context=camera)
    candidates, rejected = [], []
    for row in rows:
        sync = int(row['sync_index'])
        records = {c: prediction_rows[c][sync] for c in cameras}
        hands = associate_hands(records, calibrations, threshold_px=args.reprojection_threshold_px,
                                max_detections=args.max_detections)
        for hand in hands:
            hand['sync_index'] = sync
            seed_camera = max(hand['selected'], key=lambda c: float(hand['selected'][c]['hand']['confidence']))
            observation = hand['selected'][seed_camera]
            seed = native[seed_camera][sync][int(observation['hand']['hypothesis_index'])]
            if int(seed['is_right']) != hand['side'] or int(seed['detection_index']) != int(observation['hand']['detection_index']):
                raise ValueError('native MANO hypothesis does not match associated observation')
            hand['seed'], hand['seed_camera'] = seed, seed_camera
            candidates.append(hand)
        if not hands:
            rejected.append({'sync_index': sync, 'reason': 'insufficient_consistent_multiview_observations'})
    print(f'[shared MANO] associated {len(candidates)} hand/frame models over {len(rows)} frames', flush=True)
    shapes = {}
    for side in (0, 1):
        group = [c for c in candidates if c['side'] == side]
        if group:
            shapes[side] = np.median(np.stack([np.asarray(c['seed']['mano']['betas']).reshape(10) for c in group]), axis=0)
    layer = load_mano(args.mano_model_dir, device=args.device)
    output.mkdir(parents=True)
    (output / 'run_config.json').write_text(json.dumps(config, indent=2) + '\n')
    model_frames = {int(row['sync_index']): {'sync_index': int(row['sync_index']), 'hands': []} for row in rows}
    accepted = {index: {'sync_index': index, 'hands': []} for index in model_frames}
    topology = {'left': physical_faces(layer, 0), 'right': physical_faces(layer, 1)}
    fitted = 0
    for side in (0, 1):
        group = [c for c in candidates if c['side'] == side]
        for start in range(0, len(group), args.batch_size):
            batch = group[start:start + args.batch_size]
            models, _ = fit_batch(batch, shapes[side], layer, calibrations, device=args.device, steps=args.steps)
            for model, candidate in zip(models, batch):
                valid, views, support = fit_quality(model, candidate, calibrations, args.reprojection_threshold_px)
                if not valid:
                    rejected.append({'sync_index': model['sync_index'], 'side': side,
                                     'reason': 'shared_model_reprojection_or_geometry_failed', 'views': views})
                    continue
                model['views'], model['supporting_cameras'] = views, support
                model_frames[model['sync_index']]['hands'].append(model)
                accepted[model['sync_index']]['hands'].append({
                    'side': side, 'joints_base_m': model['joints_base_m'].tolist(),
                    'supporting_cameras': support, 'views': views,
                    'geometry_source': 'shared_mano_model', 'model_scale': model['model_scale'],
                })
            fitted += len(batch)
            print(f'[shared MANO] fitted {fitted}/{len(candidates)} models | {time.perf_counter() - started:.1f}s', flush=True)
    torch.save({'format': 'shared_wilor_mano_v1', 'coordinate_system': manifest['coordinate_system'],
                'mano_convention': WILOR_RIGHT_CANONICAL, 'topology': topology,
                'frames': list(model_frames.values())}, output / 'shared_pose.pt')
    with (output / 'accepted.jsonl').open('w') as stream:
        for row in accepted.values():
            if row['hands']:
                stream.write(json.dumps(row, allow_nan=False) + '\n')
    with (output / 'rejected.jsonl').open('w') as stream:
        for row in rejected:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
    hands = [h for row in accepted.values() for h in row['hands']]
    errors = [v['median_px'] for h in hands for v in h['views'].values() if v['supported']]
    summary = {'stage': 'shared_wilor_mano_multiview', 'frame_count': len(rows), 'candidate_hand_count': len(candidates),
               'accepted_frame_count': sum(bool(row['hands']) for row in accepted.values()),
               'accepted_hand_count': len(hands), 'camera_ids': cameras,
               'supporting_view_counts': dict(Counter(len(h['supporting_cameras']) for h in hands)),
               'supporting_view_median_error_px': float(np.median(errors)) if errors else None,
               'fit_seconds': time.perf_counter() - started,
               'skeleton_and_mesh_share_model': True, 'independent_view_fallback': False,
               'mano_convention': WILOR_RIGHT_CANONICAL,
               'identity_shape_per_side': {str(k): v.tolist() for k, v in shapes.items()},
               'rejected_hand_or_frame_reasons': dict(Counter(row['reason'] for row in rejected))}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
