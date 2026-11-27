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
from math import prod
import os
from pathlib import Path
import time

import numpy as np
import torch

from ego_hand_inference.ego_data.calibration import CameraCalibration
from ego_hand_inference.camera_models.kb import KB_PROJECTION_VERSION
from ego_hand_inference.shared_hand_geometry import (
    ASSOCIATION_VERSION, associate_hands, camera_domain_penalty_torch, project_base, project_base_torch, rigid_initialization, rotation_exp,
)
from mano_conventions import WILOR_RIGHT_CANONICAL
from wilor_native_recovery import (
    MAX_WRIST_REFINEMENT_M, RECOVERY_VERSION, aliases_claimed_glove, measure_view, physical_observations,
    recover_native_hand,
)
from wilor_source_transition import apply_source_transitions, transition_policy

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANO_PARAMETER_SHAPES = {'global_orient': (1, 3, 3), 'hand_pose': (15, 3, 3), 'betas': (10,)}
NATIVE_REPLAY_TOLERANCE_M = 1e-3  # Allow FP16 inference / FP32 MANO buffer rounding.
NATIVE_CHRONOLOGY_VERSION = 'current_native_at_published_wrist_across_recovery_v2'


def mano_wrapper_path():
    upstream = PROJECT_ROOT / 'third_party/WiLoR'
    wrapper = Path('wilor/models/mano_wrapper.py')
    default_root = upstream if (upstream / wrapper).is_file() else PROJECT_ROOT / 'training'
    return Path(os.environ.get('EGO_WILOR_ROOT', default_root)) / wrapper


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
    for name, shape in MANO_PARAMETER_SHAPES.items():
        value = np.asarray(parameters.get(name, []))
        if value.size != prod(shape) or not np.isfinite(value).all():
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
        # Wrist error remains diagnostic; assess support from the whole hand.
        trusted = bool(support >= 16 and median is not None and median <= threshold_px * .7)
        if trusted:
            supported.append(camera)
        hand = observation['hand']
        views[camera] = {'detection_index': int(hand['detection_index']),
                         'hypothesis_index': int(hand['hypothesis_index']),
                         'bbox_xyxy': hand['bbox_xyxy'], 'confidence': float(hand['confidence']),
                         'observed_joints_2d': observation['pixels'].tolist(),
                         'median_px': median, 'p95_px': p95,
                         'wrist_px': float(errors[0]) if good[0] else None,
                         'inlier_joint_count': support, 'supported': trusted}
    good_shape = np.isfinite(points).all() and .04 < np.linalg.norm(points - points[0], axis=1).max() < .40
    sufficient_angle = False
    if len(supported) >= 2:
        rays = [points[0] - calibrations[c].T_base_camera[:3, 3] for c in supported]
        rays = [ray / np.linalg.norm(ray) for ray in rays]
        sufficient_angle = any(np.degrees(np.arccos(np.clip(abs(a @ b), 0, 1))) >= 1.
                               for n, a in enumerate(rays) for b in rays[n + 1:])
    return bool(good_shape and sufficient_angle and len(supported) >= 2), views, supported


def native_hand_for_view(native, candidate, camera):
    sync = candidate['sync_index']
    observed = candidate['selected'][camera]['hand']
    index = int(observed['hypothesis_index'])
    hand = native[camera][sync].get(index)
    if (hand is None or int(hand['is_right']) != candidate['side']
            or int(hand['detection_index']) != int(observed['detection_index'])):
        raise ValueError(f'{camera} sync {sync} hypothesis {index}: native MANO does not match associated observation')
    return hand


def check_native_joint_replay(local, hand):
    """Cross-check the wrapper and native MANO parameters when native joints were exported."""
    if 'joints_3d_camera_relative' not in hand:
        return None, None
    try:
        saved = np.asarray(hand['joints_3d_camera_relative'], dtype=float)
    except (TypeError, ValueError):
        return None, 'invalid_native_joints'
    if saved.shape != (21, 3) or not np.isfinite(saved).all():
        return None, 'invalid_native_joints'
    error = float(np.abs(local.numpy() - (saved - saved[0])).max())
    if not np.isfinite(error) or error > NATIVE_REPLAY_TOLERANCE_M:
        return error, 'native_mano_replay_mismatch'
    return error, None


def choose_native_camera(scores, previous_camera, margin_px):
    """Return the chosen camera and the framewise winner among eligible sources."""
    best = min(scores, key=lambda camera: (scores[camera], camera))
    if margin_px > 0 and previous_camera in scores and scores[previous_camera] <= scores[best] + margin_px:
        return previous_camera, best
    return best, best


@torch.no_grad()
def native_pose_proposals(qualified, native, layer, calibrations, *, threshold_px, batch_size):
    """Decode and score only the native cameras qualified at the original fitted wrist."""
    requests = []
    proposals = [{} for _ in qualified]
    for index, (model, candidate) in enumerate(qualified):
        support = model['original_supporting_cameras']
        observations = {'selected': {c: candidate['selected'][c] for c in support}}
        for camera in sorted(support):
            hand = native_hand_for_view(native, candidate, camera)
            requests.append((index, observations, camera, hand))

    device = next(layer.buffers()).device
    for start in range(0, len(requests), batch_size):
        batch = requests[start:start + batch_size]
        side = qualified[batch[0][0]][0]['side']
        assert all(qualified[index][0]['side'] == side for index, _, _, _ in batch)
        # MANO's FP32 kernels can round identical poses differently for a
        # partial batch. Keep the forward shape fixed so tied cameras stay tied.
        padded = batch + [batch[-1]] * (batch_size - len(batch))
        parameters = {key: torch.as_tensor(np.stack([np.reshape(hand['mano'][key], shape)
                                                     for _, _, _, hand in padded]), dtype=torch.float32)
                      for key, shape in MANO_PARAMETER_SHAPES.items()}
        joints, vertices = physical_geometry(layer, {k: v.to(device) for k, v in parameters.items()}, side)
        joints, vertices = joints[:len(batch)].cpu(), vertices[:len(batch)].cpu()
        for i, (index, observations, camera, hand) in enumerate(batch):
            model, _ = qualified[index]
            local = joints[i] - joints[i, 0]
            replay_error, reason = check_native_joint_replay(local, hand)
            if reason:
                model.setdefault('native_pose_exclusions', {})[camera] = reason
                continue

            rotation = torch.as_tensor(calibrations[camera].T_base_camera[:3, :3], dtype=local.dtype)
            wrist = model['joints_base_m'][0]
            points = local @ rotation.T + wrist
            passes, views, support = fit_quality({'joints_base_m': points}, observations, calibrations, threshold_px)
            medians = [view['median_px'] for view in views.values()]
            if not passes or camera not in support or None in medians:
                continue
            proposals[index][camera] = dict(
                joints_base_m=points, vertices_base_m=(vertices[i] - joints[i, 0]) @ rotation.T + wrist,
                parameters={k: v[i].clone() for k, v in parameters.items()},
                rotation_base_from_model=rotation, translation_base_m=wrist - joints[i, 0] @ rotation.T,
                model_scale=1., pose_source='native_camera', pose_source_camera=camera,
                pose_source_frame_index=observations['selected'][camera]['source_frame_index'],
                pose_score_px=float(np.mean(medians)), native_joint_replay_max_abs_m=replay_error)
    return proposals


def select_native_poses(qualified, native, layer, calibrations, *, threshold_px, batch_size,
                        switch_margin_px=1., previous_by_side=None):
    """Select in sync order using only the previous camera ID and current-frame proposals."""
    if previous_by_side is None:
        previous_by_side = {}
    proposals = native_pose_proposals(qualified, native, layer, calibrations,
                                      threshold_px=threshold_px, batch_size=batch_size)
    for (model, candidate), choices in zip(qualified, proposals):
        side, sync = model['side'], model['sync_index']
        if not choices:
            previous_by_side.pop(side, None)
            continue
        last_sync, last_camera = previous_by_side.get(side, (None, None))
        previous_camera = last_camera if last_sync == sync - 1 else None
        scores = {camera: pose['pose_score_px'] for camera, pose in choices.items()}
        camera, best = choose_native_camera(scores, previous_camera, switch_margin_px)
        model.update(choices[camera], pose_min_score_px=scores[best], pose_hysteresis_retained=camera != best)
        _, model['views'], model['supporting_cameras'] = fit_quality(model, candidate, calibrations, threshold_px)
        previous_by_side[side] = (sync, camera)


@torch.no_grad()
def decode_recovery_template(layer, hand, side, camera, calibration, sync, source_frame_index):
    """Keep the source camera's native articulation, orientation, scale and shape."""
    device = next(layer.buffers()).device
    parameters = {key: torch.as_tensor(hand['mano'][key], device=device, dtype=torch.float32).reshape(1, *shape)
                  for key, shape in MANO_PARAMETER_SHAPES.items()}
    rotations = torch.cat((parameters['global_orient'], parameters['hand_pose']), dim=1)
    if (not torch.allclose(rotations @ rotations.transpose(-1, -2), torch.eye(3, device=device).expand_as(rotations),
                           rtol=0, atol=.005) or (torch.linalg.det(rotations) <= 0).any()):
        return None, 'invalid_native_rotation'
    joints, vertices = physical_geometry(layer, parameters, side)
    joints, vertices = joints[0].cpu(), vertices[0].cpu()
    if (joints.shape != (21, 3) or vertices.shape != (778, 3)
            or not torch.isfinite(joints).all() or not torch.isfinite(vertices).all()):
        return None, 'invalid_native_geometry'
    error, reason = check_native_joint_replay(joints - joints[0], hand)
    if reason:
        return None, reason
    rotation = torch.as_tensor(calibration.T_base_camera[:3, :3], dtype=joints.dtype)
    if torch.linalg.det(rotation) <= 0:
        return None, 'invalid_native_camera_rotation'
    return {
        'sync_index': sync, 'side': side, 'pose_source': 'native_camera', 'pose_source_camera': camera,
        'seed_camera': camera, 'pose_source_frame_index': source_frame_index,
        'parameters': {key: value[0].cpu() for key, value in parameters.items()},
        'rotation_base_from_model': rotation, 'model_scale': 1., 'mano_convention': WILOR_RIGHT_CANONICAL,
        'joints_base_m': joints @ rotation.T, 'vertices_base_m': vertices @ rotation.T,
        'native_joint_replay_max_abs_m': error,
    }, None


def recover_missing_sides(model_frames, rejected, prediction_rows, native, layer, calibrations, args):
    """Freeze qualified hands and use their boxes as exclusions for the missing-side pass."""
    initial_failures = {(row['sync_index'], row.get('side')): row for row in rejected}
    remaining = []
    for sync, frame in sorted(model_frames.items()):
        accepted = frame['hands']
        if len(accepted) == 2:
            continue
        records = {c: rows[sync] for c, rows in prediction_rows.items()}
        native_frame = {c: frames[sync] for c, frames in native.items()}
        for side in sorted({0, 1} - {m['side'] for m in accepted}):
            original = initial_failures.get((sync, side), {})
            observations, detected = physical_observations(records, native_frame, calibrations, side, accepted,
                                                           args.max_detections)
            if not detected and not original:
                continue

            def decode(camera, observation):
                hand = native_frame[camera][int(observation['hand']['hypothesis_index'])]
                return decode_recovery_template(layer, hand, side, camera, calibrations[camera], sync,
                                                observation['source_frame_index'])

            choices, audit = recover_native_hand(observations, calibrations, accepted,
                                                 threshold=args.reprojection_threshold_px,
                                                 decode=decode, fit_quality=fit_quality)
            audit['detected_boxes'] = detected
            audit['initial_reason'] = original.get('reason', 'insufficient_consistent_multiview_observations')
            if not choices:
                if len(detected) < 2:
                    reason = 'too_few_same_side_detection_views'
                elif len(observations) < 2:
                    reason = 'too_few_unclaimed_same_side_hypotheses'
                elif not audit['triangulated_pairs']:
                    reason = 'no_consistent_multiview_wrist_pair'
                elif not audit['replayable_sources']:
                    reason = 'no_replayable_native_pose'
                else:
                    reason = 'no_supported_native_multiview_fit'
                remaining.append({**original, 'sync_index': sync, 'side': side, 'reason': reason, 'recovery': audit})
                continue
            previous = model_frames.get(sync - 1, {}).get('hands', [])
            previous_camera = next((m['pose_source_camera'] for m in previous if m['side'] == side), None)
            scores = {c: m['pose_score_px'] for c, m in choices.items()}
            source, best = choose_native_camera(scores, previous_camera, args.pose_switch_margin_px)
            model = choices[source]
            initial_views = original.get('views', {})
            audit['initial_views'] = {c: {k: v[k] for k in ('detection_index', 'hypothesis_index', 'supported',
                                                            'median_px', 'inlier_joint_count')}
                                      for c, v in initial_views.items()}
            if 'association' in original:
                audit['initial_association'] = original['association']
            model.update(recovery=audit, original_supporting_cameras=[c for c, v in initial_views.items()
                                                                      if v['supported']],
                         pose_min_score_px=scores[best], pose_hysteresis_retained=source != best)
            accepted.append(model)
        if (not accepted and (sync, None) in initial_failures
                and (not remaining or remaining[-1]['sync_index'] != sync)):
            remaining.append(initial_failures[sync, None])
    return remaining


def boundary_pose_choices(model, others, raw_rows, layer, calibrations, threshold_px, batch_size):
    """Regenerate alternatives at this wrist, keeping all existing support and checking new boxes."""
    sync, side = model['sync_index'], model['side']
    native, selected = {}, {}
    for camera, view in model['views'].items():
        row = raw_rows[camera][sync]
        if int(row['sync_index']) != sync:
            raise ValueError(f'{camera}/{sync}: unmatched native frame in chronology replay')
        native[camera] = {sync: {int(h['hypothesis_index']): h for h in row['hands']}}
        selected[camera] = {'hand': view, 'pixels': np.asarray(view['observed_joints_2d']),
                            'source_frame_index': int(row['source_frame_index'])}
    candidate = dict(sync_index=sync, side=side, selected=selected)
    recovered = 'recovery' in model
    if recovered:
        # The current wrist was established by these two specific detections.
        # A source tied to a different pair/wrist is not interchangeable here.
        pair = model['association']['reference_cameras']
        for camera in selected:
            hand = native_hand_for_view(native, candidate, camera)
            if (hand.get('detector_is_right') != side
                    or not np.array_equal(hand['bbox_xyxy'], model['views'][camera]['bbox_xyxy'])
                    or hand.get('proposal_source', 'detector') not in ('detector', 'cached_original_detector')):
                raise ValueError(f'{camera}/{sync}: recovered hand does not match physical detection')
        proposals = {}
        wrist = model['joints_base_m'][0]
        for camera in pair:
            hand = native_hand_for_view(native, candidate, camera)
            proposal, _ = decode_recovery_template(layer, hand, side, camera, calibrations[camera], sync,
                                                   selected[camera]['source_frame_index'])
            if proposal is None:
                continue
            origin = proposal['joints_base_m'][0]
            proposal.update(translation_base_m=wrist - origin, joints_base_m=(proposal['joints_base_m'] - origin) + wrist,
                            vertices_base_m=(proposal['vertices_base_m'] - origin) + wrist)
            probes = {'selected': {c: selected[c] for c in pair}}
            valid, _, _ = fit_quality(proposal, probes, calibrations, threshold_px)
            if valid:
                proposals[camera] = proposal
    else:
        proposals, = native_pose_proposals([(model, candidate)], native, layer, calibrations,
                                           threshold_px=threshold_px, batch_size=batch_size)

    choices = {}
    for camera, proposal in proposals.items():
        if recovered:
            views = {c: measure_view(proposal, c, obs, calibrations, threshold_px, fit_quality)
                     for c, obs in selected.items()}
            support = [c for c, view in views.items() if view['supported']]
        else:
            _, views, support = fit_quality(proposal, candidate, calibrations, threshold_px)
        published = set(model['supporting_cameras'])
        if not published.issubset(support) or camera not in support:
            continue
        conflicts = False
        for c in set(support) - published:
            view = views[c]
            for other in others:
                claim = other['views'].get(c)
                if claim and (view['detection_index'] == claim['detection_index']
                              or aliases_claimed_glove(np.asarray(view['bbox_xyxy']),
                                                      np.asarray(view['observed_joints_2d']), claim)):
                    conflicts = True
        if conflicts:
            continue
        if recovered:
            proposal['pose_score_px'] = float(np.mean([max(views[c]['wrist_px'], views[c]['median_px']) for c in support]))
        choices[camera] = dict(proposal, views=views, supporting_cameras=support)
    return choices


def apply_native_source_chronology(model_frames, native_rows, layer, calibrations, *, threshold_px,
                                   switch_margin_px, batch_size):
    """Repair two-pass history at recovered-to-ordinary boundaries, without re-fitting any wrist.

    The caller may supply a completed RC2 shared pose and its matched native
    prediction rows. Revisit a successor only while its incoming source differs
    from the one used by the original passes. All evidence belongs to its own frame.
    """
    counts = {'reassessed_hands': 0, 'changed_source_hands': 0}
    previous = {}
    for sync, frame in sorted(model_frames.items()):
        for model in frame['hands']:
            side = model['side']
            if model['pose_source'] != 'native_camera':
                previous.pop(side, None)
                continue
            recovered = 'recovery' in model
            if recovered:
                for other in frame['hands']:
                    if other['side'] == side:
                        continue
                    for c in set(model['supporting_cameras']).intersection(other['views']):
                        view = model['views'][c]
                        if aliases_claimed_glove(np.asarray(view['bbox_xyxy']), np.asarray(view['observed_joints_2d']),
                                                other['views'][c]):
                            raise ValueError(f'{c}/{sync}: saved recovered hand aliases another glove; rerun recovery')
            camera = model['pose_source_camera']
            prior_sync, prior_camera, prior_recovered, prior_changed = previous.get(side, (None, None, False, False))
            affected = prior_sync == sync - 1 and (prior_changed or (prior_recovered and not recovered))
            original = model.get('source_chronology', {}).get('original_source_camera', camera)
            if affected and switch_margin_px > 0:
                others = [h for h in frame['hands'] if h['side'] != side]
                choices = boundary_pose_choices(model, others, native_rows, layer, calibrations, threshold_px, batch_size)
                scores = {c: p['pose_score_px'] for c, p in choices.items()}
                # This saved current-frame geometry already passed every gate.
                # Keep its exact score/geometry even if replay rounds differently.
                scores[camera] = model['pose_score_px']
                if recovered and choices:
                    strongest = max(len(p['supporting_cameras']) for p in choices.values())
                    scores = {c: score for c, score in scores.items()
                              if len(choices.get(c, model)['supporting_cameras']) >= strongest}
                chosen, best = choose_native_camera(scores, prior_camera, switch_margin_px)
                if chosen != camera:
                    model.update(choices[chosen])
                model.update(pose_min_score_px=scores[best], pose_hysteresis_retained=chosen != best,
                             source_chronology=dict(previous_camera=prior_camera, original_source_camera=original,
                                                    qualified_scores_px=scores))
                camera = chosen
                counts['reassessed_hands'] += 1
                counts['changed_source_hands'] += int(camera != original)
            previous[side] = sync, camera, recovered, camera != original
    return counts


def accepted_hand(model):
    return {
        'joints_base_m': model['joints_base_m'].tolist(), 'geometry_source': 'shared_mano_model',
        **{key: model[key] for key in ('side', 'supporting_cameras', 'views', 'association', 'model_scale',
                                       'original_supporting_cameras', 'pose_source', 'pose_source_camera',
                                       'pose_score_px')},
        **{key: model[key] for key in ('pose_source_frame_index', 'pose_min_score_px', 'pose_hysteresis_retained', 'pose_transition',
                                       'native_pose_exclusions', 'native_joint_replay_max_abs_m', 'recovery',
                                       'source_chronology')
           if key in model},
    }


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
    p.add_argument('--pose-policy', choices=('native-wrist', 'shared-fit'), default='native-wrist',
                   help='native-wrist selects a supporting WiLoR articulation at the qualified shared wrist '
                        '(default); shared-fit retains the fitted MANO articulation')
    p.add_argument('--pose-switch-margin-px', type=float, default=1.,
                   help='keep the previous native camera when its current score is within this many pixels '
                        'of the best (default 1, 0 selects the framewise minimum with lexical ties)')
    p.add_argument('--no-native-recovery', action='store_true',
                   help='disable the missing-hand recovery pass for the native-wrist policy')
    p.add_argument('--no-pose-transition', action='store_true',
                   help='save unblended source poses for replay with transition_wilor_sources.py')
    p.add_argument('--max-detections', type=int, default=3)
    p.add_argument('--reprojection-threshold-px', type=float, default=30.)
    args = p.parse_args()
    if (args.max_frames < 0 or min(args.steps, args.batch_size, args.max_detections) < 1
            or not np.isfinite(args.reprojection_threshold_px) or args.reprojection_threshold_px <= 0):
        p.error('invalid shared fitting options')
    if not np.isfinite(args.pose_switch_margin_px) or args.pose_switch_margin_px < 0:
        p.error('--pose-switch-margin-px must be finite and nonnegative')
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
              mano_wrapper_path(), Path(__file__), Path(__file__).with_name('shared_hand_geometry.py')]
    recovery_enabled = args.pose_policy == 'native-wrist' and not args.no_native_recovery
    transition_enabled = args.pose_policy == 'native-wrist' and not args.no_pose_transition
    if transition_enabled:
        inputs.append(PROJECT_ROOT / 'scripts/wilor_source_transition.py')
    if recovery_enabled:
        inputs.extend([PROJECT_ROOT / 'scripts/wilor_native_recovery.py', PROJECT_ROOT / 'scripts/fusion_temporal_guard.py'])
    for camera in cameras:
        inputs.extend([dataset / 'calibration' / f'{camera}.json', args.predictions / camera / 'predictions.jsonl',
                       native_root / camera / 'predictions.jsonl'])
    config = {'algorithm': 'shared_wilor_mano_multiview_v1', 'inputs': [signature(path) for path in inputs],
              'pose_policy': args.pose_policy,
              'cameras': cameras, 'frame_count': len(rows), 'steps': args.steps, 'batch_size': args.batch_size,
              'max_detections': args.max_detections, 'threshold_px': args.reprojection_threshold_px,
              'device': args.device, 'mano_model_dir': str(args.mano_model_dir.resolve()),
              'mano_convention': WILOR_RIGHT_CANONICAL, 'min_supporting_views': 2,
              'no_independent_view_fallback': True, 'camera_projection_style': KB_PROJECTION_VERSION,
              'association_version': ASSOCIATION_VERSION, 'fit_quality': {'wrist_hard_gate': False}}
    if args.pose_policy == 'native-wrist':
        config['native_pose_transition'] = dict(transition_policy(), enabled=transition_enabled)
        if recovery_enabled:
            config['algorithm'] = 'shared_wilor_mano_native_wrist_multiview_v2'
            config['native_source_chronology'] = NATIVE_CHRONOLOGY_VERSION
        config['native_recovery'] = {'enabled': recovery_enabled, 'version': RECOVERY_VERSION,
                                     'require_source_support': True, 'min_same_side_detection_views': 2,
                                     'selection': 'max_supported_views_then_mean_max_of_wrist_and_median_px',
                                     'max_wrist_refinement_m': MAX_WRIST_REFINEMENT_M,
                                     'wrist_priority': 'triangulated_pair_before_bounded_refinement',
                                     'scope': 'missing_side_after_original_fits'}
        config['native_pose_selection'] = {
            'candidate_views': 'original_shared_fit_support', 'require_source_support': True,
            'score': 'mean_of_original_support_view_medians_px', 'tie_break': 'camera_id_ascending',
            'pose_switch_margin_px': args.pose_switch_margin_px,
            'retention': 'previous_native_camera_same_side_at_sync_minus_one_within_inclusive_margin',
            'zero_margin': 'framewise_minimum',
            'geometry': 'native_MANO_parameters_FP32_replay_at_fitted_wrist', 'scale': 1.,
            'native_joint_replay_tolerance_m': NATIVE_REPLAY_TOLERANCE_M,
        }
    if output.exists():
        if not (output / 'summary.json').is_file() or not (output / 'shared_pose.pt').is_file():
            raise RuntimeError(f'incomplete shared output exists: {output}; use a new output directory')
        if json.loads((output / 'run_config.json').read_text()) != config:
            raise RuntimeError(f'shared configuration changed: use a new output directory instead of {output} '
                               'for refitting; to reproject saved geometry only, run render_shared_wilor_multiview.py')
        print(f'[shared MANO] reuse completed fit: {output}', flush=True)
        return
    started = time.perf_counter()
    prediction_rows, native, native_rows = {}, {}, {}
    for camera in cameras:
        path = args.predictions / camera / 'predictions.jsonl'
        prediction_rows[camera] = read_prediction_records(path)
        raw_path = native_root / camera / 'predictions.jsonl'
        raw_rows = prediction_rows[camera] if raw_path.resolve() == path.resolve() else read_prediction_records(raw_path)
        native_rows[camera] = raw_rows
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
            hand['seed'], hand['seed_camera'] = native_hand_for_view(native, hand, seed_camera), seed_camera
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
    topology = {'left': physical_faces(layer, 0), 'right': physical_faces(layer, 1)}
    fitted = 0
    for side in (0, 1):
        group = [c for c in candidates if c['side'] == side]
        qualified = []
        for start in range(0, len(group), args.batch_size):
            batch = group[start:start + args.batch_size]
            models, _ = fit_batch(batch, shapes[side], layer, calibrations, device=args.device, steps=args.steps)
            for model, candidate in zip(models, batch):
                valid, views, support = fit_quality(model, candidate, calibrations, args.reprojection_threshold_px)
                if not valid:
                    rejected.append({'sync_index': model['sync_index'], 'side': side,
                                     'reason': 'shared_model_reprojection_or_geometry_failed', 'views': views,
                                     **({'association': candidate['association']} if recovery_enabled else {})})
                    continue
                model.update(views=views, supporting_cameras=support, original_supporting_cameras=support,
                             pose_source='shared_fit', pose_source_camera=None, pose_score_px=None)
                model['association'] = candidate['association']
                model_frames[model['sync_index']]['hands'].append(model)
                if args.pose_policy == 'native-wrist':
                    qualified.append((model, candidate))
            fitted += len(batch)
            print(f'[shared MANO] fitted {fitted}/{len(candidates)} models | {time.perf_counter() - started:.1f}s', flush=True)
        # Preserve the original optimizer batches even when the CSV rows are out of order.
        qualified.sort(key=lambda pair: pair[0]['sync_index'])
        previous_by_side = {}
        for start in range(0, len(qualified), args.batch_size):
            select_native_poses(qualified[start:start + args.batch_size], native, layer, calibrations,
                                threshold_px=args.reprojection_threshold_px, batch_size=args.batch_size,
                                switch_margin_px=args.pose_switch_margin_px, previous_by_side=previous_by_side)
    if recovery_enabled:
        rejected = recover_missing_sides(model_frames, rejected, prediction_rows, native, layer, calibrations, args)
        chronology_counts = apply_native_source_chronology(model_frames, native_rows, layer, calibrations,
                                                           threshold_px=args.reprojection_threshold_px,
                                                           switch_margin_px=args.pose_switch_margin_px,
                                                           batch_size=args.batch_size)
    if transition_enabled:
        transition_counts = apply_source_transitions(model_frames.values(), layer, calibrations,
                                                     args.reprojection_threshold_px, fit_quality, physical_geometry)
    accepted = {sync: {'sync_index': sync, 'hands': [accepted_hand(m) for m in frame['hands']]}
                for sync, frame in model_frames.items()}
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
               'pose_policy': args.pose_policy, 'pose_source_counts': dict(Counter(h['pose_source'] for h in hands)),
               'native_pose_exclusion_counts': dict(Counter(
                   reason for h in hands for reason in h.get('native_pose_exclusions', {}).values())),
               'supporting_view_counts': dict(Counter(len(h['supporting_cameras']) for h in hands)),
               'supporting_view_median_error_px': float(np.median(errors)) if errors else None,
               'fit_seconds': time.perf_counter() - started,
               'skeleton_and_mesh_share_model': True, 'independent_view_fallback': False,
               'association_version': ASSOCIATION_VERSION,
               'fit_quality': config['fit_quality'],
               'mano_convention': WILOR_RIGHT_CANONICAL,
               'identity_shape_per_side': {str(k): v.tolist() for k, v in shapes.items()},
               'identity_shape_scope': 'original_shared_fit; selected native poses use their source betas',
               'rejected_hand_or_frame_reasons': dict(Counter(row['reason'] for row in rejected))}
    if recovery_enabled:
        summary['native_source_chronology'] = chronology_counts
        summary['recovered_hand_count'] = sum('recovery' in h for h in hands)
        summary['recovery_initial_reasons'] = dict(Counter(h['recovery']['initial_reason'] for h in hands
                                                          if 'recovery' in h))
    if transition_enabled:
        summary['native_pose_transition'] = transition_counts
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
