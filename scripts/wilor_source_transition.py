"""Briefly blend native MANO parameters when the selected source camera changes."""
from collections import Counter

import numpy as np
from scipy.spatial.transform import Rotation
import torch

from ego_hand_inference.shared_hand_geometry import project_base


# Each coefficient weights the previous published pose at this age of the switch.
PREVIOUS_WEIGHTS = (.70, .60, .45, .25, .10)
MEDIAN_SLACK_PX = 1.
TIP_SLACK_PX = 2.
TIPS = [4, 8, 12, 16, 20]


def transition_policy():
    return dict(version='native_switch_so3_v1', previous_weights=list(PREVIOUS_WEIGHTS),
                median_slack_px=MEDIAN_SLACK_PX, mean_tip_slack_px=TIP_SLACK_PX,
                backoff_weight_factor=.5, shape='current_target', recovered_hands='reset')


def interpolate_rotations(current, previous, weight):
    """Move from each current rotation toward its predecessor along the short arc."""
    current = Rotation.from_matrix(np.asarray(current, dtype=np.float64))
    previous = Rotation.from_matrix(np.asarray(previous, dtype=np.float64))
    difference = (current.inv() * previous).as_rotvec()
    return torch.from_numpy((current * Rotation.from_rotvec(difference * weight)).as_matrix()).float()


def blend_geometry(previous, target, weight, layer, physical_geometry):
    # Express the previous wrist orientation in the target's right-canonical
    # camera coordinates. Conjugation by the physical left reflection occurs
    # on both sides of the inter-camera rotation. Finger rotations are local.
    side = target['side']
    mirror = torch.diag(torch.tensor([1. if side else -1., 1., 1.]))
    rotation = target['rotation_base_from_model']
    change_camera = mirror @ rotation.T @ previous['rotation_base_from_model'] @ mirror
    raw, prior = target['parameters'], previous['parameters']
    # Historical predictions store FP16 matrices. A finite but degenerate or
    # reflected matrix can be decoded by MANO but has no physical SO(3) arc.
    rotations = np.concatenate((raw['global_orient'], raw['hand_pose'], prior['global_orient'],
                                prior['hand_pose'], rotation[None], previous['rotation_base_from_model'][None]))
    if (not np.isfinite(rotations).all() or np.any(np.linalg.det(rotations) <= 0)
            or np.max(np.abs(rotations.swapaxes(1, 2) @ rotations - np.eye(3))) > .02):
        return None
    parameters = {
        'global_orient': interpolate_rotations(raw['global_orient'], change_camera @ prior['global_orient'], weight),
        'hand_pose': interpolate_rotations(raw['hand_pose'], prior['hand_pose'], weight),
        'betas': raw['betas'],
    }
    device = next(layer.buffers()).device
    joints, vertices = physical_geometry(layer, {k: v[None].to(device) for k, v in parameters.items()}, side)
    joints, vertices = joints[0].cpu(), vertices[0].cpu()
    if not torch.isfinite(joints).all() or not torch.isfinite(vertices).all():
        return None
    wrist = target['joints_base_m'][0]
    return dict(parameters=parameters, translation_base_m=wrist - joints[0] @ rotation.T,
                joints_base_m=(joints - joints[0]) @ rotation.T + wrist,
                vertices_base_m=(vertices - joints[0]) @ rotation.T + wrist)


def tip_score(points, observations, calibrations):
    errors = []
    for camera, view in observations.items():
        pixels, valid, _ = project_base(calibrations[camera], np.asarray(points)[TIPS])
        if not valid.all():
            return np.inf
        errors.extend(np.linalg.norm(pixels - view['pixels'][TIPS], axis=1))
    return float(np.mean(errors))


def transition_hand(previous, target, others, age, layer, calibrations, threshold_px, fit_quality, physical_geometry):
    views = {cam: {'pixels': np.asarray(view['observed_joints_2d']), 'hand': view}
             for cam, view in target['views'].items()}
    original = {cam: views[cam] for cam in target['original_supporting_cameras']}
    camera = target['pose_source_camera']
    raw_score = target['pose_score_px']
    raw_tips = tip_score(target['joints_base_m'], original, calibrations)
    if not np.isfinite(raw_tips):
        return False
    for weight in (PREVIOUS_WEIGHTS[age], PREVIOUS_WEIGHTS[age] * .5):
        proposal = blend_geometry(previous, target, weight, layer, physical_geometry)
        if proposal is None:
            return False
        passes, metrics, support = fit_quality(proposal, {'selected': original}, calibrations, threshold_px)
        if not passes or camera not in support or any(v['median_px'] is None for v in metrics.values()):
            continue
        score = float(np.mean([v['median_px'] for v in metrics.values()]))
        if score > raw_score + MEDIAN_SLACK_PX:
            continue
        tips = tip_score(proposal['joints_base_m'], original, calibrations)
        if not np.isfinite(tips) or tips > raw_tips + TIP_SLACK_PX:
            continue
        _, published, supported = fit_quality(proposal, {'selected': views}, calibrations, threshold_px)
        if not set(target['supporting_cameras']).issubset(supported):
            continue
        if any(published[c]['detection_index'] == other['views'][c]['detection_index']
               for other in others for c in set(supported).intersection(other['supporting_cameras'])):
            continue
        target.update(proposal, views=published, supporting_cameras=supported, pose_score_px=score,
                      pose_transition=dict(switch_sync_index=target['sync_index'] - age, age=age,
                                           previous_source_camera=previous['pose_source_camera'],
                                           target_source_camera=camera, previous_weight=weight,
                                           raw_pose_score_px=raw_score, raw_tip_score_px=raw_tips,
                                           tip_score_px=tips))
        return True
    return False


@torch.no_grad()
def apply_source_transitions(frames, layer, calibrations, threshold_px, fit_quality, physical_geometry):
    """Modify eligible models in chronological order using only present and past frames."""
    previous_by_side = {}
    counts = Counter()
    for frame in sorted(frames, key=lambda f: f['sync_index']):
        sync = frame['sync_index']
        for model in frame['hands']:
            side = model['side']
            if model['pose_source'] != 'native_camera' or 'recovery' in model:
                previous_by_side.pop(side, None)
                continue
            previous = previous_by_side.get(side)
            if previous is not None and previous['sync_index'] == sync - 1:
                switched = previous['pose_source_camera'] != model['pose_source_camera']
                age = 0 if switched else previous.get('pose_transition', {}).get('age', len(PREVIOUS_WEIGHTS)) + 1
                if age < len(PREVIOUS_WEIGHTS):
                    others = [other for other in frame['hands'] if other['side'] != side]
                    accepted = transition_hand(previous, model, others, age, layer, calibrations,
                                               threshold_px, fit_quality, physical_geometry)
                    counts['switches' if switched else 'followup_attempts'] += 1
                    counts['applied' if accepted else 'kept_raw'] += 1
            previous_by_side[side] = model
    return dict(counts)
