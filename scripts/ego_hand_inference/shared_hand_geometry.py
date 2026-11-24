"""Calibrated association/projection for shared MANO hand fitting."""
from __future__ import annotations

import itertools
import numpy as np

from ego_hand_inference.camera_models import project_points, unproject_points
from ego_hand_inference.camera_models.kb import theta_limit


ASSOCIATION_VERSION = 'wrist_gated_incremental_consensus_v2'


def project_base(camera, points):
    xyz = (np.asarray(points) - camera.T_base_camera[:3, 3]) @ camera.T_base_camera[:3, :3]
    pixels, valid = project_points(camera, xyz)
    return pixels, valid, xyz


def intersect_observations(observations):
    """Batched least-squares ray intersection for the same 21 joint indices."""
    directions = np.stack([o['rays'] for o in observations])
    centers = np.stack([o['center'] for o in observations])
    projectors = np.eye(3) - directions[..., :, None] * directions[..., None, :]
    matrix = projectors.sum(axis=0)
    target = np.einsum('vjkl,vl->jk', projectors, centers)
    points = np.linalg.solve(matrix + 1e-10 * np.eye(3), target[..., None])[..., 0]
    angles = [np.degrees(np.arccos(np.clip(np.abs((a * b).sum(axis=-1)), 0, 1)))
              for a, b in itertools.combinations(directions, 2)]
    angle = np.max(angles, axis=0)
    forward = np.all(np.einsum('vjk,vjk->vj', points[None] - centers[:, None], directions) > 0, axis=0)
    return points, angle, forward


def prepare_observations(records, calibrations, max_detections=3):
    output = {}
    for camera, row in records.items():
        cal = calibrations[camera]
        scores = {int(h['detection_index']): float(h['confidence']) for h in row.get('hands', [])}
        keep = sorted(scores, key=scores.get, reverse=True)[:max_detections]
        output[camera] = []
        for hand in row.get('hands', []):
            if int(hand['detection_index']) not in keep:
                continue
            pixels = np.asarray(hand['joints_2d'], dtype=float)
            rays, valid = unproject_points(cal, pixels)
            if pixels.shape != (21, 2) or not valid.all():
                continue
            output[camera].append({
                'camera': camera, 'hand': hand, 'pixels': pixels,
                'rays': rays @ cal.T_base_camera[:3, :3].T,
                'center': cal.T_base_camera[:3, 3],
                'source_frame_index': int(row['source_frame_index']),
            })
    return output


def _residual(points, observation, calibrations):
    pixels, valid, _ = project_base(calibrations[observation['camera']], points)
    errors = np.linalg.norm(pixels - observation['pixels'], axis=1)
    errors[~valid] = np.inf
    return errors


def _consistent_geometry(selected, calibrations, threshold_px, min_angle_deg):
    """Accept a view set only if its triangulation preserves wrist consensus."""
    values = list(selected.values())
    points, angle, forward = intersect_observations(values)
    errors = np.stack([_residual(points, observation, calibrations) for observation in values])
    good = forward & (angle >= min_angle_deg) & ((errors <= threshold_px).sum(axis=0) >= 2)
    # A good median can hide one catastrophic wrist. Every participating view
    # must support the root, which defines both hand identity and placement.
    if good.sum() < 12 or not good[0] or np.any(errors[:, 0] > threshold_px):
        return None
    radius = np.linalg.norm(points[good] - points[0], axis=1).max()
    median = float(np.median(errors[:, good]))
    if (not .04 < radius < .40 or np.linalg.norm(points[0]) > 3.
            or median > threshold_px or np.any(np.median(errors[:, good], axis=1) > threshold_px)):
        return None
    return points, angle, good, median


def _association_audit(candidate, observations, calibrations, threshold_px):
    excluded = []
    for camera, options in observations.items():
        if camera in candidate['selected']:
            continue
        alternatives = []
        for observation in options:
            if int(observation['hand']['is_right']) != candidate['side']:
                continue
            errors = _residual(candidate['points'], observation, calibrations)
            median = float(np.median(errors[candidate['valid']]))
            alternatives.append((median, observation, errors))
        if not alternatives:
            excluded.append({'camera': camera, 'reason': 'no_valid_hypothesis'})
            continue
        median, observation, errors = min(alternatives, key=lambda item: item[0])
        hand = observation['hand']
        reason = ('wrist_reprojection_outlier' if errors[0] > threshold_px else
                  'joint_reprojection_outlier' if median > threshold_px else 'inconsistent_triangulation')
        excluded.append({'camera': camera, 'detection_index': int(hand['detection_index']),
                         'hypothesis_index': int(hand['hypothesis_index']), 'reason': reason,
                         'wrist_px': float(errors[0]) if np.isfinite(errors[0]) else None,
                         'median_px': median if np.isfinite(median) else None})
    return {'version': ASSOCIATION_VERSION, 'wrist_threshold_px': float(threshold_px),
            'reference_cameras': list(candidate['selected']), 'excluded_views': excluded}


def associate_hands(records, calibrations, *, threshold_px=30.0, min_angle_deg=1.0, max_detections=3):
    """Associate one or two hands independently; sides are soft detector evidence.

    Both handedness hypotheses compete, but a detector box can belong to
    only one physical hand. Never require both hands to be visible together.
    Extend each seed with one consistent view at a time; a rejected extension
    must not destroy the valid subset that already reconstructs the hand.
    """
    observations = prepare_observations(records, calibrations, max_detections)
    candidates = {}
    for first, second in itertools.combinations(observations, 2):
        for a, b in itertools.product(observations[first], observations[second]):
            side = int(a['hand']['is_right'])
            if side != int(b['hand']['is_right']):
                continue
            selected = {first: a, second: b}
            geometry = _consistent_geometry(selected, calibrations, threshold_px, min_angle_deg)
            if geometry is None:
                continue
            points, angle, good, median = geometry
            for camera, options in observations.items():
                if camera in selected:
                    continue
                choices = []
                for observation in options:
                    if int(observation['hand']['is_right']) != side:
                        continue
                    errors = _residual(points, observation, calibrations)
                    err = errors[good]
                    if (errors[0] <= threshold_px and np.median(err) <= threshold_px
                            and (err <= threshold_px * 1.5).sum() >= 10):
                        choices.append((float(np.median(err)), observation))
                for _, observation in sorted(choices, key=lambda item: item[0]):
                    trial = {**selected, camera: observation}
                    geometry = _consistent_geometry(trial, calibrations, threshold_px, min_angle_deg)
                    if geometry is not None:
                        selected = trial
                        points, angle, good, median = geometry
                        break
            votes = [float(o['hand']['confidence']) * (1 if o['hand'].get('detector_is_right') == side else -1)
                     for o in selected.values()]
            score = 8 * len(selected) + good.sum() - median / 2 + 4 * sum(votes)
            key = (side, tuple(sorted((c, int(o['hand']['detection_index'])) for c, o in selected.items())))
            candidate = dict(side=side, selected=selected, points=points, valid=good,
                             ray_angles_deg=angle, score=float(score), median_px=median)
            if key not in candidates or candidates[key]['score'] < score:
                candidates[key] = candidate
    chosen, used_boxes = [], set()
    for candidate in sorted(candidates.values(), key=lambda c: c['score'], reverse=True):
        if any(c['side'] == candidate['side'] for c in chosen):
            continue
        boxes = {(c, int(o['hand']['detection_index'])) for c, o in candidate['selected'].items()}
        if used_boxes & boxes:
            continue
        if any(np.linalg.norm(candidate['points'][0] - other['points'][0]) < 0.06 for other in chosen):
            continue
        chosen.append(candidate)
        used_boxes.update(boxes)
    for candidate in chosen:
        candidate['association'] = _association_audit(candidate, observations, calibrations, threshold_px)
    return chosen


def rigid_initialization(source, target, valid):
    """Proper rotation and wrist translation; no reflection or independent joints."""
    x, y = source[valid], target[valid]
    xc, yc = x.mean(axis=0), y.mean(axis=0)
    u, _, vt = np.linalg.svd((x - xc).T @ (y - yc))
    correction = np.eye(3)
    correction[-1, -1] = np.linalg.det(vt.T @ u.T)
    rotation = vt.T @ correction @ u.T
    translation = yc - xc @ rotation.T
    return rotation, translation


def rotation_exp(vectors):
    import torch
    x, y, z = vectors.unbind(-1)
    zero = torch.zeros_like(x)
    skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), dim=-1).reshape(*vectors.shape[:-1], 3, 3)
    angle = torch.linalg.vector_norm(vectors, dim=-1)[..., None, None]
    return (torch.eye(3, device=vectors.device, dtype=vectors.dtype)
            + torch.sinc(angle / torch.pi) * skew
            + 0.5 * torch.sinc(angle / (2 * torch.pi)).square() * (skew @ skew))


def project_base_torch(camera, points):
    """Differentiable counterpart of the repository's KB / DS / pinhole models."""
    import torch
    rotation = torch.as_tensor(camera.T_base_camera[:3, :3], device=points.device, dtype=points.dtype)
    center = torch.as_tensor(camera.T_base_camera[:3, 3], device=points.device, dtype=points.dtype)
    xyz = (points - center) @ rotation
    x, y, z = xyz.unbind(-1)
    if camera.model == 'KB':
        r = torch.sqrt(x.square() + y.square() + 1e-16)
        theta = torch.atan2(r, z)
        theta2 = theta.square()
        k1, k2, k3, k4 = camera.distortion
        radial = theta * (1 + theta2 * (k1 + theta2 * (k2 + theta2 * (k3 + theta2 * k4)))) / r
        u, v = camera.fx * x * radial + camera.cx, camera.fy * y * radial + camera.cy
        valid = (torch.isfinite(xyz).all(dim=-1)
                 & (torch.linalg.vector_norm(xyz, dim=-1) > 1e-12)
                 & ((x != 0) | (y != 0) | (z > 0)) & (theta < theta_limit(camera)))
    elif camera.model == 'DS':
        fx, fy, cx, cy, xi, alpha = camera.distortion
        d1 = torch.linalg.vector_norm(xyz, dim=-1).clamp_min(1e-8)
        z1 = alpha * d1 + (1 - alpha) * z
        d2 = torch.sqrt(x.square() + y.square() + z1.square() + 1e-16)
        denominator = xi * d2 + z1
        u, v = fx * x / denominator.clamp_min(1e-6) + cx, fy * y / denominator.clamp_min(1e-6) + cy
        valid = denominator > 1e-6
    else:
        u, v = camera.fx * x / z.clamp_min(1e-6) + camera.cx, camera.fy * y / z.clamp_min(1e-6) + camera.cy
        valid = z > 1e-6
    return torch.stack((u, v), dim=-1), valid, xyz


def camera_domain_penalty_torch(camera, xyz):
    """Keep fits in the camera's domain without forcing KB rays into z > 0."""
    import torch
    if camera.model == 'KB':
        distance = torch.linalg.vector_norm(xyz, dim=-1)
        radius = torch.sqrt(xyz[..., 0].square() + xyz[..., 1].square() + 1e-16)
        theta = torch.atan2(radius, xyz[..., 2])
        outside = torch.relu(theta - (theta_limit(camera) - 1e-6)) * distance
        return torch.relu(.02 - distance).square() + outside.square()
    # Preserve the existing depth constraint for the other camera models.
    return torch.relu(.02 - xyz[..., 2]).square()
