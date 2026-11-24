"""Recover a missing physical hand from current-frame detector boxes and native MANO poses."""
from __future__ import annotations

from collections import Counter
from itertools import combinations, product

import numpy as np
import torch
from scipy.optimize import least_squares

from ego_hand_inference.shared_hand_geometry import intersect_observations, prepare_observations, project_base
from fusion_temporal_guard import _bbox_iou


RECOVERY_VERSION = 'detector_pair_native_at_triangulated_wrist_v3'
MAX_WRIST_REFINEMENT_M = .01


def aliases_claimed_glove(box, pixels, claimed):
    """Overlapping gloves remain distinct when their image wrists or landmarks separate."""
    other_box = claimed.get('bbox_xyxy')
    if other_box is None or _bbox_iou(box, other_box) < .8:
        return False
    other = np.asarray(claimed.get('observed_joints_2d', []))
    if other.shape != (21, 2):
        return False
    distances = np.linalg.norm(pixels - other, axis=1)
    tolerance = .10 * min(np.linalg.norm(box[2:] - box[:2]),
                          np.linalg.norm(np.subtract(other_box[2:], other_box[:2])))
    return bool(distances[0] <= tolerance and np.median(distances) <= tolerance)


def physical_observations(records, native_frame, calibrations, side, accepted, max_detections):
    """Join observations to unclaimed physical detector boxes and their exact native hypotheses."""
    reserved = {(camera, view['detection_index']) for model in accepted
                for camera, view in model['views'].items()}
    detected, eligible = {}, {}
    for camera, row in records.items():
        raw = native_frame[camera]
        physical = {i: h for i, h in raw.items() if h.get('detector_is_right') == side
                    and h.get('proposal_source', 'detector') in ('detector', 'cached_original_detector')}
        boxes = {int(h['detection_index']) for h in physical.values()}
        if boxes:
            detected[camera] = sorted(boxes)
        scores = {int(h['detection_index']): float(h['confidence']) for h in row['hands']}
        keep = set(sorted(scores, key=scores.get, reverse=True)[:max_detections])
        hands = []
        for hand in row['hands']:
            index, detection = int(hand['hypothesis_index']), int(hand['detection_index'])
            source = physical.get(index)
            if (detection not in keep or (camera, detection) in reserved or source is None
                    or int(source['detection_index']) != detection
                    or int(hand['is_right']) != side or int(source['is_right']) != side
                    or hand.get('detector_is_right') != side):
                continue
            box = np.asarray(hand['bbox_xyxy'], dtype=float)
            xy = np.asarray(hand['joints_2d'], dtype=float)
            width, height = calibrations[camera].image_size
            if (box.shape != (4,) or not np.isfinite(box).all() or not np.array_equal(box, source['bbox_xyxy'])
                    or not (0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height)
                    or not 0 <= float(hand['confidence']) <= 1
                    or xy.shape != (21, 2) or not np.isfinite(xy).all()):
                continue
            if any(camera in model['views'] and aliases_claimed_glove(box, xy, model['views'][camera])
                   for model in accepted):
                continue
            hands.append(hand)
        eligible[camera] = dict(row, hands=hands)
    observations = prepare_observations(eligible, calibrations, max_detections)
    return {cam: options for cam, options in observations.items() if options}, detected


def triangulate_pair(pair, calibrations, threshold):
    """Keep the association wrist, ray-angle, distance and per-joint image consensus gates.

    Finger depth can be poorly conditioned even with a well-localized wrist. The
    physical hand-size gate is applied to the MANO model in fit_quality.
    """
    values = list(pair.values())
    points, angle, forward = intersect_observations(values)
    errors = []
    for camera, observation in pair.items():
        xy, valid, _ = project_base(calibrations[camera], points)
        errors.append(np.where(valid, np.linalg.norm(xy - observation['pixels'], axis=1), np.inf))
    errors = np.stack(errors)
    good = np.isfinite(points).all(1) & forward & (angle >= 1.) & (errors.max(axis=0) <= threshold)
    if not good[0] or good.sum() < 12 or np.linalg.norm(points[0]) > 3.:
        return None
    return points[0], {
        'reference_cameras': list(pair), 'wrist_base_m': points[0].tolist(),
        'wrist_angle_deg': float(angle[0]), 'consistent_joint_count': int(good.sum()),
        'wrist_errors_px': dict(zip(pair, errors[:, 0].tolist())),
        'free_joint_radius_m': float(np.linalg.norm(points[good] - points[0], axis=1).max()),
    }


def refine_wrist(relative, pair, wrist, calibrations, threshold):
    """Use the fingers to correct a noisy wrist by at most one centimeter."""

    def residual(position):
        parts = []
        for camera, observation in pair.items():
            xy, valid, _ = project_base(calibrations[camera], relative + position)
            parts.append(np.where(valid[:, None], xy - observation['pixels'], 10 * threshold))
        return np.concatenate(parts).ravel()

    fit = least_squares(residual, wrist, loss='soft_l1', f_scale=threshold / 3., max_nfev=40)
    if not fit.success or not np.isfinite(fit.x).all():
        return None
    delta = fit.x - wrist
    return wrist + delta * min(1., MAX_WRIST_REFINEMENT_M / max(1e-12, np.linalg.norm(delta)))


def place_template(template, wrist):
    joints = template['joints_base_m']
    translation = torch.as_tensor(wrist, dtype=joints.dtype) - joints[0]
    return dict(template, translation_base_m=translation, joints_base_m=joints + translation,
                vertices_base_m=template['vertices_base_m'] + translation)


def measure_view(model, camera, observation, calibrations, threshold, fit_quality):
    """Require visible inliers on the detector box as well as the existing whole-hand gate."""
    selected = {camera: observation}
    _, views, _ = fit_quality(model, {'selected': selected}, calibrations, threshold)
    view = views[camera]
    box = np.asarray(view['bbox_xyxy'])
    xy, valid, _ = project_base(calibrations[camera], model['joints_base_m'])
    observed = observation['pixels']
    inliers = valid & (np.linalg.norm(xy - observed, axis=1) <= threshold)
    size = calibrations[camera].image_size
    visible = ((xy >= 0) & (xy < size) & (observed >= 0) & (observed < size)).all(1)
    in_box = ((xy >= box[:2]) & (xy <= box[2:]) & (observed >= box[:2]) & (observed <= box[2:])).all(1)
    view.update(detector_is_right=model['side'], source_frame_index=observation['source_frame_index'],
                visible_inlier_count=int((visible & inliers).sum()), box_inlier_count=int((in_box & inliers).sum()))
    view['supported'] = bool(view['supported'] and valid.all() and view['wrist_px'] <= threshold
                             and view['visible_inlier_count'] >= 16 and view['box_inlier_count'] >= 12
                             and project_base(calibrations[camera], model['vertices_base_m'])[1].all())
    return view


def supported_model(template, wrist, pair, observations, calibrations, accepted, threshold, fit_quality):
    if np.linalg.norm(wrist) > 3. or any(np.linalg.norm(wrist - h['joints_base_m'][0].numpy()) < .06
                                         for h in accepted):
        return None
    model = place_template(template, wrist)
    passes, _, _ = fit_quality(model, {'selected': pair}, calibrations, threshold)
    if not passes:
        return None
    views = {}
    for name, options in observations.items():
        # Never swap out a seed box when a camera has multiple same-side detections.
        candidates = [pair[name]] if name in pair else options
        measurements = [measure_view(model, name, ob, calibrations, threshold, fit_quality) for ob in candidates]
        views[name] = min(measurements, key=lambda v: (
            not v['supported'], v['median_px'] if v['median_px'] is not None else np.inf, v['detection_index']))
    support = [name for name, view in views.items() if view['supported']]
    if not pair.keys() <= set(support):
        return None
    return dict(model, views=views, supporting_cameras=support,
                pose_score_px=float(np.mean([max(views[c]['wrist_px'], views[c]['median_px']) for c in support])))


def recover_native_hand(observations, calibrations, accepted, *, threshold, decode, fit_quality):
    """Rank source poses only after triangulated-wrist, box and geometry qualification."""
    templates, trials, choices, exclusions = {}, [], {}, {}
    counts = Counter()
    for a, b in combinations(sorted(observations), 2):
        for first, second in product(observations[a], observations[b]):
            pair = {a: first, b: second}
            counts['tested_pairs'] += 1
            result = triangulate_pair(pair, calibrations, threshold)
            if result is None:
                continue
            wrist, triangulation = result
            counts['triangulated_pairs'] += 1
            for camera, observation in pair.items():
                key = (camera, int(observation['hand']['hypothesis_index']))
                if key not in templates:
                    templates[key], reason = decode(camera, observation)
                    if reason:
                        exclusions[f'{camera}/{key[1]}'] = reason
                if templates[key] is not None:
                    trials.append((camera, templates[key], pair, wrist, triangulation))

    # A lower finger reprojection residual can pull a hand far along the viewing
    # rays. Keep the triangulated wrist whenever any native pose explains it.
    for refine in (False, True):
        for camera, template, pair, wrist, triangulation in trials:
            position = wrist
            if refine:
                joints = template['joints_base_m'].numpy()
                position = refine_wrist(joints - joints[0], pair, wrist, calibrations, threshold)
                if position is None:
                    continue
            model = supported_model(template, position, pair, observations, calibrations,
                                    accepted, threshold, fit_quality)
            if model is None:
                continue
            rank = (-len(model['supporting_cameras']), model['pose_score_px'])
            counts['qualified_native_fits'] += 1
            if camera not in choices or rank < choices[camera][0]:
                model['association'] = {'version': RECOVERY_VERSION, **triangulation,
                                        'wrist_refinement_m': float(np.linalg.norm(position - wrist))}
                choices[camera] = rank, model
        if choices:
            break
    audit = {name: counts[name] for name in ('tested_pairs', 'triangulated_pairs', 'qualified_native_fits')}
    audit['eligible_cameras'] = sorted(observations)
    audit['replayable_sources'] = sum(template is not None for template in templates.values())
    if exclusions:
        audit['native_pose_exclusions'] = exclusions
    strongest = min((rank[0] for rank, _ in choices.values()), default=0)
    return {c: model for c, (rank, model) in choices.items() if rank[0] == strongest}, audit
