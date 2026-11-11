#!/usr/bin/env python3
"""Fit one fixed rig mount from reviewed native-image/NOKOV marker pairs.

Only measured 3D markers enter calibration. The saved MANO reconstruction is
read afterwards for residual analysis; its mesh is never an alignment target.
All views share a right-multiplied SE(3) correction in head-marker coordinates.
"""
from __future__ import annotations
import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, Slerp

from prepare_basalt_dataset import read_timestamps
from visualize_ego_nokov_sync import STREAMS


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def statistics(values):
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    if not len(a):
        return {'count': 0}
    return dict(count=len(a), mean=float(a.mean()), median=float(np.median(a)),
                p95=float(np.percentile(a, 95)), max=float(a.max()),
                rmse=float(np.sqrt(np.mean(a * a))))


def correction_matrix(parameters):
    t = np.eye(4)
    t[:3, :3] = Rotation.from_rotvec(np.asarray(parameters[:3])).as_matrix()
    t[:3, 3] = parameters[3:6]
    return t


def corrected_config(baseline, parameters):
    config = copy.deepcopy(baseline)
    delta = correction_matrix(parameters)
    for camera in config['cameras']:
        camera['T_camera_marker'] = (np.asarray(camera['T_camera_marker']) @ delta).tolist()
    return config


def project_local(points, camera):
    t = np.asarray(camera['T_camera_marker'])
    optical = np.asarray(points) @ t[:3, :3].T + t[:3, 3]
    intr = camera['intrinsics']
    k = np.array([[intr['fx'], 0, intr['cx']], [0, intr['fy'], intr['cy']], [0, 0, 1.]])
    uv = cv2.fisheye.projectPoints(optical.reshape(-1, 1, 3), np.zeros(3), np.zeros(3),
                                  k, np.asarray(camera['distortion']))[0].reshape(-1, 2)
    return uv


class Recording:
    def __init__(self, baseline, mano):
        self.baseline = Path(baseline)
        self.mano = Path(mano)
        self.config = json.loads((self.baseline / 'projection_config.json').read_text())
        self.summary = json.loads((self.baseline / 'overlay_summary.json').read_text())
        self.calibration = json.loads((self.baseline / 'calibration/nokov_calibration.json').read_text())
        self.ego = Path(self.summary['ego_session'])
        self.files = json.loads(next(self.ego.glob('meta_*.json')).read_text())['files']
        self.nokov = Path(self.calibration['reference_directory'])
        self.data = dict(np.load(self.mano / 'reconstruction.npz'))
        with (self.nokov / 'nokov_rigid_bodies.csv').open(encoding='utf-8-sig') as f:
            rows = [r for r in csv.DictReader(f) if r['rigid_body_name'] == 'orbbec_ego4' and r['valid_numeric'] == '1']
        self.frames = np.array([int(r['frame_no']) for r in rows])
        if not np.array_equal(self.frames, self.data['frame_no']):
            raise ValueError('Head and measured-marker frame numbers differ')
        self.motion_ns = np.array([int(r['device_timestamp_raw']) * 1_000_000 for r in rows], np.int64)
        self.camera_ns = np.array([np.array(read_timestamps(self.ego / self.files[f'video_{s}_pts']), np.int64) * 1000 for s in STREAMS])
        self.origin_ns = int(self.camera_ns[0, 0])
        self.motion_s = (self.motion_ns - self.origin_ns) * 1e-9
        self.camera_s = (self.camera_ns - self.origin_ns) * 1e-9
        self.offset_s = float(self.calibration['mocap_to_imu_offset_s'])
        self.position = np.array([[float(r[f'{a}_mm']) * .001 for a in 'xyz'] for r in rows])
        self.rotations = Rotation.from_quat([[float(r[f'q{a}']) for a in 'xyzw'] for r in rows])
        self.slerp = Slerp(self.motion_s, self.rotations)

    def local_points(self, observations, delta_s=0., key='markers_world_m'):
        # t_camera = t_NOKOV_raw + original_offset + delta. Subtract only once.
        query = np.array([self.camera_s[o['camera'], o['video_frame']] for o in observations]) - self.offset_s - delta_s
        if np.any(query < self.motion_s[0]) or np.any(query > self.motion_s[-1]):
            raise ValueError('Observation outside measured motion coverage; no extrapolation')
        right = np.clip(np.searchsorted(self.motion_s, query, side='right'), 1, len(self.motion_s) - 1)
        left = right - 1
        alpha = (query - self.motion_s[left]) / (self.motion_s[right] - self.motion_s[left])
        marker = np.array([o['marker_index'] for o in observations])
        endpoints = self.data[key][left, marker], self.data[key][right, marker]
        if not np.isfinite(endpoints).all():
            raise ValueError('Missing 3D marker at an interpolation endpoint')
        world = endpoints[0] * (1 - alpha[:, None]) + endpoints[1] * alpha[:, None]
        pos = self.position[left] * (1 - alpha[:, None]) + self.position[right] * alpha[:, None]
        return self.slerp(query).inv().apply(world - pos)

    def source_bindings(self):
        sources = [self.mano / 'reconstruction.npz', self.baseline / 'projection_config.json',
                   self.baseline / 'calibration/nokov_calibration.json', self.nokov / 'nokov_rigid_bodies.csv',
                   self.nokov / 'nokov_markers.csv']
        sources += [self.ego / self.files[k] for k in ('calibration_camera', 'calibration_imu')]
        sources += [self.ego / self.files[f'video_{s}_pts'] for s in STREAMS]
        return {str(p.resolve()): sha256(p) for p in sources}


def project_observations(local, observations, config):
    result = np.zeros((len(observations), 2))
    for camera in range(len(config['cameras'])):
        select = np.array([o['camera'] == camera for o in observations])
        if select.any():
            result[select] = project_local(local[select], config['cameras'][camera])
    return result


def observation_depths(local, observations, config):
    """Optical z, checked separately because fisheye projection also maps z < 0."""
    return np.array([(np.asarray(config['cameras'][o['camera']]['T_camera_marker'])
                      @ np.r_[point, 1.])[2] for point, o in zip(local, observations)])


def fit_mount(local, observations, baseline, initial=None):
    target = np.array([o['pixel'] for o in observations])
    uncertainty = np.array([o['uncertainty_px'] for o in observations])[:, None]
    if len(observations) < 6 or not np.isfinite(local).all() or not np.isfinite(target).all():
        raise ValueError('Need at least six finite 3D-2D correspondences')
    if not np.isfinite(uncertainty).all() or np.any(uncertainty <= 0):
        raise ValueError('Pixel uncertainty must be finite and positive')
    if np.linalg.matrix_rank(local - np.mean(local, axis=0)) < 2:
        raise ValueError('Collinear marker geometry cannot determine a mount')
    def residual(x):
        return ((project_observations(local, observations, corrected_config(baseline, x)) - target) / uncertainty).ravel()
    fit = least_squares(residual, np.zeros(6) if initial is None else initial,
                        loss='soft_l1', f_scale=2., max_nfev=300, xtol=1e-12, ftol=1e-12, gtol=1e-12)
    if not fit.success or np.linalg.matrix_rank(fit.jac) < 6:
        raise ValueError('Fixed mount optimization did not converge to a full-rank solution')
    if np.any(observation_depths(local, observations, corrected_config(baseline, fit.x)) <= .001):
        raise ValueError('Fitted marker lies behind the camera or too close to its optical plane')
    return fit


def evaluate(recording, observations, config, delta_s=0., key='markers_world_m'):
    local = recording.local_points(observations, delta_s, key)
    if np.any(observation_depths(local, observations, config) <= .001):
        raise ValueError('Observed marker lies behind the camera or too close to its optical plane')
    pixels = project_observations(local, observations, config)
    return pixels, np.linalg.norm(pixels - np.array([o['pixel'] for o in observations]), axis=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--mano', type=Path, required=True)
    parser.add_argument('--observations', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    recording = Recording(args.baseline, args.mano)
    annotations = json.loads(args.observations.read_text())
    observations = annotations['observations']
    seen = set()
    for o in observations:
        if o['split'] not in ('fit', 'development', 'validation', 'final_test'):
            raise ValueError('Unknown observation role')
        if not (0 <= o['camera'] < len(recording.config['cameras'])
                and 0 <= o['video_frame'] < recording.camera_ns.shape[1]
                and 0 <= o['marker_index'] < len(recording.data['marker_names'])):
            raise ValueError('Observation index out of range')
        if (np.asarray(o['pixel']).shape != (2,) or not np.isfinite(o['pixel']).all()
                or not np.isfinite(o['uncertainty_px']) or o['uncertainty_px'] <= 0):
            raise ValueError('Invalid pixel or localization uncertainty')
        identity = o['camera'], o['video_frame'], o['marker_index']
        if identity in seen or str(recording.data['marker_names'][o['marker_index']]) != o['marker_name']:
            raise ValueError('Duplicate or mismatched marker identity')
        seen.add(identity)
        if sha256(args.observations.parent / o['source']) != o['source_sha256']:
            raise ValueError('Annotation source image changed')
    train = [o for o in observations if o['split'] == 'fit']
    val = [o for o in observations if o['split'] != 'fit']
    if {o['video_frame'] for o in train} & {o['video_frame'] for o in val}:
        raise ValueError('Fit and validation frames overlap')
    fit = fit_mount(recording.local_points(train), train, recording.config)
    config = corrected_config(recording.config, fit.x)
    before_px, before = evaluate(recording, observations, recording.config)
    after_px, after = evaluate(recording, observations, config)
    profile = []
    for milliseconds in [-60, -40, -20, -10, 0, 10, 20, 40, 60]:
        try:
            local = recording.local_points(train, milliseconds / 1000)
            recording.local_points(val, milliseconds / 1000)
        except ValueError as error:
            # A diagnostic time shift must never remove valid nominal-time labels.
            profile.append(dict(additional_offset_ms=milliseconds, status='unavailable', reason=str(error)))
            continue
        trial = fit_mount(local, train, recording.config, fit.x)
        trial_config = corrected_config(recording.config, trial.x)
        _, train_error = evaluate(recording, train, trial_config, milliseconds / 1000)
        _, validation_error = evaluate(recording, val, trial_config, milliseconds / 1000)
        _, fixed_error = evaluate(recording, train, config, milliseconds / 1000)
        profile.append(dict(additional_offset_ms=milliseconds, refit_train_px=statistics(train_error),
                            refit_evaluation_px=statistics(validation_error), fixed_mount_train_px=statistics(fixed_error),
                            parameters=trial.x.tolist()))
    result = dict(status='candidate_requires_visual_validation', success=bool(fit.success),
                  correction_frame='head rigid marker; T_camera_marker_new = T_camera_marker_old @ correction',
                  correction_matrix=correction_matrix(fit.x).tolist(), parameters_rotvec_rad_translation_m=fit.x.tolist(),
                  rotation_degrees=float(np.linalg.norm(fit.x[:3]) * 180 / np.pi),
                  translation_norm_mm=float(np.linalg.norm(fit.x[3:]) * 1000),
                  optimizer=dict(success=bool(fit.success), evaluations=fit.nfev,
                                 jacobian_rank=int(np.linalg.matrix_rank(fit.jac)),
                                 jacobian_condition=float(np.linalg.cond(fit.jac)),
                                 minimum_observed_depth_m=float(observation_depths(recording.local_points(observations), observations, config).min())),
                  time_offset_change_ms=0., nokov_to_ego_offset_ms=recording.offset_s * 1000,
                  camera_imu_offset_ms=0., time_profile=profile,
                  time_policy='Retain original inertial offset. Profile is sensitivity only; sparse mostly static annotations cannot certify sync accuracy.',
                  sampling='Linear interpolation of measured markers and head translation; SLERP head orientation at camera time. No extrapolation.',
                  source_bindings=recording.source_bindings(), observations_sha256=sha256(args.observations),
                  fit_frames=sorted({o['video_frame'] for o in train}), validation_frames=sorted({o['video_frame'] for o in val}),
                  planned_validation_frames=annotations['validation_frames'],
                  data_roles=annotations.get('roles',{}), role_limitation=annotations.get('role_limitation',''),
                  limitations=annotations['exclusions'] + ['Labels are agent-reviewed, not independently approved human ground truth.',
                     'Only labeled visible spheres are measured; no silhouette/anatomical shape truth.',
                     'Unobserved camera/time/pose regions are not certified by the sparse validation.'])
    result['evaluation'] = dict(before_px=statistics(before[[o['split']!='fit' for o in observations]]),
                               after_px=statistics(after[[o['split']!='fit' for o in observations]]))
    for split in sorted({o['split'] for o in observations}):
        select = np.array([o['split'] == split for o in observations])
        result[split] = dict(before_px=statistics(before[select]), after_px=statistics(after[select]))
    result['by_camera'] = {}
    for c in range(4):
        result['by_camera'][str(c)] = {}
        for split in sorted({o['split'] for o in observations}):
            select = np.array([o['camera'] == c and o['split'] == split for o in observations])
            result['by_camera'][str(c)][split] = dict(before_px=statistics(before[select]), after_px=statistics(after[select]))
    result['by_frame'] = {str(f): dict(before_px=statistics(before[[o['video_frame'] == f for o in observations]]),
                                        after_px=statistics(after[[o['video_frame'] == f for o in observations]]))
                          for f in sorted({o['video_frame'] for o in observations})}
    result['coverage'] = dict(observations=len(observations), image_count=len({(o['camera'], o['video_frame']) for o in observations}),
                              camera_time_s=[min(recording.camera_s[o['camera'], o['video_frame']] for o in observations),
                                             max(recording.camera_s[o['camera'], o['video_frame']] for o in observations)],
                              observed_pixel_min=np.min([o['pixel'] for o in observations], axis=0).tolist(),
                              observed_pixel_max=np.max([o['pixel'] for o in observations], axis=0).tolist())
    config['installation'] = 'Candidate fixed shared rig image-marker correction; acceptance requires separate held-out image checks; original inertial time offset retained'
    config['image_alignment'] = {k: result[k] for k in ['correction_matrix', 'source_bindings', 'observations_sha256', 'time_offset_change_ms', 'nokov_to_ego_offset_ms', 'limitations']}
    (args.output / 'alignment_result.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    (args.output / 'projection_config.json').write_text(json.dumps(config, indent=2, allow_nan=False) + '\n')
    with (args.output / 'reprojection_observations.csv').open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['camera','video_frame','split','marker_index','marker_name','observed_u','observed_v','before_u','before_v','after_u','after_v','before_error_px','after_error_px','uncertainty_px'])
        for o, b, a, eb, ea in zip(observations, before_px, after_px, before, after):
            w.writerow([o['camera'],o['video_frame'],o['split'],o['marker_index'],o['marker_name'],*o['pixel'],*b,*a,eb,ea,o['uncertainty_px']])
    print(json.dumps({k: result[k] for k in ['fit','evaluation','by_frame','rotation_degrees','translation_norm_mm']}, indent=2))


if __name__ == '__main__':
    main()
