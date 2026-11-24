#!/usr/bin/env python3
"""Recover recorded SDK offsets and calibrate provisional direction correspondences.

Recovered offsets describe the recorded solver output, not verified anatomy or
the original NOKOV template configuration. The Finger0 origin is a reference;
no SDK root position or SDK bone length becomes a MANO position target.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools/personal_hand_calibration'))
from context import load_context, load_rigid

MP_PARENT = np.array([k + d for k in [1, 5, 9, 13, 17] for d in range(3)])
MP_CHILD = MP_PARENT + 1
SDK_PARENT = np.array([k + d for k in [2, 7, 12, 17, 22] for d in range(3)])
SDK_CHILD = SDK_PARENT + 1


def unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def summary(x):
    x = np.asarray(x); x = x[np.isfinite(x)]
    return dict(count=int(x.size), median=float(np.median(x)),
                p95=float(np.percentile(x, 95)), max=float(x.max()))


def proper_rotation(matrix):
    u, _, vt = np.linalg.svd(matrix)
    return u @ np.diag([1, 1, np.linalg.det(u @ vt)]) @ vt


def angular_deg(a, b):
    return np.degrees(np.arccos(np.clip((unit(a) * unit(b)).sum(-1), -1, 1)))


def raw_profile(nokov):
    assets = json.loads((nokov / 'asset_descriptions.json').read_text())
    with (nokov / 'nokov_frames.csv').open(encoding='utf-8-sig') as f:
        times = [int(r['device_timestamp_raw']) for r in csv.DictReader(f)]
    lookup = {t: i for i, t in enumerate(times)}
    pos = np.full((2, len(times), 26, 3), np.nan)
    quat = np.full((2, len(times), 26, 4), np.nan)
    markers = np.full((2, len(times), 24, 3), np.nan)
    names, parents = [], []
    for hand in ['left', 'right']:
        desc = next(d for d in assets['skeletons'].values() if d['name'] == hand + '_hand_4mm')
        names.append([desc['segments'][str(i)] for i in range(1, 27)])
        parents.append([int(desc['parent_ids'][str(i)]) - 1 for i in range(1, 27)])
    with (nokov / 'nokov_skeleton_segments.csv').open(encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            side = {'left_hand_4mm': 0, 'right_hand_4mm': 1}.get(r['skeleton_name'])
            if side is None or r['valid_numeric'] != '1': continue
            i, j = lookup[int(r['device_timestamp_raw'])], int(r['segment_id']) - 1
            pos[side, i, j] = [float(r[k]) * .001 for k in ['x_mm', 'y_mm', 'z_mm']]
            quat[side, i, j] = [float(r[k]) for k in ['qx', 'qy', 'qz', 'qw']]
    with (nokov / 'nokov_markers.csv').open(encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            side = {'left_hand_4mm': 0, 'right_hand_4mm': 1}.get(r['markerset_name'])
            if side is not None and r['valid'] == '1':
                markers[side, lookup[int(r['device_timestamp_raw'])], int(r['marker_index'])] = [float(r[k]) * .001 for k in ['x_mm', 'y_mm', 'z_mm']]
    hands = {}
    for side, hand in enumerate(['left', 'right']):
        if not np.isfinite(pos[side]).all() or not np.isfinite(quat[side]).all():
            raise ValueError('Incomplete SDK model: do not silently replace missing data')
        rotations = Rotation.from_quat(quat[side].reshape(-1, 4)).as_matrix().reshape(-1, 26, 3, 3)
        bones = []
        for child, parent in enumerate(parents[side]):
            if parent < 0: continue
            local = np.einsum('fji,fj->fi', rotations[:, parent], pos[side, :, child] - pos[side, :, parent])
            offset = np.median(local, axis=0)
            bones.append(dict(parent=names[side][parent], child=names[side][child],
                              parent_local_offset_mm=(offset * 1000).tolist(),
                              offset_repeatability_mm=summary(np.linalg.norm(local - offset, axis=-1) * 1000)))
        attachment_nodes = [0] * 4 + [k + d for k in [2, 7, 12, 17, 22] for d in range(4)]
        marker_names = ['WristM', 'WristIn', 'WristOut', 'HandOffset'] + [finger + str(i) for finger in ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky'] for i in range(1, 5)]
        offsets = []
        for marker, node in enumerate(attachment_nodes):
            local = np.einsum('fji,fj->fi', rotations[:, node], markers[side, :, marker] - pos[side, :, node])
            valid = np.isfinite(local).all(-1); local = local[valid]
            if not len(local): raise ValueError('Marker has no valid samples')
            offset = np.median(local, axis=0)
            offsets.append(dict(marker_index=marker, marker=marker_names[marker], provisional_sdk_node=names[side][node],
                                inferred_local_offset_mm=(offset * 1000).tolist(),
                                residual_spread_mm=summary(np.linalg.norm(local - offset, axis=-1) * 1000),
                                original_template_offset_verified=False))
        hands[hand] = dict(recorded_frames=len(times), bones=bones, inferred_marker_offsets=offsets)
    return hands


def main():
    ap = argparse.ArgumentParser()
    for name in ['context', 'source', 'skeleton', 'selection', 'output']:
        ap.add_argument('--' + name, type=Path, required=True)
    a = ap.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    c = load_context(a.context); sk = np.load(a.skeleton)
    alignment = json.loads(c['alignment'].read_text())
    seed = json.loads(Path(alignment['seed_alignment']).read_text()); m = seed['time_mapping']
    target = m['nokov_origin_timestamp_raw'] + ((sk['reference_timestamp_ns'] - m['ego_origin_timestamp_ns']) * 1e-9 * m.get('a', 1) + m['b_s']) / m['nokov_seconds_per_timestamp_unit']
    rt, _, head_q = load_rigid(Path(alignment['source_nokov']) / 'nokov_rigid_bodies.csv')
    valid = (target >= rt[0]) & (target <= rt[-1])
    head_R = np.full((len(target), 3, 3), np.nan)
    head_R[valid] = Slerp(rt, Rotation.from_quat(head_q))(target[valid]).as_matrix()
    T = np.asarray(alignment['T_cam0_head']); selection = json.loads(a.selection.read_text())
    report = dict(strategy='sdk_direction_mapping_v1', approved_for_training=False,
                  root_policy='finger0', user_preferred_root='SDK Finger0 common origin',
                  sdk_root_position_used_as_mano_target=False, sdk_bone_lengths_used_as_mano_targets=False,
                  original_nokov_template_available=False, anatomical_mapping_verified=False,
                  caveat='Data-derived solver offsets and baseline-calibrated directions; not verified anatomical joints.',
                  source_skeleton=str(a.skeleton.resolve()), source_fit=str(a.source.resolve()),
                  source_selection=str(a.selection.resolve()),
                  recovered_recorded_model=raw_profile(Path(alignment['source_nokov'])), hands={})
    directions = np.full((2, len(target), 15, 3), np.nan)
    confidences = []; palm_targets = np.full((2, len(target), 3, 3), np.nan)
    common_origins = sk['positions_cam0_m'][:, :, 1].copy()
    for side, hand in enumerate(['left', 'right']):
        z = np.load(a.source / f'{hand}_pilot.npz'); pairs = z['pair_indices']
        if not np.isfinite(sk['positions_cam0_m'][side, pairs]).all(): raise ValueError('Missing SDK positions at fitting samples')
        common = sk['positions_cam0_m'][side][:, [1, 6, 11, 16, 21]]
        if np.nanmax(np.linalg.norm(common - common[:, :1], axis=-1)) > 1e-6: raise ValueError('Finger0 origins are not common')
        S = np.diag([-1, 1, 1] if side == 0 else [1, 1, 1])
        # Hand supplies the rigid palm orientation. Finger0 supplies the chosen origin.
        # The Thumb0 rotation can articulate and must not silently become the palm frame.
        q = sk['sdk_quaternions_xyzw'][side, :, 0]; sdk_world = np.full((len(target), 3, 3), np.nan)
        good = valid & np.isfinite(q).all(-1)
        sdk_world[good] = Rotation.from_quat(q[good]).as_matrix()
        sdk_physical = T[:3, :3] @ head_R.transpose(0, 2, 1) @ sdk_world
        sdk_R = S @ sdk_physical @ S
        p = sk['positions_cam0_m'][side] @ S
        d = unit(p[:, SDK_CHILD] - p[:, SDK_PARENT])
        local = np.einsum('fji,fkj->fki', sdk_R, d)
        cal_frames = selection['report']['hands'][hand]['calibration_sync']
        lookup = {int(f): i for i, f in enumerate(pairs)}
        ids = np.array([lookup[f] for f in cal_frames]); cal = np.array(cal_frames)
        mano_R = Rotation.from_rotvec(z['global_orient'][ids]).as_matrix()
        mano_d = unit(z['joints'][ids][:, MP_CHILD] - z['joints'][ids][:, MP_PARENT])
        y = np.einsum('fji,fkj->fki', mano_R, mano_d)
        x = local[cal]
        common_R = proper_rotation((mano_R.transpose(0, 2, 1) @ sdk_R[cal]).mean(0))
        corrections = np.array([proper_rotation(y[:, k].T @ x[:, k] / len(cal) + .25 * common_R) for k in range(15)])
        mapped = np.einsum('kij,fkj->fki', corrections, local)
        error = angular_deg(mapped[cal], y)
        confidence = 1 / (1 + (np.median(error, axis=0) / 10) ** 2)
        directions[side] = mapped; confidences.append(confidence)
        palm_targets[side] = sdk_R @ common_R.T
        root_can = common_origins[side] @ S
        # Diagnostic offset only: an offset from Finger0 is not assumed to vanish.
        wrist_local = np.einsum('fji,fj->fi', sdk_R[cal], z['joints'][ids, 0] - root_can[cal])
        links = []
        for k, (parent, child) in enumerate(zip(SDK_PARENT, SDK_CHILD)):
            links.append(dict(sdk_parent=str(sk['names'][side, parent]), sdk_child=str(sk['names'][side, child]),
                              mano_parent=int(MP_PARENT[k]), mano_child=int(MP_CHILD[k]),
                              sdk_palm_local_to_mano_palm_local=corrections[k].tolist(),
                              calibration_error_deg=summary(error[:, k]), confidence=float(confidence[k])))
        report['hands'][hand] = dict(calibration_sync=cal_frames, directions=links,
                                    sdk_hand_frame_to_mano_frame=common_R.tolist(),
                                    finger0_to_mano_wrist_in_sdk_palm_local_mm=(np.median(wrist_local, axis=0) * 1000).tolist(),
                                    wrist_offset_used_as_fitting_target=False,
                                    calibration_error_deg=summary(error),
                                    reflection_determinant=float(np.linalg.det(S)),
                                    canonical_root_rotation_determinant=summary(np.linalg.det(sdk_R[good])))
        print(hand, 'direction calibration deg', summary(error), flush=True)
    a.output.mkdir(parents=True)
    (a.output / 'mapping.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    np.savez_compressed(a.output / 'direction_targets.npz', local_directions=directions.astype(np.float32),
                        confidence=np.asarray(confidences, np.float32), palm_rotation_targets=palm_targets.astype(np.float32),
                        finger0_origins_cam0_m=common_origins, mano_parent=MP_PARENT, mano_child=MP_CHILD,
                        sdk_parent=SDK_PARENT, sdk_child=SDK_CHILD, approved_for_training=False)
    print(a.output, flush=True)


if __name__ == '__main__': main()
