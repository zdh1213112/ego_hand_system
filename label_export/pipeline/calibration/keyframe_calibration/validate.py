#!/usr/bin/env python3
"""Re-render mesh silhouettes on images excluded from shared calibration.

Scores measure agreement with filtered automatic masks, not independently measured
hand-surface truth. All comparisons use identical masks and a fixed wrist cut.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'label_export/pipeline/calibration/personal_hand_calibration'))
from context import load_context
from refine_contours import project
from run import PROJECT, import_mano, run_model


def distance(vertices, faces, record, side, wrist_joints, size):
    uv = project(torch.tensor(vertices, dtype=torch.float64), record, side).numpy()
    jp = project(torch.tensor(wrist_joints, dtype=torch.float64), record, side).numpy()
    axis = jp[[5, 9, 13, 17]].mean(0) - jp[0]
    axis /= max(np.linalg.norm(axis), 1e-9)
    mask = np.zeros((size[1], size[0]), np.uint8)
    for triangle in np.rint(uv[faces]).astype(np.int32):
        cv2.fillConvexPoly(mask, triangle, 1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError('No mesh silhouette')
    boundary = np.concatenate([q.reshape(-1, 2) for q in contours])
    boundary = boundary[(boundary - jp[0]) @ axis > 8]
    if not len(boundary):
        raise ValueError('No distal mesh boundary')
    boundary = boundary[np.linspace(0, len(boundary) - 1, min(200, len(boundary))).astype(int)]
    target = np.asarray(record['points'])
    residual = np.r_[cKDTree(target).query(boundary)[0], cKDTree(boundary).query(target)[0]]
    return float(residual.mean())


def aggregate(rows):
    if not rows:
        return {'views': 0}
    result = {'views': len(rows)}
    for mode in ['baseline', 'previous_joint', 'keyframe']:
        values = np.array([r[mode + '_px'] for r in rows])
        result[mode + '_mean_px'] = float(values.mean())
        result[mode + '_median_px'] = float(np.median(values))
    result['improvement_vs_baseline_percent'] = 100 * (1 - result['keyframe_mean_px'] / result['baseline_mean_px'])
    result['worsened_vs_baseline_over_1px'] = sum(r['keyframe_px'] - r['baseline_px'] > 1 for r in rows)
    result['improved_vs_baseline_over_1px'] = sum(r['keyframe_px'] - r['baseline_px'] < -1 for r in rows)
    return result


def main():
    ap = argparse.ArgumentParser()
    for name in ['context', 'source', 'selection', 'fit', 'previous', 'output']:
        ap.add_argument('--' + name, type=Path, required=True)
    ap.add_argument('--stability-a', type=Path)
    ap.add_argument('--stability-b', type=Path)
    a = ap.parse_args()
    c = load_context(a.context)
    selected = json.loads(a.selection.read_text())
    size = json.loads((c['dataset'] / 'manifest.json').read_text())['image_size']
    cv2.setNumThreads(2)
    torch.set_num_threads(4)
    model = import_mano(PROJECT / 'third_party/MANO').load(
        model_path=str(PROJECT / 'models/mano'), is_rhand=True, use_pca=True,
        num_pca_comps=45, flat_hand_mean=False, batch_size=1)
    report = {'approved_for_training': False,
              'metric': 'Symmetric external silhouette distance, true KB projection, fixed wrist cut; automatic masks only',
              'independent_surface_ground_truth': False, 'hands': {}}
    for side, hand in enumerate(['left', 'right']):
        fit = np.load(a.fit / f'{hand}_pilot.npz')
        source = np.load(a.source / f'{hand}_pilot.npz')
        previous = np.load(a.previous / f'{hand}_pilot.npz')
        indices = {int(f): i for i, f in enumerate(fit['pair_indices'])}
        si = {int(f): i for i, f in enumerate(source['pair_indices'])}
        pi = {int(f): i for i, f in enumerate(previous['pair_indices'])}
        cal = set(selected['report']['hands'][hand]['calibration_sync'])
        rows = []
        for r in selected['records']:
            if r['side'] != side or not r['accepted_quality'] or r['sync'] in cal:
                continue
            i, j, k = indices[r['sync']], si[r['sync']], pi[r['sync']]
            row = {'camera': r['camera'], 'sync': r['sync'],
                   'time_holdout': c['holdout'][0] <= r['sync'] < c['holdout'][1]}
            for mode, vertices in [('baseline', fit['baseline_vertices'][i]),
                                   ('previous_joint', previous['vertices'][k]),
                                   ('keyframe', fit['vertices'][i])]:
                row[mode + '_px'] = distance(vertices, fit['faces'], r, side, source['joints'][j], size)
            rows.append(row)
        result = {'noncalibration': aggregate(rows),
                  'time_holdout': aggregate([r for r in rows if r['time_holdout']]),
                  'per_camera': {camera: aggregate([r for r in rows if r['camera'] == camera])
                                 for camera in sorted({r['camera'] for r in rows})},
                  'worst_regressions': sorted(rows, key=lambda r: r['keyframe_px'] - r['baseline_px'], reverse=True)[:6],
                  'per_view': rows}
        # Canonical neutral pose isolates shape from frame-dependent orientation and motion.
        neutral_pose = -model.hand_mean[None]
        def neutral(beta):
            with torch.no_grad():
                vertices, joints, _ = run_model(model, torch.tensor(beta)[None], torch.zeros((1, 3)),
                                               neutral_pose, torch.zeros((1, 3)))
            return (vertices[0] - joints[0, :1]).numpy(), (joints[0] - joints[0, :1]).numpy()
        vertices, joints = neutral(fit['betas'])
        np.savez_compressed(a.fit / f'{hand}_reference_hand.npz', vertices=vertices, joints=joints,
                            faces=fit['faces'], betas=fit['betas'], geometry_space='right_canonical',
                            approved_for_training=False)
        if a.stability_a and a.stability_b:
            split_a = np.load(a.stability_a / f'{hand}_pilot.npz')
            split_b = np.load(a.stability_b / f'{hand}_pilot.npz')
            va, _ = neutral(split_a['betas'])
            vb, _ = neutral(split_b['betas'])
            err = np.linalg.norm(va - vb, axis=-1) * 1000
            result['split_calibration_stability'] = {
                'meaning': 'Different disjoint keyframe groups; reproducibility, not accuracy',
                'neutral_vertex_median_mm': float(np.median(err)),
                'neutral_vertex_p95_mm': float(np.percentile(err, 95)),
                'neutral_vertex_max_mm': float(err.max()),
                'beta_l2_difference': float(np.linalg.norm(split_a['betas'] - split_b['betas']))}
        report['hands'][hand] = result
        print(hand, json.dumps({k: v for k, v in result.items() if k not in ['per_view', 'worst_regressions']}), flush=True)
    a.output.write_text(json.dumps(report, indent=2, allow_nan=False))
    print(a.output, flush=True)


if __name__ == '__main__':
    main()
