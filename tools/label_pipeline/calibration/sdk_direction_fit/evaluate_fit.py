#!/usr/bin/env python3
"""Fixed image evidence for baseline / marker-only / marker+direction controls."""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'tools/label_pipeline/calibration/full_silhouette'))
from evaluate import score
from render_masks import load_hand_mask
sys.path.insert(0, str(ROOT / 'tools/label_pipeline/calibration/personal_hand_calibration'))
from context import load_context
from refine_contours import project


def aggregate(rows):
    out = dict(images=len(rows))
    if not rows: return out
    for mode in ['current', 'marker_only', 'direction']:
        out[mode + '_iou'] = float(np.mean([r[mode]['iou'] for r in rows]))
        out[mode + '_boundary_px'] = float(np.mean([r[mode]['boundary_mean_px'] for r in rows]))
    for mode, reference in [('marker_only', 'current'), ('direction', 'current'), ('direction', 'marker_only')]:
        delta = np.array([r[mode]['iou'] - r[reference]['iou'] for r in rows])
        out[f'{mode}_vs_{reference}_delta_pp'] = float(delta.mean() * 100)
        out[f'{mode}_vs_{reference}_worse_over_2pp'] = int((delta < -.02).sum())
    return out


def main():
    ap = argparse.ArgumentParser()
    for name in ['context', 'selection', 'source', 'result']:
        ap.add_argument('--' + name, type=Path, required=True)
    a = ap.parse_args(); c = load_context(a.context); selection = json.loads(a.selection.read_text())
    cv2.setNumThreads(2); torch.set_num_threads(2)
    report = dict(approved_for_training=False, sdk_root_policy='finger0', sdk_absolute_positions_used_as_targets=False,
                  metric='Native KB image silhouette agreement; fixed baseline wrist domain; not anatomical ground truth',
                  images_used_for_this_motion_fitting=False, calibration_reuses_baseline_pose=True, hands={}, per_view=[])
    for side, hand in enumerate(['left', 'right']):
        tracks = {mode: np.load(path / f'{hand}_pilot.npz') for mode, path in [('current', a.source), ('marker_only', a.result / 'marker_only'), ('direction', a.result / 'direction')]}
        lookup = {mode: {int(f): i for i, f in enumerate(z['pair_indices'])} for mode, z in tracks.items()}
        chosen = set(selection['report']['hands'][hand]['calibration_sync']); rows = []
        for r in selection['records']:
            if r['side'] != side or not r['accepted_quality']: continue
            old = tracks['current']; jp = project(torch.tensor(old['joints'][lookup['current'][r['sync']]], dtype=torch.float64), r, side).numpy()
            axis = jp[[5, 9, 13, 17]].mean(0) - jp[0]; axis /= max(np.linalg.norm(axis), 1e-9)
            mask = load_hand_mask(r['mask']); yy, xx = np.mgrid[:mask.shape[0], :mask.shape[1]]
            domain = (xx - jp[0, 0]) * axis[0] + (yy - jp[0, 1]) * axis[1] > 8
            row = dict(camera=r['camera'], sync=r['sync'], side=side, used_for_shape_or_direction_calibration=r['sync'] in chosen,
                       time_holdout=c['holdout'][0] <= r['sync'] < c['holdout'][1])
            for mode, z in tracks.items():
                uv = project(torch.tensor(z['vertices'][lookup[mode][r['sync']]], dtype=torch.float64), r, side).numpy()
                row[mode] = score(uv, z['faces'], mask, domain, jp[0], axis)
            rows.append(row)
        independent = [r for r in rows if not r['used_for_shape_or_direction_calibration']]
        report['hands'][hand] = dict(all_accepted_masks=aggregate(rows), images_not_used_for_shape_or_direction_calibration=aggregate(independent),
                                    time_holdout_images=aggregate([r for r in independent if r['time_holdout']]),
                                    worst_direction_vs_control=sorted(independent, key=lambda r: r['direction']['iou'] - r['marker_only']['iou'])[:5])
        report['per_view'] += rows
        print(hand, json.dumps(report['hands'][hand]['images_not_used_for_shape_or_direction_calibration']), flush=True)
    (a.result / 'comparison_metrics.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__': main()
