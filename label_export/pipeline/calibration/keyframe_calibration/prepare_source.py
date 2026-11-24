#!/usr/bin/env python3
"""Prepare finite NOKOV24 samples and the existing baseline without fitting shape or attachments."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'label_export/pipeline/calibration/personal_hand_calibration'))
from context import load_context, raw24_context
from run import NAMES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--context', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--sample-stride', type=int, default=8)
    a = ap.parse_args()
    if a.sample_stride < 1:
        ap.error('sample-stride must be positive')
    if a.output.exists():
        raise FileExistsError(a.output)
    c = load_context(a.context)
    xyz, _ = raw24_context(c)
    a.output.mkdir(parents=True)
    summary = {'strategy': 'nokov24_input_only_no_parameter_fit', 'approved_for_training': False, 'hands': {}}
    present = [h for side, h in enumerate(['left', 'right']) if (c['baseline_fit'] / f'track_{side}.npz').is_file()]
    if not present: raise ValueError('baseline fit has no hand track')
    for side, hand in enumerate(['left', 'right']):
        if hand not in present: continue
        b = np.load(c['baseline_fit'] / f'track_{side}.npz')
        pairs = b['pair_indices']
        q = xyz[side, pairs].copy()
        q[:, :, 0] *= -1 if side == 0 else 1
        speed = np.zeros(len(q))
        speed[1:] = np.nanmax(np.linalg.norm(np.diff(q, axis=0), axis=-1), axis=1)
        good = np.isfinite(q).all((1, 2)) & (speed < .05) & (pairs % a.sample_stride == 0) & b['render_valid']
        ids = np.flatnonzero(good)
        if not len(ids):
            raise ValueError(f'{hand}: no finite renderable NOKOV24 frames')
        lo, hi = c['holdout']
        np.savez_compressed(a.output / f'{hand}_pilot.npz',
                            pair_indices=pairs[ids], vertices=b['vertices'][ids], joints=b['joints'][ids],
                            baseline_vertices=b['vertices'][ids], baseline_joints=b['joints'][ids],
                            target24=q[ids].astype(np.float32), betas=b['betas'],
                            global_orient=b['global_orient'][ids], hand_pose_pca=b['hand_pose_pca'][ids],
                            translation=b['translation'][ids], faces=b['faces'], marker_names=np.asarray(NAMES),
                            calibration_mask=np.zeros(len(ids), bool),
                            heldout_mask=(pairs[ids] >= lo) & (pairs[ids] < hi), geometry_space='right_canonical')
        summary['hands'][hand] = {'frames': len(ids), 'raw_frames': len(pairs), 'sample_stride': a.sample_stride}
    (a.output / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
