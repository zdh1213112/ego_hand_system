#!/usr/bin/env python3
"""Dense source for the whole recording: every baseline-fit frame with raw NOKOV24 targets and the calibrated shared parameters."""
import argparse, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2].parent / 'wilor_training'
sys.path.insert(0, str(ROOT / 'tools/personal_hand_calibration'))
from context import load_context


def main():
    ap = argparse.ArgumentParser()
    for n in ['context', 'calibrated', 'skeleton', 'output']: ap.add_argument('--' + n, type=Path, required=True)
    a = ap.parse_args(); a.output.mkdir(parents=True); c = load_context(a.context); sk = np.load(a.skeleton); assert str(sk['geometry_space']) == 'physical_camera0'
    present = [h for side, h in enumerate(['left', 'right']) if (c['baseline_fit'] / f'track_{side}.npz').is_file()]
    if not present: raise ValueError('baseline fit has no hand track')
    for side, hand in enumerate(['left', 'right']):
        if hand not in present: continue
        b = np.load(c['baseline_fit'] / f'track_{side}.npz'); cal = np.load(a.calibrated / f'{hand}_pilot.npz'); pairs = b['pair_indices']
        assert str(b['geometry_space']) == str(cal['geometry_space']), (b['geometry_space'], cal['geometry_space'])
        t = sk['marker24_cam0_m'][side][pairs].copy(); t[..., 0] *= -1 if side == 0 else 1
        i = np.array([np.where(pairs == s)[0][0] for s in cal['pair_indices']]); d = np.abs(b['global_orient'][i] - np.load(c['baseline_fit'] / f'track_{side}.npz')['global_orient'][i]).max()
        values = dict(pair_indices=pairs, target24=t.astype(np.float32), vertices=b['vertices'], joints=b['joints'], faces=cal['faces'], global_orient=b['global_orient'], hand_pose_pca=b['hand_pose_pca'],
                      translation=b['translation'], marker_names=cal['marker_names'], geometry_space=cal['geometry_space'], approved_for_training=np.array(False))
        for k in ['betas', 'attachment_faces', 'barycentric', 'attachment_local_offsets_m', 'normal_offsets_m', 'wrist_local', 'root_anchor_local_m']: values[k] = cal[k]
        np.savez_compressed(a.output / f'{hand}_pilot.npz', **values); print(hand, 'frames', len(pairs), 'frames with missing markers', int(np.isnan(t).any((1, 2)).sum()), flush=True)


if __name__ == '__main__': main()
