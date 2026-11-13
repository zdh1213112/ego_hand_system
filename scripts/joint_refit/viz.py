#!/usr/bin/env python3
"""Overlay a fit (mesh outline, predicted/observed markers) on original images for visual diagnosis."""
import argparse, json, sys
from pathlib import Path
import cv2, numpy as np, torch
PROJECT = Path(__file__).resolve().parents[2]; ROOT = PROJECT.parent / 'wilor_training'  # shared fitting tools live in wilor_training/tools
sys.path.insert(0, str(ROOT / 'tools/full_silhouette'))
from render_masks import project_native, load_hand_mask

EXP = ROOT / 'output/native_kb_training_20261111'

def proj(x, r, side):
    uv, _ = project_native(torch.as_tensor(x[None], dtype=torch.float64), [r], side)
    return uv[0].numpy()

def panel(r, z, i, side, size=560, markers=True, label=''):
    img = cv2.imread(r['image']); v = proj(z['vertices'][i], r, side)
    lo, hi = v.min(0) - 60, v.max(0) + 60; s = max(hi - lo); c = (lo + hi) / 2; lo = c - s / 2
    M = np.array([[size / s, 0, -lo[0] * size / s], [0, size / s, -lo[1] * size / s]])
    out = cv2.warpAffine(img, M, (size, size), flags=cv2.INTER_CUBIC)
    tf = lambda p: (p * size / s - lo * size / s)
    sil = np.zeros((size, size), np.uint8); pv = tf(v)
    for f in z['faces']: cv2.fillConvexPoly(sil, np.round(pv[f]).astype(np.int32), 255)
    cs, _ = cv2.findContours(sil, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    raw = out.copy(); cv2.drawContours(out, cs, -1, (255, 220, 0), 1, cv2.LINE_AA)
    if Path(r['mask']).exists():
        m = cv2.warpAffine(load_hand_mask(r['mask']) * 255, M, (size, size), flags=cv2.INTER_NEAREST)
        mc, _ = cv2.findContours(m, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE); cv2.drawContours(out, mc, -1, (0, 255, 0), 1, cv2.LINE_AA)
    if markers:
        a, b = tf(proj(z['target24'][i], r, side)), tf(proj(z['predicted24'][i], r, side))
        for p, q in zip(a, b):
            cv2.line(out, tuple(np.round(p).astype(int)), tuple(np.round(q).astype(int)), (0, 0, 255), 1, cv2.LINE_AA)
            cv2.circle(out, tuple(np.round(p).astype(int)), 3, (0, 255, 0), 1, cv2.LINE_AA); cv2.circle(out, tuple(np.round(q).astype(int)), 2, (0, 0, 255), -1, cv2.LINE_AA)
    cv2.putText(out, f"{label} {['L','R'][side]} {r['camera']} {r['sync']} q={int(r['accepted_quality'])}", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1, cv2.LINE_AA)
    return raw, out

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--fits', nargs='+', required=True, help='label=dir'); ap.add_argument('--sync', type=int, nargs='+', required=True)
    ap.add_argument('--output', type=Path, required=True); ap.add_argument('--sides', nargs='+', default=['left', 'right']); ap.add_argument('--cameras', nargs='+', default=None)
    a = ap.parse_args(); a.output.mkdir(parents=True, exist_ok=True)
    recs = json.loads((EXP / 'keyframe_calibration_v3/selection.json').read_text())['records']
    for hand in a.sides:
        side = ['left', 'right'].index(hand); fits = [(x.split('=')[0], np.load(Path(x.split('=')[1]) / f'{hand}_pilot.npz')) for x in a.fits]
        for s in a.sync:
            rows = []
            for r in sorted([r for r in recs if r['side'] == side and r['sync'] == s and (not a.cameras or r['camera'] in a.cameras)], key=lambda r: r['camera']):
                row = []
                for k, (lab, z) in enumerate(fits):
                    i = np.where(z['pair_indices'] == s)[0]
                    if not len(i): continue
                    raw, out = panel(r, fits[0][1] if False else z, i[0], side, label=lab)
                    row += ([raw] if k == 0 else []) + [out]
                if row: rows.append(np.hstack(row))
            if rows:
                w = max(x.shape[1] for x in rows); cv2.imwrite(str(a.output / f'{hand}_{s:06d}.jpg'), np.vstack([np.pad(x, ((0, 0), (0, w - x.shape[1]), (0, 0))) for x in rows]))
if __name__ == '__main__': main()
