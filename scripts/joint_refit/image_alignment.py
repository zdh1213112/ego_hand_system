#!/usr/bin/env python3
"""Per-frame rigid correction of the NOKOV -> camera alignment from marker blobs detected in the original images.

Projected NOKOV markers miss the marker blobs in the images by ~5 px, by a different amount every frame but the same
for both hands and consistent across cameras: a per-frame error of the mocap-to-camera transform (~6 mm / 1 deg), not
of the hand markers. For every sync, match projected markers of both hands to bright blobs in all cameras (per-view
translation voting), solve one 6-DoF correction in camera0 space, re-match under the correction and re-solve.
Writes a copy of the source fit with corrected target24 / finger0_target; uncorrected frames are flagged.
"""
import argparse, csv, json, sys
from pathlib import Path
import cv2, numpy as np, torch
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as Rot
ROOT = Path(__file__).resolve().parents[2].parent / 'wilor_training'
sys.path[:0] = [str(ROOT / 'tools/full_silhouette'), str(ROOT / 'tools/personal_hand_calibration')]
from render_masks import project_native
from context import load_context


def blobs(im):
    if im is None: return np.zeros((0, 2), np.float32)
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32); dog = cv2.GaussianBlur(g, (0, 0), 1.3) - cv2.GaussianBlur(g, (0, 0), 4)
    ys, xs = np.nonzero((dog == cv2.dilate(dog, np.ones((5, 5), np.uint8))) & (dog > 6)); return np.c_[xs, ys].astype(np.float32)


def proj(X, rec): return project_native(torch.as_tensor(X[None], dtype=torch.float64), [rec], 1)[0][0].numpy()  # physical camera0 points


def apply(X, p, c): return (X - c) @ Rot.from_rotvec(p[:3]).as_matrix().T + c + p[3:]


def vote(P, B, radius=30, tol=2.5):
    d = (B[None] - P[:, None]).reshape(-1, 2); d = d[np.linalg.norm(d, axis=1) < radius]
    if not len(d): return None
    sc = np.array([(np.linalg.norm(B[None] - (P + o)[:, None], axis=-1).min(1) < tol).sum() for o in d]); b = sc.argmax()
    far = sc[np.linalg.norm(d - d[b], axis=1) > 6]; return d[b], int(sc[b]), int(far.max()) if len(far) else 0


def match(P, B, tol):
    D = np.linalg.norm(B[None] - P[:, None], axis=-1); j = D.argmin(1); return j, D[np.arange(len(P)), j] < tol


def solve(views, c):
    def f(p): return np.r_[np.concatenate([(proj(apply(v['X'], p, c), v['rec']) - v['uv']).ravel() for v in views]), p[:3] / .02, p[3:] / .02]
    return least_squares(f, np.zeros(6), loss='soft_l1', f_scale=2.).x


def main():
    ap = argparse.ArgumentParser()
    for n in ['context', 'selection', 'source', 'skeleton', 'output']: ap.add_argument('--' + n, type=Path, required=True)
    ap.add_argument('--min-points', type=int, default=12); ap.add_argument('--max-rms-px', type=float, default=2.5)
    a = ap.parse_args(); a.output.mkdir(parents=True)
    recs = json.loads(a.selection.read_text())['records']; sk = np.load(a.skeleton); assert str(sk['geometry_space']) == 'physical_camera0'
    present = [h for h in ['left', 'right'] if (a.source / f'{h}_pilot.npz').is_file()]
    if not present: raise ValueError('source has no hand fit')
    z = {h: dict(np.load(a.source / f'{h}_pilot.npz')) for h in present}; mirror = {0: np.array([-1., 1, 1]), 1: np.ones(3)}
    phys = {}  # (side, sync) -> 24 physical marker positions
    for h in z:
        side = ['left', 'right'].index(h)
        for i, s in enumerate(z[h]['pair_indices']): phys[(side, int(s))] = z[h]['target24'][i].astype(np.float64) * mirror[side]
    # Every fitted sync is checked in all four cameras, read from the original videos (camera parameters are constant per camera).
    c0 = load_context(a.context); frames = list(csv.DictReader((c0['dataset'] / 'multiview_frames.csv').open())); cams = {}
    for r in recs: cams.setdefault(r['camera'], r)
    cache, by = {}, {}; syncs = sorted({s for _, s in phys})
    for cam in sorted(cams):
        cap = cv2.VideoCapture(str(c0['dataset'] / 'cameras' / cam / 'video.mp4')); pos = -1
        for s in syncs:
            fi = int(frames[s][f'{cam}_frame_index'])
            if fi != pos + 1: cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, im = cap.read(); pos = fi; cache[(cam, s)] = blobs(im if ok else None)
            for side in (0, 1):
                if (side, s) in phys: by.setdefault(s, []).append(dict(cams[cam], side=side, sync=s))
        cap.release()
    corr, rows = {}, []
    for s, rs in sorted(by.items()):
        views = []
        for r in rs:
            X = phys[(r['side'], s)]; P = proj(X, r); ok = (P[:, 0] > 5) & (P[:, 1] > 5) & (P[:, 0] < 1915) & (P[:, 1] < 1195); B = cache[(r['camera'], s)]
            if ok.sum() < 8 or not len(B): continue
            B = B[(B[:, 0] > P[ok, 0].min() - 35) & (B[:, 0] < P[ok, 0].max() + 35) & (B[:, 1] > P[ok, 1].min() - 35) & (B[:, 1] < P[ok, 1].max() + 35)]
            if len(B) < 5: continue
            views.append(dict(rec=r, Xall=X[ok], B=B, v=vote(P[ok], B)))
        seed = [v for v in views if v['v'] and v['v'][1] >= 9 and v['v'][1] - v['v'][2] >= 3]
        row = dict(sync=int(s), views=len(views), seed_views=len(seed), corrected=False)
        if seed:
            c = np.concatenate([v['Xall'] for v in views]).mean(0)
            for v in seed:
                P = proj(v['Xall'], v['rec']) + v['v'][0]; j, m = match(P, v['B'], 2.5); v['X'], v['uv'] = v['Xall'][m], v['B'][j[m]]
            p = solve(seed, c)
            for _ in range(2):  # re-match every view under the current correction
                use = []
                for v in views:
                    j, m = match(proj(apply(v['Xall'], p, c), v['rec']), v['B'], 3.)
                    if m.sum() >= 6: v['X'], v['uv'] = v['Xall'][m], v['B'][j[m]]; use.append(v)
                if not use: break
                p = solve(use, c)
            if use:
                e0 = np.concatenate([np.linalg.norm(proj(v['X'], v['rec']) - v['uv'], axis=1) for v in use]); e1 = np.concatenate([np.linalg.norm(proj(apply(v['X'], p, c), v['rec']) - v['uv'], axis=1) for v in use])
                row.update(points=len(e1), used_views=len(use), px_before=float(np.sqrt((e0 ** 2).mean())), px_after=float(np.sqrt((e1 ** 2).mean())), rotation_deg=float(np.degrees(np.linalg.norm(p[:3]))), translation_mm=float(np.linalg.norm(p[3:]) * 1000))
                if len(e1) >= a.min_points and row['px_after'] <= a.max_rms_px: row['corrected'] = True; corr[int(s)] = (p, c)
        rows.append(row)
    for h in z:
        side = ['left', 'right'].index(h)
        pairs = z[h]['pair_indices']; f0 = sk['positions_cam0_m'][side][pairs, 1].copy(); t = z[h]['target24'].copy(); flag = np.zeros(len(pairs), bool)
        for i, s in enumerate(pairs):
            if int(s) in corr:
                p, c = corr[int(s)]; t[i] = apply(t[i].astype(np.float64) * mirror[side], p, c) * mirror[side]; f0[i] = apply(f0[i], p, c); flag[i] = True
        z[h].update(target24_uncorrected=z[h]['target24'], target24=t.astype(np.float32), finger0_target=(f0 * mirror[side]).astype(np.float32), image_alignment_corrected=flag)
        np.savez_compressed(a.output / f'{h}_pilot.npz', **z[h]); print(h, 'frames', len(pairs), 'corrected', int(flag.sum()), flush=True)
    ok = [r for r in rows if r['corrected']]
    summary = dict(source=str(a.source.resolve()), approved_for_training=False, syncs=len(rows), corrected_syncs=len(ok),
                   px_rms_before_median=float(np.median([r['px_before'] for r in ok])), px_rms_after_median=float(np.median([r['px_after'] for r in ok])),
                   rotation_deg_median=float(np.median([r['rotation_deg'] for r in ok])), translation_mm_median=float(np.median([r['translation_mm'] for r in ok])), frames=rows)
    (a.output / 'image_alignment.json').write_text(json.dumps(summary, indent=1)); print({k: v for k, v in summary.items() if k != 'frames'}, flush=True)


if __name__ == '__main__': main()
