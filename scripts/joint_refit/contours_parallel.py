#!/usr/bin/env python3
"""Glove contours for the calibration sample only, decoded in order and processed in parallel.

Per-view segmentation is the same as wilor_training/tools/personal_hand_calibration/image_constraints.py, but only
for the syncs keyframe selection can use (the frames in --sample), instead of every baseline frame. Frames are read
forward instead of seeking for each one, and camera/sync chunks run in separate processes. The output layout
(data.js, observations.json, images/, masks/) is what select_keyframes.py reads; preview crops are not written.
"""
import argparse, csv, json, os, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import cv2, numpy as np
TOOLS = Path(__file__).resolve().parents[3] / 'wilor_training/tools'
sys.path.insert(0, str(TOOLS / 'personal_hand_calibration'))
import image_constraints as ic  # noqa: E402  (sets up the calibration tool import paths)
from context import load_context  # noqa: E402

HANDS = ['left', 'right']
MAX_FORWARD = 40  # read forward over gaps up to this many frames, otherwise seek


def load(directory):
    return [np.load(directory / f'{h}_pilot.npz') if (directory / f'{h}_pilot.npz').is_file() else None for h in HANDS]


def chunk(cam, syncs, a):
    """Process one camera over an ordered list of syncs; returns (records, rejected) in sync/side order."""
    cv2.setNumThreads(1)
    ds = load_context(a.context)['dataset']; tracks = load(a.source); sample = load(a.sample)
    maps = [{int(f): i for i, f in enumerate(z['pair_indices'])} if z is not None else {} for z in tracks]
    wanted = [set(int(f) for f in z['pair_indices']) if z is not None else set() for z in sample]
    with (ds / 'multiview_frames.csv').open() as f: frames = list(csv.DictReader(f))
    C0 = ic.CameraCalibration.load(ds / 'calibration/camera0.json'); C = ic.CameraCalibration.load(ds / 'calibration' / f'{cam}.json')

    def proj(q, side):
        q = q.copy(); q[:, 0] *= (-1 if side == 0 else 1); base = q @ C0.T_base_camera[:3, :3].T + C0.T_base_camera[:3, 3]
        xyz = (base - C.T_base_camera[:3, 3]) @ C.T_base_camera[:3, :3]; uv, _ = ic.project_points(C, xyz); return uv, xyz[:, 2]

    cap = cv2.VideoCapture(str(ds / 'cameras' / cam / 'video.mp4')); pos = -1; records, rejected = [], []
    for sync in syncs:
        fi = int(frames[sync][cam + '_frame_index'])
        if pos < fi <= pos + MAX_FORWARD:
            for _ in range(fi - pos - 1): cap.grab()
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, image = cap.read(); pos = fi
        if not ok: raise RuntimeError(f'Image missing: {cam} sync {sync}')
        written = False
        for side, z in enumerate(tracks):
            if z is None or sync not in maps[side] or sync not in wanted[side]: continue
            i = maps[side][sync]; seed, depth = proj(z['target24'][i], side); j, _ = proj(z['joints'][i], side); other = np.empty((0, 2))
            if sync in maps[1 - side]:
                other, od = proj(tracks[1 - side]['target24'][maps[1 - side][sync]], 1 - side); other = other[(od > .03) & np.isfinite(other).all(1)]
            if (depth < .03).any() or not np.isfinite(seed).all() or ((seed < 0).any(1) | (seed[:, 0] >= image.shape[1]) | (seed[:, 1] >= image.shape[0])).sum() > 2:
                rejected.append(dict(camera=cam, sync=sync, side=side, reason='partial_or_behind_camera')); continue
            axis = j[[5, 9, 13, 17]].mean(0) - j[0]; obs, why = ic.segment(image, seed, other, j[0], axis)
            if obs is None: rejected.append(dict(camera=cam, sync=sync, side=side, reason=why)); continue
            uv, _ = proj(z['vertices'][i], side)
            mm = np.zeros(image.shape[:2], np.uint8)  # full triangle raster keeps the gaps between fingers
            for tri in np.rint(uv[z['faces']]).astype(np.int32): cv2.fillConvexPoly(mm, tri, 1)
            edge = cv2.morphologyEx(mm, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)); distance = cv2.distanceTransform(1 - edge, cv2.DIST_L2, 3)
            idx = np.rint(uv).astype(int); inside = (idx[:, 0] >= 0) & (idx[:, 0] < image.shape[1]) & (idx[:, 1] >= 0) & (idx[:, 1] < image.shape[0])
            ids = np.flatnonzero(inside); ids = ids[(distance[idx[ids, 1], idx[ids, 0]] < 2.5) & ((uv[ids] - j[0]) @ axis / max(np.linalg.norm(axis), 1e-9) > 8)]
            if len(ids) < 25: rejected.append(dict(camera=cam, sync=sync, side=side, reason='insufficient_mesh_boundary')); continue
            stem = f'{cam}_{sync:06d}'
            if not written: cv2.imwrite(str(a.output / 'images' / f'{stem}.jpg'), image, [cv2.IMWRITE_JPEG_QUALITY, 97]); written = True
            rgba = np.zeros((*image.shape[:2], 4), np.uint8); rgba[:, :, 3] = obs['mask'] * 255; cv2.imwrite(str(a.output / 'masks' / f'{stem}_{side}.png'), rgba)
            records.append(dict(camera=cam, sync=sync, side=side, accepted=True, replay_exact=True, mask=f'masks/{stem}_{side}.png', image=f'images/{stem}.jpg',
                                calibration=bool(z['calibration_mask'][i]), points=obs['points'].tolist(), boundary_vertices=ids.tolist(), stable_fraction=obs['stable_fraction'],
                                component_marker_hits=obs['component_marker_hits'], preview=None, K=C.K.tolist(), D=C.distortion.tolist(),
                                T_base_camera=C.T_base_camera.tolist(), T_base_cam0=C0.T_base_camera.tolist()))
    cap.release(); return records, rejected


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--context', type=Path, required=True)
    ap.add_argument('--source', type=Path, required=True, help='baseline fit in <hand>_pilot.npz form (meshes and markers)')
    ap.add_argument('--sample', type=Path, required=True, help='prepare_source.py output; only its frames are segmented')
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--workers', type=int, default=min(16, os.cpu_count() or 4))
    a = ap.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    (a.output / 'images').mkdir(parents=True); (a.output / 'masks').mkdir()
    sample = load(a.sample)
    if not any(z is not None for z in sample): raise ValueError('sample has no hand')
    syncs = sorted(set().union(*[set(int(f) for f in z['pair_indices']) for z in sample if z is not None]))
    per_cam = max(1, a.workers // 4); jobs = []
    for cam in [f'camera{i}' for i in range(4)]:
        for part in np.array_split(np.asarray(syncs), per_cam):
            if len(part): jobs.append((cam, [int(s) for s in part]))
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        results = list(pool.map(chunk, [c for c, _ in jobs], [s for _, s in jobs], [a] * len(jobs)))
    key = lambda r: (r['camera'], r['sync'], r['side'])
    records = sorted([r for rec, _ in results for r in rec], key=key); rejected = sorted([r for _, rej in results for r in rej], key=key)
    for cam in [f'camera{i}' for i in range(4)]: print(cam, 'accepted', sum(r['camera'] == cam for r in records), flush=True)
    data = dict(schema='automatic_native_contours_v1', records=records, rejected=rejected, sampled_syncs=len(syncs),
                limitations=['Color-threshold proposals are not independently verified silhouettes.',
                             'NOKOV projection chooses component identity; baseline mesh determines distal wrist exclusion and initial silhouette vertex indices.',
                             'Shared hand components are rejected; no per-pixel occlusion model.', 'No inferred fingertip identities are used.',
                             'Only the calibration sample frames are segmented.'])
    (a.output / 'observations.json').write_text(json.dumps(data))
    (a.output / 'data.js').write_text('window.CONTOUR_AUDIT=' + json.dumps(dict(report={}, records=records), separators=(',', ':')) + ';\n')
    print('DONE', len(records), 'contours from', len(syncs), 'sampled syncs', flush=True)


if __name__ == '__main__': main()
