#!/usr/bin/env python3
"""End-to-end check of exported labels against image evidence.

For exported labels, carry the fitted marker positions into the label's own camera frame (via the label vertices),
project them with the label's own K / distortion and compare with marker blobs detected in the label's own image.
The same is done with the uncorrected NOKOV markers as a reference. Independent of the fitting code path.
"""
import argparse, json, sys
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from image_alignment import blobs


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--result', type=Path, required=True, help='<fit>/final'); ap.add_argument('--samples', type=int, default=1500)
    a = ap.parse_args(); ds = a.result / 'export/merged'; rows = [json.loads(l) for l in (ds / 'index.jsonl').read_text().splitlines() if l.strip()]; rows = [rows[i] for i in sorted(np.random.default_rng(0).choice(len(rows), min(a.samples, len(rows)), replace=False))]
    z = [np.load(a.result / f'{h}_pilot.npz') if (a.result / f'{h}_pilot.npz').is_file() else None for h in ['left', 'right']]
    if not any(x is not None for x in z): raise ValueError('result has no hand fit')
    look = [{int(s): i for i, s in enumerate(x['pair_indices'])} if x is not None else {} for x in z]
    out = {k: [] for k in ['replay', 'fit', 'nokov_corrected', 'nokov_uncorrected']}; per = {}
    for r in rows:
        lab = np.load(ds / r['label'], allow_pickle=True).item(); side = int(r['side']); i = look[side][int(r['sync_index'])]; x = z[side]
        V = np.asarray(lab['vertices'], np.float64); K = np.asarray(lab['K'], np.float64); D = np.asarray(lab['distortion'], np.float64)
        px = lambda X: cv2.fisheye.projectPoints(X[None], np.zeros(3), np.zeros(3), K, D)[0][0]
        best = None
        for cand in (V, V + np.asarray(lab['trans'], np.float64)):
            if (cand[:, 2] > .05).all():
                e = np.abs(px(cand) - np.asarray(lab['joints_2d'], np.float64)).max()
                if best is None or e < best[0]: best = (e, cand)
        if best is None: continue
        out['replay'].append(best[0]); Vc = best[1]
        src = np.c_[x['vertices'][i].astype(np.float64), np.ones(778)]; A = np.linalg.lstsq(src, Vc, rcond=None)[0]  # canonical fit frame -> label camera frame
        B = blobs(cv2.imread(str(ds / r['image']))); valid = x['marker_valid'][i].copy()
        # only markers whose attachment face looks at this camera can be seen as blobs (wrist markers 0..2 are kept as tracked)
        tri = x['vertices'][i].astype(np.float64)[x['faces'][x['attachment_faces']]]; nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]) @ A[:3]; M = np.c_[x['predicted24'][i].astype(np.float64), np.ones(24)] @ A
        cosv = (nrm * M[3:]).sum(1) / (np.linalg.norm(nrm, axis=1) * np.linalg.norm(M[3:], axis=1)); valid[3:] &= cosv < -.35; valid[:3] = False
        for name, key in [('fit', 'predicted24'), ('nokov_corrected', 'target24'), ('nokov_uncorrected', 'target24_uncorrected')]:
            P = px(np.c_[x[key][i].astype(np.float64), np.ones(24)] @ A)[valid]
            P = P[(P[:, 0] > 5) & (P[:, 1] > 5) & (P[:, 0] < 1915) & (P[:, 1] < 1195)]
            if len(P) and len(B):
                d = np.linalg.norm(B[None] - P[:, None], axis=-1).min(1); out[name] += d.tolist()
                if name == 'fit': per[(r['camera'], side)] = per.get((r['camera'], side), []) + d.tolist()
    s = lambda v: dict(n=len(v), median_px=float(np.median(v)), p90_px=float(np.percentile(v, 90)), within_3px=float((np.asarray(v) < 3).mean()))
    rep = dict(labels_checked=len(out['replay']), label_projection_replay_max_px=float(np.max(out['replay'])), nearest_blob_distance={k: s(out[k]) for k in ['fit', 'nokov_corrected', 'nokov_uncorrected']},
               fit_by_camera_side={f'{c}_{"LR"[sd]}': s(v) for (c, sd), v in sorted(per.items())})
    (a.result / 'export/image_evidence_check.json').write_text(json.dumps(rep, indent=1)); print(json.dumps(rep, indent=1))


if __name__ == '__main__': main()
