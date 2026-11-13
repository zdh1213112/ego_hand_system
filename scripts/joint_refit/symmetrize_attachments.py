#!/usr/bin/env python3
"""Make marker attachment sites left/right consistent and dorsal.

Both gloves carry the same marker layout, yet the automatically estimated attachment faces of some markers differ
between hands by >1 cm, one of them sitting on the side of the finger. Markers nearly collinear along a finger cannot
resolve that: the phalanx just rolls about the marker line. For every marker whose two sites disagree, keep the more
dorsal one (flat-hand normal); if neither is dorsal, snap to the most dorsal face of the same phalanx at the same
axial position. Writes a copy of the source fit with replaced attachment_faces / barycentric (poses untouched).
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[2].parent / 'wilor_training'
sys.path.insert(0, str(ROOT / 'tools/personal_hand_calibration'))
from run import PROJECT, import_mano, run_model


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--source', type=Path, required=True); ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--disagree-mm', type=float, default=8.); ap.add_argument('--dorsal-min', type=float, default=.6)
    a = ap.parse_args(); a.output.mkdir(parents=True)
    model = import_mano(PROJECT / 'third_party/MANO').load(model_path=str(PROJECT / 'models/mano'), is_rhand=True, use_pca=True, num_pca_comps=45, flat_hand_mean=False, batch_size=1)
    F = model.faces_tensor.numpy(); v, _, _ = run_model(model, torch.zeros(1, 10), torch.zeros(1, 3), -model.hand_mean[None], torch.zeros(1, 3)); v = v[0].numpy()  # flat hand: dorsal = +y, fingers along x
    cen = v[F].mean(1); nrm = np.cross(v[F[:, 1]] - v[F[:, 0]], v[F[:, 2]] - v[F[:, 0]]); nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    bone = model.lbs_weights.numpy()[F].mean(1).argmax(1)
    present = [h for h in ['left', 'right'] if (a.source / f'{h}_pilot.npz').is_file()]
    if not present: raise ValueError('source has no hand fit')
    z = {h: dict(np.load(a.source / f'{h}_pilot.npz')) for h in present}; names = [str(x) for x in z[present[0]]['marker_names'][3:]]
    site = {h: (v[F[z[h]['attachment_faces']]] * z[h]['barycentric'][:, :, None]).sum(1) for h in z}; dors = {h: nrm[z[h]['attachment_faces'], 1] for h in z}
    report = {}
    if len(present) < 2:
        print('single hand recorded; no other hand to compare, attachments copied unchanged', flush=True)
    for k, name in enumerate(names):
        if len(present) < 2: break
        gap = float(np.linalg.norm(site['left'][k] - site['right'][k]) * 1000)
        if gap < a.disagree_mm or name == 'HandOffset': continue  # the palm marker has no roll ambiguity; keep per-hand estimate
        best = max(z, key=lambda h: dors[h][k]); face, bary = int(z[best]['attachment_faces'][k]), z[best]['barycentric'][k].copy(); how = f'copied from {best}'
        if dors[best][k] < a.dorsal_min and 'Thumb' not in name:
            cand = np.flatnonzero((bone == bone[face]) & (np.abs(cen[:, 0] - site[best][k][0]) < .004))
            face = int(cand[nrm[cand, 1].argmax()]); bary = np.full(3, 1 / 3, np.float32); how = 'snapped to dorsal face'
        for h in z:
            z[h]['attachment_faces'] = z[h]['attachment_faces'].copy(); z[h]['barycentric'] = z[h]['barycentric'].copy()
            z[h]['attachment_faces'][k] = face; z[h]['barycentric'][k] = bary
        report[name] = dict(left_right_gap_mm=round(gap, 1), dorsal_left=round(float(dors['left'][k]), 2), dorsal_right=round(float(dors['right'][k]), 2), action=how, dorsal_new=round(float(nrm[face, 1]), 2))
        print(name, report[name], flush=True)
    for h in z: np.savez_compressed(a.output / f'{h}_pilot.npz', **z[h])
    (a.output / 'attachment_changes.json').write_text(json.dumps(dict(source=str(a.source.resolve()), approved_for_training=False, changes=report), indent=1, ensure_ascii=False))


if __name__ == '__main__': main()
