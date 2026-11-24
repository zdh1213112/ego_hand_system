# Source work/attach_ring_prior_group003_20261118/prepare.py, SHA-256 a44a26c8ea82b397329e6d4fa8fdcfbc77325b8b16bd0cb0e1490cb0b6ba46f5

import case_settings as case

import hashlib, json, sys

from pathlib import Path

import numpy as np, torch

from fit_mano_sequence import import_mano

PROJECT=case.ego

ROW = {'FingerRing2': 14}

BONE = {'FingerRing2': [10, 11], 'FingerPinky2': [7, 8]}      # proximal and middle phalanx of the finger

CHAIN = {'FingerRing2': [10, 11, 12, 19], 'FingerPinky2': [7, 8, 9, 20]}

MIN_DORSAL = .85

def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def rest(model, betas):
    with torch.no_grad():
        out = model(betas=torch.as_tensor(betas, dtype=torch.float32)[None], global_orient=torch.zeros(1, 3),
                    hand_pose=-model.hand_mean[None], transl=torch.zeros(1, 3), return_tips=True)
    return out.vertices[0].numpy(), out.joints[0].numpy()

def chain_pos(p, pts):
    """Position of point p along a joint chain: integer part = segment, fraction = along it."""
    best = None
    for s in range(len(pts) - 1):
        a, b = pts[s], pts[s + 1]; t = float(np.clip((p - a) @ (b - a) / ((b - a) @ (b - a)), 0, 1))
        d = np.linalg.norm(a + t * (b - a) - p)
        if best is None or d < best[0]: best = (d, s + t, a + t * (b - a), (b - a) / np.linalg.norm(b - a))
    return best[1], best[2], best[3]

def plane_bary(tri, dist):
    pts = [np.eye(3)[k] for k in range(3) if abs(dist[k]) < 1e-12]
    for a, b in [(0, 1), (1, 2), (2, 0)]:
        if dist[a] * dist[b] < 0:
            t = dist[a] / (dist[a] - dist[b]); w = np.zeros(3); w[a], w[b] = 1 - t, t; pts.append(w)
    return np.mean(pts, 0) if pts else None

def prepare_recording(optimized, output):
    out = output / 'aligned'
    if out.exists(): raise FileExistsError(out)
    model = import_mano(PROJECT / 'third_party/MANO').load(model_path=str(PROJECT / 'models/mano'), is_rhand=True, use_pca=True,
                                                         num_pca_comps=45, flat_hand_mean=False, batch_size=1)
    faces = model.faces_tensor.long().numpy(); W = model.lbs_weights.detach().numpy(); owner = W[faces].mean(1).argmax(1)
    report = dict(experiment='Ring2 only moved to the dorsal surface of their own finger at calibration axial position',
                  source=str(optimized / 'aligned'), approved_for_training=False, inputs={}, hands={})
    out.mkdir(parents=True)
    for hand in ['left', 'right']:
        calib = np.load(optimized / 'calibration' / f'{hand}_pilot.npz'); src = optimized / 'aligned' / f'{hand}_pilot.npz'; z = dict(np.load(src))
        v, j = rest(model, z['betas']); tri = v[faces]
        nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]); nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
        lat = j[7] - j[1]; lat /= np.linalg.norm(lat); fwd = j[4] - j[0]; fwd /= np.linalg.norm(fwd)
        dorsal = np.cross(fwd, lat); dorsal /= np.linalg.norm(dorsal)
        if dorsal[1] < 0: dorsal = -dorsal
        dcos = nrm @ dorsal; new = {k: z[k].copy() for k in z}; rows = {}
        for name, r in ROW.items():
            pts = j[CHAIN[name]]
            old_site = (v[faces[calib['attachment_faces'][r]]] * calib['barycentric'][r][:, None]).sum(0)
            pos, foot, axis = chain_pos(old_site, pts)                 # calibration axial position, kept
            dist = (tri - foot) @ axis                                 # signed distance to the plane through `foot`
            cand = np.flatnonzero(np.isin(owner, BONE[name]) & (dcos >= MIN_DORSAL) & (dist.min(1) <= 0) & (dist.max(1) >= 0))
            best = None
            for fidx in cand:
                bary = plane_bary(tri[fidx], dist[fidx])
                if bary is None: continue
                p = (tri[fidx] * bary[:, None]).sum(0)
                off_axis = p - foot - ((p - foot) @ axis) * axis          # radial vector from bone axis
                score = float((off_axis / np.linalg.norm(off_axis)) @ dorsal)  # most dorsal-centred point of the section
                if best is None or score > best[0]: best = (score, int(fidx), bary, p)
            if best is None: raise ValueError(f'{hand} {name}: no dorsal face on the axial plane')
            _, fidx, bary, p = best
            new['attachment_faces'][r] = fidx; new['barycentric'][r] = bary.astype(z['barycentric'].dtype)
            rows[name] = dict(old_face=int(z['attachment_faces'][r]), old_dorsal_cos=round(float(dcos[z['attachment_faces'][r]]), 3),
                              old_chain_pos=round(chain_pos((v[faces[z['attachment_faces'][r]]] * z['barycentric'][r][:, None]).sum(0), pts)[0], 3),
                              calibration_chain_pos=round(pos, 3), new_face=fidx, new_dorsal_cos=round(float(dcos[fidx]), 3),
                              new_chain_pos=round(chain_pos(p, pts)[0], 3), new_bone=int(owner[fidx]))
        for k in z:
            if k not in ('attachment_faces', 'barycentric'): np.testing.assert_equal(z[k], new[k])
        keep = np.ones(len(z['attachment_faces']), bool); keep[list(ROW.values())] = False
        np.testing.assert_equal(z['attachment_faces'][keep], new['attachment_faces'][keep]); np.testing.assert_equal(z['barycentric'][keep], new['barycentric'][keep])
        np.savez_compressed(out / src.name, **new)
        report['inputs'][str(src)] = digest(src); report['hands'][hand] = rows
        print(hand, json.dumps(rows), flush=True)
    (output / 'preparation.json').write_text(json.dumps(report, indent=1))
