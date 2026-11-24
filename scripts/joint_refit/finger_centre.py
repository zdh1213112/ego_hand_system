"""Finger centre-line term: projected MANO finger bones should run along the middle of the glove finger in the image.

For every accepted glove mask view, sample points on the index/middle/ring/pinky bones (joints in MediaPipe order),
project them with the start pose, and find the glove finger's centre along the image direction perpendicular to the
bone (midpoint of the mask run crossing the point). Samples where the run is not closed within the search range
(finger merged with a neighbour or the palm), too narrow, or outside the mask are dropped. During fitting the
residual is the sideways (perpendicular) image distance from the current projected sample to that fixed centre;
the along-finger direction is not constrained, so finger length and flexion stay with the markers.
"""
import numpy as np
import torch
import torch.nn.functional as F
from render_masks import project_native, load_hand_mask

CHAINS = {'index': [5, 6, 7, 8], 'middle': [9, 10, 11, 12], 'ring': [13, 14, 15, 16], 'pinky': [17, 18, 19, 20]}
# (segment index in the chain, fraction along it): distal half of the proximal phalanx, middle and distal phalanx
SAMPLES = [(0, .6), (0, .9), (1, .2), (1, .5), (1, .8), (2, .3), (2, .6)]
SEARCH_PX, MIN_WIDTH_PX = 45, 10


def add_arguments(ap):
    ap.add_argument('--centre-weight', type=float, default=0., help='0 = off. Pull projected finger bones to the glove finger centre line')
    ap.add_argument('--centre-sigma-px', type=float, default=3.)


def _sample(J):
    """J: (..., 21, 2|3) -> (..., 4 fingers * len(SAMPLES), dim) points on the bones."""
    out = []
    for ids in CHAINS.values():
        for s, t in SAMPLES:
            out.append(J[..., ids[s], :] * (1 - t) + J[..., ids[s + 1], :] * t)
    return torch.stack(out, -2) if torch.is_tensor(J) else np.stack(out, -2)


def _seg_dir(J):
    out = []
    for ids in CHAINS.values():
        for s, _ in SAMPLES:
            out.append(J[..., ids[s + 1], :] - J[..., ids[s], :])
    return np.stack(out, -2)


class CentreTargets:
    """Fixed per-view glove finger centres, measured once from the start pose."""

    def __init__(self, records, lookup, joints, side, device):
        self.side, self.device = side, device
        frames, recs, centres, valid = [], [], [], []
        for r in records:
            i = lookup[r['sync']]
            uv, _ = project_native(torch.as_tensor(joints[i:i + 1], dtype=torch.float64), [r], side)
            P = _sample(uv[0].numpy()); D = _seg_dir(uv[0].numpy())
            mask = load_hand_mask(r['mask']) > 0; H, W = mask.shape
            c = np.zeros_like(P); ok = np.zeros(len(P), bool)
            for k, (p, d) in enumerate(zip(P, D)):
                n = np.linalg.norm(d)
                if n < 1e-6: continue
                perp = np.array([-d[1], d[0]]) / n
                ts = np.arange(-SEARCH_PX, SEARCH_PX + 1); q = np.rint(p + ts[:, None] * perp).astype(int)
                inb = (q[:, 0] >= 0) & (q[:, 0] < W) & (q[:, 1] >= 0) & (q[:, 1] < H)
                if not inb.all() or not mask[q[SEARCH_PX, 1], q[SEARCH_PX, 0]]: continue
                ins = mask[q[:, 1], q[:, 0]]; lo = hi = SEARCH_PX
                while lo > 0 and ins[lo - 1]: lo -= 1
                while hi < 2 * SEARCH_PX and ins[hi + 1]: hi += 1
                if lo == 0 or hi == 2 * SEARCH_PX or hi - lo < MIN_WIDTH_PX: continue
                c[k] = p + ((lo + hi) / 2 - SEARCH_PX) * perp; ok[k] = True
            if ok.any():
                frames.append(i); recs.append(r); centres.append(c); valid.append(ok)
        self.records = recs
        self.frame = torch.as_tensor(frames, device=device, dtype=torch.long)
        self.centre = torch.as_tensor(np.array(centres), dtype=torch.float32, device=device) if recs else None
        self.valid = torch.as_tensor(np.array(valid), dtype=torch.float32, device=device) if recs else None
        self.count = int(np.sum(valid)) if recs else 0

    def residual_px(self, joints):
        """joints: (frames, 21, 3) current MANO joints (MediaPipe order) -> (views, samples) signed sideways px."""
        uv, _ = project_native(joints[self.frame].double(), self.records, self.side)
        uv = uv.float(); P = _sample(uv)
        D = []
        for ids in CHAINS.values():
            for s, _ in SAMPLES: D.append(uv[:, ids[s + 1]] - uv[:, ids[s]])
        D = F.normalize(torch.stack(D, 1), dim=-1); perp = torch.stack([-D[..., 1], D[..., 0]], -1)
        side = torch.sign(((uv[:, 17] - uv[:, 5])[:, None] * perp).sum(-1)).clamp_min(-1)  # + = toward the pinky side in this image
        side = torch.where(side == 0, torch.ones_like(side), side)
        return ((P - self.centre) * perp).sum(-1) * side

    def penalty(self, joints, weight, sigma_px):
        if weight <= 0 or self.centre is None: return joints.new_zeros(())
        r = self.residual_px(joints) / sigma_px
        return weight * ((torch.sqrt(1 + r.square()) - 1) * self.valid).sum() / self.valid.sum().clamp_min(1)

    def report(self, joints):
        """Median signed offset per finger in px (+ = toward the pinky side of the finger in this image)."""
        if self.centre is None: return {}
        with torch.no_grad(): r = self.residual_px(joints).cpu().numpy()
        v = self.valid.cpu().numpy() > 0; n = len(SAMPLES); out = {}
        for f, name in enumerate(CHAINS):
            x = r[:, f * n:(f + 1) * n][v[:, f * n:(f + 1) * n]]
            out[name] = dict(median_abs_px=float(np.median(np.abs(x))) if len(x) else None, median_signed_px=float(np.median(x)) if len(x) else None, samples=int(len(x)))
        return out
