"""Physiological finger-joint prior for MANO (num_pca_comps=45 -> plain axis-angle hand_pose).

Each finger joint rotation is split (swing-twist) into twist about the rest-pose bone axis, flexion about the joint's
own hinge axis, and sideways abduction. The penalty keeps PIP/DIP as hinges that do not bend backwards, limits MCP
twist / over-abduction, and ties DIP flexion to PIP flexion (human DIP ~ 2/3 of PIP). All four fingers are treated
alike; the thumb is left to the existing MANO scan prior.

MANO hand_pose joint order: index(0-2) middle(3-5) pinky(6-8) ring(9-11) thumb(12-14); within a finger MCP, PIP, DIP.
"""
import numpy as np
import torch
import torch.nn.functional as F

FINGERS = ('index', 'middle', 'ring', 'pinky')
SLOT = {'index': 0, 'middle': 3, 'pinky': 6, 'ring': 9, 'thumb': 12}
JID = {'index': 1, 'middle': 4, 'pinky': 7, 'ring': 10, 'thumb': 13}
TIP = {'thumb': 16, 'index': 17, 'middle': 18, 'ring': 19, 'pinky': 20}
DEG = 180 / np.pi

DEFAULTS = dict(weight=0.05, min_flex_deg=-5., twist_deg=10., abd_deg=10., mcp_twist_deg=15., mcp_abd_deg=25.,
                ratio=2 / 3, ratio_deg=15., ratio_min_pip_deg=10.)


def add_arguments(ap):
    g = ap.add_argument_group('finger joint prior')
    g.add_argument('--joint-limit-weight', type=float, default=0., help='0 disables (previous behaviour)')
    for k in ('min_flex_deg', 'twist_deg', 'abd_deg', 'mcp_twist_deg', 'mcp_abd_deg', 'ratio', 'ratio_deg', 'ratio_min_pip_deg'):
        g.add_argument('--joint-limit-' + k.replace('_', '-'), dest='jl_' + k, type=float, default=DEFAULTS[k])


def config(a):
    return dict(weight=a.joint_limit_weight, **{k: getattr(a, 'jl_' + k) for k in DEFAULTS if k != 'weight'})


def rest_axes(model, beta, dev, include_thumb=False):
    """Per finger joint (index, middle, ring, pinky x MCP, PIP, DIP): slot, bone, flexion and lateral unit axes."""
    with torch.no_grad():
        out = model(betas=beta[None], global_orient=torch.zeros(1, 3, device=dev), hand_pose=-model.hand_mean[None],
                    transl=torch.zeros(1, 3, device=dev), return_tips=True)
    j = out.joints[0]; mean = model.hand_mean.view(15, 3); slots, bones, flexes, lats = [], [], [], []
    for f in (FINGERS + ('thumb',) if include_thumb else FINGERS):
        for k in range(3):
            jid = JID[f] + k; child = jid + 1 if k < 2 else TIP[f]
            b = F.normalize(j[child] - j[jid], dim=0)
            m = mean[SLOT[f] + k]; fl = F.normalize(m - (m @ b) * b, dim=0)   # MANO mean bend sets the flexion sign
            slots.append(SLOT[f] + k); bones.append(b); flexes.append(fl); lats.append(torch.cross(b, fl, dim=0))
    st = lambda x: torch.stack(x)[None]
    return torch.tensor(slots, device=dev), st(bones), st(flexes), st(lats)


def decompose(pose, model, axes):
    """pose: (N,45) relative to the hand mean. Returns twist, flexion, abduction in degrees, shape (N, 4 fingers, 3 joints)."""
    slots, bone, flex, lat = axes
    rv = (pose + model.hand_mean).view(-1, 15, 3)[:, slots]
    ang = rv.norm(dim=-1, keepdim=True).clamp_min(1e-8); ax = rv / ang
    qw, qv = torch.cos(ang / 2), ax * torch.sin(ang / 2)
    twist = 2 * torch.atan2((qv * bone).sum(-1, keepdim=True), qw)
    tw, tv = torch.cos(twist / 2), -bone * torch.sin(twist / 2)                 # inverse twist quaternion
    sw = qw * tw - (qv * tv).sum(-1, keepdim=True); sv = qw * tv + tw * qv + torch.cross(qv, tv, dim=-1)
    sang = 2 * torch.atan2(sv.norm(dim=-1, keepdim=True), sw.abs()) * torch.sign(sw)
    srv = sv / sv.norm(dim=-1, keepdim=True).clamp_min(1e-8) * sang
    shape = (-1, len(slots) // 3, 3)
    return (twist[..., 0] * DEG).view(shape), ((srv * flex).sum(-1) * DEG).view(shape), ((srv * lat).sum(-1) * DEG).view(shape)


def penalty(pose, model, axes, c):
    if c['weight'] <= 0: return pose.new_zeros(())
    tw, fl, ab = decompose(pose, model, axes)
    hinge = tw[..., 1:] / c['twist_deg'], ab[..., 1:] / c['abd_deg'], F.relu(c['min_flex_deg'] - fl[..., 1:]) / 2.
    mcp = tw[..., 0] / c['mcp_twist_deg'], F.relu(ab[..., 0].abs() - c['mcp_abd_deg']) / 2.
    pip, dip = fl[..., 1], fl[..., 2]
    gate = torch.sigmoid((pip - c['ratio_min_pip_deg']) / 3.)                   # only once the PIP is clearly bent
    # one-way: DIP follows PIP; PIP is detached so an unobserved PIP is never straightened to satisfy the ratio
    gate = gate.detach(); ratio = gate * (dip - c['ratio'] * pip.detach()) / c['ratio_deg']
    terms = [x.square().mean() for x in (*hinge, *mcp, ratio)]
    return c['weight'] * sum(terms)


def report(pose, model, axes):
    """Per finger/joint statistics in degrees, for summaries."""
    with torch.no_grad():
        tw, fl, ab = [x.cpu().numpy() for x in decompose(pose, model, axes)]
    out = {}
    for i, f in enumerate(FINGERS):
        for k, jn in enumerate(('MCP', 'PIP', 'DIP')):
            out[f'{f}_{jn}'] = dict(flex_median=float(np.median(fl[:, i, k])), backward_pct=float(100 * np.mean(fl[:, i, k] < -5)),
                                    abd_abs_median=float(np.median(abs(ab[:, i, k]))), twist_abs_median=float(np.median(abs(tw[:, i, k]))))
    pip, dip = fl[:, :, 1], fl[:, :, 2]; bent = pip > 20
    for i, f in enumerate(FINGERS):
        out[f'{f}_dip_over_pip_when_bent'] = float(np.median(dip[bent[:, i], i] / pip[bent[:, i], i])) if bent[:, i].any() else None
    return out
