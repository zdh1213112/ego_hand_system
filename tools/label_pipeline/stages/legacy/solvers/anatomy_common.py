# Source work/group_010_anatomy_20261119/joint_anatomy_20261119/anatomy_common.py, SHA-256 b2b9ab00f3d6fec2a653fcf86d6e58226a3953de3d96c21b946601270d8bf800

import case_settings as case

import json

import numpy as np

import torch

ROOT = case.stage('joint_anatomy_20261119')

from common import EGO, HANDS, HERE, OPT, TIPS, gate_and_errors, geometry, output, project, save, track_file, write_json

from fit_cuff import components, compose, finger_prior, import_mano, run_model

REFERENCE = "pair_20261111_174236"

DIGITS = ("thumb", "index", "middle", "ring", "pinky")

JOINTS = (("CMC", "MCP", "IP"),) + (("MCP", "PIP", "DIP"),) * 4

AXES = ("twist", "flexion", "lateral")

def make_model(device):
    return (
        import_mano(EGO / "third_party/MANO")
        .load(
            model_path=str(EGO / "models/mano"),
            is_rhand=True,
            use_pca=True,
            num_pca_comps=45,
            flat_hand_mean=False,
            batch_size=1,
        )
        .to(device)
        .eval()
        .requires_grad_(False)
    )

def paths(group, hand):
    softroot = output(group)
    outward = softroot / "outward"
    cuff = softroot / "baseline"
    return dict(
        bounded=HERE / group / "bounded/final" / (hand + "_pilot.npz"),
        outward=outward / "tracks" / (hand + ".npz"),
        cuff_hard=cuff / "tracks" / (hand + ".npz"),
        soft35=track_file(group, hand),
    )

def keys(path, names):
    with np.load(path) as z:
        return {k: z[k] for k in names if k in z}

def signed_flex(joints, right):
    # Vectorized form of /tmp/kiro_analysis/flex.py. Mirror left canonical MANO before calling.
    def unit(v):
        return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)

    w = joints[:, 0]
    n = unit(np.cross(joints[:, 5] - w, joints[:, 17] - w)) * (1 if right else -1)
    chains = np.concatenate(
        (np.broadcast_to(w[:, None, None, :], (len(w), 5, 1, 3)), joints[:, 1:].reshape(-1, 5, 4, 3)),
        axis=2,
    )
    b = unit(np.diff(chains, axis=2))
    u, v = b[:, :, :-1], b[:, :, 1:]
    ax = unit(np.cross(u, n[:, None, None, :]))
    return np.rad2deg(np.arctan2((np.cross(u, v) * ax).sum(-1), (u * v).sum(-1)))

def flex_torch(joints):
    # MANO is right-canonical for both hands during optimization.
    unit = torch.nn.functional.normalize
    w = joints[:, 0]
    n = unit(torch.cross(joints[:, 5] - w, joints[:, 17] - w, dim=-1), dim=-1)
    chains = torch.cat(
        (w[:, None, None, :].expand(-1, 5, 1, 3), joints[:, 1:].reshape(-1, 5, 4, 3)), dim=2
    )
    b = unit(torch.diff(chains, dim=2), dim=-1)
    u, v = b[:, :, :-1], b[:, :, 1:]
    ax = unit(torch.cross(u, n[:, None, None, :], dim=-1), dim=-1)
    return torch.rad2deg(torch.atan2((torch.cross(u, v, dim=-1) * ax).sum(-1), (u * v).sum(-1)))

def flags(flex):
    zig = (flex[:, :, 1] * flex[:, :, 2] < 0) & (np.abs(flex[:, :, 1:]).min(axis=2) > 10)
    return flex < -15, zig

def anatomy_stats(flex):
    hyper, zig = flags(flex)
    return dict(
        n=len(flex),
        hyper_percent=(100 * hyper.mean(0)).tolist(),
        zig_percent=(100 * zig.mean(0)).tolist(),
        quantiles=np.percentile(flex, [1, 50, 99], axis=0).tolist(),
    )

def limits(group, hand):
    data = json.loads((HERE / group / "bounded/summary.json").read_text())["hands"][hand]
    return np.array(data["lower_degrees"]), np.array(data["upper_degrees"])

def physical(j, hand):
    return j * np.array([1 if hand == "right" else -1, 1, 1], dtype=j.dtype)

def checkpoint(group, version, hand):
    return ROOT / group / version / (hand + ".npz")

def percentiles(values):
    a = np.asarray(values)
    a = a[np.isfinite(a)]
    return (
        dict(n=len(a), median=float(np.median(a)), p95=float(np.percentile(a, 95)), max=float(a.max()))
        if a.size
        else dict(n=0, median=None, p95=None, max=None)
    )
