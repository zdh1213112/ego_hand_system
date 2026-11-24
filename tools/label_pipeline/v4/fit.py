"""Improve observed right distal pinky poses on the training split using fixed NOKOV mounts."""

import json
import numpy as np
import torch
from torch.nn import functional as F

from scipy.spatial.transform import Rotation
from fit_mano_sequence import run_model
import finger_prior
from anatomy_common import signed_flex, flags

DEVICE = "cuda"


def put(a):
    return torch.as_tensor(a, device=DEVICE, dtype=torch.float32)


def mount(v, tri, bary, offset):
    pts = v[:, tri]
    tan = F.normalize(pts[:, :, 1] - pts[:, :, 0], dim=-1)
    norm = F.normalize(torch.cross(tan, pts[:, :, 2] - pts[:, :, 0], dim=-1), dim=-1)
    bit = torch.cross(norm, tan, dim=-1)
    return (
        (pts * bary[None, :, :, None]).sum(2)
        + tan * offset[None, :, :1]
        + bit * offset[None, :, 1:2]
        + norm * offset[None, :, 2:3]
    )


def propose(track, model, rows, output, cfg):
    with np.load(track) as f:
        keys = (
            "pair_indices",
            "hand_pose_pca",
            "betas",
            "vertices",
            "joints",
            "translation",
            "global_orient",
            "target24",
            "marker_valid",
            "attachment_faces",
            "barycentric",
            "attachment_local_offsets_m",
            "predicted24",
            "anatomy_lower_degrees",
            "anatomy_upper_degrees",
            "marker_error_mm",
        )
        z = {k: f[k] for k in keys}
    ix = np.where(np.isin(z["pair_indices"], list(rows)))[0]
    assert len(ix) == len(rows), (track, len(ix), len(rows))
    assert z["attachment_faces"].shape == (21,)
    tri = model.faces_tensor.long()[z["attachment_faces"][-4:]]
    bary = put(z["barycentric"][-4:])
    off = put(z["attachment_local_offsets_m"][-4:])
    beta = put(z["betas"])
    axes = finger_prior.rest_axes(model, beta, DEVICE, include_thumb=True)
    basis = torch.zeros(4, 15, 3, device=DEVICE)
    for j in (0, 1):
        basis[2 * j, 7 + j] = axes[2][0, 10 + j]
        basis[2 * j + 1, 7 + j] = axes[3][0, 10 + j]
    basis = basis.reshape(4, 45) * (np.pi / 180)
    ids = []
    old_err = []
    new_err = []
    before_all = []
    sol = []
    delta_v = []
    delta_j = []
    ori_mat = []
    pts_old = []
    pts_new = []
    maxreplay = 0.0
    maxmount = 0.0
    for start in range(0, len(ix), cfg["batch_size"]):
        pi = ix[start : start + cfg["batch_size"]]
        n = len(pi)
        ori = put(z["global_orient"][pi])
        trans = put(z["translation"][pi])
        p0 = put(z["hand_pose_pca"][pi])
        target = put(z["target24"][pi, -4:])
        observed = torch.as_tensor(z["marker_valid"][pi, -4:], device=DEVICE) & torch.isfinite(target).all(-1)
        eligible = observed[:, -1] & observed[:, -2] & (observed.sum(1) >= cfg["min_observed_markers"])

        def forward(delta):
            p = p0 + delta @ basis
            v, j, out = run_model(model, beta.expand(n, -1), ori, p, trans)
            return mount(v, tri, bary, off), v, j, p

        zero = put(np.zeros((n, 4)))
        m0, v0, j0, _ = forward(zero)
        maxreplay = max(
            maxreplay,
            float((v0 - put(z["vertices"][pi])).abs().max()),
            float((j0 - put(z["joints"][pi])).abs().max()),
        )
        maxmount = max(maxmount, float((m0 - put(z["predicted24"][pi, -4:])).abs().max()))
        assert max(maxreplay, maxmount) < cfg["replay_tolerance_m"], (track, maxreplay, maxmount)
        target = torch.where(observed[..., None], target, m0)
        w = (observed * put(cfg["marker_weights"])).sqrt()[..., None]
        jac = torch.stack(
            [((forward(zero + v * cfg["finite_difference_deg"])[0] - m0) * (1000 / cfg["finite_difference_deg"]) * w).flatten(1) for v in torch.eye(4, device=DEVICE)],
            dim=-1,
        )
        resid = ((target - m0) * 1000 * w).flatten(1)
        step = torch.linalg.solve(
            jac.mT @ jac + torch.diag(put(cfg["regularization"])), (jac.mT @ resid.unsqueeze(-1))
        ).squeeze(-1)
        bounds = put(cfg["max_angle_step_deg"])
        step = torch.maximum(torch.minimum(step, bounds), -bounds)
        error0 = ((m0 - target) * 1000).norm(dim=-1)
        before_all.extend(error0[eligible, -1].cpu().numpy().tolist())
        eligible = eligible & (error0[:, -1] > cfg["min_nail_residual_mm"])
        initial_angle = torch.stack(finger_prior.decompose(p0, model, axes), -1)[:, 3]
        lo = put(z["anatomy_lower_degrees"][3])
        hi = put(z["anatomy_upper_degrees"][3])
        h0, zig0 = flags(signed_flex(j0.cpu().numpy(), True))
        for factor in cfg["line_search"]:
            if not eligible.any():
                break
            m, v, j, p = forward(step * factor)
            err = ((m - target) * 1000).norm(dim=-1)
            a = torch.stack(finger_prior.decompose(p, model, axes), -1)[:, 3]
            anatomy = (
                (a >= torch.minimum(initial_angle, lo) - cfg["angle_tolerance_deg"])
                & (a <= torch.maximum(initial_angle, hi) + cfg["angle_tolerance_deg"])
            ).all((1, 2))
            h1, zig1 = flags(signed_flex(j.cpu().numpy(), True))
            new_flags = torch.from_numpy((h1 & ~h0).any((1, 2)) | ((zig1 & ~zig0).any(1))).to(DEVICE)
            candidates = (
                eligible
                & anatomy
                & ~new_flags
                & (err[:, -1] < error0[:, -1] - cfg["min_nail_gain_mm"])
                & ((err[:, :3] <= error0[:, :3] + cfg["max_other_marker_loss_mm"]) | ~observed[:, :3]).all(1)
            )
            candidates &= ((j[:, 20] - j0[:, 20]) * 1000).norm(dim=-1) <= cfg["max_tip_displacement_mm"]
            candidates &= ((err.square() - error0.square()) * observed).sum(1) < 0
            ids0 = torch.where(candidates)[0]
            if len(ids0):
                ki = ids0.cpu().numpy()
                old_err.extend(error0[ids0, -1].cpu().numpy())
                new_err.extend(err[ids0, -1].cpu().numpy())
                ids.extend(z["pair_indices"][pi[ki]])
                sol.append(
                    Rotation.from_rotvec((p[ids0] + model.hand_mean).cpu().numpy().reshape(-1, 3))
                    .as_matrix()
                    .reshape(-1, 15, 3, 3)
                    .astype("f")
                )
                delta_v.append((v[ids0] - v0[ids0]).cpu().numpy())
                delta_j.append((j[ids0] - j0[ids0]).cpu().numpy())
                ori_mat.append(Rotation.from_rotvec(ori[ids0].cpu().numpy()).as_matrix().astype("f"))
                pts_old.append(m0[ids0, -1].cpu().numpy())
                pts_new.append(m[ids0, -1].cpu().numpy())
                eligible &= ~candidates
    if len(ids):
        np.savez_compressed(
            output,
            sync=np.asarray(ids),
            pose=np.concatenate(sol),
            dv=np.concatenate(delta_v),
            dj=np.concatenate(delta_j),
            ori=np.concatenate(ori_mat),
            nail_before=np.concatenate(pts_old),
            nail_after=np.concatenate(pts_new),
        )
    metric = dict(
        train_right_poses=len(ix),
        observed=len(before_all),
        corrected=len(ids),
        baseline_observed_p50_mm=float(np.median(before_all)) if before_all else None,
        replay_abs_max_m=maxreplay,
        mount_abs_max_m=maxmount,
    )
    if ids:
        metric.update(
            paired_before_p50_mm=float(np.median(old_err)),
            paired_after_p50_mm=float(np.median(new_err)),
            mean_gain_mm=float(np.mean(np.array(old_err) - new_err)),
        )
    print(json.dumps(metric), flush=True)
    return metric
