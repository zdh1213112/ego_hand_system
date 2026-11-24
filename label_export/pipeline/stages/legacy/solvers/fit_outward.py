# Source work/mcp_outward_20261119/fit_outward.py, SHA-256 d07228f7f211b0b22c81b4119f83210956b9b7b94147aa3eba8c5b55a5a6a234

import case_settings as case

import numpy as np

import torch

import torch.nn.functional as F

from scipy.optimize import least_squares

from scipy.spatial.transform import Rotation

STUDY_SOURCE = case.stage('mcp_outward_20261119')

HERE = STUDY_SOURCE

WORK = STUDY_SOURCE.parent

PREV = WORK / "direction_acceptance_20261119"

from fit_wide import HANDS, WEIGHTS

from anatomy_common import TIPS, components, compose, finger_prior, flex_torch, run_model, write_json

from rounds import Fit, apply_mounts, load, make_split

from calibrate_mounts import patch_fit, units

from scipy.spatial.distance import cdist

ROOTS = np.array([5, 9, 13, 17])

STRENGTHS = {"mild": {"left": 1.5, "right": 3.0}, "strong": {"left": 3.0, "right": 5.0}}

def base_path(group, hand):
    return PREV / group / "distal_anatomy" / (hand + ".npz")

def unit(x):
    return x / np.linalg.norm(x)

def palm_calibration(fit, old_beta, distance):
    """One small rotation and one shape per recording/hand; fix J0 as the origin."""
    model = fit.model
    vertices = model.v_template.cpu().numpy().astype(float)
    basis = model.shapedirs.cpu().numpy().astype(float)
    regressor = model.J_regressor.cpu().numpy().astype(float)
    tips = list(model.tip_ids.values())
    order = fit.kin.order.cpu().numpy()
    rest = np.concatenate((regressor @ vertices, vertices[tips]))[order] * 1000
    basis = (
        np.concatenate((np.einsum("jv,vck->jck", regressor, basis), basis[tips]))[order]
        * 1000
    )

    def points(beta):
        return rest + np.einsum("jck,k->jc", basis, beta)

    old = points(old_beta)
    long = unit(old[ROOTS].mean(0) - old[0])
    across = old[17] - old[5]
    outer = unit(across - (across @ long) * long)
    target = old[ROOTS] - old[0] + distance * outer

    def lengths(joints):
        return np.linalg.norm(np.diff(joints[1:].reshape(5, 4, 3), axis=1), axis=-1)

    old_lengths = lengths(old)

    def residual(p):
        j = points(old_beta + p[3:])
        moved = Rotation.from_rotvec(p[:3]).apply(j - j[0])
        return np.r_[
            (moved[ROOTS] - target).ravel(),
            14 * (moved[1] - old[1] + old[0]),
            (
                (lengths(j) - old_lengths)
                * np.array([2.2, 0.85, 0.85, 0.85, 0.85])[:, None]
            ).ravel(),
            0.70 * p[3:],
            4 * p[:3],
        ]

    bounds = np.r_[np.repeat(0.17, 3), np.repeat(1.0, 10)]
    p = least_squares(
        residual, np.zeros(13), bounds=(-bounds, bounds), ftol=1e-12, xtol=1e-12
    ).x
    beta = (old_beta + p[3:]).astype(np.float32)
    rot = Rotation.from_rotvec(p[:3])
    j = points(beta)
    shift = rot.apply(j - j[0]) - (old - old[0])
    calibration = dict(
        target_outward_mm=distance,
        basis="Index to little finger across the palm, perpendicular to the wrist-to-knuckle line.",
        palm_axis_local=outer.tolist(),
        local_root_rotation_rad=p[:3].tolist(),
        local_root_rotation_deg=float(np.rad2deg(np.linalg.norm(p[:3]))),
        old_betas=old_beta.tolist(),
        betas=beta.tolist(),
        achieved_outward_mm=(shift[ROOTS] @ outer).tolist(),
        nonlateral_mm=np.linalg.norm(
            shift[ROOTS] - np.outer(shift[ROOTS] @ outer, outer), axis=1
        ).tolist(),
        thumb_cmc_shift_mm=float(np.linalg.norm(shift[1])),
        bone_length_delta_mm=(lengths(j) - old_lengths).tolist(),
    )
    return beta, rot, calibration

def prepare(fit, z, beta, relative):
    # Counter-rotate each finger base so that the palm rotation by itself does not rotate the fingertips.
    rot = Rotation.from_rotvec(z["global_orient"]) * relative
    mean = fit.model.hand_mean.cpu().numpy()
    pose = (z["hand_pose_pca"] + mean).reshape(-1, 15, 3).astype(float)
    slots = [0, 3, 6, 9, 12]
    pose[:, slots] = (
        (relative.inv() * Rotation.from_rotvec(pose[:, slots].reshape(-1, 3)))
        .as_rotvec()
        .reshape(-1, 5, 3)
    )
    pose = fit.tensor(pose.reshape(-1, 45) - mean)
    fit.axes = finger_prior.rest_axes(
        fit.model, fit.tensor(beta), fit.dev, include_thumb=True
    )
    apply_mounts(fit, {**z, "betas": beta})
    fingers = (
        components(pose, fit.model, fit.axes).flatten(1).clamp(fit.lo[:45], fit.hi[:45])
    )
    translation = fit.tensor(z["joints"][:, 0]) * 1000 - fit.kin.rest[:1] * 1000
    return torch.cat(
        (fingers, fit.tensor(np.rad2deg(rot.as_rotvec())), translation), dim=1
    )

@torch.no_grad()
def bounded_step(gram, rhs, lower, upper):
    """Keep the derivative of parameters at bounds out of the remaining Newton solve."""
    n, dims = rhs.shape
    fixed = torch.zeros((n, dims), dtype=torch.bool, device=rhs.device)
    value = torch.zeros_like(rhs)
    for _ in range(8):
        free = ~fixed
        a = gram * (free[:, :, None] & free[:, None, :]) + torch.diag_embed(
            fixed.to(gram.dtype)
        )
        b = torch.where(fixed, value, rhs - (gram @ value[:, :, None]).squeeze(2))
        step = torch.linalg.solve(a, b.unsqueeze(2)).squeeze(2)
        hit = free & ((step < lower - 1e-5) | (step > upper + 1e-5))
        if not hit.any():
            return step.clamp(lower, upper)
        fixed |= hit
        value = torch.where(hit, step.clamp(lower, upper), value)
    return step.clamp(lower, upper)

@torch.no_grad()
def fit_fingers(fit, z, seed, indices):
    """Bounded batched Gauss–Newton with the measured surface points and original fingertips."""
    device = fit.dev
    old_joints = fit.tensor(z["joints"]) * 1000
    old_error = fit.tensor(np.nan_to_num(z["marker_error_mm"][:, 3:]))
    cap = fit.tensor(z["tip_cap_per_pose_mm"])
    outputs = []
    dirs = torch.eye(45, device=device)
    selectable = fit.tensor([1.0, 0.55, 0.25, 0.09, 0.0])
    weights = np.tile([0.0, 1.0, 1.15, 8.0], 5)
    weights[:4] = [0.0, 9.0, 10.0, 14.0]
    joint_w = fit.tensor(weights)
    for start in range(0, len(indices), 32):
        ids = indices[start : start + 32]
        n = len(ids)
        q0 = seed[ids]
        initial = old_joints[ids]
        flex0 = flex_torch(initial)
        f0a, f0b = flex0[:, :, 1], flex0[:, :, 2]
        zig0 = torch.maximum(torch.minimum(f0a, -f0b), torch.minimum(-f0a, f0b))
        zig_limit = torch.maximum(zig0, zig0.new_tensor(9.88))
        lower = torch.minimum(flex0, fit.range_min + 0.02)
        with torch.no_grad():
            delta = fit.kin.evaluate(q0, False)[:, ROOTS] - initial[:, ROOTS]
        target = initial[:, 1:].clone()
        target[:, ROOTS] += 0.38 * delta  # next joint (PIP); joint 0 has been removed
        target[:, ROOTS + 1] += 0.12 * delta
        valid = fit.valid[ids].bool()
        base_e = old_error[ids]
        med = torch.nanquantile(base_e.masked_fill(~valid, torch.nan), 0.5, dim=1)
        surface_med = med.clamp_min(2.48)
        maximum = base_e.amax(1).clamp_min(7.95)
        max_tip = torch.maximum(
            cap[ids], (initial[:, TIPS] - fit.tip[ids]).norm(dim=-1).amax(1)
        )

        def residual(x):
            k = x.shape[1]
            fixed = q0[:, None, 45:].expand(-1, k, -1)
            full = torch.cat((x, fixed), dim=-1)
            joints, m, _, _ = fit.kin.evaluate(full.reshape(-1, 51))
            flex = flex_torch(joints).reshape(n, k, 5, 3)
            joints = joints.reshape(n, k, 21, 3)
            m = m.reshape(n, k, 21, 3)
            marker = (m - fit.targets[ids, None]) * valid[:, None, :, None]
            err = marker.norm(dim=-1)
            med = torch.nanquantile(
                err.masked_fill(~valid[:, None], torch.nan), 0.5, dim=2
            )
            tip = (joints[:, :, TIPS] - fit.tip[ids, None]).norm(dim=-1)
            a, b = flex[:, :, :, 1], flex[:, :, :, 2]
            zig = torch.maximum(torch.minimum(a, -b), torch.minimum(-a, b))
            pieces = [
                (joints[:, :, 1:] - target[:, None]) * joint_w[None, None, :, None],
                0.78 * marker,
                0.18 * (flex[:, :, :, 1:] - flex0[:, None, :, 1:]),
                0.09 * (x - q0[:, None, :45]),
                55 * F.relu(tip - max_tip[:, None, None] + 0.055),
                8 * F.relu(med - surface_med[:, None]),
                4 * F.relu(err.amax(2) - maximum[:, None]),
                3 * F.relu(zig - zig_limit[:, None]),
                2 * F.relu(lower[:, None] - flex),
            ]
            return torch.cat([a.reshape(n, k, -1) for a in pieces], dim=-1)

        x = q0[:, :45].clone()
        eye = torch.eye(45, device=device)
        for it in range(24):
            perturbed = torch.cat((x[:, None], x[:, None] + 0.055 * dirs), dim=1)
            r = residual(perturbed)
            jac = (r[:, 1:] - r[:, :1]) / 0.055
            gram = jac @ jac.transpose(1, 2) + (0.15 if it < 6 else 0.04) * eye
            rhs = -(jac @ r[:, 0, :, None]).squeeze(2)
            step = bounded_step(
                gram,
                rhs,
                (fit.lo[:45] - x).clamp_min(-5),
                (fit.hi[:45] - x).clamp_max(5),
            )
            proposals = (x[:, None] + selectable[None, :, None] * step[:, None]).clamp(
                fit.lo[:45], fit.hi[:45]
            )
            candidates = residual(proposals).square().sum(2)
            choice = candidates.argmin(1)
            updated = proposals[torch.arange(n, device=device), choice]
            change = (updated - x).abs().max().item()
            x = updated
            if change < 0.01:
                break
        outputs.append(torch.cat((x, q0[:, 45:]), dim=1))
    return torch.cat(outputs)

@torch.no_grad()
def fit_surface_patches(fit, z, seed, beta):
    """Refit the same named mounts on fixed neighboring surface patches using training times only."""
    train, held = make_split(fit.group, fit.hand, fit.ts)
    ii = np.flatnonzero(train & ~z["isolated_pose"])
    ii = ii[np.linspace(0, len(ii) - 1, min(116, len(ii))).round().astype(int)]
    assert not held[ii].any()
    q = fit_fingers(fit, z, seed, ii)
    pose = compose(q[:, :45].reshape(-1, 5, 3, 3), fit.model, fit.axes)
    verts, _, _ = run_model(
        fit.model,
        fit.tensor(beta).expand(len(ii), -1),
        torch.deg2rad(q[:, 45:48]),
        pose,
        q[:, 48:] / 1000,
    )
    mesh = verts.cpu().numpy().astype(float) * 1000
    faces = z["faces"].astype(int)
    normal = units(
        np.cross(
            mesh[:, faces[:, 1]] - mesh[:, faces[:, 0]],
            mesh[:, faces[:, 2]] - mesh[:, faces[:, 0]],
        )
    )
    face = z["attachment_faces"].copy()
    bary = z["barycentric"].copy()
    offset = z["attachment_local_offsets_m"].copy()
    assert np.max(np.abs(offset[:, :2])) < 1e-6
    old_full = fit.kin.evaluate(q)[1].cpu().numpy()
    rest = (
        fit.model.v_template
        + (fit.model.shapedirs * fit.tensor(beta)[None, None, :]).sum(2)
    ).cpu().numpy() * 1000
    rows = {}
    for m in [3, *range(8, 24)]:
        a = m - 3
        use = z["marker_valid"][ii, m]
        if use.sum() < 6:
            continue
        tr = mesh[use]
        obs = z["target24"][ii[use], m].astype(float) * 1000
        old = old_full[use, a]
        e0 = np.linalg.norm(old - obs, axis=1)
        anchor = int(face[a])
        src = bary[a] @ rest[faces[anchor]]
        # Retain the connected surface around the existing face, including for close-by folded fingers.
        radius = 2.5 if m % 4 == 3 and m != 3 else 4.5
        near = (cdist(rest, src[None])[:, 0] <= radius + 2.0)[faces].any(1)
        near &= (
            np.median((normal * normal[:, anchor : anchor + 1]).sum(-1), axis=0) >= 0.82
        )
        near[anchor] = True
        pool = np.flatnonzero(near)
        connected = {anchor}
        for _ in range(5):
            vertices = np.unique(faces[list(connected)])
            next_faces = (
                set(pool[(np.isin(faces[pool], vertices).sum(1) >= 2)].tolist())
                | connected
            )
            if next_faces == connected:
                break
            connected = next_faces
        best_cost = np.mean(e0**2)
        best = (anchor, bary[a], float(offset[a, 2] * 1000), 0.0, e0)
        for f in sorted(connected):
            bc, height, e = patch_fit(tr[:, faces[f]], obs, (-2.0, 8.0))
            pt = (tr[:, faces[f]] * bc[None, :, None]).sum(1) + height * normal[use, f]
            travel = np.linalg.norm(pt - old, axis=1)
            cost = np.mean(e**2) + 0.10 * np.mean(travel**2)
            if travel.max() <= radius and cost < best_cost:
                best_cost = cost
                best = (f, bc, height, float(np.median(travel)), e)
        f, b, h, travel, e = best
        face[a] = f
        bary[a] = b
        offset[a] = [0, 0, h / 1000]
        rows[str(z["marker_names"][m])] = dict(
            n=int(use.sum()),
            old_face=anchor,
            face=int(f),
            model_mount_move_median_mm=travel,
            before_mm=float(np.median(e0)),
            after_mm=float(np.median(e)),
            height_mm=float(h),
        )
    return dict(
        attachment_faces=face,
        barycentric=bary,
        attachment_local_offsets_m=offset,
        normal_offsets_m=offset[:, 2].copy(),
        mcp_mount_calibration_sync=fit.ts[ii],
    ), rows

def run(
    group,
    hand,
    variant,
    pilot=False,
    calibration_only=False,
    *,
    study=HERE,
    target_mm=None,
):
    dest = study / group / variant
    dest.mkdir(parents=True, exist_ok=True)
    source = base_path(group, hand)
    z = load(source)
    fit = Fit(group, hand, bounds_folder=dest / "bounds")
    distance = STRENGTHS[variant][hand] if target_mm is None else target_mm
    beta, relative, record = palm_calibration(fit, z["betas"], distance)
    print(
        group,
        hand,
        variant,
        "root displacement",
        record["achieved_outward_mm"],
        "thumb CMC",
        record["thumb_cmc_shift_mm"],
        flush=True,
    )
    write_json(dest / (hand + "_palm.json"), record)
    if calibration_only:
        return
    seed = prepare(fit, z, beta, relative)
    mounts, details = fit_surface_patches(fit, z, seed, beta)
    write_json(
        dest / (hand + "_mounts.json"),
        dict(
            calibration_sync=mounts["mcp_mount_calibration_sync"].tolist(),
            markers=details,
        ),
    )
    apply_mounts(fit, {**z, **mounts, "betas": beta})
    indices = np.arange(len(seed))
    if pilot:
        examples = [883, 1018, 3955, 3963, 4129, 4136]
        ids = np.linspace(0, len(seed) - 1, 68).astype(int)
        indices = np.unique(np.r_[ids, np.flatnonzero(np.isin(fit.ts, examples))])
    q = fit_fingers(fit, z, seed, indices)
    if pilot:
        with torch.no_grad():
            j, m, *_ = fit.kin.evaluate(q)
        np.savez_compressed(
            dest / (hand + "_pilot.npz"),
            sync_index=fit.ts[indices],
            solution_q=q.cpu().numpy(),
            joints=j.cpu().numpy() / 1000,
            mounts=m.cpu().numpy() / 1000,
        )
        return
    fit.store(
        "mcp_outward_" + variant,
        q,
        indices,
        z["tip_relaxed"],
        WEIGHTS[hand],
        source=source,
        destination=dest / (hand + ".npz"),
        mounts=dict(**mounts, betas=beta, approved_for_training=np.array(False)),
    )
