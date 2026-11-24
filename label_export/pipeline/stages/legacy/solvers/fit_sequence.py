# Source work/group_003_wrist_outward_20261119/fit_sequence.py, SHA-256 94307fd8dcfb7d418cf2aba1154d4b15d30c38f2a492996a291ed2844a196e60

import case_settings as case

import json

import numpy as np

from scipy.spatial.transform import Rotation

import torch

from torch.nn import functional as F

DEST = case.stage('group_003_wrist_outward_20261119')

WORK = DEST.parent

PREVIOUS = WORK / "joint_motion_20261119"

GROUP = case.recording

ANCHOR = None

from fit_pivot import EGO, HERE, TIPS, components, compose, finger_prior, import_mano, marker_positions, run_model

from observations import geometry

def source_hand(hand, group=GROUP):
    with np.load(PREVIOUS / group / "final" / (hand + "_pilot.npz")) as stream:
        data = {k: stream[k] for k in stream.files}
    clip = PREVIOUS / "fast_fits" / group / hand
    if (clip / "summary.json").is_file() and json.loads((clip / "summary.json").read_text())["accept_for_this_clip"]:
        with np.load(clip / "candidate.npz") as patch:
            shared, positions, subset = np.intersect1d(data["pair_indices"], patch["times"][1:-1], return_indices=True)
            for key in ("joints", "vertices", "hand_pose_pca", "predicted24"):
                data[key][positions] = patch[key][subset + 1]
            residual = np.linalg.norm(data["predicted24"][positions] - data["target24"][positions], axis=-1) * 1000
            data["marker_error_mm"][positions] = np.where(data["marker_valid"][positions], residual, data["marker_error_mm"][positions])
    return data

from wrist_transfer import prepared_targets

def fit_from_reference(data, endpoint, hand, pin_tips_3d=False, *, group=GROUP, anchor=ANCHOR):
    """Transfer a reference in this hand's right-canonical fit space and its own calibrated bounds."""
    times = data["pair_indices"]
    ref = np.flatnonzero(times == anchor).item()
    goals = prepared_targets(data, endpoint, ref)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    def tensor(x): return torch.as_tensor(x, dtype=torch.float32, device=device)
    mano = import_mano(EGO / "third_party/MANO")
    model = mano.load(model_path=str(EGO / "models/mano"), is_rhand=True, use_pca=True,
                      num_pca_comps=45, flat_hand_mean=False, batch_size=1).to(device).eval().requires_grad_(False)
    from mano.lbs import batch_rodrigues
    beta = tensor(data["betas"])
    axes = finger_prior.rest_axes(model, beta, device, include_thumb=True)
    base_angles = components(tensor(data["hand_pose_pca"]), model, axes)
    cfg = json.loads((HERE / group / "bounded/summary.json").read_text())["hands"][hand]
    lower, upper = (tensor(cfg[k]) for k in ("lower_degrees", "upper_degrees"))
    shape = (-1, 1, 1, 1)
    seed = (base_angles + tensor(goals["reach"]).reshape(shape) *
            (tensor(endpoint["angles"]) - base_angles[ref])).clamp(lower, upper)
    orient = tensor(goals["orient"])
    shift = tensor(goals["wrist"] - (data["joints"][:, 0] - data["translation"]))
    tips = tensor(goals["tips"])
    rays = F.normalize(tips, dim=-1)
    target = tensor(goals["mounts"]).nan_to_num()
    usable = torch.as_tensor(data["marker_valid"].copy(), device=device)
    usable[:, :3] = False
    triangles = model.faces_tensor.long()[torch.as_tensor(data["attachment_faces"], device=device).long()]
    attachments = [tensor(data[k]) for k in ("barycentric", "attachment_local_offsets_m", "wrist_local")]
    count = len(times)
    result = {k: np.empty_like(data[k]) for k in ("joints", "vertices", "global_orient", "translation", "hand_pose_pca", "predicted24")}
    for start in range(0, count, 144):
        stop = min(count, start + 144)
        lo, hi = max(0, start - 6), min(count, stop + 6)
        ix = slice(lo, hi)
        local_anchor = ref-lo if lo <= ref < hi else None
        angles = torch.nn.Parameter(seed[ix].clone())
        rot = batch_rodrigues(orient[ix]); n = hi-lo
        connected = np.flatnonzero(np.diff(times[ix]) == 1)
        # No temporal term crosses a missing camera exposure or an unmeasured pose.
        connected = connected[(np.diff(data["_timestamp_ns"][ix])[connected] < data["_max_dt_ns"]) &
                              (np.diff(data["_timestamp_ns"][ix])[connected] > 0)]
        links = torch.as_tensor(connected, device=device)
        optimizer = torch.optim.Adam([angles], lr=.55)
        steps = 106
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps, eta_min=.035)
        def forward():
            pose = compose(angles, model, axes)
            vertices, joints, _ = run_model(model, beta.expand(n, -1), orient[ix], pose, shift[ix])
            mounts = marker_positions(vertices, joints[:, 0], rot, triangles, *attachments)
            return pose, vertices, joints, mounts
        for step in range(steps):
            optimizer.zero_grad(set_to_none=True)
            pose, verts, joints, mounts = forward()
            error = (joints[:, TIPS] - tips[ix]) * 1000
            depth = (error * rays[ix]).sum(-1, keepdim=True)
            tangent = error - depth * rays[ix]
            if pin_tips_3d:
                tip_weight = 14. if step < steps // 2 else 70.
                loss = tip_weight * error.square().mean()
            else:
                loss = 14. * tangent.square().mean() + .7 * depth.square().mean()
            surface = (mounts - target[ix]) * 1000
            loss = loss + .23 * surface.square().sum(-1).masked_fill(~usable[ix], 0.).sum() / usable[ix].sum().clamp_min(1)
            correction = angles - seed[ix]
            loss = loss + .08 * correction.square().mean() + .12 * correction[:, :4, 1:, [0, 2]].square().mean()
            if len(connected):
                loss = loss + .16 * (correction[links + 1] - correction[links]).square().mean()
            loss.backward()
            optimizer.step()
            schedule.step()
            with torch.no_grad():
                angles.clamp_(lower, upper)
                if local_anchor is not None:
                    angles[local_anchor].copy_(tensor(endpoint["angles"]))
        with torch.no_grad():
            pose, vertices, joints, mounts = forward()
            section = slice(start-lo, stop-lo)
            for key, values in (("joints", joints), ("vertices", vertices), ("global_orient", orient[ix]),
                                ("translation", shift[ix]), ("hand_pose_pca", pose), ("predicted24", mounts)):
                result[key][start:stop] = values[section].cpu().numpy()
        if start % 576 == 0 or stop == count: print(hand, stop, "/", count, flush=True)
    np.testing.assert_allclose(result["joints"][ref], endpoint["joints"], rtol=0, atol=2e-6)
    np.testing.assert_allclose(result["vertices"][ref], endpoint["vertices"], rtol=0, atol=2e-6)
    fitted = dict(data, **result)
    fitted.pop("_timestamp_ns"); fitted.pop("_max_dt_ns")
    norm = np.linalg.norm(result["predicted24"] - data["target24"], axis=-1) * 1000
    fitted["marker_error_mm"] = np.where(data["marker_valid"], norm, data["marker_error_mm"]).astype(np.float32)
    root = Rotation.from_rotvec(result["global_orient"]).apply(np.broadcast_to(data["root_anchor_local_m"], (count, 3))) + result["joints"][:, 0]
    fitted["root_anchor_error_mm"] = (np.linalg.norm(root - data["finger0_target"], axis=1)*1000).astype(np.float32)
    fitted["approved_for_training"] = np.array(False)
    return fitted

def save(path, data):
    tmp = path.with_suffix(".partial.npz")
    np.savez_compressed(tmp, **data)
    tmp.replace(path)
