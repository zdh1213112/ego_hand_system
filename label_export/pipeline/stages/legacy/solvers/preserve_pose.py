# Source work/mcp_outward_20261119/preserve_pose.py, SHA-256 b6b148a6c23ffb3735fe4c5d7bf5bce51b70b4c180f611870a6b65d20b6e9b2f

import case_settings as case

import json

import numpy as np

import torch

import torch.nn.functional as F

from scipy.spatial.transform import Rotation

from fit_outward import HERE, HANDS, ROOTS, base_path, bounded_step

from anatomy_common import OPT, TIPS, finger_prior, flex_torch, geometry, physical

from fit_wide import WEIGHTS

from rounds import Fit, apply_mounts, load

from observed import measure, sdk_in_fit_space, uv_all

from refine_distal import CameraTurns

@torch.no_grad()
def run(group, hand, *, study=HERE):
    folder = study / group
    dest = folder / "selected"
    dest.mkdir(exist_ok=True)
    src = folder / "mild" / (hand + ".npz")
    x = load(src)
    old = load(base_path(group, hand))
    fit = Fit(group, hand, bounds_folder=dest / "bounds")
    fit.axes = finger_prior.rest_axes(
        fit.model, fit.tensor(x["betas"]), fit.dev, include_thumb=True
    )
    apply_mounts(fit, x)
    _, cams, rt, _ = geometry(group)
    project_turn = CameraTurns(fit, cams, rt)
    oj = old["joints"].astype(float)
    orig_uv = uv_all(physical(oj, hand), cams, rt)
    oa, ov = measure(orig_uv, cams)
    with np.load(OPT[group] / "skeleton/aligned_skeleton.npz") as s:
        sdk = sdk_in_fit_space(
            group,
            HANDS.index(hand),
            s["positions_cam0_m"][:, :, s["provisional21_indices"]].astype(float),
            s["marker24_cam0_m"].astype(float),
            fit.ts,
        )
    sa, sv = measure(uv_all(sdk, cams, rt), cams)
    ov[:, :, ::3] = (
        False  # Only 2D turns of the distal finger chain; MCP positions have their own targets.
    )
    usable = ov & (ov.sum(1) >= 2)[:, None, :]
    signs = usable & sv & (abs(sa) >= 10) & (sa * oa > 0)
    setting = json.loads((folder / "mild" / (hand + "_palm.json")).read_text())
    axis = fit.tensor(
        Rotation.from_rotvec(old["global_orient"]).apply(setting["palm_axis_local"])
    )
    target_mm = setting["target_outward_mm"]
    q = fit.tensor(x["solution_q"])
    result = q.clone()
    initial = fit.tensor(oj) * 1000
    start_j = fit.tensor(x["joints"]) * 1000
    oturn = fit.tensor(np.nan_to_num(oa))
    vis = fit.tensor(usable)
    agreed = fit.tensor(signs)
    caps = fit.tensor(old["tip_cap_per_pose_mm"])
    e0 = fit.tensor(np.nan_to_num(old["marker_error_mm"][:, 3:]))
    eye = torch.eye(48, device=fit.dev)
    choice_scale = fit.tensor([1.0, 0.55, 0.27, 0.12, 0.04, 0.0])
    protected_count = 0
    for begin in range(0, len(q), 28):
        sl = slice(begin, min(len(q), begin + 28))
        first = q[sl]
        n = len(first)
        ori = initial[sl]
        mild = start_j[sl]
        mask = fit.valid[sl].bool()
        sref = fit.tip[sl]
        cap = caps[sl]
        old_f = flex_torch(ori)
        olda, oldb = old_f[:, :, 1], old_f[:, :, 2]
        ziglim = torch.maximum(
            torch.minimum(olda, -oldb), torch.minimum(-olda, oldb)
        ).clamp_min(9.8)
        lowflex = torch.minimum(old_f, fit.range_min + 0.02)
        max0 = e0[sl].amax(1).clamp_min(7.95)
        md0 = torch.nanquantile(
            e0[sl].masked_fill(~mask, torch.nan), 0.5, dim=1
        ).clamp_min(2.48)
        oldsurf = torch.as_tensor(old["pass_surface"][sl], device=fit.dev)
        # Global translation stays fixed, which pins J0 exactly even when the palm rotates.
        lower = torch.cat((fit.lo[:45].expand(n, -1), first[:, 45:48] - 3.5), 1)
        upper = torch.cat((fit.hi[:45].expand(n, -1), first[:, 45:48] + 3.5), 1)

        def values(param):
            k = param.shape[1]
            qq = torch.cat((param, first[:, None, 48:].expand(-1, k, -1)), dim=-1)
            j, m, _, _ = fit.kin.evaluate(qq.reshape(-1, 51))
            turn = project_turn(j).reshape(n, k, 4, 15) * 57.2957795
            dturn = ((turn - oturn[sl, None] + 180) % 360 - 180) * vis[sl, None]
            align = (
                F.relu(
                    torch.minimum(oturn[sl, None].abs(), turn.new_tensor(3.0))
                    + 0.2
                    - oturn[sl, None].sign() * turn
                )
                * agreed[sl, None]
            )
            ft = flex_torch(j).reshape(n, k, 5, 3)
            a, b = ft[:, :, :, 1], ft[:, :, :, 2]
            zig = torch.maximum(torch.minimum(a, -b), torch.minimum(-a, b))
            j = j.reshape(n, k, 21, 3)
            marker = (m.reshape(n, k, 21, 3) - fit.targets[sl, None]) * mask[
                :, None, :, None
            ]
            e = marker.norm(dim=-1)
            med = torch.nanquantile(
                e.masked_fill(~mask[:, None], torch.nan), 0.5, dim=2
            )
            mx = e.amax(2)
            tip = (j[:, :, TIPS] - sref[:, None]).norm(dim=-1)
            tc = (j[:, :, TIPS] - ori[:, None, TIPS]).norm(dim=-1)
            thumb = (j[:, :, 1:5] - ori[:, None, 1:5]).norm(dim=-1)
            lat = (
                (j[:, :, ROOTS] - ori[:, None, ROOTS]) * axis[sl, None, None, :]
            ).sum(-1)

            def flat(v):
                return v.reshape(n, k, -1)

            loss = torch.cat(
                [
                    flat(v)
                    for v in [
                        0.90 * (j[:, :, ROOTS] - mild[:, None, ROOTS]),
                        0.38 * (j[:, :, 1:] - ori[:, None, 1:]),
                        0.5 * marker,
                        2.5 * (j[:, :, TIPS] - ori[:, None, TIPS]),
                        4.0 * (j[:, :, 1:5] - ori[:, None, 1:5]),
                        12 * F.relu(tc - 0.65),
                        20 * F.relu(thumb - 0.43),
                        125 * F.relu(tip - cap[:, None, None] + 0.045),
                        22 * F.relu(med - md0[:, None]),
                        12 * F.relu(mx - max0[:, None]),
                        0.28 * dturn,
                        1.25 * F.relu(dturn.abs() - 2.2),
                        3.4 * align,
                        4 * F.relu(zig - ziglim[:, None]),
                        4 * F.relu(lowflex[:, None] - ft),
                        20 * F.relu(0.40 * target_mm - lat),
                        0.10 * (param[:, :, :45] - first[:, None, :45]),
                        0.6 * (param[:, :, 45:] - first[:, None, 45:48]),
                    ]
                ],
                dim=-1,
            )
            cap_ex = (tip.amax(2) - cap[:, None] + 0.001).clamp_min(0)
            surf_ex = (F.relu(med - 2.499) + 0.5 * F.relu(mx - 7.997)) * oldsurf[
                :, None
            ]
            root_ex = F.relu(0.30 * target_mm - lat.amin(2))
            rank = (
                1e7 * (cap_ex > 0)
                + 2e6 * cap_ex
                + 1e6 * (root_ex > 0)
                + 2e5 * root_ex
                + 3500 * (surf_ex > 0)
                + 1000 * surf_ex
                + 450 * F.relu(thumb.amax(2) - 0.5)
                + 350 * F.relu(tc.amax(2) - 0.85)
                + 60 * (align.amax((2, 3)))
                + 110 * F.relu(zig - ziglim[:, None]).amax(2)
                + 0.3 * F.relu(dturn.abs() - 3).square().sum((2, 3))
                + 0.04 * loss.square().sum(2)
            )
            return loss, rank

        p = first[:, :48].clone()
        best = p.clone()
        bestscore = values(p[:, None])[1][:, 0]
        for it in range(22):
            r, _ = values(torch.cat((p[:, None], p[:, None] + 0.055 * eye), dim=1))
            jac = (r[:, 1:] - r[:, :1]) / 0.055
            a = jac @ jac.transpose(1, 2) + (0.16 if it < 5 else 0.065) * eye
            rhs = -(jac @ r[:, 0, :, None]).squeeze(2)
            step = bounded_step(
                a, rhs, (lower - p).clamp_min(-3), (upper - p).clamp_max(3)
            )
            options = (p[:, None] + choice_scale[None, :, None] * step[:, None]).clamp(
                lower[:, None], upper[:, None]
            )
            rs, rank = values(options)
            batch = torch.arange(n, device=fit.dev)
            minimum, k = rank.min(1)
            keep = minimum < bestscore
            best[keep] = options[batch, k][keep]
            bestscore = torch.minimum(bestscore, minimum)
            k = rs.square().sum(2).argmin(1)
            moved = options[batch, k]
            change = (moved - p).abs().max().item()
            p = moved
            if change < 0.01:
                break
        result[sl, :48] = best
        protected_count += int(((first[:, :48] - best).abs().amax(1) > 0.005).sum())
    fit.store(
        "mcp_outward",
        result,
        np.arange(len(q)),
        x["tip_relaxed"],
        WEIGHTS[hand],
        source=src,
        destination=dest / (hand + ".npz"),
        mounts=dict(approved_for_training=np.array(False)),
    )
    print("refined", protected_count, flush=True)
