# Source work/mcp_outward_20261119/guard_mcp_angle.py, SHA-256 73c79696fb164d2f92a6c38b96513c0eb06310cf2fa571a5750eb2c9b02143b3

import case_settings as case

import os

from collections import defaultdict

import numpy as np

import torch

import torch.nn.functional as F

from fit_outward import HERE, ROOTS, base_path, bounded_step

from anatomy_common import TIPS, write_json, finger_prior, flex_torch

from fit_wide import WEIGHTS

from rounds import Fit, apply_mounts, load

@torch.no_grad()
def run(group, hand):
    archived = HERE / group / "before_mcp_angle_guard"
    primary = archived if archived.is_dir() else HERE / group / "final"
    src = primary / (hand + ".npz")
    dest = HERE / "angle_guard" / group / "final"
    dest.mkdir(parents=True, exist_ok=True)
    before = load(src)
    with np.load(base_path(group, hand)) as f:
        previous = f["joints"]
    fit = Fit(group, hand, bounds_folder=primary / "bounds")
    fit.axes = finger_prior.rest_axes(
        fit.model, fit.tensor(before["betas"]), fit.dev, include_thumb=True
    )
    apply_mounts(fit, before)
    joints = fit.tensor(before["joints"]) * 1000
    f0 = flex_torch(joints)
    prev = flex_torch(fit.tensor(previous) * 1000)
    flagged = (f0[:, 1:, 0] < -15) & (prev[:, 1:, 0] >= -15)
    panels = defaultdict(list)
    for i, row in enumerate(flagged.cpu().numpy()):
        if row.any():
            panels[tuple(np.flatnonzero(row))].append(i)
    q = fit.tensor(before["solution_q"])
    answer = q.clone()
    caps = fit.tensor(before["tip_cap_per_pose_mm"])
    scales = fit.tensor([1, 0.65, 0.35, 0.16, 0])
    accepted = 0
    for digits, indices in panels.items():
        active = np.concatenate([np.arange(9 * d, 9 * d + 9) for d in digits])
        nf = len(active)
        eye = torch.eye(nf, device=fit.dev)
        for ix in np.array_split(indices, 1 + (len(indices) - 1) // 32):
            n = len(ix)
            batch = torch.arange(n, device=fit.dev)
            initial = q[ix]
            x0 = initial[:, active]
            x = x0.clone()
            first, markers, _, _ = fit.kin.evaluate(initial)
            mask = fit.valid[ix].bool()
            target = fit.targets[ix]
            baseline = fit.tip[ix]
            dist0 = (first[:, TIPS] - baseline).norm(dim=2)
            limit = torch.maximum(dist0 + 0.0005, caps[ix, None] - 0.001)
            err0 = (markers - target).norm(dim=-1).masked_fill(~mask, torch.nan)
            boundmed = torch.nanquantile(err0, 0.5, dim=1).clamp_min(2.496) + 0.0002
            boundmax = err0.nan_to_num().amax(1).clamp_min(7.995) + 0.0002
            f_init = flex_torch(first)
            flag = flagged[ix]
            prior = prev[ix]
            lo = torch.maximum(x0 - 3.5, fit.lo[active])
            hi = torch.minimum(x0 + 3.5, fit.hi[active])

            def evaluate(values):
                k = values.shape[1]
                qq = initial[:, None].expand(-1, k, -1).clone()
                qq[:, :, active] = values
                j, m, _, _ = fit.kin.evaluate(qq.reshape(-1, 51))
                f = flex_torch(j).reshape(n, k, 5, 3)
                j = j.reshape(n, k, 21, 3)
                m = m.reshape(n, k, 21, 3)
                tips = (j[:, :, TIPS] - baseline[:, None]).norm(dim=3)
                err = (
                    (m - target[:, None])
                    .norm(dim=3)
                    .masked_fill(~mask[:, None], torch.nan)
                )
                med = torch.nanquantile(err, 0.5, dim=2)
                mx = err.nan_to_num().amax(2)
                deficit = F.relu(-14.94 - f[:, :, 1:, 0]) * flag[:, None]
                guard = F.relu(
                    torch.minimum(f_init, f_init.new_tensor(-14.998))[:, None] - f
                )
                d = (j - first[:, None]).norm(dim=-1)
                rs = [
                    8 * deficit,
                    40 * F.relu(tips - limit[:, None]),
                    18 * F.relu(med - boundmed[:, None]),
                    18 * F.relu(mx - boundmax[:, None]),
                    2 * guard,
                    0.3 * (f - f_init[:, None]),
                    1.3 * d[:, :, TIPS],
                    0.18 * d,
                    0.15 * (values - x0[:, None]),
                    0.35 * (m - markers[:, None]) * mask[:, None, :, None],
                ]
                loss = torch.cat([a.reshape(n, k, -1) for a in rs], 2)
                # Accept only improvements that retain eligible marker fits and the previous tip caps.
                flip = (f[:, :, 1:, 0] < -15) & (prior[:, None, 1:, 0] >= -15)
                zig = (f[:, :, :, 1] * f[:, :, :, 2] < 0) & (
                    f[:, :, :, 1:].abs().amin(3) > 10
                )
                zig0 = (f_init[:, :, 1] * f_init[:, :, 2] < 0) & (
                    f_init[:, :, 1:].abs().amin(2) > 10
                )
                legal = (
                    (tips <= limit[:, None] + 0.00065).all(2)
                    & (med <= boundmed[:, None] + 0.001)
                    & (mx <= boundmax[:, None] + 0.001)
                )
                legal &= (
                    (d[:, :, TIPS].amax(2) <= 0.48)
                    & (d.amax(2) < 1.1)
                    & ~(zig & ~zig0[:, None]).any(2)
                )
                legal &= ~(
                    (f[:, :, :, 1:] < -15) & (f_init[:, None, :, 1:] >= -15)
                ).any((2, 3))
                merit = (
                    flip.sum(2) * 10000
                    + deficit.sum(2) * 20
                    + loss.square().sum(2) * 0.01
                )
                return loss, torch.where(legal, merit, torch.inf)

            best = x0.clone()
            best_score = evaluate(x[:, None])[1][:, 0]
            for _ in range(18):
                r = evaluate(
                    torch.cat(
                        (
                            x[:, None],
                            x[:, None] + 0.055 * eye,
                            x[:, None] - 0.055 * eye,
                        ),
                        1,
                    )
                )[0]
                jac = (r[:, 1 : nf + 1] - r[:, nf + 1 :]) / 0.11
                gram = jac @ jac.transpose(1, 2) + 0.025 * eye
                step = bounded_step(
                    gram, -(jac @ r[:, 0, :, None]).squeeze(2), lo - x, hi - x
                )
                options = x[:, None] + step[:, None] * scales[None, :, None]
                loss, score = evaluate(options)
                vv, ii = score.min(1)
                best = torch.where((vv < best_score)[:, None], options[batch, ii], best)
                best_score = torch.minimum(best_score, vv)
                nextx = options[batch, loss.square().sum(2).argmin(1)]
                change = (nextx - x).abs().max().item()
                x = nextx
                if change < 0.002:
                    break
            answer[torch.as_tensor(ix, device=fit.dev)[:, None], active] = best
            accepted += int(((best - x0).abs().amax(1) > 0.0001).sum())
    output = dest / src.name
    if accepted:
        fit.store(
            "four_mcp_angle_guard",
            answer,
            np.arange(len(q)),
            before["tip_relaxed"],
            WEIGHTS[hand],
            source=src,
            destination=output,
            mounts=dict(approved_for_training=np.array(False)),
        )
    elif not output.exists():
        os.link(src, output)
    with np.load(output) as z:
        after = flex_torch(fit.tensor(z["joints"]) * 1000)
        remaining = (after[:, 1:, 0] < -15) & (prev[:, 1:, 0] >= -15)
        delta = np.linalg.norm(z["joints"] - before["joints"], axis=-1) * 1000
        cap = before["tip_cap_per_pose_mm"]
        over0 = before["tip_drift_mm"].max(1) > cap
        over = z["tip_drift_mm"].max(1) > cap
        loss = int((before["pass_surface"] & ~z["pass_surface"]).sum())
        new_caps = int((over & ~over0).sum())
        report = dict(
            events_before=int(flagged.sum()),
            events_after=int(remaining.sum()),
            modified_poses=accepted,
            root_max_mm=float(delta[:, ROOTS].max()),
            tip_max_mm=float(delta[:, TIPS].max()),
            new_cap_failures=new_caps,
            lost_surface=loss,
        )
        assert (
            report["root_max_mm"] < 0.001
            and not loss
            and not new_caps
            and int(remaining.sum()) <= int(flagged.sum())
        ), report
    write_json(dest / (hand + "_angle_guard.json"), report)
    print(group, hand, report, flush=True)
