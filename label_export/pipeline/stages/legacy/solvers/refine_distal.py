# Source work/direction_acceptance_20261119/refine_distal.py, SHA-256 78e1397d36b356963e22ae9f4f6dbe3f3cfe250c46b8e616225e4f64e26573a7

import case_settings as case

import numpy as np

import torch

import torch.nn.functional as F

from fit_wide import HANDS, ROOT, WEIGHTS, source_track

from rounds import Fit, apply_mounts, load

from anatomy_common import OPT, TIPS, flex_torch, geometry, physical

from observed import TRIPLES, measure, sdk_in_fit_space, uv_all

class CameraTurns:
    def __init__(self, fit, cams, rt):
        t = fit.tensor
        self.mirror = t([1 if fit.hand == "right" else -1, 1, 1])
        self.rot = t(np.stack([x[0] for x in rt]))
        self.trans = t(np.stack([x[1] for x in rt])) * 1000
        self.fx = t(np.array([[c.K[0, 0], c.K[1, 1]] for c in cams]))
        self.center = t(np.array([[c.K[0, 2], c.K[1, 2]] for c in cams]))
        self.dist = t(np.stack([c.distortion.reshape(-1) for c in cams]))
        self.triples = torch.tensor(TRIPLES, device=fit.dev)

    def __call__(self, j):
        cam = (
            torch.einsum("nji,cki->ncjk", j * self.mirror, self.rot)
            + self.trans[None, :, None, :]
        )
        xy = cam[..., :2] / cam[..., 2:].clamp_min(1e-4)
        r = xy.norm(dim=-1).clamp_min(1e-9)
        theta = r.atan()
        t2 = theta.square()
        a, b, c, d = self.dist.T[:, :, None]
        rad = theta * (1 + t2 * (a + t2 * (b + t2 * (c + t2 * d))))
        uv = (xy * (rad / r)[..., None]) * self.fx[None, :, None] + self.center[
            None, :, None
        ]
        v = uv[:, :, self.triples]
        u = v[:, :, :, 1] - v[:, :, :, 0]
        v = v[:, :, :, 2] - v[:, :, :, 1]
        cross = u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]
        dot = (u * v).sum(-1)
        return torch.atan2(cross, dot + 1e-7)

def solve(group, hand, example_only=False, preserve_anatomy=False):
    dst = ROOT / group / ("distal_anatomy" if preserve_anatomy else "distal")
    dst.mkdir(parents=True, exist_ok=True)
    source = source_track(group, hand)
    z = load(source)
    fit = Fit(group, hand, bounds_folder=dst / "bounds")
    apply_mounts(fit, z)
    _, cams, rt, _ = geometry(group)
    proj = CameraTurns(fit, cams, rt)
    with np.load(OPT[group] / "skeleton/aligned_skeleton.npz") as s:
        sdk = sdk_in_fit_space(
            group,
            HANDS.index(hand),
            s["positions_cam0_m"][:, :, s["provisional21_indices"]].astype(float),
            s["marker24_cam0_m"].astype(float),
            fit.ts,
        )
    truth, mu = measure(uv_all(sdk, cams, rt), cams)
    base = z["joints"]
    original, vis = measure(uv_all(physical(base.astype(float), hand), cams, rt), cams)
    good = mu & vis
    good[:, :, ::3] = False
    good &= (good.sum(1) >= 2)[:, None, :]
    # The new target contains NOKOV angles only. WiLoR observations are never passed into the fit.
    truth = fit.tensor(np.deg2rad(np.nan_to_num(truth)))
    eligible = fit.tensor(good)
    start = fit.tensor(z["solution_q"])
    change = fit.tensor(np.tile(np.r_[np.zeros(3), np.ones(6)], 5).tolist() + [0.0] * 6)
    cap = fit.tensor(z["tip_cap_per_pose_mm"])
    isolated = z["isolated_pose"]
    t = fit.ts
    if example_only:
        targets = case.config.get('review_syncs', [])
        ii = np.flatnonzero(np.min(np.abs(t[:, None] - targets), axis=1) <= 6)
    else:
        ii = np.arange(len(t))
    result = start.clone()
    fractions = np.zeros(len(t), np.float32)
    with torch.no_grad():
        compare = (
            torch.rad2deg(proj(fit.kin.evaluate(start[ii[:12]], False))).cpu().numpy()
        )
        error = np.abs((compare - original[ii[:12]] + 180) % 360 - 180)
        assert np.nanmax(np.where(good[ii[:12]], error, 0)) < 0.012
    for lo in range(0, len(ii), 160):
        sub = ii[max(lo - 3, 0) : lo + 163]
        initial = start[sub]
        targets = truth[sub]
        ok = eligible[sub]
        observed = fit.valid[sub].bool()
        surface = fit.targets[sub]
        tip = fit.tip[sub]
        budget = cap[sub]
        with torch.no_grad():
            j0, m0, p0, _ = fit.kin.evaluate(initial)
            e0 = (m0 - surface).norm(dim=-1)
            f0 = flex_torch(j0)
        diff = torch.nn.Parameter(torch.zeros_like(initial))
        opt = torch.optim.Adam([diff], lr=0.75)
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 95, eta_min=0.04)
        links = np.flatnonzero(
            (np.diff(t[sub]) == 1)
            & (np.diff(fit.stamps[sub]) > 0)
            & (np.diff(fit.stamps[sub]) < fit.gap)
        )

        def terms(q):
            j, m, p, _ = fit.kin.evaluate(q)
            dt = proj(j) - targets
            dir_loss = ((2 * torch.sin(dt / 2) * 57.29578).square() * ok).sum(
                (1, 2)
            ) / ok.sum((1, 2)).clamp_min(1)
            tipd = (j[:, TIPS] - tip).norm(dim=-1)
            e = (m - surface).norm(dim=-1)
            f = flex_torch(j)
            return dir_loss, tipd, e, f, p

        for _ in range(95):
            opt.zero_grad(set_to_none=True)
            q = (initial + diff * change).clamp(fit.lo, fit.hi)
            direction, tipd, err, flex, pose = terms(q)
            sq = err.square() * observed
            cost = (
                (0.8 if preserve_anatomy else 0.22) * direction.mean()
                + sq.sum() / observed.sum()
                + 0.18 * sq.topk(2, dim=1).values.mean()
            )
            cost += (
                12 * F.relu(tipd - budget[:, None] + 0.15).square().mean()
                + 0.15 * tipd.square().mean()
            )
            cost += (
                0.012 * (q - initial).square().mean()
                + WEIGHTS[hand] * ((pose - fit.mean) @ fit.white).square().mean()
            )
            cost += 1.2 * F.relu(fit.range_min - flex).square().sum(2).mean()
            if preserve_anatomy:
                a, b = flex[:, :, 1], flex[:, :, 2]
                zig = F.relu(torch.minimum(a, -b) - 9.8) + F.relu(
                    torch.minimum(-a, b) - 9.8
                )
                cost += 2.0 * zig.square().sum(1).mean()
            if len(links):
                cost += (
                    0.055
                    * ((q - initial)[links + 1] - (q - initial)[links]).square().mean()
                )
            cost.backward()
            opt.step()
            schedule.step()
        with torch.no_grad():
            final = (initial + diff * change).clamp(fit.lo, fit.hi)
            best = initial.clone()
            best_dir = terms(initial)[0]
            alpha = np.zeros(len(sub), np.float32)
            med0 = torch.nanquantile(e0.masked_fill(~observed, torch.nan), 0.5, dim=1)
            max0 = e0.masked_fill(~observed, 0).amax(1)

            def retain(candidate, fraction):
                nonlocal best, best_dir
                direction, tipd, e, flex, _ = terms(candidate)
                med = torch.nanquantile(e.masked_fill(~observed, torch.nan), 0.5, dim=1)
                maxerr = e.masked_fill(~observed, 0).amax(1)
                legal = (
                    (tipd.amax(1) <= budget - 0.0001)
                    & (med <= torch.maximum(med0, med0.new_tensor(2.4998)))
                    & (maxerr <= torch.maximum(max0, max0.new_tensor(7.9998)))
                )
                legal &= ((flex + 0.12) >= torch.minimum(f0, fit.range_min)).all((1, 2))
                if preserve_anatomy:
                    a, b = flex[:, :, 1], flex[:, :, 2]
                    x, y = f0[:, :, 1], f0[:, :, 2]
                    new_zig = (a * b < 0) & (torch.minimum(a.abs(), b.abs()) > 9.99)
                    old_zig = (x * y < 0) & (torch.minimum(x.abs(), y.abs()) > 10)
                    legal &= ~((new_zig & ~old_zig).any(1))
                    new_hyper = (flex < -14.99) & (f0 >= -15) & (flex < f0 - 0.002)
                    new_hyper[:, 0, 0] = False
                    legal &= ~new_hyper.any((1, 2))
                use = (
                    legal
                    & (direction < best_dir - 0.01)
                    & ~torch.as_tensor(isolated[sub], device=fit.dev)
                )
                best = torch.where(use[:, None], candidate, best)
                best_dir = torch.where(use, direction, best_dir)
                take = use.cpu().numpy()
                alpha[take] = np.maximum(alpha[take], fraction)

            for a in (1.0, 0.85, 0.65, 0.45, 0.3, 0.18, 0.08):
                retain(initial + (final - initial) * a, a)
            # A different finger can be at its tip limit. Test each phalanx chain
            # separately against the same whole-hand constraints before discarding its update.
            for digit in range(5):
                sl = slice(digit * 9 + 3, digit * 9 + 9)
                for a in (1.0, 0.70, 0.40, 0.18):
                    proposal = best.clone()
                    proposal[:, sl] = (
                        initial[:, sl] + (final[:, sl] - initial[:, sl]) * a
                    )
                    retain(proposal, a)
            at = lo - max(lo - 3, 0)
            size = min(160, len(ii) - lo)
            out = ii[lo : lo + size]
            result[out] = best[at : at + size]
            fractions[out] = alpha[at : at + size]
    np.testing.assert_equal(
        result[:, change == 0].cpu().numpy(), start[:, change == 0].cpu().numpy()
    )
    if example_only:
        with torch.no_grad():
            j, m, *_ = fit.kin.evaluate(result[ii])
        np.savez_compressed(
            dst / (hand + "_examples.npz"),
            times=t[ii],
            joints=j.cpu().numpy() / 1000,
            mounts=m.cpu().numpy() / 1000,
            alpha=fractions[ii],
        )
        return
    fields = {
        k: z[k]
        for k in (
            "mount_calibration_sync",
            "cuff_local_calibration_sync",
            "cuff_reference_calibration_sync",
            "isolated_pose",
        )
    }
    fields["sdk_distal_step_fraction"] = fractions
    fit.store(
        "sdk_distal",
        result,
        ii,
        z["tip_relaxed"],
        WEIGHTS[hand],
        source=source,
        destination=dst / (hand + ".npz"),
        mounts=fields,
    )
    print("changed", int((fractions > 0).sum()), "/", len(fractions), flush=True)
