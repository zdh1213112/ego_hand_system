# Source work/mcp_outward_20261119/tip_guard.py, SHA-256 c1ab15e5097028fa4ab75130d3f4cad93c34bf64e77fbef37f6718560ac0b5b4

import case_settings as case

import os

import numpy as np

import torch

import torch.nn.functional as F

from fit_outward import HERE, ROOTS, base_path, bounded_step

from anatomy_common import TIPS, finger_prior, flex_torch, write_json

from fit_wide import WEIGHTS

from rounds import Fit, apply_mounts, load

@torch.no_grad()
def run(group, hand, palm=False, *, study=HERE):
    folder = study / group
    dest = folder / "final"
    dest.mkdir(exist_ok=True)
    src = (dest if palm else folder / "outward") / (hand + ".npz")
    out = dest / (("." + hand + "_palm.npz") if palm else src.name)
    before = load(src)
    reference = before["tip_reference_m"].astype(float)
    cap = before["tip_cap_per_pose_mm"]

    def excess(joints):
        return (
            np.linalg.norm(joints[:, TIPS].astype(float) - reference, axis=2) * 1000
            - cap[:, None]
        )

    first_excess = excess(before["joints"])
    ids = np.flatnonzero(first_excess.max(1) > 0)
    if len(ids):
        fit = Fit(group, hand, bounds_folder=folder / "outward/bounds")
        fit.axes = finger_prior.rest_axes(
            fit.model, fit.tensor(before["betas"]), fit.dev, include_thumb=True
        )
        apply_mounts(fit, before)
        q = fit.tensor(before["solution_q"])
        result = q.clone()
        with np.load(base_path(group, hand)) as base:
            old = fit.tensor(base["joints"]) * 1000
        dims = 48 if palm else 45
        I = torch.eye(dims, device=fit.dev)
        scales = fit.tensor([1, 0.5, 0.25, 0.125, 0])
        for ix in np.array_split(ids, 1 + (len(ids) - 1) // 24):
            first = q[ix]
            x = first[:, :dims].clone()
            n = len(first)
            batch = torch.arange(n, device=fit.dev)
            initial, marker, _, _ = fit.kin.evaluate(first)
            ref = fit.tensor(reference[ix]) * 1000
            caps = fit.tensor(cap[ix])
            orig = old[ix]
            rel = initial[:, TIPS] - ref
            dist = rel.norm(dim=2)
            target = (
                ref
                + rel
                * ((caps[:, None] - 0.015) / dist.clamp_min(0.001)).clamp_max(1)[
                    :, :, None
                ]
            )
            mask = fit.valid[ix].bool()
            measured = fit.targets[ix]
            orig_flex = flex_torch(initial)
            err0 = (marker - measured).norm(dim=-1).masked_fill(~mask, torch.nan)
            med0 = torch.nanquantile(err0, 0.5, dim=1).clamp_min(2.493)
            max0 = torch.nan_to_num(err0).amax(1).clamp_min(7.992)
            lo = torch.maximum(fit.lo[:45], x[:, :45] - 5)
            hi = torch.minimum(fit.hi[:45], x[:, :45] + 5)
            if palm:
                lo = torch.cat((lo, x[:, 45:48] - 1.35), dim=1)
                hi = torch.cat((hi, x[:, 45:48] + 1.35), dim=1)

            def residual(angles):
                k = angles.shape[1]
                full = torch.cat(
                    (angles, first[:, None, dims:].expand(-1, k, -1)), dim=2
                )
                j, m, _, _ = fit.kin.evaluate(full.reshape(-1, 51))
                flex = flex_torch(j).reshape(n, k, 5, 3)
                j = j.reshape(n, k, 21, 3)
                m = m.reshape(n, k, 21, 3)
                err = (m - measured[:, None]).norm(dim=-1)
                med = torch.nanquantile(
                    err.masked_fill(~mask[:, None], torch.nan), 0.5, dim=2
                )
                mx = err.masked_fill(~mask[:, None], 0).amax(2)
                over = (
                    (j[:, :, TIPS] - ref[:, None]).norm(dim=-1)
                    - caps[:, None, None]
                    + 0.005
                )
                tip_dist = (j[:, :, TIPS] - orig[:, None, TIPS]).norm(dim=-1)
                thumb_dist = (j[:, :, 1:5] - orig[:, None, 1:5]).norm(dim=-1)
                upper_tip = (
                    (initial[:, TIPS] - orig[:, TIPS]).norm(dim=-1).clamp_min(0.99)
                )
                upper_thumb = (
                    (initial[:, 1:5] - orig[:, 1:5]).norm(dim=-1).clamp_min(0.69)
                )
                values = [
                    18 * (j[:, :, TIPS] - target[:, None]),
                    55 * F.relu(over),
                    0.36 * (j - initial[:, None]),
                    0.38 * (m - marker[:, None]) * mask[:, None, :, None],
                    0.13 * (angles - first[:, None, :dims]),
                    0.17 * (flex - orig_flex[:, None]),
                    7 * F.relu(med - med0[:, None]),
                    7 * F.relu(mx - max0[:, None]),
                    1.5 * F.relu(tip_dist - upper_tip[:, None]),
                    1.5 * F.relu(thumb_dist - upper_thumb[:, None]),
                ]
                if palm:
                    mcp = (j[:, :, ROOTS] - initial[:, None, ROOTS]).norm(dim=-1)
                    values.extend((0.6 * mcp, 35 * F.relu(mcp - 0.48)))
                return torch.cat([a.reshape(n, k, -1) for a in values], dim=2)

            for _ in range(18):
                f = residual(
                    torch.cat(
                        (x[:, None], x[:, None] + 0.06 * I, x[:, None] - 0.06 * I),
                        dim=1,
                    )
                )
                jac = (f[:, 1 : dims + 1] - f[:, dims + 1 :]) / 0.12
                gram = jac @ jac.transpose(1, 2) + 0.02 * I
                rhs = -(jac @ f[:, 0, :, None]).squeeze(2)
                step = bounded_step(gram, rhs, lo - x, hi - x)
                options = x[:, None] + scales[None, :, None] * step[:, None]
                pick = residual(options).square().sum(2).argmin(1)
                chosen = options[batch, pick]
                change = (chosen - x).abs().max().item()
                x = chosen
                if change < 0.002:
                    break
            result[ix, :dims] = x
        fit.store(
            "four_mcp_tip_guard",
            result,
            np.arange(len(q)),
            before["tip_relaxed"],
            WEIGHTS[hand],
            source=src,
            destination=out,
            mounts=dict(approved_for_training=np.array(False)),
        )
    elif not out.exists():
        os.link(src, out)
    with np.load(out) as z:
        last_excess = excess(z["joints"])
        change = np.linalg.norm(z["joints"] - before["joints"], axis=-1) * 1000
        report = dict(
            considered=len(ids),
            remaining=int((last_excess.max(1) > 0).sum()),
            worst_mm=max(0.0, float(last_excess.max())),
            mcp_change_max_mm=float(change[:, ROOTS].max()),
            additional_lost_surface=int(
                (before["pass_surface"] & ~z["pass_surface"]).sum()
            ),
            additional_regained_surface=int(
                (~before["pass_surface"] & z["pass_surface"]).sum()
            ),
        )
        assert report["mcp_change_max_mm"] < (0.65 if palm else 0.001)
        if len(ids):
            untouched = first_excess.max(1) <= 0
            np.testing.assert_array_equal(
                z["solution_q"][untouched], before["solution_q"][untouched]
            )
            np.testing.assert_array_equal(
                z["solution_q"][:, dims:], before["solution_q"][:, dims:]
            )
    if palm:
        assert report["remaining"] <= len(ids)
        out.replace(src)
    write_json(
        dest / (hand + ("_palm_guard.json" if palm else "_tip_guard.json")), report
    )
    print(group, hand, report, flush=True)
