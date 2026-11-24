# Source work/tip_pivot_20261119/lock_tips.py, SHA-256 aeebc5442db81caca96abdbb8f44b296df89797f86c9404992744e0ed47838b0

import case_settings as case

import json

import numpy as np

import torch

from fit_pivot import HANDS, MCP, OPT, TIPS, compose, pivot_targets, run_model

def visible_contours(group, hand, op):
    from preserve_visible_contours import FullMaskBatch, silhouette_per_frame
    lookup = {int(t): i for i, t in enumerate(op.times)}
    possible = json.loads((OPT[group] / "selection/selection.json").read_text())["records"]
    by_time = {}
    for r in possible:
        if r["side"] == HANDS.index(hand) and r["sync"] in lookup and r["accepted_quality"]:
            by_time.setdefault(r["sync"], []).append(r)
    used = []
    for views in by_time.values():
        views.sort(key=lambda r: r["camera"])
        used.extend(views[:-1] if len(views) > 1 else views)
    if not used:
        return lambda vertices: torch.ones(len(op.times), dtype=torch.bool, device=op.device)
    masks = FullMaskBatch(used, lookup, op.arr["joints"], op.arr["vertices"], op.arr["faces"], HANDS.index(hand),
                          192, str(op.device))
    with torch.no_grad():
        base = silhouette_per_frame(masks, op.base["vertices"])
    def compatible(vertices):
        current = silhouette_per_frame(masks, vertices)
        total = torch.zeros(len(op.times), device=op.device)
        total.scatter_add_(0, masks.frame, current - base)
        count = torch.bincount(masks.frame, minlength=len(op.times)).clamp_min(1)
        return total / count >= -.0003
    return compatible

def solve_tips(op, start, orientation, translation):
    angles = start.detach().clone()
    # Prefer proximal rotation and ordinary hinge flexion when choosing between equally good tip solutions.
    weights = torch.ones_like(angles)
    weights[:, :4, 1:, 0] = .12
    weights[:, :4, 1:, 2] = .16
    weights[:, :4, 0, 0] = .4
    eye = torch.eye(15, device=op.device, dtype=angles.dtype)
    lo, hi = op.limits
    for _ in range(10):
        angles.requires_grad_(True)
        _, j, _ = run_model(op.model, op.beta, orientation, compose(angles, op.model, op.axes), translation)
        residual = (j[:, TIPS] - op.joints[:, TIPS]).flatten(1) * 1000
        if float(residual.abs().max().detach()) < .002:
            return angles.detach()
        rows = [torch.autograd.grad(residual[:, k].sum(), angles, retain_graph=k < 14)[0].flatten(1)
                for k in range(15)]
        with torch.no_grad():
            jac = torch.stack(rows, dim=1)
            available = weights.flatten(1).clone()
            for _ in range(3):
                jac_t = available[:, :, None] * jac.transpose(1, 2)
                delta = -(jac_t @ torch.linalg.solve(jac @ jac_t + .0005 * eye, residual.detach()[:, :, None])).squeeze(2)
                limit = ((angles - lo < .006) & (delta.view_as(angles) < 0)) | (
                         (hi - angles < .006) & (delta.view_as(angles) > 0))
                available.masked_fill_(limit.flatten(1), 0.)
            delta = delta.view_as(angles)
            scale = 1. / delta.abs().amax((1, 2, 3), keepdim=True).clamp_min(1.)
            angles = (angles + delta * scale).clamp(lo + .002, hi - .002)
    return angles.detach()

def lock(group, hand, op, folder, source):
    with np.load(folder / (source + ".npz")) as original:
        target = {k: op.tensor(original[k]) for k in ("global_orient", "translation", "angles", "fraction")}
    base_hinge = op.angles[:, :4, 1:, [0, 2]].square().mean((1, 2, 3))
    da = target["angles"] - op.angles
    do = torch.rad2deg(target["global_orient"] - op.orient)
    dt = (target["translation"] - op.trans) * 1000
    chosen = torch.zeros_like(op.root_before)
    best_cost = torch.full_like(chosen, float("inf"))
    picked = op.angles.clone()
    measured = source != "geometry6"
    axial = source == "ulnar"
    silhouette_ok = visible_contours(group, hand, op) if axial else None
    _, outward = pivot_targets(op.joints, 0)
    scales = (1., .6, .3, .12) if measured else (1.,)
    damping = (1., .82, .64) if axial else (1.,)
    for scale, side_factor in ((s, f) for s in scales for f in damping):
        aim = op.angles + scale * da
        if side_factor != 1.:
            aim[:, :4, 1:, 0] *= side_factor
            aim[:, :4, 1:, 2] *= side_factor
        result = solve_tips(op, aim, op.orient + torch.deg2rad(do * scale), op.trans + dt * scale * .001)
        with torch.no_grad():
            a = op.forward(result, do * scale, dt * scale)
            good = ((a["joints"][:, TIPS] - op.joints[:, TIPS]) * 1000).norm(dim=-1).amax(1) <= .01
            if measured:
                within, _ = op.check(a)
                hinge = result[:, :4, 1:, [0, 2]].square().mean((1, 2, 3))
                good &= within & (hinge <= base_hinge + .001) & (target["fraction"] > 0)
                if axial:
                    direction = ((a["joints"] - op.joints) * outward[:, None]).sum(-1)
                    good &= (direction[:, [0, *MCP]] > 0).all(1) & silhouette_ok(a["vertices"])
            metric = (a["err"].square().masked_fill(~op.valid, 0.).sum(1) / op.valid.sum(1)).sqrt()
            metric = metric + .09 * result[:, :4, 1:, [0, 2]].square().mean((1, 2, 3)).sqrt()
            choose = good & ((chosen == 0) | ((chosen == scale) & (metric < best_cost)))
            chosen[choose] = scale
            picked[choose] = result[choose]
            best_cost[choose] = metric[choose]
            if not measured:
                # Keep all geometry-only stress-test frames, and report their actual tip residuals.
                chosen[:] = scale
                picked = result
    with torch.no_grad():
        a = op.forward(picked, do * chosen[:, None], dt * chosen[:, None])
        snap = op.snapshot(a, chosen * target["fraction"])
    name = {"supported": "anchored", "bounded3": "anchored3", "geometry6": "geometry6_locked",
            "ulnar": "palm_pivot"}[source]
    tmp = folder / (name + ".partial.npz")
    np.savez_compressed(tmp, times=op.times, betas=op.arr["betas"], marker_valid=op.arr["marker_valid"],
                        target24=op.arr["target24"], calibrated_root_p90_mm=op.arr["calibrated_root_p90"], **snap)
    tmp.replace(folder / (name + ".npz"))
    print(group, hand, name, "samples", len(chosen), "modified", int((chosen > 0).sum()),
          "max_tip_mm", round(float(snap["tip_mm"].max()), 4), flush=True)
