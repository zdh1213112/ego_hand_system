# Source work/mcp_outward_20261119/batch_constraints.py, SHA-256 0fe3e60bf6d5d35a9e58fdd0bce56524f13e79f7f443b732740ca3286abd4b98

import case_settings as case

import json

import queue

import threading

import time

from collections import defaultdict

from concurrent.futures import ThreadPoolExecutor

from functools import lru_cache

import numpy as np

import torch

from scipy.optimize import minimize

from scipy.spatial.transform import Rotation

from constraints import palm

from fit_outward import HERE, ROOTS, base_path

from fit_wide import WEIGHTS

from anatomy_common import TIPS, geometry, physical, finger_prior, flex_torch, write_json

from rounds import Fit, apply_mounts, load

from observed import measure, uv_all

from refine_distal import CameraTurns

@torch.no_grad()
def run(group, h, check_only=False, tips_only=False, *, study=HERE):
    dest = study / group / ("final" if tips_only else "outward")
    dest.mkdir(exist_ok=True)
    src = (dest if tips_only else study / group / "selected") / (h + ".npz")
    a = load(src)
    b = load(base_path(group, h))
    fit = Fit(group, h, bounds_folder=dest / "bounds")
    fit.axes = finger_prior.rest_axes(
        fit.model, fit.tensor(a["betas"]), fit.dev, include_thumb=True
    )
    apply_mounts(fit, a)
    q = fit.tensor(a["solution_q"])
    result = q.clone()
    jo = fit.tensor(b["joints"]) * 1000
    jn = fit.tensor(a["joints"]) * 1000
    olddir = palm(jo)
    newdir = palm(jn)
    cap = fit.tensor(a["tip_cap_per_pose_mm"])
    gap = (jn[:, TIPS] - fit.tip).norm(dim=-1) - cap[:, None]
    tipmove = (jn[:, TIPS] - jo[:, TIPS]).norm(dim=-1)
    thumb = (jn[:, 1:5] - jo[:, 1:5]).norm(dim=-1).amax(1)
    chosen = (gap > -0.001) | (tipmove > 1.02)
    chosen[:, 0] |= thumb > 0.7
    chosen[:, 1:] |= (olddir * newdir <= 0) & (olddir.abs() > 0.01)
    if tips_only:
        with np.load(study / group / "outward" / (h + ".npz")) as prev:
            surface_floor = prev["pass_surface"]
            center = fit.tensor(prev["joints"]) * 1000
            surface_errors = fit.tensor(prev["marker_error_mm"][:, 3:])
        new_lost = surface_floor & ~a["pass_surface"]
        excess = a["tip_drift_mm"].max(1) > a["tip_cap_per_pose_mm"]
        # A small change of palm orientation may require compensation even in digits still inside their caps.
        chosen = (gap > -0.9) | (tipmove > 0.85)
        chosen[:, 0] = True
        chosen &= torch.as_tensor(new_lost | excess, device=fit.dev)[:, None]
    ids_by_digits = defaultdict(list)
    for i, flags in enumerate(chosen.cpu().numpy()):
        if flags.any():
            ids_by_digits[tuple(np.flatnonzero(flags))].append(i)
    _, cams, rt, _ = geometry(group)
    turn = CameraTurns(fit, cams, rt)
    ang, visible = measure(
        uv_all(physical(b["joints"].astype(float), h), cams, rt), cams
    )
    visible[:, :, ::3] = False
    has = fit.tensor(visible & (np.abs(ang) >= 3)).bool()
    ot = fit.tensor(np.nan_to_num(ang))
    oldf = flex_torch(jo)
    a0, b0 = oldf[:, :, 1], oldf[:, :, 2]
    ziglim = torch.maximum(torch.minimum(a0, -b0), torch.minimum(-a0, b0)).clamp_min(
        9.98
    )
    lowf = torch.minimum(oldf - 0.025, fit.range_min - 0.025)
    raw_e = fit.tensor(a["marker_error_mm"][:, 3:])
    med0 = (torch.nanquantile(raw_e, 0.5, dim=1) + 0.008).clamp_min(2.495)
    max0 = (raw_e.nan_to_num().amax(1) + 0.015).clamp_min(7.992)
    if tips_only:
        med0 = torch.where(
            torch.as_tensor(surface_floor, device=fit.dev),
            2.495,
            torch.nanquantile(surface_errors, 0.5, dim=1).clamp_min(2.495) + 0.025,
        )
        max0 = torch.where(
            torch.as_tensor(surface_floor, device=fit.dev),
            7.992,
            surface_errors.nan_to_num().amax(1).clamp_min(7.992) + 0.04,
        )
    setting = json.loads((study / group / "mild" / (h + "_palm.json")).read_text())
    axis = fit.tensor(
        Rotation.from_rotvec(b["global_orient"]).apply(setting["palm_axis_local"])
    )
    lo = fit.lo[:45].cpu().numpy()
    hi = fit.hi[:45].cpu().numpy()
    results = {}

    def solve_panel(indices, active):
        jobs = queue.Queue()
        count = len(active)
        eps = 0.045
        dirs = eps * torch.eye(count, device=fit.dev)

        def features(ii, x, relaxed):
            n, k, _ = x.shape
            qq = q[ii, None].expand(-1, k, -1).clone()
            qq[:, :, active] = x
            j, m, _, _ = fit.kin.evaluate(qq.reshape(-1, 51))
            pp = palm(j).reshape(n, k, 4)
            ft = flex_torch(j).reshape(n, k, 5, 3)
            aa, bb = ft[:, :, :, 1], ft[:, :, :, 2]
            zig = torch.maximum(torch.minimum(aa, -bb), torch.minimum(-aa, bb))
            ang = turn(j).reshape(n, k, 4, 15) * 57.2957795
            j = j.reshape(n, k, 21, 3)
            da = ((ang - ot[ii, None] + 180) % 360 - 180) * has[ii, None]
            er = (
                (m.reshape(n, k, 21, 3) - fit.targets[ii, None])
                .norm(dim=-1)
                .masked_fill(~fit.valid[ii, None].bool(), torch.nan)
            )
            tip = (j[:, :, TIPS] - fit.tip[ii, None]).norm(dim=-1)
            dr = (j[:, :, TIPS] - jo[ii, None][:, :, TIPS]).norm(dim=-1)
            th = (j[:, :, 1:5] - jo[ii, None][:, :, 1:5]).norm(dim=-1)
            lat = (
                (j[:, :, ROOTS] - jo[ii, None][:, :, ROOTS]) * axis[ii, None, None]
            ).sum(-1)
            score = (
                0.55
                * (j[:, :, ROOTS] - jn[ii, None][:, :, ROOTS]).square().sum(-1).mean(2)
                + 0.06 * (qq[:, :, :48] - q[ii, None, :48]).square().sum(2)
                + 0.1 * (j - jn[ii, None]).square().sum(-1).mean(2)
                + 0.14 * er.square().nanmean(2)
                + 0.75 * dr.square().mean(2)
                + 1.1 * th.square().mean(2)
                + 0.018
                * da.square().sum((2, 3))
                / has[ii].sum((1, 2)).clamp_min(1)[:, None]
            )
            vl = relaxed[:, None, None]
            parts = [
                cap[ii, None, None] - 0.010 - tip,
                1.0 + 0.5 * vl - dr,
                0.68 + 0.19 * vl - th,
                lat - (0.28 - 0.11 * vl) * setting["target_outward_mm"],
                pp * olddir[ii, None].sign()
                - olddir[ii, None].abs().mul(0.5).clamp_max(0.25),
                med0[ii, None] + 0.10 * vl[:, :, 0] - torch.nanquantile(er, 0.5, dim=2),
                max0[ii, None] + 0.2 * vl[:, :, 0] - er.nan_to_num().amax(2),
                ft - lowf[ii, None],
                ziglim[ii, None] - zig,
                torch.where(has[ii, None], ang * ot[ii, None].sign() + 0.02, 1.0),
            ]
            if tips_only:
                root_change = (j[:, :, ROOTS] - center[ii, None][:, :, ROOTS]).norm(
                    dim=-1
                )
                keep_angle = torch.where(
                    has[ii, None] & (ot[ii, None].abs() >= 10),
                    ang * ot[ii, None].sign() + 0.10,
                    1.0,
                )
                parts = [
                    cap[ii, None, None] - 0.008 - tip,
                    0.65 + 0.40 * vl - root_change,
                    lat - 0.18 * setting["target_outward_mm"],
                    1.85 + 0.30 * vl - dr,
                    1.30 + 0.3 * vl - th,
                    med0[ii, None] - torch.nanquantile(er, 0.5, dim=2),
                    max0[ii, None] - er.nan_to_num().amax(2),
                    ft - lowf[ii, None],
                    ziglim[ii, None] - zig,
                    keep_angle,
                ]
            return torch.cat(
                [score.unsqueeze(2), *[v.reshape(n, k, -1) for v in parts]], dim=2
            )

        def worker(i):
            seed = q[i].cpu().numpy()
            x0 = seed[active].astype(float)
            l = np.r_[lo, seed[45:48] - 3.3][active]
            u = np.r_[hi, seed[45:48] + 3.3][active]
            for relaxed in (False, True):
                best = [np.inf, None]

                @lru_cache(maxsize=20)
                def state(key):
                    event = threading.Event()
                    response = []
                    jobs.put((i, np.asarray(key), relaxed, event, response))
                    event.wait()
                    val, jac = response[0]
                    if val[1:].min() >= -0.0015 and val[0] < best[0]:
                        best[:] = [val[0], np.array(key)]
                    return val, jac

                out = minimize(
                    lambda p: state(tuple(p))[0][0],
                    x0,
                    jac=lambda p: state(tuple(p))[1][0],
                    method="SLSQP",
                    bounds=list(zip(l, u)),
                    constraints=[
                        {
                            "type": "ineq",
                            "fun": lambda p: state(tuple(p))[0][1:],
                            "jac": lambda p: state(tuple(p))[1][1:],
                        }
                    ],
                    options={"maxiter": 32 if not relaxed else 44, "ftol": 1e-5},
                )
                state(tuple(out.x))
                if best[1] is not None:
                    return best[1], relaxed
                x0 = np.clip(out.x, l, u)
            return None, None

        with ThreadPoolExecutor(max_workers=len(indices)) as pool:
            futures = {i: pool.submit(worker, int(i)) for i in indices}
            while not all(f.done() for f in futures.values()):
                try:
                    items = [jobs.get(timeout=0.01)]
                except queue.Empty:
                    continue
                deadline = time.monotonic() + 0.0012
                while len(items) < len(indices):
                    try:
                        items.append(
                            jobs.get(timeout=max(0, deadline - time.monotonic()))
                        )
                    except queue.Empty:
                        break
                ii = np.array([r[0] for r in items])
                xx = fit.tensor(np.stack([r[1] for r in items]))
                vals = (
                    features(
                        ii,
                        torch.cat(
                            (xx[:, None], xx[:, None] + dirs, xx[:, None] - dirs), 1
                        ),
                        fit.tensor([r[2] for r in items]),
                    )
                    .cpu()
                    .numpy()
                    .astype(float)
                )
                jac = (
                    (vals[:, 1 : count + 1] - vals[:, count + 1 :]) / (2 * eps)
                ).transpose(0, 2, 1)
                for k, (i, _, _, event, response) in enumerate(items):
                    # Constant constraints for unobserved camera/joint pairs do not enter the QP.
                    keep = np.r_[np.ones(vals.shape[2] - 60, bool), visible_mask[i]]
                    response.append((vals[k, 0, keep], jac[k][keep]))
                    event.set()
            for i, f in futures.items():
                sol, relaxed = f.result()
                if sol is not None:
                    result[i, active] = fit.tensor(sol)
                results[str(int(fit.ts[i]))] = dict(
                    solved=sol is not None, relaxed=relaxed
                )

    visible_mask = has.cpu().numpy().reshape(len(q), -1)
    count = sum(map(len, ids_by_digits.values()))
    for digits, indices in sorted(ids_by_digits.items(), key=lambda r: -len(r[1])):
        active = np.r_[
            np.concatenate(
                [np.arange(9 * ((d + 4) % 5), 9 * ((d + 4) % 5 + 1)) for d in digits]
            ),
            45,
            46,
            47,
        ].astype(int)
        for panel in np.array_split(indices, max(1, int(np.ceil(len(indices) / 28)))):
            solve_panel(panel[:2] if check_only else panel, active)
            print("processed", len(results), "/", count, flush=True)
            if check_only:
                return
    destination = dest / (("." + h + "_cap.npz") if tips_only else h + ".npz")
    fit.store(
        "mcp_outward_final",
        result,
        np.arange(len(q)),
        a["tip_relaxed"],
        WEIGHTS[h],
        source=src,
        destination=destination,
        mounts=dict(approved_for_training=np.array(False)),
    )
    if tips_only:
        with np.load(destination) as out:
            cap_left = int(
                (out["tip_drift_mm"].max(1) > out["tip_cap_per_pose_mm"]).sum()
            )
            restored = int(
                (surface_floor & out["pass_surface"] & ~a["pass_surface"]).sum()
            )
            print("remaining", cap_left, "recovered surface", restored, flush=True)
        destination.replace(src)
    write_json(
        dest / (h + ("_cap_guard.json" if tips_only else "_guard.json")),
        dict(
            considered=count,
            solved=sum(x["solved"] for x in results.values()),
            poses=results,
        ),
    )
    print(
        "checked",
        count,
        "solved",
        sum(x["solved"] for x in results.values()),
        flush=True,
    )
