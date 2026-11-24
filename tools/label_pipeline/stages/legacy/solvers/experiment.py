# Source work/group_010_anatomy_20261119/joint_anatomy_20261119/experiment.py, SHA-256 c253503d5f9c4bbcf40e861d5926d857b967c083c21a21c9be27b8973675843f

import case_settings as case

import json

from scipy.spatial.transform import Rotation

import torch.nn.functional as F

import numpy as np

import torch

from anatomy_common import AXES, HANDS, REFERENCE, ROOT, TIPS, checkpoint, components, compose, finger_prior, flex_torch, gate_and_errors, geometry, keys, limits, make_model, paths, percentiles, run_model, save, write_json

from fit_cuff import marker_positions

from assess import held_times

class CompactHand:
    """Exact LBS at the 21 mount triangles, all joints and all five fingertip vertices."""

    def __init__(self, model, beta, axes, z):
        from mano.lbs import blend_shapes

        self.model = model
        self.axes = axes
        dev = beta.device
        tri = z["faces"][z["attachment_faces"]].astype(int)
        vids = np.unique(np.r_[tri.ravel(), list(model.tip_ids.values())])
        self.tri = torch.tensor(np.searchsorted(vids, tri), device=dev)
        self.tips = torch.tensor(np.searchsorted(vids, list(model.tip_ids.values())), device=dev)
        shaped = model.v_template + blend_shapes(beta[None], model.shapedirs)[0]
        self.base = shaped[vids]
        self.rest = model.J_regressor @ shaped
        self.ps = model.posedirs.reshape(135, -1, 3)[:, vids].reshape(135, -1)
        self.weights = model.lbs_weights[vids]
        self.eye = torch.eye(3, device=dev)
        self.bar = torch.as_tensor(z["barycentric"], device=dev)
        self.offset = torch.as_tensor(z["attachment_local_offsets_m"], device=dev) * 1000
        self.order = torch.tensor(
            [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20], device=dev
        )

    def evaluate(self, q, mounts=True):
        from mano.lbs import batch_rodrigues, batch_rigid_transform

        count = len(q)
        angle = q[:, :45].reshape(-1, 5, 3, 3)
        pose = compose(angle, self.model, self.axes) + self.model.hand_mean
        r = batch_rodrigues(torch.cat((torch.deg2rad(q[:, 45:48]), pose), 1).reshape(-1, 3)).reshape(
            -1, 16, 3, 3
        )
        j, aff = batch_rigid_transform(r, self.rest.expand(count, -1, -1), self.model.parents)
        v = self.base + ((r[:, 1:] - self.eye).flatten(1) @ self.ps).reshape(count, -1, 3)
        a = (self.weights @ aff.reshape(count, 16, 16)).reshape(count, -1, 4, 4)
        v = ((a[:, :, :3, :3] @ v[..., None]).squeeze(-1) + a[:, :, :3, 3]) * 1000 + q[:, None, 48:]
        joints = torch.cat((j * 1000 + q[:, None, 48:], v[:, self.tips]), 1)[:, self.order]
        if not mounts:
            return joints
        tri = v[:, self.tri]
        tx = F.normalize(tri[:, :, 1] - tri[:, :, 0], dim=-1)
        nz = F.normalize(torch.cross(tx, tri[:, :, 2] - tri[:, :, 0], dim=-1), dim=-1)
        by = torch.cross(nz, tx, dim=-1)
        surf = (
            (tri * self.bar[None, :, :, None]).sum(2)
            + tx * self.offset[:, 0:1]
            + by * self.offset[:, 1:2]
            + nz * self.offset[:, 2:3]
        )
        return joints, surf, pose, r[:, 0]

class Fit:
    def __init__(self, group, hand, bounds_folder=None):
        self.group = group
        self.hand = hand
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        t = self.tensor
        self.model = make_model(self.dev)
        path = paths(group, hand)
        self.soft = keys(
            path["soft35"],
            (
                "pair_indices",
                "global_orient",
                "translation",
                "hand_pose_pca",
                "betas",
                "attachment_faces",
                "faces",
                "barycentric",
                "attachment_local_offsets_m",
                "joints",
                "predicted24",
                "target24",
                "marker_valid",
                "tip_reference_m",
                "cuff_pivot_camera0_canonical_m",
                "root_anchor_local_m",
                "finger0_target",
            ),
        )
        z = self.soft
        self.ts = z["pair_indices"]
        beta = t(z["betas"])
        self.axes = finger_prior.rest_axes(self.model, beta, self.dev, include_thumb=True)
        self.kin = CompactHand(self.model, beta, self.axes, z)
        cuff = keys(
            path["cuff_hard"],
            ("pair_indices", "hand_pose_pca", "global_orient", "translation", "joints"),
        )
        np.testing.assert_equal(cuff["pair_indices"], self.ts)
        self.seed = self.parameters(cuff)
        self.start = self.parameters(z)
        self.seedj = t(cuff["joints"]) * 1000
        self.targets = t(np.nan_to_num(z["target24"][:, 3:])) * 1000
        self.valid = t(z["marker_valid"][:, 3:])
        self.cuff = t(z["cuff_pivot_camera0_canonical_m"]) * 1000
        self.sdk = t(z["finger0_target"]) * 1000
        self.sdk_local = t(z["root_anchor_local_m"]) * 1000
        self.tip = t(z["tip_reference_m"]) * 1000
        with torch.no_grad():
            for start in range(0, len(self.ts), 192):
                sl = slice(start, start + 192)
                j, m, *_ = self.kin.evaluate(self.start[sl])
                torch.testing.assert_close(j, t(z["joints"][sl]) * 1000, atol=0.003, rtol=0)
                torch.testing.assert_close(m, t(z["predicted24"][sl, 3:]) * 1000, atol=0.003, rtol=0)
        with np.load(case.assets / "wilor_pose2906.npz") as ref:
            selection = ref["selection"]
            samples = t(ref["local_rotvec"][selection[:, 3] == HANDS.index(hand)])
            reference_recording = REFERENCE
            source_names = json.loads((case.assets / "prior_recordings.json").read_text())
            same_ref = (selection[:, 3] == HANDS.index(hand)) & (
                selection[:, 0] == source_names.index(reference_recording)
            )
            cmc_samples = t(ref["local_rotvec"][same_ref])
        self.mean = samples.mean(0)
        cov = torch.cov(samples.T)
        cov = (
            0.95 * cov + 0.05 * cov.diag().diag() + np.deg2rad(2.0) ** 2 * torch.eye(45, device=self.dev)
        )
        self.white = torch.linalg.inv(torch.linalg.cholesky(cov)).T
        local = components(cmc_samples - self.model.hand_mean, self.model, self.axes)
        quant = torch.quantile(local, local.new_tensor([0.01, 0.99]), dim=0)
        lo, hi = [t(x) for x in limits(group, hand)]
        lo[4, 0] = quant[0, 4, 0] - 5
        hi[4, 0] = quant[1, 4, 0] + 5
        lo[:4, :, 1] = t([-15, -5, -10])
        lo[4, 1, 1] = -10
        lo[4, 2, 1] = -15
        for a in [0, 2]:
            lo[:4, 1:, a] = -5
            hi[:4, 1:, a] = 5
            lo[4, 2, a] = -5
            hi[4, 2, a] = 5
        self.lo = torch.cat((lo.flatten(), t([-np.inf] * 6)))
        self.hi = torch.cat((hi.flatten(), t([np.inf] * 6)))
        self.range_min = t([[-120, -10, -15]] + [[-15, -5, -10]] * 4)
        reference = json.loads((case.assets / "wilor_reference.json").read_text())[reference_recording][hand]
        self.range_min[0, 0] = reference["quantiles"][0][0][0] - 5
        self.cmc_max = reference["quantiles"][2][0][0] + 5
        _, _, _, rows = geometry(group)
        self.stamps = np.array([int(rows[int(s)]["reference_timestamp_ns"]) for s in self.ts])
        dd = np.diff(self.stamps)
        self.gap = 1.5 * np.median(dd[np.diff(self.ts) == 1])
        folder = ROOT / group if bounds_folder is None else bounds_folder
        folder.mkdir(parents=True, exist_ok=True)
        write_json(
            folder / (hand + "_anatomy_bounds.json"),
            dict(
                order=["index", "middle", "ring", "pinky", "thumb"],
                axes=list(AXES),
                lower_deg=lo.cpu().tolist(),
                upper_deg=hi.cpu().tolist(),
                signed_geometric_min_deg=self.range_min.cpu().tolist(),
                signed_geometric_thumb_cmc_upper_deg=self.cmc_max,
                cmc_reference=reference_recording,
                cmc_local_poses=len(cmc_samples),
                cmc_geometric_camera_observations=reference["n"],
                prior_samples=len(samples),
                covariance="95% full empirical covariance + 5% diagonal + (2 degrees)^2 I on local rotvec radians",
            ),
        )

    def tensor(self, x):
        return torch.as_tensor(x, dtype=torch.float32, device=self.dev)

    def parameters(self, z):
        return torch.cat(
            (
                components(self.tensor(z["hand_pose_pca"]), self.model, self.axes).flatten(1),
                torch.rad2deg(self.tensor(z["global_orient"])),
                self.tensor(z["translation"]) * 1000,
            ),
            1,
        )

    def pilot(self):
        eligible = ~np.isin(self.ts, held_times(self.group, self.hand)) & (abs(self.ts - 221) > 12)
        candidates = np.flatnonzero(eligible)
        centers = candidates[np.linspace(0, len(candidates) - 1, 26).round().astype(int)]
        return np.unique(
            np.concatenate(
                [
                    np.arange(max(i - 3, 0), min(i + 4, len(self.ts)))[
                        eligible[max(i - 3, 0) : min(i + 4, len(self.ts))]
                    ]
                    for i in centers
                ]
            )
        )

    def violations(self, j, target, cap):
        f = flex_torch(j)
        distances = (j[:, TIPS] - target).norm(dim=-1)
        return torch.cat(
            (
                F.relu(distances - cap + 0.003),
                0.25 * F.relu(self.range_min - f + 0.015).flatten(1),
                0.25 * F.relu(f[:, 0, 0:1] - self.cmc_max + 0.015),
            ),
            1,
        )

    @torch.no_grad()
    def project(self, q, targets, cap, steps=30):
        # Alternating active-set Gauss–Newton for the five tip balls AND signed-flex limits.
        # A failed intersection is retained and measured; local rotations never leave the box.
        q = q.clone()
        eps = 0.035
        dirs = torch.eye(51, device=self.dev) * eps
        mobile = torch.ones(51, device=self.dev)
        mobile[48:] = 0.2
        eye = torch.eye(21, device=self.dev)

        def residual(x, refs):
            return self.violations(self.kin.evaluate(x, False), refs, cap)

        for _ in range(steps):
            res = residual(q, targets)
            ids = torch.nonzero(res.amax(1) > 0.0006).flatten()
            if not len(ids):
                break
            x = q[ids]
            g = targets[ids]
            base = res[ids]
            n = len(ids)
            xx = x[:, None] + dirs[None]
            jac = (
                residual(
                    xx.reshape(-1, 51), g[:, None].expand(-1, 51, -1, -1).reshape(-1, 5, 3)
                ).reshape(n, 51, 21)
                - base[:, None, :]
            ).transpose(1, 2) / eps
            # Ignore inactive inequality rows, so other fingertips can move freely inside their own ball.
            jac = jac * (base > 0.0005)[:, :, None]
            weights = mobile.expand(n, -1).clone()
            for _ in range(4):
                jt = weights[:, :, None] * jac.transpose(1, 2)
                delta = -(jt @ torch.linalg.solve(jac @ jt + 0.0025 * eye, base[:, :, None])).squeeze(-1)
                block = ((x < self.lo + 0.015) & (delta < 0)) | ((x > self.hi - 0.015) & (delta > 0))
                if not (block & (weights > 0)).any():
                    break
                weights.masked_fill_(block, 0)
            scale = torch.minimum(
                4 / delta[:, :48].abs().amax(1).clamp_min(4),
                1.5 / delta[:, 48:].abs().amax(1).clamp_min(1.5),
            )
            trial = (
                x[:, None]
                + delta[:, None] * scale[:, None, None] * x.new_tensor([1, 0.5, 0.2, 0])[None, :, None]
            ).clamp(self.lo, self.hi)
            err = (
                residual(trial.reshape(-1, 51), g[:, None].expand(-1, 4, -1, -1).reshape(-1, 5, 3))
                .reshape(n, 4, -1)
                .square()
                .sum(2)
            )
            best = err.argmin(1)
            q[ids] = trial[torch.arange(n, device=self.dev), best]
            if not (best < 3).any():
                break
        return q

    def solve(self, indices, weight, cap=case.params['tip_cap_mm'], initial=None):
        if not len(indices):
            return self.start[:0], np.array([], dtype=int)
        qsource = self.start if initial is None else initial
        indices = np.asarray(indices)
        res = []
        for s in range(0, len(indices), 192):
            e = min(s + 192, len(indices))
            a = max(s - 4, 0)
            b = min(e + 4, len(indices))
            ii = indices[a:b]
            idx = torch.tensor(ii, device=self.dev)
            q = torch.nn.Parameter(qsource[idx].clamp(self.lo, self.hi).clone())
            seed = self.seed[idx]
            mask = self.valid[idx]
            target = self.targets[idx]
            tip = self.tip[idx]
            init = self.seedj[idx]
            rays = F.normalize(init[:, 1:], dim=-1)
            delta_t = np.diff(self.stamps[ii])
            links = np.flatnonzero((np.diff(self.ts[ii]) == 1) & (delta_t > 0) & (delta_t < self.gap))
            links = torch.as_tensor(links, device=self.dev)
            opt = torch.optim.Adam([q], lr=0.55)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 140, eta_min=0.005)
            for it in range(140):
                opt.zero_grad(set_to_none=True)
                j, m, pose, rot = self.kin.evaluate(q)
                err = (m - target).square().sum(-1) * mask
                loss = err.sum() / mask.sum() + 0.18 * err.topk(2, dim=1).values.mean()
                loss += 0.28 * (j[:, 0] - self.cuff[idx]).square().sum(1).mean()
                root = j[:, 0] + (rot @ self.sdk_local)
                loss += 0.10 * (root - self.sdk[idx]).square().sum(1).mean()
                move = j[:, 1:] - init[:, 1:]
                loss += (
                    0.08 * (move - (move * rays).sum(-1, keepdim=True) * rays).square().sum(-1).mean()
                )
                delta = q - seed
                ang = delta[:, :45].reshape(-1, 5, 3, 3)
                loss += (
                    0.025 * ang.square().mean()
                    + 0.055 * ang[:, :4, 1:, [0, 2]].square().mean()
                    + 0.026 * delta[:, 45:48].square().mean()
                )
                if len(links):
                    change = delta[links + 1] - delta[links]
                    loss += (
                        0.09 * change[:, :45].square().mean()
                        + 0.10 * change[:, 45:48].square().mean()
                        + 0.20 * change[:, 48:].square().mean()
                    )
                if weight:
                    loss += weight * ((pose - self.mean) @ self.white).square().mean()
                dist = (j[:, TIPS] - tip).norm(dim=-1)
                loss += 0.22 * dist.square().mean() + 13 * F.relu(dist - (cap - 0.5)).square().mean()
                f = flex_torch(j)
                loss += 1.2 * (
                    F.relu(self.range_min - f).square().sum(2).mean()
                    + F.relu(f[:, 0, 0] - self.cmc_max).square().mean()
                )
                loss.backward()
                opt.step()
                sched.step()
                with torch.no_grad():
                    q.clamp_(self.lo, self.hi)
            for offset in range(s, e, 32):
                length = min(32, e - offset)
                part = slice(offset - a, offset - a + length)
                res.append(self.project(q.detach()[part], tip[part], cap))
        return torch.cat(res), indices

    @torch.no_grad()
    def score(self, q, indices):
        j, m, *_ = self.kin.evaluate(q)
        f = flex_torch(j)
        dr = (j[:, TIPS] - self.tip[indices]).norm(dim=-1)
        err = (m - self.targets[indices]).norm(dim=-1).cpu().numpy()
        err[self.valid[indices].cpu().numpy() == 0] = np.nan
        complete = self.soft["marker_valid"][indices, 4:].reshape(-1, 5, 4).sum(2).min(1) >= case.params['min_markers_per_finger']
        passed = complete & (np.nanmedian(err, 1) <= case.params['surface_median_mm']) & (np.nanmax(err, 1) <= case.params['surface_max_mm'])
        violation = torch.maximum(
            F.relu(self.range_min - f).amax((1, 2)), F.relu(f[:, 0, 0] - self.cmc_max)
        )
        return f.cpu().numpy(), dr.cpu().numpy(), passed, violation.cpu().numpy()

    def run(self, name, weight=0.0, indices=None, prior_q=None):
        if indices is None:
            indices = np.arange(len(self.ts))
        if name == "relaxed":
            if prior_q is None:
                with np.load(checkpoint(self.group, "prior", self.hand)) as old:
                    prior_q = self.tensor(old["solution_q"])
            q0 = prior_q
            _, dr, _, gap = self.score(q0[indices], indices)
            conflict = (dr.max(1) > case.params['tip_cap_mm']) | ((gap > 0.15) & (dr.max(1) >= (case.params['tip_cap_mm'] - .05)))
            ii = indices[conflict]
            q = q0[indices].clone()
            chosen = np.zeros(len(indices), bool)
            if len(ii):
                # Test nominal feasibility once more from B before spending the larger fingertip budget.
                strict = torch.cat(
                    [
                        self.project(q0[i], self.tip[i], case.params['tip_cap_mm'], steps=60)
                        for i in np.array_split(ii, int(np.ceil(len(ii) / 32)))
                    ]
                )
                _, sd, _, sg = self.score(strict, ii)
                q[np.flatnonzero(conflict)] = strict
                still = (sd.max(1) > case.params['tip_cap_mm']) | ((sg > 0.15) & (sd.max(1) >= (case.params['tip_cap_mm'] - .05)))
                jj = ii[still]
                if len(jj):
                    relaxed, _ = self.solve(jj, weight, case.params['conflict_tip_cap_mm'], q0)
                    _, rd, _, rg = self.score(relaxed, jj)
                    # Accept a relaxed pose only when it reduces the witnessed conflict.
                    hardcost = np.maximum(sd[still].max(1) - case.params['tip_cap_mm'], 0) + 0.25 * sg[still]
                    softcost = np.maximum(rd.max(1) - 5, 0) + 0.25 * rg
                    use = softcost + 1.0e-5 < hardcost
                    loc = np.flatnonzero(conflict)[still][use]
                    q[loc] = relaxed[torch.as_tensor(use, device=self.dev)]
                    chosen[loc] = True
        else:
            q, indices = self.solve(indices, weight, case.params['tip_cap_mm'])
            chosen = np.zeros(len(indices), bool)
        return q, indices, chosen

    @torch.no_grad()
    def store(self, name, q, ii, relaxed, weight, *, source=None, destination=None, mounts=None):
        assert len(ii) == len(self.ts)
        src = paths(self.group, self.hand)["soft35"] if source is None else source
        with np.load(src) as a:
            z = {k: a[k] for k in a.files}
        if mounts is not None:
            z.update(mounts)
        z["hand_pose_pca"] = compose(q[:, :45].view(-1, 5, 3, 3), self.model, self.axes).cpu().numpy()
        z["global_orient"] = torch.deg2rad(q[:, 45:48]).cpu().numpy()
        z["translation"] = q[:, 48:].cpu().numpy() / 1000
        z["solution_q"] = q.cpu().numpy()
        z["tip_relaxed"] = relaxed
        z["tip_cap_per_pose_mm"] = np.where(relaxed, case.params['conflict_tip_cap_mm'], case.params['tip_cap_mm']).astype(np.float32)
        z["tip_constraint_mode"] = np.array("anatomy_" + name)
        z["prior_weight"] = np.array(weight)
        for key in ("projection_needed", "fallback_to_baseline_count", "tip_cap_mm", "tip_kink_mm"):
            z.pop(key, None)
        z["anatomy_lower_degrees"] = self.lo[:45].cpu().numpy().reshape(5, 3, 3)
        z["anatomy_upper_degrees"] = self.hi[:45].cpu().numpy().reshape(5, 3, 3)
        t = self.tensor
        tri = self.model.faces_tensor.long()[
            torch.as_tensor(z["attachment_faces"], device=self.dev).long()
        ]
        for s in range(0, len(q), 128):
            sl = slice(s, min(len(q), s + 128))
            n = len(q[sl])
            v, j, _ = run_model(
                self.model,
                t(z["betas"]).expand(n, -1),
                *[t(z[k][sl]) for k in ("global_orient", "hand_pose_pca", "translation")],
            )
            rot = t(Rotation.from_rotvec(z["cuff_global_orient"][sl]).as_matrix())
            m = marker_positions(
                v,
                t(z["cuff_pivot_camera0_canonical_m"][sl]),
                rot,
                tri,
                *[t(z[k]) for k in ("barycentric", "attachment_local_offsets_m", "wrist_local")],
            )
            refj, refm, _, _ = self.kin.evaluate(q[sl])
            torch.testing.assert_close(refj, j * 1000, atol=0.003, rtol=0)
            torch.testing.assert_close(refm, m[:, 3:] * 1000, atol=0.003, rtol=0)
            for k, val in (("vertices", v), ("joints", j), ("predicted24", m)):
                z[k][sl] = val.cpu().numpy()
        passed, err = gate_and_errors(z)
        if mounts is not None:
            z["pass_surface"] = passed
            z["cuff_points_observed"] = z["marker_valid"][:, :3].sum(axis=1).astype(np.uint8)
        z["marker_error_mm"] = err.astype(np.float32)
        z["tip_drift_mm"] = np.linalg.norm(z["joints"][:, TIPS] - z["tip_reference_m"], axis=-1) * 1000
        rotation = Rotation.from_rotvec(z["global_orient"])
        root = z["joints"][:, 0] + rotation.apply(np.broadcast_to(z["root_anchor_local_m"], (len(q), 3)))
        z["root_anchor_error_mm"] = (np.linalg.norm(root - z["finger0_target"], axis=1) * 1000).astype(
            np.float32
        )
        z["joint_minus_cuff_pivot_local_m"] = np.einsum(
            "nji,nj->ni",
            Rotation.from_rotvec(z["cuff_global_orient"]).as_matrix(),
            z["joints"][:, 0] - z["cuff_pivot_camera0_canonical_m"],
        )
        dest = checkpoint(self.group, name, self.hand) if destination is None else destination
        dest.parent.mkdir(parents=True, exist_ok=True)
        save(dest, z)
        print(
            self.group,
            self.hand,
            name,
            "poses",
            len(q),
            "pass",
            gate_and_errors(z)[0].mean(),
            "tip",
            percentiles(z["tip_drift_mm"]),
            "relaxed",
            relaxed.sum(),
            flush=True,
        )
