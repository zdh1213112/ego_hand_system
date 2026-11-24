# Source work/tip_pivot_20261119/fit_pivot.py, SHA-256 1d1fcdf1001fb1b9841433526d4c5d9e3216a73aaa13ab9bc119dfc2b0cb6abe

import case_settings as case

import argparse

import json

import numpy as np

import torch

import torch.nn.functional as F

OUT = case.stage('tip_pivot_20261119')

from cohort import HERE, OPT, EGO, HANDS

from bounded_articulation import MARKER_ALLOWANCE_MM, RecordedTurns, angular_error, components, compose, finger_prior, import_mano, local_turns, marker_positions, measured_anchors, run_model

MCP = [2, 5, 9, 13, 17]  # Thumb CMC is index 1, thumb MCP is index 2.

TIPS = [4, 8, 12, 16, 20]

MODES = ("bounded1", "bounded3", "bounded6", "geometry3", "geometry6", "smooth", "supported", "ulnar")

SHARED = ("betas", "faces", "attachment_faces", "barycentric", "attachment_local_offsets_m",
          "wrist_local", "root_anchor_local_m")

FIELDS = ("hand_pose_pca", "global_orient", "translation", "target24", "marker_valid", "joints",
          "marker_error_mm", "finger0_target", "predicted24", "root_anchor_error_mm", "vertices")

def pivot_targets(joints, degrees):
    """Rotate each digit about its own tip in the palm plane. Average the five proposed wrist targets."""
    width = F.normalize(joints[:, 17] - joints[:, 5], dim=-1)
    forward = joints[:, MCP[1:]].mean(1) - joints[:, 0]
    normal = F.normalize(torch.cross(forward, width, dim=-1), dim=-1)[:, None]
    stem = joints[:, MCP] - joints[:, TIPS]
    tangent = torch.cross(normal.expand_as(stem), stem, dim=-1)
    sign = torch.sign((tangent * width[:, None]).sum(-1)).detach()
    angle = torch.deg2rad(sign * degrees)[..., None, None]
    finger = joints[:, 1:].reshape(-1, 5, 4, 3)
    wrist = joints[:, None, :1].expand(-1, 5, -1, -1)
    before = torch.cat([wrist, finger], dim=2)
    tip = finger[:, :, -1:]
    arm = before - tip
    axis = normal[:, :, None].expand_as(arm)
    turned = (tip + torch.cos(angle) * arm + torch.sin(angle) * torch.cross(axis, arm, dim=-1)
              + (1 - torch.cos(angle)) * (axis * arm).sum(-1, keepdim=True) * axis)
    target = torch.cat([turned[:, :, 0].mean(1, keepdim=True), turned[:, :, 1:].reshape(-1, 20, 3)], dim=1)
    return target, width

def sampled_indices(times, group, calibration, extra, mask_times):
    holdout = np.asarray([t for t in times if min(abs(t - np.asarray(calibration))) > 8])
    selected = set(holdout[np.linspace(0, len(holdout) - 1, min(extra, len(holdout))).round().astype(int)])
    visible = np.intersect1d(holdout, list(mask_times))
    if len(visible):
        selected.update(visible[np.linspace(0, len(visible) - 1, min(14, len(visible))).round().astype(int)])
    cases = {group: case.config.get('review_syncs', [])}
    for center in cases.get(group, []):
        selected.update(range(center - 5, center + 6))
    selected.update(case.config.get('wrist_sync', {}).values())
    return np.flatnonzero(np.isin(times, sorted(selected)))

def array_pack(group, hand, extra, calibration=False, model=None, interval=None):
    side = HANDS.index(hand)
    path = HERE / group
    name = hand + "_pilot.npz"
    selection = json.loads((OPT[group] / "selection/selection.json").read_text())
    cal_sync = selection["report"]["hands"][hand]["calibration_sync"]
    masks = {r["sync"] for r in selection["records"] if r["side"] == side and r["accepted_quality"]}
    with np.load(path / "contour/final" / name) as z:
        times = z["pair_indices"]
        index = sampled_indices(times, group, cal_sync, extra, masks)
        requested=case.config.get('wrist_sync', {}).get(hand)
        if requested is not None:
            if requested not in times:raise ValueError(f'{hand} 腕参照 {requested} 缺少有效姿态')
            index=np.union1d(index,np.flatnonzero(times==requested))
        if interval is not None:
            index = np.flatnonzero((times >= interval[0]) & (times <= interval[1]))
        if calibration:
            # Keep the entire test time blocks out of attachment calibration.
            cutoff = json.loads((HERE / "release_dataset/splits.json").read_text())["val_starts_at"][group]
            eligible = np.flatnonzero((times < cutoff) & ~np.isin(times // 60, np.unique(times[index] // 60)))
            if len(eligible) < 12:
                raise ValueError(f"Too few independent time blocks for {group} {hand} calibration")
            index = eligible[np.linspace(0, len(eligible) - 1, min(48, len(eligible))).round().astype(int)]
        arr = {k: z[k] for k in SHARED}
        arr.update({k: z[k][index] for k in FIELDS})
        arr["times"] = z["pair_indices"][index]
    with np.load(path / "reference/final" / name) as z:
        ref_idx = np.searchsorted(z["pair_indices"], arr["times"])
        np.testing.assert_array_equal(z["pair_indices"][ref_idx], arr["times"])
        arr["reference_error"] = z["marker_error_mm"][ref_idx]
    with np.load(path / "calibrated" / name) as z:
        arr["calibrated_root_p90"] = np.quantile(z["root_anchor_error_mm"][np.isin(z["pair_indices"], cal_sync)], .9)
    cfg = json.loads((path / "bounded/summary.json").read_text())["hands"][hand]
    anchors = None
    if cfg["calibration"].get("refit_within_bounds"):
        device = model.faces_tensor.device
        axes = finger_prior.rest_axes(model, torch.as_tensor(arr["betas"], device=device), device, include_thumb=True)
        limits = tuple(torch.as_tensor(cfg[k], device=device) for k in ("lower_degrees", "upper_degrees"))
        anchors, _ = measured_anchors(model, path / "calibrated" / name, cal_sync, axes, limits)
    sensor = RecordedTurns(OPT[group] / "skeleton/aligned_skeleton.npz", side, hand, path / "calibrated" / name,
                           cal_sync, anchors)
    return arr, sensor, cfg

class Trial:
    def __init__(self, model, arr, sensor, cfg, device):
        self.model, self.device = model, device
        self.arr = arr
        self.sensor = sensor
        tensor = self.tensor
        self.times = arr["times"]
        self.ndata = len(self.times)
        self.axes = finger_prior.rest_axes(model, tensor(arr["betas"]), device, include_thumb=True)
        self.limits = tuple(tensor(cfg[k]) for k in ("lower_degrees", "upper_degrees"))
        self.beta = tensor(arr["betas"]).expand(self.ndata, -1)
        self.orient = tensor(arr["global_orient"])
        self.trans = tensor(arr["translation"])
        self.pose = tensor(arr["hand_pose_pca"])
        self.joints = tensor(arr["joints"])
        self.angles = components(self.pose, model, self.axes)
        self.valid = torch.as_tensor(arr["marker_valid"], device=device)
        self.obs = tensor(arr["target24"]).nan_to_num()
        self.triangles = model.faces_tensor.long()[torch.as_tensor(arr["attachment_faces"], device=device).long()]
        self.mount = [tensor(arr[k]) for k in ("barycentric", "attachment_local_offsets_m", "wrist_local")]
        self.finger0 = tensor(arr["finger0_target"])
        self.root_offset = tensor(arr["root_anchor_local_m"])
        self.root_before = tensor(arr["root_anchor_error_mm"])
        self.root_cap = self.root_before.clamp_min(float(arr["calibrated_root_p90"]))
        self.marker_cap = tensor(arr["reference_error"]) + MARKER_ALLOWANCE_MM
        self.goals = sensor.expected[:, self.times].to(device)
        observed = self.valid[:, 4:].reshape(-1, 5, 4)
        self.sensor_valid = (sensor.valid[self.times].to(device) & self.valid[:, 8, None, None] &
                             self.valid[:, 20, None, None] & observed[:, :, :-2] & observed[:, :, 1:-1] &
                             observed[:, :, 2:])
        self.limit = sensor.limits.to(device).clamp_min(10.)
        self.neighbors = torch.as_tensor(np.flatnonzero(np.diff(self.times) == 1), device=device)
        self.base = self.forward(self.angles, torch.zeros_like(self.orient), torch.zeros_like(self.trans))
        for key, original in (("joints", "joints"), ("vertices", "vertices"), ("mounted", "predicted24")):
            np.testing.assert_allclose(self.base[key].cpu().numpy(), arr[original], atol=3e-6, rtol=0)

    def tensor(self, x):
        return torch.as_tensor(x, device=self.device, dtype=torch.float32)

    def forward(self, angles, dorient, dtrans):
        from mano.lbs import batch_rodrigues
        pose = compose(angles, self.model, self.axes)
        ori = self.orient + torch.deg2rad(dorient)
        tr = self.trans + dtrans * .001
        vertices, joints, _ = run_model(self.model, self.beta, ori, pose, tr)
        rotation = batch_rodrigues(ori)
        mount = marker_positions(vertices, joints[:, 0], rotation, self.triangles, *self.mount)
        err = (mount - self.obs).norm(dim=-1) * 1000
        root = (joints[:, 0] + rotation @ self.root_offset - self.finger0).norm(dim=-1) * 1000
        curve, curve_valid = local_turns(joints[:, 1:].reshape(-1, 5, 4, 3), self.obs[:, 8] - self.obs[:, 20])
        return dict(joints=joints, vertices=vertices, mounted=mount, err=err, root=root, bend=curve,
                    bend_valid=curve_valid, angles=angles, hand_pose_pca=pose, global_orient=ori, translation=tr)

    def fit(self, mode, steps):
        amount = {"smooth": 0, "supported": 1.5, "ulnar": 3, "bounded1": 1, "bounded3": 3, "bounded6": 6,
                  "geometry3": 3, "geometry6": 6}[mode]
        measured = not mode.startswith("geometry")
        target, outward = pivot_targets(self.joints, amount)
        tip_base = self.joints[:, TIPS]
        angles = torch.nn.Parameter(self.angles.clone())
        ulnar = mode == "ulnar"
        ori = torch.nn.Parameter(torch.zeros_like(self.orient), requires_grad=not ulnar)
        shift = torch.nn.Parameter(torch.zeros_like(self.trans[:, :1] if ulnar else self.trans))
        optimizer = torch.optim.Adam([{"params": [angles], "lr": .33}, {"params": [ori], "lr": .21},
                                      {"params": [shift], "lr": .32}])
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)
        base_error = self.base["err"].masked_fill(~self.valid, 0.).square().sum(1) / self.valid.sum(1)
        for _ in range(steps):
            optimizer.zero_grad()
            move = shift * outward if ulnar else shift
            a = self.forward(angles, ori, move)
            diff = (a["joints"] - target) * 1000
            tip = (a["joints"][:, TIPS] - tip_base) * 1000
            goal = diff[:, MCP].square().sum(-1).mean() + .65 * diff[:, 0].square().sum(-1).mean()
            chain = diff[:, [3, 6, 7, 10, 11, 14, 15, 18, 19]].square().sum(-1).mean()
            loss = (goal + .15 * chain) / 5.0 + 2.8 * tip.square().sum(-1).mean()
            # Reduce the lateral/twist component of PIP/DIP kinks, retaining the measured flexion.
            hinge = angles[:, :4, 1:, [0, 2]]
            loss = loss + .33 * (hinge / 4.).square().mean()
            loss = loss + .06 * ((angles - self.angles) / 8.).square().mean()
            if measured:
                err = a["err"].masked_fill(~self.valid, 0.)
                above = (err - self.marker_cap).clamp_min(0.).masked_fill(~self.valid, 0.)
                residual = err.square().sum(1) / self.valid.sum(1)
                root_excess = (a["root"] - self.root_cap).clamp_min(0.)
                turn = (a["bend"] - self.goals[0]).square().sum(-1)
                valid = self.sensor_valid & a["bend_valid"]
                loss = loss + .45 * residual.mean() + 5. * above.amax(1).square().mean()
                loss = loss + .30 * a["root"].square().mean() + 3. * root_excess.square().mean()
                loss = loss + .38 * (turn * valid).sum() / valid.sum().clamp_min(1) / np.deg2rad(8.)**2
                if mode in ("supported", "ulnar"):
                    loss = loss + 2.5 * residual.mean() + 7.0 * (residual - base_error).clamp_min(0.).mean()
                    loss = loss + .65 * (hinge / 4.).square().mean()
            near = self.neighbors
            if len(near):
                correction = angles - self.angles
                loss = loss + .06 * ((correction[near + 1] - correction[near]) / 4.).square().mean()
                loss = loss + .025 * (move[near + 1] - move[near]).square().mean()
            loss.backward()
            optimizer.step()
            schedule.step()
            with torch.no_grad():
                angles.clamp_(self.limits[0] + .002, self.limits[1] - .002)
                if ulnar:
                    shift.clamp_min_(0.)
        with torch.no_grad():
            move = shift * outward if ulnar else shift
            raw = self.forward(angles, ori, move)
            if not measured:
                return raw, torch.ones_like(self.root_before), target
            chosen = torch.zeros_like(self.root_before)
            for fraction in (.0625, .125, .25, .375, .5, .625, .75, .875, 1.):
                a = self.forward(torch.lerp(self.angles, angles, fraction), ori * fraction, move * fraction)
                within, _ = self.check(a)
                travel = ((a["joints"] - self.joints) * outward[:, None]).sum(-1)
                if amount:
                    within &= (travel[:, 0] > 0) & (travel[:, MCP[1:]] > 0).all(1)
                chosen[within] = fraction
            result = self.forward(torch.lerp(self.angles, angles, chosen[:, None, None, None]),
                                  ori * chosen[:, None], move * chosen[:, None])
            return result, chosen, target

    def check(self, a):
        err = a["err"]
        masked = err.masked_fill(~self.valid, float("nan"))
        turn_error = angular_error(a["bend"][None].expand_as(self.goals), self.goals)
        disputed = self.sensor_valid & a["bend_valid"] & (turn_error > self.limit).all(0)
        tip = ((a["joints"][:, TIPS] - self.joints[:, TIPS]) * 1000).norm(dim=-1)
        accepted = ((err <= self.marker_cap + .0005) | ~self.valid).all(1)
        accepted &= (torch.nanquantile(masked, .5, dim=1) <= 2.5) & ((err <= 8.) | ~self.valid).all(1)
        accepted &= (a["root"] <= self.root_cap + .001) & (tip.amax(1) <= 1.) & ~disputed.any((1, 2))
        return accepted, turn_error[0].masked_fill(~(self.sensor_valid & a["bend_valid"]), float("nan"))

    def snapshot(self, a, chosen):
        accepted, turn = self.check(a)
        _, outward = pivot_targets(self.joints, 0)
        delta = (a["joints"] - self.joints) * 1000
        side = (delta * outward[:, None]).sum(-1)
        hinge = a["angles"][:, :4, 1:, [0, 2]]
        result = {k: a[k] for k in ("hand_pose_pca", "global_orient", "translation", "joints", "vertices", "mounted")}
        result.update(marker_error_mm=a["err"].masked_fill(~self.valid, float("nan")), anchor_mm=a["root"],
                      tip_mm=delta[:, TIPS].norm(dim=-1), mcp_out_mm=side[:, MCP], wrist_out_mm=side[:, 0],
                      hinge_rms_deg=hinge.square().mean((1, 2, 3)).sqrt(), turn_error_deg=turn,
                      angles=a["angles"], feasible=accepted, fraction=chosen)
        return {k: v.detach().cpu().numpy() for k, v in result.items()}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--groups", nargs="+", default=[case.recording])
    p.add_argument("--modes", nargs="+", choices=MODES, default=MODES)
    p.add_argument("--extra", type=int, default=28)
    p.add_argument("--steps", type=int, default=125)
    p.add_argument("--interval", nargs=2, type=int, metavar=("FIRST", "LAST"))
    args = p.parse_args()
    torch.set_num_threads(2)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = import_mano(EGO / "third_party/MANO").load(model_path=str(EGO / "models/mano"), is_rhand=True,
        use_pca=True, num_pca_comps=45, flat_hand_mean=False, batch_size=1).to(device).eval().requires_grad_(False)
    for group in args.groups:
        for hand in HANDS:
            root = OUT if args.interval is None else OUT / f"sequence_{args.interval[0]}_{args.interval[1]}"
            folder = root / group / hand
            folder.mkdir(parents=True, exist_ok=True)
            arr, sensor, cfg = array_pack(group, hand, args.extra, model=model, interval=args.interval)
            op = Trial(model, arr, sensor, cfg, device)
            common = dict(times=arr["times"], betas=arr["betas"], target24=arr["target24"],
                          marker_valid=arr["marker_valid"], calibrated_root_p90_mm=arr["calibrated_root_p90"])
            def save(mode, data, chosen):
                snap = op.snapshot(data, chosen)
                tmp = folder / (mode + ".partial.npz")
                np.savez_compressed(tmp, **common, **snap)
                tmp.replace(folder / (mode + ".npz"))
                print(group, hand, mode, "n", len(chosen), "median_wrist", np.median(snap["wrist_out_mm"]),
                      "median_mcp", np.median(snap["mcp_out_mm"][:, 1:]),
                      "mean_hinge", snap["hinge_rms_deg"].mean(), "mean_fraction", snap["fraction"].mean(), flush=True)
            save("before", op.base, torch.zeros_like(op.root_before))
            for mode in args.modes:
                result, weights, _ = op.fit(mode, args.steps)
                save(mode, result, weights)
