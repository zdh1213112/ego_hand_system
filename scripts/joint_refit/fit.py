#!/usr/bin/env python3
"""Joint multi-view refit of shared glove shape / marker attachments and per-frame MANO motion.

Stage A (calibration): on training frames, optimise shared betas, shared marker attachments
(face-local 3D offsets) and per-frame pose with NOKOV markers + accepted multi-view silhouettes.
Stage B (drive): freeze shared parameters and fit every frame from markers only.
Stage C (optional final): like B but accepted silhouettes of all frames also constrain the pose.
Validation frames never contribute to stage A, so stage B IoU on them measures what a
marker-driven label looks like on frames without usable masks.
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]; ROOT = PROJECT_ROOT.parent / 'wilor_training'  # shared fitting tools live in wilor_training/tools
sys.path[:0] = [str(ROOT / 'tools/personal_hand_calibration'), str(ROOT / 'tools/full_silhouette')]
from context import load_context
from run import PROJECT, import_mano, run_model, stats
from render_masks import FullMaskBatch

F = torch.nn.functional


def main():
    ap = argparse.ArgumentParser()
    for name in ['context', 'selection', 'source', 'output']: ap.add_argument('--' + name, type=Path, required=True)
    ap.add_argument('--steps', type=int, default=700); ap.add_argument('--drive-steps', type=int, default=400)
    ap.add_argument('--resolution', type=int, default=256); ap.add_argument('--val-every', type=int, default=4)
    ap.add_argument('--beta-limit', type=float, default=3.0); ap.add_argument('--marker-weight', type=float, default=1.0)
    ap.add_argument('--normal-min', type=float, default=.0015); ap.add_argument('--adaptive-sigma', action='store_true'); ap.add_argument('--sigma-scale', type=float, default=1.)
    ap.add_argument('--debug-warm', action='store_true'); ap.add_argument('--final-silhouette', action='store_true', help='also write stage C (markers + accepted silhouettes on all frames)')
    ap.add_argument('--anat-weight', type=float, default=.03); ap.add_argument('--pose-floor', type=float, default=.1)
    ap.add_argument('--hands', nargs='+', default=['left', 'right'])
    # Root anchor: the SDK Finger0 common origin is rigid w.r.t. the palm (unlike the cuff wrist markers) and is the user's preferred wrist root.
    ap.add_argument('--skeleton', type=Path, default=None, help='aligned_skeleton.npz; enables the Finger0 root anchor')
    ap.add_argument('--root-offset', choices=['free', 'zero'], default='free', help='free: calibrate one shared palm-local offset Finger0-MANO wrist; zero: MANO wrist joint = Finger0')
    ap.add_argument('--root-weight', type=float, default=3.); ap.add_argument('--wrist-marker-weight', type=float, default=.2)
    a = ap.parse_args()
    if (a.output / 'summary.json').exists() or (a.output / 'final').exists(): raise FileExistsError(a.output)
    c = load_context(a.context); selection = json.loads(a.selection.read_text())
    torch.manual_seed(0); torch.set_num_threads(4); dev = 'cuda'
    model = import_mano(PROJECT / 'third_party/MANO').load(model_path=str(PROJECT / 'models/mano'), is_rhand=True, use_pca=True,
                                                         num_pca_comps=45, flat_hand_mean=False, batch_size=1).to(dev)
    from mano.lbs import batch_rodrigues
    for p in model.parameters(): p.requires_grad_(False)
    faces = model.faces_tensor.long(); to = lambda x: torch.as_tensor(x, dtype=torch.float32, device=dev); cpu = lambda x: x.detach().cpu().numpy()
    import pickle
    raw = pickle.load(open(PROJECT / 'models/mano/MANO_RIGHT.pkl', 'rb'), encoding='latin1')
    scans = np.asarray(raw['hands_coeffs']) @ np.asarray(raw['hands_components'])  # centred axis-angle poses of the MANO scans
    pose_precision = to(np.linalg.inv(np.cov(scans.T) + a.pose_floor ** 2 * np.eye(45)))
    a.output.mkdir(parents=True, exist_ok=True)
    report = dict(strategy='joint_refit_v4' if a.skeleton else 'joint_refit_v1', approved_for_training=False, source=str(a.source.resolve()), args={k: str(v) for k, v in vars(a).items()}, hands={})
    for hand in a.hands:
        side = ['left', 'right'].index(hand); start = time.monotonic()
        z = np.load(a.source / f'{hand}_pilot.npz'); pairs = z['pair_indices']; n = len(pairs); lookup = {int(f): i for i, f in enumerate(pairs)}
        records = [r for r in selection['records'] if r['side'] == side and r['accepted_quality'] and r['sync'] in lookup]
        masked = sorted({r['sync'] for r in records}); val_sync = set(masked[a.val_every // 2::a.val_every])
        train_frames = np.array([i for i, f in enumerate(pairs) if int(f) not in val_sync]); tf = torch.as_tensor(train_frames, device=dev)
        train_rec = [r for r in records if r['sync'] not in val_sync]; val_rec = [r for r in records if r['sync'] in val_sync]
        # Additionally hold out one view of every third multi-view training frame to test 3D consistency of silhouette-refined poses.
        multi = sorted({r['sync'] for r in train_rec if sum(q['sync'] == r['sync'] for q in train_rec) >= 2})[1::3]
        held_rec = [max((r for r in train_rec if r['sync'] == f), key=lambda r: r['camera']) for f in multi]
        train_rec = [r for r in train_rec if not any(r is h for h in held_rec)]
        mk = lambda rs: FullMaskBatch(rs, lookup, z['joints'], z['vertices'], z['faces'], side, a.resolution)
        bt, bv, bh, ball = mk(train_rec), mk(val_rec), mk(held_rec), mk(records)
        beta0, o0, p0, t0 = [to(z[k]) for k in ['betas', 'global_orient', 'hand_pose_pca', 'translation']]
        goal = to(z['target24']); attach = torch.as_tensor(z['attachment_faces'], device=dev).long(); bary = to(z['barycentric'])
        off0 = torch.zeros(21, 3, device=dev); off0[:, 2] = to(z['normal_offsets_m']).clamp_min(.002)
        wrist0 = to(z['wrist_local']); weights = to([a.wrist_marker_weight] * 3 + [1.] * 21)
        root_goal = None
        if a.skeleton is not None:
            sk = np.load(a.skeleton); assert str(sk['geometry_space']) == 'physical_camera0'
            f0 = sk['positions_cam0_m'][side][:, [1, 6, 11, 16, 21]]; assert np.nanmax(np.linalg.norm(f0 - f0[:, :1], axis=-1)) < 1e-6, 'Finger0 origins differ'
            f0 = f0[pairs, 0].copy(); f0[:, 0] *= -1 if side == 0 else 1
            if 'finger0_target' in z.files: f0 = z['finger0_target']  # already corrected by image_alignment.py
            assert np.isfinite(f0).all(); root_goal = to(f0)
        c0 = torch.zeros(3, device=dev)
        order = np.argsort(pairs); near = np.diff(pairs[order]) <= 16
        prev, nxt = torch.as_tensor(order[:-1][near], device=dev), torch.as_tensor(order[1:][near], device=dev)

        def forward(db, doff, dw, dc, dp, do, dt):
            v, j, _ = run_model(model, (beta0 + db)[None].expand(n, -1), o0 + do, p0 + dp, t0 + dt)
            tri = v[:, faces[attach]]; e1 = F.normalize(tri[:, :, 1] - tri[:, :, 0], dim=-1)
            nrm = F.normalize(torch.cross(e1, tri[:, :, 2] - tri[:, :, 0], dim=-1), dim=-1); e2 = torch.cross(nrm, e1, dim=-1)
            o = off0 + doff
            surf = (tri * bary[None, :, :, None]).sum(2) + e1 * o[None, :, :1] + e2 * o[None, :, 1:2] + nrm * o[None, :, 2:]
            rot = batch_rodrigues(o0 + do); wr = torch.einsum('fij,kj->fki', rot, wrist0 + dw) + j[:, :1]
            forward.root = None if root_goal is None else (j[:, 0] + rot @ (c0 + dc) - root_goal).norm(dim=-1)
            pred = torch.cat([wr, surf], 1); return v, j, pred, (pred - goal).norm(dim=-1)

        def root_term(idx=None):
            if root_goal is None: return 0.
            e = forward.root if idx is None else forward.root[idx]
            return a.root_weight * (torch.sqrt(1 + (e / .004).square()) - 1).mean() / weights.sum()

        sigma = torch.full((24,), .002, device=dev)  # per-marker noise scale, re-estimated from stage A residuals

        def marker_term(err, idx=None):
            e = err if idx is None else err[idx]; w = weights * (.002 / sigma).square()
            return ((torch.sqrt(1 + (e / (2 * sigma)).square()) - 1) * w).sum() / (len(e) * weights.sum())

        def update_sigma(err):
            with torch.no_grad(): sigma.copy_((err[tf].square().mean(0).sqrt() * a.sigma_scale).clamp(.0007, .02))

        # Pose prior: dorsal markers leave each phalanx free to roll about the marker line, which shifts the finger
        # sideways under the markers. Resolve it with the Mahalanobis distance to the MANO scan pose distribution
        # (hinge-like PIP/DIP, little twist) instead of anchoring to the previous, anatomically implausible fit.
        def motion_prior(dp, do, dt, anchor=True):
            t = 0 if not len(prev) else ((dp[nxt] - dp[prev]) / .15).square().mean() + ((do[nxt] - do[prev]) / .15).square().mean()
            pose = p0 + dp
            return a.anat_weight * torch.einsum('fi,ij,fj->f', pose, pose_precision, pose).mean() / 45 + .02 * t

        def evaluate(v, batch):
            with torch.no_grad():
                pred = batch.render(v).clamp(0, 1); inter = (pred * batch.target * batch.domain).sum((1, 2))
                union = ((pred + batch.target - pred * batch.target) * batch.domain).sum((1, 2)).clamp_min(1)
                return cpu(inter / union)

        zero = lambda x: torch.nn.Parameter(torch.zeros_like(x))
        # ---------------- stage A: joint calibration on training frames
        db, doff, dw, dc, dp, do, dt = zero(beta0), zero(off0), zero(wrist0), zero(c0), zero(p0), zero(o0), zero(t0)
        if a.root_offset == 'zero' or root_goal is None: dc.requires_grad_(False)
        opt = torch.optim.Adam([dict(params=[db], lr=.01), dict(params=[doff, dw, dc], lr=.0004), dict(params=[dp], lr=.004),
                                dict(params=[do], lr=.002), dict(params=[dt], lr=.0004)])
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps, eta_min=0)
        with torch.no_grad(): v, j, pred, err = forward(db, doff, dw, dc, dp, do, dt)
        root_before = None if root_goal is None else cpu(forward.root * 1000)
        base = dict(iou_train=evaluate(v, bt), iou_val=evaluate(v, bv), iou_held=evaluate(v, bh), marker_mm=cpu(err * 1000))
        history = []
        for step in range(a.steps):
            opt.zero_grad(); v, j, pred, err = forward(db, doff, dw, dc, dp, do, dt)
            if a.adaptive_sigma and step >= a.steps // 4 and step % 25 == 0: update_sigma(err)
            li, ld, iou = bt.terms(v); lm = marker_term(err, tf); o = off0 + doff
            shared = .002 * ((beta0 + db) / 2).square().mean() + 2 * (o[:, :2] / .02).square().mean() \
                + 50 * (F.relu(a.normal_min - o[:, 2]) / .002).square().mean() + 5 * (F.relu(o[:, 2] - .008) / .002).square().mean() + .5 * (dw / .02).square().mean()
            loss = 4 * li + ld + a.marker_weight * lm + root_term(tf) + motion_prior(dp, do, dt) + shared
            if not torch.isfinite(loss): raise ValueError('nonfinite')
            loss.backward(); opt.step(); sched.step()
            with torch.no_grad(): (beta0 + db).clamp_(-a.beta_limit, a.beta_limit); db.copy_((beta0 + db).clamp(-a.beta_limit, a.beta_limit) - beta0)
            if step % 50 == 0 or step == a.steps - 1:
                row = dict(step=step, loss=float(loss), soft_iou=float(iou), sdf=float(ld), marker=float(lm), marker_med_mm=float(err[tf][:, 3:].median() * 1000))
                history.append(row); print(hand, 'A', json.dumps(row), flush=True)
        with torch.no_grad(): vA, jA, predA, errA = forward(db, doff, dw, dc, dp, do, dt)
        if a.adaptive_sigma: update_sigma(errA)
        print(hand, 'sigma mm', cpu(sigma * 1000).round(2).tolist(), flush=True)
        A = dict(iou_train=evaluate(vA, bt), marker_mm=cpu(errA * 1000)[train_frames])
        shared_p = [x.detach().clone() for x in [db, doff, dw, dc]]; poseA = [x.detach().clone() for x in [dp, do, dt]]

        # ---------------- stage B / C: drive all frames with frozen shared parameters
        def drive(batch, steps, init=None):
            dp, do, dt = [torch.nn.Parameter(x.clone()) for x in init] if init is not None else (zero(p0), zero(o0), zero(t0))
            opt = torch.optim.Adam([dict(params=[dp], lr=.004), dict(params=[do], lr=.002), dict(params=[dt], lr=.0004)])
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=0)
            for step in range(steps):
                opt.zero_grad(); v, j, pred, err = forward(*shared_p, dp, do, dt)
                loss = a.marker_weight * marker_term(err) + root_term() + motion_prior(dp, do, dt, anchor=False)
                if batch is not None: li, ld, _ = batch.terms(v); loss = loss + 4 * li + ld
                loss.backward(); opt.step(); sched.step()
            with torch.no_grad():
                out = forward(*shared_p, dp, do, dt); return [x.detach() for x in [dp, do, dt]], out + (None if root_goal is None else cpu(forward.root * 1000),)

        rootA = None if root_goal is None else cpu(forward.root * 1000)
        results = {'calib': (poseA, (vA, jA, predA, errA, rootA)), 'drive': drive(None, a.drive_steps)}
        d = (results['drive'][0][0] - poseA[0])[tf].view(-1, 15, 3).norm(dim=-1).mean(0)
        print(hand, 'calib-vs-drive per-joint pose diff rad', cpu(d).round(3).tolist(), 'orient', float((results['drive'][0][1] - poseA[1])[tf].norm(dim=-1).mean()),
              'transl mm', float((results['drive'][0][2] - poseA[2])[tf].norm(dim=-1).mean() * 1000), flush=True)
        if a.debug_warm: results['warm'] = drive(None, a.drive_steps, poseA)
        if a.final_silhouette: results['final'] = drive(ball, a.steps)
        summary = dict(frames=n, train_frames=len(train_frames), val_sync=sorted(val_sync), train_views=len(train_rec), val_views=len(val_rec),
                       betas=cpu(beta0 + db).tolist(), sigma_mm=cpu(sigma * 1000).tolist(), offsets_mm=cpu((off0 + doff) * 1000).round(2).tolist(), wrist_local_mm=cpu((wrist0 + dw) * 1000).round(2).tolist(), root_anchor=None if root_goal is None else dict(target='sdk_finger0', offset=a.root_offset, local_offset_mm=cpu((c0 + dc) * 1000).round(2).tolist(), before_mm=stats(root_before)),
                       before=dict(iou_train=float(base['iou_train'].mean()), iou_val=float(base['iou_val'].mean()), marker_mm=stats(base['marker_mm'][:, 3:])),
                       stageA=dict(iou_train=float(A['iou_train'].mean()), marker_mm=stats(A['marker_mm'][:, 3:])), history=history)
        for name, ((dpx, dox, dtx), (v, j, pred, err, rooterr)) in results.items():
            it, iv, ih = evaluate(v, bt), evaluate(v, bv), evaluate(v, bh)
            summary[name] = dict(iou_train=float(it.mean()), iou_val=float(iv.mean()), iou_heldview=float(ih.mean()), iou_heldview_before=float(base['iou_held'].mean()), iou_val_per_view=iv.round(4).tolist(), before_val_per_view=base['iou_val'].round(4).tolist(),
                                 marker_mm=stats(cpu(err * 1000)[:, 3:]), wrist_marker_mm=stats(cpu(err * 1000)[:, :3]),
                                 pose_change_rad=stats(cpu(dpx.view(n, 15, 3).norm(dim=-1))))
            if rooterr is not None: summary[name]['root_anchor_mm'] = stats(rooterr)
            values = {k: z[k] for k in z.files}
            values.update(vertices=cpu(v), joints=cpu(j), baseline_vertices=z['vertices'], baseline_joints=z['joints'], betas=cpu(beta0 + db),
                          global_orient=cpu(o0 + dox), hand_pose_pca=cpu(p0 + dpx), translation=cpu(t0 + dtx), predicted24=cpu(pred), marker_error_mm=cpu(err * 1000),
                          attachment_local_offsets_m=cpu(off0 + doff), wrist_local=cpu(wrist0 + dw), normal_offsets_m=cpu((off0 + doff)[:, 2]),
                          validation_sync=np.array(sorted(val_sync)), approved_for_training=np.array(False))
            if root_goal is not None: values.update(root_anchor_target=cpu(root_goal), root_anchor_local_m=cpu(c0 + dc), root_anchor_error_mm=rooterr)
            (a.output / name).mkdir(exist_ok=True); np.savez_compressed(a.output / name / f'{hand}_pilot.npz', **values)
            print(hand, name, 'IoU train %.4f -> %.4f | heldview(%d) %.4f -> %.4f | valframes(%d) %.4f -> %.4f | marker med %.2f p95 %.2f' % (base['iou_train'].mean(), it.mean(), len(ih), base['iou_held'].mean(), ih.mean(), len(iv), base['iou_val'].mean(), iv.mean(),
                  summary[name]['marker_mm']['median'], summary[name]['marker_mm']['p95']), flush=True)
        summary['elapsed_seconds'] = time.monotonic() - start; report['hands'][hand] = summary
        (a.output / 'summary.json').write_text(json.dumps(report, indent=1))
    print(a.output)


if __name__ == '__main__': main()
