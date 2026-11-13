#!/usr/bin/env python3
"""Drive every frame of the recording with frozen calibrated shared parameters, then gate each label.

Phase 1: markers (missing ones masked) + Finger0 root anchor + MANO scan pose prior + frame-to-frame smoothness.
Phase 2: accepted-quality silhouettes additionally constrain the frames that have them.
Gate: a hand-frame is kept only if its mocap-to-camera alignment was corrected from the images, enough markers
were tracked and the marker residual is small. Rejected frames are listed with reasons and not written.
"""
import argparse, json, pickle, sys
from pathlib import Path
import numpy as np, torch
PROJECT_ROOT = Path(__file__).resolve().parents[2]; ROOT = PROJECT_ROOT.parent / 'wilor_training'
sys.path[:0] = [str(ROOT / 'tools/personal_hand_calibration'), str(ROOT / 'tools/full_silhouette')]
from run import PROJECT, import_mano, run_model, stats
from render_masks import FullMaskBatch
F = torch.nn.functional


def main():
    ap = argparse.ArgumentParser()
    for n in ['selection', 'source', 'output']: ap.add_argument('--' + n, type=Path, required=True)
    ap.add_argument('--steps', type=int, default=400); ap.add_argument('--silhouette-steps', type=int, default=300); ap.add_argument('--anat-weight', type=float, default=.01); ap.add_argument('--root-weight', type=float, default=3.)
    ap.add_argument('--min-finger-markers', type=int, default=19); ap.add_argument('--max-median-mm', type=float, default=2.5); ap.add_argument('--max-worst-mm', type=float, default=8.)
    a = ap.parse_args(); a.output.mkdir(parents=True); dev = 'cuda'; torch.manual_seed(0); selection = json.loads(a.selection.read_text())
    model = import_mano(PROJECT / 'third_party/MANO').load(model_path=str(PROJECT / 'models/mano'), is_rhand=True, use_pca=True, num_pca_comps=45, flat_hand_mean=False, batch_size=1).to(dev)
    from mano.lbs import batch_rodrigues
    for p in model.parameters(): p.requires_grad_(False)
    faces = model.faces_tensor.long(); to = lambda x: torch.as_tensor(x, dtype=torch.float32, device=dev); cpu = lambda x: x.detach().cpu().numpy()
    raw = pickle.load(open(PROJECT / 'models/mano/MANO_RIGHT.pkl', 'rb'), encoding='latin1'); scans = np.asarray(raw['hands_coeffs']) @ np.asarray(raw['hands_components'])
    precision = to(np.linalg.inv(np.cov(scans.T) + .01 * np.eye(45)))
    report = dict(strategy='joint_refit_v5_dense', approved_for_training=False, source=str(a.source.resolve()), gate=dict(min_finger_markers=a.min_finger_markers, max_median_mm=a.max_median_mm, max_worst_mm=a.max_worst_mm, alignment_corrected_required=True), hands={})
    (a.output / 'final').mkdir()
    present = [h for h in ['left', 'right'] if (a.source / f'{h}_pilot.npz').is_file()]
    if not present: raise ValueError('source has no hand fit')
    for side, hand in enumerate(['left', 'right']):
        if hand not in present: continue
        z = np.load(a.source / f'{hand}_pilot.npz'); pairs = z['pair_indices']; n = len(pairs); lookup = {int(f): i for i, f in enumerate(pairs)}
        beta, o0, p0, t0 = [to(z[k]) for k in ['betas', 'global_orient', 'hand_pose_pca', 'translation']]
        goal_np = z['target24']; valid = np.isfinite(goal_np).all(-1); valid[:, :3] = False  # cuff wrist markers are not used
        goal = to(np.nan_to_num(goal_np)); w = to(valid); attach = torch.as_tensor(z['attachment_faces'], device=dev).long(); bary = to(z['barycentric']); off = to(z['attachment_local_offsets_m'])
        root_goal = to(z['finger0_target']); c = to(z['root_anchor_local_m']); wrist = to(z['wrist_local'])
        order = np.argsort(pairs); near = np.diff(pairs[order]) == 1; prev, nxt = torch.as_tensor(order[:-1][near], device=dev), torch.as_tensor(order[1:][near], device=dev)
        records = [r for r in selection['records'] if r['side'] == side and r['accepted_quality'] and r['sync'] in lookup]
        batch = FullMaskBatch(records, lookup, z['joints'], z['vertices'], z['faces'], side, 256)

        def forward(dp, do, dt):
            v, j, _ = run_model(model, beta[None].expand(n, -1), o0 + do, p0 + dp, t0 + dt)
            tri = v[:, faces[attach]]; e1 = F.normalize(tri[:, :, 1] - tri[:, :, 0], dim=-1); nrm = F.normalize(torch.cross(e1, tri[:, :, 2] - tri[:, :, 0], dim=-1), dim=-1); e2 = torch.cross(nrm, e1, dim=-1)
            surf = (tri * bary[None, :, :, None]).sum(2) + e1 * off[None, :, :1] + e2 * off[None, :, 1:2] + nrm * off[None, :, 2:]
            rot = batch_rodrigues(o0 + do); pred = torch.cat([torch.einsum('fij,kj->fki', rot, wrist) + j[:, :1], surf], 1)
            return v, j, pred, (pred - goal).norm(dim=-1), (j[:, 0] + rot @ c - root_goal).norm(dim=-1)

        def iou(v):
            with torch.no_grad():
                p = batch.render(v).clamp(0, 1); i = (p * batch.target * batch.domain).sum((1, 2)); return cpu(i / ((p + batch.target - p * batch.target) * batch.domain).sum((1, 2)).clamp_min(1))

        dp, do, dt = [torch.nn.Parameter(torch.zeros_like(x)) for x in [p0, o0, t0]]

        def run(steps, silhouette):
            opt = torch.optim.Adam([dict(params=[dp], lr=.004), dict(params=[do], lr=.002), dict(params=[dt], lr=.0006)]); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
            for step in range(steps):
                opt.zero_grad(); v, j, pred, err, root = forward(dp, do, dt); pose = p0 + dp
                lm = ((torch.sqrt(1 + (err / .004).square()) - 1) * w).sum() / w.sum(); lr = a.root_weight * (torch.sqrt(1 + (root / .004).square()) - 1).mean() / 21
                prior = a.anat_weight * torch.einsum('fi,ij,fj->f', pose, precision, pose).mean() / 45
                smooth = .02 * (((dp[nxt] - dp[prev]) / .05).square().mean() + ((do[nxt] - do[prev]) / .05).square().mean() + ((dt[nxt] - dt[prev]) / .01).square().mean())
                loss = lm + lr + prior + smooth
                if silhouette: li, ld, _ = batch.terms(v); loss = loss + 4 * li + ld
                loss.backward(); opt.step(); sched.step()
                if step % 100 == 0 or step == steps - 1: print(hand, 'sil' if silhouette else 'mk ', step, 'loss %.4f marker med mm %.2f' % (float(loss), float(err[w > 0].median() * 1000)), flush=True)

        with torch.no_grad(): v, *_ = forward(dp, do, dt)
        iou0 = iou(v); run(a.steps, False)
        with torch.no_grad(): v, *_ = forward(dp, do, dt)
        iou1 = iou(v); run(a.silhouette_steps, True)
        with torch.no_grad(): v, j, pred, err, root = forward(dp, do, dt)
        iou2 = iou(v); e = cpu(err * 1000); e[~valid] = np.nan; fingers = valid[:, 3:].sum(1); med = np.nanmedian(np.where(fingers[:, None] > 0, e, 0), 1); worst = np.nanmax(np.where(fingers[:, None] > 0, e, 0), 1)
        aligned = z['image_alignment_corrected']; reasons = [[] for _ in range(n)]
        for i in range(n):
            if not aligned[i]: reasons[i].append('alignment_not_corrected')
            if fingers[i] < a.min_finger_markers: reasons[i].append('markers_missing')
            elif med[i] > a.max_median_mm or worst[i] > a.max_worst_mm: reasons[i].append('marker_residual')
        keep = np.array([not r for r in reasons]); counts = {k: sum(k in r for r in reasons) for k in ['alignment_not_corrected', 'markers_missing', 'marker_residual']}
        values = {k: z[k] for k in z.files}
        values.update(vertices=cpu(v), joints=cpu(j), global_orient=cpu(o0 + do), hand_pose_pca=cpu(p0 + dp), translation=cpu(t0 + dt), predicted24=cpu(pred), marker_error_mm=np.nan_to_num(e), marker_valid=valid,
                      root_anchor_error_mm=cpu(root * 1000), baseline_vertices=z['vertices'], baseline_joints=z['joints'], target24=np.nan_to_num(goal_np), approved_for_training=np.array(False))
        for k in list(values):
            if values[k].ndim and values[k].shape[0] == n and k != 'faces': values[k] = values[k][keep]
        np.savez_compressed(a.output / 'final' / f'{hand}_pilot.npz', **values)
        (a.output / f'{hand}_rejected.json').write_text(json.dumps([dict(sync=int(pairs[i]), reasons=reasons[i]) for i in range(n) if reasons[i]]))
        rk = np.array([keep[lookup[r['sync']]] for r in records])
        report['hands'][hand] = dict(frames=n, kept=int(keep.sum()), rejected=counts, marker_mm_kept=stats(e[keep][valid[keep]]), root_anchor_mm_kept=stats(cpu(root * 1000)[keep]),
                                     silhouette_views=len(records), iou_baseline=float(iou0.mean()), iou_marker_only=float(iou1.mean()), iou_final=float(iou2.mean()), iou_final_kept_views=float(iou2[rk].mean()), iou_final_p10_kept=float(np.percentile(iou2[rk], 10)))
        print(hand, json.dumps(report['hands'][hand]), flush=True); (a.output / 'summary.json').write_text(json.dumps(report, indent=1))


if __name__ == '__main__': main()
