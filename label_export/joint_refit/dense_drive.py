#!/usr/bin/env python3
"""Drive every frame of the recording with frozen calibrated shared parameters, then gate each label.

Phase 1: markers (missing ones masked) + Finger0 root anchor + MANO scan pose prior + frame-to-frame smoothness.
Phase 2: accepted-quality silhouettes additionally constrain the frames that have them.
The default gate requires image-corrected alignment. A source checked per hand can instead use
--require-verified-hand-alignment, which also accepts image-verified raw positions. Both modes
require sufficient tracked markers and small marker residuals.
"""
import argparse, json, pickle, sys
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'label_export/pipeline/calibration/personal_hand_calibration'), str(ROOT / 'label_export/pipeline/calibration/full_silhouette')]
from run import PROJECT, import_mano, run_model, stats
from render_masks import FullMaskBatch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import finger_prior
F = torch.nn.functional


def marker_positions(vertices, root, rotation, attachment_triangles, barycentric, offsets, wrist_local):
    triangles = vertices[:, attachment_triangles]
    tangent = F.normalize(triangles[:, :, 1] - triangles[:, :, 0], dim=-1)
    normal = F.normalize(torch.cross(tangent, triangles[:, :, 2] - triangles[:, :, 0], dim=-1), dim=-1)
    bitangent = torch.cross(normal, tangent, dim=-1)
    surface = (triangles * barycentric[None, :, :, None]).sum(2)
    surface = surface + tangent * offsets[None, :, :1] + bitangent * offsets[None, :, 1:2] + normal * offsets[None, :, 2:]
    cuff = torch.einsum('fij,kj->fki', rotation, wrist_local) + root[:, None]
    return torch.cat([cuff, surface], 1)


def main():
    ap = argparse.ArgumentParser()
    for n in ['selection', 'source', 'output']: ap.add_argument('--' + n, type=Path, required=True)
    ap.add_argument('--steps', type=int, default=400); ap.add_argument('--silhouette-steps', type=int, default=300); ap.add_argument('--anat-weight', type=float, default=.01); ap.add_argument('--root-weight', type=float, default=3.)
    finger_prior.add_arguments(ap)
    ap.add_argument('--sdk-directions', type=Path,
                    help='optional recording-specific directions.npz from sdk_directions.py')
    ap.add_argument('--sdk-marker-gaps', type=Path,
                    help='optional marker_gaps.npz for fingers with three real observations and one missing marker')
    ap.add_argument('--require-verified-hand-alignment', action='store_true',
                    help='gate each hand by its own image evidence, including verified uncorrected NOKOV targets')
    ap.add_argument('--min-finger-markers', type=int, default=19, help='minimum observed points among HandOffset and the twenty finger markers'); ap.add_argument('--min-markers-per-finger', type=int, default=0, help='reject a frame if any finger has fewer of its 4 markers (0 = off)'); ap.add_argument('--max-median-mm', type=float, default=2.5); ap.add_argument('--max-worst-mm', type=float, default=8.)
    a = ap.parse_args(); a.output.mkdir(parents=True); dev = 'cuda'; torch.manual_seed(0); selection = json.loads(a.selection.read_text())
    model = import_mano(PROJECT / 'third_party/MANO').load(model_path=str(PROJECT / 'models/mano'), is_rhand=True, use_pca=True, num_pca_comps=45, flat_hand_mean=False, batch_size=1).to(dev)
    from mano.lbs import batch_rodrigues
    for p in model.parameters(): p.requires_grad_(False)
    faces = model.faces_tensor.long(); to = lambda x: torch.as_tensor(x, dtype=torch.float32, device=dev); cpu = lambda x: x.detach().cpu().numpy()
    raw = pickle.load(open(PROJECT / 'models/mano/MANO_RIGHT.pkl', 'rb'), encoding='latin1'); scans = np.asarray(raw['hands_coeffs']) @ np.asarray(raw['hands_components'])
    precision = to(np.linalg.inv(np.cov(scans.T) + .01 * np.eye(45)))
    report = dict(strategy='joint_refit_v5_dense', approved_for_training=False, source=str(a.source.resolve()), gate=dict(min_finger_markers=a.min_finger_markers, max_median_mm=a.max_median_mm, max_worst_mm=a.max_worst_mm, alignment_corrected_required=True), hands={})
    if a.require_verified_hand_alignment:
        report['gate'].update(alignment_corrected_required=False, hand_alignment_verified_required=True)
    if a.min_markers_per_finger:
        report['gate']['min_markers_per_finger'] = a.min_markers_per_finger
    (a.output / 'final').mkdir()
    present = [h for h in ['left', 'right'] if (a.source / f'{h}_pilot.npz').is_file()]
    if not present: raise ValueError('source has no hand fit')
    for side, hand in enumerate(['left', 'right']):
        if hand not in present: continue
        z = np.load(a.source / f'{hand}_pilot.npz'); pairs = z['pair_indices']; n = len(pairs); lookup = {int(f): i for i, f in enumerate(pairs)}
        alignment_key = 'image_alignment_verified' if a.require_verified_hand_alignment else 'image_alignment_corrected'
        aligned = z[alignment_key]
        if a.require_verified_hand_alignment and (aligned.dtype != np.bool_ or aligned.shape != (n,)):
            raise ValueError(f'{hand}: expected a boolean {alignment_key} entry for each source frame')
        alignment_reason = 'hand_alignment_unverified' if a.require_verified_hand_alignment else 'alignment_not_corrected'
        beta, o0, p0, t0 = [to(z[k]) for k in ['betas', 'global_orient', 'hand_pose_pca', 'translation']]
        goal_np = z['target24']; valid = np.isfinite(goal_np).all(-1); valid[:, :3] = False  # cuff wrist markers are not used
        goal = to(np.nan_to_num(goal_np)); w = to(valid); attach = torch.as_tensor(z['attachment_faces'], device=dev).long(); bary = to(z['barycentric']); off = to(z['attachment_local_offsets_m'])
        root_goal = to(z['finger0_target']); c = to(z['root_anchor_local_m']); wrist = to(z['wrist_local'])
        sdk = None
        if a.sdk_directions is not None:
            import sdk_directions
            sdk = sdk_directions.DirectionTargets(a.sdk_directions, side, pairs, z['target24_uncorrected'], dev)
            usage = report.setdefault('sdk_directions', dict(profile=str(a.sdk_directions.resolve()),
                                      weight=sdk_directions.WEIGHT, usable_segments={}))
            usage['usable_segments'][hand] = int((sdk.weights>0).sum().item())
        gaps = None
        if a.sdk_marker_gaps is not None:
            from sdk_marker_gaps import MarkerGapTargets
            gaps = MarkerGapTargets(a.sdk_marker_gaps, side, pairs, z['target24_uncorrected'], goal_np, dev)
            usage = report.setdefault('sdk_marker_gaps', dict(profile=str(a.sdk_marker_gaps.resolve()), inferred_observations={}))
            usage['inferred_observations'][hand] = gaps.count
        order = np.argsort(pairs); near = np.diff(pairs[order]) == 1; prev, nxt = torch.as_tensor(order[:-1][near], device=dev), torch.as_tensor(order[1:][near], device=dev)
        records = [r for r in selection['records'] if r['side'] == side and r['accepted_quality'] and r['sync'] in lookup]
        batch = FullMaskBatch(records, lookup, z['joints'], z['vertices'], z['faces'], side, 256)
        jl = finger_prior.config(a); jl_axes = finger_prior.rest_axes(model, beta, dev)

        attachment_triangles = faces[attach]

        def forward(dp, do, dt):
            v, j, _ = run_model(model, beta[None].expand(n, -1), o0 + do, p0 + dp, t0 + dt)
            rot = batch_rodrigues(o0 + do)
            pred = marker_positions(v, j[:, 0], rot, attachment_triangles, bary, off, wrist)
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
                prior = a.anat_weight * torch.einsum('fi,ij,fj->f', pose, precision, pose).mean() / 45 + finger_prior.penalty(pose, model, jl_axes, jl)
                smooth = .02 * (((dp[nxt] - dp[prev]) / .05).square().mean() + ((do[nxt] - do[prev]) / .05).square().mean() + ((dt[nxt] - dt[prev]) / .01).square().mean())
                loss = lm + lr + prior + smooth
                if sdk is not None:
                    loss = loss + sdk_directions.WEIGHT * sdk.penalty(j, batch_rodrigues(o0 + do))
                if gaps is not None and gaps.count:
                    loss = loss + gaps.penalty(pred, w.sum())
                if silhouette: li, ld, _ = batch.terms(v); loss = loss + 4 * li + ld
                loss.backward(); opt.step(); sched.step()
                if step % 100 == 0 or step == steps - 1: print(hand, 'sil' if silhouette else 'mk ', step, 'loss %.4f marker med mm %.2f' % (float(loss), float(err[w > 0].median() * 1000)), flush=True)

        with torch.no_grad(): v, *_ = forward(dp, do, dt)
        iou0 = iou(v); run(a.steps, False)
        with torch.no_grad(): v, *_ = forward(dp, do, dt)
        iou1 = iou(v); run(a.silhouette_steps, True)
        with torch.no_grad(): v, j, pred, err, root = forward(dp, do, dt)
        iou2 = iou(v); e = cpu(err * 1000); e[~valid] = np.nan; fingers = valid[:, 3:].sum(1); med = np.nanmedian(np.where(fingers[:, None] > 0, e, 0), 1); worst = np.nanmax(np.where(fingers[:, None] > 0, e, 0), 1)
        reasons = [[] for _ in range(n)]; per_finger = valid[:, 4:].reshape(n, 5, 4).sum(-1)  # thumb..pinky, 4 markers each
        for i in range(n):
            if not aligned[i]: reasons[i].append(alignment_reason)
            if fingers[i] < a.min_finger_markers or per_finger[i].min() < a.min_markers_per_finger: reasons[i].append('markers_missing')
            elif med[i] > a.max_median_mm or worst[i] > a.max_worst_mm: reasons[i].append('marker_residual')
        keep = np.array([not r for r in reasons]); counts = {k: sum(k in r for r in reasons) for k in [alignment_reason, 'markers_missing', 'marker_residual']}
        values = {k: z[k] for k in z.files}
        values.update(vertices=cpu(v), joints=cpu(j), global_orient=cpu(o0 + do), hand_pose_pca=cpu(p0 + dp), translation=cpu(t0 + dt), predicted24=cpu(pred), marker_error_mm=np.nan_to_num(e), marker_valid=valid,
                      root_anchor_error_mm=cpu(root * 1000), baseline_vertices=z['vertices'], baseline_joints=z['joints'], target24=np.nan_to_num(goal_np), approved_for_training=np.array(False))
        for k in list(values):
            if values[k].ndim and values[k].shape[0] == n and k != 'faces': values[k] = values[k][keep]
        np.savez_compressed(a.output / 'final' / f'{hand}_pilot.npz', **values)
        (a.output / f'{hand}_rejected.json').write_text(json.dumps([dict(sync=int(pairs[i]), reasons=reasons[i]) for i in range(n) if reasons[i]]))
        rk = np.array([keep[lookup[r['sync']]] for r in records])
        report['finger_prior'] = jl
        report['hands'][hand] = dict(finger_joints_kept=finger_prior.report((p0 + dp)[torch.as_tensor(keep, device=dev)], model, jl_axes), frames=n, kept=int(keep.sum()), rejected=counts, marker_mm_kept=stats(e[keep][valid[keep]]), root_anchor_mm_kept=stats(cpu(root * 1000)[keep]),
                                     silhouette_views=len(records), iou_baseline=float(iou0.mean()), iou_marker_only=float(iou1.mean()), iou_final=float(iou2.mean()), iou_final_kept_views=float(iou2[rk].mean()) if rk.any() else None, iou_final_p10_kept=float(np.percentile(iou2[rk], 10)) if rk.any() else None, silhouette_views_kept=int(rk.sum()))
        print(hand, json.dumps(report['hands'][hand]), flush=True); (a.output / 'summary.json').write_text(json.dumps(report, indent=1))


if __name__ == '__main__': main()
