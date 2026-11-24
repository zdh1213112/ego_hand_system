#!/usr/bin/env python3
"""Controlled marker refit and weak calibrated SDK direction auxiliary fit."""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools/personal_hand_calibration'))
from context import load_context
from run import PROJECT, import_mano, run_model, stats


def main():
    ap = argparse.ArgumentParser()
    for name in ['context', 'source', 'profile', 'output']:
        ap.add_argument('--' + name, type=Path, required=True)
    ap.add_argument('--variant', choices=['marker_only', 'direction'], required=True)
    ap.add_argument('--steps', type=int, default=300)
    a = ap.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    if a.steps < 1: ap.error('steps must be positive')
    c = load_context(a.context); target_file = np.load(a.profile / 'direction_targets.npz')
    mapping = json.loads((a.profile / 'mapping.json').read_text())
    if mapping['anatomical_mapping_verified'] or mapping['root_policy'] != 'finger0': raise ValueError('Unexpected mapping policy')
    torch.set_num_threads(4); torch.manual_seed(0); device = c['device']
    to = lambda x: torch.as_tensor(x, dtype=torch.float32, device=device)
    model = import_mano(PROJECT / 'third_party/MANO').load(model_path=str(PROJECT / 'models/mano'), is_rhand=True, use_pca=True, num_pca_comps=45, flat_hand_mean=False, batch_size=1).to(device)
    from mano.lbs import batch_rodrigues
    for p in model.parameters(): p.requires_grad_(False)
    if model.use_pca: raise ValueError('Expected 45 axis-angle components, not a PCA subspace')
    parents = torch.as_tensor(target_file['mano_parent'], device=device).long()
    children = torch.as_tensor(target_file['mano_child'], device=device).long()
    a.output.mkdir(parents=True)
    report = dict(strategy='marker24_with_optional_sdk_direction_v1', variant=a.variant,
                  source=str(a.source.resolve()), profile=str(a.profile.resolve()),
                  approved_for_training=False, anatomical_mapping_verified=False, root_policy='finger0',
                  shared_shape_frozen=True, marker_attachments_frozen=True, root_translation_frozen=True,
                  sdk_joint_positions_used_as_targets=False, sdk_bone_lengths_used_as_targets=False,
                  additional_marker_radius_subtraction_mm=0, fit_uses_images=False,
                  direction_weight=.10 if a.variant == 'direction' else 0,
                  limits=dict(finger_rotation_vector_increment_norm_rad=.10, root_rotation_vector_increment_norm_rad=.035),
                  hands={})
    for side, hand in enumerate(['left', 'right']):
        start = time.monotonic(); z = np.load(a.source / f'{hand}_pilot.npz'); pairs = z['pair_indices']; n = len(pairs)
        beta, o0, p0, t0 = [to(z[k]) for k in ['betas', 'global_orient', 'hand_pose_pca', 'translation']]
        goal = to(z['target24']); dirs = to(target_file['local_directions'][side, pairs]); confidence = to(target_file['confidence'][side])
        palm_target = to(target_file['palm_rotation_targets'][side, pairs])
        if not torch.isfinite(dirs).all() or not torch.isfinite(palm_target).all(): raise ValueError('Missing SDK directions')
        dp = torch.nn.Parameter(torch.zeros_like(p0)); do = torch.nn.Parameter(torch.zeros_like(o0))
        optimizer = torch.optim.Adam([dp, do], lr=.0015)
        order = np.argsort(pairs); near = np.diff(pairs[order]) <= 16
        prev = torch.as_tensor(order[:-1][near], device=device); following = torch.as_tensor(order[1:][near], device=device)
        faces = model.faces_tensor.long(); attach = torch.as_tensor(z['attachment_faces'], device=device).long()
        bary, offset, wrist = [to(z[k]) for k in ['barycentric', 'normal_offsets_m', 'wrist_local']]
        weights = to([.2] * 3 + [1.] * 21)

        def objective():
            v, j, out = run_model(model, beta[None].expand(n, -1), o0 + do, p0 + dp, t0)
            R = batch_rodrigues(o0 + do)
            tri = v[:, faces[attach]]
            normal = torch.nn.functional.normalize(torch.cross(tri[:, :, 1] - tri[:, :, 0], tri[:, :, 2] - tri[:, :, 0], dim=-1), dim=-1)
            surface = (tri * bary[None, :, :, None]).sum(2) + normal * offset[None, :, None]
            prediction = torch.cat([torch.einsum('fij,kj->fki', R, wrist) + j[:, :1], surface], 1)
            error = (prediction - goal).norm(dim=-1)
            marker = ((torch.sqrt(1 + (error / .006).square()) - 1) * weights).sum() / (n * weights.sum())
            local = torch.einsum('fji,fkj->fki', R, torch.nn.functional.normalize(j[:, children] - j[:, parents], dim=-1))
            chord = (local - dirs).norm(dim=-1)
            direction = ((torch.sqrt(1 + (chord / .174533).square()) - 1) * confidence).sum() / (n * confidence.sum())
            palm = (R - palm_target).square().sum((-1, -2)).mean() / (2 * .174533 ** 2)
            prior = .03 * (dp / .10).square().mean() + .03 * (do / .035).square().mean()
            temporal = 0 if not len(prev) else .01 * ((dp[following] - dp[prev]) / .10).square().mean()
            loss = marker + prior + temporal + (.10 * direction + .002 * palm if a.variant == 'direction' else 0)
            return loss, marker, direction, v, j, out, prediction, error, local

        history = []; best_loss = float('inf'); best = None
        for step in range(a.steps):
            optimizer.zero_grad(); loss, marker, direction, *_ = objective()
            if not torch.isfinite(loss): raise ValueError('Nonfinite direction fit')
            if float(loss.detach()) < best_loss:
                best_loss = float(loss.detach()); best = [p.detach().clone() for p in [dp, do]]
            loss.backward(); optimizer.step()
            with torch.no_grad():
                view = dp.view(n, 15, 3); view.mul_(torch.clamp(.10 / view.norm(dim=-1, keepdim=True).clamp_min(1e-9), max=1))
                do.mul_(torch.clamp(.035 / do.norm(dim=-1, keepdim=True).clamp_min(1e-9), max=1))
            if step % 60 == 0 or step == a.steps - 1:
                row = dict(step=step, loss=float(loss.detach()), marker=float(marker.detach()), direction=float(direction.detach()))
                history.append(row); print(hand, a.variant, json.dumps(row), flush=True)
        with torch.no_grad():
            loss, *_ = objective()
            if float(loss) < best_loss: best = [p.detach().clone() for p in [dp, do]]
            for p, q in zip([dp, do], best): p.copy_(q)
            loss, marker, direction, v, j, out, prediction, error, local = objective()
            r0 = batch_rodrigues((p0 + model.hand_mean).reshape(-1, 3)); r1 = batch_rodrigues((p0 + dp + model.hand_mean).reshape(-1, 3))
            rr = r0.transpose(-1, -2) @ r1
            angle = torch.acos(((rr.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1, 1)) * 180 / torch.pi
            local0 = torch.einsum('fji,fkj->fki', batch_rodrigues(o0), torch.nn.functional.normalize(to(z['joints'])[:, children] - to(z['joints'])[:, parents], dim=-1))
            def angles(x): return torch.acos((x * dirs).sum(-1).clamp(-1, 1)) * 180 / torch.pi
        cpu = lambda x: x.detach().cpu().numpy()
        values = {k: z[k] for k in z.files}
        values.update(vertices=cpu(v), joints=cpu(j), global_orient=cpu(o0 + do), hand_pose_pca=cpu(p0 + dp),
                      baseline_vertices=z['vertices'], baseline_joints=z['joints'], predicted24=cpu(prediction), marker_error_mm=cpu(error * 1000),
                      sdk_local_directions=cpu(dirs), sdk_direction_confidence=cpu(confidence),
                      sdk_direction_enabled=np.array(a.variant == 'direction'), root_policy=np.array('finger0'), approved_for_training=np.array(False))
        for key in ['betas', 'attachment_faces', 'barycentric', 'normal_offsets_m', 'wrist_local', 'translation']:
            assert np.array_equal(values[key], z[key]), f'Frozen parameter changed: {key}'
        np.savez_compressed(a.output / f'{hand}_pilot.npz', **values)
        report['hands'][hand] = dict(frames=n, marker_before_mm=stats(z['marker_error_mm']), marker_after_mm=stats(cpu(error * 1000)),
                                    direction_before_deg=stats(cpu(angles(local0))), direction_after_deg=stats(cpu(angles(local))),
                                    finger_angle_change_deg=stats(cpu(angle)), root_rotation_increment_deg=stats(cpu(do.norm(dim=-1) * 180 / torch.pi)),
                                    frozen_parameters_exactly_unchanged=True, history=history, elapsed_seconds=time.monotonic() - start)
        (a.output / 'summary.json').write_text(json.dumps(report, indent=2))
        print(hand, 'DONE', report['hands'][hand]['marker_after_mm'], flush=True)
    print(a.output, flush=True)


if __name__ == '__main__': main()
