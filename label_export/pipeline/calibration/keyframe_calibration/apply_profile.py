#!/usr/bin/env python3
"""Fit sampled motion using NOKOV24 and a read-only quality-keyframe calibration.

No segmentation, per-frame shape fitting, or attachment refinement is performed.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'label_export/pipeline/calibration/personal_hand_calibration'))
from context import load_context
from run import PROJECT, import_mano, run_model, stats
from calibration_profile import file_sha256, validate_profile


def main():
    ap = argparse.ArgumentParser()
    for name in ['context', 'source', 'calibration', 'output']:
        ap.add_argument('--' + name, type=Path, required=True)
    ap.add_argument('--steps', type=int, default=180)
    a = ap.parse_args()
    if a.steps < 1:
        ap.error('steps must be positive')
    if a.output.exists():
        raise FileExistsError(a.output)
    c = load_context(a.context)
    profile_bytes = a.calibration.read_bytes()
    data = json.loads(profile_bytes)
    mano = import_mano(PROJECT / 'third_party/MANO')
    model = mano.load(model_path=str(PROJECT / 'models/mano'), is_rhand=True, use_pca=True,
                      num_pca_comps=45, flat_hand_mean=False, batch_size=1).to(c['device'])
    from mano.lbs import batch_rodrigues
    validate_profile(data, model.faces, json.loads(c['placement'].read_text()), PROJECT / 'models/mano/MANO_RIGHT.pkl')
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    torch.set_num_threads(4)
    torch.manual_seed(0)
    a.output.mkdir(parents=True)
    to = lambda x: torch.as_tensor(x, dtype=torch.float32, device=c['device'])
    faces = model.faces_tensor.long()
    same_recording = data['provenance']['source_dataset'] == str(c['dataset'].resolve())
    summary = {'strategy': 'quality_keyframe_fixed_calibration_motion_v1', 'approved_for_training': False,
               'quality_keyframe_calibration_enabled': True, 'fixed_calibration_applied': True,
               'calibration_profile': str(a.calibration.resolve()), 'calibration_sha256': file_sha256(a.calibration),
               'shape_sharing': 'one_beta_vector_per_physical_hand',
               'shared_parameters_frozen': True, 'pose_images_used': False,
               'same_recording_as_calibration': same_recording, 'hands': {}}
    for side, hand in enumerate(['left', 'right']):
        z = np.load(a.source / f'{hand}_pilot.npz')
        b = np.load(c['baseline_fit'] / f'track_{side}.npz')
        lookup = {int(f): i for i, f in enumerate(b['pair_indices'])}
        pairs = z['pair_indices']
        ids = np.array([lookup[int(f)] for f in pairs])
        n = len(ids)
        q = data['hands'][hand]
        # Shared quantities are ordinary constant tensors, never optimizer parameters.
        beta, bary, offset, wrist = [to(q[k]) for k in ['betas', 'barycentric', 'normal_offsets_m', 'wrist_local_m']]
        face_ids = torch.as_tensor(q['attachment_faces'], dtype=torch.long, device=c['device'])
        fixed = [beta, bary, offset, wrist]
        snapshots = [p.clone() for p in fixed]
        p0, o0, t0 = [to(b[k][ids]) for k in ['hand_pose_pca', 'global_orient', 'translation']]
        target = to(z['target24'])
        dp, do, dt = [torch.nn.Parameter(torch.zeros_like(x)) for x in [p0, o0, t0]]
        optimizer = torch.optim.Adam([{'params': [dp, do], 'lr': .0015}, {'params': [dt], 'lr': .00015}])
        weights = to([.2] * 3 + [1] * 21)
        order = np.argsort(pairs)
        near = np.diff(pairs[order]) <= 16
        left = torch.tensor(order[:-1][near], device=c['device'])
        right = torch.tensor(order[1:][near], device=c['device'])
        def geometry():
            return run_model(model, beta[None].expand(n, -1), o0 + do, p0 + dp, t0 + dt)[:2]
        def markers(v, j):
            tri = v[:, faces[face_ids]]
            normal = torch.nn.functional.normalize(torch.cross(tri[:, :, 1] - tri[:, :, 0],
                                                               tri[:, :, 2] - tri[:, :, 0], dim=-1), dim=-1)
            surface = (tri * bary[None, :, :, None]).sum(2) + normal * offset[None, :, None]
            wrists = torch.einsum('fij,kj->fki', batch_rodrigues(o0 + do), wrist) + j[:, :1]
            return torch.cat([wrists, surface], 1)
        for _ in range(a.steps):
            optimizer.zero_grad()
            v, j = geometry()
            error = (markers(v, j) - target).norm(dim=-1) / .006
            loss = ((torch.sqrt(1 + error.square()) - 1) * weights).mean()
            loss += .03 * (dp / .12).square().mean() + .03 * (do / .06).square().mean() + .03 * (dt / .008).square().mean()
            if len(left):
                loss += .01 * ((dp[right] - dp[left]) / .12).square().mean() + .01 * ((dt[right] - dt[left]) / .008).square().mean()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                dp.clamp_(-.12, .12)
                do.mul_(torch.clamp(.06 / do.norm(dim=-1, keepdim=True).clamp_min(1e-9), max=1))
                dt.mul_(torch.clamp(.008 / dt.norm(dim=-1, keepdim=True).clamp_min(1e-9), max=1))
        assert all(torch.equal(x, y) for x, y in zip(fixed, snapshots)), 'Shared calibration changed'
        with torch.no_grad():
            v, j = geometry()
            predicted = markers(v, j)
            residual = (predicted - target).norm(dim=-1) * 1000
        cpu = lambda t: t.detach().cpu().numpy()
        out = {k: z[k] for k in z.files}
        calibration_mask = np.isin(pairs, q['calibration_sync']) if same_recording else np.zeros(n, bool)
        out.update(vertices=cpu(v), joints=cpu(j), baseline_vertices=b['vertices'][ids], baseline_joints=b['joints'][ids],
                   betas=cpu(beta), global_orient=cpu(o0 + do), hand_pose_pca=cpu(p0 + dp), translation=cpu(t0 + dt),
                   predicted24=cpu(predicted), marker_error_mm=cpu(residual), attachment_faces=cpu(face_ids),
                   barycentric=cpu(bary), normal_offsets_m=cpu(offset), wrist_local=cpu(wrist),
                   calibration_mask=calibration_mask, calibration_frame_has_image_constraints=np.zeros(n, bool),
                   quality_keyframe_calibration_enabled=np.array(True), approved_for_training=np.array(False))
        np.savez_compressed(a.output / f'{hand}_pilot.npz', **out)
        summary['hands'][hand] = {'frames': n, 'calibration_sync': q['calibration_sync'] if same_recording else [],
                                  'marker_residual_mm': stats(cpu(residual)),
                                  'shared_parameters_unchanged': True, 'image_constraints_used': False}
        print(hand, json.dumps(summary['hands'][hand]), flush=True)
    assert a.calibration.read_bytes() == profile_bytes, 'Input calibration file changed'
    # Save a snapshot rather than mutate the original accepted calibration artifact.
    (a.output / 'shared_calibration.json').write_bytes(profile_bytes)
    (a.output / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(a.output, flush=True)


if __name__ == '__main__':
    main()
