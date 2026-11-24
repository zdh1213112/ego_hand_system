#!/usr/bin/env python3
"""Check coordinate invariance, frozen parameters, bounded rotations and exports."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'tools/label_pipeline/calibration/personal_hand_calibration'))
from run import PROJECT, import_mano, run_model
from build_mapping import unit, proper_rotation, MP_PARENT, MP_CHILD


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', type=Path, required=True); ap.add_argument('--result', type=Path, required=True)
    a = ap.parse_args(); model = import_mano(PROJECT / 'third_party/MANO').load(model_path=str(PROJECT / 'models/mano'), is_rhand=True, use_pca=True, num_pca_comps=45, flat_hand_mean=False, batch_size=1)
    torch.set_num_threads(2); report = dict(approved_for_training=False, hands={})
    rng = np.random.default_rng(0)
    A = Rotation.from_rotvec(rng.normal(size=3)).as_matrix(); translation = rng.normal(size=3)
    v = rng.normal(size=(10, 21, 3)); R = Rotation.from_rotvec(rng.normal(size=(10, 3))).as_matrix()
    def local(p, rot): return np.einsum('fji,fkj->fki', rot, unit(p[:, MP_CHILD] - p[:, MP_PARENT]))
    error = np.max(abs(local(v, R) - local(3.2 * v @ A.T + translation, A @ R)))
    assert error < 1e-12
    # Reflect camera and local axes together: canonical rotations stay proper.
    S = np.diag([-1, 1, 1]); mirrored = local(v @ S, S @ R @ S)
    assert np.max(abs(mirrored - local(v, R) @ S)) < 1e-12
    assert np.max(abs(np.linalg.det(S @ R @ S) - 1)) < 1e-12
    assert np.linalg.det(proper_rotation(np.diag([-1., 1, 1]))) > .999999
    report['direction_invariant_to_camera_rotation_translation_and_uniform_scale_max_error'] = float(error)
    report['left_canonical_reflection_verified'] = True
    for variant in ['marker_only', 'direction']:
        fit = json.loads((a.result / variant / 'summary.json').read_text())
        assert fit['sdk_joint_positions_used_as_targets'] is False and fit['sdk_bone_lengths_used_as_targets'] is False
        report['hands'][variant] = {}
        for hand in ['left', 'right']:
            z = np.load(a.result / variant / f'{hand}_pilot.npz'); old = np.load(a.source / f'{hand}_pilot.npz')
            assert z['approved_for_training'].item() is False
            for key in ['betas', 'attachment_faces', 'barycentric', 'normal_offsets_m', 'wrist_local', 'translation', 'target24', 'pair_indices']:
                assert np.array_equal(z[key], old[key]), f'{variant}/{hand}/{key} changed'
            dp = (z['hand_pose_pca'] - old['hand_pose_pca']).reshape(-1, 15, 3)
            do = z['global_orient'] - old['global_orient']
            assert np.linalg.norm(dp, axis=-1).max() < .100002 and np.linalg.norm(do, axis=-1).max() < .035002
            n = len(z['pair_indices'])
            with torch.no_grad():
                vv, jj, _ = run_model(model, torch.tensor(z['betas'])[None].expand(n, -1), torch.tensor(z['global_orient']), torch.tensor(z['hand_pose_pca']), torch.tensor(z['translation']))
            replay = max(float(abs(vv - torch.tensor(z['vertices'])).max()), float(abs(jj - torch.tensor(z['joints'])).max()))
            assert replay < 2e-6
            report['hands'][variant][hand] = dict(frames=n, frozen_parameters_exact=True, max_mano_replay_error_m=replay,
                                                  max_finger_increment_rad=float(np.linalg.norm(dp, axis=-1).max()), max_root_increment_rad=float(np.linalg.norm(do, axis=-1).max()))
        dataset = a.result / variant / 'export/merged'
        summary = json.loads((dataset / 'summary.json').read_text())
        assert summary['approved_for_training'] is False
        rows = [json.loads(r) for r in (dataset / 'index.jsonl').read_text().splitlines()]
        assert len(rows) == summary['label_count']
        groups = {}
        for row in rows:
            assert row['sdk_direction_enabled'] == (variant == 'direction')
            assert row['sdk_joint_positions_used_as_targets'] is False
            assert row['approved_for_training'] is False
            assert (dataset / row['label']).is_file() and (dataset / row['image']).is_file()
            groups.setdefault((row['sync_index'], row['camera']), set()).add(row['side'])
        assert all(sides == {0, 1} for sides in groups.values())
        report[variant + '_export'] = dict(images=summary['image_count'], labels=summary['label_count'], all_exported_images_have_both_hands=True)
    (a.result / 'verification.json').write_text(json.dumps(report, indent=2)); print(json.dumps(report, indent=2))


if __name__ == '__main__': main()
