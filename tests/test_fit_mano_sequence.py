#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import unittest
from types import SimpleNamespace

import cv2
import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "fit_mano_sequence.py"
SPEC = importlib.util.spec_from_file_location("fit_mano_sequence", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class FitManoSequenceTests(unittest.TestCase):
    def test_joint_mapping_is_a_permutation(self):
        self.assertEqual(MODULE.MANO_TO_MEDIAPIPE.tolist(), [
            0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20,
        ])
        self.assertEqual(sorted(MODULE.MANO_TO_MEDIAPIPE.tolist()), list(range(21)))

    def test_required_model_assets_are_reported(self):
        source = Path(os.environ.get(
            "MANO_SOURCE", SCRIPT.parents[1] / "third_party" / "MANO"
        ))
        with self.assertRaisesRegex(FileNotFoundError, "MANO model data"):
            MODULE.validate_source_and_assets(
                source, Path("/definitely/missing/mano/models")
            )

    def test_official_mano_models_load_and_run(self):
        import torch
        source = Path(os.environ.get("MANO_SOURCE", SCRIPT.parents[1] / "third_party" / "MANO"))
        model_dir = SCRIPT.parents[1] / "models" / "mano"
        assets_available = all(
            (model_dir / name).is_file() for name in ("MANO_LEFT.pkl", "MANO_RIGHT.pkl")
        )
        if not assets_available or not (source / "mano" / "model.py").is_file():
            self.skipTest("external MANO source/licensed assets are not installed")
        mano = MODULE.import_mano(source)
        for is_right in (True, False):
            model = mano.load(
                str(model_dir), is_rhand=is_right, num_pca_comps=15,
                batch_size=1, flat_hand_mean=False,
            )
            output = model(return_tips=True)
            self.assertEqual(tuple(output.vertices.shape), (1, 778, 3))
            self.assertEqual(tuple(output.joints.shape), (1, 21, 3))
            self.assertTrue(torch.isfinite(output.vertices).all())

    def test_weighted_loss_ignores_nan_at_zero_weight(self):
        import torch
        residual = torch.tensor([[[0.01, 0.0, 0.0], [float("nan"), 0.0, 0.0]]])
        weights = torch.tensor([[1.0, 0.0]])
        loss = MODULE.robust_weighted_loss(residual, weights, 0.006)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(loss), 0.0)

    def test_rectified_projection(self):
        import torch
        points = torch.tensor([[[0.1, 0.0, 1.0]]])
        rotation = torch.eye(3)
        projection = torch.tensor([
            [100.0, 0.0, 50.0, 0.0],
            [0.0, 100.0, 60.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ])
        pixel = MODULE.project_rectified(points, rotation, projection)
        np.testing.assert_allclose(pixel.numpy(), [[[60.0, 60.0]]], atol=1e-6)

    def test_weighted_kabsch_recovers_rotation(self):
        source = np.asarray([
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]
        ])
        expected, _ = cv2.Rodrigues(np.asarray([0.2, -0.3, 0.1]))
        target = (expected @ source.T).T + np.asarray([0.3, -0.2, 0.7])
        recovered = MODULE.weighted_kabsch(source, target, np.ones(4))
        np.testing.assert_allclose(recovered, expected, atol=1e-7)

    def test_transition_profile_tightens_when_observation_quality_drops(self):
        frames = 3
        valid = np.ones((frames, 21), dtype=bool)
        valid[2, 4:] = False
        confidence = np.ones((frames, 21), dtype=np.float32)
        confidence[2] = 0.25
        pixels = np.zeros((frames, 21, 2), dtype=np.float32)
        pixels[1, :, 0] = 30.0
        pixels[2, :, 0] = 31.0
        profile = MODULE.observation_transition_profile(
            valid, confidence, pixels, pixels,
            max_orient_step_deg=40.0, max_translation_step_m=0.05,
        )
        self.assertGreater(profile["temporal_strength"][2], profile["temporal_strength"][1])
        self.assertLess(profile["orient_limit_rad"][2], profile["orient_limit_rad"][1])
        self.assertLess(profile["translation_limit_m"][2], profile["translation_limit_m"][1])

    def test_parameter_transition_limits_reject_large_solution_switch(self):
        import torch

        pose = torch.tensor([[0.0, 0.0], [4.0, 0.0]], dtype=torch.float32)
        orient = torch.tensor([[0.0, 0.0, 0.0], [2.5, 0.0, 0.0]], dtype=torch.float32)
        transl = torch.tensor([[0.0, 0.0, 0.2], [0.2, 0.0, 0.2]], dtype=torch.float32)
        profile = {
            "pose_limit": torch.tensor([1.0, 0.6]),
            "orient_limit_rad": torch.tensor([1.0, 0.4]),
            "translation_limit_m": torch.tensor([1.0, 0.03]),
        }
        MODULE.limit_parameter_transitions_(
            pose, orient, transl, torch.tensor([0, 1]), profile
        )
        self.assertLessEqual(float(torch.linalg.vector_norm(pose[1] - pose[0])), 0.60001)
        self.assertLessEqual(float(torch.linalg.vector_norm(orient[1] - orient[0])), 0.40001)
        self.assertLessEqual(float(torch.linalg.vector_norm(transl[1] - transl[0])), 0.03001)

    def test_image_alignment_moves_palm_without_rotating_pose(self):
        import torch

        joints = torch.zeros((3, 21, 3), dtype=torch.float32)
        joints[..., 2] = 1.0
        rotation = torch.eye(3)
        projection = torch.tensor([
            [100.0, 0.0, 50.0, 0.0],
            [0.0, 100.0, 60.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ])
        observed = MODULE.project_rectified(joints, rotation, projection)
        observed[..., 0] += 10.0
        correction = MODULE.image_space_translation_alignment(
            joints, observed, rotation, projection, torch.ones(3), maximum_m=0.2,
        )
        np.testing.assert_allclose(correction.numpy()[:, 0], 0.1, atol=1e-6)
        np.testing.assert_allclose(correction.numpy()[:, 1:], 0.0, atol=1e-6)

    def test_synthetic_differentiable_fit_reduces_joint_error(self):
        import torch

        class FakeMano(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("faces_tensor", torch.tensor([[0, 1, 2]], dtype=torch.long))
                base = torch.zeros((21, 3), dtype=torch.float32)
                base[:, 0] = torch.linspace(-0.04, 0.04, 21)
                base[:, 1] = torch.sin(torch.linspace(0, 3.0, 21)) * 0.03
                self.register_buffer("base", base)

            def forward(self, betas, global_orient, hand_pose, transl, **_):
                batch = len(global_orient)
                joints = self.base[None].expand(batch, -1, -1).clone()
                joints[:, :, 0] += 0.004 * betas[:, :1]
                joints[:, :, 1] += 0.003 * hand_pose[:, :1]
                joints += global_orient[:, None, :] * 0.01
                joints += transl[:, None, :]
                vertices = joints.repeat_interleave(2, dim=1)
                return SimpleNamespace(vertices=vertices, joints=joints)

        model = FakeMano()
        frames = 4
        known_translation = np.asarray([[0.02 + 0.002 * frame, -0.01, 0.24] for frame in range(frames)])
        with torch.no_grad():
            target_vertices, target_joints, _ = MODULE.run_model(
                model, torch.full((frames, 10), 0.5), torch.zeros((frames, 3)),
                torch.full((frames, 3), 0.4), torch.tensor(known_translation, dtype=torch.float32),
            )
        target = target_joints.numpy()
        rotation = np.eye(3, dtype=np.float32)
        p1 = np.asarray([[200, 0, 100, 0], [0, 200, 80, 0], [0, 0, 1, 0]], dtype=np.float32)
        left_px = MODULE.project_rectified(target_joints, torch.eye(3), torch.tensor(p1)).numpy()
        args = SimpleNamespace(
            pca_components=3, shape_frames=4, shape_iterations=25, pose_iterations=35,
            pose_window=32, pose_overlap=4, rigid_initialization=False,
            learning_rate=0.04, w_3d=1.0, w_2d=0.02, w_pose=0.0001,
            w_shape=0.0001, w_temporal=0.001, w_rigid_temporal=0.0001,
            w_acceleration=0.0, boundary_weight=0.0,
            max_orient_step_deg=180.0, max_translation_step_m=0.5,
            max_pose_step=10.0,
        )
        result = MODULE.optimize_track(model, {
            "positions": target, "valid": np.ones((frames, 21), dtype=bool),
            "confidence": np.ones((frames, 21), dtype=np.float32),
            "left_px": left_px, "right_px": left_px, "rotation": rotation,
            "p1": p1, "p2": p1,
        }, args, torch.device("cpu"))
        error_mm = result["joint_error_m"] * 1000.0
        self.assertLess(float(np.nanmedian(error_mm)), 5.0)


if __name__ == "__main__":
    unittest.main()
