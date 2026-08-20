from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from check_wilor_training_dataset import _contract, _replay_mano, _signature
from export_multiview_wilor_training_dataset import (
    _full_pose_matrices, _mesh_bbox, _project, _rectify_mano_geometry,
)
from prepare_multiview_mano_input import _project_rectified


class MultiviewWilorTrainingExportTests(unittest.TestCase):
    def test_rectification_moves_root_offset_into_translation(self):
        vertices = np.asarray([[0.03, 0.01, 0.0], [0.05, -0.02, 0.01]], np.float32)
        joints = np.asarray([[0.02, 0.01, 0.0], [0.04, 0.03, 0.0]], np.float32)
        translation = np.asarray([0.1, -0.04, 0.7], np.float32)
        rotation = np.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], np.float32)
        rectified_vertices, rectified_joints, rectified_translation, offset = (
            _rectify_mano_geometry(vertices, joints, translation, rotation)
        )
        np.testing.assert_allclose(
            rectified_vertices + rectified_translation,
            (rotation @ (vertices + translation).T).T,
            atol=1e-7,
        )
        np.testing.assert_allclose(
            rectified_joints + rectified_translation,
            (rotation @ (joints + translation).T).T,
            atol=1e-7,
        )
        np.testing.assert_allclose(offset, rotation @ joints[0] - joints[0])

    def test_full_pose_matrices_use_mean_inclusive_hand_pose(self):
        full_pose = np.zeros((16, 3), dtype=np.float32)
        full_pose[0, 2] = np.pi / 4
        full_pose[3, 0] = -0.25
        rectification = np.asarray(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        global_orient, hand_pose = _full_pose_matrices(full_pose, rectification)
        expected_global = rectification @ np.asarray([
            [np.sqrt(0.5), -np.sqrt(0.5), 0.0],
            [np.sqrt(0.5), np.sqrt(0.5), 0.0],
            [0.0, 0.0, 1.0],
        ])
        np.testing.assert_allclose(global_orient[0], expected_global, atol=1e-6)
        self.assertFalse(np.allclose(hand_pose[2], np.eye(3)))

    def test_exported_full_pose_replays_rectified_geometry(self):
        import torch
        from scipy.spatial.transform import Rotation
        from fit_mano_sequence import import_mano, run_model

        root = SCRIPTS.parent
        source = root / "third_party" / "MANO"
        model_dir = root / "models" / "mano"
        if (not (source / "mano" / "model.py").is_file()
                or not all((model_dir / name).is_file()
                           for name in ("MANO_LEFT.pkl", "MANO_RIGHT.pkl"))):
            self.skipTest("external MANO source/licensed assets are not installed")
        mano = import_mano(source)
        rotation = Rotation.from_rotvec([0.12, -0.08, 0.2]).as_matrix().astype(np.float32)
        records = []
        for side in (0, 1):
            model = mano.load(
                str(model_dir), is_rhand=bool(side), use_pca=False,
                num_pca_comps=45, batch_size=1, flat_hand_mean=False,
            ).eval()
            with torch.no_grad():
                vertices, joints, output = run_model(
                    model,
                    torch.zeros((1, 10)),
                    torch.zeros((1, 3)),
                    torch.zeros((1, 45)),
                    torch.zeros((1, 3)),
                )
            rectified_vertices, rectified_joints, _, _ = _rectify_mano_geometry(
                vertices[0].numpy(), joints[0].numpy(), np.zeros(3, np.float32), rotation
            )
            global_orient, hand_pose = _full_pose_matrices(
                output.full_pose[0].reshape(16, 3).numpy(), rotation
            )
            records.append((Path(f"side_{side}.npy"), {
                "side": np.asarray(float(side), np.float32),
                "vertices": rectified_vertices,
                "joints_3d": rectified_joints,
                "mano": {
                    "global_orient": global_orient,
                    "hand_pose": hand_pose,
                    "betas": np.zeros(10, np.float32),
                },
            }))
        vertex_error, joint_error = _replay_mano(
            records,
            source,
            model_dir,
            {"left": "MANO_LEFT.pkl", "right": "MANO_RIGHT.pkl"},
            tolerance_m=1e-5,
        )
        self.assertLess(vertex_error, 1e-6)
        self.assertLess(joint_error, 1e-6)

        # The production history is irrelevant: declaring a right-hand model
        # for a physical left hand makes the checker compare against mirrored
        # label geometry automatically.
        right_canonical = records[1][1]
        mirrored_left = {
            "side": np.asarray(0.0, np.float32),
            "vertices": right_canonical["vertices"].copy(),
            "joints_3d": right_canonical["joints_3d"].copy(),
            "mano": right_canonical["mano"],
        }
        mirrored_left["vertices"][:, 0] *= -1.0
        mirrored_left["joints_3d"][:, 0] *= -1.0
        vertex_error, joint_error = _replay_mano(
            [(Path("left_with_right_model.npy"), mirrored_left)],
            source,
            model_dir,
            {"left": "MANO_RIGHT.pkl", "right": "MANO_RIGHT.pkl"},
            tolerance_m=1e-5,
        )
        self.assertLess(vertex_error, 1e-6)
        self.assertLess(joint_error, 1e-6)

    def test_rectified_projection_and_reference_projection_are_consistent(self):
        points = np.asarray([
            [-0.05, -0.03, 0.50], [0.04, 0.02, 0.60], [0.00, 0.08, 0.70],
        ])
        rotation = np.eye(3)
        K = np.asarray([[500.0, 0.0, 800.0], [0.0, 500.0, 650.0], [0.0, 0.0, 1.0]])
        projection = np.column_stack((K, np.zeros(3)))
        pixels, valid = _project_rectified(points, rotation, projection)
        self.assertTrue(valid.all())
        local = points - np.asarray([0.0, 0.0, 0.55])
        reference_pixels = _project(local.astype(np.float32), np.asarray([0.0, 0.0, 0.55], np.float32), K.astype(np.float32))
        np.testing.assert_allclose(reference_pixels, pixels, atol=1e-4)

    def test_mesh_bbox_is_clipped_and_has_margin(self):
        projected = np.column_stack((
            np.linspace(100.0, 200.0, 778), np.linspace(300.0, 500.0, 778)
        )).astype(np.float32)
        bbox = _mesh_bbox(projected, (1600, 1300), 0.1, 0.5)
        np.testing.assert_allclose(bbox, [90.0, 280.0, 210.0, 520.0], atol=1e-5)

    def test_export_contract_matches_000865_types_and_shapes(self):
        import torch

        sample = {
            "bbox": np.zeros(4, dtype=np.float64),
            "vertices": np.zeros((778, 3), dtype=np.float32),
            "joints_3d": np.zeros((21, 3), dtype=np.float32),
            "joints_2d": torch.zeros((778, 2), dtype=torch.float32),
            "side": np.asarray(0.0, dtype=np.float32),
            "trans": np.zeros(3, dtype=np.float32),
            "K": np.eye(3, dtype=np.float32),
            "mano": {
                "global_orient": np.eye(3, dtype=np.float32)[None],
                "hand_pose": np.repeat(np.eye(3, dtype=np.float32)[None], 15, axis=0),
                "betas": np.zeros(10, dtype=np.float32),
            },
        }
        _contract(sample, torch)
        self.assertEqual(_signature(sample)["joints_2d"], ("torch.float32", (778, 2)))


if __name__ == "__main__":
    unittest.main()
