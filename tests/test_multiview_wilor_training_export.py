from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from check_wilor_training_dataset import _contract, _replay_mano, _signature
from export_multiview_wilor_training_dataset import (
    _full_pose_matrices, _limit_to_sync_groups, _mesh_bbox, _project,
    _rectify_mano_geometry, _select_sync_indices,
)
from prepare_multiview_mano_input import _project_rectified
from mano_conventions import (
    MIRROR_X,
    canonical_projection_rotation,
    canonical_rectification_rotation,
    mirror_left_points,
    physicalize_geometry,
)


class MultiviewWilorTrainingExportTests(unittest.TestCase):
    @staticmethod
    def _sampling_pending(sync_indices, roots):
        pending = []
        for sync_index, root in zip(sync_indices, roots):
            points = np.zeros((21, 3), dtype=np.float32)
            points[0] = np.asarray(root, dtype=np.float32)
            for camera in ("camera2", "camera3"):
                pending.append({
                    "sync_index": sync_index,
                    "side": 0,
                    "camera": camera,
                    "motion_points": points.copy(),
                })
        return pending

    def test_fixed_stride_uses_sync_index_and_keeps_final_frame(self):
        sync_indices = [0, 1, 3, 4, 6, 7]
        pending = self._sampling_pending(sync_indices, [(0, 0, 0)] * len(sync_indices))
        rows = {
            index: {"reference_timestamp_ns": str(index * 33_333_333)}
            for index in sync_indices
        }
        selected, summary = _select_sync_indices(pending, rows, 3)
        self.assertEqual(selected, [0, 3, 6, 7])
        self.assertEqual(summary["mode"], "fixed_stride")

    def test_motion_adaptive_sampling_keeps_motion_and_reduces_static_frames(self):
        sync_indices = list(range(13))
        roots = []
        for index in sync_indices:
            if index < 3:
                roots.append((0.0, 0.0, 0.0))
            elif index < 6:
                roots.append((0.01, 0.0, 0.0))
            else:
                roots.append((0.02, 0.0, 0.0))
        pending = self._sampling_pending(sync_indices, roots)
        rows = {
            index: {"reference_timestamp_ns": str(index * 33_333_333)}
            for index in sync_indices
        }
        selected, summary = _select_sync_indices(pending, rows, 0)
        self.assertEqual(selected, [0, 3, 6, 12])
        self.assertEqual(summary["mode"], "motion_adaptive")
        self.assertGreater(summary["selected_by_motion"], 0)

    def test_sample_cap_does_not_split_sync_group(self):
        pending = self._sampling_pending([0, 3], [(0, 0, 0), (0.01, 0, 0)])
        limited = _limit_to_sync_groups(pending, [0, 3], 3)
        self.assertEqual({item["sync_index"] for item in limited}, {0})
        self.assertEqual(len(limited), 2)

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

    def test_left_export_restores_physical_rectified_geometry(self):
        canonical_vertices = np.asarray(
            [[0.03, 0.01, 0.0], [0.05, -0.02, 0.01]], np.float32
        )
        canonical_joints = np.asarray(
            [[0.02, 0.01, 0.0], [0.04, 0.03, 0.0]], np.float32
        )
        canonical_translation = np.asarray([0.1, -0.04, 0.7], np.float32)
        physical_rotation = np.asarray(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            np.float32,
        )
        canonical_rotation = canonical_rectification_rotation(
            physical_rotation, "Left"
        )
        vertices, joints, translation, _ = _rectify_mano_geometry(
            canonical_vertices, canonical_joints,
            canonical_translation, canonical_rotation,
        )
        physical_vertices = mirror_left_points(vertices, "Left")
        physical_joints = mirror_left_points(joints, "Left")
        physical_translation = mirror_left_points(translation, "Left")
        np.testing.assert_allclose(
            physical_vertices + physical_translation,
            (physical_rotation @ (
                mirror_left_points(canonical_vertices, "Left")
                + mirror_left_points(canonical_translation, "Left")
            ).T).T,
            atol=1e-7,
        )
        np.testing.assert_allclose(
            physical_joints + physical_translation,
            (physical_rotation @ (
                mirror_left_points(canonical_joints, "Left")
                + mirror_left_points(canonical_translation, "Left")
            ).T).T,
            atol=1e-7,
        )

    def test_exported_full_pose_replays_rectified_geometry(self):
        import torch
        from scipy.spatial.transform import Rotation
        from fit_mano_sequence import import_mano, run_model

        root = SCRIPTS.parent
        source = root / "third_party" / "MANO"
        model_dir = root / "models" / "mano"
        if (not (source / "mano" / "model.py").is_file()
                or not (model_dir / "MANO_RIGHT.pkl").is_file()):
            self.skipTest("external MANO source/licensed assets are not installed")
        mano = import_mano(source)
        rotation = Rotation.from_rotvec([0.12, -0.08, 0.2]).as_matrix().astype(np.float32)
        records = []
        for side in (0, 1):
            model = mano.load(
                str(model_dir), is_rhand=True, use_pca=False,
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
            side_rotation = canonical_rectification_rotation(rotation, side)
            rectified_vertices, rectified_joints, _, _ = _rectify_mano_geometry(
                vertices[0].numpy(), joints[0].numpy(),
                np.zeros(3, np.float32), side_rotation,
            )
            global_orient, hand_pose = _full_pose_matrices(
                output.full_pose[0].reshape(16, 3).numpy(), side_rotation
            )
            records.append((Path(f"side_{side}.npy"), {
                "side": np.asarray(float(side), np.float32),
                "vertices": mirror_left_points(rectified_vertices, side),
                "joints_3d": mirror_left_points(rectified_joints, side),
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
            {"left": "MANO_RIGHT.pkl", "right": "MANO_RIGHT.pkl"},
            tolerance_m=1e-5,
        )
        self.assertLess(vertex_error, 1e-6)
        self.assertLess(joint_error, 1e-6)

        with self.assertRaisesRegex(ValueError, "MANO_RIGHT replay requires"):
            _replay_mano(
                records, source, model_dir,
                {"left": "MANO_LEFT.pkl", "right": "MANO_RIGHT.pkl"},
                tolerance_m=1e-5,
            )

    def test_left_canonical_projection_preserves_physical_pixels(self):
        rotation = np.asarray(
            [[0.98, -0.1, 0.17], [0.12, 0.99, -0.03], [-0.16, 0.05, 0.98]],
            dtype=np.float32,
        )
        physical = np.asarray([[0.08, -0.03, 0.55], [-0.04, 0.06, 0.72]], np.float32)
        canonical = mirror_left_points(physical, "Left")
        effective = canonical_projection_rotation(rotation, "Left")
        np.testing.assert_allclose(
            (effective @ canonical.T).T, (rotation @ physical.T).T, atol=1e-7
        )

    def test_physical_visualization_roundtrip_and_winding(self):
        canonical_vertices = np.asarray([[0.1, 0.2, 0.3], [-0.2, 0.1, 0.4]], np.float32)
        canonical_joints = np.asarray([[0.03, -0.04, 0.5]], np.float32)
        faces = np.asarray([[1, 4, 7]], np.int32)
        vertices, joints, physical_faces = physicalize_geometry(
            canonical_vertices, canonical_joints, faces, "Left"
        )
        np.testing.assert_allclose(vertices, canonical_vertices @ MIRROR_X)
        np.testing.assert_allclose(joints, canonical_joints @ MIRROR_X)
        np.testing.assert_array_equal(physical_faces, [[1, 7, 4]])

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
