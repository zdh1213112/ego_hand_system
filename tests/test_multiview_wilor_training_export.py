from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from check_wilor_training_dataset import _contract, _signature
from export_multiview_wilor_training_dataset import _mesh_bbox, _project
from prepare_multiview_mano_input import _project_rectified


class MultiviewWilorTrainingExportTests(unittest.TestCase):
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
