"""Finger angles and metric 3D mapping used by the live display."""

import unittest

import numpy as np

from ego_data.calibration import CameraCalibration
from ego_live_wilor_preview import (
    _display_joints, _draw_mano_mesh, fit_mano_mesh_to_stereo,
    geometric_bend_degrees, project_metric_scene, reproject_metric_joints,
)


class BendAngleTests(unittest.TestCase):
    @staticmethod
    def identity_camera():
        return CameraCalibration(
            camera_id="cam_0", frame_id="cam_0", model="KB",
            image_size=(640, 480),
            K=np.asarray(((500, 0, 320), (0, 500, 240), (0, 0, 1))),
            distortion=np.zeros(4), T_base_camera=np.eye(4),
        )

    def test_all_panels_select_the_same_display_pose(self):
        raw = np.zeros((21, 3)).tolist()
        displayed = np.ones((21, 3)).tolist()
        self.assertEqual(_display_joints({"joints_3d_base_m": raw,
                                          "joints_3d_base_m_display": displayed}),
                         displayed)
        self.assertEqual(_display_joints({"joints_3d_base_m": raw}), raw)

    def test_straight_bent_and_missing_joints(self):
        points = np.zeros((21, 3), np.float32)
        for root in (1, 5, 9, 13, 17):
            for offset in range(4):
                points[root + offset] = (0, offset + 1, 0)
        points[7] = (0, 2, 1)  # Index PIP: incoming +Y, outgoing +Z.

        angles = geometric_bend_degrees(points)
        self.assertEqual(angles.shape, (5, 3))
        self.assertAlmostEqual(float(angles[1, 1]), 90.0, places=4)
        self.assertAlmostEqual(float(angles[2, 1]), 0.0, places=4)

        points[6] = np.nan
        missing = geometric_bend_degrees(points)
        self.assertTrue(np.isnan(missing[1, 0]))
        self.assertTrue(np.isnan(missing[1, 1]))
        self.assertTrue(np.isfinite(missing[2, 1]))

    def test_both_hands_share_one_fixed_metric_scale(self):
        points = np.asarray(((-0.1, 0.25, 0.3), (-0.05, 0.25, 0.3),
                             (0.2, 0.30, 0.3), (0.25, 0.30, 0.3)))
        pixels = project_metric_scene(points)
        np.testing.assert_allclose(pixels[1] - pixels[0],
                                   pixels[3] - pixels[2], atol=1e-4)
        self.assertGreater(np.linalg.norm(pixels[2] - pixels[0]),
                           np.linalg.norm(pixels[1] - pixels[0]))

    def test_mano_mesh_rigid_alignment_and_filled_surface(self):
        rng = np.random.default_rng(4)
        model_joints = rng.normal(0, 0.04, (21, 3))
        vertices = np.zeros((778, 3))
        vertices[:3] = ((-0.1, -0.1, 0), (0.1, -0.1, 0), (0, 0.1, 0))
        rotation = np.asarray(((0, -1, 0), (1, 0, 0), (0, 0, 1)))
        translation = np.asarray((0, 0.3, 0.3))
        stereo = model_joints @ rotation + translation
        fitted, error = fit_mano_mesh_to_stereo(
            vertices, model_joints, stereo, self.identity_camera(),
        )
        np.testing.assert_allclose(fitted[:3], vertices[:3] @ rotation + translation,
                                   atol=1e-6)
        self.assertLess(error, 1e-6)

        panel = np.zeros((440, 1400, 3), dtype=np.uint8)
        _draw_mano_mesh(
            panel, {0: {"joints_3d_base_m_display": stereo.tolist()}},
            {0: {"camera": "cam_0", "vertices": vertices, "joints": model_joints}},
            np.asarray(((0, 1, 2),)), {"cam_0": self.identity_camera()},
        )
        self.assertGreater(int(panel[280, 700].max()), 44)

    def test_mano_fit_rejects_missing_stereo_anchor(self):
        joints = np.zeros((21, 3))
        joints[0] = np.nan
        vertices, error = fit_mano_mesh_to_stereo(
            np.zeros((778, 3)), np.zeros((21, 3)), joints,
            self.identity_camera(),
        )
        self.assertIsNone(vertices)
        self.assertTrue(np.isinf(error))

    def test_stereo_joints_reproject_to_camera_without_filling_missing_points(self):
        camera = self.identity_camera()
        joints = np.full((21, 3), np.nan)
        joints[0] = (0, 0, 1)
        joints[8] = (0.1, 0, 1)
        pixels = reproject_metric_joints(joints, camera)
        np.testing.assert_allclose(pixels[0], (320, 240), atol=1e-5)
        self.assertGreater(pixels[8, 0], pixels[0, 0])
        self.assertTrue(np.isnan(pixels[1]).all())


if __name__ == "__main__":
    unittest.main()
