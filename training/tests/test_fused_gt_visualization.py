import unittest

import numpy as np

from visualize_fused_gt import (
    base_to_camera, partial_scene_bounds, project_double_sphere, root_relative_hands,
)


class FusedGtVisualizationTests(unittest.TestCase):
    def setUp(self):
        self.calibration = {
            "camera_id": "camera2", "model": "DS",
            "distortion": [100, 100, 50, 40, 0, 0],
            "T_base_camera": [
                [1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3], [0, 0, 0, 1]
            ],
        }

    def test_base_to_camera_applies_inverse_extrinsics(self):
        points = np.tile([1, 2, 4], (21, 1))
        np.testing.assert_allclose(base_to_camera(points, self.calibration), np.tile([0, 0, 1], (21, 1)))

    def test_double_sphere_projection_preserves_principal_point(self):
        points = np.array([[0, 0, 1], [0.1, -0.2, 1]])
        pixels, valid = project_double_sphere(points, self.calibration)
        np.testing.assert_array_equal(valid, [True, True])
        np.testing.assert_allclose(pixels[0], [50, 40])
        np.testing.assert_allclose(pixels[1], [60, 20])

    def test_partial_joints_are_kept_and_bounds_ignore_nan(self):
        points = np.tile([1, 2, 4], (21, 1)).astype(float)
        points[5] = np.nan
        result = {"hands": [{"side": 0, "joints_base_m": points.tolist()}]}
        hands = root_relative_hands(result, self.calibration)
        self.assertEqual(len(hands), 1)
        np.testing.assert_allclose(hands[0][1][0], [-0.14, 0, 0])
        self.assertTrue(np.isnan(hands[0][1][5]).all())
        center, radius = partial_scene_bounds([hands])
        self.assertTrue(np.isfinite(center).all())
        self.assertEqual(radius, 0.22)


if __name__ == "__main__":
    unittest.main()
