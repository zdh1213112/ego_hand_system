from __future__ import annotations

from pathlib import Path
import sys
import unittest

import cv2
import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from camera_models import RectificationOptions, create_stereo_rectification, project_points
from camera_models.double_sphere import unproject
from ego_data.calibration import CameraCalibration, StereoCalibration


def ds_camera(camera_id: str, tx: float) -> CameraCalibration:
    transform = np.eye(4)
    transform[0, 3] = tx
    return CameraCalibration(
        camera_id, camera_id, "DS", (320, 240),
        np.asarray([[180.0, 0.0, 160.0], [0.0, 182.0, 120.0], [0.0, 0.0, 1.0]]),
        np.asarray([180.0, 182.0, 160.0, 120.0, -0.003, 0.572]),
        transform,
    )


class DoubleSphereTests(unittest.TestCase):
    def test_projection_matches_gen_reference_formula(self):
        camera = ds_camera("camera2", 0.0)
        points = np.asarray([[0.2, -0.1, 1.2], [-0.1, 0.3, 0.8]])
        pixels, valid = project_points(camera, points)
        x, y, z = points.T
        d1 = np.linalg.norm(points, axis=1)
        alpha = camera.distortion[5]
        xi = camera.distortion[4]
        z1 = alpha * d1 + (1.0 - alpha) * z
        d2 = np.sqrt(x * x + y * y + z1 * z1)
        expected = np.column_stack((
            camera.distortion[0] * x / (xi * d2 + z1) + camera.distortion[2],
            camera.distortion[1] * y / (xi * d2 + z1) + camera.distortion[3],
        ))
        self.assertTrue(valid.all())
        np.testing.assert_allclose(pixels, expected, atol=1e-12)

    def test_unprojection_round_trip(self):
        camera = ds_camera("camera2", 0.0)
        points = np.asarray([
            [0.2, -0.1, 1.2], [-0.1, 0.3, 0.8], [0.45, 0.25, 0.7],
        ])
        pixels, projected = project_points(camera, points)
        rays, unprojected = unproject(camera, pixels)
        expected = points / np.linalg.norm(points, axis=1, keepdims=True)
        self.assertTrue(projected.all())
        self.assertTrue(unprojected.all())
        np.testing.assert_allclose(rays, expected, atol=1e-10)

    def test_rectification_produces_positive_disparity_and_metric_3d(self):
        left = ds_camera("camera2", 0.0)
        right = ds_camera("camera3", 0.06)
        stereo = StereoCalibration.from_cameras(left, right)
        result = create_stereo_rectification(
            stereo, RectificationOptions(output_size=(320, 240)), "ds"
        )
        points_left = np.asarray([
            [-0.10, -0.05, 0.5], [0.0, 0.0, 0.8], [0.15, 0.08, 1.2],
        ])
        points_rectified = (result.R1 @ points_left.T).T
        homogeneous = np.column_stack((points_rectified, np.ones(len(points_rectified))))
        left_px_h = homogeneous @ result.P1.T
        right_px_h = homogeneous @ result.P2.T
        left_px = left_px_h[:, :2] / left_px_h[:, 2:]
        right_px = right_px_h[:, :2] / right_px_h[:, 2:]
        self.assertTrue(np.all(left_px[:, 0] > right_px[:, 0]))
        np.testing.assert_allclose(left_px[:, 1], right_px[:, 1], atol=1e-12)
        triangulated = cv2.triangulatePoints(result.P1, result.P2, left_px.T, right_px.T)
        triangulated = (triangulated[:3] / triangulated[3]).T
        recovered_left = (result.R1.T @ triangulated.T).T
        self.assertLess(float(np.max(np.linalg.norm(recovered_left - points_left, axis=1))), 1e-9)
        self.assertEqual(result.map_left_x.shape, (240, 320))


if __name__ == "__main__":
    unittest.main()
