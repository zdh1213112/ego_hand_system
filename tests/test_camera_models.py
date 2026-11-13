from __future__ import annotations

from pathlib import Path
import sys
import unittest

import cv2
import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from camera_models import RectificationOptions, create_stereo_rectification, project_points, unproject_points
from camera_models.double_sphere import unproject as unproject_double_sphere
from camera_models.kb import theta_limit
from ego_data.calibration import CameraCalibration, StereoCalibration
from ego_data.calibration import stereo_from_ego_yaml


class CameraModelTests(unittest.TestCase):
    def test_kb_spherical_projection_crosses_90_degrees_without_folding(self):
        K = np.array([[200., 0, 500.], [0, 210., 400.], [0, 0, 1.]])
        camera = CameraCalibration('wide', 'optical', 'KB', (1000, 800), K, np.zeros(4), np.eye(4))
        angles = np.radians([0., 85., 90., 92., 99., 120.])
        points = np.column_stack((np.sin(angles), np.zeros(len(angles)), np.cos(angles)))
        pixels, valid = project_points(camera, points)
        self.assertTrue(valid.all())
        np.testing.assert_allclose(pixels[:, 0], 500. + 200. * angles, atol=1e-10)
        np.testing.assert_allclose(pixels[:, 1], 400., atol=1e-10)
        rays, valid = unproject_points(camera, pixels)
        self.assertTrue(valid.all())
        np.testing.assert_allclose(rays, points, atol=1e-10)
        self.assertTrue((rays[3:, 2] < 0).all())

    def test_kb_front_hemisphere_matches_opencv(self):
        K = np.array([[505., 0, 944.], [0, 504., 590.], [0, 0, 1.]])
        D = np.array([.081, -.023, .0069, -.00083])
        camera = CameraCalibration('wide', 'optical', 'KB', (1920, 1200), K, D, np.eye(4))
        points = np.array([[0, 0, 1], [.2, -.3, .8], [1., .8, .001]])
        expected, _ = cv2.fisheye.projectPoints(points[:, None], np.zeros(3), np.zeros(3), K, D)
        actual, valid = project_points(camera, points)
        self.assertTrue(valid.all())
        np.testing.assert_allclose(actual, expected[:, 0], atol=1e-9)

    def test_kb_distorted_spherical_roundtrip_for_mixed_angles(self):
        camera = CameraCalibration('wide', 'optical', 'KB', (1920, 1200),
            [[505., 0, 944.], [0, 504., 590.], [0, 0, 1.]],
            [.081, -.023, .0069, -.00083], np.eye(4))
        rng = np.random.default_rng(42)
        angles = np.concatenate(([0., 1e-9], rng.uniform(0., min(theta_limit(camera) - .01, 2.), 1000)))
        azimuth = rng.uniform(-np.pi, np.pi, len(angles))
        points = np.column_stack((np.sin(angles) * np.cos(azimuth),
                                  np.sin(angles) * np.sin(azimuth), np.cos(angles)))
        pixels, valid = project_points(camera, points)
        self.assertTrue(valid.all())
        rays, valid = unproject_points(camera, pixels)
        self.assertTrue(valid.all())
        np.testing.assert_allclose(rays, points, atol=1e-10)

    def test_kb_rejects_ambiguous_domain_and_invalid_points(self):
        K = np.array([[100., 0, 200.], [0, 100., 200.], [0, 0, 1.]])
        # rho(theta) = theta - theta^3/3 folds beyond theta = 1 rad.
        camera = CameraCalibration('fold', 'optical', 'KB', (400, 400), K,
                                   [-1 / 3, 0, 0, 0], np.eye(4))
        self.assertAlmostEqual(theta_limit(camera), 1.)
        angles = np.array([.8, 1.1])
        points = np.column_stack((np.sin(angles), np.zeros(2), np.cos(angles)))
        pixels, valid = project_points(camera, points)
        np.testing.assert_array_equal(valid, [True, False])
        self.assertTrue(np.isnan(pixels[1]).all())
        rays, valid = unproject_points(camera, [pixels[0], [300., 200.], [np.nan, 0]])
        np.testing.assert_array_equal(valid, [True, False, False])
        np.testing.assert_allclose(rays[0], points[0], atol=1e-9)
        self.assertTrue(np.isnan(rays[1:]).all())
        for distortion in (np.zeros(4), [-1 / 3, 0, 0, 0]):
            camera = CameraCalibration('invalid', 'optical', 'KB', (400, 400), K, distortion, np.eye(4))
            pixels, valid = project_points(camera, [[0, 0, 0], [0, 0, -1], [np.nan, 0, 1]])
            self.assertFalse(valid.any())
            self.assertTrue(np.isnan(pixels).all())

    def test_pinhole_still_rejects_nonpositive_depth(self):
        camera = CameraCalibration('pinhole', 'optical', 'PINHOLE', (100, 100),
                                   [[50, 0, 50], [0, 50, 50], [0, 0, 1]], [], np.eye(4))
        _, valid = project_points(camera, [[1., 0, -.1], [1., 0, 0], [1., 0, .1]])
        np.testing.assert_array_equal(valid, [False, False, True])

    def test_kb_unprojection_roundtrip_and_invalid_pixels(self):
        K = np.asarray([[505.0, 0.0, 800.0], [0.0, 505.0, 600.0], [0.0, 0.0, 1.0]])
        D = np.asarray([0.076, -0.015, 0.0027, -0.00009])
        camera = CameraCalibration("camera0", "left", "KB", (1600, 1200), K, D, np.eye(4))
        points = np.asarray([[0.0, 0.0, 1.0], [0.4, -0.2, 0.8], [-0.5, 0.3, 1.2]])
        pixels, projected_valid = project_points(camera, points)
        self.assertTrue(projected_valid.all())
        pixels = np.concatenate((pixels, [[np.nan, 10.0]]))
        rays, valid = unproject_points(camera, pixels)
        self.assertEqual(rays.shape, (4, 3))
        np.testing.assert_array_equal(valid, [True, True, True, False])
        np.testing.assert_allclose(rays[:3], points / np.linalg.norm(points, axis=1)[:, None], atol=1e-6)
        self.assertTrue(np.isnan(rays[3]).all())

    def test_ds_unprojection_dispatch_keeps_existing_values(self):
        K = np.asarray([[200.0, 0.0, 160.0], [0.0, 200.0, 120.0], [0.0, 0.0, 1.0]])
        camera = CameraCalibration("camera0", "ds", "DS", (320, 240), K,
                                   np.asarray([200.0, 200.0, 160.0, 120.0, 0.5, 0.5]), np.eye(4))
        pixels = np.asarray([[160.0, 120.0], [175.0, 130.0]])
        expected, expected_valid = unproject_double_sphere(camera, pixels)
        actual, actual_valid = unproject_points(camera, pixels)
        np.testing.assert_array_equal(actual_valid, expected_valid)
        np.testing.assert_array_equal(actual, expected)

    def test_mano_kb_rectification_matches_opencv(self):
        from prepare_multiview_mano_input import _rectify_view_pixels
        K = np.asarray([[505.0, 0.0, 800.0], [0.0, 505.0, 600.0], [0.0, 0.0, 1.0]])
        D = np.asarray([0.076, -0.015, 0.0027, -0.00009])
        camera = CameraCalibration("camera0", "left", "KB", (1600, 1200), K, D, np.eye(4))
        rotation, _ = cv2.Rodrigues(np.asarray([0.02, -0.01, 0.005]))
        projection = np.column_stack((K, np.asarray([[-50.0], [0.0], [0.0]])))
        pixels = np.asarray([[800.0 + i * 7, 600.0 - i * 4] for i in range(21)])
        actual, valid = _rectify_view_pixels(camera, pixels, rotation, projection)
        expected = cv2.fisheye.undistortPoints(pixels.reshape(-1, 1, 2), K, D,
                                               R=rotation, P=projection[:, :3])[:, 0]
        self.assertTrue(valid.all())
        np.testing.assert_allclose(actual, expected, atol=1e-6)

    def test_kb_dispatch_matches_opencv_contract(self):
        K = np.asarray([[220.0, 0.0, 160.0], [0.0, 221.0, 120.0], [0.0, 0.0, 1.0]])
        D = np.asarray([-0.01, 0.002, -0.0002, 0.00001])
        left_transform = np.eye(4)
        right_transform = np.eye(4)
        right_transform[0, 3] = 0.06
        left = CameraCalibration("left", "left", "KB", (320, 240), K, D, left_transform)
        right = CameraCalibration("right", "right", "KB", (320, 240), K, D, right_transform)
        stereo = StereoCalibration.from_cameras(left, right)
        result = create_stereo_rectification(stereo, RectificationOptions(), "auto")
        expected = cv2.fisheye.stereoRectify(
            K, D, K, D, (320, 240), stereo.R_left_to_right,
            stereo.t_left_to_right_m, flags=cv2.CALIB_ZERO_DISPARITY,
            newImageSize=(320, 240), balance=0.0, fov_scale=1.0,
        )
        for actual, reference in zip(
            (result.R1, result.R2, result.P1, result.P2, result.Q), expected
        ):
            np.testing.assert_allclose(actual, reference, atol=1e-12)
        self.assertEqual(result.model, "KB")

    def test_explicit_model_mismatch_is_rejected(self):
        K = np.eye(3)
        K[0, 0] = K[1, 1] = 200
        left_transform = np.eye(4)
        right_transform = np.eye(4)
        right_transform[0, 3] = 0.05
        left = CameraCalibration("left", "left", "KB", (10, 10), K, np.zeros(4), left_transform)
        right = CameraCalibration("right", "right", "KB", (10, 10), K, np.zeros(4), right_transform)
        with self.assertRaisesRegex(ValueError, "requested DS"):
            create_stereo_rectification(StereoCalibration.from_cameras(left, right), model="ds")

    def test_ego_yaml_reference_to_camera_extrinsics_are_preserved(self):
        import tempfile
        import yaml
        left_rotation, _ = cv2.Rodrigues(np.asarray([0.01, -0.02, 0.005]))
        right_rotation, _ = cv2.Rodrigues(np.asarray([-0.015, 0.03, -0.004]))
        left_translation_mm = np.asarray([3.0, -2.0, 1.0])
        right_translation_mm = np.asarray([-57.0, -1.0, 2.0])
        cameras = []
        for index, (rotation, translation) in enumerate((
            (left_rotation, left_translation_mm), (right_rotation, right_translation_mm)
        )):
            cameras.append({
                "id": f"cam_{index}", "name": f"cam_{index}", "distortion_model": "KB",
                "image_width": 320, "image_height": 240,
                "intrinsics": {"fx": 200.0, "fy": 201.0, "cx": 160.0, "cy": 120.0},
                "distortion": {"k1": 0.0, "k2": 0.0, "k3": 0.0, "k4": 0.0},
                "extrinsics": {"rotation": rotation.tolist(), "translation": translation.tolist()},
            })
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calibration.yaml"
            path.write_text(yaml.safe_dump({"cameras": cameras}), encoding="utf-8")
            stereo = stereo_from_ego_yaml(path)
        expected_rotation = right_rotation @ left_rotation.T
        expected_translation = right_translation_mm * 1e-3 - expected_rotation @ (left_translation_mm * 1e-3)
        np.testing.assert_allclose(stereo.R_left_to_right, expected_rotation, atol=1e-12)
        np.testing.assert_allclose(stereo.t_left_to_right_m, expected_translation, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
