from __future__ import annotations

from pathlib import Path
import sys
import unittest

import cv2
import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from camera_models import RectificationOptions, create_stereo_rectification
from ego_data.calibration import CameraCalibration, StereoCalibration
from ego_data.calibration import stereo_from_ego_yaml


class CameraModelTests(unittest.TestCase):
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
