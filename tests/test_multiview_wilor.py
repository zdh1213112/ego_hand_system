from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from camera_models import project_points
from ego_data.calibration import CameraCalibration
from fuse_multiview_wilor import triangulate_ransac
from fuse_multiview_wilor_guided import _match_camera, _ordered_hand_pairs
from normalize_multiview_recording import synchronize_rows


def ds_camera(camera_id: str, center_x: float) -> CameraCalibration:
    transform = np.eye(4)
    transform[0, 3] = center_x
    return CameraCalibration(
        camera_id, camera_id, "DS", (1600, 1300),
        np.asarray([[510.0, 0.0, 799.5], [0.0, 512.0, 649.5], [0.0, 0.0, 1.0]]),
        np.asarray([510.0, 512.0, 799.5, 649.5, -0.003, 0.572]),
        transform,
    )


class MultiviewWilorTests(unittest.TestCase):
    def test_synchronization_uses_each_frame_at_most_once(self):
        rows = {
            "camera0": [{"timestamp_ns": value} for value in (1000, 2000, 3000)],
            "camera1": [{"timestamp_ns": value} for value in (990, 2010, 3020)],
            "camera2": [{"timestamp_ns": value} for value in (995, 2005, 3015)],
        }
        result = synchronize_rows(rows, tuple(rows), "camera1", 25)
        self.assertEqual(len(result), 3)
        self.assertEqual([row["camera0_frame_index"] for row in result], [0, 1, 2])
        self.assertEqual([row["camera2_frame_index"] for row in result], [0, 1, 2])

    def test_native_ds_multiview_triangulation(self):
        cameras = [ds_camera(f"camera{index}", center) for index, center in enumerate((-0.09, -0.03, 0.03, 0.09))]
        expected = np.asarray([0.04, -0.025, 0.72])
        observations = []
        for camera in cameras:
            point_camera = expected - camera.T_base_camera[:3, 3]
            pixel, valid = project_points(camera, point_camera[None])
            self.assertTrue(bool(valid[0]))
            observations.append((camera.camera_id, camera, pixel[0]))
        recovered, errors, inliers = triangulate_ransac(observations, 0.01)
        np.testing.assert_allclose(recovered, expected, atol=1e-10)
        self.assertTrue(inliers.all())
        self.assertLess(float(errors.max()), 1e-9)

    def test_guided_matching_rejects_background_candidate(self):
        camera = ds_camera("camera0", -0.09)
        left_points = np.tile(np.asarray([-0.06, -0.02, 0.70]), (21, 1))
        right_points = np.tile(np.asarray([0.08, -0.01, 0.72]), (21, 1))
        left_pixels, _ = project_points(
            camera, left_points - camera.T_base_camera[:3, 3]
        )
        right_pixels, _ = project_points(
            camera, right_points - camera.T_base_camera[:3, 3]
        )
        false_pixels = left_pixels + np.asarray([350.0, -220.0])
        groups = {
            0: {
                0: {"detection_index": 0, "confidence": 0.8, "detector_is_right": 0, "joints_2d": left_pixels.tolist()},
                1: {"detection_index": 0, "confidence": 0.8, "detector_is_right": 0, "joints_2d": false_pixels.tolist()},
            },
            1: {
                0: {"detection_index": 1, "confidence": 0.7, "detector_is_right": 1, "joints_2d": false_pixels.tolist()},
                1: {"detection_index": 1, "confidence": 0.7, "detector_is_right": 1, "joints_2d": right_pixels.tolist()},
            },
        }
        selected, errors = _match_camera(
            "camera0", groups, {0: {"points": left_points}, 1: {"points": right_points}},
            camera, 55.0, 4, "strict",
        )
        self.assertEqual(selected[0]["detection_index"], 0)
        self.assertEqual(selected[1]["detection_index"], 1)
        self.assertLess(errors[0], 1e-9)
        self.assertLess(errors[1], 1e-9)
        self.assertEqual(_ordered_hand_pairs([0, 1], groups, "strict"), [(0, 1)])


if __name__ == "__main__":
    unittest.main()
