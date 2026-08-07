from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from ego_data.calibration import CameraCalibration, StereoCalibration, quaternion_transform
from ego_data.pairing import pair_timestamps_ns, pairing_statistics
from ego_data.genrobot_mcap import h264_nal_types


class GenrobotDataTests(unittest.TestCase):
    def test_quaternion_transform_and_relative_extrinsics(self):
        left_transform = quaternion_transform([0.03, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        right_transform = quaternion_transform([-0.03, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        K = np.asarray([[500.0, 0.0, 800.0], [0.0, 500.0, 650.0], [0.0, 0.0, 1.0]])
        D = np.asarray([500.0, 500.0, 800.0, 650.0, 0.0, 0.57])
        left = CameraCalibration("camera2", "left", "DS", (1600, 1300), K, D, left_transform)
        right = CameraCalibration("camera3", "right", "DS", (1600, 1300), K, D, right_transform)
        stereo = StereoCalibration.from_cameras(left, right)
        np.testing.assert_allclose(stereo.R_left_to_right, np.eye(3), atol=1e-12)
        np.testing.assert_allclose(stereo.t_left_to_right_m, [0.06, 0.0, 0.0], atol=1e-12)
        self.assertAlmostEqual(stereo.baseline_m, 0.06)

    def test_nanosecond_pairing_contract(self):
        left = [1_000_000, 2_000_000, 3_000_000]
        right = [1_019_000, 2_021_000, 4_000_000]
        pairs = pair_timestamps_ns(left, right, 30_000)
        self.assertEqual([(p.left_frame_index, p.right_frame_index) for p in pairs], [(0, 0), (1, 1)])
        stats = pairing_statistics(len(left), len(right), pairs)
        self.assertEqual(stats["pair_count"], 2)
        self.assertEqual(stats["abs_delta_ns_max"], 21_000)

    def test_annex_b_nal_type_detection(self):
        payload = b"\x00\x00\x00\x01\x67abc\x00\x00\x01\x68d\x00\x00\x00\x01\x65frame"
        self.assertEqual(h264_nal_types(payload), [7, 8, 5])


if __name__ == "__main__":
    unittest.main()
