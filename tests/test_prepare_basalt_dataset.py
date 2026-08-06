#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import numpy as np
import yaml


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_basalt_dataset.py"
SPEC = importlib.util.spec_from_file_location("prepare_basalt_dataset", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)

SESSION = Path(
    "/home/zdh/ego_hand_system/recordings/"
    "Orbbec_Ego_AZER764008C_20260805_171119"
)


def pose_to_transform(pose: dict) -> np.ndarray:
    x, y, z, w = pose["qx"], pose["qy"], pose["qz"], pose["qw"]
    rotation = np.asarray([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    return MODULE.make_transform(rotation, (pose["px"], pose["py"], pose["pz"]))


class PrepareBasaltDatasetTests(unittest.TestCase):
    def test_timestamp_pairing_skips_unmatched_prefix(self):
        left = [100, 200, 300, 400]
        right = [295, 405]
        self.assertEqual(MODULE.pair_timestamps(left, right, 10), [(2, 0), (3, 1)])

    def test_transform_inverse_round_trip(self):
        rotation, _ = __import__("cv2").Rodrigues(np.asarray((0.1, -0.2, 0.05)))
        transform = MODULE.make_transform(rotation, (0.2, -0.1, 0.4))
        np.testing.assert_allclose(
            transform @ MODULE.invert_transform(transform), np.eye(4), atol=1e-12
        )

    def test_quaternion_round_trip(self):
        rotation, _ = __import__("cv2").Rodrigues(np.asarray((-0.3, 0.1, 0.4)))
        pose = MODULE.basalt_pose(MODULE.make_transform(rotation, (0.1, 0.2, 0.3)))
        reconstructed = pose_to_transform(pose)
        np.testing.assert_allclose(reconstructed[:3, :3], rotation, atol=1e-12)
        np.testing.assert_allclose(reconstructed[:3, 3], (0.1, 0.2, 0.3), atol=1e-12)

    def test_ego_calibration_maps_to_kb4_and_preserves_baseline(self):
        with MODULE.unique_file(SESSION, "_calibration_camera.yaml").open() as stream:
            cameras = yaml.safe_load(stream)
        with MODULE.unique_file(SESSION, "_calibration_imu.yaml").open() as stream:
            imu = yaml.safe_load(stream)
        calibration, metadata = MODULE.build_basalt_calibration(cameras, imu, 0.5, True)
        value = calibration["value0"]
        self.assertEqual([entry["camera_type"] for entry in value["intrinsics"]], ["kb4", "kb4"])
        self.assertEqual(value["resolution"], [[800, 650], [800, 650]])
        self.assertAlmostEqual(metadata["stereo_baseline_m"], 0.1222669808, places=7)
        t_i_c0 = pose_to_transform(value["T_imu_cam"][0])
        t_i_c1 = pose_to_transform(value["T_imu_cam"][1])
        relative = MODULE.invert_transform(t_i_c0) @ t_i_c1
        self.assertAlmostEqual(float(np.linalg.norm(relative[:3, 3])), 0.1222669808, places=7)
        self.assertEqual(value["cam_time_offset_ns"], 0)

    def test_imu_calibration_uses_full_vendor_matrices(self):
        imu0 = {
            "M_acc": [[2.0, 0.1, 0.0], [0.0, 3.0, 0.2], [0.0, 0.0, 4.0]],
            "M_gyr": [[1.0, 0.0, 0.0], [0.1, 2.0, 0.0], [0.2, 0.3, 3.0]],
            "AccBias": [0.5, 0.6, 0.7],
            "GyrBias": [0.1, 0.2, 0.3],
        }
        accel, gyro = MODULE.calibrate_imu_sample(
            np.asarray((1.0, 2.0, 3.0)), np.asarray((0.5, 1.0, 1.5)), imu0
        )
        np.testing.assert_allclose(accel, np.asarray(imu0["M_acc"]) @ (1.0, 2.0, 3.0) - imu0["AccBias"])
        np.testing.assert_allclose(gyro, np.asarray(imu0["M_gyr"]) @ (0.5, 1.0, 1.5) - imu0["GyrBias"])


if __name__ == "__main__":
    unittest.main()
