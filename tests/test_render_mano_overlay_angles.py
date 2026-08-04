#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import cv2
import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "render_mano_overlay_angles.py"
SPEC = importlib.util.spec_from_file_location("render_mano_overlay_angles", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class RenderManoOverlayAnglesTests(unittest.TestCase):
    def test_bend_angle_is_zero_when_straight(self):
        angle = MODULE.bend_angle(
            np.asarray([0.0, 0.0, 0.0]),
            np.asarray([1.0, 0.0, 0.0]),
            np.asarray([2.0, 0.0, 0.0]),
        )
        self.assertAlmostEqual(angle, 0.0, places=7)

    def test_bend_angle_is_ninety_for_right_angle(self):
        angle = MODULE.bend_angle(
            np.asarray([0.0, 0.0, 0.0]),
            np.asarray([1.0, 0.0, 0.0]),
            np.asarray([1.0, 1.0, 0.0]),
        )
        self.assertAlmostEqual(angle, 90.0, places=7)

    def test_signed_planar_angle_preserves_direction(self):
        normal = np.asarray([0.0, 0.0, 1.0])
        reference = np.asarray([1.0, 0.0, 0.0])
        self.assertAlmostEqual(
            MODULE.signed_angle_on_plane(reference, np.asarray([0.0, 1.0, 0.0]), normal),
            90.0,
        )
        self.assertAlmostEqual(
            MODULE.signed_angle_on_plane(reference, np.asarray([0.0, -1.0, 0.0]), normal),
            -90.0,
        )

    def test_spread_angle_treats_projected_bone_as_axis(self):
        normal = np.asarray([0.0, 0.0, 1.0])
        reference = np.asarray([1.0, 0.0, 0.0])
        vector = np.asarray([-1.0, 1.0, 0.0])
        self.assertAlmostEqual(MODULE.signed_angle_on_plane(reference, vector, normal), 135.0)
        self.assertAlmostEqual(MODULE.spread_angle_on_plane(reference, vector, normal), -45.0)

    def test_median_filter_removes_single_frame_spike(self):
        values = np.asarray([[10.0], [10.0], [90.0], [10.0], [10.0]])
        filtered = MODULE.median_filter(values, radius=1)
        np.testing.assert_allclose(filtered[:, 0], 10.0)

    def test_raw_and_rectified_fisheye_projection_agree(self):
        camera_matrix = np.asarray([
            [500.0, 0.0, 800.0], [0.0, 500.0, 650.0], [0.0, 0.0, 1.0]
        ])
        distortion = np.asarray([0.06, -0.01, -0.004, 0.002])
        rotation, _ = cv2.Rodrigues(np.asarray([0.01, -0.02, 0.005]))
        projection = np.asarray([
            [260.0, 0.0, 800.0, 0.0],
            [0.0, 260.0, 650.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ])
        points = np.asarray([
            [-0.08, -0.04, 0.22], [0.03, 0.02, 0.25], [0.12, -0.05, 0.30]
        ])
        residual = MODULE.raw_rectified_projection_residual(
            points, camera_matrix, distortion, rotation, projection
        )
        self.assertLess(float(np.max(residual)), 1e-8)

    def test_angle_contract_contains_fifteen_bends_and_five_spreads(self):
        self.assertEqual(len(MODULE.ANGLE_KEYS), 20)
        self.assertEqual(sum("bend" in key for key in MODULE.ANGLE_KEYS), 15)
        self.assertEqual(sum("spread" in key for key in MODULE.ANGLE_KEYS), 5)

    def test_kinematic_contract_is_five_plus_four_times_four(self):
        self.assertEqual(len(MODULE.KINEMATIC_KEYS), 21)
        self.assertEqual(len(MODULE.KINEMATIC_LAYOUT["thumb"]), 5)
        for finger in ("index", "middle", "ring", "pinky"):
            self.assertEqual(len(MODULE.KINEMATIC_LAYOUT[finger]), 4)

    def test_mano_pose_projects_to_flexion_and_abduction_axes(self):
        joints = np.zeros((21, 3), dtype=np.float64)
        joints[0] = (0.0, 0.0, 0.0)
        chains = {
            "thumb": ((1, (-0.025, 0.015, 0.0)), (2, (-0.045, 0.035, 0.0)),
                      (3, (-0.055, 0.055, 0.0)), (4, (-0.060, 0.072, 0.0))),
            "index": ((5, (0.030, 0.040, 0.0)), (6, (0.030, 0.070, 0.0)),
                      (7, (0.030, 0.095, 0.0)), (8, (0.030, 0.115, 0.0))),
            "middle": ((9, (0.008, 0.045, 0.0)), (10, (0.008, 0.080, 0.0)),
                       (11, (0.008, 0.108, 0.0)), (12, (0.008, 0.130, 0.0))),
            "ring": ((13, (-0.012, 0.042, 0.0)), (14, (-0.012, 0.074, 0.0)),
                     (15, (-0.012, 0.100, 0.0)), (16, (-0.012, 0.120, 0.0))),
            "pinky": ((17, (-0.032, 0.032, 0.0)), (18, (-0.032, 0.058, 0.0)),
                      (19, (-0.032, 0.080, 0.0)), (20, (-0.032, 0.098, 0.0))),
        }
        for entries in chains.values():
            for index, point in entries:
                joints[index] = point
        axes = MODULE.build_kinematic_axes(joints)
        pose = np.zeros((1, 15, 3), dtype=np.float64)
        pose[0, 0] = 0.5 * axes["index_mcp_flex"] + 0.2 * axes["palm_normal"]
        values = MODULE.extract_kinematic_sequence(pose, axes, "Right")
        by_name = dict(zip(MODULE.KINEMATIC_KEYS, values[0]))
        self.assertAlmostEqual(by_name["index_mcp_flex_rad"], 0.5, places=7)
        self.assertAlmostEqual(by_name["index_mcp_abduction_rad"], 0.2, places=7)

    def test_left_hand_flexion_sign_is_mirrored(self):
        axes = {
            "palm_normal": np.asarray([0.0, 0.0, 1.0]),
            "thumb_cmc_flex": np.asarray([1.0, 0.0, 0.0]),
            "thumb_cmc_opposition": np.asarray([0.0, 1.0, 0.0]),
            "thumb_mcp_flex": np.asarray([1.0, 0.0, 0.0]),
            "thumb_ip_flex": np.asarray([1.0, 0.0, 0.0]),
        }
        for finger in ("index", "middle", "ring", "pinky"):
            for joint in ("mcp", "pip", "dip"):
                axes[f"{finger}_{joint}_flex"] = np.asarray([1.0, 0.0, 0.0])
        pose = np.zeros((1, 15, 3), dtype=np.float64)
        pose[0, 0, 0] = 0.4
        right = MODULE.extract_kinematic_sequence(pose, axes, "Right")[0]
        left = MODULE.extract_kinematic_sequence(pose, axes, "Left")[0]
        index = MODULE.KINEMATIC_KEYS.index("index_mcp_flex_rad")
        self.assertAlmostEqual(right[index], 0.4)
        self.assertAlmostEqual(left[index], -0.4)


if __name__ == "__main__":
    unittest.main()
