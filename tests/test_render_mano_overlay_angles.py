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


if __name__ == "__main__":
    unittest.main()
