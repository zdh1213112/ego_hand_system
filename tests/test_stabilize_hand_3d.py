#!/usr/bin/env python3
"""Focused regression tests for MANO preparation missing-data handling."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "stabilize_hand_3d.py"
SPEC = importlib.util.spec_from_file_location("stabilize_hand_3d", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class StabilizeHand3DTests(unittest.TestCase):
    def test_only_short_internal_gaps_are_interpolated(self):
        points = np.full((1, 9, 21, 3), np.nan)
        valid = np.zeros((1, 9, 21), dtype=bool)
        confidence = np.zeros((1, 9, 21))
        for frame, x in ((0, 0.0), (3, 0.03), (8, 0.08)):
            points[0, frame, 0] = (x, 0.0, 0.2)
            valid[0, frame, 0] = True
            confidence[0, frame, 0] = 1.0

        result, output_valid, _, interpolated = MODULE.interpolate_short_gaps(
            points, valid, confidence, [(0, 8)], max_gap=2
        )

        self.assertTrue(np.all(output_valid[0, 1:3, 0]))
        self.assertTrue(np.all(interpolated[0, 1:3, 0]))
        self.assertAlmostEqual(result[0, 1, 0, 0], 0.01)
        self.assertFalse(np.any(output_valid[0, 4:8, 0]))

    def test_temporal_depth_spike_is_rejected(self):
        points = np.zeros((1, 9, 21, 3), dtype=np.float64)
        points[..., 2] = 0.2
        observed = np.zeros((1, 9, 21), dtype=bool)
        observed[0, :, 0] = True
        points[0, 4, 0] = (0.8, -0.4, 1.5)

        accepted, rejected = MODULE.reject_observation_outliers(
            points, observed, [(0, 8)], temporal_radius=4,
            temporal_distance_m=0.12, max_hand_radius_m=0.22,
        )

        self.assertFalse(accepted[0, 4, 0])
        self.assertTrue(rejected[0, 4, 0])
        self.assertEqual(np.count_nonzero(rejected), 1)

    def test_gross_bone_outlier_rejects_lower_confidence_endpoint(self):
        points = np.zeros((1, 1, 21, 3), dtype=np.float64)
        observed = np.zeros((1, 1, 21), dtype=bool)
        confidence = np.zeros((1, 1, 21), dtype=np.float64)
        observed[0, 0, 0] = True
        observed[0, 0, 1] = True
        confidence[0, 0, 0] = 0.9
        confidence[0, 0, 1] = 0.1
        points[0, 0, 1] = (0.5, 0.0, 0.0)
        targets = np.full((1, len(MODULE.SKELETON_EDGES)), 0.03)

        accepted, rejected = MODULE.reject_bone_outliers(
            points, observed, confidence, targets,
            absolute_tolerance_m=0.05, relative_tolerance=0.8,
        )

        self.assertTrue(accepted[0, 0, 0])
        self.assertFalse(accepted[0, 0, 1])
        self.assertTrue(rejected[0, 0, 1])


if __name__ == "__main__":
    unittest.main()
