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
    @staticmethod
    def moving_pixel_skeleton() -> np.ndarray:
        base = np.zeros((21, 2), dtype=np.float64)
        base[0] = (0.0, 50.0)
        base[1:5] = ((-20, 35), (-30, 20), (-38, 5), (-44, -10))
        base[5:9] = ((-30, 15), (-32, -5), (-34, -25), (-36, -42))
        base[9:13] = ((-10, 8), (-10, -15), (-10, -38), (-10, -58))
        base[13:17] = ((12, 10), (14, -12), (16, -32), (18, -50))
        base[17:21] = ((30, 18), (35, 0), (39, -17), (42, -32))
        frames = np.zeros((1, 11, 21, 2), dtype=np.float64)
        for frame in range(11):
            angle = 0.025 * frame
            rotation = np.asarray([
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle), np.cos(angle)],
            ])
            frames[0, frame] = (base @ rotation.T) * (1.0 + 0.01 * frame)
            frames[0, frame] += (200.0 + 3.0 * frame, 300.0 - 2.0 * frame)
        return frames

    def test_temporal_pixel_shape_spike_is_rejected(self):
        pixels = self.moving_pixel_skeleton()
        pixels[0, 5, 8] += (75.0, -50.0)
        rejected = MODULE.detect_temporal_pixel_outliers(
            pixels, radius=4, normalized_distance=0.45, maximum_scale_ratio=1.8
        )
        self.assertTrue(rejected[0, 5, 8])
        self.assertFalse(rejected[0, 5, 7])

    def test_global_hand_motion_does_not_trigger_pixel_outlier(self):
        rejected = MODULE.detect_temporal_pixel_outliers(
            self.moving_pixel_skeleton(), radius=4,
            normalized_distance=0.45, maximum_scale_ratio=1.8,
        )
        self.assertEqual(np.count_nonzero(rejected), 0)

    def test_palm_frame_smoothing_rejects_finger_spike_and_keeps_palm_attached(self):
        pixels = self.moving_pixel_skeleton()
        expected_tip = pixels[0, 5, 8].copy()
        pixels[0, 5, 8] += (75.0, -50.0)
        valid = np.ones(pixels.shape[:-1], dtype=bool)

        filtered = MODULE.smooth_pixel_landmarks_in_palm_frame(
            pixels, valid, radius=3, strength=1.0
        )

        raw_error = np.linalg.norm(pixels[0, 5, 8] - expected_tip)
        filtered_error = np.linalg.norm(filtered[0, 5, 8] - expected_tip)
        self.assertLess(filtered_error, raw_error * 0.2)
        raw_centres = np.median(pixels[0][:, MODULE.PALM_FRAME_JOINTS], axis=1)
        filtered_centres = np.median(
            filtered[0][:, MODULE.PALM_FRAME_JOINTS], axis=1
        )
        np.testing.assert_allclose(filtered_centres, raw_centres, atol=1e-9)

    def test_palm_frame_smoothing_reduces_local_shape_step(self):
        pixels = self.moving_pixel_skeleton()
        pixels[0, 5, 8] += (75.0, -50.0)
        valid = np.ones(pixels.shape[:-1], dtype=bool)
        filtered = MODULE.smooth_pixel_landmarks_in_palm_frame(
            pixels, valid, radius=3, strength=0.8
        )
        raw = MODULE.palm_normalized_pixel_step_metric(pixels, valid)
        smooth = MODULE.palm_normalized_pixel_step_metric(filtered, valid)
        self.assertLess(smooth["p95"], raw["p95"])

    def test_3d_palm_frame_smoothing_keeps_palm_and_rejects_finger_spike(self):
        pixels = self.moving_pixel_skeleton()
        points = np.zeros((1, 11, 21, 3), dtype=np.float64)
        points[..., :2] = pixels / 1000.0
        points[..., 2] = np.arange(21, dtype=np.float64)[None, None, :] * 0.0002
        expected_tip = points[0, 5, 8].copy()
        points[0, 5, 8] += (0.075, -0.050, 0.040)
        valid = np.ones(points.shape[:-1], dtype=bool)

        filtered = MODULE.smooth_3d_landmarks_in_palm_frame(
            points, valid, radius=3, strength=1.0
        )

        raw_error = np.linalg.norm(points[0, 5, 8] - expected_tip)
        filtered_error = np.linalg.norm(filtered[0, 5, 8] - expected_tip)
        self.assertLess(filtered_error, raw_error * 0.25)
        raw_centres = np.median(points[0][:, MODULE.PALM_FRAME_JOINTS], axis=1)
        filtered_centres = np.median(
            filtered[0][:, MODULE.PALM_FRAME_JOINTS], axis=1
        )
        np.testing.assert_allclose(filtered_centres, raw_centres, atol=1e-12)
        raw_metric = MODULE.palm_normalized_3d_step_metric(points, valid)
        filtered_metric = MODULE.palm_normalized_3d_step_metric(filtered, valid)
        self.assertLess(filtered_metric["p95"], raw_metric["p95"])

    def test_stereo_confidence_uses_disparity_and_refinement_quality(self):
        row = {
            "landmark_index": "8",
            "epipolar_error_px": "0.5",
            "reprojection_error_px": "0.4",
            "disparity_px": "40.0",
            "left_handedness_score": "0.95",
            "right_handedness_score": "0.95",
            "refinement_attempted": "1",
            "refinement_used": "1",
            "refinement_quality": "0.9",
        }
        good = MODULE.confidence_from_row(row)
        failed = dict(row, refinement_used="0", refinement_quality="0.0")
        low_disparity = dict(row, disparity_px="4.0")
        self.assertLess(MODULE.confidence_from_row(failed), good * 0.5)
        self.assertLess(MODULE.confidence_from_row(low_disparity), good * 0.2)

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
