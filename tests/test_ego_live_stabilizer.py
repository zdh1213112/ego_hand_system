"""Causal filtering must preserve missing-data semantics and reset on gaps."""

import unittest

import numpy as np

from ego_live_stabilizer import BoundedDisplayFilter, CausalJointFilter, LiveHandStabilizer


class CausalJointFilterTests(unittest.TestCase):
    def test_display_filter_suppresses_static_noise_without_frame_delay(self):
        smooth = BoundedDisplayFilter()
        base = np.zeros((21, 3), dtype=float)
        smooth.update(base, 0)
        small = base.copy()
        small[0, 0] = 0.002
        small[8, 0] = 0.003
        filtered = np.asarray(smooth.update(small, 33_333))
        self.assertAlmostEqual(filtered[0, 0], 0.0004)
        self.assertAlmostEqual(filtered[8, 0], 0.0006)

        moved = base.copy()
        moved[:, 0] = 0.10
        filtered = np.asarray(smooth.update(moved, 66_666))
        self.assertLess(np.linalg.norm(filtered[0] - moved[0]), 0.00251)
        self.assertLess(np.linalg.norm(filtered[8] - moved[8]), 0.00351)

        missing = moved.copy()
        missing[8] = np.nan
        self.assertTrue(np.isnan(np.asarray(smooth.update(missing, 99_999))[8]).all())
        self.assertAlmostEqual(
            np.asarray(smooth.update(moved, 133_332))[8, 0], 0.10,
        )

    def test_one_frame_spike_is_reduced_without_predicting_invalid_joints(self):
        smooth = CausalJointFilter(alpha=0.55)
        base = np.zeros((21, 3), dtype=float)
        spike = base.copy()
        spike[8, 0] = 0.10
        smooth.update(base, 0)
        second = np.asarray(smooth.update(spike, 33_333))
        third = np.asarray(smooth.update(base, 66_666))
        self.assertLess(second[8, 0], 0.04)
        self.assertLess(third[8, 0], 0.02)

        missing = base.copy()
        missing[8] = np.nan
        fourth = np.asarray(smooth.update(missing, 99_999))
        self.assertTrue(np.isnan(fourth[8]).all())
        self.assertTrue(np.isfinite(fourth[7]).all())

    def test_gap_resets_old_pose(self):
        smooth = CausalJointFilter(alpha=0.55)
        first = np.zeros((21, 3), dtype=float)
        next_pose = np.ones((21, 3), dtype=float)
        smooth.update(first, 1_000_000)
        result = np.asarray(smooth.update(next_pose, 1_300_000))
        np.testing.assert_allclose(result, next_pose)

    def test_raw_fields_are_preserved(self):
        raw = np.zeros((21, 3), dtype=float).tolist()
        row = {
            "timestamp_us": {"cam_0": 1_000_000},
            "camera_hands": {"cam_0": {0: {
                "joints_2d": np.zeros((21, 2)).tolist(),
                "joints_3d_camera_relative": raw,
            }}},
            "hands": [{"is_right": 0, "joints_3d_base_m": raw}],
        }
        LiveHandStabilizer().update_row(row)
        self.assertEqual(row["hands"][0]["joints_3d_base_m"], raw)
        self.assertIn("joints_3d_base_m_stabilized", row["hands"][0])
        self.assertIn("joints_3d_base_m_display", row["hands"][0])
        self.assertIn("joints_3d_camera_relative_stabilized",
                      row["camera_hands"]["cam_0"][0])

if __name__ == "__main__":
    unittest.main()
