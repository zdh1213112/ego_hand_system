import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from pico_sync_quality import (
    full_motion_quality,
    motion_window_quality,
    nearest_time_pairs,
)


class PicoSyncQualityTest(unittest.TestCase):
    def test_full_sequence_lag_uses_whole_recording_and_labels_comparison(self):
        pg, ng = np.arange(0.04, 32, 0.02), np.arange(0, 34, 0.005)
        signal = lambda t: 1.4 + 0.6 * np.sin(t * 0.9) + 0.35 * np.cos(t * 2.3)
        support = lambda t, r: (t - r >= ng[0]) & (t + r <= ng[-1])
        result = full_motion_quality(
            pg, signal(pg), ng, signal(ng), 1.0, -0.007, support
        )
        self.assertEqual(result["status"], "ok")
        self.assertAlmostEqual(result["residual_offset_ms"], 7, delta=0.03)
        self.assertGreater(result["end_s"], 31)
        self.assertGreater(result["samples"], 1500)
        self.assertIn("PICO head VIO", result["pico_source"])
        self.assertIn("head rigid-body", result["nokov_source"])
        self.assertFalse(result["global_clock_refitted"])
        self.assertFalse(result["residual_applied"])

    def test_nearest_pair_direction_ties_and_outside_samples(self):
        pt = np.array([0.0, 0.005, 0.014, 0.035])
        nt = np.array([0.0, 0.01, 0.02, 0.03])
        summary, data = nearest_time_pairs(pt, nt, 1.0, 0.0, np.ones(4, bool))
        np.testing.assert_array_equal(data["nearest"][:3], [0, 0, 1])
        np.testing.assert_allclose(data["delta_s"][:3], [0, -0.005, -0.004])
        self.assertFalse(data["used"][-1])
        self.assertEqual(summary["count"], 3)
        self.assertEqual(summary["reused_frame_pairs"], 1)

    def test_sampling_pair_error_remains_nonzero_with_perfect_clock(self):
        pt, nt = np.arange(0, 1, 0.014), np.arange(0, 1, 0.011)
        summary, _ = nearest_time_pairs(pt, nt, 1.0, 0.0, np.ones(len(pt), bool))
        self.assertGreater(summary["mean_abs_ms"], 1)
        self.assertLessEqual(summary["max_abs_ms"], 5.5 + 1e-9)

    def test_ineligible_samples_are_excluded_from_pair_statistics(self):
        summary, _ = nearest_time_pairs(
            np.array([0.0, 0.004, 0.014]),
            np.array([0.0, 0.01, 0.02]),
            1.0,
            0.0,
            [True, False, True],
        )
        self.assertEqual(summary["count"], 2)
        self.assertAlmostEqual(summary["mean_abs_ms"], 2)

    def test_window_residual_recovers_known_remaining_delay_without_refitting(self):
        pg, ng = np.arange(0.04, 32, 0.02), np.arange(0, 34, 0.005)
        signal = lambda t: 1.4 + 0.6 * np.sin(t * 0.9) + 0.35 * np.cos(t * 2.3)
        support = lambda t, r: (t - r >= ng[0]) & (t + r <= ng[-1])
        expected = 0.007
        report = motion_window_quality(
            pg, signal(pg), ng, signal(ng), 1.0, -expected, support
        )
        self.assertGreaterEqual(report["accepted_windows"], 3)
        self.assertFalse(report["global_clock_refitted"])
        for row in report["windows"]:
            if row["status"] == "ok":
                self.assertAlmostEqual(row["residual_offset_ms"], 7, delta=0.03)
                self.assertLess(
                    row["speed_rmse_after_rad_s"], row["speed_rmse_before_rad_s"]
                )

    def test_weak_motion_and_search_boundary_are_not_accepted(self):
        pg, ng = np.arange(0.04, 24, 0.02), np.arange(0, 26, 0.005)
        support = lambda t, r: (t - r >= ng[0]) & (t + r <= ng[-1])
        report = motion_window_quality(
            pg, np.ones(len(pg)), ng, np.ones(len(ng)), 1.0, 0.0, support
        )
        self.assertEqual(report["accepted_windows"], 0)
        self.assertTrue(
            all(r["status"] == "insufficient_motion" for r in report["windows"])
        )
        signal = lambda t: 1.5 + 0.5 * np.sin(t) + 0.3 * np.cos(2.1 * t)
        report = motion_window_quality(
            pg, signal(pg), ng, signal(ng), 1.0, -0.2, support
        )
        self.assertEqual(report["accepted_windows"], 0)
        self.assertTrue(
            any(r["status"] == "search_boundary" for r in report["windows"])
        )


if __name__ == "__main__":
    unittest.main()
