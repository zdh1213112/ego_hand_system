from __future__ import annotations

import copy
from pathlib import Path
import sys
import unittest

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from fusion_anatomy_refinement import (  # noqa: E402
    FusionAnatomyConfig,
    refine_accepted_rows,
)


def _hand_pose(frame: int) -> np.ndarray:
    points = np.zeros((21, 3), dtype=np.float64)
    points[0] = (0.0, 0.0, 0.65)
    roots = (
        (0.030, 0.018),
        (0.034, 0.040),
        (0.010, 0.045),
        (-0.014, 0.040),
        (-0.036, 0.032),
    )
    increments = (
        (0.014, 0.014),
        (0.000, 0.026),
        (0.000, 0.028),
        (0.000, 0.026),
        (-0.003, 0.022),
    )
    for finger, ((root_x, root_y), (step_x, step_y)) in enumerate(
        zip(roots, increments)
    ):
        start = 1 + 4 * finger
        for offset in range(4):
            points[start + offset] = (
                root_x + offset * step_x,
                root_y + offset * step_y,
                0.65 + 0.002 * offset,
            )
    points[:, 0] += 0.004 * frame
    return points


def _rows_with_one_spike() -> list[dict]:
    rows = []
    for frame in range(9):
        points = _hand_pose(frame)
        if frame == 4:
            points[8] += np.asarray((0.32, -0.18, 0.24))
        rows.append({
            "sync_index": frame,
            "hands": [{
                "side": 0,
                "joints_base_m": points.tolist(),
                "inlier_view_counts": [4] * 21,
            }],
        })
    return rows


class FusionAnatomyRefinementTests(unittest.TestCase):
    def test_isolated_joint_spike_is_repaired_without_moving_clean_hands(self):
        rows = _rows_with_one_spike()
        original = copy.deepcopy(rows)
        summary = refine_accepted_rows(rows, 6, FusionAnatomyConfig())

        expected = _hand_pose(4)[8]
        raw_error = np.linalg.norm(
            np.asarray(original[4]["hands"][0]["joints_base_m"])[8] - expected
        )
        refined_error = np.linalg.norm(
            np.asarray(rows[4]["hands"][0]["joints_base_m"])[8] - expected
        )
        self.assertLess(refined_error, 0.25 * raw_error)
        np.testing.assert_allclose(
            rows[0]["hands"][0]["joints_base_m"],
            original[0]["hands"][0]["joints_base_m"],
            atol=1e-12,
        )
        self.assertGreaterEqual(summary["repaired_outlier_count"], 1)
        self.assertGreater(summary["corrected_joint_count"], 0)
        self.assertFalse(rows[0]["hands"][0]["anatomy_refinement"]["applied"])
        self.assertTrue(rows[4]["hands"][0]["anatomy_refinement"]["applied"])

    def test_empty_input_has_explicit_enabled_summary(self):
        summary = refine_accepted_rows([], 6, FusionAnatomyConfig())
        self.assertTrue(summary["enabled"])
        self.assertEqual(summary["corrected_joint_count"], 0)

    def test_configuration_rejects_invalid_adjustment_strength(self):
        with self.assertRaises(ValueError):
            FusionAnatomyConfig(reliable_adjustment_blend=1.1).validate()


if __name__ == "__main__":
    unittest.main()
