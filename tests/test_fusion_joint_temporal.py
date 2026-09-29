from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from fusion_joint_temporal import (  # noqa: E402
    _gaussian_smooth_runs,
    _limit_temporal_steps,
    _remove_isolated_step_spikes,
    stabilize_hand_identities,
    stabilize_joint_temporal_rows,
)


def _pose(frame: int) -> np.ndarray:
    points = np.zeros((21, 3), dtype=np.float64)
    points[0] = (0.004 * frame, 0.0, 0.65)
    for finger in range(5):
        root = 1 + 4 * finger
        for offset in range(4):
            points[root + offset] = (
                0.004 * frame + 0.025 * finger,
                0.018 * offset + 0.004 * finger,
                0.65 + 0.006 * offset,
            )
    return points


def _hand(points: np.ndarray, side: int = 0) -> dict:
    return {
        "side": side,
        "joints_base_m": points.tolist(),
        "inlier_view_counts": [6] * 21,
        "views": {},
    }


def _view(points: np.ndarray, bbox=(0.0, 0.0, 100.0, 100.0)) -> dict:
    return {
        "joints_2d": points.tolist(),
        "wilor_joints_2d": points.tolist(),
        "bbox_xyxy": list(bbox),
    }


class FusionJointTemporalTests(unittest.TestCase):
    def test_identity_reappearance_keeps_both_hands(self):
        prior = _hand(_pose(0), side=1)
        near = _hand(_pose(1), side=0)
        far_points = _pose(1).copy()
        far_points[:, 0] += 0.5
        far = _hand(far_points, side=1)
        rows = [
            {"sync_index": 0, "hands": [prior]},
            {"sync_index": 1, "hands": [near, far]},
        ]
        stabilize_hand_identities(rows)
        self.assertEqual(len(rows[1]["hands"]), 2)
        self.assertEqual({hand["side"] for hand in rows[1]["hands"]}, {0, 1})

    def test_gaussian_smoothing_does_not_cross_missing_gap(self):
        points = np.zeros((1, 7, 1, 3), dtype=np.float64)
        points[0, :, 0, 0] = np.arange(7, dtype=np.float64)
        valid = np.ones((1, 7, 1), dtype=bool)
        valid[0, 3, 0] = False
        points[~valid] = np.nan
        smoothed = _gaussian_smooth_runs(points, valid, 1.3)
        self.assertTrue(np.isnan(smoothed[0, 3, 0]).all())
        self.assertGreaterEqual(float(smoothed[0, 0, 0, 0]), 0.0)
        self.assertLessEqual(float(smoothed[0, 0, 0, 0]), 2.0)
        self.assertGreaterEqual(float(smoothed[0, 6, 0, 0]), 4.0)
        self.assertLessEqual(float(smoothed[0, 6, 0, 0]), 6.0)

    def test_isolated_joint_depth_spike_is_repaired(self):
        rows = []
        for frame in range(9):
            points = _pose(frame)
            if frame == 4:
                points[8] += np.asarray((0.28, -0.12, 0.30))
            rows.append({"sync_index": frame, "hands": [_hand(points)]})
        original = np.asarray(rows[4]["hands"][0]["joints_base_m"])[8]
        expected = _pose(4)[8]
        summary = stabilize_joint_temporal_rows(
            rows, {}, smoothing_sigma=0.0, bone_iterations=8,
        )
        repaired = np.asarray(rows[4]["hands"][0]["joints_base_m"])[8]
        self.assertLess(np.linalg.norm(repaired - expected), 0.35 * np.linalg.norm(original - expected))
        self.assertGreater(summary["corrected_joint_count"], 0)

    def test_missing_3d_point_is_not_resurrected(self):
        rows = []
        for frame in range(5):
            points = _pose(frame)
            if frame == 2:
                points[12] = np.nan
            rows.append({"sync_index": frame, "hands": [_hand(points)]})
        stabilize_joint_temporal_rows(rows, {}, smoothing_sigma=0.0)
        self.assertTrue(
            np.isnan(np.asarray(rows[2]["hands"][0]["joints_base_m"])[12]).all()
        )

    def test_step_spike_filter_keeps_sustained_motion(self):
        points = np.zeros((5, 1, 3), dtype=np.float64)
        points[:, 0, 0] = np.asarray((0.00, 0.01, 0.10, 0.11, 0.12))
        valid = np.ones((5, 1), dtype=bool)
        filtered, changed = _remove_isolated_step_spikes(
            points, valid, np.arange(5), 0.04,
        )
        # The 90 mm one-frame excursion is replaced by the midpoint of its
        # neighbours, while the monotonic 10 mm/frame motion is retained.
        self.assertTrue(changed[2, 0])
        self.assertAlmostEqual(float(filtered[2, 0, 0]), 0.06, places=6)
        self.assertFalse(changed[1, 0])
        self.assertFalse(changed[3, 0])

    def test_final_continuity_guard_caps_branch_transition(self):
        points = np.zeros((5, 1, 3), dtype=np.float64)
        # The large middle excursion is a depth branch, not sustained motion:
        # the trusted endpoints remain nearly stationary.  The final guard
        # should interpolate that interior branch and must not hard-clamp a
        # genuinely fast monotonic gesture.
        points[:, 0, 0] = np.asarray((0.00, 0.01, 0.16, 0.02, 0.03))
        valid = np.ones((5, 1), dtype=bool)
        filtered, changed = _limit_temporal_steps(
            points, valid, np.arange(5), 0.04,
        )
        steps = np.linalg.norm(np.diff(filtered[:, 0], axis=0), axis=1)
        accelerations = np.linalg.norm(
            filtered[2:, 0] - 2.0 * filtered[1:-1, 0] + filtered[:-2, 0],
            axis=1,
        )
        self.assertTrue(np.any(changed))
        self.assertLessEqual(float(np.max(steps)), 0.040001)
        self.assertLessEqual(float(np.max(accelerations)), 0.040001)

    def test_sparse_hand_is_hidden_as_one_visibility_unit(self):
        rows = []
        for frame in range(3):
            points = _pose(frame)
            if frame == 1:
                points[:4] = np.nan
            rows.append({"sync_index": frame, "hands": [_hand(points)]})
        summary = stabilize_joint_temporal_rows(
            rows, {}, smoothing_sigma=0.0, bone_iterations=0,
            min_hand_joints=18,
        )
        self.assertEqual(
            int(np.isfinite(np.asarray(rows[1]["hands"][0]["joints_base_m"])).all(axis=1).sum()),
            0,
        )
        self.assertEqual(summary["sparse_hand_cleared_count"], 1)

    def test_camera_view_jump_hides_merged_box_sample(self):
        base_2d = np.stack([np.asarray((20.0 + j, 30.0 + j)) for j in range(21)])
        jumped_2d = base_2d + np.asarray((210.0, 0.0))
        first = _hand(_pose(0))
        second = _hand(_pose(1))
        first["views"] = {"camera2": _view(base_2d)}
        second["views"] = {"camera2": _view(jumped_2d, (0.0, 0.0, 180.0, 180.0))}
        rows = [
            {"sync_index": 0, "hands": [first]},
            {"sync_index": 1, "hands": [second]},
        ]
        summary = stabilize_joint_temporal_rows(
            rows, {}, smoothing_sigma=0.0, bone_iterations=0,
            view_wrist_jump_px=180.0,
        )
        self.assertEqual(
            int(np.isfinite(np.asarray(rows[1]["hands"][0]["joints_base_m"])).all(axis=1).sum()),
            0,
        )
        self.assertEqual(summary["view_jump_cleared_hand_count"], 1)


if __name__ == "__main__":
    unittest.main()
