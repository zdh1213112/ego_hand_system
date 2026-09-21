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
    _apply_bidirectional_joint_spike_guard,
    _apply_confidence_guided_tracking,
    _apply_final_palm_trajectory_filter,
    _apply_robust_temporal_refinement,
    _drop_unreconstructable_hands,
    _final_center_step_guard,
    _hand_identity_collision,
    _limit_refinement_center_displacement,
    refine_accepted_rows,
)
from stabilize_hand_3d import SKELETON_EDGES  # noqa: E402


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


def _direct_sequence() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    points = np.full((2, 9, 21, 3), np.nan, dtype=np.float64)
    for frame in range(9):
        points[0, frame] = _hand_pose(frame)
    observed = np.isfinite(points).all(axis=-1)
    confidence = np.where(observed, 0.8, 0.0)
    rejected = np.zeros_like(observed)
    reference = _hand_pose(0)
    lengths = np.asarray([
        np.linalg.norm(reference[child] - reference[parent])
        for parent, child in SKELETON_EDGES
    ])
    bone_lengths = np.stack((lengths, lengths))
    return points, observed, confidence, bone_lengths


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

    def test_whole_hand_depth_spike_is_repaired_by_bidirectional_context(self):
        rows = []
        for frame in range(9):
            points = _hand_pose(frame)
            if frame == 4:
                points += np.asarray((0.05, -0.03, 0.42))
            rows.append({
                "sync_index": frame,
                "hands": [{
                    "side": 0,
                    "joints_base_m": points.tolist(),
                    "inlier_view_counts": [4] * 21,
                }],
            })
        raw_error = np.linalg.norm(
            np.asarray(rows[4]["hands"][0]["joints_base_m"])[0]
            - _hand_pose(4)[0]
        )
        summary = refine_accepted_rows(rows, 6, FusionAnatomyConfig())
        refined_error = np.linalg.norm(
            np.asarray(rows[4]["hands"][0]["joints_base_m"])[0]
            - _hand_pose(4)[0]
        )
        self.assertLess(refined_error, 0.35 * raw_error)
        self.assertGreaterEqual(summary["temporal_global_corrected_hand_count"], 1)
        self.assertTrue(
            rows[4]["hands"][0]["anatomy_refinement"]["temporal_global_corrected"]
        )

    def test_configuration_rejects_invalid_adjustment_strength(self):
        with self.assertRaises(ValueError):
            FusionAnatomyConfig(reliable_adjustment_blend=1.1).validate()

    def test_nonphysical_hand_is_not_temporally_transformed(self):
        points, observed, confidence, bone_lengths = _direct_sequence()
        points[0, 4, 17] += np.asarray((12.0, 0.5, -2.0))
        rejected = np.zeros_like(observed)
        rejected[0, 4, 17] = True
        output, global_corrected, local_corrected, _global, _local = (
            _apply_robust_temporal_refinement(
                points,
                observed,
                rejected,
                confidence,
                bone_lengths,
                FusionAnatomyConfig(),
            )
        )
        np.testing.assert_allclose(output[0, 4], points[0, 4])
        self.assertFalse(global_corrected[0, 4])
        self.assertFalse(local_corrected[0, 4].any())

    def test_bone_constraint_cannot_bypass_local_displacement_cap(self):
        points, observed, confidence, bone_lengths = _direct_sequence()
        points[0, 4, 8] += np.asarray((0.08, -0.08, 0.06))
        points[0, 4, 12] += np.asarray((-0.08, 0.08, -0.06))
        rejected = np.zeros_like(observed)
        rejected[0, 4, (8, 12)] = True
        output, _global, local_corrected, _gres, _lres = (
            _apply_robust_temporal_refinement(
                points,
                observed,
                rejected,
                confidence,
                bone_lengths,
                FusionAnatomyConfig(),
            )
        )
        self.assertGreaterEqual(np.count_nonzero(local_corrected[0, 4]), 2)
        displacement = np.linalg.norm(output - points, axis=-1)
        self.assertLessEqual(
            float(np.nanmax(displacement)),
            FusionAnatomyConfig().temporal_max_local_adjustment_m + 1e-9,
        )

    def test_unreconstructable_whole_hand_is_dropped_instead_of_kept(self):
        refined = np.ones((1, 1, 21, 3), dtype=np.float64)
        observed = np.ones((1, 1, 21), dtype=bool)
        rejected = np.ones_like(observed)
        candidate_finite = np.zeros_like(observed)
        output, dropped = _drop_unreconstructable_hands(
            refined, observed, rejected, candidate_finite
        )
        self.assertTrue(dropped[0, 0])
        self.assertFalse(np.isfinite(output[0, 0]).any())

    def test_partially_reconstructed_whole_hand_is_also_dropped(self):
        refined = np.ones((1, 1, 21, 3), dtype=np.float64)
        observed = np.ones((1, 1, 21), dtype=bool)
        rejected = np.ones_like(observed)
        candidate_finite = np.zeros_like(observed)
        candidate_finite[0, 0, 5:] = True
        output, dropped = _drop_unreconstructable_hands(
            refined, observed, rejected, candidate_finite
        )
        self.assertTrue(dropped[0, 0])
        self.assertFalse(np.isfinite(output[0, 0]).any())

    def test_tracking_repairs_low_support_shape_spike_without_touching_good_frame(self):
        points, observed, confidence, _bone_lengths = _direct_sequence()
        points[0, 4, 8] += np.asarray((0.10, -0.04, 0.02))
        hands = {}
        for frame in range(points.shape[1]):
            support = [2] * 21 if frame == 4 else [5] * 21
            hands[(frame, 0)] = {
                "inlier_view_counts": support,
                "quality": {
                    "multiview_reprojection_median_px": 6.0,
                    "all_view_reprojection_p95_px": 20.0,
                },
                "anatomy_refinement": {"corrected_joint_count": 0},
            }
        before = points.copy()
        output, corrected, hand_corrected, *_ = _apply_confidence_guided_tracking(
            points,
            observed,
            confidence,
            hands,
            FusionAnatomyConfig(),
        )
        self.assertTrue(hand_corrected[0, 4])
        self.assertGreater(np.count_nonzero(corrected[0, 4]), 0)
        self.assertLess(
            np.linalg.norm(output[0, 4, 8] - _hand_pose(4)[8]),
            np.linalg.norm(before[0, 4, 8] - _hand_pose(4)[8]),
        )
        np.testing.assert_allclose(output[0, 3], before[0, 3])

    def test_tracking_repairs_severe_palm_jump_even_with_high_support(self):
        points, observed, confidence, _bone_lengths = _direct_sequence()
        # Simulate a wrong depth branch selected by several cameras: support
        # remains high, but the whole hand is far from the bidirectional track.
        points[0, 4] += np.asarray((0.04, -0.03, 0.22))
        hands = {
            (frame, 0): {
                "inlier_view_counts": [5] * 21,
                "quality": {
                    "multiview_reprojection_median_px": 6.0,
                    "all_view_reprojection_p95_px": 20.0,
                },
                "anatomy_refinement": {"corrected_joint_count": 0},
            }
            for frame in range(points.shape[1])
        }
        output, corrected, hand_corrected, *_ = _apply_confidence_guided_tracking(
            points, observed, confidence, hands, FusionAnatomyConfig()
        )
        self.assertTrue(hand_corrected[0, 4])
        self.assertGreater(np.count_nonzero(corrected[0, 4]), 10)
        self.assertLess(
            np.linalg.norm(output[0, 4, 0] - _hand_pose(4)[0]),
            np.linalg.norm(points[0, 4, 0] - _hand_pose(4)[0]),
        )

    def test_identity_collision_detected_from_shared_camera_wrist(self):
        hands = {
            (4, 0): {"views": {"camera2": {"joints_2d": [[100.0, 200.0]] * 21}}},
            (4, 1): {"views": {"camera2": {"joints_2d": [[103.0, 204.0]] * 21}}},
        }
        self.assertTrue(_hand_identity_collision(4, 1, hands))

    def test_identity_collision_never_enables_reprojection_override(self):
        points, observed, confidence, _bone_lengths = _direct_sequence()
        points[0, 4] += np.asarray((0.04, -0.03, 0.22))
        hands = {
            (frame, side): {
                "inlier_view_counts": [2] * 21 if frame == 4 else [5] * 21,
                "quality": {
                    "multiview_reprojection_median_px": 6.0,
                    "all_view_reprojection_p95_px": 20.0,
                },
                "anatomy_refinement": {"corrected_joint_count": 0},
                "views": {
                    "camera2": {
                        "joints_2d": ([[100.0, 200.0]] * 21),
                    }
                },
            }
            for frame in range(points.shape[1])
            for side in [0]
        }
        # Add the opposite hand only at the ambiguous frame.
        hands[(4, 1)] = {
            "inlier_view_counts": [2] * 21,
            "quality": {
                "multiview_reprojection_median_px": 6.0,
                "all_view_reprojection_p95_px": 20.0,
            },
            "anatomy_refinement": {"corrected_joint_count": 0},
            "views": {"camera2": {"joints_2d": ([[101.0, 201.0]] * 21)}},
        }
        result = _apply_confidence_guided_tracking(
            points, observed, confidence, hands, FusionAnatomyConfig()
        )
        self.assertFalse(result[6][0, 4])

    def test_joint_spike_guard_repairs_incomplete_palm_depth_branch(self):
        points, observed, _confidence, _bone_lengths = _direct_sequence()
        # Keep only a subset of the palm valid and place the wrist on a wrong
        # depth branch. Complete-palm temporal passes intentionally skip this
        # case; the per-joint guard must still repair the isolated wrist.
        points[0, 4, 0] += np.asarray((0.0, 0.0, 0.30))
        observed[0, 4, 5] = False
        observed[0, 4, 9] = False
        observed[0, 4, 13] = False
        observed[0, 4, 17] = False
        output, corrected, strong, _residuals = _apply_bidirectional_joint_spike_guard(
            points, observed, FusionAnatomyConfig()
        )
        self.assertTrue(corrected[0, 4, 0])
        self.assertTrue(strong[0, 4, 0])
        self.assertLess(
            np.linalg.norm(output[0, 4, 0] - _hand_pose(4)[0]),
            0.10,
        )

    def test_final_palm_filter_repairs_low_support_branch_across_missing_frames(self):
        points = np.full((2, 9, 21, 3), np.nan, dtype=np.float64)
        for frame in (0, 1, 2, 4, 6, 7, 8):
            points[0, frame] = _hand_pose(frame)
        points[0, 4] += np.asarray((0.25, -0.20, 0.30))
        observed = np.isfinite(points).all(axis=-1)
        hands = {
            (frame, 0): {
                "inlier_view_counts": ([2] * 21 if frame == 4 else [4] * 21),
                "quality": {"multiview_reprojection_median_px": 6.0},
            }
            for frame in (0, 1, 2, 4, 6, 7, 8)
        }
        output, corrected, residuals = _apply_final_palm_trajectory_filter(
            points, observed, hands, FusionAnatomyConfig()
        )
        self.assertTrue(corrected[0, 4])
        self.assertGreater(residuals[0, 4], 0.20)
        self.assertLess(
            np.linalg.norm(output[0, 4, 0] - _hand_pose(4)[0]),
            np.linalg.norm(points[0, 4, 0] - _hand_pose(4)[0]),
        )
        np.testing.assert_allclose(output[0, 2], points[0, 2])

    def test_final_palm_filter_preserves_fast_monotonic_motion(self):
        points = np.full((2, 9, 21, 3), np.nan, dtype=np.float64)
        for frame in range(9):
            points[0, frame] = _hand_pose(0) + np.asarray(
                (0.055 * frame, 0.010 * frame, 0.0)
            )
        observed = np.isfinite(points).all(axis=-1)
        hands = {
            (frame, 0): {
                "inlier_view_counts": [2] * 21,
                "quality": {"multiview_reprojection_median_px": 6.0},
            }
            for frame in range(9)
        }
        output, corrected, _residuals = _apply_final_palm_trajectory_filter(
            points, observed, hands, FusionAnatomyConfig()
        )
        self.assertFalse(corrected.any())
        np.testing.assert_allclose(output, points, equal_nan=True)

    def test_final_center_guard_reverts_tracking_created_jump(self):
        raw, observed, _confidence, _bone_lengths = _direct_sequence()
        refined = raw.copy()
        refined[0, 4] += np.asarray((0.0, 0.0, 0.12))
        hands = {(frame, 0): {} for frame in range(9)}
        output, reverted = _final_center_step_guard(
            refined, raw, observed, hands, FusionAnatomyConfig()
        )
        self.assertTrue(reverted[0, 4])
        np.testing.assert_allclose(output[0, 4], raw[0, 4])

    def test_final_center_guard_preserves_nonworsening_correction(self):
        raw, observed, _confidence, _bone_lengths = _direct_sequence()
        # The raw frame contains a small centre spike; moving it toward the
        # bidirectional trajectory reduces both adjacent steps.
        raw[0, 4] += np.asarray((0.0, 0.0, 0.08))
        refined = raw.copy()
        refined[0, 4] -= np.asarray((0.0, 0.0, 0.06))
        hands = {(frame, 0): {} for frame in range(9)}
        output, reverted = _final_center_step_guard(
            refined, raw, observed, hands, FusionAnatomyConfig()
        )
        self.assertFalse(reverted[0, 4])
        np.testing.assert_allclose(output[0, 4], refined[0, 4])

    def test_refinement_rigid_component_is_capped_at_hand_level(self):
        raw, observed, _confidence, _bone_lengths = _direct_sequence()
        refined = raw.copy()
        refined[0, 4] += np.asarray((0.0, 0.0, 0.24))
        output, limited = _limit_refinement_center_displacement(
            refined, raw, observed, 0.10
        )
        self.assertTrue(limited[0, 4])
        displacement = output[0, 4] - raw[0, 4]
        np.testing.assert_allclose(
            np.linalg.norm(displacement, axis=1), 0.10, atol=1e-10
        )


if __name__ == "__main__":
    unittest.main()
