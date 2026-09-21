from __future__ import annotations

import argparse
from pathlib import Path
import sys
import unittest

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from camera_models import project_points
from ego_data.calibration import CameraCalibration
from fuse_multiview_wilor import _triangulate_hand, triangulate_ransac
from fuse_multiview_wilor_guided import (
    _complete_detections,
    _depth_condition_reasons, _depth_reconnect_contradiction, _fuse_primary_frame,
    _hypothesis_branch_quality_reasons, _hypothesis_transition_cost,
    _independent_candidate_quality, _independent_pair_decision,
    _apply_temporal_depth_guard, _near_coincident_independent_pair_is_unsafe,
    _marker_assist_cost,
    _anchor_border_cost, _tip_support_penalty,
    _marker_evidence_rank, _selection_has_marker_evidence,
    _match_camera, _ordered_hand_pairs, _prune_unusable_hands,
    _row_to_hypothesis,
    _same_detector_observation, _duplicate_hand_observation_count,
    _stabilize_hand_identities, _stabilize_partial_side_labels,
    _identity_2d_link_policy, _stabilize_hand_identities_2d,
    _final_sequence_guard, _final_sequence_neighbours,
)
from normalize_multiview_recording import synchronize_rows
from render_multiview_wilor import _final_overlay_for_view


def ds_camera(camera_id: str, center_x: float) -> CameraCalibration:
    transform = np.eye(4)
    transform[0, 3] = center_x
    return CameraCalibration(
        camera_id, camera_id, "DS", (1600, 1300),
        np.asarray([[510.0, 0.0, 799.5], [0.0, 512.0, 649.5], [0.0, 0.0, 1.0]]),
        np.asarray([510.0, 512.0, 799.5, 649.5, -0.003, 0.572]),
        transform,
    )


class MultiviewWilorTests(unittest.TestCase):
    def test_tip_support_penalty_targets_only_sub_three_view_tips(self):
        result = {
            "points": np.zeros((21, 3), dtype=np.float64),
            "inlier_counts": np.full(21, 3, dtype=np.int32),
        }
        args = argparse.Namespace(tip_three_view_penalty=12.0)
        self.assertEqual(_tip_support_penalty(result, args), 0.0)
        result["inlier_counts"][[4, 8, 12, 16, 20]] = 2
        self.assertEqual(_tip_support_penalty(result, args), 60.0)

    def test_anchor_border_cost_penalizes_truncated_anchor_only(self):
        camera = ds_camera("camera0", 0.0)
        args = argparse.Namespace(
            anchor_border_penalty=80.0,
            anchor_border_margin_px=3.0,
        )
        clear = {"camera0": {"bbox_xyxy": [100.0, 100.0, 300.0, 400.0]}}
        clipped = {"camera0": {"bbox_xyxy": [100.0, 1100.0, 300.0, 1299.0]}}
        calibrations = {"camera0": camera}
        self.assertEqual(
            _anchor_border_cost(("camera0",), [clear], calibrations, args), 0.0
        )
        self.assertEqual(
            _anchor_border_cost(("camera0",), [clipped], calibrations, args), 80.0
        )

    def test_marker_evidence_is_ranked_ahead_of_detector_confidence(self):
        groups = {
            0: {
                0: {"confidence": 0.99, "marker_assist": {"applied": False}},
                1: {"confidence": 0.99, "marker_assist": {"applied": False}},
            },
            1: {
                0: {
                    "confidence": 0.55,
                    "marker_assist": {
                        "applied": False,
                        "evidence_only": True,
                        "matched_marker_count": 6,
                    },
                },
                1: {"confidence": 0.55, "marker_assist": {"applied": False}},
            },
        }
        self.assertEqual(_complete_detections(groups, 1), [1])
        self.assertGreater(
            _marker_evidence_rank(groups[1][0]),
            _marker_evidence_rank(groups[0][0]),
        )
        self.assertTrue(_selection_has_marker_evidence({"camera0": groups[1][0]}))
        self.assertFalse(_selection_has_marker_evidence({"camera0": groups[0][0]}))

    def test_final_overlay_requires_an_observed_geometrically_consistent_view(self):
        camera = ds_camera("camera0", 0.0)
        points = self._guard_pose()
        pixels, valid = project_points(camera, points)
        self.assertTrue(valid.all())
        lower = np.min(pixels, axis=0) - 20.0
        upper = np.max(pixels, axis=0) + 20.0
        hand = {
            "joints_base_m": points.tolist(),
            "views": {
                "camera0": {
                    "joints_2d": pixels.tolist(),
                    "bbox_xyxy": [lower[0], lower[1], upper[0], upper[1]],
                    "inlier_joint_count": 21,
                }
            },
        }
        overlay, reason = _final_overlay_for_view(camera, hand, "camera0")
        self.assertEqual(reason, "safe")
        self.assertEqual(overlay.shape, (21, 2))
        overlay, reason = _final_overlay_for_view(camera, hand, "camera1")
        self.assertIsNone(overlay)
        self.assertEqual(reason, "no_view")

    @staticmethod
    def _continuity_args() -> argparse.Namespace:
        return argparse.Namespace(
            continuity_weight=20000.0,
            continuity_max_step_m=0.11,
            continuity_wrist_max_step_m=0.14,
            continuity_switch_penalty=8.0,
            track_reconnect_max_gap=8,
            track_reconnect_max_2d_motion_px=55.0,
            track_reconnect_max_3d_step_m=0.095,
            track_reconnect_depth_penalty=2500.0,
        )

    @staticmethod
    def _hypothesis(center, anchors=("camera2", "camera3")):
        results = []
        for side in (0, 1):
            points = np.tile(np.asarray(center[side], dtype=np.float64), (21, 1))
            results.append({"points": points})
        return {"anchor_cameras": list(anchors), "results": results}

    @staticmethod
    def _quality_args() -> argparse.Namespace:
        return argparse.Namespace(
            min_valid_joints=12,
            max_reprojection_median_px=15.0,
            max_reprojection_p95_px=40.0,
            continuity_min_branch_valid_joints=8,
            continuity_min_palm_points=3,
            continuity_min_palm_support=2.0,
            depth_preferred_ray_angle_deg=8.0,
            depth_min_palm_three_view=2,
            depth_max_pair_spread_m=0.075,
            secondary_hand_min_valid_joints=10,
            secondary_hand_min_palm_points=3,
            secondary_hand_min_palm_angle_deg=7.0,
            secondary_hand_max_cost_gap=420.0,
        )

    @staticmethod
    def _candidate(*, selected_count=2, valid=18, angle=12.0,
                   pair_spread=0.01, palm_three=5, cost=10.0):
        points = np.tile(np.asarray([0.0, 0.0, 0.72]), (21, 1))
        result = {
            "points": points,
            "inlier_counts": np.full(21, selected_count, dtype=np.int32),
            "valid_joints": valid,
            "median_px": 3.0,
            "p95_px": 8.0,
            "palm_ray_angle_p25_deg": angle,
            "ray_angle_p25_deg": angle,
            "palm_pair_spread_p75_m": pair_spread,
            "pair_spread_p75_m": pair_spread,
            "palm_three_view_count": palm_three,
            "palm_two_view_fraction": 1.0 if selected_count <= 2 else 0.0,
            "two_view_joint_fraction": 1.0 if selected_count <= 2 else 0.0,
            "geometry_rejected_joints": 0,
        }
        return {
            "cost": cost,
            "result": result,
            "selected": {f"camera{i}": {} for i in range(selected_count)},
            "anchor_cameras": ["camera0", "camera1"],
            "side": 0,
        }

    @staticmethod
    def _depth_guard_args(**overrides):
        values = {
            "temporal_depth_guard": 1,
            "temporal_depth_guard_radius": 4,
            "temporal_depth_guard_trigger_m": 0.055,
            "temporal_depth_guard_max_2d_motion_px": 75.0,
            "temporal_depth_guard_max_step_m": 0.115,
            "temporal_depth_guard_max_bracket_speed_m": 0.120,
            "temporal_depth_guard_max_shift_m": 0.160,
            "temporal_depth_guard_max_run_length": 3,
            "temporal_depth_guard_max_reprojection_regression_px": 12.0,
            "temporal_depth_guard_max_reprojection_p95_regression_px": 24.0,
            "temporal_depth_guard_min_shared_cameras": 2,
            "depth_regularization_blend": 0.85,
            "depth_min_palm_angle_deg": 4.5,
            "depth_min_palm_three_view": 2,
            "depth_max_pair_spread_m": 0.075,
            "depth_unsafe_pair_spread_m": 0.090,
            "secondary_hand_min_palm_angle_deg": 7.0,
            "min_valid_joints": 12,
        }
        values.update(overrides)
        return argparse.Namespace(**values)

    @staticmethod
    def _guard_pose(offset=(0.0, 0.0, 0.0)):
        """Compact, physically plausible 21-joint pose for temporal tests."""
        base = np.asarray([
            [0.00, 0.00, 0.72], [-0.018, -0.010, 0.72], [-0.030, -0.020, 0.72],
            [-0.040, -0.030, 0.72], [-0.050, -0.040, 0.72],
            [-0.022, -0.018, 0.72], [-0.022, -0.040, 0.72],
            [-0.022, -0.062, 0.72], [-0.022, -0.084, 0.72],
            [0.000, -0.020, 0.72], [0.000, -0.043, 0.72],
            [0.000, -0.066, 0.72], [0.000, -0.089, 0.72],
            [0.022, -0.018, 0.72], [0.022, -0.040, 0.72],
            [0.022, -0.062, 0.72], [0.022, -0.084, 0.72],
            [0.044, -0.014, 0.72], [0.044, -0.034, 0.72],
            [0.044, -0.054, 0.72], [0.044, -0.074, 0.72],
        ], dtype=np.float64)
        return base + np.asarray(offset, dtype=np.float64)

    @staticmethod
    def _guard_hand(
        points, cameras, *, projection_points=None, image_shift=(0.0, 0.0),
        segment=0,
    ):
        views = {}
        true_points = np.asarray(
            points if projection_points is None else projection_points,
            dtype=np.float64,
        )
        # ``points`` are the serialized 3-D estimate; views intentionally use
        # a separate projection pose in the caller to emulate a wrong depth
        # branch with continuous detector pixels.
        for camera_id, camera in cameras.items():
            pixels, valid = project_points(
                camera, true_points - camera.T_base_camera[:3, 3]
            )
            assert valid.all()
            pixels = pixels + np.asarray(image_shift, dtype=np.float64)
            views[camera_id] = {
                "joints_2d": pixels.tolist(),
                "bbox_xyxy": [100.0 + image_shift[0], 100.0 + image_shift[1],
                              500.0 + image_shift[0], 700.0 + image_shift[1]],
            }
        return {
            "side": 0,
            "joints_base_m": np.asarray(points, dtype=np.float64).tolist(),
            "views": views,
            "quality": {
                "valid_joint_count": 21,
                "palm_ray_angle_p25_deg": 15.0,
                "ray_angle_p25_deg": 15.0,
                "palm_three_view_count": 5,
                "palm_two_view_fraction": 0.0,
                "pair_spread_p75_m": 0.01,
                "palm_pair_spread_p75_m": 0.01,
            },
            "temporal_segment_id": int(segment),
        }

    def test_synchronization_uses_each_frame_at_most_once(self):
        rows = {
            "camera0": [{"timestamp_ns": value} for value in (1000, 2000, 3000)],
            "camera1": [{"timestamp_ns": value} for value in (990, 2010, 3020)],
            "camera2": [{"timestamp_ns": value} for value in (995, 2005, 3015)],
        }
        result = synchronize_rows(rows, tuple(rows), "camera1", 25)
        self.assertEqual(len(result), 3)
        self.assertEqual([row["camera0_frame_index"] for row in result], [0, 1, 2])
        self.assertEqual([row["camera2_frame_index"] for row in result], [0, 1, 2])

    def test_native_ds_multiview_triangulation(self):
        cameras = [ds_camera(f"camera{index}", center) for index, center in enumerate((-0.09, -0.03, 0.03, 0.09))]
        expected = np.asarray([0.04, -0.025, 0.72])
        observations = []
        for camera in cameras:
            point_camera = expected - camera.T_base_camera[:3, 3]
            pixel, valid = project_points(camera, point_camera[None])
            self.assertTrue(bool(valid[0]))
            observations.append((camera.camera_id, camera, pixel[0]))
        recovered, errors, inliers = triangulate_ransac(observations, 0.01)
        np.testing.assert_allclose(recovered, expected, atol=1e-10)
        self.assertTrue(inliers.all())
        self.assertLess(float(errors.max()), 1e-9)

    def test_nearly_parallel_rays_do_not_produce_far_depth(self):
        cameras = [
            ds_camera("camera0", -0.0005),
            ds_camera("camera1", 0.0005),
        ]
        expected = np.asarray([0.02, -0.01, 20.0])
        observations = []
        for camera in cameras:
            point_camera = expected - camera.T_base_camera[:3, 3]
            pixel, valid = project_points(camera, point_camera[None])
            self.assertTrue(bool(valid[0]))
            observations.append((camera.camera_id, camera, pixel[0]))
        recovered, _errors, inliers = triangulate_ransac(observations, 0.01)
        self.assertFalse(np.isfinite(recovered).all())
        self.assertFalse(inliers.any())

    def test_low_angle_candidate_is_not_safe_secondary_hand(self):
        args = self._quality_args()
        weak = self._candidate(angle=2.0, selected_count=2, palm_three=0)
        strong = self._candidate(angle=14.0, selected_count=4, palm_three=5, cost=12.0)
        self.assertIn(
            "near_parallel_palm_rays",
            _depth_condition_reasons(weak["result"], 2, args),
        )
        self.assertFalse(_independent_candidate_quality(weak, args)["single_safe"])
        allowed, reason, _first, _second = _independent_pair_decision(
            strong, weak, args
        )
        self.assertFalse(allowed)
        self.assertIn(reason, {"weak_secondary_hand", "weak_secondary_depth_condition"})

    def test_two_low_angle_candidates_are_not_emitted_as_complete_pair(self):
        args = self._quality_args()
        first = self._candidate(selected_count=2, angle=5.0, palm_three=0, cost=10.0)
        second = self._candidate(selected_count=2, angle=5.2, palm_three=0, cost=11.0)
        allowed, reason, _first, _second = _independent_pair_decision(
            first, second, args
        )
        self.assertFalse(allowed)
        self.assertEqual(reason, "weak_secondary_depth_condition")

    def test_two_well_conditioned_candidates_can_form_pair(self):
        args = self._quality_args()
        first = self._candidate(selected_count=4, angle=14.0, palm_three=5, cost=10.0)
        second = self._candidate(selected_count=4, angle=13.0, palm_three=5, cost=11.0)
        allowed, reason, _first, _second = _independent_pair_decision(
            first, second, args
        )
        self.assertTrue(allowed)
        self.assertEqual(reason, "both_reliable")

    def test_continuity_alternate_rejects_large_pair_spread(self):
        args = self._quality_args()
        bad = self._candidate(selected_count=4, angle=12.0, palm_three=3,
                              pair_spread=0.14)
        hypothesis = {
            "results": [bad["result"], bad["result"]],
            "selected": {0: bad["selected"], 1: bad["selected"]},
        }
        reasons = _hypothesis_branch_quality_reasons(hypothesis, args)
        self.assertTrue(any("depth_pair_disagreement" in value for value in reasons))

    def test_row_hypothesis_preserves_2d_views_for_identity_cost(self):
        points = np.tile(np.asarray([0.0, 0.0, 0.72]), (21, 1)).tolist()
        views = {
            "camera0": {
                "joints_2d": np.zeros((21, 2)).tolist(),
                "bbox_xyxy": [0.0, 0.0, 100.0, 100.0],
            }
        }
        row = {
            "sync_index": 0,
            "anchor_cameras": ["camera0", "camera1"],
            "hands": [{
                "side": 0,
                "joints_base_m": points,
                "inlier_view_counts": [2] * 21,
                "quality": {
                    "valid_joint_count": 21,
                    "multiview_reprojection_median_px": 1.0,
                    "multiview_reprojection_p95_px": 2.0,
                },
                "views": views,
            }],
        }
        hypothesis = _row_to_hypothesis(row)
        self.assertIsNotNone(hypothesis)
        self.assertIn("camera0", hypothesis["selected"][0])

    def test_continuous_2d_but_discontinuous_3d_is_depth_contradiction(self):
        def hand(depth: float, image_shift: float):
            points = np.tile([0.0, 0.0, depth], (21, 1))
            joints = np.tile([320.0 + image_shift, 240.0], (21, 1))
            return {
                "joints_base_m": points.tolist(),
                "views": {
                    camera: {
                        "joints_2d": joints.tolist(),
                        "bbox_xyxy": [250.0 + image_shift, 160.0,
                                      390.0 + image_shift, 360.0],
                    }
                    for camera in ("camera0", "camera1")
                },
            }

        contradictory, diagnostics = _depth_reconnect_contradiction(
            hand(0.70, 0.0), hand(0.94, 8.0), 1,
            self._continuity_args(),
        )
        self.assertTrue(contradictory)
        self.assertEqual(diagnostics["shared_camera_count"], 2)
        self.assertLess(diagnostics["robust_px_per_frame"], 10.0)

    def test_fast_image_motion_does_not_trigger_depth_contradiction(self):
        def hand(depth: float, image_x: float):
            points = np.tile([0.0, 0.0, depth], (21, 1))
            joints = np.tile([image_x, 240.0], (21, 1))
            return {
                "joints_base_m": points.tolist(),
                "views": {
                    "camera0": {
                        "joints_2d": joints.tolist(),
                        "bbox_xyxy": [image_x - 50.0, 180.0,
                                      image_x + 50.0, 330.0],
                    }
                },
            }

        contradictory, _diagnostics = _depth_reconnect_contradiction(
            hand(0.70, 200.0), hand(0.94, 420.0), 1,
            self._continuity_args(),
        )
        self.assertFalse(contradictory)

    def test_partial_identity_hard_2d_break_is_reentry(self):
        def hand(x, side):
            points = np.tile([x, 0.0, 0.72], (21, 1))
            joints = np.tile([320.0 + 1000.0 * x, 240.0], (21, 1))
            return {
                "side": side,
                "joints_base_m": points.tolist(),
                "views": {
                    "camera0": {
                        "joints_2d": joints.tolist(),
                        "bbox_xyxy": [300.0 + 1000.0 * x, 180.0,
                                      400.0 + 1000.0 * x, 330.0],
                    },
                    "camera1": {
                        "joints_2d": joints.tolist(),
                        "bbox_xyxy": [300.0 + 1000.0 * x, 180.0,
                                      400.0 + 1000.0 * x, 330.0],
                    },
                },
            }
        args = argparse.Namespace(
            identity_2d_soft_motion_px=80.0,
            identity_2d_hard_motion_px=125.0,
            identity_2d_max_gap=8,
            identity_2d_hard_break_penalty=24.0,
        )
        policy, diagnostics = _identity_2d_link_policy(
            hand(0.0, 0),
            hand(0.30, 1),
            1,
            args,
        )
        self.assertEqual(policy, "break")
        self.assertGreater(diagnostics["robust_px_per_frame"], 125.0)

    def test_2d_identity_keeps_track_and_marks_reentry(self):
        def row(frame, x, side):
            joints = np.tile([320.0 + 1000.0 * x, 240.0], (21, 1))
            hand = {
                "side": side,
                "joints_base_m": np.tile([x, 0.0, 0.72], (21, 1)).tolist(),
                "views": {
                    "camera0": {"joints_2d": joints.tolist(),
                                "bbox_xyxy": [x * 1000, 0, x * 1000 + 100, 100]},
                    "camera1": {"joints_2d": joints.tolist(),
                                "bbox_xyxy": [x * 1000, 0, x * 1000 + 100, 100]},
                },
            }
            return {"sync_index": frame, "hands": [hand]}
        rows = [row(0, 0.0, 0), row(1, 0.01, 1), row(2, 0.30, 0)]
        args = argparse.Namespace(
            identity_2d_soft_motion_px=80.0,
            identity_2d_hard_motion_px=125.0,
            identity_2d_max_gap=8,
        )
        _stabilize_hand_identities_2d(rows, args)
        self.assertEqual(rows[1]["hands"][0]["side"], 0)
        self.assertEqual(rows[2]["hands"][0]["side"], 1)
        self.assertIn("identity_reentry_2d", rows[2])

    def test_2d_identity_keeps_complete_row_orientation(self):
        """A long partial interval must not globally swap complete rows."""
        def hand(x, side):
            joints = np.tile([320.0 + 1000.0 * x, 240.0], (21, 1))
            return {
                "side": side,
                "joints_base_m": np.tile([x, 0.0, 0.72], (21, 1)).tolist(),
                "views": {
                    camera: {
                        "detection_index": side,
                        "joints_2d": joints.tolist(),
                        "bbox_xyxy": [270.0 + 1000.0 * x, 180.0,
                                      370.0 + 1000.0 * x, 330.0],
                    }
                    for camera in ("camera0", "camera1")
                },
            }

        # The two physical trajectories cross in x and the one-hand rows have
        # deliberately flipped detector labels.  Full rows are authoritative
        # orientation anchors; only the partial rows may be canonicalised.
        rows = [
            {"sync_index": 0, "hands": [hand(-0.10, 0), hand(0.10, 1)]},
            {"sync_index": 1, "hands": [hand(-0.08, 1)]},
            {"sync_index": 2, "hands": [hand(-0.06, 1)]},
            {"sync_index": 3, "hands": [hand(-0.04, 0), hand(0.04, 1)]},
        ]
        args = argparse.Namespace(
            identity_2d_soft_motion_px=80.0,
            identity_2d_hard_motion_px=125.0,
            identity_2d_max_gap=8,
        )
        _stabilize_hand_identities_2d(rows, args)
        self.assertEqual([hand["side"] for hand in rows[0]["hands"]], [0, 1])
        self.assertEqual([hand["side"] for hand in rows[3]["hands"]], [0, 1])
        self.assertEqual(rows[1]["hands"][0]["side"], 0)
        self.assertEqual(rows[2]["hands"][0]["side"], 0)

    def test_hand_geometry_gate_removes_isolated_far_joint(self):
        cameras = {
            f"camera{index}": ds_camera(f"camera{index}", center)
            for index, center in enumerate((-0.09, -0.03, 0.03, 0.09))
        }
        points = np.tile(np.asarray([0.0, 0.0, 0.72]), (21, 1))
        points[:, 0] = np.linspace(-0.06, 0.06, 21)
        points[16] = np.asarray([0.90, 0.0, 0.72])
        selected = {}
        for camera_id, camera in cameras.items():
            pixels, valid = project_points(
                camera, points - camera.T_base_camera[:3, 3]
            )
            self.assertTrue(valid.all())
            selected[camera_id] = {"joints_2d": pixels.tolist()}
        result = _triangulate_hand(1, selected, cameras, 0.01, 2)
        self.assertFalse(np.isfinite(result["points"][16]).all())
        self.assertEqual(result["inlier_counts"][16], 0)
        self.assertGreaterEqual(result["geometry_rejected_joints"], 1)

    def test_guided_matching_rejects_background_candidate(self):
        camera = ds_camera("camera0", -0.09)
        left_points = np.tile(np.asarray([-0.06, -0.02, 0.70]), (21, 1))
        right_points = np.tile(np.asarray([0.08, -0.01, 0.72]), (21, 1))
        left_pixels, _ = project_points(
            camera, left_points - camera.T_base_camera[:3, 3]
        )
        right_pixels, _ = project_points(
            camera, right_points - camera.T_base_camera[:3, 3]
        )
        false_pixels = left_pixels + np.asarray([350.0, -220.0])
        groups = {
            0: {
                0: {"detection_index": 0, "confidence": 0.8, "detector_is_right": 0, "joints_2d": left_pixels.tolist()},
                1: {"detection_index": 0, "confidence": 0.8, "detector_is_right": 0, "joints_2d": false_pixels.tolist()},
            },
            1: {
                0: {"detection_index": 1, "confidence": 0.7, "detector_is_right": 1, "joints_2d": false_pixels.tolist()},
                1: {"detection_index": 1, "confidence": 0.7, "detector_is_right": 1, "joints_2d": right_pixels.tolist()},
            },
        }
        selected, errors = _match_camera(
            "camera0", groups, {0: {"points": left_points}, 1: {"points": right_points}},
            camera, 55.0, 4, "strict",
        )
        self.assertEqual(selected[0]["detection_index"], 0)
        self.assertEqual(selected[1]["detection_index"], 1)
        self.assertLess(errors[0], 1e-9)
        self.assertLess(errors[1], 1e-9)
        self.assertEqual(_ordered_hand_pairs([0, 1], groups, "strict"), [(0, 1)])

    def test_marker_evidence_mildly_rewards_well_matched_hypothesis(self):
        selected = {
            0: {
                "camera0": {
                    "marker_assist": {
                        "applied": True,
                        "matched_marker_count": 12,
                        "finger_group_count": 5,
                        "match_residual_median_px": 4.0,
                    }
                }
            },
            1: {"camera0": {"marker_assist": {"applied": False}}},
        }
        applied_cost = _marker_assist_cost(selected)
        self.assertLess(applied_cost, 0.0)
        selected[0]["camera0"]["marker_assist"] = {
            "applied": False,
            "evidence_only": True,
            "matched_marker_count": 12,
            "finger_group_count": 5,
            "match_residual_median_px": 4.0,
        }
        evidence_only_cost = _marker_assist_cost(selected)
        self.assertLess(evidence_only_cost, 0.0)
        self.assertLess(abs(evidence_only_cost), abs(applied_cost))
        self.assertEqual(_marker_assist_cost({0: {}, 1: {}}), 0.0)

    def test_overlapping_duplicate_detector_boxes_are_not_two_hands(self):
        first = {
            "detection_index": 2,
            "bbox_xyxy": [100.0, 120.0, 400.0, 600.0],
            "joints_2d": (np.arange(42, dtype=float).reshape(21, 2) + 100.0).tolist(),
        }
        second = {
            "detection_index": 7,
            "bbox_xyxy": [105.0, 118.0, 398.0, 602.0],
            "joints_2d": (np.arange(42, dtype=float).reshape(21, 2) + 101.0).tolist(),
        }
        self.assertTrue(_same_detector_observation(first, second))
        self.assertEqual(
            _duplicate_hand_observation_count(
                {0: {"camera0": first}, 1: {"camera0": second}},
            ),
            1,
        )

    def test_separated_detector_boxes_remain_distinct_hands(self):
        first = {
            "detection_index": 0,
            "bbox_xyxy": [0.0, 0.0, 200.0, 300.0],
            "joints_2d": (np.zeros((21, 2)) + 50.0).tolist(),
        }
        second = {
            "detection_index": 1,
            "bbox_xyxy": [900.0, 0.0, 1200.0, 300.0],
            "joints_2d": (np.zeros((21, 2)) + 1000.0).tolist(),
        }
        self.assertFalse(_same_detector_observation(first, second))

    def test_near_coincident_cross_camera_branches_are_not_complete_pair(self):
        args = argparse.Namespace(
            duplicate_center_distance_m=0.055,
            duplicate_center_min_distinct_cameras=2,
            duplicate_bbox_iou=0.65,
            duplicate_joint_distance_px=18.0,
        )
        first_points = np.tile([0.00, 0.00, 0.72], (21, 1))
        second_points = np.tile([0.028, 0.00, 0.72], (21, 1))
        first = {
            "result": {"points": first_points},
            "selected": {
                "camera2": {"detection_index": 0},
                "camera4": {"detection_index": 0},
            },
        }
        second = {
            "result": {"points": second_points},
            "selected": {
                "camera1": {"detection_index": 1},
                "camera3": {"detection_index": 1},
            },
        }
        unsafe, diagnostics = _near_coincident_independent_pair_is_unsafe(
            first, second, args
        )
        self.assertTrue(unsafe)
        self.assertEqual(diagnostics["shared_camera_count"], 0)
        self.assertEqual(diagnostics["distinct_camera_count"], 0)

    def test_near_coincident_branches_with_two_distinct_views_are_allowed(self):
        args = argparse.Namespace(
            duplicate_center_distance_m=0.055,
            duplicate_center_min_distinct_cameras=2,
            duplicate_bbox_iou=0.65,
            duplicate_joint_distance_px=18.0,
        )
        points = np.tile([0.00, 0.00, 0.72], (21, 1))
        first = {
            "result": {"points": points},
            "selected": {
                "camera2": {"detection_index": 0, "bbox_xyxy": [0, 0, 100, 100]},
                "camera3": {"detection_index": 0, "bbox_xyxy": [0, 0, 100, 100]},
            },
        }
        second = {
            "result": {"points": points + [0.02, 0.00, 0.00]},
            "selected": {
                "camera2": {"detection_index": 1, "bbox_xyxy": [300, 0, 400, 100]},
                "camera3": {"detection_index": 1, "bbox_xyxy": [300, 0, 400, 100]},
            },
        }
        unsafe, diagnostics = _near_coincident_independent_pair_is_unsafe(
            first, second, args
        )
        self.assertFalse(unsafe)
        self.assertEqual(diagnostics["distinct_camera_count"], 2)

    def test_temporal_depth_guard_repairs_isolated_wrong_branch(self):
        cameras = {
            f"camera{index}": ds_camera(f"camera{index}", center)
            for index, center in enumerate((-0.09, -0.03, 0.03, 0.09))
        }
        true_before = self._guard_pose((0.00, 0.00, 0.00))
        true_after = self._guard_pose((0.008, 0.002, 0.002))
        wrong = self._guard_pose((0.004, 0.001, 0.20))
        rows = [
            {"sync_index": 0, "hands": [self._guard_hand(true_before, cameras)]},
            {"sync_index": 1, "hands": [self._guard_hand(
                wrong, cameras, projection_points=self._guard_pose((0.004, 0.001, 0.001))
            )]},
            {"sync_index": 2, "hands": [self._guard_hand(true_after, cameras)]},
        ]
        changed = _apply_temporal_depth_guard(
            rows, cameras, self._depth_guard_args()
        )
        self.assertEqual(changed, 1)
        corrected = np.asarray(rows[1]["hands"][0]["joints_base_m"])
        before_error = np.linalg.norm(
            np.nanmedian(wrong[[0, 5, 9, 13, 17]], axis=0)
            - np.nanmedian(true_before[[0, 5, 9, 13, 17]], axis=0)
        )
        after_error = np.linalg.norm(
            np.nanmedian(corrected[[0, 5, 9, 13, 17]], axis=0)
            - np.nanmedian(true_before[[0, 5, 9, 13, 17]], axis=0)
        )
        self.assertLess(after_error, before_error)
        self.assertIn("temporal_depth_guard", rows[1]["hands"][0])

    def test_temporal_depth_guard_does_not_smooth_fast_image_motion(self):
        cameras = {
            f"camera{index}": ds_camera(f"camera{index}", center)
            for index, center in enumerate((-0.09, -0.03, 0.03, 0.09))
        }
        before = self._guard_pose()
        current = self._guard_pose((0.004, 0.001, 0.20))
        after = self._guard_pose((0.008, 0.002, 0.002))
        rows = [
            {"sync_index": 0, "hands": [self._guard_hand(before, cameras)]},
            {"sync_index": 1, "hands": [self._guard_hand(
                current, cameras, projection_points=self._guard_pose((0.004, 0.001, 0.001)),
                image_shift=(220.0, 0.0),
            )]},
            {"sync_index": 2, "hands": [self._guard_hand(after, cameras)]},
        ]
        changed = _apply_temporal_depth_guard(
            rows, cameras, self._depth_guard_args()
        )
        self.assertEqual(changed, 0)
        self.assertNotIn("temporal_depth_guard", rows[1]["hands"][0])

    def test_continuity_penalizes_low_support_depth_branch(self):
        previous = self._hypothesis(((-0.1, 0.0, 0.7), (0.1, 0.0, 0.7)))
        continuous = self._hypothesis(((-0.08, 0.0, 0.7), (0.12, 0.0, 0.7)))
        wrong_depth = self._hypothesis(((-0.08, 0.0, 0.95), (0.12, 0.0, 0.95)))
        self.assertEqual(
            _hypothesis_transition_cost(
                previous, continuous, 1, self._continuity_args()
            ),
            0.0,
        )
        self.assertGreater(
            _hypothesis_transition_cost(
                previous, wrong_depth, 1, self._continuity_args()
            ),
            100.0,
        )

    def test_camera_switch_is_only_mildly_penalized_when_geometry_is_continuous(self):
        previous = self._hypothesis(((-0.1, 0.0, 0.7), (0.1, 0.0, 0.7)))
        switched = self._hypothesis(
            ((-0.08, 0.0, 0.7), (0.12, 0.0, 0.7)),
            ("camera1", "camera3"),
        )
        self.assertEqual(
            _hypothesis_transition_cost(
                previous, switched, 1, self._continuity_args()
            ),
            self._continuity_args().continuity_switch_penalty,
        )

    def test_transition_is_invariant_to_detector_side_permutation(self):
        previous = self._hypothesis(((-0.1, 0.0, 0.7), (0.1, 0.0, 0.7)))
        # The physical poses are unchanged, but the serialized side labels are
        # exchanged in the next frame.
        swapped = self._hypothesis(((0.1, 0.0, 0.7), (-0.1, 0.0, 0.7)))
        self.assertEqual(
            _hypothesis_transition_cost(
                previous, swapped, 1, self._continuity_args()
            ),
            0.0,
        )

    def test_fast_monotonic_motion_has_bounded_robust_penalty(self):
        previous = self._hypothesis(((-0.1, 0.0, 0.7), (0.1, 0.0, 0.7)))
        fast = self._hypothesis(((0.9, 0.0, 0.7), (1.1, 0.0, 0.7)))
        penalty = _hypothesis_transition_cost(
            previous, fast, 1, self._continuity_args()
        )
        self.assertTrue(np.isfinite(penalty))
        self.assertLessEqual(penalty, 15000.0)

    def test_adaptive_handedness_retries_detector_rejected_frame(self):
        cameras = {
            "camera0": ds_camera("camera0", -0.06),
            "camera1": ds_camera("camera1", 0.06),
        }
        base = np.asarray([
            [0.00, 0.00, 0.72], [-0.03, -0.01, 0.72], [-0.04, -0.02, 0.72],
            [-0.05, -0.03, 0.72], [-0.06, -0.04, 0.72],
            [-0.02, -0.02, 0.72], [-0.02, -0.04, 0.72], [-0.02, -0.06, 0.72],
            [-0.02, -0.08, 0.72], [0.00, -0.02, 0.72], [0.00, -0.04, 0.72],
            [0.00, -0.06, 0.72], [0.00, -0.08, 0.72], [0.02, -0.02, 0.72],
            [0.02, -0.04, 0.72], [0.02, -0.06, 0.72], [0.02, -0.08, 0.72],
            [0.04, -0.02, 0.72], [0.04, -0.04, 0.72], [0.04, -0.06, 0.72],
            [0.04, -0.08, 0.72],
        ])
        physical_hands = (base - [0.12, 0.0, 0.0], base + [0.12, 0.0, 0.0])
        prediction_rows = {}
        for camera_id, camera in cameras.items():
            hands = []
            for detection, points in enumerate(physical_hands):
                pixels, valid = project_points(
                    camera, points - camera.T_base_camera[:3, 3]
                )
                self.assertTrue(valid.all())
                for is_right in (0, 1):
                    hands.append({
                        "detection_index": detection,
                        "detector_is_right": 1,
                        "is_right": is_right,
                        "confidence": 0.9,
                        "bbox_xyxy": [0, 0, 100, 100],
                        "joints_2d": pixels.tolist(),
                    })
            prediction_rows[camera_id] = {0: {"hands": hands}}
        args = argparse.Namespace(
            max_anchor_detections=3,
            max_side_detections=4,
            detector_handedness="adaptive",
            association_threshold_px=55.0,
            anchor_threshold_px=60.0,
            ransac_threshold_px=20.0,
            min_valid_joints=12,
            max_reprojection_median_px=15.0,
            max_reprojection_p95_px=40.0,
        )

        accepted, result = _fuse_primary_frame(
            0, prediction_rows, cameras, tuple(cameras),
            ("camera0", "camera1"), args,
        )

        self.assertTrue(accepted)
        self.assertEqual(result["handedness_mode"], "ignore_fallback")

    def test_partial_hand_keeps_visible_side_when_other_side_is_absent(self):
        cameras = {
            "camera0": ds_camera("camera0", -0.08),
            "camera1": ds_camera("camera1", 0.08),
        }
        points = np.asarray([
            [0.0, 0.0, 0.72], [-0.02, -0.01, 0.72], [-0.03, -0.02, 0.72],
            [-0.04, -0.03, 0.72], [-0.05, -0.04, 0.72], [-0.02, -0.02, 0.72],
            [-0.02, -0.04, 0.72], [-0.02, -0.06, 0.72], [-0.02, -0.08, 0.72],
            [0.0, -0.02, 0.72], [0.0, -0.04, 0.72], [0.0, -0.06, 0.72],
            [0.0, -0.08, 0.72], [0.02, -0.02, 0.72], [0.02, -0.04, 0.72],
            [0.02, -0.06, 0.72], [0.02, -0.08, 0.72], [0.04, -0.02, 0.72],
            [0.04, -0.04, 0.72], [0.04, -0.06, 0.72], [0.04, -0.08, 0.72],
        ])
        prediction_rows = {}
        for camera_id, camera in cameras.items():
            pixels, valid = project_points(
                camera, points - camera.T_base_camera[:3, 3]
            )
            self.assertTrue(valid.all())
            prediction_rows[camera_id] = {0: {"hands": [{
                "detection_index": 0,
                "detector_is_right": 0,
                "is_right": 0,
                "confidence": 0.95,
                "bbox_xyxy": [0, 0, 100, 100],
                "joints_2d": pixels.tolist(),
            }]}}
        args = argparse.Namespace(
            max_anchor_detections=3, max_side_detections=4,
            detector_handedness="strict", association_threshold_px=55.0,
            anchor_threshold_px=60.0, ransac_threshold_px=20.0,
            min_valid_joints=12, max_reprojection_median_px=15.0,
            max_reprojection_p95_px=40.0,
        )
        accepted, row = _fuse_primary_frame(
            0, prediction_rows, cameras, tuple(cameras), ("camera0", "camera1"), args
        )
        self.assertTrue(accepted)
        self.assertTrue(row["partial_hand_acceptance"])
        self.assertEqual([hand["side"] for hand in row["hands"]], [0])
        self.assertEqual(row["missing_sides"], [1])

    def test_partial_side_identity_follows_3d_track(self):
        def hand(side, x):
            return {
                "side": side,
                "joints_base_m": np.tile([x, 0.0, 0.7], (21, 1)).tolist(),
                "camera_ids": ["camera0", "camera1"],
            }
        rows = [
            {"sync_index": 0, "hands": [hand(0, 0.0), hand(1, 0.3)]},
            {"sync_index": 1, "partial_hand_acceptance": True,
             "missing_sides": [1], "hands": [hand(1, 0.01)]},
        ]
        self.assertEqual(_stabilize_partial_side_labels(rows), 1)
        self.assertEqual(rows[1]["hands"][0]["side"], 0)

    def test_full_hand_identity_follows_3d_track_when_labels_swap(self):
        def hand(side, x):
            return {
                "side": side,
                "joints_base_m": np.tile([x, 0.0, 0.7], (21, 1)).tolist(),
                "camera_ids": ["camera0", "camera1"],
                "inlier_view_counts": [2] * 21,
                "views": {"camera0": {"detector_is_right": side}},
            }
        rows = [
            {"sync_index": 0, "hands": [hand(0, 0.0), hand(1, 0.3)]},
            # Detector/serialization labels are exchanged, while the hands
            # themselves move only a few millimetres.
            {"sync_index": 1, "hands": [hand(0, 0.3), hand(1, 0.01)]},
        ]
        self.assertEqual(_stabilize_hand_identities(rows), 2)
        centers = [
            float(np.asarray(hand["joints_base_m"])[0, 0])
            for hand in rows[1]["hands"]
        ]
        self.assertEqual(centers, [0.01, 0.3])
        self.assertTrue(rows[1]["identity_tracking"]["relabelled"])

    def test_prune_unreconstructable_hand_removes_empty_row(self):
        rows = [{
            "sync_index": 3,
            "hands": [{
                "side": 0,
                "joints_base_m": np.full((21, 3), np.nan).tolist(),
                "anatomy_refinement": {"dropped_unreconstructable": True},
            }],
        }]
        kept, dropped = _prune_unusable_hands(rows)
        self.assertEqual(kept, [])
        self.assertEqual(dropped[0]["reason"], "all_hands_unreconstructable")

    def test_final_guard_does_not_use_unreconstructable_bracket(self):
        """A row removed after anatomy must not drive a replacement jump."""
        def hand(center, side=0, dropped=False):
            points = np.tile(np.asarray(center, dtype=np.float64), (21, 1))
            value = {
                "side": side,
                "joints_base_m": points.tolist(),
                "camera_ids": ["camera0", "camera1"],
                "views": {
                    camera: {
                        "joints_2d": np.tile([320.0, 240.0], (21, 1)).tolist(),
                        "bbox_xyxy": [280.0, 180.0, 360.0, 320.0],
                        "detection_index": 0,
                    }
                    for camera in ("camera0", "camera1")
                },
            }
            if dropped:
                value["anatomy_refinement"] = {"dropped_unreconstructable": True}
            return value

        # The middle frame is a large excursion.  The immediately following
        # row is marked invalid, so the guard must not use it as its bracket.
        rows = [
            {"sync_index": 0, "hands": [hand([0.0, 0.0, 0.7])]},
            {"sync_index": 1, "hands": [hand([0.3, 0.0, 0.7])]},
            {"sync_index": 2, "hands": [hand([0.3, 0.0, 0.7], dropped=True)]},
            {"sync_index": 3, "hands": [hand([0.02, 0.0, 0.7])]},
        ]
        before, after = _final_sequence_neighbours(
            rows, 1, 0, 4, excluded_positions={2}
        )
        self.assertEqual(before[0]["sync_index"], 0)
        self.assertEqual(after[0]["sync_index"], 3)


if __name__ == "__main__":
    unittest.main()
