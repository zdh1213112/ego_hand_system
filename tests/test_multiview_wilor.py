from __future__ import annotations

import argparse
import itertools
from pathlib import Path
import sys
import unittest

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from camera_models import project_points
from ego_data.calibration import CameraCalibration
from fuse_multiview_wilor import (
    _intersect_rays, _ray_in_base, triangulate_ransac,
)
from fuse_multiview_wilor_guided import (
    _fuse_primary_frame, _marker_assist_cost, _match_camera,
    _nearest_reference_row, _ordered_hand_pairs,
)
from normalize_multiview_recording import synchronize_rows


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
    def test_nearest_reference_uses_binary_neighbors_and_earlier_tie(self):
        references = [{"sync_index": value} for value in (2, 100, 205, 900)]
        reference, distance = _nearest_reference_row(
            references, [2, 100, 205, 900], 150
        )
        self.assertEqual(reference["sync_index"], 100)
        self.assertEqual(distance, 50)
        reference, distance = _nearest_reference_row(
            references, [2, 100, 205, 900], 207
        )
        self.assertEqual(reference["sync_index"], 205)
        self.assertEqual(distance, 2)

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

    def test_vectorized_ransac_matches_scalar_candidate_search(self):
        cameras = [
            ds_camera(f"camera{index}", center)
            for index, center in enumerate(np.linspace(-0.12, 0.12, 6))
        ]
        expected = np.asarray([0.035, -0.018, 0.73])
        rng = np.random.default_rng(17)
        observations = []
        for camera in cameras:
            point_camera = expected - camera.T_base_camera[:3, 3]
            pixel, _ = project_points(camera, point_camera[None])
            observations.append((
                camera.camera_id, camera, pixel[0] + rng.normal(0.0, 1.5, 2)
            ))
        observations[-1] = (
            observations[-1][0], observations[-1][1],
            observations[-1][2] + np.asarray([90.0, -60.0]),
        )

        rays = [
            _ray_in_base(camera, pixel)
            for _, camera, pixel in observations
        ]
        best = None
        for first, second in itertools.combinations(range(len(observations)), 2):
            point = _intersect_rays([rays[first], rays[second]])
            if point is None:
                continue
            errors = np.full(len(observations), np.inf)
            for index, (_, camera, pixel) in enumerate(observations):
                center = camera.T_base_camera[:3, 3]
                if np.dot(point - center, rays[index][1]) <= 0:
                    continue
                point_camera = camera.T_base_camera[:3, :3].T @ (point - center)
                projected, valid = project_points(camera, point_camera[None])
                if bool(valid[0]):
                    errors[index] = np.linalg.norm(projected[0] - pixel)
            inliers = np.isfinite(errors) & (errors <= 20.0)
            count = int(inliers.sum())
            score = float(np.median(errors[inliers])) if count else np.inf
            candidate = (count, -score, point, inliers)
            if best is None or candidate[:2] > best[:2]:
                best = candidate
        reference_point = _intersect_rays([
            rays[index] for index in np.flatnonzero(best[3])
        ])
        reference_errors = np.full(len(observations), np.inf)
        for index, (_, camera, pixel) in enumerate(observations):
            center = camera.T_base_camera[:3, 3]
            point_camera = camera.T_base_camera[:3, :3].T @ (
                reference_point - center
            )
            projected, valid = project_points(camera, point_camera[None])
            if bool(valid[0]) and np.dot(
                reference_point - center, rays[index][1]
            ) > 0:
                reference_errors[index] = np.linalg.norm(projected[0] - pixel)
        reference_inliers = np.isfinite(reference_errors) & (
            reference_errors <= 20.0
        )

        point, errors, inliers = triangulate_ransac(observations, 20.0)
        np.testing.assert_allclose(point, reference_point, atol=1e-12)
        np.testing.assert_allclose(errors, reference_errors, atol=1e-12)
        np.testing.assert_array_equal(inliers, reference_inliers)

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


if __name__ == "__main__":
    unittest.main()
