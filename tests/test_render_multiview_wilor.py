from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from render_multiview_wilor import (  # noqa: E402
    _has_renderable_final_pose,
    _interpolate_all_gap_results,
)
from render_native_2d_multiview import _select_camera_tracks  # noqa: E402
from render_hybrid_multiview import _drop_short_trusted_runs  # noqa: E402


def _hand(value: float) -> dict:
    points = np.full((21, 3), value, dtype=np.float64)
    return {
        "side": 0,
        "joints_base_m": points.tolist(),
        "unrefined_joints_base_m": points.tolist(),
        "views": {},
        "camera_ids": [],
    }


class RenderMultiviewTests(unittest.TestCase):
    def test_native_tracks_do_not_duplicate_one_detector_box(self):
        points = np.zeros((21, 2), dtype=np.float64).tolist()
        record = {
            0: {
                "sync_index": 0,
                "hands": [
                    {"is_right": 0, "bbox_xyxy": [10, 10, 50, 50], "joints_2d": points},
                    {"is_right": 1, "bbox_xyxy": [10, 10, 50, 50], "joints_2d": points},
                ],
            }
        }
        tracks = _select_camera_tracks(record, [0], max_hold_frames=0, wrist_jump_px=150)
        self.assertIsNotNone(tracks[0][0])
        self.assertIsNone(tracks[1][0])

    def test_native_tracks_collapse_duplicate_indices_for_same_box(self):
        points = np.zeros((21, 2), dtype=np.float64).tolist()
        record = {
            0: {
                "sync_index": 0,
                "hands": [
                    {
                        "detection_index": 0,
                        "is_right": 0,
                        "bbox_xyxy": [10, 10, 50, 50],
                        "joints_2d": points,
                    },
                    {
                        "detection_index": 1,
                        "is_right": 1,
                        "bbox_xyxy": [10, 10, 50, 50],
                        "joints_2d": points,
                    },
                ],
            }
        }
        tracks = _select_camera_tracks(record, [0], max_hold_frames=0, wrist_jump_px=150)
        self.assertIsNotNone(tracks[0][0])
        self.assertIsNone(tracks[1][0])

    def test_native_quality_gate_hides_clipped_projected_pose(self):
        points = np.zeros((21, 2), dtype=np.float64)
        points[:, 1] = 1300.0
        record = {
            0: {
                "sync_index": 0,
                "hands": [
                    {"is_right": 0, "joints_2d": points.tolist()},
                ],
            }
        }
        tracks = _select_camera_tracks(
            record, [0], max_hold_frames=0, wrist_jump_px=150,
            image_size=(1600, 1200), min_visible_joints=15,
        )
        self.assertIsNone(tracks[0][0])

    def test_native_track_reacquires_after_large_occluded_move(self):
        def detection(x: float) -> dict:
            points = np.tile([x, 100.0], (21, 1))
            return {
                "is_right": 0,
                "detector_is_right": 0,
                "confidence": 0.9,
                "bbox_xyxy": [x - 20, 80, x + 20, 120],
                "joints_2d": points.tolist(),
            }

        records = {
            index: {"sync_index": index, "hands": [detection(100.0 if index == 0 else 400.0)]}
            for index in range(5)
        }
        tracks = _select_camera_tracks(records, list(range(5)), max_hold_frames=2,
                                       wrist_jump_px=150)
        self.assertEqual([item is not None and item.get("native_2d_hold", False)
                          for item in tracks[0]], [False, True, True, False, False])
        self.assertIsNone(tracks[0][3])
        self.assertAlmostEqual(tracks[0][4]["joints_2d"][0][0], 400.0)

    def test_native_tracks_keep_detector_identity_after_occlusion(self):
        def hand(side: int, x: float, branch: int) -> dict:
            points = np.tile([x, 100.0], (21, 1))
            return {
                "is_right": branch,
                "detector_is_right": side,
                "confidence": 0.9,
                "bbox_xyxy": [x - 20, 80, x + 20, 120],
                "joints_2d": points.tolist(),
            }

        records = {
            0: {"hands": [hand(1, 300, 1)]},
            1: {"hands": [hand(0, 100, 0), hand(0, 100, 1)]},
            2: {"hands": [hand(0, 100, 0), hand(0, 100, 1), hand(1, 300, 1)]},
        }
        tracks = _select_camera_tracks(records, [0, 1, 2],
                                       max_hold_frames=0, wrist_jump_px=150)
        self.assertIsNotNone(tracks[0][1])
        self.assertIsNone(tracks[1][1])
        self.assertEqual(tracks[0][2]["detector_is_right"], 0)
        self.assertEqual(tracks[1][2]["detector_is_right"], 1)

    def test_hybrid_hides_isolated_fused_frame(self):
        self.assertEqual(
            _drop_short_trusted_runs([False, True, False, True, True, False], 2),
            [False, False, False, True, True, False],
        )

    def test_final_pose_requires_all_joints_by_default(self):
        points = np.full((21, 3), 1.0, dtype=np.float64)
        points[8] = np.nan
        self.assertFalse(_has_renderable_final_pose(points))
        self.assertTrue(_has_renderable_final_pose(points, min_joints=20))

    def test_long_gap_hold_does_not_morph_fingers(self):
        rows = [{"sync_index": index} for index in range(5)]
        accepted = {
            0: {"sync_index": 0, "hands": [_hand(0.0)]},
            4: {"sync_index": 4, "hands": [_hand(1.0)]},
        }
        predicted = _interpolate_all_gap_results(
            rows, accepted, prediction_mode="hold"
        )
        values = [
            float(np.asarray(predicted[index]["hands"][0]["joints_base_m"])[0, 0])
            for index in (1, 2, 3)
        ]
        self.assertEqual(values, [0.0, 1.0, 1.0])

    def test_linear_mode_remains_explicit_legacy_behavior(self):
        rows = [{"sync_index": index} for index in range(5)]
        accepted = {
            0: {"sync_index": 0, "hands": [_hand(0.0)]},
            4: {"sync_index": 4, "hands": [_hand(1.0)]},
        }
        predicted = _interpolate_all_gap_results(
            rows, accepted, prediction_mode="linear"
        )
        values = [
            float(np.asarray(predicted[index]["hands"][0]["joints_base_m"])[0, 0])
            for index in (1, 2, 3)
        ]
        self.assertTrue(np.allclose(values, [0.25, 0.50, 0.75]))

    def test_hold_previous_keeps_one_pose_through_gap(self):
        rows = [{"sync_index": index} for index in range(5)]
        accepted = {
            0: {"sync_index": 0, "hands": [_hand(0.0)]},
            4: {"sync_index": 4, "hands": [_hand(1.0)]},
        }
        predicted = _interpolate_all_gap_results(
            rows, accepted, prediction_mode="hold-previous"
        )
        values = [
            float(np.asarray(predicted[index]["hands"][0]["joints_base_m"])[0, 0])
            for index in (1, 2, 3)
        ]
        self.assertEqual(values, [0.0, 0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
