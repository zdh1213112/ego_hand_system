from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from fusion_temporal_guard import guard_temporal_rows  # noqa: E402


def make_hand(side: int, center: float, cameras: int = 6) -> dict:
    """Build a small synthetic hand with a known palm center on x."""
    points = np.zeros((21, 3), dtype=np.float64)
    points[:, 0] = center
    return {
        "side": side,
        "joints_base_m": points.tolist(),
        "camera_ids": [f"camera{index}" for index in range(cameras)],
        "inlier_view_counts": [3] * 21,
        "quality": {"multiview_reprojection_median_px": 5.0},
        "views": {},
    }


class FusionTemporalGuardTests(unittest.TestCase):
    def test_interpolates_low_support_depth_spike(self):
        rows = []
        side1_centers = (0.0, 0.005, 0.10, 0.015, 0.020)
        for sync_index, side1_center in enumerate(side1_centers):
            rows.append({
                "sync_index": sync_index,
                "hands": [
                    make_hand(0, 0.0),
                    make_hand(1, side1_center, cameras=2 if sync_index == 2 else 6),
                ],
            })

        summary = guard_temporal_rows(
            rows,
            radius=4,
            replacement_threshold_m=0.020,
            min_anchor_cameras=5,
            min_anchor_palm_inliers=3,
        )

        corrected = next(
            hand for hand in rows[2]["hands"] if hand["side"] == 1
        )
        np.testing.assert_allclose(
            np.asarray(corrected["joints_base_m"])[0], [0.01, 0.0, 0.0]
        )
        self.assertEqual(
            corrected["temporal_guard"]["method"],
            "trusted_anchor_interpolation",
        )
        self.assertEqual(summary["corrected_hand_count"], 1)
        self.assertLess(
            summary["motion_after"]["1"]["max_mm_per_frame"],
            summary["motion_before"]["1"]["max_mm_per_frame"],
        )

    def test_duplicate_detector_frame_is_not_trusted_as_anchor(self):
        left = make_hand(0, 0.0)
        right = make_hand(1, 0.2)
        left["views"] = {
            "camera2": {
                "detection_index": 4,
                "bbox_xyxy": [10, 10, 100, 100],
            }
        }
        right["views"] = {
            "camera2": {
                "detection_index": 4,
                "bbox_xyxy": [10, 10, 100, 100],
            }
        }
        rows = [{"sync_index": 0, "hands": [left, right]}]

        summary = guard_temporal_rows(rows, duplicate_iou=0.90)

        self.assertEqual(summary["duplicate_detector_frame_count"], 1)
        self.assertEqual(summary["trusted_anchor_frame_count"], {"0": 0, "1": 0})

    def test_unrecoverable_two_view_tail_is_cleared(self):
        rows = []
        for sync_index, side1_center in enumerate((0.0, 0.01, 0.12)):
            rows.append({
                "sync_index": sync_index,
                "hands": [
                    make_hand(0, 0.0),
                    make_hand(1, side1_center, cameras=2 if sync_index == 2 else 6),
                ],
            })

        summary = guard_temporal_rows(
            rows,
            replacement_threshold_m=0.020,
            max_anchor_speed_m_per_frame=0.050,
            clear_unrecoverable_endpoint=True,
        )
        endpoint = next(hand for hand in rows[-1]["hands"] if hand["side"] == 1)

        self.assertEqual(summary["cleared_hand_count"], 1)
        self.assertEqual(
            endpoint["temporal_guard"]["method"],
            "clear_untrusted_temporal_endpoint",
        )
        self.assertTrue(np.isnan(np.asarray(endpoint["joints_base_m"])).all())


if __name__ == "__main__":
    unittest.main()
