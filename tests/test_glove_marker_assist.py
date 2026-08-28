from __future__ import annotations

from pathlib import Path
import sys
import unittest

import cv2
import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from glove_marker_assist import (
    BrightBlobs,
    FINGER_JOINT_INDICES,
    MarkerAssistConfig,
    _remove_same_finger_reversals,
    assist_wilor_hand,
    assist_wilor_hypotheses,
    associate_marker_blobs,
    detect_bright_blobs,
)


def hand_joints() -> np.ndarray:
    points = [[100.0, 190.0]]
    finger_x = (65.0, 85.0, 105.0, 125.0, 145.0)
    for x in finger_x:
        points.extend([
            [x, 165.0], [x, 140.0], [x, 115.0], [x, 90.0],
        ])
    return np.asarray(points, dtype=np.float64)


class GloveMarkerAssistTests(unittest.TestCase):
    def test_hsv_components_keep_small_round_low_saturation_spots(self):
        config = MarkerAssistConfig()
        image = np.zeros((240, 220, 3), dtype=np.uint8)
        expected = hand_joints()[1:] + np.asarray([6.0, -4.0])
        for center in np.rint(expected).astype(np.int32):
            cv2.circle(image, tuple(center), 4, (230, 230, 230), -1, cv2.LINE_8)
        cv2.circle(image, (190, 40), 4, (0, 0, 255), -1, cv2.LINE_8)
        cv2.rectangle(image, (175, 170), (210, 173), (240, 240, 240), -1)
        cv2.rectangle(image, (0, 0), (40, 40), (220, 220, 220), -1)

        blobs = detect_bright_blobs(image, config)

        self.assertEqual(len(blobs.centers), 20)
        distances = np.linalg.norm(
            expected[:, None, :] - blobs.centers[None, :, :], axis=2
        )
        self.assertTrue(np.all(np.min(distances, axis=1) < 0.1))

    def test_shift_then_hungarian_recovers_unique_finger_markers(self):
        config = MarkerAssistConfig(min_matches=8, min_finger_groups=4)
        joints = hand_joints()
        shift = np.asarray([8.0, -6.0])
        components = joints[1:] + shift
        components[3] += [2.0, -1.0]
        components[11] += [-1.5, 1.0]
        components = np.vstack((components, [[25.0, 25.0], [205.0, 210.0]]))
        components = components[np.asarray([
            7, 2, 18, 0, 13, 5, 20, 9, 1, 16, 4,
            11, 19, 6, 15, 3, 12, 8, 17, 10, 14, 21,
        ])]

        association = associate_marker_blobs(joints, components, config)

        self.assertEqual(len(association.joint_indices), 20)
        self.assertEqual(len(set(association.component_indices.tolist())), 20)
        np.testing.assert_allclose(association.shift_px, shift, atol=0.1)
        self.assertEqual(association.finger_group_count, 5)
        self.assertLess(float(np.median(association.residuals_px)), 0.1)

    def test_reliable_matches_shift_wrist_and_blend_finger_observations(self):
        config = MarkerAssistConfig(marker_blend=0.5, min_matches=8)
        joints = hand_joints()
        shift = np.asarray([5.0, -3.0])
        observations = joints[1:] + shift
        observations[0] += [4.0, 0.0]
        blobs = BrightBlobs(
            centers=observations,
            areas=np.full(20, 20.0),
            circularities=np.full(20, 0.8),
        )
        hand = {"joints_2d": joints.tolist(), "is_right": 0}

        assisted = assist_wilor_hand(hand, blobs, config)

        self.assertTrue(assisted["marker_assist"]["applied"])
        self.assertLess(
            assisted["marker_assist"]["assisted_marker_residual_median_px"],
            assisted["marker_assist"]["raw_marker_residual_median_px"],
        )
        adjusted = np.asarray(assisted["joints_2d"])
        estimated_shift = np.asarray(assisted["marker_assist"]["coarse_shift_px"])
        np.testing.assert_allclose(adjusted[0], joints[0] + estimated_shift)
        expected_first = (
            joints[1] + estimated_shift
            + 0.5 * (observations[0] - joints[1] - estimated_shift)
        )
        np.testing.assert_allclose(adjusted[1], expected_first)
        np.testing.assert_allclose(assisted["wilor_joints_2d"], joints)

    def test_insufficient_marker_coverage_preserves_wilor_joints(self):
        config = MarkerAssistConfig(min_matches=5, min_finger_groups=3)
        joints = hand_joints()
        blobs = BrightBlobs(
            centers=joints[1:5] + np.asarray([2.0, 1.0]),
            areas=np.full(4, 20.0),
            circularities=np.full(4, 0.8),
        )

        assisted = assist_wilor_hand(
            {"joints_2d": joints.tolist(), "is_right": 0}, blobs, config
        )

        self.assertFalse(assisted["marker_assist"]["applied"])
        np.testing.assert_allclose(assisted["joints_2d"], joints)

    def test_five_matches_are_evidence_only_and_do_not_shift_whole_hand(self):
        config = MarkerAssistConfig(
            min_matches=3,
            min_finger_groups=2,
            global_min_matches=8,
            global_min_finger_groups=4,
        )
        joints = hand_joints()
        selected = np.asarray([1, 2, 5, 6, 9])
        blobs = BrightBlobs(
            centers=joints[selected] + np.asarray([9.0, -6.0]),
            areas=np.full(len(selected), 20.0),
            circularities=np.full(len(selected), 0.8),
        )

        assisted = assist_wilor_hand(
            {"joints_2d": joints.tolist(), "is_right": 0}, blobs, config
        )

        self.assertFalse(assisted["marker_assist"]["applied"])
        self.assertTrue(assisted["marker_assist"]["evidence_only"])
        self.assertEqual(
            assisted["marker_assist"]["reason_code"], "too_few_global_matches"
        )
        np.testing.assert_allclose(assisted["joints_2d"], joints)

    def test_detector_boxes_prevent_cross_hand_blob_contamination(self):
        config = MarkerAssistConfig(bbox_padding_px=8.0)
        first = hand_joints()
        second = first + np.asarray([130.0, 0.0])
        marker_shift = np.asarray([4.0, -3.0])
        image = np.zeros((260, 330, 3), dtype=np.uint8)
        for center in np.vstack((first[1:], second[1:])) + marker_shift:
            cv2.circle(
                image, tuple(np.rint(center).astype(np.int32)),
                4, (235, 235, 235), -1, cv2.LINE_8,
            )
        hands = [
            {
                "detection_index": 0,
                "bbox_xyxy": [45.0, 75.0, 155.0, 205.0],
                "joints_2d": first.tolist(),
                "is_right": 0,
            },
            {
                "detection_index": 1,
                "bbox_xyxy": [175.0, 75.0, 285.0, 205.0],
                "joints_2d": second.tolist(),
                "is_right": 1,
            },
        ]

        assisted, blobs = assist_wilor_hypotheses(image, hands, config)

        self.assertEqual(len(blobs.centers), 40)
        self.assertEqual(assisted[0]["marker_assist"]["candidate_blob_count"], 20)
        self.assertEqual(assisted[1]["marker_assist"]["candidate_blob_count"], 20)
        self.assertTrue(all(hand["marker_assist"]["applied"] for hand in assisted))

    def test_same_finger_reversal_guard_discards_one_conflicting_match(self):
        shifted = hand_joints()[1:]
        components = shifted.copy()
        rows = np.asarray([0, 3], dtype=np.int32)
        columns = np.asarray([3, 0], dtype=np.int32)
        distances = np.linalg.norm(
            shifted[:, None, :] - components[None, :, :], axis=2
        )

        kept_rows, kept_columns, rejected = _remove_same_finger_reversals(
            shifted,
            FINGER_JOINT_INDICES,
            components,
            rows,
            columns,
            distances,
            -0.10,
        )

        self.assertEqual(rejected, 1)
        self.assertEqual(len(kept_rows), 1)
        self.assertEqual(len(kept_columns), 1)

    def test_local_marker_pull_keeps_bone_deformation_bounded(self):
        config = MarkerAssistConfig(
            marker_blend=1.0,
            max_local_adjustment_px=2.0,
            max_bone_length_change_p95_ratio=0.10,
        )
        joints = hand_joints()
        observations = joints[1:] + np.asarray([5.0, -3.0])
        observations[::2] += np.asarray([7.0, 0.0])
        blobs = BrightBlobs(
            centers=observations,
            areas=np.full(20, 20.0),
            circularities=np.full(20, 0.8),
        )

        assisted = assist_wilor_hand(
            {"joints_2d": joints.tolist(), "is_right": 0}, blobs, config
        )

        self.assertTrue(assisted["marker_assist"]["applied"])
        self.assertLessEqual(
            assisted["marker_assist"]["bone_length_change_p95_ratio"], 0.10
        )
        self.assertLessEqual(
            assisted["marker_assist"]["local_adjustment_norm_max_px"], 2.0
        )


if __name__ == "__main__":
    unittest.main()
