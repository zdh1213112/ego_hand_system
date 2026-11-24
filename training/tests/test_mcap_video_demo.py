import unittest

import numpy as np

from mcap_video_demo import (
    absolute_hands, camera_topic, estimated_fps, overlay_predictions,
    select_detections, visualization_hands,
)


class McapVideoDemoTests(unittest.TestCase):
    def test_camera_topic_validation(self):
        self.assertEqual(camera_topic("camera2"), "/robot0/sensor/camera2/compressed")
        for value in ("2", "camera", "cameraA"):
            with self.assertRaises(ValueError):
                camera_topic(value)

    def test_selects_highest_confidence_per_hand_class(self):
        boxes = np.arange(20, dtype=np.float32).reshape(5, 4)
        scores = np.array([0.5, 0.9, 0.7, 0.8, 0.4], np.float32)
        sides = np.array([0, 0, 0, 1, 1], np.float32)
        selected_boxes, selected_scores, selected_sides = select_detections(
            boxes, scores, sides, 1
        )
        np.testing.assert_array_equal(selected_boxes, boxes[[1, 3]])
        np.testing.assert_array_equal(selected_scores, scores[[1, 3]])
        np.testing.assert_array_equal(selected_sides, sides[[1, 3]])

    def test_absolute_hands_adds_full_camera_translation(self):
        joints = np.arange(63, dtype=np.float64).reshape(21, 3) * 0.001
        result = absolute_hands([{
            "joints_3d": joints.tolist(), "camera_translation": [0.2, -0.1, 0.5],
            "is_right": 1,
        }])
        self.assertEqual(result[0][0], 1)
        np.testing.assert_allclose(result[0][1][0], [0.2, -0.099, 0.502])

    def test_root_relative_visualization_ignores_unreliable_camera_depth(self):
        joints = np.arange(63, dtype=np.float64).reshape(21, 3) * 0.001
        hands = [{"joints_3d": joints.tolist(), "camera_translation": [0, 0, 99],
                  "is_right": side} for side in (0, 1)]
        left, right = visualization_hands(hands, "root-relative")
        np.testing.assert_allclose(left[1][0], [-0.14, 0, 0])
        np.testing.assert_allclose(right[1][0], [0.14, 0, 0])
        np.testing.assert_allclose(left[1][1] - left[1][0], [0.003, 0.003, 0.003])

    def test_fps_uses_timestamp_median(self):
        records = [{"timestamp_ns": value} for value in (1_000_000_000, 1_033_333_333, 1_066_666_666)]
        self.assertAlmostEqual(estimated_fps(records, 25), 30.0000003, places=4)
        self.assertEqual(estimated_fps(records[:1], 25), 25)

    def test_duplicate_classes_keep_identity_with_separate_display_wrists(self):
        joints = np.zeros((21, 3)).tolist()
        hands = [
            {"joints_3d": joints, "is_right": 1, "bbox_xyxy": [500, 0, 600, 100]},
            {"joints_3d": joints, "is_right": 1, "bbox_xyxy": [100, 0, 200, 100]},
        ]
        result = visualization_hands(hands, "root-relative")
        self.assertEqual([side for side, _ in result], [1, 1])
        np.testing.assert_allclose(result[0][1][0], [0.14, 0, 0])
        np.testing.assert_allclose(result[1][1][0], [-0.14, 0, 0])

    def test_overlay_draws_box_and_projected_joints_without_mutating_rgb(self):
        image = np.zeros((160, 240, 3), dtype=np.uint8)
        original = image.copy()
        joints = np.stack((np.linspace(40, 190, 21), np.linspace(40, 120, 21)), axis=1)
        output = overlay_predictions(image, [{
            "bbox_xyxy": [30, 25, 210, 135], "joints_2d": joints.tolist(),
            "is_right": 1, "confidence": 0.93,
        }])
        np.testing.assert_array_equal(image, original)
        self.assertTrue(np.any(output != image))
        self.assertTrue(np.any(output[25, 30:211] != 0))


if __name__ == "__main__":
    unittest.main()
