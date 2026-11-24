"""CPU-only checks; run with python -m unittest discover -s tests -p test_hand_3d_video.py."""

import json
from pathlib import Path
import tempfile
import unittest

import av
import cv2
import numpy as np

from visualize_hand_3d import (
    Frame, camera_joints, load_hands, parse_args, process_dataset,
    project_camera_joints, read_frames, root_relative_hands, scene_bounds,
)


class HandVideoTests(unittest.TestCase):
    def test_camera_translation_and_invalid_coordinates(self):
        joints = np.arange(63).reshape(21, 3) * 0.001
        points = camera_joints({"joints_3d": joints, "trans": [0.2, -0.1, 0.5]})
        np.testing.assert_allclose(points[0], [0.2, -0.099, 0.502])
        np.testing.assert_allclose(points[20], [0.260, -0.039, 0.562])
        for bad in (np.zeros((20, 3)), np.full((21, 3), np.nan)):
            with self.assertRaises(ValueError):
                camera_joints({"joints_3d": bad, "trans": [0, 0, 0]})

    def test_projection_and_root_relative_anchors(self):
        points = np.tile([0.0, 0.0, 1.0], (21, 1))
        points[1] = [0.1, -0.2, 1.0]
        pixels = project_camera_joints(points, [[100, 0, 320], [0, 100, 240], [0, 0, 1]])
        np.testing.assert_allclose(pixels[0], [320, 240])
        np.testing.assert_allclose(pixels[1], [330, 220])
        relative = root_relative_hands([(0, points), (1, points)])
        np.testing.assert_allclose(relative[0][1][0], [-0.14, 0, 0])
        np.testing.assert_allclose(relative[1][1][0], [0.14, 0, 0])

    def test_grouping_keeps_recordings_and_cameras_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [
                {"source_dataset_id": source, "camera": camera, "sync_index": frame,
                 "source_frame_index": frame, "side": side, "image": f"{i}.jpg"}
                for i, (source, camera, frame, side) in enumerate([
                    (10, "camera2", 12, 0), (2, "camera2", 12, 0),
                    (2, "camera3", 12, 0), (2, "camera2", 12, 1),
                    (2, "camera2", 2, 1),
                ])
            ]
            (root / "index.jsonl").write_text("\n".join(map(json.dumps, rows)))
            frames = read_frames(root)
            self.assertEqual(len(frames), 4)
            self.assertEqual([f.index for f in frames], [2, 12, 12, 12])
            self.assertEqual(len(frames[1].rows), 2)
            self.assertEqual(frames[-1].source, "source_dataset_id:10")

    def test_bounds_cover_all_hands_and_frames_with_equal_scale(self):
        a = np.tile([1, 2, 3], (21, 1))
        b = np.tile([-1, 0, 5], (21, 1))
        center, radius = scene_bounds([[(0, a)], [(1, b)], []])
        np.testing.assert_allclose(center, [0, 4, -1])
        self.assertAlmostEqual(radius, 1.2)

    def test_corrupt_label_and_duplicate_side_are_skipped(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.save(root / "valid.npy", {"joints_3d": np.zeros((21, 3)),
                                         "trans": np.zeros(3), "side": 0})
            (root / "corrupt.npy").write_bytes(b"invalid label")
            frame = Frame("recording", "camera2", 0, [
                {"label": "corrupt.npy"}, {"label": "valid.npy"}, {"label": "valid.npy"},
            ])
            stats = {"skipped_hands": 0}
            with self.assertLogs("hand_3d", level="WARNING"):
                hands = load_hands(root, frame, stats)
            self.assertEqual(len(hands), 1)
            self.assertEqual(stats["skipped_hands"], 2)

    def test_video_handles_two_hands_invalid_label_and_missing_rgb(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "sample"
            root.mkdir()
            output = Path(directory) / "videos"
            image = np.full((100, 180, 3), (30, 80, 170), np.uint8)
            cv2.imwrite(str(root / "rgb.jpg"), image)
            for side in (0, 1):
                points = np.zeros((21, 3))
                points[:, 0] = np.linspace(0, 0.1, 21)
                np.save(root / f"{side}.npy", {"joints_3d": points, "trans": [side * 0.2, 0, 0.5], "side": side})
            np.save(root / "bad.npy", {"joints_3d": np.zeros((20, 3)), "trans": np.zeros(3)})
            rows = [
                {"sync_index": 0, "side": 0, "image": "rgb.jpg", "label": "0.npy"},
                {"sync_index": 0, "side": 1, "image": "rgb.jpg", "label": "1.npy"},
                {"sync_index": 1, "side": 1, "image": "rgb.jpg", "label": "1.npy"},
                {"sync_index": 2, "side": 0, "image": "rgb.jpg", "label": "bad.npy"},
                {"sync_index": 3, "side": 0, "image": "missing.jpg", "label": "0.npy"},
            ]
            (root / "index.jsonl").write_text("\n".join(map(json.dumps, rows)))
            args = parse_args(["--output-dir", str(output), "--width", "640", "--height", "480"])
            with self.assertLogs("hand_3d", level="WARNING"):
                stats = process_dataset(root, args)
            self.assertEqual(stats["written_frames"], 3)
            self.assertEqual(stats["skipped_frames"], 1)
            self.assertEqual(stats["skipped_hands"], 1)
            self.assertEqual(stats["empty_frames"], 1)
            self.assertEqual([p.suffix for p in output.iterdir()], [".mp4"])
            with av.open(stats["output"]) as container:
                stream = container.streams.video[0]
                self.assertEqual(stream.codec_context.name, "h264")
                self.assertEqual(stream.codec_context.pix_fmt, "yuv420p")
                self.assertEqual(stream.frames, 3)
            capture = cv2.VideoCapture(stats["output"])
            try:
                self.assertAlmostEqual(capture.get(cv2.CAP_PROP_FPS), 10)
                decoded = []
                while True:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    self.assertEqual(frame.shape, (480, 640, 3))
                    decoded.append(frame)
                self.assertEqual(len(decoded), 3)
                # The original RGB remains on the left, with BGR channel order intact.
                np.testing.assert_allclose(decoded[0][220, 160], [30, 80, 170], atol=12)
                self.assertFalse(np.array_equal(decoded[0][:, 320:], decoded[2][:, 320:]))
            finally:
                capture.release()


if __name__ == "__main__":
    unittest.main()
