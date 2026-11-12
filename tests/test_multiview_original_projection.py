from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from camera_models import project_points, unproject_points
from ego_data.calibration import CameraCalibration
from ego_data.dataset import SequentialVideoReader
from multiview_original_projection import OriginalVideoProjection, box_to_original, pixels_to_original
import render_hybrid_multiview as renderer
from test_multiview_undistortion import camera, make_dataset, write_json
from undistort_multiview_dataset import make_pinhole_view, undistort_dataset, undistortion_options


class OriginalProjectionTests(unittest.TestCase):
    def test_kb_and_ds_pixels_match_direct_projection(self):
        points = np.asarray([[.02, .01, .5], [-.1, .03, .8], [.1, -.02, .7]])
        for model in ("KB", "DS"):
            with self.subTest(model=model):
                raw = camera(model=model)
                pinhole, *_ = make_pinhole_view(raw, 110, (80, 40))
                pixels, _ = project_points(pinhole, points)
                actual = pixels_to_original(pixels, pinhole, raw)
                expected, _ = project_points(raw, points)
                np.testing.assert_allclose(actual, expected, atol=1e-10)

    def test_outside_inference_view_is_not_revived_in_raw_view(self):
        raw = camera()
        pinhole, *_ = make_pinhole_view(raw, 80)
        pixels = np.asarray([[-2., 23.5], [31.5, 23.5], [np.nan, 10.]])
        rays, _ = unproject_points(pinhole, pixels[:1])
        raw_pixel, _ = project_points(raw, rays)
        self.assertTrue(0 < raw_pixel[0, 0] < 63)  # Still inside the original image.
        result = pixels_to_original(pixels, pinhole, raw)
        self.assertTrue(np.isnan(result[[0, 2]]).all())
        self.assertTrue(np.isfinite(result[1]).all())

    def test_box_uses_warped_edges_not_only_corners(self):
        raw = camera()
        pinhole, *_ = make_pinhole_view(raw, 100)
        box = box_to_original([-20, -10, 90, 70], pinhole, raw)
        corners = pixels_to_original(np.asarray([[0, 0], [63, 0], [63, 47], [0, 47]]), pinhole, raw)
        self.assertLess(box[0], float(corners[:, 0].min()) - 1)
        self.assertGreater(box[2], float(corners[:, 0].max()) + 1)
        ys = np.linspace(0, 47, 1001)
        dense = pixels_to_original(np.column_stack((np.full_like(ys, 63), ys)), pinhole, raw)
        self.assertAlmostEqual(box[2], float(np.nanmax(dense[:, 0])), places=3)
        self.assertEqual(box_to_original([70, 0, 80, 10], pinhole, raw), [])
        self.assertEqual(box_to_original([0, 0, np.nan, 5], pinhole, raw), [])

    def test_pair_validation_and_nonmutating_view_conversion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw, undistorted = root / "raw", root / "undistorted"
            make_dataset(raw)
            undistort_dataset(raw, undistorted, undistortion_options(100, 80, 40))
            adapter = OriginalVideoProjection(undistorted, raw, ("camera0", "camera1"))
            source_view = {"joints_2d": np.tile([40., 20.], (21, 1)).tolist(),
                           "wilor_joints_2d": np.tile([42., 20.], (21, 1)).tolist(),
                           "bbox_xyxy": [25, 10, 50, 30], "confidence": .9}
            before = copy.deepcopy(source_view)
            mapped = adapter.native_view("camera0", source_view)
            self.assertEqual(source_view, before)
            self.assertNotEqual(mapped["joints_2d"], source_view["joints_2d"])
            self.assertNotEqual(mapped["wilor_joints_2d"], source_view["wilor_joints_2d"])
            self.assertNotEqual(mapped["bbox_xyxy"], source_view["bbox_xyxy"])
            self.assertEqual(mapped["confidence"], .9)
            hidden = {"joints_2d": np.tile([-20., 20.], (21, 1)).tolist()}
            self.assertIsNone(adapter.native_view("camera0", hidden))
            table = raw / "multiview_frames.csv"
            original_table = table.read_text()
            table.write_text(original_table.replace("1000000000000000000", "1000000000000000001"))
            with self.assertRaisesRegex(ValueError, "timestamps differ"):
                OriginalVideoProjection(undistorted, raw, ("camera0",))
            table.write_text(original_table)
            path = raw / "calibration/camera0.json"
            data = json.loads(path.read_text())
            data["T_base_camera"][0][3] += .1
            write_json(path, data)
            with self.assertRaisesRegex(ValueError, "calibration/optical frame differs"):
                OriginalVideoProjection(undistorted, raw, ("camera0",))

    def test_renderer_reads_raw_frames_and_maps_3d_2d_and_boxes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw, undistorted = root / "raw", root / "undistorted"
            rows = make_dataset(raw)
            undistort_dataset(raw, undistorted, undistortion_options(100, 80, 40))
            pinhole = CameraCalibration.load(undistorted / "calibration/camera0.json")
            original = CameraCalibration.load(raw / "calibration/camera0.json")
            camera_points = np.column_stack((np.linspace(-.025, .025, 21), np.linspace(-.01, .01, 21), np.full(21, .5)))
            world_points = camera_points @ original.T_base_camera[:3, :3].T + original.T_base_camera[:3, 3]
            observed, _ = project_points(pinhole, camera_points)
            expected, _ = project_points(original, camera_points)
            view = {"detection_index": 0, "is_right": 0, "detector_is_right": 0, "confidence": .99,
                    "joints_2d": observed.tolist(), "bbox_xyxy": [25, 10, 50, 30]}
            prediction_path = root / "predictions/camera0/predictions.jsonl"
            prediction_path.parent.mkdir(parents=True)
            prediction_path.write_text("\n".join(json.dumps({"sync_index": row["sync_index"], "hands": [view]}) for row in rows))
            fusion = root / "fusion"
            fusion.mkdir()
            hand = {"side": 0, "joints_base_m": world_points.tolist(), "views": {"camera0": view},
                    "inlier_view_counts": [4] * 21, "quality": {"ray_condition_numbers": [1] * 21},
                    "stereo_baseline_comparison": {"multiview_3d_difference_median_mm": 0}}
            (fusion / "accepted.jsonl").write_text("\n".join(json.dumps({"sync_index": row["sync_index"], "hands": [hand]}) for row in rows[:2]))
            frames, fused, native, boxes = [], [], [], []
            class Writer:
                def write(self, frame):
                    frames.append(frame.copy())
                def release(self):
                    pass
            def create_writer(path, *args):
                path.write_bytes(b"fake video")
                return Writer(), "test"
            args = ["render_hybrid_multiview.py", "--dataset", str(undistorted), "--original-dataset", str(raw),
                    "--fusion", str(fusion), "--predictions", str(root / "predictions"),
                    "--output", str(root / "original.mp4"), "--cameras", "camera0", "--tile-width", "64",
                    "--smoothing-radius", "0", "--max-hold-frames", "0", "--residual-threshold-px", "2",
                    "--wrist-threshold-px", "2", "--max-joint-residual-px", "2"]
            with (patch.object(sys, "argv", args),
                  patch.object(renderer, "_create_video_writer", side_effect=create_writer),
                  patch.object(renderer, "_draw_refined_hand", side_effect=lambda image, points, *a, **kw: fused.append(points.copy())),
                  patch.object(renderer, "_draw_final_detection", side_effect=lambda image, item, *a: boxes.append(item)),
                  patch.object(renderer, "_draw_final_2d_fallback", side_effect=lambda image, item, *a: native.append(item))):
                self.assertEqual(renderer.main(), 0)
            self.assertTrue((root / "original.mp4").is_file())
            self.assertFalse((root / "original.partial.mp4").exists())
            self.assertEqual(len(frames), 3)
            self.assertEqual(frames[0].shape, (48, 64, 3))  # Raw aspect ratio, not 80x40.
            reader = SequentialVideoReader(raw / "cameras/camera0/video.mkv", (64, 48))
            try:
                for output, index in zip(frames, (0, 2, 4)):
                    original_frame = reader.read(index)
                    np.testing.assert_array_equal(output[-1, 0], original_frame[-1, 0])
            finally:
                reader.close()
            self.assertEqual(len(fused), 2)  # Trust evaluated in pinhole pixels.
            self.assertEqual(len(native), 1)
            for points in fused:
                np.testing.assert_allclose(points, expected, atol=1e-9)
            np.testing.assert_allclose(native[0]["joints_2d"], expected, atol=1e-9)
            self.assertNotEqual(boxes[0]["bbox_xyxy"], view["bbox_xyxy"])
            summary = json.loads((root / "original.json").read_text())
            self.assertEqual(summary["fused_camera_tiles"], 2)
            self.assertEqual(summary["native_2d_camera_hand_tiles"], 1)
            self.assertEqual(summary["original_display"]["dataset"], str(raw))


if __name__ == "__main__":
    unittest.main()
