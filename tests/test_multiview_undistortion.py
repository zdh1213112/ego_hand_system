from __future__ import annotations

import csv
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from camera_models import project_points, unproject_points
from compare_multiview_wilor_undistortion import compare_runs, original_pixel_residuals
from ego_data.calibration import CameraCalibration
from ego_data.dataset import SequentialVideoReader
from fuse_multiview_wilor import _intersect_rays, _ray_in_base
from undistort_multiview_dataset import make_pinhole_view, undistort_dataset, undistortion_options


def camera(name="camera0", model="KB", center=0.0):
    intrinsic = np.asarray([[35., 0., 31.5], [0., 36., 23.5], [0., 0., 1.]])
    transform = np.eye(4)
    transform[0, 3] = center
    rotation, _ = cv2.Rodrigues(np.asarray([0.01, center, -0.02]))
    transform[:3, :3] = rotation
    distortion = np.asarray([0.01, -0.002, 0.0001, 0.0]) if model == "KB" else np.asarray([35., 36., 31.5, 23.5, -0.003, 0.572])
    return CameraCalibration(name, name, model, (64, 48), intrinsic, distortion, transform)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_dataset(root):
    root.mkdir(parents=True)
    (root / "calibration").mkdir()
    cameras = ("camera0", "camera1")
    for number, name in enumerate(cameras):
        directory = root / "cameras" / name
        directory.mkdir(parents=True)
        camera(name, "KB" if number == 0 else "DS", number * .06).save(root / "calibration" / f"{name}.json")
        writer = cv2.VideoWriter(str(directory / "video.mkv"), cv2.VideoWriter_fourcc(*"FFV1"), 30, (64, 48))
        if not writer.isOpened():
            raise RuntimeError("test requires FFV1 support")
        for frame in range(6):
            x, y = np.meshgrid(np.arange(64), np.arange(48))
            image = np.stack((x + frame * 20, y + number * 30, x + y), axis=-1).astype(np.uint8)
            writer.write(image)
        writer.release()
        write_csv(directory / "timestamps.csv", [{"frame_index": i, "timestamp_ns": 10**18 + i * 33333333 + number * 100,
                                                  "timestamp_us": (10**18 + i * 33333333 + number * 100) // 1000} for i in range(6)])
    rows = []
    for index, frame_index in enumerate((0, 2, 4)):
        row = {"sync_index": index, "reference_timestamp_ns": 10**18 + frame_index * 33333333}
        for number, name in enumerate(cameras):
            row.update({f"{name}_frame_index": frame_index,
                        f"{name}_timestamp_ns": 10**18 + frame_index * 33333333 + number * 100,
                        f"{name}_delta_us": number * .1})
        rows.append(row)
    write_csv(root / "multiview_frames.csv", rows)
    write_json(root / "manifest.json", {"dataset_type": "normalized_multiview", "camera_ids": list(cameras),
               "reference_camera": "camera0", "source": {"sha256": "same-recording"}, "image_size": [64, 48],
               "storage": {"video_filename": "video.mkv"}, "synchronization": {"frame_count": len(rows)}})
    return rows


class UndistortionGeometryTests(unittest.TestCase):
    def test_kb_remap_matches_opencv(self):
        raw = camera()
        pinhole, x, y, valid = make_pinhole_view(raw, 90)
        expected_x, expected_y = cv2.fisheye.initUndistortRectifyMap(
            raw.K, raw.distortion, np.eye(3), pinhole.K, raw.image_size, cv2.CV_32FC1)
        np.testing.assert_allclose(x[valid], expected_x[valid], atol=1e-5)
        np.testing.assert_allclose(y[valid], expected_y[valid], atol=1e-5)
        self.assertEqual(pinhole.distortion.size, 0)
        np.testing.assert_array_equal(raw.T_base_camera, pinhole.T_base_camera)

    def test_ds_map_roundtrip_and_fusion_preserve_rig_coordinates(self):
        point = np.asarray([.04, .02, .5])
        native_rays, pinhole_rays = [], []
        for name, center in (("camera0", 0), ("camera1", .06), ("camera2", .14)):
            raw = camera(name, "DS", center)
            pinhole, x, y, valid = make_pinhole_view(raw, 120)
            grid_x, grid_y = np.meshgrid(np.arange(64), np.arange(48))
            native, _ = unproject_points(raw, np.stack((x[valid], y[valid]), axis=-1))
            converted, _ = unproject_points(pinhole, np.stack((grid_x[valid], grid_y[valid]), axis=-1))
            np.testing.assert_allclose(native, converted, atol=3e-7)
            point_camera = (point - raw.T_base_camera[:3, 3]) @ raw.T_base_camera[:3, :3]
            for calib, collection in ((raw, native_rays), (pinhole, pinhole_rays)):
                pixels, is_valid = project_points(calib, point_camera[None])
                self.assertTrue(is_valid[0])
                collection.append(_ray_in_base(calib, pixels[0]))
        np.testing.assert_allclose(_intersect_rays(native_rays), point, atol=1e-9)
        np.testing.assert_allclose(_intersect_rays(pinhole_rays), point, atol=1e-9)

    def test_pinhole_roundtrip_and_invalid_points(self):
        pinhole, *_ = make_pinhole_view(camera(), 100)
        points = np.asarray([[.1, .2, 1], [0, 0, -1], [np.nan, 1, 1], [0, 0, 0]])
        pixels, valid = project_points(pinhole, points)
        np.testing.assert_array_equal(valid, [True, False, False, False])
        rays, ray_valid = unproject_points(pinhole, pixels)
        np.testing.assert_array_equal(valid, ray_valid)
        np.testing.assert_allclose(rays[0], points[0] / np.linalg.norm(points[0]))
        with self.assertRaises(ValueError):
            replace(pinhole, distortion=np.zeros(4))

    def test_invalid_options(self):
        for fov, width, height in ((0, 0, 0), (175, 0, 0), (float("nan"), 0, 0), (120, 64, 0), (120, 63, 48)):
            with self.subTest(fov=fov, size=(width, height)), self.assertRaises(ValueError):
                undistortion_options(fov, width, height)

    def test_residuals_use_original_pixels(self):
        raw = camera()
        pinhole, *_ = make_pinhole_view(raw, 130)
        observed_points = np.tile([0.02, .01, .5], (21, 1))
        predicted = observed_points + [.005, 0, 0]
        points_base = predicted @ raw.T_base_camera[:3, :3].T + raw.T_base_camera[:3, 3]
        results = []
        for calib in (raw, pinhole):
            observed, _ = project_points(calib, observed_points)
            hand = {"joints_base_m": points_base.tolist(), "views": {"camera0": {"joints_2d": observed.tolist()}}}
            results.append(original_pixel_residuals(hand, {"camera0": calib}, {"camera0": raw}))
        self.assertEqual(len(results[0]), 21)
        np.testing.assert_allclose(list(results[0].values()), list(results[1].values()), atol=1e-10)


class UndistortionDatasetTests(unittest.TestCase):
    def test_video_timing_calibration_and_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "source", root / "output"
            make_dataset(source)
            options = undistortion_options(100, 80, 60)
            undistort_dataset(source, output, options)
            self.assertEqual((source / "multiview_frames.csv").read_bytes(), (output / "multiview_frames.csv").read_bytes())
            for name in ("camera0", "camera1"):
                raw = CameraCalibration.load(source / "calibration" / f"{name}.json")
                converted = CameraCalibration.load(output / "calibration" / f"{name}.json")
                self.assertEqual(converted.image_size, (80, 60))
                _, mx, my, _ = make_pinhole_view(raw, 100, (80, 60))
                src = SequentialVideoReader(source / "cameras" / name / "video.mkv", (64, 48))
                dst = SequentialVideoReader(output / "cameras" / name / "video.mkv", (80, 60))
                try:
                    for index in (0, 2, 4):
                        expected = cv2.remap(src.read(index), mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                        np.testing.assert_array_equal(dst.read(index), expected)
                finally:
                    src.close()
                    dst.close()
            self.assertEqual(undistort_dataset(source, output, options, reuse=True), output)
            with self.assertRaisesRegex(ValueError, "configuration/source changed"):
                undistort_dataset(source, output, undistortion_options(110), reuse=True)
            (source / "calibration" / "camera0.json").write_text((source / "calibration" / "camera0.json").read_text() + "\n")
            with self.assertRaisesRegex(ValueError, "configuration/source changed"):
                undistort_dataset(source, output, options, reuse=True)

    def test_truncated_video_does_not_publish_partial_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "source", root / "output"
            make_dataset(source)
            (source / "cameras" / "camera1" / "video.mkv").write_bytes(b"broken")
            with self.assertRaises(RuntimeError):
                undistort_dataset(source, output, undistortion_options(100))
            self.assertFalse(output.exists())
            self.assertEqual(list(root.glob(".output.tmp-*")), [])

    def test_comparison_counts_boxes_once_and_rejects_mismatched_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw_root, new_root = root / "raw", root / "new"
            rows = make_dataset(raw_root / "normalized_multiview")
            options = undistortion_options(120)
            undistort_dataset(raw_root / "normalized_multiview", new_root / "undistorted_multiview", options)
            config = {"detector": "same-model", "max_frames": 3}
            for run, extra in ((raw_root, {}), (new_root, {"image_preprocessing": options})):
                write_json(run / "run_config.json", {**config, **extra})
                write_json(run / "wilor_multiview" / "summary.json", {})
                write_json(run / "fusion_multiview" / "summary.json", {})
                (run / "fusion_multiview" / "accepted.jsonl").write_text("")
                for name in ("camera0", "camera1"):
                    destination = run / "wilor_multiview" / name / "predictions.jsonl"
                    destination.parent.mkdir()
                    destination.write_text("\n".join(json.dumps({"sync_index": row["sync_index"],
                        "source_frame_index": row[f"{name}_frame_index"], "timestamp_ns": row[f"{name}_timestamp_ns"],
                        "hands": [{"detection_index": 0, "is_right": side} for side in (0, 1)]}) for row in rows))
            report, samples = compare_runs(raw_root, new_root)
            self.assertEqual(report["raw"]["detections"]["camera0"]["physical_box_count"], 3)
            self.assertEqual(report["raw"]["finite_3d_frame_rate"], 0)
            self.assertIsNone(report["paired_original_pixel_residual"]["raw"]["mean"])
            self.assertEqual(samples, [])
            json.dumps(report, allow_nan=False)
            # Nonempty accepted results exercise paired reporting, not just its
            # empty-data fallback. The new run has a known 3-D offset.
            for run, dataset_name, offset in ((raw_root, "normalized_multiview", 0),
                                               (new_root, "undistorted_multiview", .003)):
                points = np.tile([.02, .01, .5], (21, 1))
                views = {}
                for name in ("camera0", "camera1"):
                    calib = CameraCalibration.load(run / dataset_name / "calibration" / f"{name}.json")
                    observed, _ = project_points(calib, (points - calib.T_base_camera[:3, 3]) @ calib.T_base_camera[:3, :3])
                    views[name] = {"joints_2d": observed.tolist()}
                records = [{"sync_index": row["sync_index"], "hands": [{"side": 0,
                            "joints_base_m": (points + [offset, 0, 0]).tolist(), "views": views}]} for row in rows]
                (run / "fusion_multiview" / "accepted.jsonl").write_text("\n".join(json.dumps(row) for row in records))
            report, samples = compare_runs(raw_root, new_root)
            self.assertEqual(len(samples), 126)
            self.assertEqual(report["raw"]["complete_21_joint_hands"], 3)
            self.assertAlmostEqual(report["paired_original_pixel_residual"]["raw"]["mean"], 0, places=9)
            self.assertGreater(report["paired_original_pixel_residual"]["undistorted"]["mean"], .01)
            write_json(new_root / "run_config.json", {**config, "image_preprocessing": options, "max_frames": 5})
            with self.assertRaisesRegex(ValueError, "configurations differ"):
                compare_runs(raw_root, new_root)


class UndistortionRunnerTests(unittest.TestCase):
    def test_shared_pipeline_routes_all_stages_and_guards_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = root / "fixture"
            make_dataset(fixture)
            fake = root / "bin"
            fake.mkdir()
            conda = fake / "conda"
            conda.write_text(f"#!{sys.executable}\n" + '''
import json, os, pathlib, shutil, subprocess, sys
args = sys.argv[sys.argv.index("python") + 1:]
name = pathlib.Path(args[0]).name
with open(os.environ["CALL_LOG"], "a") as stream:
    stream.write(json.dumps(args) + "\\n")
if name in ("-", "undistort_multiview_dataset.py"):
    raise SystemExit(subprocess.run([sys.executable, *args]).returncode)
output = pathlib.Path(args[args.index("--output") + 1])
if name == "normalize_multiview_recording.py":
    shutil.copytree(os.environ["FIXTURE"], output)
elif name.startswith("render_"):
    output.write_bytes(b"fake video")
    output.with_suffix(".json").write_text("{}")
else:
    output.mkdir(parents=True, exist_ok=True)
    summary = {"save_mano_params": "--save-mano-params" in args} if name == "wilor_multiview_inference.py" else {}
    (output / "summary.json").write_text(json.dumps(summary))
''')
            conda.chmod(0o755)
            source, detector = root / "input.mcap", root / "detector.pt"
            source.touch()
            detector.touch()
            env = {key: value for key, value in os.environ.items() if not key.startswith("EGO_WILOR_UNDISTORT_")}
            env.update(PATH=str(fake) + os.pathsep + env["PATH"], PYTHONPATH=str(ROOT / "scripts"),
                       FIXTURE=str(fixture), CALL_LOG=str(root / "calls.jsonl"))
            common = ["--mcap", str(source), "--detector", str(detector), "--cameras", "camera0", "camera1",
                      "--reference-camera", "camera0", "--max-frames", "0", "--no-video",
                      "--wilor-mode", "both"]
            for mode in ("raw", "undistorted"):
                runner = ROOT / "scripts" / ("run_multiview_wilor_experiment.sh" if mode == "raw" else "run_multiview_wilor_undistorted_experiment.sh")
                command = ["bash", str(runner), *common, "--output", str(root / mode)]
                result = subprocess.run(command, env=env, text=True, capture_output=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            calls = [json.loads(line) for line in (root / "calls.jsonl").read_text().splitlines()]
            for mode, dataset_name in (("raw", "normalized_multiview"), ("undistorted", "undistorted_multiview")):
                for name in ("wilor_multiview_inference.py", "fuse_multiview_wilor_guided.py"):
                    matching = [args for args in calls if Path(args[0]).name == name and str(root / mode) in args[args.index("--output") + 1]]
                    self.assertEqual(len(matching), 1)
                    self.assertEqual(matching[0][matching[0].index("--dataset") + 1], str(root / mode / dataset_name))
            # Adding a display-only output to a completed run must not rerun
            # inference/fusion or conflict with its saved run configuration.
            config_before = (root / "undistorted/run_config.json").read_bytes()
            render_command = [arg for arg in command if arg != "--no-video"]
            result = subprocess.run([*render_command, "--final-video", "--original-video"], env=env, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            calls = [json.loads(line) for line in (root / "calls.jsonl").read_text().splitlines()]
            originals = [args for args in calls if "--original-dataset" in args]
            self.assertEqual(len(originals), 1)
            original = originals[0]
            self.assertEqual(original[original.index("--dataset") + 1], str(root / "undistorted/undistorted_multiview"))
            self.assertEqual(original[original.index("--original-dataset") + 1], str(root / "undistorted/normalized_multiview"))
            self.assertTrue(original[original.index("--output") + 1].endswith("final_on_original_2view.mp4"))
            self.assertEqual(sum(Path(args[0]).name == "wilor_multiview_inference.py" for args in calls), 2)
            self.assertEqual((root / "undistorted/run_config.json").read_bytes(), config_before)
            # A second render request reuses its completed video.
            result = subprocess.run([*render_command, "--original-video"], env=env, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            calls_after = [json.loads(line) for line in (root / "calls.jsonl").read_text().splitlines()]
            self.assertEqual(sum("--original-dataset" in args for args in calls_after), 1)
            repeat = subprocess.run([*command, "--undistort-fov-deg", "100"], env=env, text=True, capture_output=True)
            self.assertNotEqual(repeat.returncode, 0)
            self.assertIn("configuration differs", repeat.stderr)


if __name__ == "__main__":
    unittest.main()
