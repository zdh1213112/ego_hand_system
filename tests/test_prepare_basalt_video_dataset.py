from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np


SPEC = importlib.util.spec_from_file_location(
    "basalt_video_dataset", Path(__file__).resolve().parents[1] / "scripts/prepare_basalt_video_dataset.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class VideoDatasetTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def fixture(self, count=4):
        videos, timestamps = [], []
        for camera in range(count):
            video = self.root / f"cam{camera}.avi"
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 30, (48, 32))
            if not writer.isOpened():
                self.skipTest("MJPG encoder unavailable")
            for frame in range(4):
                writer.write(np.full((32, 48, 3), 20 + camera * 25 + frame * 8, np.uint8))
            writer.release()
            times = self.root / f"cam{camera}.csv"
            times.write_text("timestamp_ns\n1000000000\n1033333333\n1066666666\n1100000000\n")
            videos.append(video)
            timestamps.append(times)
        identity = dict(px=0., py=0., pz=0., qx=0., qy=0., qz=0., qw=1.)
        calibration = self.root / "calibration.json"
        calibration.write_text(json.dumps({"value0": {
            "resolution": [[48, 32] for _ in range(count)],
            "intrinsics": [{"camera_type": "ds", "intrinsics": {
                "fx": 30, "fy": 30, "cx": 24, "cy": 16, "xi": 0, "alpha": 0.5
            }} for _ in range(count)],
            "T_imu_cam": [{**identity, "px": camera * 0.1} for camera in range(count)],
            "vignette": [{} for _ in range(count)], "cam_time_offset_ns": 0,
        }}))
        config = self.root / "config.json"
        config.write_text(json.dumps({"value0": {"config.optical_flow_type": "frame_to_frame"}}))
        return argparse.Namespace(videos=videos, timestamps=timestamps, calibration=calibration,
                                  config=config, imu=None, output=self.root / "dataset",
                                  max_delta_us=0, max_frames=0)

    def test_four_and_six_camera_real_video_roundtrip(self):
        for count in (4, 6):
            with self.subTest(count=count):
                args = self.fixture(count)
                args.output = self.root / f"dataset{count}"
                output = MODULE.prepare(args)
                summary = json.loads((output / "summary.json").read_text())
                self.assertEqual(summary["image_groups"], 4)
                self.assertEqual(summary["camera_count"], count)
                for camera in range(count):
                    directory = output / "mav0" / f"cam{camera}"
                    image = cv2.imread(str(directory / "data/1066666666.png"), cv2.IMREAD_UNCHANGED)
                    self.assertEqual(image.shape, (32, 48))
                    self.assertAlmostEqual(float(image.mean()), 20 + camera * 25 + 16, delta=3)
                    self.assertEqual(len((directory / "data.csv").read_text().splitlines()), 5)
                with self.assertRaises(FileExistsError):
                    MODULE.prepare(args)

    def test_no_false_sync_and_preserved_source_offsets(self):
        args = self.fixture(2)
        args.timestamps[1].write_text("timestamp_ns\n1000000500\n1033333833\n1066667166\n1100000500\n")
        with self.assertRaisesRegex(ValueError, "synchronized"):
            MODULE.prepare(args)
        self.assertFalse(args.output.exists())
        args.max_delta_us = 1
        output = MODULE.prepare(args)
        with (output / "frame_map.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(rows[0]["cam1_source_timestamp_ns"], "1000000500")
        self.assertEqual(rows[0]["cam1_delta_ns"], "500")

    def test_resize_changes_pixel_intrinsics_and_preserves_extrinsics(self):
        args = self.fixture(2)
        args.image_scale = 0.5
        original = json.loads(args.calibration.read_text())
        output = MODULE.prepare(args)
        calibration = json.loads((output / "calibration.json").read_text())["value0"]
        image = cv2.imread(str(output / "mav0/cam0/data/1000000000.png"), 0)
        self.assertEqual(image.shape, (16, 24))
        self.assertEqual(calibration["intrinsics"][0]["intrinsics"]["fx"], 15)
        self.assertEqual(calibration["intrinsics"][0]["intrinsics"]["alpha"], 0.5)
        self.assertEqual(calibration["T_imu_cam"], original["value0"]["T_imu_cam"])
        self.assertEqual(json.loads(args.calibration.read_text()), original)

    def test_video_ends_early_without_partial_dataset(self):
        args = self.fixture(2)
        for path in args.timestamps:
            with path.open("a") as stream:
                stream.write("1133333333\n")
        with self.assertRaisesRegex(RuntimeError, "ended before"):
            MODULE.prepare(args)
        self.assertFalse(args.output.exists())
        self.assertFalse(list(self.root.glob(".dataset-*")))

    def test_nonzero_calibration_time_offset_is_rejected(self):
        args = self.fixture(2)
        calibration = json.loads(args.calibration.read_text())
        calibration['value0']['cam_time_offset_ns'] = 1801936
        args.calibration.write_text(json.dumps(calibration))
        with self.assertRaisesRegex(ValueError, 'time correction is disabled'):
            MODULE.prepare(args)
        self.assertFalse(args.output.exists())

    def test_vio_requires_covering_imu_and_preserves_frontend(self):
        args = self.fixture(2)
        args.imu = self.root / "imu.csv"
        args.imu.write_text("timestamp_ns,wx,wy,wz,ax,ay,az\n1000000000,0,0,0,0,0,9.81\n1100000000,0,0,0,0,0,9.81\n")
        with self.assertRaisesRegex(ValueError, "extend beyond"):
            MODULE.prepare(args)
        with args.imu.open("a") as stream:
            stream.write("1133333333,0,0,0,0,0,9.81\n")
        output = MODULE.prepare(args)
        self.assertEqual(json.loads((output / "summary.json").read_text())["mode"], "vio")
        self.assertEqual(json.loads((output / "config.json").read_text())["value0"]["config.optical_flow_type"], "frame_to_frame")

    def test_missing_video_frame_is_skipped_without_reuse(self):
        result = MODULE.synchronize([[10, 20, 30, 40], [10, 30, 40]], 0)
        self.assertEqual(result, [(0, 0), (2, 1), (3, 2)])
        result = MODULE.synchronize([[10, 11, 12], [10, 12]], 2)
        self.assertEqual(result, [(0, 0), (1, 1)])

    def test_imu_window_keeps_original_video_indices(self):
        args = self.fixture(2)
        args.first_timestamp_ns = 1000000001
        args.last_timestamp_ns = 1100000000
        output = MODULE.prepare(args)
        with (output / "frame_map.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual([int(r["cam0_frame_index"]) for r in rows], [1, 2])
        image = cv2.imread(str(output / "mav0/cam0/data/1033333333.png"), 0)
        self.assertAlmostEqual(float(image.mean()), 28, delta=3)


if __name__ == "__main__":
    unittest.main()
