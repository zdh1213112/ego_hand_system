from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

import cv2
import numpy as np
import yaml


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from normalize_orbbec_ego4_session import CAMERA_SOURCES, normalize
from ego_data.calibration import CameraCalibration


class NormalizeOrbbecEgo4Tests(unittest.TestCase):
    def test_four_video_export_maps_calibration_and_syncs_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            session = root / "session"
            session.mkdir()
            cameras = []
            for index, (camera_id, (stream, source_id)) in enumerate(CAMERA_SOURCES.items()):
                video = session / f"recording_camera_{stream}.mp4"
                writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (64, 48))
                self.assertTrue(writer.isOpened())
                for frame in range(2):
                    writer.write(np.full((48, 64, 3), index * 30 + frame, dtype=np.uint8))
                writer.release()
                with (session / f"recording_camera_{stream}_pts.csv").open("w", newline="") as output:
                    rows = csv.writer(output)
                    rows.writerow(["timestamp_us"])
                    rows.writerow([1_000_000 + index * 3])
                    rows.writerow([1_033_333 + index * 3])
                cameras.append({
                    "id": source_id, "name": stream, "distortion_model": "KB",
                    "image_width": 64, "image_height": 48,
                    "intrinsics": {"fx": 40, "fy": 40, "cx": 32, "cy": 24},
                    "distortion": {"k1": 0, "k2": 0, "k3": 0, "k4": 0},
                    "extrinsics": {"rotation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                                   "translation": [-100 * index, 0, 0]},
                })
            (session / "recording_calibration_camera.yaml").write_text(
                yaml.safe_dump({"cameras": cameras}), encoding="utf-8"
            )
            output = normalize(type("Args", (), {
                "session": session, "output": root / "normalized",
                "reference_camera": "camera0", "max_delta_us": 1500, "max_frames": 0,
            })())
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["synchronization"]["frame_count"], 2)
            self.assertEqual(manifest["camera_sources"]["camera2"]["stream"], "right_side")
            camera2 = CameraCalibration.load(output / "calibration" / "camera2.json")
            self.assertEqual(camera2.camera_id, "camera2")
            self.assertAlmostEqual(camera2.T_base_camera[0, 3], 0.2)
            with (output / "multiview_frames.csv").open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(int(rows[1]["camera3_frame_index"]), 1)
            self.assertEqual(float(rows[0]["camera3_delta_us"]), 9.0)


if __name__ == "__main__":
    unittest.main()
