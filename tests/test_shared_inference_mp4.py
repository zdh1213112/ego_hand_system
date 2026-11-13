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

from ego_hand_inference.normalize_orbbec_ego4_session import CAMERA_SOURCES, normalize
from ego_hand_inference.normalize_orbbec_multiview_mcap import DEFAULT_TOPICS, _copy_calibration
from ego_hand_inference.ego_data.calibration import CameraCalibration


class NormalizeOrbbecEgo4Tests(unittest.TestCase):
    def test_external_calibration_rejects_wrong_camera_or_image_size(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "calibration"
            source.mkdir()
            destination = root / "output"
            destination.mkdir()
            path = source / "camera0.json"
            path.write_text(json.dumps({"camera_id": "camera1", "image_size": [64, 48]}))
            with self.assertRaisesRegex(ValueError, "camera_id"):
                _copy_calibration(source, destination, ("camera0",), (64, 48))
            path.write_text(json.dumps({"camera_id": "camera0", "image_size": [1920, 1200]}))
            other_destination = root / "other_output"
            other_destination.mkdir()
            with self.assertRaisesRegex(ValueError, "image_size"):
                _copy_calibration(source, other_destination, ("camera0",), (64, 48))

    def test_mcap_topics_match_four_video_calibration_ids(self):
        self.assertEqual(DEFAULT_TOPICS, {
            "camera0": "/camera/left",
            "camera1": "/camera/right",
            "camera2": "/camera/right_side",
            "camera3": "/camera/left_side",
        })
        self.assertEqual(tuple(DEFAULT_TOPICS), tuple(CAMERA_SOURCES))

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

            # A requested part uses the recording-root calibration but only
            # streams inside that part, even when another part is present.
            part = session / "part0001"
            part.mkdir()
            sibling = session / "part0002"
            sibling.mkdir()
            for stream, _ in CAMERA_SOURCES.values():
                for suffix in (".mp4", "_pts.csv"):
                    original = session / f"recording_camera_{stream}{suffix}"
                    original.rename(part / f"recording_camera_{stream}_part0001{suffix}")
                    (sibling / f"recording_camera_{stream}_part0002{suffix}").touch()
            (session / "meta_recording.json").write_text('{"test": true}')
            part_output = normalize(type("Args", (), {
                "session": part, "output": root / "normalized_part",
                "reference_camera": "camera0", "max_delta_us": 1500, "max_frames": 0,
            })())
            part_manifest = json.loads((part_output / "manifest.json").read_text())
            self.assertEqual(part_manifest["source"]["session"], str(part.resolve()))
            self.assertEqual(part_manifest["synchronization"]["frame_count"], 2)
            self.assertTrue((part_output / "source_meta.json").is_file())
            source_videos = [Path(item["path"]) for item in part_manifest["source"]["files"]
                             if item["path"].endswith(".mp4")]
            self.assertEqual(len(source_videos), 4)
            self.assertTrue(all(path.parent == part for path in source_videos))


if __name__ == "__main__":
    unittest.main()
