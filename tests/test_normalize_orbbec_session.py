from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

import yaml


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from ego_data.dataset import NormalizedStereoDataset
from normalize_orbbec_session import normalize


class NormalizeOrbbecSessionTests(unittest.TestCase):
    def test_session_becomes_self_contained_normalized_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            session = root / "session"
            session.mkdir()
            left_video = session / "recording_camera_left.mp4"
            right_video = session / "recording_camera_right.mp4"
            left_video.write_bytes(b"left-video")
            right_video.write_bytes(b"right-video")
            (session / "recording_camera_left_pts.csv").write_text(
                "frame_index,timestamp_us\n0,1000000\n1,1033333\n", encoding="utf-8"
            )
            (session / "recording_camera_right_pts.csv").write_text(
                "frame_index,timestamp_us\n0,1000200\n1,1033500\n", encoding="utf-8"
            )
            cameras = []
            for camera_id, tx in (("left", 0.0), ("right", -60.0)):
                cameras.append({
                    "id": camera_id, "name": camera_id, "distortion_model": "KB",
                    "image_width": 320, "image_height": 240,
                    "intrinsics": {"fx": 200.0, "fy": 201.0, "cx": 160.0, "cy": 120.0},
                    "distortion": {"k1": 0.0, "k2": 0.0, "k3": 0.0, "k4": 0.0},
                    "extrinsics": {"rotation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                                   "translation": [tx, 0.0, 0.0]},
                })
            (session / "recording_calibration_camera.yaml").write_text(
                yaml.safe_dump({"cameras": cameras}), encoding="utf-8"
            )

            output = normalize(type("Args", (), {
                "session": session, "output": root / "normalized", "max_delta_us": 1500,
            })())
            dataset = NormalizedStereoDataset(output)
            self.assertEqual(dataset.manifest["source"]["kind"], "orbbec_session")
            self.assertEqual(dataset.manifest["pairing"]["pair_count"], 2)
            self.assertEqual(dataset.stereo.left.model, "KB")
            self.assertEqual((output / "cameras" / "left" / "video.mp4").read_bytes(), b"left-video")
            self.assertEqual(int(dataset.pairs[0]["left_timestamp_ns"]), 1_000_000_000)
            self.assertTrue((output / "calibration" / "stereo.json").is_file())
            json.loads((output / "manifest.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
