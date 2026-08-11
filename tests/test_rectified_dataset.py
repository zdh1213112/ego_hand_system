from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

import cv2
import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from ego_data.dataset import RectifiedStereoDataset


class RectifiedDatasetTests(unittest.TestCase):
    def test_reader_exposes_existing_triangulation_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "manifest.json").write_text(json.dumps({
                "dataset_type": "rectified_stereo", "image_size": [8, 6],
                "calibration_hash": "test",
            }))
            with (root / "stereo_pairs.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=[
                    "pair_index", "left_frame_index", "right_frame_index",
                    "left_timestamp_ns", "right_timestamp_ns",
                ])
                writer.writeheader()
                writer.writerow({"pair_index": 0, "left_frame_index": 4, "right_frame_index": 5,
                                 "left_timestamp_ns": 1_000_000, "right_timestamp_ns": 1_020_000})
            image = np.zeros((6, 8, 3), dtype=np.uint8)
            for name in ("left.mkv", "right.mkv"):
                writer = cv2.VideoWriter(
                    str(root / name), cv2.VideoWriter_fourcc(*"FFV1"), 30.0, (8, 6)
                )
                self.assertTrue(writer.isOpened())
                writer.write(image)
                writer.release()
            np.savez(root / "rectification.npz", R1=np.eye(3), R2=np.eye(3),
                     P1=np.zeros((3, 4)), P2=np.zeros((3, 4)), Q=np.zeros((4, 4)))
            dataset = RectifiedStereoDataset(root)
            pair = next(iter(dataset))
            self.assertEqual(pair["left_index"], 4)
            self.assertEqual(pair["right_timestamp_us"], 1020)
            self.assertEqual(pair["left_image"].shape, (6, 8, 3))
            self.assertEqual(dataset.rectification["r1"].shape, (3, 3))


if __name__ == "__main__":
    unittest.main()
