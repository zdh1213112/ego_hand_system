from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from ego_data.calibration import CameraCalibration
from fit_mano_sequence import load_input
from prepare_nokov21_mano_input import prepare


class PrepareNokov21ManoInputTests(unittest.TestCase):
    def test_aligned_truth_becomes_fitter_input_and_export_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "normalized"
            calibration_dir = dataset / "calibration"
            calibration_dir.mkdir(parents=True)
            for camera, offset in (("camera2", 0.0), ("camera3", 0.05)):
                transform = np.eye(4)
                transform[0, 3] = offset
                CameraCalibration(
                    camera_id=camera, frame_id=camera, model="KB", image_size=(64, 48),
                    K=np.asarray([[70, 0, 32], [0, 70, 24], [0, 0, 1]]),
                    distortion=np.zeros(4), T_base_camera=transform,
                ).save(calibration_dir / f"{camera}.json")
            (dataset / "manifest.json").write_text(json.dumps({
                "dataset_type": "normalized_multiview",
                "camera_ids": ["camera2", "camera3"],
                "image_size": [64, 48],
                "calibration": {"available": True},
            }))
            stamps = np.asarray([1_793_070_505_343_647_000,
                                 1_793_070_505_376_980_000], dtype=np.int64)
            with (dataset / "multiview_frames.csv").open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(("sync_index", "reference_timestamp_ns"))
                writer.writerows((index, int(stamp)) for index, stamp in enumerate(stamps))
            points = np.full((2, 2, 21, 3), np.nan, dtype=np.float32)
            valid = np.zeros((2, 2, 21), dtype=bool)
            points[1, :, :, :] = np.asarray([0.02, 0.0, 0.5], dtype=np.float32)
            valid[1] = True
            aligned = root / "aligned.npz"
            np.savez_compressed(
                aligned, joints_base_m=points, valid=valid,
                reference_timestamp_ns=stamps,
            )
            summary = prepare(
                aligned, dataset, root / "mano_input.npz", root / "rectification.npz",
                fusion_output=root / "observations", source_group="session_001",
            )
            fitted_input = load_input(root / "mano_input.npz")
            self.assertEqual(fitted_input["positions_left_camera_m"].shape, (2, 2, 21, 3))
            self.assertEqual(summary["accepted_frames"], 2)
            row = json.loads((root / "observations" / "accepted.jsonl").read_text().splitlines()[0])
            self.assertEqual(row["hands"][0]["side"], 1)
            self.assertEqual(row["hands"][0]["views"]["camera2"]["inlier_joint_count"], 21)
            self.assertEqual(json.loads((root / "observations" / "summary.json").read_text())["source_group"], "session_001")


if __name__ == "__main__":
    unittest.main()
