from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from ego_data.time_sync import camera_offset_policy, nearest_indices, require_zero_camera_offset
from associate_vio_imu import associate


class EgoTimeSyncTest(unittest.TestCase):
    def test_epoch_precision_correction_before_matching_and_gap_rejection(self):
        epoch = 1793793532731618000
        raw = [epoch, epoch + 1_000_000, epoch + 10_000_000]
        target = [epoch + 161_781_812, epoch + 162_281_812, epoch + 166_781_812]
        self.assertEqual(nearest_indices(target, raw, 600_000), [None]*3)
        self.assertEqual(nearest_indices(target, raw, 600_000, sample_offset_ns=161_781_812),
                         [0, 0, None])
        self.assertEqual(raw[0], epoch)

    def test_offset_policy_always_preserves_recorded_timestamps(self):
        yaml = {"cam0": {"timeshift_cam_imu": .001801936421667}}
        self.assertEqual(camera_offset_policy(argparse.Namespace(), yaml)["camera_to_imu_offset_ns"], 0)
        self.assertEqual(camera_offset_policy(None, yaml)["application_count"], 0)
        for args in [argparse.Namespace(camera_imu_offset_ns=123),
                     argparse.Namespace(use_factory_camera_imu_offset=True)]:
            with self.assertRaisesRegex(ValueError, 'disabled'):
                camera_offset_policy(args, yaml)

    def test_historical_shifted_results_require_a_real_rerun(self):
        for value in [{'camera_time_shift_ns':1801936},
                      {'calibration':{'camera_time_shift_s':.001801936}},
                      {'time_sync_policy':{'camera_to_imu_offset_ns':-1}},
                      {'value0':{'cam_time_offset_ns':1}}]:
            with self.assertRaisesRegex(ValueError, 'Reprepare and rerun VIO'):
                require_zero_camera_offset(value)
        require_zero_camera_offset({'time_sync_policy':camera_offset_policy(),
                                    'nokov_to_imu_offset_s':.161781812})

    def test_association_preserves_native_pose_times_and_rate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trajectory, imu = root / "trajectory.csv", root / "imu.csv"
            epoch = 1793793532731618000
            native = [epoch+400_000, epoch+33_400_000, epoch+66_400_000]
            trajectory.write_text("#timestamp [ns],x\n"+"".join(f"{t},0\n" for t in native))
            original_bytes = trajectory.read_bytes()
            imu.write_text("timestamp_ns,wx\n"+"".join(f"{epoch+i*1_000_000},0\n" for i in range(68)))
            summary = associate(trajectory, imu, root / "association", 500_000)
            with (root / "association/vio_imu_association.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([int(r["vio_timestamp_ns"]) for r in rows], native)
            self.assertEqual([int(r["imu_minus_vio_ns"]) for r in rows], [-400_000]*3)
            self.assertEqual(summary["exact_imu_timestamp_count"], 0)
            self.assertEqual(summary["vio"]["count"], 3)
            self.assertEqual(summary["imu_sampling"]["average_hz"], 1000)
            self.assertEqual(trajectory.read_bytes(), original_bytes)
            self.assertFalse(json.loads((root / "association/vio_imu_timing.json").read_text())["trajectory_retimed"])

    def test_invalid_time_order_and_negative_tolerance_are_rejected(self):
        for reference, samples, tolerance in [([1, 1], [1, 2], 0), ([1], [2, 1], 0), ([1], [1], -1)]:
            with self.assertRaises(ValueError):
                nearest_indices(reference, samples, tolerance)


if __name__ == "__main__":
    unittest.main()
