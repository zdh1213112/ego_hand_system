import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from analyze_pico_vio_nokov import (
    Series,
    analyze_session,
    compare_transforms,
    fit_extrinsics,
    gate_reference,
    inv,
    load_series,
    metrics,
    support,
    synchronize,
    unpack,
)


def motion(t):
    result = np.tile(np.eye(4), (len(t), 1, 1))
    angles = np.c_[
        0.5 * np.sin(t * 0.7), 0.4 * np.cos(t * 0.43), 0.6 * np.sin(t * 0.53)
    ]
    result[:, :3, :3] = Rotation.from_euler("xyz", angles).as_matrix()
    result[:, :3, 3] = np.c_[
        0.3 * np.sin(t * 0.9) + 0.7,
        0.2 * np.cos(t * 0.6),
        0.15 * np.sin(t * 0.31) + 1.1,
    ]
    return result


def series(times, poses, host=None):
    return Series(
        [],
        times,
        np.arange(len(times), dtype=np.int64) if host is None else host,
        poses,
        np.ones(len(times), bool),
        0,
        [set() for _ in times],
    )


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class PicoVioNokovTest(unittest.TestCase):
    def test_blank_invalid_pose_is_retained_as_an_interpolation_hole(self):
        rows = []
        for i in range(40):
            rows.append(
                {
                    "head_timestamp_ns": 4000000000000 + i * 14000000,
                    "host_receive_perf_ns": 5000000000000 + i * 14000000,
                    "valid_numeric": 1,
                    "x_m": 0,
                    "y_m": 0,
                    "z_m": 0,
                    "qx": 0,
                    "qy": 0,
                    "qz": 0,
                    "qw": 1,
                }
            )
        rows[20]["qw"] = ""
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "head.csv"
            write_csv(path, rows)
            data = load_series(path, True)
        self.assertEqual(len(data.t), 40)
        self.assertFalse(data.good[20])
        self.assertFalse(support(data, np.array([data.t[20] - 0.001]))[0])

    def test_two_sided_calibration_recovers_world_and_mount_at_unit_scale(self):
        t = np.arange(0, 20, 0.1)
        x = unpack([0.15, -0.2, 0.3, 0.03, -0.04, 0.02])
        y = unpack([0.4, -0.1, -0.5, -0.3, 0.2, 0.7])
        n = motion(t)
        p = inv(y) @ n @ x
        xx, yy, details = fit_extrinsics(n, p, t)
        np.testing.assert_allclose(xx, x, atol=1e-8)
        np.testing.assert_allclose(yy, y, atol=1e-8)
        self.assertEqual(details["jacobian_rank"], 12)
        self.assertFalse(details["scale_fitted"])

    def test_unexcited_single_axis_rotations_do_not_yield_fake_mount(self):
        t = np.arange(0, 20, 0.1)
        matrices = np.tile(np.eye(4), (len(t), 1, 1))
        matrices[:, :3, :3] = Rotation.from_euler("z", 0.5 * np.sin(t)).as_matrix()
        with self.assertRaisesRegex(ValueError, "three-axis"):
            fit_extrinsics(matrices, matrices, t)

    def test_inverse_support_and_missing_interpolation_intervals(self):
        t = np.array([0.0, 0.01, 0.02, 0.2, 0.21])
        data = series(t, motion(t))
        np.testing.assert_allclose(
            inv(data.poses) @ data.poses, np.tile(np.eye(4), (5, 1, 1)), atol=1e-12
        )
        np.testing.assert_array_equal(
            support(data, np.array([-0.1, 0.005, 0.1, 0.205, 0.3])),
            [False, True, False, True, False],
        )
        data.good[1] = False
        self.assertFalse(support(data, np.array([0.005]))[0])
        self.assertFalse(support(data, np.array([0.01]), 0.01)[0])

    def test_host_clock_prior_plus_motion_offset_preserves_clock_domains(self):
        nt, pt = np.arange(0, 32, 0.01), np.arange(0, 27, 0.014)
        a, b = 1.00012, 2.4
        origin = 1230000000000000
        nh = origin + (nt * 1e9).astype(np.int64) + 10_000_000
        ph = origin + ((a * pt + b) * 1e9).astype(np.int64) + 7_000_000
        x, y = (
            unpack([0.1, 0.2, -0.1, 0.02, 0.03, 0.04]),
            unpack([-0.3, 0.2, 0.4, 0.3, 0.4, -0.1]),
        )
        n = series(nt, motion(nt), nh)
        p = series(pt, inv(y) @ motion(a * pt + b) @ x, ph)
        aa, bb, clock, _ = synchronize(p, n, 20)
        self.assertAlmostEqual(aa, a, places=8)
        self.assertAlmostEqual(bb, b, delta=0.0003)
        self.assertAlmostEqual(clock["motion_offset_s"], 0.003, delta=0.0003)
        self.assertGreater(
            clock["motion_validation"]["validation"]["correlation"], 0.999
        )

    def test_reference_marker_sentinel_and_guard_are_independent_of_pico(self):
        t = np.arange(0, 1, 0.01)
        data = series(t, motion(t))
        data.rows = [{"frame_no": str(i)} for i in range(len(t))]
        local = np.array([[0, 0, 0], [0.04, 0, 0], [0, 0.03, 0], [0.01, 0.01, 0.02]])
        rows = []
        for i, matrix in enumerate(data.poses):
            world = local @ matrix[:3, :3].T + matrix[:3, 3]
            for j, point in enumerate(world):
                invalid = i == 50 and j == 2
                point = [9999999] * 3 if invalid else point * 1000
                rows.append(
                    {
                        "rigid_body_name": "pico_head",
                        "frame_no": i,
                        "marker_id": j + 1,
                        "valid": int(not invalid),
                        "x_mm": point[0],
                        "y_mm": point[1],
                        "z_mm": point[2],
                    }
                )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "markers.csv"
            write_csv(path, rows)
            quality = gate_reference(data, path, guard_s=0.1)
        self.assertEqual(quality["raw_rejected_frames"], 1)
        self.assertFalse(data.good[50])
        self.assertFalse(data.good[45])
        self.assertTrue(data.good[30])

    def test_pairwise_transform_difference_is_rigid_and_not_euler_subtraction(self):
        identity = np.eye(4)
        other = unpack([0, 0, np.radians(2), 0.003, 0.004, 0])
        first = {
            "session": "one",
            **{k: identity.tolist() for k in ("T_B_H", "T_N_P", "T_P_N")},
        }
        second = {
            "session": "two",
            **{k: other.tolist() for k in ("T_B_H", "T_N_P", "T_P_N")},
        }
        pairs = compare_transforms([first, second])
        self.assertEqual(len(pairs), 3)
        self.assertAlmostEqual(pairs[0]["translation_difference_mm"], 5)
        self.assertAlmostEqual(pairs[0]["rotation_difference_deg"], 2)

    def test_fixed_calibration_exposes_later_drift_and_handles_empty_metric(self):
        t = np.arange(0, 20, 0.1)
        n = motion(t)
        p = n.copy()
        p[t >= 10, 0, 3] += 0.02
        x, y, _ = fit_extrinsics(n[t < 10], p[t < 10], t[t < 10])
        measured, _, _ = metrics(y @ p, n @ x, t >= 10)
        self.assertAlmostEqual(measured["position_rmse_mm"], 20, places=6)
        empty, _, _ = metrics(y @ p, n @ x, np.zeros(len(t), bool))
        self.assertEqual(empty["poses"], 0)
        self.assertIsNone(empty["position_rmse_mm"])
        self.assertNotIn("position_rmse_m", empty)

    def test_end_to_end_raw_csv_analysis_uses_held_out_poses(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "session"
            (root / "pico").mkdir(parents=True)
            (root / "nokov").mkdir()
            x, y = (
                unpack([0.1, -0.2, 0.15, 0.03, -0.04, 0.02]),
                unpack([0.2, 0.1, -0.4, -0.4, 0.2, 0.7]),
            )
            nt = np.round(np.arange(0, 21, 1 / 90), 3)
            pt = np.arange(0, 17, 0.014)
            a, b = 1.00008, 2.2
            n, p = motion(nt), inv(y) @ motion(a * pt + b) @ x
            head, tracker, bodies, markers = [], [], [], []
            template = np.array(
                [
                    [0, 0, 0],
                    [0.04, 0, 0],
                    [0, 0.03, 0],
                    [0.015, 0.018, 0.02],
                    [-0.01, 0.02, 0.005],
                ]
            )
            host_origin = 1230000000000000
            for i, matrix in enumerate(p):
                timestamp = 4355000000000 + round(pt[i] * 1e9)
                host = host_origin + round((a * pt[i] + b) * 1e9) + 7_000_000
                row = {
                    "sample_index": i,
                    "host_receive_perf_ns": host,
                    "host_receive_unix_ns": host + 10**18,
                    "valid_numeric": 1,
                    **dict(zip(("x_m", "y_m", "z_m"), matrix[:3, 3])),
                    **dict(
                        zip(
                            ("qx", "qy", "qz", "qw"),
                            Rotation.from_matrix(matrix[:3, :3]).as_quat(),
                        )
                    ),
                }
                head.append({**row, "head_timestamp_ns": timestamp})
                tracker.append(
                    {
                        **row,
                        "motion_timestamp_ns": timestamp,
                        "tracker_index": 0,
                        "tracker_serial": "serial",
                        "selected": 1,
                    }
                )
            for i, matrix in enumerate(n):
                row = {
                    "frame_no": i,
                    "device_timestamp_raw": 1793876000000 + round(nt[i] * 1000),
                    "receive_perf_ns": host_origin + round(nt[i] * 1e9) + 10_000_000,
                    "valid_numeric": 1,
                    "rigid_body_name": "pico_head",
                    **dict(zip(("x_mm", "y_mm", "z_mm"), matrix[:3, 3] * 1000)),
                    **dict(
                        zip(
                            ("qx", "qy", "qz", "qw"),
                            Rotation.from_matrix(matrix[:3, :3]).as_quat(),
                        )
                    ),
                }
                bodies.append(row)
                world = template @ matrix[:3, :3].T + matrix[:3, 3]
                for j, point in enumerate(world):
                    markers.append(
                        {
                            "frame_no": i,
                            "rigid_body_name": "pico_head",
                            "marker_id": j + 1,
                            "valid": 1,
                            **dict(zip(("x_mm", "y_mm", "z_mm"), point * 1000)),
                        }
                    )
            write_csv(root / "pico/pico_head_poses.csv", head)
            write_csv(root / "pico/pico_tracker_poses.csv", tracker)
            write_csv(root / "nokov/nokov_rigid_bodies.csv", bodies)
            write_csv(root / "nokov/nokov_rigid_body_markers.csv", markers)
            (root / "pico/capture_metadata.json").write_text(
                json.dumps(
                    {
                        "schema": "pico_raw_tracking_capture_v1",
                        "status": "complete",
                        "rows": {
                            "head": len(head),
                            "tracker": len(tracker),
                            "motion_packets": len(head),
                        },
                        "selected_tracker_serials": ["serial"],
                    }
                )
            )
            (root / "nokov/capture_metadata.json").write_text("{}")
            report, data = analyze_session(root, 10)
        self.assertEqual(report["status"], "ok")
        self.assertTrue(report["raw_inputs_unchanged"])
        self.assertGreater(report["metrics"]["validation"]["poses"], 100)
        self.assertLess(report["metrics"]["validation"]["position_rmse_mm"], 0.1)
        np.testing.assert_allclose(report["T_B_H"], x, atol=0.0003)
        self.assertTrue((data["pico"].t[data["validation"]] > 10.2).all())


if __name__ == "__main__":
    unittest.main()
