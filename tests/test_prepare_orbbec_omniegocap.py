from __future__ import annotations

import contextlib
import csv
import io
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from prepare_orbbec_omniegocap import interpolate, neutral_head_rotation, prepare, wrist_from_joints
from filter_omniegocap_wrist_rotations import rotation_rejections


def palm(side):
    p = np.full((21, 3), np.nan)
    sign = 1 if side == 0 else -1
    p[0] = [0.1, 0.2, 0.3]
    p[5] = p[0] + [sign * 0.08, 0, 0.025]
    p[9] = p[0] + [sign * 0.09, 0, 0]
    p[17] = p[0] + [sign * 0.07, 0, -0.025]
    return p


class PrepareOmniTests(unittest.TestCase):
    def test_rotation_gate_preserves_gaps_and_independent_hand_masks(self):
        w = np.tile(np.eye(4), (4, 2, 1, 1))
        w[:, 0, :3, :3] = Rotation.from_euler("x", [0, 90, 90, 0], degrees=True).as_matrix()
        v = np.ones((4, 2), dtype=bool)
        v[2, 0] = False
        rejected, pairs = rotation_rejections(w, v, 60)
        np.testing.assert_array_equal(rejected[:, 0], [True, True, False, False])
        self.assertFalse(rejected[:, 1].any())
        self.assertEqual(len(pairs), 1)
        self.assertTrue(v[0, 0])  # the audit must not mutate source masks

    def test_handedness_and_rigid_equivariance(self):
        R = Rotation.from_rotvec([0.3, -0.2, 0.7]).as_matrix()
        translation = np.array([1, 2, 3])
        for side in (0, 1):
            points = palm(side)
            T = wrist_from_joints(points, side)
            np.testing.assert_allclose(T[:3, :3], np.eye(3), atol=1e-8)
            moved = points @ R.T + translation
            T_moved = wrist_from_joints(moved, side)
            np.testing.assert_allclose(T_moved[:3, :3], R, atol=1e-8)
            np.testing.assert_allclose(T_moved[:3, 3], R @ points[0] + translation)
        points[17] = points[5]
        self.assertIsNone(wrist_from_joints(points, 1))

    def test_interpolation_rejects_extrapolation(self):
        poses = np.tile(np.eye(4), (2, 1, 1))
        poses[1, :3, :3] = Rotation.from_euler("z", 90, degrees=True).as_matrix()
        poses[1, 0, 3] = 2
        mid = interpolate(poses, np.array([0., 1.]), np.array([0.5]))[0]
        np.testing.assert_allclose(mid[:3, :3], Rotation.from_euler("z", 45, degrees=True).as_matrix())
        self.assertAlmostEqual(mid[0, 3], 1)
        with self.assertRaises(ValueError):
            interpolate(poses, np.array([0., 1.]), np.array([1.1]))

    def test_imu_head_neutral_orientation_and_relative_motion(self):
        imu0 = np.eye(4)
        imu0[:3, :3] = Rotation.from_euler("xyz", [8, -6, 32], degrees=True).as_matrix()
        cam0 = np.eye(4)
        cam0[:3, :3] = imu0[:3, :3] @ Rotation.from_euler("y", 50, degrees=True).as_matrix()
        R_ih = neutral_head_rotation(imu0, cam0)
        H0 = imu0[:3, :3] @ R_ih
        np.testing.assert_allclose(H0[:, 1], [0, 0, 1], atol=1e-8)
        self.assertAlmostEqual(H0[2, 2], 0)
        self.assertAlmostEqual(np.linalg.det(H0), 1)
        delta = Rotation.from_rotvec([0.3, -0.1, 0.2]).as_matrix()
        H1 = delta @ imu0[:3, :3] @ R_ih
        np.testing.assert_allclose(H1 @ H0.T, delta, atol=1e-8)

    def test_clock_transform_masks_floor_and_missing_calibration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, fusion, vio = (root / n for n in ("dataset", "fusion", "vio"))
            (dataset / "calibration").mkdir(parents=True)
            fusion.mkdir()
            (vio / "input_adapter").mkdir(parents=True)
            (dataset / "manifest.json").write_text(json.dumps({"source": {"kind": "orbbec_ego4_export"}}))
            (dataset / "calibration/camera0.json").write_text(json.dumps({"T_base_camera": np.eye(4).tolist()}))
            origin = 1790577829710891000
            shift = 0
            timestamps = [origin + i * 33334000 for i in range(4)]
            with (dataset / "multiview_frames.csv").open("w") as f:
                w = csv.writer(f)
                w.writerow(["sync_index", "camera0_timestamp_ns"])
                w.writerows(enumerate(timestamps))
            with (vio / "headring_poses.csv").open("w") as f:
                w = csv.writer(f)
                w.writerow(["timestamp_ns", "source_cam0_timestamp_us", "x_m", "y_m", "z_m", "qx", "qy", "qz", "qw"])
                for i, t in enumerate(timestamps[:3]):
                    w.writerow([t + shift, t // 1000, i * 0.1, 0, 0, 0, 0, 0, 1])
            (vio / "input_adapter/metadata.json").write_text(json.dumps({"camera_to_imu_shift_ns": shift}))
            rows = []
            for i in range(4):
                counts = np.full(21, 3)
                if i == 1:
                    counts[5] = 1  # an unsupported palm cannot supply orientation
                rows.append({"sync_index": i, "hands": [{"side": 0, "joints_base_m": palm(0).tolist(), "inlier_view_counts": counts.tolist()}]})
            (fusion / "accepted.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
            args = SimpleNamespace(dataset=dataset, fusion=fusion, vio=vio,
                                   output=root / "partial", head_to_camera=None, floor_z=None,
                                   min_palm_views=3, calibration_note="synthetic test")
            with contextlib.redirect_stdout(io.StringIO()):
                prepare(args)
            self.assertFalse((args.output / "omniegocap_input.npz").exists())
            data = np.load(args.output / "wrists_world.npz")
            np.testing.assert_array_equal(data["visible"][:, 0], [True, False, True, False])
            np.testing.assert_allclose(data["wrist"][2, 0, :3, 3], [0.3, 0.2, 0.3])
            np.testing.assert_allclose(data["wrist"][~data["visible"]], np.broadcast_to(np.eye(4), (6, 4, 4)))
            H = np.eye(4)
            H[:3, 3] = [0, 0, 0.1]
            np.save(root / "head.npy", H)
            args.head_to_camera, args.floor_z = root / "head.npy", -1.5
            args.output = root / "complete"
            with contextlib.redirect_stdout(io.StringIO()):
                prepare(args)
            final = np.load(args.output / "omniegocap_input.npz")
            np.testing.assert_allclose(final["head"][:, 2, 3], 1.6)
            self.assertEqual(final["head"].shape, (3, 4, 4))
            self.assertEqual(final["visible"].dtype, np.bool_)
            self.assertEqual(final["head"].dtype, np.float32)
            np.testing.assert_array_equal(final["visible"][:, 0], [True, False, True])
            self.assertAlmostEqual(final["raw_time_s"][0], 0)

            # IMU position must come from the IMU trajectory, not camera position.
            with (vio / "trajectory.csv").open("w") as f:
                f.write("#timestamp,px,py,pz,qw,qx,qy,qz\n")
                w = csv.writer(f)
                for i, t in enumerate(timestamps[:3]):
                    q = Rotation.from_euler("y", 6 + i, degrees=True).as_quat()
                    w.writerow([t + shift, i * 0.03, 0, 0, q[3], *q[:3]])
            with (vio / "headring_poses.csv").open("w") as f:
                w = csv.writer(f)
                w.writerow(["timestamp_ns", "source_cam0_timestamp_us", "x_m", "y_m", "z_m", "qx", "qy", "qz", "qw"])
                for i, t in enumerate(timestamps[:3]):
                    q = Rotation.from_euler("y", 60 + i, degrees=True).as_quat()
                    w.writerow([t + shift, t // 1000, i * 0.1, 0, 0, *q])
            args.head_to_camera = None
            args.head_at_imu = args.neutral_first_frame = True
            args.floor_z = -1.7
            args.output = root / "imu_approx"
            with contextlib.redirect_stdout(io.StringIO()):
                prepare(args)
            imu_final = np.load(args.output / "omniegocap_input.npz")
            expected_x = np.interp(imu_final["raw_time_s"], np.array([0, .033334, .066668]), [0, .03, .06])
            np.testing.assert_allclose(imu_final["head"][:, 0, 3], expected_x, atol=1e-7)
            np.testing.assert_allclose(imu_final["head"][:, 2, 3], 1.7)
            np.testing.assert_allclose(imu_final["head"][0, :3, 1], [0, 0, 1], atol=1e-7)
            info = json.loads(str(imu_final["meta"]))
            self.assertFalse(info["head_calibration_provided"])
            self.assertEqual(info["head_position_reference"], "IMU origin")


if __name__ == "__main__":
    unittest.main()
