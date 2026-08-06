#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import cv2
import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "fuse_basalt_hand_trajectory.py"
SPEC = importlib.util.spec_from_file_location("fuse_basalt_hand_trajectory", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class FuseBasaltHandTrajectoryTests(unittest.TestCase):
    def test_quaternion_matrix_round_trip(self):
        rotation, _ = cv2.Rodrigues(np.asarray((0.2, -0.35, 0.1)))
        quaternion = MODULE.quaternion_xyzw_from_matrix(rotation)
        np.testing.assert_allclose(
            MODULE.quaternion_xyzw_to_matrix(quaternion), rotation, atol=1e-12
        )

    def test_static_world_hand_is_constant_when_camera_moves(self):
        t_world_camera0 = MODULE.make_transform(np.eye(3), (0.0, 0.0, 0.0))
        rotation1, _ = cv2.Rodrigues(np.asarray((0.0, 0.0, 0.2)))
        t_world_camera1 = MODULE.make_transform(rotation1, (0.1, -0.03, 0.02))
        t_world_hand = MODULE.make_transform(np.eye(3), (0.4, 0.2, 0.5))
        t_camera0_hand = MODULE.invert_transform(t_world_camera0) @ t_world_hand
        t_camera1_hand = MODULE.invert_transform(t_world_camera1) @ t_world_hand
        reconstructed0 = t_world_camera0 @ t_camera0_hand
        reconstructed1 = t_world_camera1 @ t_camera1_hand
        np.testing.assert_allclose(reconstructed0, reconstructed1, atol=1e-12)

    def test_origin_shift_preserves_gravity_aligned_rotation(self):
        rotation, _ = cv2.Rodrigues(np.asarray((0.1, -0.2, 0.3)))
        first = MODULE.make_transform(rotation, (0.4, -0.2, 1.1))
        shift = MODULE.make_transform(np.eye(3), -first[:3, 3])
        shifted = shift @ first
        np.testing.assert_allclose(shifted[:3, 3], np.zeros(3), atol=1e-12)
        np.testing.assert_allclose(shifted[:3, :3], rotation, atol=1e-12)

    def test_projected_world_point_tracks_current_camera(self):
        point_world = np.asarray([[0.0, 0.0, 1.0]])
        t_world_camera = MODULE.make_transform(np.eye(3), (0.1, 0.0, 0.0))
        point_camera = MODULE.transform_points(
            MODULE.invert_transform(t_world_camera), point_world
        )
        np.testing.assert_allclose(point_camera[0], (-0.1, 0.0, 1.0), atol=1e-12)


if __name__ == "__main__":
    unittest.main()
