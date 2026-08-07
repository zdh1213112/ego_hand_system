"""OpenCV Kannala-Brandt fisheye projection and rectification."""

from __future__ import annotations

import cv2
import numpy as np

from ego_data.calibration import CameraCalibration, StereoCalibration
from .base import RectificationOptions, RectificationResult


def project(camera: CameraCalibration, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if camera.model != "KB":
        raise ValueError("KB projection requires a KB camera")
    points = np.asarray(points, dtype=np.float64)
    flat = points.reshape(-1, 3)
    projected, _ = cv2.fisheye.projectPoints(
        flat.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), camera.K, camera.distortion
    )
    valid = np.all(np.isfinite(flat), axis=1) & (flat[:, 2] > 1e-12)
    pixels = projected[:, 0]
    pixels[~valid] = np.nan
    return pixels.reshape(points.shape[:-1] + (2,)), valid.reshape(points.shape[:-1])


def create_rectification(
    calibration: StereoCalibration, options: RectificationOptions
) -> RectificationResult:
    if calibration.left.model != "KB" or calibration.right.model != "KB":
        raise ValueError("KB rectification requires two KB cameras")
    output_size = options.output_size or calibration.left.image_size
    R1, R2, P1, P2, Q = cv2.fisheye.stereoRectify(
        calibration.left.K, calibration.left.distortion,
        calibration.right.K, calibration.right.distortion,
        calibration.left.image_size,
        calibration.R_left_to_right, calibration.t_left_to_right_m,
        flags=cv2.CALIB_ZERO_DISPARITY,
        newImageSize=output_size,
        balance=options.balance,
        fov_scale=1.0 / options.focal_scale,
    )
    map_left_x, map_left_y = cv2.fisheye.initUndistortRectifyMap(
        calibration.left.K, calibration.left.distortion, R1, P1,
        output_size, cv2.CV_32FC1,
    )
    map_right_x, map_right_y = cv2.fisheye.initUndistortRectifyMap(
        calibration.right.K, calibration.right.distortion, R2, P2,
        output_size, cv2.CV_32FC1,
    )
    raw_width, raw_height = calibration.left.image_size
    valid_left = (
        (map_left_x >= 0) & (map_left_x <= raw_width - 1)
        & (map_left_y >= 0) & (map_left_y <= raw_height - 1)
    )
    valid_right = (
        (map_right_x >= 0) & (map_right_x <= raw_width - 1)
        & (map_right_y >= 0) & (map_right_y <= raw_height - 1)
    )
    return RectificationResult(
        output_size, map_left_x, map_left_y, map_right_x, map_right_y,
        valid_left, valid_right, R1, R2, P1, P2, Q, "KB"
    )
