"""Calibrated projection and translation placement for WiLoR's oriented MANO geometry."""
from __future__ import annotations
import cv2
import numpy as np
from scipy.optimize import least_squares


def proper_rotation(matrix):
    """Remove AMP rounding error from a learned rotation before exporting SE(3)."""
    U, _, Vt = np.linalg.svd(np.asarray(matrix, dtype=np.float64))
    if np.linalg.det(U @ Vt) < 0:
        U[:, -1] *= -1
    return (U @ Vt).astype(np.float32)


def project_pinhole(points, K):
    points = np.asarray(points, dtype=np.float64)
    return points[:, :2] / points[:, 2:] * np.asarray([K[0, 0], K[1, 1]]) + K[:2, 2]


def fit_translation(points, pixels, K):
    """Keep the predicted shape/orientation and fit translation to learned 2D projections.

    This is calibrated monocular placement, not stereo depth or independent 2D evidence.
    """
    points, pixels = np.asarray(points, dtype=np.float64), np.asarray(pixels, dtype=np.float64)
    normalized = (pixels - K[:2, 2]) / np.asarray([K[0, 0], K[1, 1]])
    A, b = [], []
    for point, uv in zip(points, normalized):
        A.extend(([1, 0, -uv[0]], [0, 1, -uv[1]]))
        b.extend((uv[0] * point[2] - point[0], uv[1] * point[2] - point[1]))
    translation = np.linalg.lstsq(np.asarray(A), np.asarray(b), rcond=None)[0]
    minimum_z = -float(points[:, 2].min()) + 0.02
    translation[2] = max(translation[2], minimum_z + 0.01)
    result = least_squares(lambda t: (project_pinhole(points + t, K) - pixels).ravel(),
                           translation, bounds=([-np.inf, -np.inf, minimum_z], [np.inf] * 3),
                           loss="soft_l1", f_scale=3.0, max_nfev=40)
    error = project_pinhole(points + result.x, K) - pixels
    return result.x.astype(np.float32), float(np.sqrt(np.mean(np.sum(error * error, axis=1))))


def native_pixels(points, calibration):
    pixels, _ = cv2.fisheye.projectPoints(np.asarray(points, dtype=np.float64).reshape(-1, 1, 3),
                                         np.zeros(3), np.zeros(3), calibration.K, calibration.distortion)
    return pixels[:, 0].astype(np.float32)


def pinhole_view(calibration, width=1000, height=750, horizontal_fov_deg=130.0):
    focal = width / (2 * np.tan(np.deg2rad(horizontal_fov_deg) / 2))
    K = np.asarray([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]], dtype=np.float64)
    maps = cv2.fisheye.initUndistortRectifyMap(calibration.K, calibration.distortion,
                                             np.eye(3), K, (width, height), cv2.CV_32FC1)
    return K, maps
