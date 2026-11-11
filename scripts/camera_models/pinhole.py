"""Undistorted pinhole projection in the original camera optical frame."""

from __future__ import annotations

import numpy as np

from ego_data.calibration import CameraCalibration


def project(camera: CameraCalibration, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=np.float64)
    if camera.model != "PINHOLE" or points.shape[-1] != 3:
        raise ValueError("pinhole projection requires a PINHOLE camera and (..., 3) points")
    flat = points.reshape(-1, 3)
    valid = np.isfinite(flat).all(axis=1) & (flat[:, 2] > 1e-12)
    pixels = np.full((len(flat), 2), np.nan)
    homogeneous = flat[valid] @ camera.K.T
    pixels[valid] = homogeneous[:, :2] / homogeneous[:, 2, None]
    return pixels.reshape(points.shape[:-1] + (2,)), valid.reshape(points.shape[:-1])


def unproject(camera: CameraCalibration, pixels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pixels = np.asarray(pixels, dtype=np.float64)
    if camera.model != "PINHOLE" or pixels.shape[-1] != 2:
        raise ValueError("pinhole unprojection requires a PINHOLE camera and (..., 2) pixels")
    flat = pixels.reshape(-1, 2)
    rays = np.column_stack((flat, np.ones(len(flat)))) @ np.linalg.inv(camera.K).T
    norms = np.linalg.norm(rays, axis=1)
    valid = np.isfinite(rays).all(axis=1) & (norms > 1e-12) & (rays[:, 2] > 0)
    rays[valid] /= norms[valid, None]
    rays[~valid] = np.nan
    return rays.reshape(pixels.shape[:-1] + (3,)), valid.reshape(pixels.shape[:-1])
