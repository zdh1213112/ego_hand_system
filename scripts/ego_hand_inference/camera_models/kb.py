"""Spherical Kannala-Brandt projection."""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from ego_hand_inference.ego_data.calibration import CameraCalibration


KB_PROJECTION_VERSION = "kb_spherical_monotonic_v2"


def _radius(theta, distortion):
    k1, k2, k3, k4 = distortion
    theta2 = theta * theta
    return theta * (1 + theta2 * (k1 + theta2 * (k2 + theta2 * (k3 + theta2 * k4))))


@lru_cache(maxsize=128)
def _theta_limit(distortion):
    # Invert only the branch connected to the optical axis. Stop before the
    # first stationary radius, or the ambiguous backward optical axis at pi.
    k1, k2, k3, k4 = distortion
    roots = np.polynomial.polynomial.polyroots([1., 3 * k1, 5 * k2, 7 * k3, 9 * k4])
    limits = [float(np.pi)]
    for root in roots:
        if abs(root.imag) <= 1e-10 * max(1., abs(root.real)) and 0 < root.real < np.pi ** 2:
            limits.append(float(np.sqrt(root.real)))
    return min(limits)


def theta_limit(camera: CameraCalibration) -> float:
    """Angular bound of the injective KB branch; image clipping is separate."""
    if camera.model != "KB":
        raise ValueError("KB angular domain requires a KB camera")
    return _theta_limit(tuple(map(float, camera.distortion)))


def project(camera: CameraCalibration, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=np.float64)
    if camera.model != "KB" or points.shape[-1] != 3:
        raise ValueError("KB projection requires a KB camera and (..., 3) points")
    flat = points.reshape(-1, 3)
    radius = np.hypot(flat[:, 0], flat[:, 1])
    theta = np.arctan2(radius, flat[:, 2])
    valid = (np.isfinite(flat).all(axis=1) & (np.hypot(radius, flat[:, 2]) > 1e-12)
             & (theta < theta_limit(camera)))
    pixels = np.full((len(flat), 2), np.nan)
    # OpenCV's x/z parametrization folds negative-z directions. atan2 keeps
    # the correct spherical branch on both sides of the 90-degree boundary.
    radial = _radius(theta[valid], camera.distortion)
    scale = np.divide(radial, radius[valid], out=np.zeros_like(radial), where=radius[valid] > 0)
    pixels[valid] = flat[valid, :2] * scale[:, None] * [camera.fx, camera.fy] + [camera.cx, camera.cy]
    valid &= np.isfinite(pixels).all(axis=1)
    pixels[~valid] = np.nan
    return pixels.reshape(points.shape[:-1] + (2,)), valid.reshape(points.shape[:-1])


def unproject(camera: CameraCalibration, pixels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert native KB image pixels to unit rays in the camera optical frame."""
    if camera.model != "KB":
        raise ValueError("KB unprojection requires a KB camera")
    pixels = np.asarray(pixels, dtype=np.float64)
    if pixels.shape[-1] != 2:
        raise ValueError("pixels must have a final dimension of two")
    flat = pixels.reshape(-1, 2)
    normalized = (flat - [camera.cx, camera.cy]) / [camera.fx, camera.fy]
    radius = np.hypot(normalized[:, 0], normalized[:, 1])
    limit = theta_limit(camera)
    valid = np.isfinite(flat).all(axis=1) & (radius < _radius(limit, camera.distortion))
    rays = np.full((len(flat), 3), np.nan, dtype=np.float64)
    if np.any(valid):
        indices = np.flatnonzero(valid)
        target = radius[indices]
        lower, upper = np.zeros_like(target), np.full_like(target, limit)
        theta = np.minimum(target, limit * .5)
        k1, k2, k3, k4 = camera.distortion
        # Safeguarded Newton iteration cannot jump to another polynomial root.
        for _ in range(48):
            residual = _radius(theta, camera.distortion) - target
            converged = np.abs(residual) <= 1e-12
            if np.all(converged):
                break
            lower = np.where(residual < 0, theta, lower)
            upper = np.where(residual >= 0, theta, upper)
            theta2 = theta * theta
            derivative = 1 + theta2 * (3 * k1 + theta2 * (5 * k2 + theta2 * (7 * k3 + theta2 * 9 * k4)))
            proposal = theta - residual / np.maximum(derivative, 1e-15)
            update = np.where((proposal > lower) & (proposal < upper), proposal, (lower + upper) * .5)
            theta = np.where(converged, theta, update)
        scale = np.divide(np.sin(theta), target, out=np.ones_like(theta), where=target > 0)
        candidates = np.column_stack((normalized[indices] * scale[:, None], np.cos(theta)))
        projected, projectable = project(camera, candidates)
        residual = np.linalg.norm(projected - flat[indices], axis=1)
        roundtrip = projectable & np.isfinite(residual) & (residual <= 0.5)
        rays[indices[roundtrip]] = candidates[roundtrip]
        valid[:] = False
        valid[indices[roundtrip]] = True
    return rays.reshape(pixels.shape[:-1] + (3,)), valid.reshape(pixels.shape[:-1])
