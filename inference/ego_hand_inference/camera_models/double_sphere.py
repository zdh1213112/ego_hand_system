"""Double Sphere projection and back-projection."""

from __future__ import annotations

import numpy as np

from ego_hand_inference.ego_data.calibration import CameraCalibration


def project(camera: CameraCalibration, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project 3D camera-frame points with the stored DS parameter order."""
    if camera.model != "DS":
        raise ValueError("Double Sphere projection requires a DS camera")
    points = np.asarray(points, dtype=np.float64)
    if points.shape[-1] != 3:
        raise ValueError("points must have a final dimension of three")
    flat = points.reshape(-1, 3)
    x, y, z = flat.T
    fx, fy, cx, cy, xi, alpha = camera.distortion
    d1 = np.linalg.norm(flat, axis=1)
    z1 = alpha * d1 + (1.0 - alpha) * z
    d2 = np.sqrt(x * x + y * y + z1 * z1)
    denominator = xi * d2 + z1
    finite = np.all(np.isfinite(flat), axis=1)
    valid = finite & (d1 > 1e-12) & (denominator > 1e-12)
    normalized = np.full((len(flat), 2), np.nan, dtype=np.float64)
    normalized[valid, 0] = x[valid] / denominator[valid]
    normalized[valid, 1] = y[valid] / denominator[valid]
    pixels = np.empty_like(normalized)
    pixels[:, 0] = fx * normalized[:, 0] + cx
    pixels[:, 1] = fy * normalized[:, 1] + cy
    return pixels.reshape(points.shape[:-1] + (2,)), valid.reshape(points.shape[:-1])


def unproject(camera: CameraCalibration, pixels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert Double Sphere image pixels to unit camera-frame rays."""
    if camera.model != "DS":
        raise ValueError("Double Sphere unprojection requires a DS camera")
    pixels = np.asarray(pixels, dtype=np.float64)
    if pixels.shape[-1] != 2:
        raise ValueError("pixels must have a final dimension of two")
    flat = pixels.reshape(-1, 2)
    fx, fy, cx, cy, xi, alpha = camera.distortion
    mx = (flat[:, 0] - cx) / fx
    my = (flat[:, 1] - cy) / fy
    radius2 = mx * mx + my * my
    # The calibration stores and projects the two DS coefficients in its own convention:
    #   z1 = alpha * ||p|| + (1-alpha) * z
    #   m  = (x, y) / (xi * sqrt(x*x+y*y+z1*z1) + z1)
    # Invert that exact equation rather than the more common Usenko parameter
    # ordering.  q is z1 divided by the unknown projection denominator.
    xi_denominator = 1.0 - xi * xi
    valid = np.all(np.isfinite(flat), axis=1) & (abs(xi_denominator) > 1e-12)
    xi_root = np.sqrt(np.maximum(1.0 + xi_denominator * radius2, 0.0))
    q = (1.0 - xi * xi_root) / xi_denominator
    scale_denominator = q * q + (1.0 - alpha) ** 2 * radius2
    root_argument = q * q + (1.0 - 2.0 * alpha) * radius2
    valid &= (scale_denominator > 1e-12) & (root_argument >= 0.0)
    scale = np.full(len(flat), np.nan, dtype=np.float64)
    scale[valid] = (
        alpha * q[valid]
        + (1.0 - alpha) * np.sqrt(root_argument[valid])
    ) / scale_denominator[valid]
    rays = np.full((len(flat), 3), np.nan, dtype=np.float64)
    rays[valid, 0] = scale[valid] * mx[valid]
    rays[valid, 1] = scale[valid] * my[valid]
    rays[valid, 2] = (q[valid] * scale[valid] - alpha) / (1.0 - alpha)
    norms = np.linalg.norm(rays, axis=1)
    valid &= norms > 1e-12
    rays[valid] /= norms[valid, None]
    return rays.reshape(pixels.shape[:-1] + (3,)), valid.reshape(pixels.shape[:-1])
