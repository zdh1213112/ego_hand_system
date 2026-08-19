"""GEN Double Sphere projection and stereo-to-pinhole rectification."""

from __future__ import annotations

import numpy as np

from ego_data.calibration import CameraCalibration, StereoCalibration
from .base import RectificationOptions, RectificationResult


def project(camera: CameraCalibration, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project 3D camera-frame points with GEN's published DS parameter order."""
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
    # GEN stores and projects the two DS coefficients in its own convention:
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


def _rectification_rotations(calibration: StereoCalibration) -> tuple[np.ndarray, np.ndarray]:
    R = calibration.R_left_to_right
    t = calibration.t_left_to_right_m
    right_center_left = -(R.T @ t)
    x_axis = right_center_left / np.linalg.norm(right_center_left)
    left_optical = np.asarray([0.0, 0.0, 1.0])
    right_optical_left = R.T @ left_optical
    z_axis = left_optical + right_optical_left
    z_axis -= x_axis * np.dot(z_axis, x_axis)
    if np.linalg.norm(z_axis) < 1e-8:
        raise ValueError("cannot construct rectified optical axis")
    z_axis /= np.linalg.norm(z_axis)
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    z_axis = np.cross(x_axis, y_axis)
    R1 = np.vstack((x_axis, y_axis, z_axis))
    R2 = R1 @ R.T
    return R1, R2


def _map_for_camera(
    camera: CameraCalibration,
    rectification_rotation: np.ndarray,
    output_size: tuple[int, int],
    focal: float,
    principal: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    width, height = output_size
    cx, cy = principal
    grid_x, grid_y = np.meshgrid(np.arange(width, dtype=np.float64), np.arange(height, dtype=np.float64))
    rays_rectified = np.stack(
        ((grid_x - cx) / focal, (grid_y - cy) / focal, np.ones_like(grid_x)), axis=-1
    )
    rays_camera = rays_rectified @ rectification_rotation
    pixels, model_valid = project(camera, rays_camera)
    map_x = pixels[..., 0].astype(np.float32)
    map_y = pixels[..., 1].astype(np.float32)
    raw_width, raw_height = camera.image_size
    valid = (
        model_valid
        & np.isfinite(map_x) & np.isfinite(map_y)
        & (map_x >= 0.0) & (map_x <= raw_width - 1.0)
        & (map_y >= 0.0) & (map_y <= raw_height - 1.0)
    )
    map_x[~valid] = -1.0
    map_y[~valid] = -1.0
    return map_x, map_y, valid


def create_rectification(
    calibration: StereoCalibration, options: RectificationOptions
) -> RectificationResult:
    if calibration.left.model != "DS" or calibration.right.model != "DS":
        raise ValueError("DS rectification requires two DS cameras")
    output_size = options.output_size or calibration.left.image_size
    R1, R2 = _rectification_rotations(calibration)
    base_focal = float(np.median([
        calibration.left.fx, calibration.left.fy,
        calibration.right.fx, calibration.right.fy,
    ]))
    focal = base_focal * options.focal_scale
    width, height = output_size
    principal = ((width - 1.0) / 2.0, (height - 1.0) / 2.0)
    map_left_x, map_left_y, valid_left = _map_for_camera(
        calibration.left, R1, output_size, focal, principal
    )
    map_right_x, map_right_y, valid_right = _map_for_camera(
        calibration.right, R2, output_size, focal, principal
    )
    rectified_translation = R2 @ calibration.t_left_to_right_m
    if abs(rectified_translation[1]) > 1e-7 or abs(rectified_translation[2]) > 1e-7:
        raise RuntimeError("rectification failed to align the stereo translation with x")
    tx = float(rectified_translation[0])
    if tx >= 0:
        raise RuntimeError("right camera must produce negative rectified translation and positive disparity")
    cx, cy = principal
    P1 = np.asarray([
        [focal, 0.0, cx, 0.0],
        [0.0, focal, cy, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ])
    P2 = P1.copy()
    P2[0, 3] = focal * tx
    Q = np.asarray([
        [1.0, 0.0, 0.0, -cx],
        [0.0, 1.0, 0.0, -cy],
        [0.0, 0.0, 0.0, focal],
        [0.0, 0.0, -1.0 / tx, 0.0],
    ])
    return RectificationResult(
        output_size, map_left_x, map_left_y, map_right_x, map_right_y,
        valid_left, valid_right, R1, R2, P1, P2, Q, "DS"
    )
