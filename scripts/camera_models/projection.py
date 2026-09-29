"""Dispatch raw-image projection by calibration model."""

from __future__ import annotations

import numpy as np

from ego_data.calibration import CameraCalibration


def project_points(camera: CameraCalibration, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if camera.model == "KB":
        from .kb import project
    elif camera.model == "DS":
        from .double_sphere import project
    else:
        raise ValueError(f"unsupported camera model: {camera.model}")
    return project(camera, points)


def unproject_points(camera: CameraCalibration, pixels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Dispatch native image pixels to unit camera-frame rays."""
    if camera.model == "KB":
        from .kb import unproject
    elif camera.model == "DS":
        from .double_sphere import unproject
    else:
        raise ValueError(f"unsupported camera model: {camera.model}")
    return unproject(camera, pixels)
