"""Dispatch stereo rectification by explicit or calibrated camera model."""

from __future__ import annotations

from ego_data.calibration import StereoCalibration
from .base import RectificationOptions, RectificationResult


def create_stereo_rectification(
    calibration: StereoCalibration,
    options: RectificationOptions | None = None,
    model: str = "auto",
) -> RectificationResult:
    options = options or RectificationOptions()
    calibrated_models = {calibration.left.model, calibration.right.model}
    if len(calibrated_models) != 1:
        raise ValueError("mixed camera models are not supported for stereo rectification")
    calibrated = calibrated_models.pop()
    requested = calibrated if model.lower() == "auto" else model.upper()
    if requested not in ("KB", "DS"):
        raise ValueError("camera model must be auto, kb, or ds")
    if requested != calibrated:
        raise ValueError(f"requested {requested}, but calibration contains {calibrated}")
    if requested == "KB":
        from .kb import create_rectification
    else:
        from .double_sphere import create_rectification
    return create_rectification(calibration, options)
