"""Common calibration and dataset contracts for offline stereo recordings."""

from .calibration import CameraCalibration, StereoCalibration
from .dataset import NormalizedStereoDataset, RectifiedStereoDataset

__all__ = [
    "CameraCalibration",
    "StereoCalibration",
    "NormalizedStereoDataset",
    "RectifiedStereoDataset",
]
