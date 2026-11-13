"""Projection dispatch for versioned WiLoR labels."""
import cv2
import numpy as np


def project_label_points(points, sample):
    points = np.asarray(points, dtype=np.float64)
    if not np.isfinite(points).all() or np.any(points[..., 2] <= 1e-8):
        raise ValueError("invalid or behind-camera geometry")
    K = np.asarray(sample["K"], dtype=np.float64)
    space = sample.get("image_space", "physical_rectified")
    if space == "native_raw":
        if sample.get("camera_model") != "KB":
            raise ValueError("native labels require KB projection")
        D = np.asarray(sample["distortion"], dtype=np.float64)
        if D.shape != (4,) or not np.isfinite(D).all():
            raise ValueError("KB requires four finite distortion coefficients")
        pixels, _ = cv2.fisheye.projectPoints(points.reshape(-1, 1, 3),
            np.zeros(3), np.zeros(3), K, D)
        return pixels.reshape(points.shape[:-1] + (2,))
    if space != "physical_rectified":
        raise ValueError(f"unsupported label image space: {space}")
    homogeneous = points @ K.T
    return homogeneous[..., :2] / homogeneous[..., 2:3]
