"""Supply calibrated crop geometry to WiLoR's trained physical camera head."""

from __future__ import annotations

import cv2
import numpy as np
import torch


def prepare_physical_batch(batch, boxes, calibration, camera_model):
    expected = {"physical_kb": "KB", "physical_perspective": "PINHOLE"}
    if calibration is None or calibration.model != expected.get(camera_model):
        raise ValueError(f"{camera_model} requires matching calibrated source images")
    device = batch["img"].device
    size = float(batch["img"].shape[-1])
    center = batch["box_center"].float().clone()
    image_size = batch["img_size"].float()
    expected_size = image_size.new_tensor(calibration.image_size)
    if not torch.all(image_size == expected_size):
        raise ValueError("Physical camera calibration/image size mismatch")
    side = batch["right"].long()
    center[:, 0] = torch.where(side == 0, image_size[:, 0] - center[:, 0] - 1, center[:, 0])
    scale = size / batch["box_size"].float()
    affine = torch.zeros((len(center), 2, 3), device=device, dtype=torch.float32)
    affine[:, 0, 0] = scale
    affine[:, 1, 1] = scale
    affine[:, :, 2] = size / 2 - scale[:, None] * center
    batch.update(
        camera_K=torch.as_tensor(calibration.K.copy(), device=device, dtype=torch.float32)
                      .expand(len(center), -1, -1),
        camera_distortion=torch.as_tensor(calibration.distortion.copy(), device=device, dtype=torch.float32)
                               .expand(len(center), -1),
        crop_transform=affine,
        crop_box_size=batch["box_size"].float(),
        image_size=image_size,
        side=side,
        bbox_xyxy=torch.as_tensor(np.asarray(boxes), device=device, dtype=torch.float32),
    )


def project_physical_points(points, calibration):
    """Project camera-space joints or vertices exactly as the trained head does."""
    if calibration.model == "KB":
        return cv2.fisheye.projectPoints(
            np.asarray(points, dtype=np.float64)[:, None], np.zeros(3), np.zeros(3),
            calibration.K, calibration.distortion,
        )[0][:, 0]
    homogeneous = points @ calibration.K.T
    return homogeneous[:, :2] / homogeneous[:, 2:]


def physical_camera_outputs(result, batch, calibration):
    """Return MANO-origin translation and joints in original image pixels.

    The trained head predicts wrist position; existing multiview consumers
    instead add camera_translation to the uncentred, mirrored MANO joints.
    """
    joints = result["pred_keypoints_3d"].detach().float().cpu().numpy().copy()
    side = batch["right"].detach().float().cpu().numpy()
    joints[:, :, 0] *= (2 * side - 1)[:, None]
    root = result["pred_root_camera_m"].detach().float().cpu().numpy()
    translation = root - joints[:, 0]
    points = joints + translation[:, None]
    pixels = np.asarray([project_physical_points(hand, calibration) for hand in points])
    if not np.isfinite(translation).all() or not np.isfinite(pixels).all():
        raise ValueError("Nonfinite physical WiLoR camera predictions")
    return translation, pixels
