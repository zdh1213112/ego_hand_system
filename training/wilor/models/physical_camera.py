"""Calibrated perspective projection for the rectified glove cameras.

The WiLoR camera is a virtual crop camera.  This module converts its wrist
estimate into a ray in the real camera, predicts a metric depth correction,
and projects the predicted hand with the label's rectified intrinsics.
"""

from __future__ import annotations

import torch
from torch import nn
from .kb_camera import (pixel_to_z_ray, physical_points_to_crop,
                        bounded_wrist_pixel, native_box_depth_prior)


def project_rectified_joints(
    joints: torch.Tensor,
    root: torch.Tensor,
    K: torch.Tensor,
    affine: torch.Tensor,
    width: torch.Tensor,
    left: torch.Tensor,
    crop_size: float,
) -> torch.Tensor:
    """Project canonical root-relative joints into a normalized image crop."""
    relative = joints - joints[:, :1]
    relative = relative * torch.where(
        left[:, None, None],
        joints.new_tensor([-1.0, 1.0, 1.0]),
        joints.new_tensor([1.0, 1.0, 1.0]),
    )
    camera_points = relative + root[:, None]
    safe_z = camera_points[..., 2].clamp_min(0.01)
    image_x = K[:, None, 0, 0] * camera_points[..., 0] / safe_z + K[:, None, 0, 2]
    image_y = K[:, None, 1, 1] * camera_points[..., 1] / safe_z + K[:, None, 1, 2]
    image_x = torch.where(left[:, None], width[:, None] - image_x - 1, image_x)
    image_xy = torch.stack((image_x, image_y), dim=-1)
    crop_xy = torch.bmm(image_xy, affine[:, :, :2].transpose(1, 2)) + affine[:, None, :, 2]
    return crop_xy / crop_size - 0.5


class PhysicalCameraHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.camera_model = "physical_perspective"
        self.native_root_prior = 'bbox_geometry'
        self.net = nn.Sequential(nn.Linear(9, 64), nn.ReLU(), nn.Linear(64, 3))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(
        self,
        joints: torch.Tensor,
        weak_joints_2d: torch.Tensor,
        weak_cam: torch.Tensor,
        weak_cam_t: torch.Tensor,
        virtual_focal_length: torch.Tensor,
        batch: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dtype = torch.float32
        joints = joints.to(dtype=dtype)
        weak_joints_2d = weak_joints_2d.to(dtype=dtype)
        weak_cam = weak_cam.to(dtype=dtype)
        weak_cam_t = weak_cam_t.to(dtype=dtype)
        virtual_focal_length = virtual_focal_length.to(dtype=dtype)
        K = batch["camera_K"].to(dtype=dtype)
        affine = batch["crop_transform"].to(dtype=dtype)
        width = batch["image_size"][:, 0].to(dtype=dtype)
        left = (batch["side"] == 0)
        crop_size = float(batch["img"].shape[-1])
        box_size = batch["crop_box_size"].to(dtype=dtype)

        # The original virtual camera gives a useful wrist ray and scale,
        # although its z value is not a physical depth in metres.
        wrist_crop = (weak_joints_2d[:, 0] + 0.5) * crop_size
        linear = affine[:, :, :2]
        offset = affine[:, :, 2]
        wrist_source = torch.linalg.solve(
            linear, (wrist_crop - offset).unsqueeze(-1)
        ).squeeze(-1)
        wrist_x = torch.where(left, width - wrist_source[:, 0] - 1, wrist_source[:, 0])
        wrist_raw = torch.stack((wrist_x, wrist_source[:, 1]), dim=-1)
        focal_crop = K[:, 0, 0] * crop_size / box_size
        depth_base = (
            weak_cam_t[:, 2] * focal_crop / virtual_focal_length[:, 0]
        ).clamp(0.05, 3.0)
        native_kb = self.camera_model == 'physical_kb'
        geometric_prior = self.camera_model in ('physical_kb', 'kb_bbox_pinhole') and self.native_root_prior == 'bbox_geometry'
        if geometric_prior:
            wrist_prior = bounded_wrist_pixel(wrist_raw, batch['bbox_xyxy'].float())
            depth_base = native_box_depth_prior(joints, wrist_prior, depth_base, batch, kb=native_kb)

        features = torch.stack((
            weak_cam[:, 0], weak_cam[:, 1], weak_cam[:, 2],
            weak_joints_2d[:, 0, 0], weak_joints_2d[:, 0, 1],
            torch.log(depth_base),
            box_size / 512.0,
            K[:, 0, 0] / 1000.0,
            left.to(dtype=dtype),
        ), dim=-1)
        delta = self.net(features).float()
        # Corrections to the wrist ray are measured in crop pixels.  The
        # bounded depth correction keeps initial and early training stable.
        wrist_raw = wrist_raw + delta[:, :2] * box_size[:, None]
        if geometric_prior:
            wrist_raw = bounded_wrist_pixel(wrist_raw, batch['bbox_xyxy'].float())
        depth = depth_base * torch.exp(delta[:, 2].clamp(-2.5, 2.5))
        if self.camera_model == "physical_kb":
            ray = pixel_to_z_ray(wrist_raw, K, batch['camera_distortion'].float())
            root = ray * depth[:, None]
            return root, physical_points_to_crop(joints, root, joints[:, :1], batch)
        root = torch.stack((
            (wrist_raw[:, 0] - K[:, 0, 2]) / K[:, 0, 0] * depth,
            (wrist_raw[:, 1] - K[:, 1, 2]) / K[:, 1, 1] * depth,
            depth,
        ), dim=-1)

        return root, project_rectified_joints(
            joints, root, K, affine, width, left, crop_size
        )


class MarkerOffsetHead(nn.Module):
    """Predict marker positions relative to the wrist from the MANO hand pose."""

    def __init__(self) -> None:
        super().__init__()
        self.root_bias = nn.Parameter(torch.zeros(3))
        self.net = nn.Sequential(nn.Linear(66, 128), nn.ReLU(), nn.Linear(128, 63))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, joints: torch.Tensor, weak_cam: torch.Tensor) -> torch.Tensor:
        relative = joints.float() - joints[:, :1].float()
        features = torch.cat((relative.flatten(1), weak_cam.float()), dim=-1)
        offset = self.net(features).float().reshape(-1, 21, 3)
        return relative + offset - offset[:, :1]
