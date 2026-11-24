"""Fit a metric camera head while preserving the trained hand pose model."""

from __future__ import annotations

from typing import Any, Mapping

import torch

from .base import TrainingRecipe


class PhysicalCameraFineTuneRecipe(TrainingRecipe):
    name = "physical_camera_finetune"
    metric_name = "crop_reprojection_px"

    def configure_model(self, model: torch.nn.Module) -> None:
        if not model.physical_camera_enabled:
            raise ValueError("physical camera head was not enabled")
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in model.physical_camera_head.parameters():
            parameter.requires_grad = True

    def set_mode(self, model: torch.nn.Module, training: bool) -> None:
        model.eval()
        model.physical_camera_head.train(training)

    def forward(
        self, model: torch.nn.Module, batch: Mapping[str, Any], training: bool
    ) -> dict[str, Any]:
        return model.forward_step(batch, train=training)

    def build_optimizer(self, model: torch.nn.Module) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            model.physical_camera_head.parameters(),
            lr=self.config.refiner_lr,
            weight_decay=self.config.weight_decay,
        )

    def metric(
        self, batch: Mapping[str, Any], output: Mapping[str, Any]
    ) -> tuple[str, torch.Tensor]:
        distance = torch.linalg.vector_norm(
            output["pred_keypoints_2d"] - batch["keypoints_2d"][..., :2], dim=-1
        ) * self.config.image_size
        visible = batch["keypoints_2d"][..., 2] > 0
        return self.metric_name, distance[visible].mean()
