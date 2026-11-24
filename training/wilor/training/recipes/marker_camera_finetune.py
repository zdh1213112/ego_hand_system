"""Train physical marker offsets and metric camera translation from Nokov."""

from __future__ import annotations

from typing import Any, Mapping

import torch

from .base import TrainingRecipe


class MarkerCameraFineTuneRecipe(TrainingRecipe):
    name = "marker_camera_finetune"
    metric_name = "marker_crop_reprojection_px"

    def configure_model(self, model: torch.nn.Module) -> None:
        if not model.physical_camera_enabled or not model.marker_head_enabled:
            raise ValueError("physical and marker heads must be enabled")
        for parameter in model.parameters():
            parameter.requires_grad = False
        for module in (model.physical_camera_head, model.marker_offset_head):
            for parameter in module.parameters():
                parameter.requires_grad = True

    def set_mode(self, model: torch.nn.Module, training: bool) -> None:
        model.eval()
        model.physical_camera_head.train(training)
        model.marker_offset_head.train(training)

    def forward(
        self, model: torch.nn.Module, batch: Mapping[str, Any], training: bool
    ) -> dict[str, Any]:
        return model.forward_step(batch, train=training)

    def build_optimizer(self, model: torch.nn.Module) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            [
                {"params": model.physical_camera_head.parameters(), "lr": self.config.refiner_lr},
                {"params": model.marker_offset_head.parameters(), "lr": self.config.refiner_lr},
            ],
            weight_decay=self.config.weight_decay,
        )

    def metric(
        self, batch: Mapping[str, Any], output: Mapping[str, Any]
    ) -> tuple[str, torch.Tensor]:
        distance = torch.linalg.vector_norm(
            output["pred_marker_keypoints_2d"] - batch["marker_keypoints_2d"][..., :2],
            dim=-1,
        ) * self.config.image_size
        visible = batch["marker_keypoints_2d"][..., 2] > 0
        return self.metric_name, distance[visible].mean()
