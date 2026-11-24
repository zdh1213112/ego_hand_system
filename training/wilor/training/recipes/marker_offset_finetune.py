"""Learn glove marker offsets without changing the calibrated camera head."""

from __future__ import annotations

import torch

from .marker_camera_finetune import MarkerCameraFineTuneRecipe


class MarkerOffsetFineTuneRecipe(MarkerCameraFineTuneRecipe):
    name = "marker_offset_finetune"

    def configure_model(self, model: torch.nn.Module) -> None:
        if not model.physical_camera_enabled or not model.marker_head_enabled:
            raise ValueError("physical and marker heads must be enabled")
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in model.marker_offset_head.parameters():
            parameter.requires_grad = True

    def set_mode(self, model: torch.nn.Module, training: bool) -> None:
        model.eval()
        model.marker_offset_head.train(training)

    def build_optimizer(self, model: torch.nn.Module) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            model.marker_offset_head.parameters(),
            lr=self.config.refiner_lr,
            weight_decay=self.config.weight_decay,
        )
