"""Camera-aligned coarse-head and RefineNet fine-tuning recipe."""

from __future__ import annotations

from typing import Any, Mapping

import torch

from .base import TrainingRecipe


class RefinerFineTuneRecipe(TrainingRecipe):
    name = "refiner_finetune"
    metric_name = "mpjpe_mm"

    def configure_model(self, model: torch.nn.Module) -> None:
        for parameter in model.parameters():
            parameter.requires_grad = False
        for module in (
            model.backbone.decpose,
            model.backbone.decshape,
            model.backbone.deccam,
            model.refine_net,
        ):
            for parameter in module.parameters():
                parameter.requires_grad = True
        if getattr(model, 'physical_camera_enabled', False):
            for parameter in model.physical_camera_head.parameters():
                parameter.requires_grad = True

    def set_mode(self, model: torch.nn.Module, training: bool) -> None:
        model.train(training)
        model.backbone.eval()
        model.backbone.decpose.train(training)
        model.backbone.decshape.train(training)
        model.backbone.deccam.train(training)
        model.refine_net.train(training)
        if getattr(model, 'physical_camera_enabled', False):
            model.physical_camera_head.train(training)

    def forward(
        self,
        model: torch.nn.Module,
        batch: Mapping[str, Any],
        training: bool,
    ) -> dict[str, Any]:
        return model.forward_step(batch, train=training)

    def build_optimizer(
        self, model: torch.nn.Module
    ) -> torch.optim.Optimizer:
        coarse = [
            parameter
            for module in (
                model.backbone.decpose,
                model.backbone.decshape,
                model.backbone.deccam,
            )
            for parameter in module.parameters()
            if parameter.requires_grad
        ]
        refiner = [
            parameter for parameter in model.refine_net.parameters()
            if parameter.requires_grad
        ]
        if not coarse or not refiner:
            raise RuntimeError(
                "refiner_finetune requires coarse heads and RefineNet"
            )
        return torch.optim.AdamW(
            [
                {"params": coarse, "lr": self.config.coarse_lr},
                {"params": refiner, "lr": self.config.refiner_lr},
            ] + ([{"params": list(model.physical_camera_head.parameters()),
                   "lr": self.config.camera_lr or self.config.refiner_lr}]
                 if getattr(model, 'physical_camera_enabled', False) else []),
            weight_decay=self.config.weight_decay,
        )

    def metric(
        self,
        batch: Mapping[str, Any],
        output: Mapping[str, Any],
    ) -> tuple[str, torch.Tensor]:
        prediction = output["pred_keypoints_3d"]
        target = batch["keypoints_3d"][..., :3]
        prediction = prediction - prediction[:, :1]
        target = target - target[:, :1]
        error = torch.linalg.vector_norm(prediction - target, dim=-1).mean() * 1000
        return "mpjpe_mm", error
