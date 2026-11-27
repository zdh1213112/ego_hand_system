"""OpenTouch glove-domain adaptation through LoRA or the full ViT encoder."""

from __future__ import annotations

from typing import Any, Mapping

import torch

from wilor.models.lora import (
    inject_vit_lora,
    iter_lora_modules,
    lora_parameters,
    set_lora_trainable,
)
from wilor.utils.hand_geometry import palm_scale_normalize

from .base import TrainingRecipe


class DomainAdaptRecipe(TrainingRecipe):
    name = "domain_adapt"
    metric_name = "local_mpjpe_normalized"
    mpjpe_name = metric_name

    @staticmethod
    def _coarse_heads(model: torch.nn.Module) -> tuple[torch.nn.Module, ...]:
        return (
            model.backbone.decpose,
            model.backbone.decshape,
            model.backbone.deccam,
        )

    def _full_vit_parameters(
        self, model: torch.nn.Module
    ) -> list[torch.nn.Parameter]:
        head_parameters = {
            id(parameter)
            for module in self._coarse_heads(model)
            for parameter in module.parameters()
        }
        return [
            parameter for parameter in model.backbone.parameters()
            if id(parameter) not in head_parameters
        ]

    def configure_model(self, model: torch.nn.Module) -> None:
        for parameter in model.parameters():
            parameter.requires_grad = False
        if self.config.vit_tuning == "lora":
            if self.config.lora is None:
                raise ValueError("LoRA Stage 1 requires a LoRA configuration")
            inject_vit_lora(model.backbone, self.config.lora)
            set_lora_trainable(model.backbone, True)
            trainable = lora_parameters(model.backbone)
        elif self.config.vit_tuning == "full":
            trainable = self._full_vit_parameters(model)
            for parameter in trainable:
                parameter.requires_grad = True
        else:
            raise ValueError(
                f"unknown Stage 1 ViT tuning mode: {self.config.vit_tuning!r}"
            )
        model.backbone.use_checkpoint = True
        if not trainable:
            raise RuntimeError("domain_adapt produced no trainable ViT parameters")

    def set_mode(self, model: torch.nn.Module, training: bool) -> None:
        model.eval()
        if self.config.vit_tuning == "lora":
            # All original WiLoR modules stay deterministic and frozen. Only
            # LoRA dropout follows train/eval mode.
            for module in iter_lora_modules(model.backbone):
                module.train(training)
        else:
            # Full ViT tuning uses the encoder's stochastic depth/dropout while
            # the frozen coarse heads and all downstream modules stay in eval.
            model.backbone.train(training)
            for module in self._coarse_heads(model):
                module.eval()

    def forward(
        self,
        model: torch.nn.Module,
        batch: Mapping[str, Any],
        training: bool,
    ) -> dict[str, Any]:
        coarse = model.forward_coarse_step(batch, train=training)
        hand_pose = coarse["coarse_mano_params"]["hand_pose"]
        batch_size = hand_pose.shape[0]
        identity = torch.eye(
            3, device=hand_pose.device, dtype=hand_pose.dtype
        ).reshape(1, 1, 3, 3).expand(batch_size, 1, 3, 3)
        mean_betas = model.backbone.init_betas.to(
            device=hand_pose.device, dtype=hand_pose.dtype
        ).reshape(1, -1).expand(batch_size, -1)
        canonical = model.mano(
            global_orient=identity,
            hand_pose=hand_pose,
            betas=mean_betas,
            pose2rot=False,
        )
        coarse["coarse_local_keypoints_3d"] = palm_scale_normalize(
            canonical.joints.reshape(batch_size, -1, 3)
        )
        return coarse

    def build_optimizer(
        self, model: torch.nn.Module
    ) -> torch.optim.Optimizer:
        if self.config.vit_tuning == "lora":
            candidates = lora_parameters(model.backbone)
        else:
            candidates = self._full_vit_parameters(model)
        parameters = [
            parameter for parameter in candidates if parameter.requires_grad
        ]
        if not parameters:
            raise RuntimeError("domain_adapt has no trainable ViT parameters")
        return torch.optim.AdamW(
            [{"params": parameters, "lr": self.config.vit_lr}],
            weight_decay=self.config.weight_decay,
        )

    def metric(
        self,
        batch: Mapping[str, Any],
        output: Mapping[str, Any],
    ) -> tuple[str, torch.Tensor]:
        error = torch.linalg.vector_norm(
            output["coarse_local_keypoints_3d"] - batch["local_keypoints_3d"],
            dim=-1,
        ).mean()
        return self.metric_name, error
