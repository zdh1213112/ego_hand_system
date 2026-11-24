"""Train ViT LoRA, coarse heads and RefineNet through the full camera path."""

from __future__ import annotations

import torch

from wilor.models.lora import (
    inject_vit_lora,
    iter_lora_modules,
    lora_parameters,
    set_lora_trainable,
)

from .refiner_finetune import RefinerFineTuneRecipe


class JointLoRAFineTuneRecipe(RefinerFineTuneRecipe):
    name = "joint_lora_finetune"

    def configure_model(self, model: torch.nn.Module) -> None:
        super().configure_model(model)
        if self.config.lora is None:
            raise ValueError("joint_lora_finetune requires recipe.lora")
        existing = getattr(model.backbone, "_wilor_lora_config", None)
        if existing is not None and existing != self.config.lora:
            raise ValueError("Loaded LoRA configuration differs from recipe.lora")
        inject_vit_lora(model.backbone, self.config.lora)
        set_lora_trainable(model.backbone, True)
        model.backbone.use_checkpoint = True

    def set_mode(self, model: torch.nn.Module, training: bool) -> None:
        super().set_mode(model, training)
        # The original ViT stays in eval mode; only LoRA dropout follows the
        # training mode alongside the coarse heads and RefineNet.
        for module in iter_lora_modules(model.backbone):
            module.train(training)

    def build_optimizer(self, model: torch.nn.Module) -> torch.optim.Optimizer:
        lora = [p for p in lora_parameters(model.backbone) if p.requires_grad]
        coarse = [
            p
            for module in (
                model.backbone.decpose,
                model.backbone.decshape,
                model.backbone.deccam,
            )
            for p in module.parameters()
            if p.requires_grad
        ]
        refiner = [p for p in model.refine_net.parameters() if p.requires_grad]
        if not lora or not coarse or not refiner:
            raise RuntimeError(
                "joint_lora_finetune requires LoRA, coarse heads and RefineNet"
            )
        return torch.optim.AdamW(
            [
                {"params": lora, "lr": self.config.vit_lr},
                {"params": coarse, "lr": self.config.coarse_lr},
                {"params": refiner, "lr": self.config.refiner_lr},
            ] + ([{"params": list(model.physical_camera_head.parameters()),
                   "lr": self.config.camera_lr or self.config.refiner_lr}]
                 if getattr(model, 'physical_camera_enabled', False) else []),
            weight_decay=self.config.weight_decay,
        )
