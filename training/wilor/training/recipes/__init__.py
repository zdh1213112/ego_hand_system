"""Training recipe registry."""

from __future__ import annotations

from ..config import TrainingConfig
from .base import TrainingRecipe
from .domain_adapt import DomainAdaptRecipe
from .joint_lora_finetune import JointLoRAFineTuneRecipe
from .refiner_finetune import RefinerFineTuneRecipe
from .physical_camera_finetune import PhysicalCameraFineTuneRecipe
from .marker_camera_finetune import MarkerCameraFineTuneRecipe
from .marker_offset_finetune import MarkerOffsetFineTuneRecipe


RECIPE_REGISTRY = {
    "domain_adapt": DomainAdaptRecipe,
    "joint_lora_finetune": JointLoRAFineTuneRecipe,
    "refiner_finetune": RefinerFineTuneRecipe,
    "physical_camera_finetune": PhysicalCameraFineTuneRecipe,
    "marker_camera_finetune": MarkerCameraFineTuneRecipe,
    "marker_offset_finetune": MarkerOffsetFineTuneRecipe,
}


def build_recipe(config: TrainingConfig) -> TrainingRecipe:
    try:
        recipe = RECIPE_REGISTRY[config.recipe_type]
    except KeyError as exc:
        raise ValueError(f"Unknown recipe: {config.recipe_type!r}") from exc
    return recipe(config)


__all__ = [
    "DomainAdaptRecipe",
    "JointLoRAFineTuneRecipe",
    "RefinerFineTuneRecipe",
    "TrainingRecipe",
    "build_recipe",
]
