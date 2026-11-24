"""Strict schema-v3 configuration for the unified glove trainer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from wilor.models.lora import LoRAConfig

from .losses import OPTIONAL_UNIFIED_LOSS_NAMES, UNIFIED_LOSS_NAMES


REPO_ROOT = Path(__file__).resolve().parents[2]


def _section(
    raw: Mapping[str, Any], name: str, allowed: set[str]
) -> Mapping[str, Any]:
    value = raw.get(name)
    if not isinstance(value, Mapping):
        raise TypeError(f"training config section {name!r} must be a mapping")
    unknown = sorted(set(value).difference(allowed))
    if unknown:
        raise ValueError(f"unknown keys in training config {name}: {unknown}")
    return value


def _value(
    section: Mapping[str, Any],
    key: str,
    expected: type,
    default: Any,
    *,
    nullable: bool = False,
) -> Any:
    value = section.get(key, default)
    if value is None and nullable:
        return None
    if expected is int:
        valid = isinstance(value, int) and not isinstance(value, bool)
    elif expected is float:
        valid = isinstance(value, (int, float)) and not isinstance(value, bool)
    else:
        valid = isinstance(value, expected)
    if not valid:
        suffix = " or null" if nullable else ""
        raise TypeError(f"{key} must be {expected.__name__}{suffix}, got {value!r}")
    return float(value) if expected is float else value


def _path(
    section: Mapping[str, Any],
    key: str,
    default: str | None,
    *,
    nullable: bool = False,
) -> Path | None:
    value = _value(section, key, str, default, nullable=nullable)
    if value is None:
        return None
    result = Path(value).expanduser()
    return result.resolve() if result.is_absolute() else (REPO_ROOT / result).resolve()


@dataclass
class TrainingConfig:
    source: Path
    experiment_name: str
    dataset_root: Path
    checkpoint: Path
    model_config: Path
    mano_path: Path
    output: Path
    train_manifest: Path | None
    val_manifest: Path | None
    resume: Path | None
    init_weights: Path | None
    data_type: str
    image_size: int
    rescale_factor: float
    npz_cache_size: int
    marker_inputs: dict[str, Path]
    recipe_type: str
    camera_model: str
    vit_tuning: str | None
    lora: LoRAConfig | None
    max_steps: int
    batch_size: int
    num_workers: int
    seed: int
    device: str
    use_amp: bool
    max_train_samples: int | None
    gradient_accumulation_steps: int
    vit_lr: float
    pose_head_lr: float
    coarse_lr: float
    refiner_lr: float
    weight_decay: float
    grad_clip: float
    save_steps: int
    val_steps: int
    val_batches: int | None
    evaluation_batch_size: int
    evaluation_max_samples: int | None
    final_visualization: bool
    losses: dict[str, float]
    camera_lr: float | None = None
    crop_jitter: float = 0.

    def validate(self) -> None:
        if not 0 <= self.crop_jitter < 1:
            raise ValueError('data.crop_jitter must be in [0, 1)')
        if self.losses.get('surface_markers_3d',0) and (self.data_type!='glove_camera' or self.camera_model!='physical_kb'):
            raise ValueError('NOKOV surface marker loss requires native glove camera data')
        if self.camera_lr is not None and not 0 < self.camera_lr < float('inf'):
            raise ValueError('camera_lr must be positive and finite')
        allowed_recipes = {
            "opentouch": {"domain_adapt"},
            "glove_camera": {"refiner_finetune", "joint_lora_finetune", "physical_camera_finetune", "marker_camera_finetune", "marker_offset_finetune"},
        }
        if self.data_type not in allowed_recipes:
            raise ValueError(f"unknown data.type: {self.data_type!r}")
        if self.recipe_type not in set().union(*allowed_recipes.values()):
            raise ValueError(f"unknown recipe.type: {self.recipe_type!r}")
        # Dataset and recipe remain separate config concepts, but invalid
        # supervision/forward combinations are rejected explicitly.
        if self.recipe_type not in allowed_recipes[self.data_type]:
            raise ValueError(
                f"data.type {self.data_type!r} cannot supply recipe "
                f"{self.recipe_type!r}"
            )
        if self.camera_model not in {"virtual", "physical_perspective", "physical_kb"}:
            raise ValueError(f"unknown recipe.camera_model: {self.camera_model!r}")
        if self.recipe_type in {"physical_camera_finetune", "marker_camera_finetune", "marker_offset_finetune"} and self.camera_model not in {"physical_perspective", "physical_kb"}:
            raise ValueError("physical camera recipes require camera_model: physical_perspective")
        if self.recipe_type in {"marker_camera_finetune", "marker_offset_finetune"} and not self.marker_inputs:
            raise ValueError("marker recipes require data.marker_inputs")
        if self.camera_model in {"physical_perspective", "physical_kb"} and self.data_type != "glove_camera":
            raise ValueError("physical_perspective requires glove_camera data")
        if self.recipe_type == "domain_adapt":
            if self.vit_tuning not in {"lora", "full"}:
                raise ValueError(
                    "domain_adapt requires recipe.vit_tuning to be 'lora' or 'full'"
                )
            if self.vit_tuning == "lora" and self.lora is None:
                raise ValueError("LoRA Stage 1 requires recipe.lora")
            if self.vit_tuning == "full" and self.lora is not None:
                raise ValueError("full-ViT Stage 1 requires recipe.lora: null")
        elif self.recipe_type == "joint_lora_finetune":
            if self.vit_tuning != "lora" or self.lora is None:
                raise ValueError(
                    "joint_lora_finetune requires recipe.vit_tuning: lora "
                    "and recipe.lora"
                )
            if self.vit_lr <= 0 or self.coarse_lr <= 0 or self.refiner_lr <= 0:
                raise ValueError(
                    "joint_lora_finetune requires positive vit_lr, "
                    "coarse_lr and refiner_lr"
                )
        elif self.vit_tuning is not None:
            raise ValueError(
                "recipe.vit_tuning is only valid for domain_adapt or "
                "joint_lora_finetune"
            )
        if self.recipe_type == "refiner_finetune" and self.lora is not None:
            raise ValueError("refiner_finetune requires recipe.lora: null")
        if self.lora is not None:
            self.lora.validate()
        if self.resume is not None and self.init_weights is not None:
            raise ValueError("resume and init_weights are mutually exclusive")
        if self.recipe_type == "domain_adapt" and self.final_visualization:
            raise ValueError("domain_adapt cannot use camera-based final_visualization")
        for name in (
            "max_steps", "batch_size", "val_steps", "save_steps",
            "evaluation_batch_size", "gradient_accumulation_steps",
            "image_size", "npz_cache_size",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be at least 1")
        if self.num_workers < 0:
            raise ValueError("num_workers cannot be negative")
        for name in ("max_train_samples", "val_batches", "evaluation_max_samples"):
            value = getattr(self, name)
            if value is not None and value < 1:
                raise ValueError(f"{name} must be at least 1 or null")
        for name in (
            "vit_lr", "pose_head_lr", "coarse_lr", "refiner_lr",
            "weight_decay", "grad_clip",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} cannot be negative")
        if not set(UNIFIED_LOSS_NAMES).difference(OPTIONAL_UNIFIED_LOSS_NAMES) <= set(self.losses) <= set(UNIFIED_LOSS_NAMES):
            raise ValueError(
                f"losses must contain exactly {list(UNIFIED_LOSS_NAMES)}"
            )
        for name, weight in self.losses.items():
            if isinstance(weight, bool) or not isinstance(weight, (int, float)):
                raise TypeError(f"loss {name} must be numeric")
            if not 0 <= float(weight) < float("inf"):
                raise ValueError(f"loss {name} must be finite and non-negative")
        if self.pose_head_lr != 0:
            raise ValueError(
                "optimization.pose_head_lr is reserved and must remain zero; "
                "domain_adapt trains the visual encoder while keeping the "
                "coarse pose head frozen"
            )
        allowed = (
            {"local_keypoints_3d"}
            if self.recipe_type == "domain_adapt"
            else set(UNIFIED_LOSS_NAMES).difference({"local_keypoints_3d"})
        )
        if self.camera_model not in {"physical_perspective", "physical_kb"} and self.losses.get("physical_root_3d", 0) > 0:
            raise ValueError("physical_root_3d requires physical_perspective camera")
        marker_losses = ("marker_keypoints_2d", "marker_keypoints_3d", "marker_root_3d")
        if self.recipe_type not in {"marker_camera_finetune", "marker_offset_finetune"} and any(self.losses.get(name, 0) > 0 for name in marker_losses):
            raise ValueError("marker losses require a marker recipe")
        if self.recipe_type in {"marker_camera_finetune", "marker_offset_finetune"} and self.losses.get("marker_keypoints_2d", 0) <= 0:
            raise ValueError("marker recipes require marker_keypoints_2d loss")
        disallowed = [
            name for name, weight in self.losses.items()
            if float(weight) > 0 and name not in allowed
        ]
        if disallowed:
            raise ValueError(
                f"recipe {self.recipe_type} does not allow losses {disallowed}"
            )


def load_training_config(path: str | Path) -> TrainingConfig:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Training config does not exist: {source}")
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping) or raw.get("schema_version") != 3:
        raise ValueError(f"{source}: expected training schema_version 3")
    allowed_top = {
        "schema_version", "experiment", "paths", "data", "recipe",
        "training", "optimization", "checkpointing", "validation", "losses",
    }
    unknown = sorted(set(raw).difference(allowed_top))
    if unknown:
        raise ValueError(f"unknown top-level training config keys: {unknown}")
    experiment = _section(raw, "experiment", {"name"})
    paths = _section(raw, "paths", {
        "dataset_root", "checkpoint", "model_config", "mano_path", "output",
        "train_manifest", "val_manifest", "resume", "init_weights",
    })
    data = _section(raw, "data", {
        "type", "image_size", "rescale_factor", "npz_cache_size", "marker_inputs", "crop_jitter",
    })
    recipe = _section(raw, "recipe", {"type", "vit_tuning", "lora", "camera_model"})
    training = _section(raw, "training", {
        "max_steps", "batch_size", "num_workers", "seed", "device", "amp",
        "max_train_samples", "gradient_accumulation_steps",
    })
    optimization = _section(raw, "optimization", {
        "vit_lr", "pose_head_lr", "coarse_lr", "refiner_lr", "weight_decay", "camera_lr",
        "grad_clip",
    })
    checkpointing = _section(raw, "checkpointing", {"save_steps"})
    validation = _section(raw, "validation", {
        "val_steps", "max_batches", "batch_size", "max_samples",
        "final_visualization",
    })
    raw_losses = raw.get("losses")
    if not isinstance(raw_losses, dict):
        raise TypeError("training config section 'losses' must be a mapping")
    lora_raw = recipe.get("lora")
    if lora_raw is not None and not isinstance(lora_raw, dict):
        raise TypeError("recipe.lora must be a mapping or null")
    recipe_type = _value(recipe, "type", str, "refiner_finetune")
    marker_input_raw = data.get("marker_inputs", {})
    if not isinstance(marker_input_raw, Mapping):
        raise TypeError("data.marker_inputs must be a mapping")
    marker_inputs = {
        str(name): (Path(value).expanduser().resolve() if Path(value).is_absolute()
                    else (REPO_ROOT / value).resolve())
        for name, value in marker_input_raw.items()
    }
    default_vit_tuning = "lora" if recipe_type == "domain_adapt" else None
    config = TrainingConfig(
        source=source,
        experiment_name=_value(experiment, "name", str, "glove_training"),
        dataset_root=_path(paths, "dataset_root", "datasets/dataset"),
        checkpoint=_path(paths, "checkpoint", "pretrained_models/wilor_final.ckpt"),
        model_config=_path(
            paths, "model_config", "pretrained_models/model_config.yaml"
        ),
        mano_path=_path(paths, "mano_path", "mano_data/mano_v1_2"),
        output=_path(paths, "output", "output/glove_training"),
        train_manifest=_path(paths, "train_manifest", None, nullable=True),
        val_manifest=_path(paths, "val_manifest", None, nullable=True),
        resume=_path(paths, "resume", None, nullable=True),
        init_weights=_path(paths, "init_weights", None, nullable=True),
        data_type=_value(data, "type", str, "glove_camera"),
        image_size=_value(data, "image_size", int, 256),
        rescale_factor=_value(data, "rescale_factor", float, 2.0),
        npz_cache_size=_value(data, "npz_cache_size", int, 8),
        marker_inputs=marker_inputs,
        crop_jitter=_value(data, "crop_jitter", float, 0.),
        recipe_type=recipe_type,
        camera_model=_value(recipe, "camera_model", str, "virtual"),
        vit_tuning=_value(
            recipe, "vit_tuning", str, default_vit_tuning, nullable=True
        ),
        lora=None if lora_raw is None else LoRAConfig.from_dict(lora_raw),
        max_steps=_value(training, "max_steps", int, 200),
        batch_size=_value(training, "batch_size", int, 16),
        num_workers=_value(training, "num_workers", int, 0),
        seed=_value(training, "seed", int, 42),
        device=_value(training, "device", str, "auto"),
        use_amp=_value(training, "amp", bool, True),
        max_train_samples=_value(
            training, "max_train_samples", int, None, nullable=True
        ),
        gradient_accumulation_steps=_value(
            training, "gradient_accumulation_steps", int, 1
        ),
        vit_lr=_value(optimization, "vit_lr", float, 1e-5),
        pose_head_lr=_value(optimization, "pose_head_lr", float, 0.0),
        coarse_lr=_value(optimization, "coarse_lr", float, 1e-4),
        refiner_lr=_value(optimization, "refiner_lr", float, 1e-5),
        camera_lr=_value(optimization, 'camera_lr', float, None, nullable=True),
        weight_decay=_value(optimization, "weight_decay", float, 1e-4),
        grad_clip=_value(optimization, "grad_clip", float, 1.0),
        save_steps=_value(checkpointing, "save_steps", int, 50),
        val_steps=_value(validation, "val_steps", int, 50),
        val_batches=_value(
            validation, "max_batches", int, None, nullable=True
        ),
        evaluation_batch_size=_value(validation, "batch_size", int, 16),
        evaluation_max_samples=_value(
            validation, "max_samples", int, None, nullable=True
        ),
        final_visualization=_value(
            validation, "final_visualization", bool, False
        ),
        losses={str(name): value for name, value in raw_losses.items()},
    )
    config.validate()
    return config


def resolved_training_config(config: TrainingConfig) -> dict[str, Any]:
    path = lambda value: None if value is None else str(value.resolve())
    return {
        "schema_version": 3,
        "experiment": {"name": config.experiment_name},
        "paths": {
            "dataset_root": path(config.dataset_root),
            "checkpoint": path(config.checkpoint),
            "model_config": path(config.model_config),
            "mano_path": path(config.mano_path),
            "output": path(config.output),
            "train_manifest": path(config.train_manifest),
            "val_manifest": path(config.val_manifest),
            "resume": path(config.resume),
            "init_weights": path(config.init_weights),
        },
        "data": {
            "type": config.data_type,
            "image_size": config.image_size,
            "rescale_factor": config.rescale_factor,
            "npz_cache_size": config.npz_cache_size,
            "marker_inputs": {name: str(path) for name, path in config.marker_inputs.items()},
            **({"crop_jitter":config.crop_jitter} if config.crop_jitter else {}),
        },
        "recipe": {
            "type": config.recipe_type,
            "vit_tuning": config.vit_tuning,
            "camera_model": config.camera_model,
            "lora": None if config.lora is None else config.lora.to_dict(),
        },
        "training": {
            "max_steps": config.max_steps,
            "batch_size": config.batch_size,
            "num_workers": config.num_workers,
            "seed": config.seed,
            "device": config.device,
            "amp": config.use_amp,
            "max_train_samples": config.max_train_samples,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
        },
        "optimization": {
            "vit_lr": config.vit_lr,
            "pose_head_lr": config.pose_head_lr,
            "coarse_lr": config.coarse_lr,
            "refiner_lr": config.refiner_lr,
            "camera_lr": config.camera_lr,
            "weight_decay": config.weight_decay,
            "grad_clip": config.grad_clip,
        },
        "checkpointing": {"save_steps": config.save_steps},
        "validation": {
            "val_steps": config.val_steps,
            "max_batches": config.val_batches,
            "batch_size": config.evaluation_batch_size,
            "max_samples": config.evaluation_max_samples,
            "final_visualization": config.final_visualization,
        },
        "losses": dict(config.losses),
    }
