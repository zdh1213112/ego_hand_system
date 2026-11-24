"""Reusable training engine for first-stage glove-domain fine-tuning."""

from __future__ import annotations

import json
import random
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from wilor.datasets.glove_dataset import GloveDataset, validate_manifest
from wilor.datasets.glove_validation import sha256_file
from wilor.models import WiLoR, load_wilor_config, resolve_mano_model_path
from wilor.models.adapter import (
    load_adapter_weights,
    resume_adapter_checkpoint,
    save_adapter_checkpoint,
    save_full_checkpoint,
)

from .evaluation import evaluate_glove_set
from .losses import (
    LOSS_NAMES,
    GloveLossComputer,
    GloveLossConfig,
    glove_loss_config_from_mapping,
)
from .reporting import write_json, write_training_curves, write_yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOSSES = {
    "keypoints_2d": 0.01,
    "keypoints_3d": 0.05,
    "coarse_keypoints_2d": 0.005,
    "coarse_keypoints_3d": 0.025,
    "global_orient": 0.0,
    "hand_pose": 0.0,
    "betas": 0.0,
}


@dataclass
class GloveTrainingConfig:
    """All inputs needed by the formal glove training workflow."""

    root: Path
    checkpoint: Path
    model_config: Path
    mano_path: Path
    output: Path
    experiment_name: str = "glove_training"
    config_source: Path | None = None
    losses: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_LOSSES))
    train_manifest: Path | None = None
    val_manifest: Path | None = None
    resume: Path | None = None
    init_adapter: Path | None = None
    max_steps: int = 200
    batch_size: int = 16
    num_workers: int = 0
    val_steps: int = 50
    save_steps: int = 50
    val_batches: int | None = None
    coarse_lr: float = 1e-4
    refiner_lr: float = 1e-5
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    seed: int = 42
    device: str = "auto"
    use_amp: bool = True
    max_train_samples: int | None = None
    final_visualization: bool = True
    evaluation_batch_size: int = 16
    evaluation_max_samples: int | None = None

    def validate(self) -> None:
        positive_values = {
            "max_steps": self.max_steps,
            "batch_size": self.batch_size,
            "val_steps": self.val_steps,
            "save_steps": self.save_steps,
            "evaluation_batch_size": self.evaluation_batch_size,
        }
        for name, value in positive_values.items():
            if value < 1:
                raise ValueError(f"{name} must be at least 1, got {value}")
        if self.num_workers < 0:
            raise ValueError("num_workers cannot be negative")
        if self.max_train_samples is not None and self.max_train_samples < 1:
            raise ValueError("max_train_samples must be at least 1")
        if self.val_batches is not None and self.val_batches < 1:
            raise ValueError("val_batches must be at least 1 or null")
        if self.evaluation_max_samples is not None and self.evaluation_max_samples < 1:
            raise ValueError("evaluation_max_samples must be at least 1 or null")
        for name in ("coarse_lr", "refiner_lr", "weight_decay", "grad_clip"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} cannot be negative")
        if self.resume is not None and self.init_adapter is not None:
            raise ValueError("resume and init_adapter are mutually exclusive")
        if set(self.losses) != set(LOSS_NAMES):
            raise ValueError(f"losses must contain exactly {list(LOSS_NAMES)}")


def _strict_section(
    config: Mapping[str, Any],
    name: str,
    allowed: set[str],
) -> Mapping[str, Any]:
    value = config.get(name)
    if not isinstance(value, Mapping):
        raise TypeError(f"training config section {name!r} must be a mapping")
    unknown = sorted(set(value).difference(allowed))
    if unknown:
        raise ValueError(f"unknown keys in training config {name}: {unknown}")
    return value


def _typed_value(
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


def _config_path(
    section: Mapping[str, Any],
    key: str,
    default: str | None,
    *,
    nullable: bool = False,
) -> Path | None:
    value = _typed_value(section, key, str, default, nullable=nullable)
    if value is None:
        return None
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def load_glove_training_config(path: str | Path) -> GloveTrainingConfig:
    """Load the complete, strict schema-v2 glove training YAML."""
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Training config does not exist: {source}")
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping) or raw.get("schema_version") != 2:
        raise ValueError(f"{source}: expected training schema_version 2")
    allowed_top = {
        "schema_version", "experiment", "paths", "training", "optimization",
        "checkpointing", "validation", "losses",
    }
    unknown_top = sorted(set(raw).difference(allowed_top))
    if unknown_top:
        raise ValueError(f"unknown top-level training config keys: {unknown_top}")

    experiment = _strict_section(raw, "experiment", {"name"})
    paths = _strict_section(raw, "paths", {
        "dataset_root", "checkpoint", "model_config", "mano_path", "output",
        "train_manifest", "val_manifest", "resume", "init_adapter",
    })
    training = _strict_section(raw, "training", {
        "max_steps", "batch_size", "num_workers", "seed", "device", "amp",
        "max_train_samples",
    })
    optimization = _strict_section(raw, "optimization", {
        "coarse_lr", "refiner_lr", "weight_decay", "grad_clip",
    })
    checkpointing = _strict_section(raw, "checkpointing", {"save_steps"})
    validation = _strict_section(raw, "validation", {
        "val_steps", "max_batches", "final_visualization", "batch_size",
        "max_samples",
    })
    losses = raw.get("losses")
    if not isinstance(losses, Mapping):
        raise TypeError("training config section 'losses' must be a mapping")

    config = GloveTrainingConfig(
        root=_config_path(paths, "dataset_root", "datasets/dataset"),
        checkpoint=_config_path(
            paths, "checkpoint", "pretrained_models/wilor_final.ckpt"
        ),
        model_config=_config_path(
            paths, "model_config", "pretrained_models/model_config.yaml"
        ),
        mano_path=_config_path(paths, "mano_path", "mano_data/mano_v1_2"),
        output=_config_path(paths, "output", "output/glove_training"),
        experiment_name=_typed_value(
            experiment, "name", str, "glove_training"
        ),
        config_source=source,
        losses={str(name): value for name, value in losses.items()},
        train_manifest=_config_path(
            paths, "train_manifest", None, nullable=True
        ),
        val_manifest=_config_path(paths, "val_manifest", None, nullable=True),
        resume=_config_path(paths, "resume", None, nullable=True),
        init_adapter=_config_path(paths, "init_adapter", None, nullable=True),
        max_steps=_typed_value(training, "max_steps", int, 200),
        batch_size=_typed_value(training, "batch_size", int, 16),
        num_workers=_typed_value(training, "num_workers", int, 0),
        seed=_typed_value(training, "seed", int, 42),
        device=_typed_value(training, "device", str, "auto"),
        use_amp=_typed_value(training, "amp", bool, True),
        max_train_samples=_typed_value(
            training, "max_train_samples", int, None, nullable=True
        ),
        coarse_lr=_typed_value(optimization, "coarse_lr", float, 1e-4),
        refiner_lr=_typed_value(optimization, "refiner_lr", float, 1e-5),
        weight_decay=_typed_value(optimization, "weight_decay", float, 1e-4),
        grad_clip=_typed_value(optimization, "grad_clip", float, 1.0),
        save_steps=_typed_value(checkpointing, "save_steps", int, 50),
        val_steps=_typed_value(validation, "val_steps", int, 50),
        val_batches=_typed_value(
            validation, "max_batches", int, None, nullable=True
        ),
        final_visualization=_typed_value(
            validation, "final_visualization", bool, True
        ),
        evaluation_batch_size=_typed_value(
            validation, "batch_size", int, 16
        ),
        evaluation_max_samples=_typed_value(
            validation, "max_samples", int, None, nullable=True
        ),
    )
    config.validate()
    return config


def resolved_training_config(config: GloveTrainingConfig) -> dict[str, Any]:
    """Return the normalized experiment config persisted with each run."""
    path_value = lambda value: None if value is None else str(Path(value).resolve())
    return {
        "schema_version": 2,
        "experiment": {"name": config.experiment_name},
        "paths": {
            "dataset_root": path_value(config.root),
            "checkpoint": path_value(config.checkpoint),
            "model_config": path_value(config.model_config),
            "mano_path": path_value(config.mano_path),
            "output": path_value(config.output),
            "train_manifest": path_value(config.train_manifest),
            "val_manifest": path_value(config.val_manifest),
            "resume": path_value(config.resume),
            "init_adapter": path_value(config.init_adapter),
        },
        "training": {
            "max_steps": config.max_steps,
            "batch_size": config.batch_size,
            "num_workers": config.num_workers,
            "seed": config.seed,
            "device": config.device,
            "amp": config.use_amp,
            "max_train_samples": config.max_train_samples,
        },
        "optimization": {
            "coarse_lr": config.coarse_lr,
            "refiner_lr": config.refiner_lr,
            "weight_decay": config.weight_decay,
            "grad_clip": config.grad_clip,
        },
        "checkpointing": {"save_steps": config.save_steps},
        "validation": {
            "val_steps": config.val_steps,
            "max_batches": config.val_batches,
            "final_visualization": config.final_visualization,
            "batch_size": config.evaluation_batch_size,
            "max_samples": config.evaluation_max_samples,
        },
        "losses": dict(config.losses),
    }


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def configure_glove_finetuning(model: WiLoR) -> None:
    """Freeze WiLoR except for its coarse heads and RefineNet adapter."""
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


def set_finetune_mode(model: WiLoR, training: bool) -> None:
    """Set modes without enabling stochastic behavior in the frozen ViT."""
    model.train(training)
    model.backbone.eval()
    model.backbone.decpose.train(training)
    model.backbone.decshape.train(training)
    model.backbone.deccam.train(training)
    model.refine_net.train(training)


def trainable_parameters(model: WiLoR) -> list[torch.nn.Parameter]:
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def build_optimizer(
    model: WiLoR,
    coarse_lr: float,
    refiner_lr: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    coarse_parameters = [
        parameter
        for module in (
            model.backbone.decpose,
            model.backbone.decshape,
            model.backbone.deccam,
        )
        for parameter in module.parameters()
        if parameter.requires_grad
    ]
    refiner_parameters = [
        parameter
        for parameter in model.refine_net.parameters()
        if parameter.requires_grad
    ]
    if not coarse_parameters or not refiner_parameters:
        raise RuntimeError("The coarse heads and RefineNet must both be trainable")
    return torch.optim.AdamW(
        [
            {"params": coarse_parameters, "lr": coarse_lr},
            {"params": refiner_parameters, "lr": refiner_lr},
        ],
        weight_decay=weight_decay,
    )


def load_glove_model(
    config: GloveTrainingConfig,
    device: torch.device,
) -> WiLoR:
    model_cfg = load_wilor_config(
        config.model_config,
        mano_path=config.mano_path,
    )
    model_cfg.defrost()
    model_cfg.LOSS_WEIGHTS.ADVERSARIAL = 0.0
    model_cfg.TRAIN.GRAD_CLIP_VAL = config.grad_clip
    model_cfg.freeze()
    model = WiLoR.load_from_checkpoint(
        str(config.checkpoint),
        cfg=model_cfg,
        init_renderer=False,
        strict=False,
        map_location="cpu",
    ).to(device)
    configure_glove_finetuning(model)
    set_finetune_mode(model, training=True)
    return model


def mpjpe_mm(prediction: torch.Tensor, target: torch.Tensor) -> float:
    prediction = prediction - prediction[:, :1]
    target = target[:, :, :3] - target[:, :1, :3]
    return float(
        torch.linalg.vector_norm(
            prediction - target, dim=-1
        ).mean().detach().cpu() * 1000.0
    )


class GloveTrainer:
    """Own model/optimizer state and the train/validation state transitions."""

    def __init__(
        self,
        model: WiLoR,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        use_amp: bool,
        grad_clip: float,
        loss_computer: GloveLossComputer,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.use_amp = use_amp and device.type == "cuda"
        self.grad_clip = grad_clip
        self.loss_computer = loss_computer
        self.last_losses: dict[str, float] = {}
        self.last_batch_size = 0

    def _move_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        def move(value: Any) -> Any:
            if isinstance(value, torch.Tensor):
                return value.to(self.device, non_blocking=True)
            if isinstance(value, dict):
                return {key: move(item) for key, item in value.items()}
            if isinstance(value, list):
                return [move(item) for item in value]
            if isinstance(value, tuple):
                return tuple(move(item) for item in value)
            return value
        return move(batch)

    def run_batch(
        self,
        batch: dict[str, Any],
        training: bool,
    ) -> tuple[float, float]:
        self.last_batch_size = int(batch["img"].shape[0])
        device_batch = self._move_batch(batch)
        if training:
            self.optimizer.zero_grad(set_to_none=True)
        amp_context = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if self.use_amp else nullcontext()
        )
        with torch.set_grad_enabled(training), amp_context:
            output = self.model.forward_step(device_batch, train=training)
            loss, components = self.loss_computer(device_batch, output)
        if training:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                trainable_parameters(self.model),
                self.grad_clip,
                error_if_nonfinite=True,
            )
            self.optimizer.step()
        self.last_losses = {
            name: float(value.detach().cpu()) for name, value in components.items()
        }
        return (
            float(loss.detach().cpu()),
            mpjpe_mm(output["pred_keypoints_3d"], device_batch["keypoints_3d"]),
        )

    @torch.no_grad()
    def evaluate(
        self,
        loader: DataLoader,
        max_batches: int | None,
    ) -> tuple[float, float, int, dict[str, float]]:
        set_finetune_mode(self.model, training=False)
        total_samples = 0
        total_loss = 0.0
        total_error = 0.0
        total_components: dict[str, float] = {}
        batch_count = 0
        try:
            for index, batch in enumerate(loader):
                if max_batches is not None and index >= max_batches:
                    break
                loss, error = self.run_batch(batch, training=False)
                count = self.last_batch_size
                total_samples += count
                total_loss += loss * count
                total_error += error * count
                batch_count += 1
                for name, value in self.last_losses.items():
                    total_components[name] = (
                        total_components.get(name, 0.0) + value * count
                    )
        finally:
            set_finetune_mode(self.model, training=True)
        if total_samples == 0:
            raise RuntimeError("Validation loader produced no batches")
        return (
            float(total_loss / total_samples),
            float(total_error / total_samples),
            batch_count,
            {
                name: float(value / total_samples)
                for name, value in total_components.items()
            },
        )


def _resolved_manifests(
    config: GloveTrainingConfig,
) -> tuple[Path, Path]:
    root = Path(config.root).expanduser().resolve()
    return (
        Path(config.train_manifest or root / "train.jsonl").expanduser().resolve(),
        Path(config.val_manifest or root / "val.jsonl").expanduser().resolve(),
    )


def _artifact(path: Path) -> dict[str, Any]:
    source = path.expanduser().resolve()
    return {
        "path": str(source),
        "size_bytes": source.stat().st_size,
        "sha256": sha256_file(source),
    }


def _training_metadata(
    config: GloveTrainingConfig,
    loss_config: GloveLossConfig,
    profile_path: Path,
    profile: dict[str, Any],
    manifests: tuple[Path, Path],
    device: torch.device,
    use_amp: bool,
) -> dict[str, Any]:
    semantic_config = {
        "schema_version": 2,
        "experiment_name": config.experiment_name,
        # The semantic fields below form the reproducibility hash.  Do not hash
        # the raw YAML because setting paths.resume would make exact resume
        # reject the checkpoint it is supposed to load.
        "training_config_schema_version": 2,
        "base_checkpoint": _artifact(config.checkpoint),
        "model_config": _artifact(config.model_config),
        "dataset_profile": _artifact(profile_path),
        "manifests": {
            name: _artifact(path)
            for name, path in zip(("train", "val"), manifests)
            if path.is_file()
        },
        "mano_assets": profile.get("mano_assets", {}),
        "losses": {
            "requested": loss_config.requested,
            "effective": loss_config.effective,
            "reduction": "batch_mean",
        },
        "keypoints_2d_visibility": "inside_source_image_and_crop",
        "optimizer": {
            "coarse_lr": config.coarse_lr,
            "refiner_lr": config.refiner_lr,
            "weight_decay": config.weight_decay,
            "grad_clip": config.grad_clip,
        },
        "data_and_runtime": {
            "batch_size": config.batch_size,
            "num_workers": config.num_workers,
            "max_train_samples": config.max_train_samples,
            "seed": config.seed,
            "device": str(device),
            "amp": use_amp,
            "val_steps": config.val_steps,
            "save_steps": config.save_steps,
            "val_batches": config.val_batches,
            "final_visualization": config.final_visualization,
            "evaluation_batch_size": config.evaluation_batch_size,
            "evaluation_max_samples": config.evaluation_max_samples,
        },
    }
    canonical = json.dumps(
        semantic_config, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    import hashlib
    return {
        "signature": hashlib.sha256(canonical).hexdigest(),
        "config": semantic_config,
    }


def _validate_resume_metadata(
    checkpoint_payload: dict[str, Any],
    expected: dict[str, Any],
    path: Path,
) -> dict[str, Any]:
    actual = checkpoint_payload.get("training_metadata")
    if not isinstance(actual, dict) or not actual.get("signature"):
        raise ValueError(
            f"{path}: checkpoint predates exact training signatures; "
            "use --init-adapter to load its weights into a new stage"
        )
    if actual["signature"] != expected["signature"]:
        raise ValueError(
            f"{path}: resume configuration differs from the checkpoint "
            f"({actual['signature']} != {expected['signature']}); use "
            "--init-adapter for a new loss/data/configuration stage"
        )
    return actual


def run_glove_training(config: GloveTrainingConfig) -> dict[str, Any]:
    """Execute the YAML-configured train/validation workflow."""
    config.validate()
    config.root = Path(config.root).expanduser().resolve()
    config.checkpoint = Path(config.checkpoint).expanduser().resolve()
    config.model_config = Path(config.model_config).expanduser().resolve()
    config.mano_path = Path(config.mano_path).expanduser().resolve()
    config.output = Path(config.output).expanduser().resolve()
    if config.resume is not None:
        config.resume = Path(config.resume).expanduser().resolve()
    if config.init_adapter is not None:
        config.init_adapter = Path(config.init_adapter).expanduser().resolve()
    if not config.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {config.checkpoint}")
    if not config.model_config.is_file():
        raise FileNotFoundError(f"Model config does not exist: {config.model_config}")

    train_manifest, val_manifest = _resolved_manifests(config)
    config.train_manifest = train_manifest
    config.val_manifest = val_manifest
    profile_path = config.root / "dataset_profile.json"
    if not profile_path.is_file():
        raise FileNotFoundError(
            f"Missing {profile_path}; run prepare_glove_dataset.py before training"
        )
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    if not profile.get("geometry_validation", {}).get("valid", False):
        raise ValueError("dataset_profile.json reports invalid geometry")
    training_mano_dir = resolve_mano_model_path(config.mano_path)
    training_mano_right = (
        training_mano_dir if training_mano_dir.is_file()
        else training_mano_dir / "MANO_RIGHT.pkl"
    )
    expected_right_hash = (
        profile.get("mano_assets", {}).get("MANO_RIGHT.pkl", {}).get("sha256")
    )
    actual_right_hash = sha256_file(training_mano_right)
    if not expected_right_hash or actual_right_hash != expected_right_hash:
        raise ValueError(
            "Training MANO_RIGHT.pkl does not match dataset_profile.json; "
            "rerun prepare_glove_dataset.py with the intended MANO assets"
        )
    loss_source = config.config_source or REPO_ROOT / "configs/glove/inline.yaml"
    loss_config = glove_loss_config_from_mapping(
        config.losses,
        profile,
        loss_source,
    )
    validation = {
        "train": validate_manifest(
            train_manifest,
            max_samples=config.max_train_samples,
        ),
        "val": validate_manifest(val_manifest),
    }
    invalid = {
        name: report for name, report in validation.items()
        if not report["valid"]
    }
    if invalid:
        raise RuntimeError(json.dumps(invalid, indent=2))

    seed_everything(config.seed)
    device = resolve_device(config.device)
    use_amp = config.use_amp and device.type == "cuda"
    config.output.mkdir(parents=True, exist_ok=True)
    resolved_config_path = config.output / "resolved_config.yaml"
    write_yaml(resolved_config_path, resolved_training_config(config))
    checkpoint_metadata = _training_metadata(
        config,
        loss_config,
        profile_path,
        profile,
        (train_manifest, val_manifest),
        device,
        use_amp,
    )

    train_set = GloveDataset(
        train_manifest,
        max_samples=config.max_train_samples,
    )
    val_set = GloveDataset(val_manifest)
    evaluation_set = (
        GloveDataset(val_manifest, max_samples=config.evaluation_max_samples)
        if config.final_visualization else None
    )
    loader_options = {
        "num_workers": config.num_workers,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(
        train_set,
        batch_size=config.batch_size,
        shuffle=True,
        **loader_options,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=config.batch_size,
        shuffle=False,
        **loader_options,
    )
    evaluation_loader = (
        DataLoader(
            evaluation_set,
            batch_size=config.evaluation_batch_size,
            shuffle=False,
            **loader_options,
        )
        if evaluation_set is not None else None
    )

    model = load_glove_model(config, device)
    initialized_from = None
    if config.init_adapter is not None:
        init_path = Path(config.init_adapter).expanduser().resolve()
        load_adapter_weights(init_path, model)
        initialized_from = _artifact(init_path)
        checkpoint_metadata["initialized_from_adapter"] = initialized_from
    optimizer = build_optimizer(
        model,
        config.coarse_lr,
        config.refiner_lr,
        config.weight_decay,
    )
    trainer = GloveTrainer(
        model,
        optimizer,
        device,
        use_amp,
        config.grad_clip,
        GloveLossComputer(model, loss_config),
    )
    start_step = 0
    best_val_loss = float("inf")
    resume_payload: dict[str, Any] | None = None
    if config.resume is not None:
        resume_path = Path(config.resume).expanduser().resolve()
        resume_payload = resume_adapter_checkpoint(resume_path, model, optimizer)
        checkpoint_metadata = _validate_resume_metadata(
            resume_payload, checkpoint_metadata, resume_path
        )
        initialized_from = checkpoint_metadata.get("initialized_from_adapter")
        start_step = int(resume_payload.get("step", 0))
        if resume_payload.get("val_loss") is not None:
            best_val_loss = float(resume_payload["val_loss"])
    if start_step > config.max_steps:
        raise ValueError(
            f"Resume step {start_step} exceeds max_steps {config.max_steps}"
        )

    print(
        f"device={device} amp={use_amp} train={len(train_set)} "
        f"val={len(val_set)} start_step={start_step}"
    )
    history: list[dict[str, Any]] = []
    loader_iterator = iter(train_loader)
    start_time = time.time()
    best_adapter_path = config.output / "best_adapter.pt"
    last_adapter_path = config.output / "last_adapter.pt"
    report_path = config.output / "train_report.json"
    curves_path = config.output / "loss_curves.png"

    def build_report(status: str, current_step: int) -> dict[str, Any]:
        return {
            "status": status,
            "experiment_name": config.experiment_name,
            "device": str(device),
            "amp": use_amp,
            "trainable_parameters": sum(
                parameter.numel() for parameter in trainable_parameters(model)
            ),
            "train_samples": len(train_set),
            "val_samples": len(val_set),
            "start_step": start_step,
            "current_step": current_step,
            "final_step": config.max_steps if status == "passed" else None,
            "best_val_loss": best_val_loss,
            "loss_config": loss_config.as_dict(),
            "loss_reduction": "batch_mean",
            "keypoints_2d_visibility": "inside_source_image_and_crop",
            "dataset_profile": str(profile_path),
            "resolved_config": str(resolved_config_path),
            "training_signature": checkpoint_metadata["signature"],
            "initialized_from_adapter": initialized_from,
            "elapsed_seconds": time.time() - start_time,
            "train_manifest": str(train_manifest),
            "val_manifest": str(val_manifest),
            "history": history,
        }

    for step in range(start_step + 1, config.max_steps + 1):
        try:
            batch = next(loader_iterator)
        except StopIteration:
            loader_iterator = iter(train_loader)
            batch = next(loader_iterator)
        train_loss, train_mpjpe = trainer.run_batch(batch, training=True)
        record: dict[str, Any] = {
            "step": step,
            "train_loss": train_loss,
            "train_mpjpe_mm": train_mpjpe,
            "train_loss_components": dict(trainer.last_losses),
        }
        should_validate = (
            step == start_step + 1
            or step % config.val_steps == 0
            or step == config.max_steps
        )
        if should_validate:
            val_loss, val_mpjpe, val_batches, val_components = trainer.evaluate(
                val_loader,
                config.val_batches,
            )
            record.update({
                "val_loss": val_loss,
                "val_mpjpe_mm": val_mpjpe,
                "val_batches": val_batches,
                "val_loss_components": val_components,
            })
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_adapter_checkpoint(
                    best_adapter_path,
                    model,
                    optimizer,
                    step,
                    config.checkpoint,
                    val_loss,
                    checkpoint_metadata,
                )
                save_full_checkpoint(config.output / "best_full.ckpt", model, step)
            print(json.dumps(record), flush=True)
        elif step % 10 == 0:
            print(json.dumps(record), flush=True)
        history.append(record)
        if should_validate:
            write_json(report_path, build_report("running", step))
            write_training_curves(history, curves_path)
        if step % config.save_steps == 0 or step == config.max_steps:
            save_adapter_checkpoint(
                last_adapter_path,
                model,
                optimizer,
                step,
                config.checkpoint,
                best_val_loss,
                checkpoint_metadata,
            )

    save_full_checkpoint(config.output / "last_full.ckpt", model, config.max_steps)
    report = build_report("passed", config.max_steps)
    if evaluation_loader is not None:
        evaluation_adapter: Path | None = None
        if best_adapter_path.is_file():
            evaluation_adapter = best_adapter_path
        elif config.resume is not None:
            evaluation_adapter = Path(config.resume).expanduser().resolve()
        elif last_adapter_path.is_file():
            evaluation_adapter = last_adapter_path
        if evaluation_adapter is None:
            raise RuntimeError("No adapter is available for final evaluation")
        load_adapter_weights(evaluation_adapter, model)
        set_finetune_mode(model, training=False)
        try:
            report["val_visualization"] = evaluate_glove_set(
                model,
                evaluation_loader,
                device,
                config.output / "val_visualization",
                use_amp,
            )
        finally:
            set_finetune_mode(model, training=True)

    write_json(report_path, report)
    write_training_curves(history, curves_path)
    return report
