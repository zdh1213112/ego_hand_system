"""Dataset-agnostic optimization engine for both glove training recipes."""

from __future__ import annotations

import hashlib
import json
import math
import random
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from wilor.datasets.glove_dataset import GloveDataset, SyncFrameBatchSampler, validate_manifest
from wilor.datasets.glove_validation import sha256_file
from wilor.datasets.opentouch_dataset import OpenTouchDataset
from wilor.models import WiLoR, load_wilor_config, resolve_mano_model_path
from wilor.models.adapter import (
    load_trainable_weights,
    restore_best_mpjpe,
    resume_trainable_checkpoint,
    save_best_mpjpe,
    save_trainable_checkpoint,
    save_full_checkpoint,
)

from .config import TrainingConfig, resolved_training_config
from .evaluation import evaluate_glove_set
from .losses import FINGERTIPS, V5_LOSS_SOURCES, UnifiedLossComputer, unified_loss_config_from_mapping
from .recipes import TrainingRecipe, build_recipe
from .reporting import write_json, write_training_curves, write_yaml


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


def _move(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: _move(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move(item, device) for item in value)
    return value


def _manifests(config: TrainingConfig) -> tuple[Path, Path]:
    return (
        Path(config.train_manifest or config.dataset_root / "train.jsonl").resolve(),
        Path(config.val_manifest or config.dataset_root / "val.jsonl").resolve(),
    )


def build_datasets(
    config: TrainingConfig,
    train_manifest: Path,
    val_manifest: Path,
) -> tuple[Dataset, Dataset, dict[str, Any]]:
    if config.pinhole_crop:
        for root in {train_manifest.parent, val_manifest.parent}:
            summary = json.loads((root / "summary.json").read_text())
            if (summary.get("input_view") != "local_pinhole"
                    or summary.get("pinhole_crop", "local_pinhole") != config.pinhole_crop
                    or (config.pinhole_crop == "kb_bbox_pinhole"
                        and summary.get("rescale_factor") != config.rescale_factor)):
                raise ValueError(f"{root}: expected a matching {config.pinhole_crop} local view export")
    profile_path = config.dataset_root / "dataset_profile.json"
    if not profile_path.is_file():
        raise FileNotFoundError(
            f"Missing {profile_path}; run prepare_glove_dataset.py first"
        )
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    if config.data_type == "opentouch":
        kwargs = {
            "image_size": config.image_size,
            "rescale_factor": config.rescale_factor,
            "npz_cache_size": config.npz_cache_size,
        }
        train = OpenTouchDataset(
            train_manifest, max_samples=config.max_train_samples, **kwargs
        )
        val = OpenTouchDataset(val_manifest, **kwargs)
    elif config.data_type == "glove_camera":
        if not profile.get("geometry_validation", {}).get("valid", False):
            raise ValueError("dataset_profile.json reports invalid camera geometry")
        validation = {
            "train": validate_manifest(
                train_manifest, max_samples=config.max_train_samples
            ),
            "val": validate_manifest(val_manifest),
        }
        invalid = {
            name: result for name, result in validation.items()
            if not result["valid"]
        }
        if invalid:
            raise RuntimeError(json.dumps(invalid, indent=2))
        mano_dir = resolve_mano_model_path(config.mano_path)
        mano_right = mano_dir if mano_dir.is_file() else mano_dir / "MANO_RIGHT.pkl"
        expected_hash = profile.get("mano_assets", {}).get(
            "MANO_RIGHT.pkl", {}
        ).get("sha256")
        if not expected_hash or sha256_file(mano_right) != expected_hash:
            raise ValueError(
                "Training MANO_RIGHT.pkl does not match dataset_profile.json"
            )
        train = GloveDataset(
            train_manifest,
            image_size=config.image_size,
            rescale_factor=config.rescale_factor,
            max_samples=config.max_train_samples,
            marker_inputs=config.marker_inputs,
            use_surface_markers=config.losses.get('surface_markers_3d',0)>0,
            crop_jitter=config.crop_jitter,
            color_jitter=config.color_jitter,
            blur_prob=config.blur_prob,
            erasing_prob=config.erasing_prob,
            detector_cache=config.detector_cache,
            offset_samples=config.offset_samples,
            detector_probability=config.detector_probability,
            camera_model=config.camera_model,
        )
        val = GloveDataset(
            val_manifest,
            image_size=config.image_size,
            rescale_factor=config.rescale_factor,
            marker_inputs=config.marker_inputs,
            use_surface_markers=config.losses.get('surface_markers_3d',0)>0,
            camera_model=config.camera_model,
        )
    else:  # guarded by config validation
        raise ValueError(f"Unknown dataset type: {config.data_type}")
    return train, val, profile


def _load_base_model(config: TrainingConfig) -> WiLoR:
    model_cfg = load_wilor_config(config.model_config, mano_path=config.mano_path)
    model_cfg.defrost()
    model_cfg.LOSS_WEIGHTS.ADVERSARIAL = 0.0
    model_cfg.TRAIN.GRAD_CLIP_VAL = config.grad_clip
    model_cfg.freeze()
    return WiLoR.load_from_checkpoint(
        str(config.checkpoint),
        cfg=model_cfg,
        init_renderer=False,
        strict=False,
        map_location="cpu",
    )


def _artifact(path: Path) -> dict[str, Any]:
    source = path.expanduser().resolve()
    return {
        "path": str(source),
        "size_bytes": source.stat().st_size,
        "sha256": sha256_file(source),
    }


def _training_metadata(
    config: TrainingConfig,
    profile_path: Path,
    manifests: tuple[Path, Path],
    effective_losses: Mapping[str, float],
) -> dict[str, Any]:
    semantic = {
        "schema_version": 3,
        "experiment_name": config.experiment_name,
        "base_checkpoint": _artifact(config.checkpoint),
        "model_config": _artifact(config.model_config),
        "dataset_profile": _artifact(profile_path),
        "manifests": {
            name: _artifact(path)
            for name, path in zip(("train", "val"), manifests)
        },
        "data": {
            "type": config.data_type,
            "image_size": config.image_size,
            "rescale_factor": config.rescale_factor,
            **({"crop_jitter":config.crop_jitter} if config.crop_jitter else {}),
            "marker_inputs": {
                name: _artifact(path) for name, path in config.marker_inputs.items()
            },
        },
        "recipe": {
            "type": config.recipe_type,
            "vit_tuning": config.vit_tuning,
            "camera_model": config.camera_model,
            "lora": None if config.lora is None else config.lora.to_dict(),
        },
        "losses": {key: value for key, value in effective_losses.items()
                   if key in config.losses or (key != "surface_markers_3d" and key not in V5_LOSS_SOURCES)},
        "optimization": {
            "vit_lr": config.vit_lr,
            "coarse_lr": config.coarse_lr,
            "refiner_lr": config.refiner_lr,
            "weight_decay": config.weight_decay,
            "grad_clip": config.grad_clip,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "batch_size": config.batch_size,
        },
        "seed": config.seed,
        "runtime": {
            "device": config.device,
            "amp": config.use_amp,
            "num_workers": config.num_workers,
            "max_train_samples": config.max_train_samples,
            "val_steps": config.val_steps,
            "val_batches": config.val_batches,
        },
    }
    if config.pinhole_crop:
        semantic["data"]["pinhole_crop"] = config.pinhole_crop
    if config.camera_lr is not None:
        semantic['optimization']['camera_lr'] = config.camera_lr
    semantic["data"].update({
        key: _artifact(value) if isinstance(value, Path) else value
        for key, value in config.augmentation_options().items()
    })
    semantic["optimization"].update(config.schedule_options())
    canonical = json.dumps(
        semantic, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return {
        "signature": hashlib.sha256(canonical).hexdigest(),
        "config": semantic,
    }


def lr_multiplier(step, max_steps, lr_schedule, warmup_steps, min_lr_ratio):
    """Use the absolute optimizer step, including after a checkpoint resume."""
    if warmup_steps and step <= warmup_steps:
        return max(0, step) / warmup_steps
    if lr_schedule == "constant":
        return 1.0
    start = max(1, warmup_steps)
    progress = min(1.0, max(0.0, (step - start) / max(1, max_steps - start)))
    return min_lr_ratio + (1 - min_lr_ratio) * (1 + math.cos(math.pi * progress)) / 2


def _validation_errors(output, batch):
    """Accumulate numerators and visible-point counts across unequal batches."""
    if "pred_keypoints_3d" not in output or "keypoints_3d" not in batch:
        return {}
    truth = batch["keypoints_3d"]
    pred = output["pred_keypoints_3d"].float()
    offset = pred - truth[..., :3]
    distances = (offset - offset[:, :1]).norm(dim=-1) * 1000
    valid = (truth[..., 3] > 0) & (truth[:, :1, 3] > 0)
    def totals(error, mask):
        return torch.stack((error.masked_fill(~mask, 0.).sum(), mask.sum()))
    values = {
        "val_mpjpe_mm": totals(distances, valid),
        "val_fingertip_mpjpe_mm": totals(distances[:, FINGERTIPS], valid[:, FINGERTIPS]),
        "val_pinky_tip_mm": totals(distances[:, 20], valid[:, 20]),
    }
    if "pred_keypoints_2d" in output and "keypoints_2d" in batch:
        truth2d = batch["keypoints_2d"]
        size = truth2d.new_tensor([batch["img"].shape[-1], batch["img"].shape[-2]])
        error = ((output["pred_keypoints_2d"].float() - truth2d[..., :2]) * size).norm(dim=-1)
        values["val_reproj_crop_px"] = totals(error, truth2d[..., 2] > 0)
    return values


class GenericTrainer:
    """Generic AMP/backward/evaluation mechanics with recipe-owned forward."""

    def __init__(
        self,
        model: WiLoR,
        recipe: TrainingRecipe,
        optimizer: torch.optim.Optimizer,
        loss_computer: UnifiedLossComputer,
        device: torch.device,
        use_amp: bool,
        grad_clip: float,
    ) -> None:
        self.model = model
        self.recipe = recipe
        self.optimizer = optimizer
        self.loss_computer = loss_computer
        self.device = device
        self.use_amp = bool(use_amp and device.type == "cuda")
        self.grad_clip = float(grad_clip)
        self.validation_metrics: dict[str, float] = {}

    def _amp_context(self):
        return (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if self.use_amp else nullcontext()
        )

    def _forward_batch(
        self,
        batch: dict[str, Any],
        training: bool,
        validation_totals: dict | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor, int]:
        device_batch = _move(batch, self.device)
        with torch.set_grad_enabled(training), self._amp_context():
            output = self.recipe.forward(
                self.model, device_batch, training=training
            )
            loss, components = self.loss_computer(device_batch, output)
            _, metric = self.recipe.metric(device_batch, output)
        if validation_totals is not None:
            for key, stats in _validation_errors(output, device_batch).items():
                validation_totals[key] = validation_totals.get(key, 0) + stats.double()
        return loss, components, metric, int(batch["img"].shape[0])

    def train_step(
        self, batches: Iterable[dict[str, Any]]
    ) -> tuple[float, float, dict[str, float]]:
        values = list(batches)
        if not values:
            raise ValueError("train_step requires at least one batch")
        self.recipe.set_mode(self.model, training=True)
        self.optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        total_metric = 0.0
        total_samples = 0
        component_totals: dict[str, float] = {}
        for batch in values:
            loss, components, metric, count = self._forward_batch(batch, True)
            (loss / len(values)).backward()
            total_loss += float(loss.detach().cpu()) * count
            total_metric += float(metric.detach().cpu()) * count
            total_samples += count
            for name, value in components.items():
                component_totals[name] = (
                    component_totals.get(name, 0.0)
                    + float(value.detach().cpu()) * count
                )
        torch.nn.utils.clip_grad_norm_(
            self.recipe.trainable_parameters(self.model),
            self.grad_clip,
            error_if_nonfinite=True,
        )
        self.optimizer.step()
        return (
            total_loss / total_samples,
            total_metric / total_samples,
            {name: value / total_samples for name, value in component_totals.items()},
        )

    @torch.no_grad()
    def evaluate(
        self,
        loader: DataLoader,
        max_batches: int | None,
    ) -> tuple[float, float, int, dict[str, float]]:
        self.recipe.set_mode(self.model, training=False)
        total_loss = 0.0
        total_metric = 0.0
        total_samples = 0
        component_totals: dict[str, float] = {}
        batch_count = 0
        validation_totals = {}
        for index, batch in enumerate(loader):
            if max_batches is not None and index >= max_batches:
                break
            loss, components, metric, count = self._forward_batch(batch, False, validation_totals)
            total_loss += float(loss.cpu()) * count
            total_metric += float(metric.cpu()) * count
            total_samples += count
            batch_count += 1
            for name, value in components.items():
                component_totals[name] = (
                    component_totals.get(name, 0.0) + float(value.cpu()) * count
                )
        self.validation_metrics = {key: float(sums[0] / sums[1]) for key, sums in validation_totals.items()
                                   if sums[1].item() > 0}
        if total_samples == 0:
            raise RuntimeError("Validation loader produced no batches")
        return (
            total_loss / total_samples,
            total_metric / total_samples,
            batch_count,
            {name: value / total_samples for name, value in component_totals.items()},
        )


def _preload_weights(config, model):
    """Restore imported modules before the recipe constructs optimizer groups."""
    if config.init_weights is None:
        if config.resume is not None:
            load_trainable_weights(config.resume, model)
        return None

    payload = load_trainable_weights(config.init_weights, model)
    previous_mode = payload.get("model_modifications", {}).get("physical_camera_model")
    if previous_mode is not None and previous_mode != config.camera_model:
        last = model.physical_camera_head.net[-1]
        torch.nn.init.zeros_(last.weight)
        torch.nn.init.zeros_(last.bias)
    return _artifact(config.init_weights)


def run_training(config: TrainingConfig) -> dict[str, Any]:
    """Execute either schema-v3 recipe through one common training loop."""
    config.validate()
    for path, description in (
        (config.checkpoint, "base checkpoint"),
        (config.model_config, "model config"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Missing {description}: {path}")
    for path, description in (
        (config.resume, "resume checkpoint"),
        (config.init_weights, "initialization checkpoint"),
    ):
        if path is not None and not path.is_file():
            raise FileNotFoundError(f"Missing {description}: {path}")
    if config.local_export is not None:
        from wilor.datasets.local_pinhole import prepare_local_views
        prepare_local_views(config)
    train_manifest, val_manifest = _manifests(config)
    config.train_manifest, config.val_manifest = train_manifest, val_manifest
    train_set, val_set, profile = build_datasets(
        config, train_manifest, val_manifest
    )
    loss_config = unified_loss_config_from_mapping(
        config.losses, profile, config.source
    )
    seed_everything(config.seed)
    device = resolve_device(config.device)
    use_amp = config.use_amp and device.type == "cuda"
    recipe = build_recipe(config)
    model = _load_base_model(config)
    initialized_from = _preload_weights(config, model)
    if config.camera_model in {"physical_perspective", "physical_kb", "kb_bbox_pinhole"}:
        model.physical_camera_enabled = True
        model.physical_camera_head.camera_model = config.camera_model
    if config.recipe_type in {"marker_camera_finetune", "marker_offset_finetune"}:
        model.marker_head_enabled = True
    model.crop_rescale_factor = config.rescale_factor
    recipe.configure_model(model)
    model = model.to(device)
    recipe.set_mode(model, training=True)
    optimizer = recipe.build_optimizer(model)
    base_lrs = [group["lr"] for group in optimizer.param_groups]
    trainer = GenericTrainer(
        model,
        recipe,
        optimizer,
        UnifiedLossComputer(model, loss_config),
        device,
        use_amp,
        config.grad_clip,
    )

    profile_path = config.dataset_root / "dataset_profile.json"
    metadata = _training_metadata(
        config, profile_path, (train_manifest, val_manifest), loss_config.effective
    )
    metadata["initialized_from"] = initialized_from
    start_step = 0
    best_val_loss = float("inf")
    selection_metric = (
        "val_metric" if recipe.name in {"marker_camera_finetune", "marker_offset_finetune"}
        else "val_loss"
    )
    best_selection_score = float("inf")
    best_selection_step = 0
    best_mpjpe_path = config.output / "best_mpjpe_full.pt"
    val_mpjpe_key = f"val_{recipe.mpjpe_name}"
    best_mpjpe_key = f"best_{recipe.mpjpe_name}"
    if config.resume is not None:
        payload = resume_trainable_checkpoint(config.resume, model, optimizer)
        actual = payload.get("training_metadata", {})
        if not isinstance(actual, dict) or actual.get("signature") != metadata["signature"]:
            raise ValueError(
                f"{config.resume}: resume configuration differs from checkpoint; "
                "use paths.init_weights to start a new stage"
            )
        metadata = actual
        restore_best_mpjpe(best_mpjpe_path, metadata, metric_name=recipe.mpjpe_name)
        initialized_from = actual.get("initialized_from")
        start_step = int(payload.get("step", 0))
        if payload.get("val_loss") is not None:
            best_val_loss = float(payload["val_loss"])
        best_selection_score = float(actual.get("best_selection_score", best_val_loss))
        best_selection_step = int(actual.get("best_selection_step", start_step))
    if start_step > config.max_steps:
        raise ValueError(
            f"Resume step {start_step} exceeds max_steps {config.max_steps}"
        )

    config.output.mkdir(parents=True, exist_ok=True)
    write_yaml(config.output / "resolved_config.yaml", resolved_training_config(config))
    loader_options = {
        "num_workers": config.num_workers,
        "pin_memory": device.type == "cuda",
    }
    if config.group_by_sync_frame:
        sampler = SyncFrameBatchSampler(train_set.rows, config.batch_size, config.seed)
        train_loader = DataLoader(train_set, batch_sampler=sampler, **loader_options)
    else:
        train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, **loader_options)
    val_loader = DataLoader(
        val_set,
        batch_size=config.evaluation_batch_size,
        shuffle=False,
        **loader_options,
    )
    train_iterator = iter(train_loader)

    def next_batch() -> dict[str, Any]:
        nonlocal train_iterator
        try:
            return next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            return next(train_iterator)

    metric_name = recipe.metric_name
    best_path = config.output / "best_trainable.pt"
    last_path = config.output / "last_trainable.pt"
    report_path = config.output / "train_report.json"
    curves_path = config.output / "loss_curves.png"
    history: list[dict[str, Any]] = []
    started = time.time()

    print(
        f"recipe={recipe.name} device={device} amp={use_amp} "
        f"train={len(train_set)} val={len(val_set)} start_step={start_step}",
        flush=True,
    )

    def report(status: str, step: int) -> dict[str, Any]:
        return {
            "status": status,
            "experiment_name": config.experiment_name,
            "recipe": recipe.name,
            "data_type": config.data_type,
            "device": str(device),
            "amp": use_amp,
            "metric_name": metric_name,
            "trainable_parameters": sum(
                parameter.numel()
                for parameter in recipe.trainable_parameters(model)
            ),
            "train_samples": len(train_set),
            "val_samples": len(val_set),
            "start_step": start_step,
            "current_step": step,
            "best_val_loss": best_val_loss,
            "selection_metric": selection_metric,
            "best_selection_score": best_selection_score,
            "best_selection_step": best_selection_step,
            best_mpjpe_key: metadata.get(best_mpjpe_key),
            "best_mpjpe_step": metadata.get("best_mpjpe_step"),
            "loss_config": loss_config.as_dict(),
            "training_signature": metadata["signature"],
            "initialized_from": initialized_from,
            "elapsed_seconds": time.time() - started,
            "history": history,
        }

    for step in range(start_step + 1, config.max_steps + 1):
        lr_scale = lr_multiplier(step, config.max_steps, config.lr_schedule,
                                 config.warmup_steps, config.min_lr_ratio)
        if config.warmup_steps or config.lr_schedule != "constant":
            for group, base_lr in zip(optimizer.param_groups, base_lrs):
                group["lr"] = base_lr * lr_scale
        batches = [
            next_batch() for _ in range(config.gradient_accumulation_steps)
        ]
        train_loss, train_metric, train_components = trainer.train_step(batches)
        record: dict[str, Any] = {
            "step": step,
            "metric_name": metric_name,
            "train_loss": train_loss,
            "train_metric": train_metric,
            "lr_scale": lr_scale,
            "train_loss_components": train_components,
        }
        should_validate = (
            step == start_step + 1
            or step % config.val_steps == 0
            or step == config.max_steps
        )
        if should_validate:
            val_loss, val_metric, val_batches, val_components = trainer.evaluate(
                val_loader, config.val_batches
            )
            record.update({
                "val_loss": val_loss,
                "val_metric": val_metric,
                "val_batches": val_batches,
                "val_loss_components": val_components,
                **trainer.validation_metrics,
            })
            if metric_name == recipe.mpjpe_name:
                record[val_mpjpe_key] = val_metric
            if val_mpjpe_key in record:
                save_best_mpjpe(
                    best_mpjpe_path, model, step, record[val_mpjpe_key], metadata,
                    train_loss=train_loss, val_loss=val_loss, metric_name=recipe.mpjpe_name,
                )
            best_val_loss = min(best_val_loss, val_loss)
            selection_score = val_metric if selection_metric == "val_metric" else val_loss
            if selection_score < best_selection_score:
                best_selection_score = selection_score
                best_selection_step = step
                metadata["best_selection_score"] = best_selection_score
                metadata["best_selection_step"] = best_selection_step
                save_trainable_checkpoint(
                    best_path, model, optimizer, step, config.checkpoint,
                    recipe.name, val_loss, metadata,
                )
                save_full_checkpoint(config.output / "best_full.pt", model, step)
            print(json.dumps(record), flush=True)
        elif step % 10 == 0:
            print(json.dumps(record), flush=True)
        history.append(record)
        if should_validate:
            write_json(report_path, report("running", step))
            write_training_curves(history, curves_path)
        if step % config.save_steps == 0 or step == config.max_steps:
            save_trainable_checkpoint(
                last_path, model, optimizer, step, config.checkpoint,
                recipe.name, best_val_loss, metadata,
            )

    save_full_checkpoint(config.output / "last_full.pt", model, config.max_steps)
    result = report("passed", config.max_steps)
    if config.final_visualization:
        if recipe.name not in {"refiner_finetune", "joint_lora_finetune", "physical_camera_finetune", "marker_camera_finetune", "marker_offset_finetune"}:
            raise ValueError(
                "final_visualization requires a camera-aligned recipe"
            )
        load_trainable_weights(best_path if best_path.is_file() else last_path, model)
        recipe.set_mode(model, training=False)
        evaluation_set = GloveDataset(
            val_manifest,
            image_size=config.image_size,
            rescale_factor=config.rescale_factor,
            max_samples=config.evaluation_max_samples,
            marker_inputs=config.marker_inputs,
            camera_model=config.camera_model,
        )
        evaluation_loader = DataLoader(
            evaluation_set,
            batch_size=config.evaluation_batch_size,
            shuffle=False,
            **loader_options,
        )
        result["val_visualization"] = evaluate_glove_set(
            model,
            evaluation_loader,
            device,
            config.output / "val_visualization",
            use_amp,
        )
    write_json(report_path, result)
    write_training_curves(history, curves_path)
    return result
