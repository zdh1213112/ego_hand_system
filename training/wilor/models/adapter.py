"""Serialization helpers for lightweight WiLoR glove adapters."""

from __future__ import annotations

import math
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Mapping

from pytorch_lightning import __version__ as lightning_version
import torch


ADAPTER_FORMAT = "wilor_glove_adapter_v1"
TRAINABLE_FORMAT = "wilor_glove_trainable_v2"
FULL_FORMAT = "wilor_full_inference_v1"
ADAPTER_PREFIXES = (
    "backbone.decpose.",
    "backbone.decshape.",
    "backbone.deccam.",
    "refine_net.",
)
CARRIED_STATE_NAMES_ATTRIBUTE = "_wilor_carried_state_names"


def adapter_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Return a detached CPU snapshot of all trainable adapter modules."""
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
        if name.startswith(ADAPTER_PREFIXES)
    }


def trainable_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Snapshot selected parameters and their module-owned persistent buffers.

    BatchNorm running statistics are buffers rather than parameters.  They
    still change while a trainable module is in training mode and must travel
    with its weights for evaluation and exact resume.
    """
    selected_parameters = {
        name for name, parameter in model.named_parameters()
        if parameter.requires_grad
        or name.endswith(("lora_A.weight", "lora_B.weight"))
    }
    carried_state = set(getattr(model, CARRIED_STATE_NAMES_ATTRIBUTE, ()))
    selected_modules = {
        name.rpartition(".")[0] for name in selected_parameters
    }
    selected_buffers = {
        name for name, _ in model.named_buffers()
        if name.rpartition(".")[0] in selected_modules
    }
    selected = selected_parameters | selected_buffers | carried_state
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
        if name in selected
    }


def _to_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_cpu(item) for item in value)
    return value


def save_adapter_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    base_checkpoint: str | Path,
    val_loss: float | None = None,
    training_metadata: Mapping[str, Any] | None = None,
) -> None:
    """Persist adapter weights and the optimizer state needed for resume."""
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": ADAPTER_FORMAT,
            "step": int(step),
            "base_checkpoint": str(Path(base_checkpoint).expanduser().resolve()),
            "state_dict": adapter_state_dict(model),
            "optimizer": _to_cpu(optimizer.state_dict()),
            "val_loss": None if val_loss is None else float(val_loss),
            "training_metadata": _to_cpu(dict(training_metadata or {})),
        },
        destination,
    )


def checkpoint_rescale_factor(payload: Mapping[str, Any]) -> float:
    data = payload.get("training_metadata", {}).get("config", {}).get("data", {})
    return float(data.get("rescale_factor", payload.get("model_modifications", {}).get("rescale_factor", 2.0)))


def _model_modifications(model: torch.nn.Module) -> dict[str, Any]:
    modifications: dict[str, Any] = {}
    if getattr(model, "crop_rescale_factor", 2.0) != 2.0:
        modifications["rescale_factor"] = model.crop_rescale_factor
    lora_config = getattr(getattr(model, "backbone", None), "_wilor_lora_config", None)
    if lora_config is not None:
        modifications["lora"] = lora_config.to_dict()
    if getattr(model, "physical_camera_enabled", False):
        modifications["physical_camera"] = True
        modifications["physical_camera_model"] = model.physical_camera_head.camera_model
        modifications['native_root_prior'] = model.physical_camera_head.native_root_prior
    if getattr(model, "marker_head_enabled", False):
        modifications["marker_head"] = True
    return modifications


def save_full_checkpoint(
    path: Path, model: torch.nn.Module, step: int, *,
    training_metadata: Mapping[str, Any] | None = None,
) -> None:
    """Write all inference weights and the configuration needed to restore them."""
    checkpoint = {
        "format": FULL_FORMAT,
        "pytorch-lightning_version": str(lightning_version),
        "global_step": step,
        "hyper_parameters": {"cfg": model.cfg.dump()},
        "model_modifications": _model_modifications(model),
        "state_dict": model.state_dict(),
    }
    if training_metadata is not None:
        checkpoint["training_metadata"] = dict(training_metadata)
    # Keep the previous complete file available until the large write succeeds.
    with TemporaryDirectory(dir=path.parent, prefix=path.name + ".") as directory:
        temporary = Path(directory) / path.name
        torch.save(checkpoint, temporary)
        temporary.replace(path)


def restore_best_mpjpe(
    path: Path, metadata: dict[str, Any], *, metric_name: str = "mpjpe_mm",
) -> None:
    """An independent validation save can be newer than the resume checkpoint."""
    if not path.is_file():
        return
    # Read the selection metadata without allocating the full model's tensors.
    saved = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    best = saved.get("training_metadata", {})
    score_key = f"best_{metric_name}"
    if (best.get("signature") == metadata["signature"]
            and best.get(score_key, math.inf) < metadata.get(score_key, math.inf)):
        metadata[score_key] = best[score_key]
        metadata["best_mpjpe_step"] = saved["global_step"]


def save_best_mpjpe(
    path: Path, model: torch.nn.Module, step: int, mpjpe: float,
    metadata: dict[str, Any], *, train_loss: float, val_loss: float,
    metric_name: str = "mpjpe_mm",
) -> None:
    """Keep the full model from the strictly lowest finite validation MPJPE."""
    if not all(math.isfinite(value) for value in (mpjpe, train_loss, val_loss)):
        return
    score_key = f"best_{metric_name}"
    if mpjpe >= metadata.get(score_key, math.inf):
        return
    best = {
        "signature": metadata["signature"],
        score_key: mpjpe,
        "best_mpjpe_step": step,
    }
    save_full_checkpoint(path, model, step, training_metadata=best)
    metadata.update(best)


def save_trainable_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    base_checkpoint: str | Path,
    recipe: str,
    val_loss: float | None = None,
    training_metadata: Mapping[str, Any] | None = None,
) -> None:
    """Save a composable schema-v2 checkpoint for either training recipe."""
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    state = trainable_state_dict(model)
    if not state:
        raise ValueError("Cannot save a checkpoint with no trainable state")
    torch.save(
        {
            "format": TRAINABLE_FORMAT,
            "recipe": str(recipe),
            "step": int(step),
            "base_checkpoint": str(Path(base_checkpoint).expanduser().resolve()),
            "model_modifications": _model_modifications(model),
            "state_dict": state,
            "optimizer": _to_cpu(optimizer.state_dict()),
            "val_loss": None if val_loss is None else float(val_loss),
            "training_metadata": _to_cpu(dict(training_metadata or {})),
        },
        destination,
    )


def _read_checkpoint(path: str | Path) -> tuple[Path, dict[str, Any]]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Adapter checkpoint does not exist: {source}")
    payload = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"Adapter checkpoint must contain a dictionary: {source}")
    if payload.get("format") != ADAPTER_FORMAT:
        raise ValueError(
            f"Unsupported adapter format {payload.get('format')!r} in {source}; "
            f"expected {ADAPTER_FORMAT!r}"
        )
    state = payload.get("state_dict")
    if not isinstance(state, Mapping) or not state:
        raise ValueError(f"Adapter checkpoint has no state_dict: {source}")
    return source, payload


def _validate_state(
    source: Path,
    model: torch.nn.Module,
    state: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    expected = adapter_state_dict(model)
    missing = sorted(set(expected).difference(state))
    unexpected = sorted(set(state).difference(expected))
    if missing or unexpected:
        raise ValueError(
            f"Adapter keys do not match {source}: "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}"
        )

    validated: dict[str, torch.Tensor] = {}
    for name, expected_value in expected.items():
        value = state[name]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Adapter value is not a tensor: {name}")
        if value.shape != expected_value.shape:
            raise ValueError(
                f"Adapter shape mismatch for {name}: "
                f"{tuple(value.shape)} != {tuple(expected_value.shape)}"
            )
        validated[name] = value
    return validated


def load_adapter_weights(
    path: str | Path,
    model: torch.nn.Module,
) -> dict[str, Any]:
    """Load only model weights and return the checkpoint metadata."""
    checkpoint_path = Path(path).expanduser().resolve()
    raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(raw, dict) and raw.get("format") == TRAINABLE_FORMAT:
        # Full-ViT checkpoints include large optimizer states. Release the
        # format probe before the trainable loader reads the checkpoint again.
        del raw
        return load_trainable_weights(checkpoint_path, model)
    source, payload = _read_checkpoint(path)
    state = _validate_state(source, model, payload["state_dict"])
    current = model.state_dict()
    with torch.no_grad():
        for name, value in state.items():
            current[name].copy_(value.to(
                device=current[name].device,
                dtype=current[name].dtype,
            ))
    return payload


def _read_trainable_checkpoint(path: str | Path) -> tuple[Path, dict[str, Any]]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Trainable checkpoint does not exist: {source}")
    payload = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("format") != TRAINABLE_FORMAT:
        raise ValueError(
            f"Unsupported trainable checkpoint format in {source}; expected "
            f"{TRAINABLE_FORMAT!r}"
        )
    state = payload.get("state_dict")
    if not isinstance(state, Mapping) or not state:
        raise ValueError(f"Trainable checkpoint has no state_dict: {source}")
    return source, payload


def restore_checkpoint_modifications(
    model: torch.nn.Module,
    payload: Mapping[str, Any],
    source: str | Path,
) -> None:
    modifications = payload.get("model_modifications", {})
    if not isinstance(modifications, Mapping):
        raise TypeError(f"{source}: model_modifications must be a mapping")
    lora = modifications.get("lora")
    if lora is not None:
        if not isinstance(lora, dict):
            raise TypeError(f"{source}: model_modifications.lora must be a mapping")
        from .lora import LoRAConfig, inject_vit_lora
        inject_vit_lora(model.backbone, LoRAConfig.from_dict(lora))
    if modifications.get("physical_camera"):
        model.physical_camera_enabled = True
        camera_model = modifications.get('physical_camera_model', 'physical_perspective')
        if camera_model not in {'physical_perspective', 'physical_kb', 'kb_bbox_pinhole'}:
            raise ValueError(f'{source}: unsupported physical camera model {camera_model}')
        model.physical_camera_head.camera_model = camera_model
        model.physical_camera_head.native_root_prior = modifications.get('native_root_prior', 'virtual')
    if modifications.get("marker_head"):
        model.marker_head_enabled = True
        model.physical_camera_enabled = True


def load_trainable_weights(
    path: str | Path,
    model: torch.nn.Module,
) -> dict[str, Any]:
    """Load matching prior-stage state without requiring trainable-set equality."""
    source, payload = _read_trainable_checkpoint(path)
    restore_checkpoint_modifications(model, payload, source)
    current = model.state_dict()
    state = payload["state_dict"]
    unknown = sorted(set(state).difference(current))
    if unknown:
        raise ValueError(f"{source}: checkpoint has unknown model keys {unknown[:5]}")
    validated: dict[str, torch.Tensor] = {}
    for name, value in state.items():
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{source}: checkpoint value is not a tensor: {name}")
        if value.shape != current[name].shape:
            raise ValueError(
                f"{source}: shape mismatch for {name}: "
                f"{tuple(value.shape)} != {tuple(current[name].shape)}"
            )
        validated[name] = value
    with torch.no_grad():
        for name, value in validated.items():
            current[name].copy_(value.to(
                device=current[name].device,
                dtype=current[name].dtype,
            ))
    # A later recipe may freeze weights imported from an earlier stage. Keep
    # those weights in subsequent composable checkpoints so a Stage 2 artifact
    # remains self-contained for inference and further initialization.
    carried_state = set(getattr(model, CARRIED_STATE_NAMES_ATTRIBUTE, ()))
    carried_state.update(validated)
    setattr(model, CARRIED_STATE_NAMES_ATTRIBUTE, frozenset(carried_state))
    return payload


def resume_trainable_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> dict[str, Any]:
    """Strictly restore weights and optimizer for the same schema-v3 run."""
    source, payload = _read_trainable_checkpoint(path)
    restore_checkpoint_modifications(model, payload, source)
    expected = trainable_state_dict(model)
    state = payload["state_dict"]
    missing = sorted(set(expected).difference(state))
    unexpected = sorted(set(state).difference(expected))
    if missing or unexpected:
        raise ValueError(
            f"Trainable keys do not match {source}: "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}"
        )
    load_trainable_weights(source, model)
    optimizer_state = payload.get("optimizer")
    if not isinstance(optimizer_state, dict):
        raise ValueError(f"Trainable checkpoint has no optimizer state: {source}")
    optimizer.load_state_dict(optimizer_state)
    return payload


def resume_adapter_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> dict[str, Any]:
    """Restore adapter weights plus optimizer state for continued training."""
    payload = load_adapter_weights(path, model)
    optimizer_state = payload.get("optimizer")
    if not isinstance(optimizer_state, dict):
        raise ValueError(f"Adapter checkpoint has no optimizer state: {path}")
    optimizer.load_state_dict(optimizer_state)
    return payload
