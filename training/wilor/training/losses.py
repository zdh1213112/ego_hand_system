"""Stable, profile-gated loss interface for glove-domain training."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F
import yaml


LOSS_NAMES = (
    "keypoints_2d",
    "keypoints_3d",
    "coarse_keypoints_2d",
    "coarse_keypoints_3d",
    "global_orient",
    "hand_pose",
    "betas",
)
OPTIONAL_UNIFIED_LOSS_NAMES = (
    "physical_root_3d", "marker_keypoints_2d", "marker_keypoints_3d", "marker_root_3d", "surface_markers_3d"
)
UNIFIED_LOSS_NAMES = ("local_keypoints_3d",) + LOSS_NAMES + OPTIONAL_UNIFIED_LOSS_NAMES
AVAILABILITY_SOURCE = {
    "local_keypoints_3d": "local_keypoints_3d",
    "keypoints_2d": "keypoints_2d",
    "keypoints_3d": "keypoints_3d",
    "coarse_keypoints_2d": "keypoints_2d",
    "coarse_keypoints_3d": "keypoints_3d",
    "global_orient": "global_orient",
    "hand_pose": "hand_pose",
    "betas": "betas",
    "physical_root_3d": "keypoints_3d",
    "marker_keypoints_2d": "keypoints_2d",
    "marker_keypoints_3d": "keypoints_3d",
    "marker_root_3d": "keypoints_3d",
    "surface_markers_3d": "surface_markers_3d",
}


@dataclass(frozen=True)
class GloveLossConfig:
    source: Path
    requested: dict[str, float]
    effective: dict[str, float]
    disabled_reasons: dict[str, str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "source": str(self.source),
            "requested": self.requested,
            "effective": self.effective,
            "disabled_reasons": self.disabled_reasons,
        }


def load_glove_loss_config(
    path: str | Path,
    dataset_profile: Mapping[str, Any],
) -> GloveLossConfig:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Loss config does not exist: {source}")
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema_version") not in (1, 2):
        raise ValueError(f"{source}: expected schema_version 1 or 2")
    losses = raw.get("losses")
    return glove_loss_config_from_mapping(losses, dataset_profile, source)


def glove_loss_config_from_mapping(
    losses: Any,
    dataset_profile: Mapping[str, Any],
    source: str | Path,
) -> GloveLossConfig:
    """Validate loss weights embedded in a complete training config."""
    source_path = Path(source).expanduser().resolve()
    if not isinstance(losses, dict) or set(losses) != set(LOSS_NAMES):
        raise ValueError(
            f"{source_path}: losses must contain exactly {list(LOSS_NAMES)}"
        )
    requested = {}
    for name in LOSS_NAMES:
        value = losses[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{source_path}: loss {name} must be numeric")
        if not 0 <= float(value) < float("inf"):
            raise ValueError(
                f"{source_path}: loss {name} must be finite and non-negative"
            )
        requested[name] = float(value)
    supervision = dataset_profile.get("supervision")
    if not isinstance(supervision, Mapping):
        raise ValueError("dataset_profile.json has no supervision map")
    effective = {}
    disabled_reasons = {}
    for name in LOSS_NAMES:
        availability_name = AVAILABILITY_SOURCE[name]
        availability = supervision.get(availability_name, {})
        available = bool(
            isinstance(availability, Mapping) and availability.get("available", False)
        )
        effective[name] = requested[name] if available else 0.0
        if not available:
            reason = (
                availability.get("reason", "supervision unavailable")
                if isinstance(availability, Mapping) else "supervision unavailable"
            )
            disabled_reasons[name] = str(reason)
    if not any(effective.values()):
        raise ValueError("No requested loss has valid supervision")
    return GloveLossConfig(
        source_path,
        requested,
        effective,
        disabled_reasons,
    )


def unified_loss_config_from_mapping(
    losses: Any,
    dataset_profile: Mapping[str, Any],
    source: str | Path,
) -> GloveLossConfig:
    """Validate the schema-v3 loss map and gate it by data supervision."""
    source_path = Path(source).expanduser().resolve()
    required = set(UNIFIED_LOSS_NAMES) - set(OPTIONAL_UNIFIED_LOSS_NAMES)
    if not isinstance(losses, dict) or not required <= set(losses) <= set(UNIFIED_LOSS_NAMES):
        raise ValueError(
            f"{source_path}: losses must contain exactly "
            f"{list(UNIFIED_LOSS_NAMES)}"
        )
    requested: dict[str, float] = {}
    for name in UNIFIED_LOSS_NAMES:
        value = losses.get(name, 0.0)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{source_path}: loss {name} must be numeric")
        if not 0 <= float(value) < float("inf"):
            raise ValueError(
                f"{source_path}: loss {name} must be finite and non-negative"
            )
        requested[name] = float(value)
    supervision = dataset_profile.get("supervision")
    if not isinstance(supervision, Mapping):
        raise ValueError("dataset_profile.json has no supervision map")
    effective: dict[str, float] = {}
    disabled_reasons: dict[str, str] = {}
    for name in UNIFIED_LOSS_NAMES:
        availability = supervision.get(AVAILABILITY_SOURCE[name], {})
        available = bool(
            isinstance(availability, Mapping) and availability.get("available", False)
        )
        effective[name] = requested[name] if available else 0.0
        if not available:
            disabled_reasons[name] = str(
                availability.get("reason", "supervision unavailable")
                if isinstance(availability, Mapping)
                else "supervision unavailable"
            )
    if not any(effective.values()):
        raise ValueError("No requested loss has valid supervision")
    return GloveLossConfig(
        source_path,
        requested,
        effective,
        disabled_reasons,
    )


def local_keypoint_3d_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor | None = None,
    beta: float = 0.01,
) -> torch.Tensor:
    """Smooth-L1 over palm-local joints, averaged over valid samples."""
    if prediction.shape != target.shape or prediction.shape[-2:] != (21, 3):
        raise ValueError(
            "local 3D prediction and target must both have shape [B, 21, 3], "
            f"got {tuple(prediction.shape)} and {tuple(target.shape)}"
        )
    per_sample = F.smooth_l1_loss(
        prediction,
        target.to(device=prediction.device, dtype=prediction.dtype),
        beta=float(beta),
        reduction="none",
    ).mean(dim=(-2, -1))
    if valid is None:
        return per_sample.mean()
    mask = valid.to(device=prediction.device, dtype=torch.bool).reshape(-1)
    if mask.numel() != prediction.shape[0]:
        raise ValueError("local 3D validity mask does not match batch size")
    if not bool(mask.any()):
        return prediction.sum() * 0.0
    return per_sample[mask].mean()


def measured_surface_loss(output, batch):
    """Align actual measured marker centres with points attached to the predicted MANO mesh."""
    with torch.autocast(device_type=output['pred_vertices'].device.type, enabled=False):
        vertices = output['pred_vertices'].float()
        row = torch.arange(len(vertices),device=vertices.device)[:,None,None]
        triangles = vertices[row,batch['surface_triangles']]
        edge = F.normalize(triangles[:,:,1]-triangles[:,:,0],dim=-1)
        normal = F.normalize(torch.cross(edge,triangles[:,:,2]-triangles[:,:,0],dim=-1),dim=-1)
        lateral = torch.cross(normal,edge,dim=-1)
        offset = batch['surface_offsets_m']
        mounted = (triangles*batch['surface_barycentric'][...,None]).sum(2)
        mounted = mounted+edge*offset[...,:1]+lateral*offset[...,1:2]+normal*offset[...,2:]
        root = output['pred_root_camera_m'].float()
        root = torch.cat([root[:,:1]*(2*batch['side'][:,None]-1),root[:,1:]],dim=-1)
        residual = mounted-output['pred_keypoints_3d'][:,:1].float()+root[:,None]-batch['surface_target_m']
        mask = batch['surface_valid'][...,None]
        count = mask.sum(1).clamp_min(1)
        translation = (residual*mask).sum(1)/count
        centered = residual-translation[:,None]
        local = F.smooth_l1_loss(centered,torch.zeros_like(centered),beta=.003,reduction='none')
        absolute = F.smooth_l1_loss(translation,torch.zeros_like(translation),beta=.015)
        return ((local*mask).sum((1,2))/count[:,0]).mean() + .5*absolute


class UnifiedLossComputer:
    """Loss dispatcher shared by domain adaptation and RefineNet fine-tuning."""

    def __init__(self, model: torch.nn.Module, config: GloveLossConfig) -> None:
        self.model = model
        self.config = config

    def __call__(
        self,
        batch: Mapping[str, Any],
        output: dict[str, Any],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        anchor = next(
            output[name]
            for name in (
                "coarse_local_keypoints_3d",
                "pred_keypoints_3d",
                "coarse_keypoints_3d",
            )
            if name in output
        )
        zero = anchor.sum() * 0.0
        raw: dict[str, torch.Tensor] = {
            name: zero for name in UNIFIED_LOSS_NAMES
        }
        if self.config.effective["local_keypoints_3d"] > 0:
            raw["local_keypoints_3d"] = local_keypoint_3d_loss(
                output["coarse_local_keypoints_3d"],
                batch["local_keypoints_3d"],
                batch.get("has_local_keypoints_3d"),
            )
        if self.config.effective["physical_root_3d"] > 0:
            if "pred_root_camera_m" not in output or "camera_root_m" not in batch:
                raise KeyError("physical root supervision requires calibrated camera output and labels")
            root_loss = F.smooth_l1_loss(output["pred_root_camera_m"], batch["camera_root_m"], beta=0.02,reduction='none')
            if 'geometry_supervised' in batch:
                trusted = batch['geometry_supervised']
                raw["physical_root_3d"] = (root_loss.mean(1)*trusted).sum()/trusted.sum().clamp_min(1)
            else:
                raw["physical_root_3d"] = root_loss.mean()
        if self.config.effective.get("surface_markers_3d",0) > 0:
            raw["surface_markers_3d"] = measured_surface_loss(output,batch)
        if self.config.effective["marker_keypoints_2d"] > 0:
            raw["marker_keypoints_2d"] = self.model.keypoint_2d_loss(
                output["pred_marker_keypoints_2d"], batch["marker_keypoints_2d"]
            )
        if self.config.effective["marker_keypoints_3d"] > 0:
            raw["marker_keypoints_3d"] = self.model.keypoint_3d_loss(
                output["pred_marker_keypoints_3d"], batch["marker_keypoints_3d"], pelvis_id=0
            )
        if self.config.effective["marker_root_3d"] > 0:
            raw["marker_root_3d"] = F.smooth_l1_loss(
                output["pred_marker_root_camera_m"], batch["marker_root_camera_m"], beta=0.02
            )
        keypoint_fields = {
            "keypoints_2d": ("pred_keypoints_2d", "keypoints_2d", False),
            "keypoints_3d": ("pred_keypoints_3d", "keypoints_3d", True),
            "coarse_keypoints_2d": (
                "coarse_keypoints_2d", "keypoints_2d", False
            ),
            "coarse_keypoints_3d": (
                "coarse_keypoints_3d", "keypoints_3d", True
            ),
        }
        for loss_name, (prediction_name, target_name, is_3d) in keypoint_fields.items():
            if self.config.effective[loss_name] <= 0:
                continue
            if is_3d:
                raw[loss_name] = self.model.keypoint_3d_loss(
                    output[prediction_name], batch[target_name], pelvis_id=0
                )
            else:
                raw[loss_name] = self.model.keypoint_2d_loss(
                    output[prediction_name], batch[target_name]
                )
        parameter_names = ("global_orient", "hand_pose", "betas")
        if any(self.config.effective[name] > 0 for name in parameter_names):
            pred_params = output["pred_mano_params"]
            gt_params = batch["mano_params"]
            masks = batch["has_mano_params"]
            batch_size = anchor.shape[0]
            for name in parameter_names:
                if self.config.effective[name] <= 0:
                    continue
                raw[name] = self.model.mano_parameter_loss(
                    pred_params[name].reshape(batch_size, -1),
                    gt_params[name].reshape(batch_size, -1),
                    masks[name],
                )
        weighted = {
            name: raw[name] * self.config.effective[name]
            for name in UNIFIED_LOSS_NAMES
        }
        total = sum(weighted.values(), zero)
        if not torch.isfinite(total):
            raise ValueError("Loss is NaN or Inf")
        detached = {"loss": total.detach()}
        for name in UNIFIED_LOSS_NAMES:
            detached[f"loss_{name}"] = raw[name].detach()
            detached[f"weighted_{name}"] = weighted[name].detach()
        output["losses"] = detached
        return total, detached


class GloveLossComputer:
    """Compose the project's official loss primitives with configured weights."""

    def __init__(self, model: torch.nn.Module, config: GloveLossConfig) -> None:
        self.model = model
        self.config = config

    def __call__(
        self,
        batch: Mapping[str, Any],
        output: dict[str, Any],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        device = output["pred_keypoints_3d"].device
        dtype = output["pred_keypoints_3d"].dtype
        zero = torch.zeros((), device=device, dtype=dtype)
        raw: dict[str, torch.Tensor] = {name: zero for name in LOSS_NAMES}
        if self.config.effective["keypoints_2d"] > 0:
            raw["keypoints_2d"] = self.model.keypoint_2d_loss(
                output["pred_keypoints_2d"], batch["keypoints_2d"]
            )
        if self.config.effective["keypoints_3d"] > 0:
            raw["keypoints_3d"] = self.model.keypoint_3d_loss(
                output["pred_keypoints_3d"], batch["keypoints_3d"], pelvis_id=0
            )
        if self.config.effective["coarse_keypoints_2d"] > 0:
            raw["coarse_keypoints_2d"] = self.model.keypoint_2d_loss(
                output["coarse_keypoints_2d"], batch["keypoints_2d"]
            )
        if self.config.effective["coarse_keypoints_3d"] > 0:
            raw["coarse_keypoints_3d"] = self.model.keypoint_3d_loss(
                output["coarse_keypoints_3d"], batch["keypoints_3d"], pelvis_id=0
            )
        pred_params = output["pred_mano_params"]
        gt_params = batch["mano_params"]
        masks = batch["has_mano_params"]
        batch_size = output["pred_keypoints_3d"].shape[0]
        for name in ("global_orient", "hand_pose", "betas"):
            if self.config.effective[name] <= 0:
                continue
            if name not in pred_params or name not in gt_params or name not in masks:
                raise KeyError(f"MANO supervision is enabled but {name} is missing")
            raw[name] = self.model.mano_parameter_loss(
                pred_params[name].reshape(batch_size, -1),
                gt_params[name].reshape(batch_size, -1),
                masks[name],
            )
        weighted = {
            name: raw[name] * self.config.effective[name] for name in LOSS_NAMES
        }
        total = sum(weighted.values(), zero)
        if not torch.isfinite(total):
            raise ValueError("Loss is NaN or Inf")
        detached = {"loss": total.detach()}
        for name in LOSS_NAMES:
            detached[f"loss_{name}"] = raw[name].detach()
            detached[f"weighted_{name}"] = weighted[name].detach()
        output["losses"] = detached
        return total, detached


def loss_config_json(config: GloveLossConfig) -> str:
    """Canonical semantic representation used by resume signatures."""
    return json.dumps(
        {"requested": config.requested, "effective": config.effective},
        sort_keys=True,
        separators=(",", ":"),
    )
