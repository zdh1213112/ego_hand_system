"""Stable, profile-gated loss interface for glove-domain training."""

from __future__ import annotations

import json
import math
from collections import defaultdict
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
FINGERTIPS = (4, 8, 12, 16, 20)
BONE_EDGES = tuple((0 if joint % 4 == 1 else joint - 1, joint) for joint in range(1, 21))
V5_LOSS_SOURCES = {
    "coarse_physical_root_3d": "keypoints_3d",
    "fingertips_3d": "keypoints_3d",
    "fingertips_2d": "keypoints_2d",
    "bone_direction_3d": "keypoints_3d",
    "anatomy_bend": "keypoints_3d",
    "multiview_pose_consistency": "keypoints_3d",
}
OPTIONAL_UNIFIED_LOSS_NAMES = (
    "physical_root_3d", "marker_keypoints_2d", "marker_keypoints_3d", "marker_root_3d", "surface_markers_3d",
    *V5_LOSS_SOURCES,
)
UNIFIED_LOSS_NAMES = ("local_keypoints_3d",) + LOSS_NAMES + OPTIONAL_UNIFIED_LOSS_NAMES
AVAILABILITY_SOURCE = {
    **V5_LOSS_SOURCES,
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
        if name in V5_LOSS_SOURCES and name not in losses:
            continue
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
    for name in requested:
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


def _weighted_mean(values, weights):
    weights = weights.to(values)
    return (torch.where(weights > 0, values, 0.) * weights).sum() / weights.sum().clamp_min(1e-8)


def _root_loss(prediction, batch):
    errors = F.smooth_l1_loss(prediction, batch["camera_root_m"], beta=0.02, reduction="none")
    mask = batch.get("geometry_supervised")
    if mask is None:
        return errors.mean()
    return (errors.mean(1) * mask).sum() / mask.sum().clamp_min(1)


def _fingertip_loss(prediction, target, wrist_relative):
    points, confidence = target[..., :-1], target[..., -1]
    if wrist_relative:
        prediction = prediction - prediction[:, :1]
        points = points - points[:, :1]
        confidence = torch.minimum(confidence, confidence[:, :1])
    difference = (prediction[:, FINGERTIPS].float() - points[:, FINGERTIPS]).abs().mean(-1)
    return _weighted_mean(difference, confidence[:, FINGERTIPS])


def _bone_direction_loss(prediction, target):
    starts, ends = zip(*BONE_EDGES)
    predicted = prediction[:, ends].float() - prediction[:, starts].float()
    actual = target[:, ends, :3] - target[:, starts, :3]
    cosine = (F.normalize(predicted, dim=-1, eps=1e-6) *
              F.normalize(actual, dim=-1, eps=1e-6)).sum(-1)
    confidence = torch.minimum(target[:, starts, 3], target[:, ends, 3])
    confidence = confidence * (actual.norm(dim=-1) > 1e-6)
    return _weighted_mean(1 - cosine.clamp(-1, 1), confidence)


def anatomy_bend(joints):
    """Mean PIP/DIP hyperextension and opposing flexion in canonical right-hand space."""
    joints = joints.float()
    palm_normal = F.normalize(torch.cross(joints[:, 5] - joints[:, 0],
                                          joints[:, 17] - joints[:, 0], dim=-1), dim=-1, eps=1e-6)
    palm_axis = F.normalize(joints[:, 9] - joints[:, 0], dim=-1, eps=1e-6)
    axis = F.normalize(torch.cross(palm_axis, palm_normal, dim=-1), dim=-1, eps=1e-6)
    fingers = joints[:, 5:].reshape(-1, 4, 4, 3)
    bones = F.normalize(fingers.diff(dim=2), dim=-1, eps=1e-6)
    first, second = bones[:, :, :-1], bones[:, :, 1:]
    sin = (torch.cross(first, second, dim=-1) * axis[:, None, None]).sum(-1)
    cos = (first * second).sum(-1)
    # atan2(0, 0) has an undefined backward on a collapsed/orthogonal bone.
    valid = sin.square() + cos.square() > 1e-12
    flex = torch.atan2(torch.where(valid, sin, 0.), torch.where(valid, cos, 1.))
    pip, dip = flex.unbind(-1)
    hyperextension = F.relu(-flex - math.radians(5.)).square().sum(-1)
    return (hyperextension + F.relu(-pip * dip - 0.02)).mean()


def _multiview_pose_consistency(params, batch):
    poses, shapes = params["hand_pose"].float(), params["betas"].float()
    groups = defaultdict(list)
    for index, (source, sync, side) in enumerate(zip(batch["source_group"], batch["sync_index"].tolist(),
                                                    batch["side"].tolist())):
        if sync >= 0:
            groups[source, sync, side].append(index)
    penalties = []
    for indices in groups.values():
        if len(indices) < 2:
            continue
        if "camera" in batch and len({batch["camera"][i] for i in indices}) < 2:
            continue
        pose, shape = poses[indices], shapes[indices]
        penalties.append((pose - pose.mean(0)).square().mean() + (shape - shape.mean(0)).square().mean())
    return torch.stack(penalties).mean() if penalties else (poses.sum() + shapes.sum()) * 0.


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
            name: zero for name in self.config.effective
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
            raw["physical_root_3d"] = _root_loss(output["pred_root_camera_m"], batch)
        active = self.config.effective
        if active.get("coarse_physical_root_3d"):
            raw["coarse_physical_root_3d"] = _root_loss(output["coarse_root_camera_m"], batch)
        for dim in ("2d", "3d"):
            key = f"fingertips_{dim}"
            if active.get(key):
                raw[key] = _fingertip_loss(output[f"pred_keypoints_{dim}"], batch[f"keypoints_{dim}"], dim == "3d")
        if active.get("bone_direction_3d"):
            raw["bone_direction_3d"] = _bone_direction_loss(output["pred_keypoints_3d"], batch["keypoints_3d"])
        if active.get("anatomy_bend"):
            raw["anatomy_bend"] = anatomy_bend(output["pred_keypoints_3d"])
        if active.get("multiview_pose_consistency"):
            raw["multiview_pose_consistency"] = _multiview_pose_consistency(output["pred_mano_params"], batch)
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
            for name in self.config.effective
        }
        total = sum(weighted.values(), zero)
        if not torch.isfinite(total):
            raise ValueError("Loss is NaN or Inf")
        detached = {"loss": total.detach()}
        for name in self.config.effective:
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
