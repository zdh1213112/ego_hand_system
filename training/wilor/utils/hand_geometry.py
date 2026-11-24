"""Coordinate transforms shared by relative-pose datasets and losses."""

from __future__ import annotations

import numpy as np
import torch


WRIST = 0
INDEX_MCP = 5
MIDDLE_MCP = 9
PINKY_MCP = 17


def palm_scale_normalize(
    joints: torch.Tensor,
    *,
    eps: float = 1e-8,
    validate: bool = True,
) -> torch.Tensor:
    """Map right-hand joints to a translation/rotation/scale-free palm frame.

    The input may have any leading dimensions but must end in ``[21, 3]``.
    The palm x axis points from pinky MCP to index MCP, y points from wrist
    towards middle MCP after Gram-Schmidt orthogonalization, and z completes a
    right-handed frame.  Palm width (index MCP to pinky MCP) is the scale.
    """
    if joints.ndim < 2 or tuple(joints.shape[-2:]) != (21, 3):
        raise ValueError(
            f"hand joints must end in [21, 3], got {tuple(joints.shape)}"
        )
    if not joints.is_floating_point():
        joints = joints.float()
    if validate and not bool(torch.isfinite(joints).all()):
        raise ValueError("hand joints contain NaN or Inf")

    wrist = joints[..., WRIST, :]
    lateral = joints[..., INDEX_MCP, :] - joints[..., PINKY_MCP, :]
    scale = torch.linalg.vector_norm(lateral, dim=-1, keepdim=True)
    x_axis = lateral / scale.clamp_min(eps)

    forward = joints[..., MIDDLE_MCP, :] - wrist
    forward = forward - (forward * x_axis).sum(dim=-1, keepdim=True) * x_axis
    forward_norm = torch.linalg.vector_norm(forward, dim=-1, keepdim=True)
    y_axis = forward / forward_norm.clamp_min(eps)
    z_axis = torch.linalg.cross(x_axis, y_axis, dim=-1)

    if validate:
        degenerate = (scale <= eps) | (forward_norm <= eps)
        if bool(degenerate.any()):
            raise ValueError("cannot construct a non-degenerate palm frame")

    basis = torch.stack((x_axis, y_axis, z_axis), dim=-1)
    centered = joints - wrist.unsqueeze(-2)
    local = torch.matmul(centered, basis)
    return local / scale.unsqueeze(-2).clamp_min(eps)


def palm_scale_normalize_numpy(
    joints: np.ndarray,
    *,
    eps: float = 1e-8,
) -> np.ndarray:
    """NumPy-facing wrapper with exactly the same semantics as the torch path."""
    array = np.asarray(joints)
    tensor = torch.from_numpy(array.astype(np.float32, copy=False))
    return palm_scale_normalize(tensor, eps=eps).numpy()
