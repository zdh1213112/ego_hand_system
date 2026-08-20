"""Coordinate transforms for the WiLoR right-hand-only MANO convention.

WiLoR decodes every crop with ``MANO_RIGHT.pkl``.  A physical left hand is
represented in a canonical right-hand coordinate system by reflecting the
camera-space X axis.  Rendering reverses that reflection and the triangle
winding exactly once.
"""

from __future__ import annotations

import numpy as np


WILOR_RIGHT_CANONICAL = "wilor_right_canonical_v1"
MIRROR_X = np.diag((-1.0, 1.0, 1.0)).astype(np.float32)


def is_right_hand(handedness: str | int | bool) -> bool:
    if isinstance(handedness, str):
        value = handedness.strip().lower()
        if value not in ("left", "right"):
            raise ValueError(f"unsupported handedness: {handedness!r}")
        return value == "right"
    return bool(int(handedness))


def mirror_left_points(points: np.ndarray, handedness: str | int | bool) -> np.ndarray:
    """Convert points between physical and right-canonical space.

    Reflection is its own inverse, so the same operation canonicalizes a
    physical left hand and restores a canonical left track for visualization.
    """
    result = np.asarray(points).copy()
    if not is_right_hand(handedness):
        result[..., 0] *= -1.0
    return result


def canonical_projection_rotation(
    physical_rotation: np.ndarray, handedness: str | int | bool,
) -> np.ndarray:
    """Map right-canonical points into a physical camera projection frame."""
    rotation = np.asarray(physical_rotation)
    if rotation.shape != (3, 3):
        raise ValueError(f"rotation must be (3, 3), got {rotation.shape}")
    if is_right_hand(handedness):
        return rotation.copy()
    return (rotation @ MIRROR_X.astype(rotation.dtype, copy=False)).astype(
        rotation.dtype, copy=False
    )


def canonical_rectification_rotation(
    physical_rotation: np.ndarray, handedness: str | int | bool,
) -> np.ndarray:
    """Rotate a canonical mesh into an equally canonical rectified frame."""
    rotation = np.asarray(physical_rotation)
    if rotation.shape != (3, 3):
        raise ValueError(f"rotation must be (3, 3), got {rotation.shape}")
    if is_right_hand(handedness):
        return rotation.copy()
    mirror = MIRROR_X.astype(rotation.dtype, copy=False)
    return (mirror @ rotation @ mirror).astype(rotation.dtype, copy=False)


def physicalize_geometry(
    vertices: np.ndarray,
    joints: np.ndarray,
    faces: np.ndarray,
    handedness: str | int | bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert canonical geometry to physical camera space for visualization."""
    physical_vertices = mirror_left_points(vertices, handedness)
    physical_joints = mirror_left_points(joints, handedness)
    physical_faces = np.asarray(faces).copy()
    if not is_right_hand(handedness):
        physical_faces = physical_faces[..., (0, 2, 1)]
    return physical_vertices, physical_joints, physical_faces
