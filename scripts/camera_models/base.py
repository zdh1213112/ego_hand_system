"""Shared rectification result contract."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

import numpy as np

from ego_data.calibration import StereoCalibration


@dataclass(frozen=True)
class RectificationOptions:
    output_size: tuple[int, int] | None = None
    balance: float = 0.0
    focal_scale: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.balance <= 1.0:
            raise ValueError("balance must be in [0, 1]")
        if self.focal_scale <= 0:
            raise ValueError("focal_scale must be positive")
        if self.output_size is not None and (self.output_size[0] <= 0 or self.output_size[1] <= 0):
            raise ValueError("output_size must be positive")


@dataclass
class RectificationResult:
    image_size: tuple[int, int]
    map_left_x: np.ndarray
    map_left_y: np.ndarray
    map_right_x: np.ndarray
    map_right_y: np.ndarray
    valid_left: np.ndarray
    valid_right: np.ndarray
    R1: np.ndarray
    R2: np.ndarray
    P1: np.ndarray
    P2: np.ndarray
    Q: np.ndarray
    model: str

    def __post_init__(self) -> None:
        width, height = self.image_size
        for name in ("map_left_x", "map_left_y", "map_right_x", "map_right_y", "valid_left", "valid_right"):
            value = np.asarray(getattr(self, name))
            if value.shape != (height, width):
                raise ValueError(f"{name} has shape {value.shape}, expected {(height, width)}")
            setattr(self, name, value)
        for name, shape in (("R1", (3, 3)), ("R2", (3, 3)), ("P1", (3, 4)), ("P2", (3, 4)), ("Q", (4, 4))):
            value = np.asarray(getattr(self, name), dtype=np.float64)
            if value.shape != shape:
                raise ValueError(f"{name} has shape {value.shape}, expected {shape}")
            setattr(self, name, value)

    def legacy_dict(self, calibration_serial: str = "unified") -> dict:
        return {
            "image_size": self.image_size,
            "map_left_x": self.map_left_x,
            "map_left_y": self.map_left_y,
            "map_right_x": self.map_right_x,
            "map_right_y": self.map_right_y,
            "r1": self.R1,
            "r2": self.R2,
            "p1": self.P1,
            "p2": self.P2,
            "q": self.Q,
            "calibration_serial": calibration_serial,
            "model": self.model,
        }

    def save_npz(self, path) -> None:
        np.savez_compressed(
            path, R1=self.R1, R2=self.R2, P1=self.P1, P2=self.P2, Q=self.Q,
            map_left_x=self.map_left_x, map_left_y=self.map_left_y,
            map_right_x=self.map_right_x, map_right_y=self.map_right_y,
            valid_left=self.valid_left, valid_right=self.valid_right,
        )


def rectification_hash(calibration: StereoCalibration, options: RectificationOptions) -> str:
    value = {
        "left": calibration.left.to_dict(),
        "right": calibration.right.to_dict(),
        "R": calibration.R_left_to_right.tolist(),
        "t": calibration.t_left_to_right_m.tolist(),
        "options": {
            "output_size": options.output_size,
            "balance": options.balance,
            "focal_scale": options.focal_scale,
        },
    }
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()
