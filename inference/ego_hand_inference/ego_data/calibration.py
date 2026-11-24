"""Orbbec calibration and camera-to-reference transforms."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Literal

import numpy as np
import yaml


CameraModel = Literal["KB", "DS", "PINHOLE"]


def _array(value, shape: tuple[int, ...], name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite array with shape {shape}, got {result.shape}")
    return result


@dataclass(frozen=True)
class CameraCalibration:
    camera_id: str
    frame_id: str
    model: CameraModel
    image_size: tuple[int, int]
    K: np.ndarray
    distortion: np.ndarray
    T_base_camera: np.ndarray

    def __post_init__(self) -> None:
        model = str(self.model).upper()
        if model not in ("KB", "DS", "PINHOLE"):
            raise ValueError(f"unsupported camera model: {self.model}")
        width, height = (int(self.image_size[0]), int(self.image_size[1]))
        if width <= 0 or height <= 0:
            raise ValueError(f"invalid image size: {self.image_size}")
        K = _array(self.K, (3, 3), "K")
        T = _array(self.T_base_camera, (4, 4), "T_base_camera")
        expected = {"KB": 4, "DS": 6, "PINHOLE": 0}[model]
        distortion = np.asarray(self.distortion, dtype=np.float64).reshape(-1)
        if distortion.size != expected or not np.all(np.isfinite(distortion)):
            raise ValueError(f"{model} distortion must contain {expected} finite values")
        if K[0, 0] <= 0 or K[1, 1] <= 0:
            raise ValueError("camera focal lengths must be positive")
        if not np.allclose(T[3], (0.0, 0.0, 0.0, 1.0), atol=1e-9):
            raise ValueError("T_base_camera must be homogeneous")
        rotation = T[:3, :3]
        if not np.allclose(rotation @ rotation.T, np.eye(3), atol=2e-4):
            raise ValueError("T_base_camera rotation is not orthogonal")
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "image_size", (width, height))
        object.__setattr__(self, "K", K)
        object.__setattr__(self, "distortion", distortion)
        object.__setattr__(self, "T_base_camera", T)

    @property
    def fx(self) -> float:
        return float(self.K[0, 0])

    @property
    def fy(self) -> float:
        return float(self.K[1, 1])

    @property
    def cx(self) -> float:
        return float(self.K[0, 2])

    @property
    def cy(self) -> float:
        return float(self.K[1, 2])

    def to_dict(self) -> dict:
        return {
            "schema_version": 1,
            "camera_id": self.camera_id,
            "frame_id": self.frame_id,
            "model": self.model,
            "image_size": list(self.image_size),
            "K": self.K.tolist(),
            "distortion": self.distortion.tolist(),
            "T_base_camera": self.T_base_camera.tolist(),
        }

    @classmethod
    def from_dict(cls, value: dict) -> "CameraCalibration":
        return cls(
            camera_id=str(value["camera_id"]),
            frame_id=str(value.get("frame_id", value["camera_id"])),
            model=str(value["model"]).upper(),
            image_size=tuple(value["image_size"]),
            K=value["K"],
            distortion=value["distortion"],
            T_base_camera=value.get("T_base_camera", np.eye(4)),
        )

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "CameraCalibration":
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


def cameras_from_ego_yaml(path: Path) -> dict[str, CameraCalibration]:
    """Load every KB camera in an EGO export, indexed by its source ID."""
    calibration = yaml.safe_load(path.read_text(encoding="utf-8"))
    cameras = calibration["cameras"]
    if not cameras:
        raise ValueError("EGO calibration has no cameras")
    result: dict[str, CameraCalibration] = {}
    for camera in cameras:
        if str(camera["distortion_model"]).upper() != "KB":
            raise ValueError("EGO loader only supports KB cameras")
        intr = camera["intrinsics"]
        dist = camera["distortion"]
        K = np.asarray([
            [intr["fx"], 0.0, intr["cx"]],
            [0.0, intr["fy"], intr["cy"]],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
        D = np.asarray([dist["k1"], dist["k2"], dist["k3"], dist["k4"]])
        # EGO YAML stores reference -> camera, while the common contract stores
        # camera -> base/reference.
        rotation_reference_camera = np.asarray(
            camera["extrinsics"]["rotation"], dtype=np.float64
        )
        translation_reference_camera = np.asarray(
            camera["extrinsics"]["translation"], dtype=np.float64
        ) * 1e-3
        transform = np.eye(4)
        transform[:3, :3] = rotation_reference_camera.T
        transform[:3, 3] = -rotation_reference_camera.T @ translation_reference_camera
        source_id = str(camera.get("id", camera.get("name", f"camera{len(result)}")))
        if source_id in result:
            raise ValueError(f"duplicate EGO camera ID: {source_id}")
        result[source_id] = CameraCalibration(
            camera_id=source_id,
            frame_id=str(camera.get("name", "")),
            model="KB",
            image_size=(int(camera["image_width"]), int(camera["image_height"])),
            K=K,
            distortion=D,
            T_base_camera=transform,
        )
    return result


def load_orbbec_four_calibrations(source):
    from dataclasses import replace
    import numpy as np
    CAMERAS = tuple(f"camera{i}" for i in range(4))
    if source.is_dir():
        result = {}
        for camera in CAMERAS:
            path = source / f'{camera}.json'
            payload = json.loads(path.read_text())
            if 'T_base_camera' not in payload:
                raise ValueError(f'{path}: explicit T_base_camera extrinsics are required')
            result[camera] = CameraCalibration.from_dict(payload)
            if result[camera].camera_id != camera:
                raise ValueError(f'{path}: camera_id must be {camera}')
    else:
        original = cameras_from_ego_yaml(source)
        if set(original) != {f'cam_{i}' for i in range(4)}:
            raise ValueError('Orbbec YAML must contain exactly cam_0..cam_3')
        result = {camera: replace(original[f'cam_{i}'], camera_id=camera) for i, camera in enumerate(CAMERAS)}
    if len({c.image_size for c in result.values()}) != 1:
        raise ValueError('the four camera image sizes must match')
    centers = np.stack([c.T_base_camera[:3, 3] for c in result.values()])
    if max(np.linalg.norm(x-y) for x in centers for y in centers) < .005:
        raise ValueError('four-camera extrinsics have no usable baseline')
    return result
