"""Camera calibration types shared by MCAP, rectification, and rendering."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Literal

import numpy as np
import yaml


CameraModel = Literal["KB", "DS"]


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
        if model not in ("KB", "DS"):
            raise ValueError(f"unsupported camera model: {self.model}")
        width, height = (int(self.image_size[0]), int(self.image_size[1]))
        if width <= 0 or height <= 0:
            raise ValueError(f"invalid image size: {self.image_size}")
        K = _array(self.K, (3, 3), "K")
        T = _array(self.T_base_camera, (4, 4), "T_base_camera")
        expected = 4 if model == "KB" else 6
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


@dataclass(frozen=True)
class StereoCalibration:
    left: CameraCalibration
    right: CameraCalibration
    R_left_to_right: np.ndarray
    t_left_to_right_m: np.ndarray

    def __post_init__(self) -> None:
        if self.left.image_size != self.right.image_size:
            raise ValueError("left and right calibration resolutions differ")
        R = _array(self.R_left_to_right, (3, 3), "R_left_to_right")
        t = np.asarray(self.t_left_to_right_m, dtype=np.float64).reshape(-1)
        if t.shape != (3,) or not np.all(np.isfinite(t)):
            raise ValueError("t_left_to_right_m must contain three finite values")
        if not np.allclose(R @ R.T, np.eye(3), atol=2e-4) or np.linalg.det(R) < 0.999:
            raise ValueError("R_left_to_right is not a rotation matrix")
        if not 0.005 < np.linalg.norm(t) < 1.0:
            raise ValueError(f"implausible stereo baseline: {np.linalg.norm(t):.6f} m")
        object.__setattr__(self, "R_left_to_right", R)
        object.__setattr__(self, "t_left_to_right_m", t)

    @property
    def baseline_m(self) -> float:
        return float(np.linalg.norm(self.t_left_to_right_m))

    @classmethod
    def from_cameras(cls, left: CameraCalibration, right: CameraCalibration) -> "StereoCalibration":
        R_base_left = left.T_base_camera[:3, :3]
        R_base_right = right.T_base_camera[:3, :3]
        t_base_left = left.T_base_camera[:3, 3]
        t_base_right = right.T_base_camera[:3, 3]
        R = R_base_right.T @ R_base_left
        t = R_base_right.T @ (t_base_left - t_base_right)
        return cls(left, right, R, t)

    def to_dict(self) -> dict:
        return {
            "schema_version": 1,
            "left_camera": self.left.camera_id,
            "right_camera": self.right.camera_id,
            "R_left_to_right": self.R_left_to_right.tolist(),
            "t_left_to_right_m": self.t_left_to_right_m.tolist(),
            "baseline_m": self.baseline_m,
            "coordinate_system": "OpenCV optical frame: +x right, +y down, +z forward",
        }


def quaternion_transform(values) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size != 7:
        raise ValueError("T_b_c must contain [tx, ty, tz, qx, qy, qz, qw]")
    translation = values[:3]
    quaternion = values[3:]
    norm = np.linalg.norm(quaternion)
    if norm <= 1e-12:
        raise ValueError("camera quaternion has zero length")
    qx, qy, qz, qw = quaternion / norm
    rotation = np.asarray([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ])
    result = np.eye(4)
    result[:3, :3] = rotation
    result[:3, 3] = translation
    return result


def camera_from_genrobot_message(camera_id: str, message) -> CameraCalibration:
    model_name = str(message.distortion_model).strip().lower()
    if model_name not in ("ds", "double_sphere", "double sphere"):
        raise ValueError(f"GEN camera {camera_id} has unsupported distortion model {model_name!r}")
    D = np.asarray(message.D, dtype=np.float64)
    K = np.asarray(message.K, dtype=np.float64)
    if D.size < 6 or K.size != 9:
        raise ValueError(f"GEN camera {camera_id} has incomplete K/D calibration")
    # GEN publishes [fx, fy, cx, cy, xi, alpha] in D.  Preserve it verbatim.
    return CameraCalibration(
        camera_id=camera_id,
        frame_id=str(message.frame_id),
        model="DS",
        image_size=(int(message.width), int(message.height)),
        K=K.reshape(3, 3),
        distortion=D[:6],
        T_base_camera=quaternion_transform(message.T_b_c),
    )


def stereo_from_ego_yaml(path: Path) -> StereoCalibration:
    calibration = yaml.safe_load(path.read_text(encoding="utf-8"))
    cameras = calibration["cameras"]
    if len(cameras) != 2:
        raise ValueError("expected two cameras in EGO calibration")
    result = []
    for camera in cameras:
        if str(camera["distortion_model"]).upper() != "KB":
            raise ValueError("legacy EGO loader only supports KB cameras")
        intr = camera["intrinsics"]
        dist = camera["distortion"]
        K = np.asarray([
            [intr["fx"], 0.0, intr["cx"]],
            [0.0, intr["fy"], intr["cy"]],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
        D = np.asarray([dist["k1"], dist["k2"], dist["k3"], dist["k4"]])
        # EGO YAML stores reference -> camera, while the common contract stores
        # camera -> base/reference (the same convention as GEN T_b_c).
        rotation_reference_camera = np.asarray(
            camera["extrinsics"]["rotation"], dtype=np.float64
        )
        translation_reference_camera = np.asarray(
            camera["extrinsics"]["translation"], dtype=np.float64
        ) * 1e-3
        transform = np.eye(4)
        transform[:3, :3] = rotation_reference_camera.T
        transform[:3, 3] = -rotation_reference_camera.T @ translation_reference_camera
        result.append(CameraCalibration(
            camera_id=str(camera.get("id", camera.get("name", f"camera{len(result)}"))),
            frame_id=str(camera.get("name", "")),
            model="KB",
            image_size=(int(camera["image_width"]), int(camera["image_height"])),
            K=K,
            distortion=D,
            T_base_camera=transform,
        ))
    return StereoCalibration.from_cameras(result[0], result[1])
