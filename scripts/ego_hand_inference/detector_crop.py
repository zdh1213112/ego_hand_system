"""Convert a glove detector box to the native mesh box used during WiLoR training."""

import json
from pathlib import Path

import numpy as np

from ego_hand_inference.orbbec_io import sha256_file


def load_crop_calibration(path: Path, detector: Path) -> dict[str, np.ndarray]:
    with path.open() as stream:
        data = json.load(stream)
    if data.get("schema") != "physical_kb_detector_crop_v1":
        raise ValueError(f"Unsupported detector crop calibration in {path}")
    if data.get("detector_sha256") != sha256_file(detector):
        raise ValueError("The crop calibration was fitted for a different detector")
    coefficients = {k: np.asarray(v, dtype=float) for k, v in data["coefficients"].items()}
    expected = {f"camera{i}/{side}" for i in range(4) for side in (0, 1)}
    if coefficients.keys() != expected or any(
        matrix.shape != (4, 4) or not np.isfinite(matrix).all() for matrix in coefficients.values()
    ):
        raise ValueError("Expected four finite crop coefficients per output for each camera and side")
    return coefficients


def box_features(box: np.ndarray, image_size: np.ndarray) -> np.ndarray:
    size = box[2:] - box[:2]
    center = (box[2:] + box[:2]) / 2
    return np.r_[1., np.clip(np.log(size[0] / size[1]), -1.6, 1.6), 2 * center / image_size - 1]


def calibrated_box(coefficients, box, camera, side, image_size) -> np.ndarray:
    box = np.asarray(box, dtype=float)
    image_size = np.asarray(image_size, dtype=float)
    size = box[2:] - box[:2]
    correction = box_features(box, image_size) @ coefficients[f"{camera}/{side}"]
    shift = np.clip(correction[:2], -.5, .5) * size
    half_size = .5 * size * np.exp(np.clip(correction[2:], -.9, .9))
    center = (box[2:] + box[:2]) * .5 + shift
    low = np.clip(center - half_size, 0, image_size - 2)
    high = np.clip(center + half_size, low + 1, image_size - 1)
    return np.r_[low, high].astype(np.float32)
