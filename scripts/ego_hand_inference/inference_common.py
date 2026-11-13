"""WiLoR device selection and hand detection."""

from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false")
    return device


def load_detector(checkpoint_path: Path, device: torch.device):
    from ultralytics import YOLO

    original_torch_load = torch.load

    def trusted_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_torch_load(*args, **kwargs)

    torch.load = trusted_load
    try:
        return YOLO(str(checkpoint_path)).to(device)
    finally:
        torch.load = original_torch_load


def _top_detections_per_class(
    confidences: np.ndarray,
    detector_sides: np.ndarray,
    limit: int,
) -> np.ndarray:
    if limit == 0:
        return np.arange(len(confidences), dtype=np.int64)
    selected = []
    for side in np.unique(detector_sides):
        candidates = np.flatnonzero(detector_sides == side)
        ranked = candidates[np.argsort(-confidences[candidates], kind="stable")]
        selected.extend(ranked[:limit].tolist())
    return np.asarray(sorted(selected), dtype=np.int64)
