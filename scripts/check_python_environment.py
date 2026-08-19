#!/usr/bin/env python3
"""Validate the complete EGO Hand Python runtime without loading model assets."""

from __future__ import annotations

import builtins
import importlib
from importlib import metadata
import inspect
import sys

import numpy as np


MODULES = (
    "av", "cv2", "mediapipe", "mcap", "torch", "torchvision", "scipy", "trimesh",
    "skimage", "pyrender", "pytorch_lightning", "smplx", "yacs", "timm", "einops",
    "xtcocotools", "pandas", "hydra", "pyrootutils", "rich", "webdataset", "gradio",
    "ultralytics", "polars", "psutil", "dill",
)

EXPECTED_ULTRALYTICS_VERSION = "8.4.56"


def import_chumpy_compatibly() -> None:
    aliases = {
        "bool": np.bool_, "int": builtins.int, "float": builtins.float,
        "complex": builtins.complex, "object": builtins.object,
        "unicode": builtins.str, "str": builtins.str,
    }
    for name, value in aliases.items():
        if name not in np.__dict__:
            setattr(np, name, value)
    if not hasattr(inspect, "getargspec"):
        inspect.getargspec = inspect.getfullargspec  # type: ignore[attr-defined]
    importlib.import_module("chumpy")


def main() -> int:
    failures = []
    for name in MODULES:
        try:
            importlib.import_module(name)
        except Exception as error:
            failures.append(f"{name}: {error}")
    try:
        import_chumpy_compatibly()
    except Exception as error:
        failures.append(f"chumpy (with compatibility shim): {error}")

    installed = {dist.metadata["Name"].lower() for dist in metadata.distributions()}
    if "opencv-python" in installed:
        failures.append(
            "opencv-python is installed alongside opencv-contrib-python; remove opencv-python"
        )
    if "opencv-contrib-python" not in installed:
        failures.append("opencv-contrib-python is not installed")

    try:
        ultralytics_version = metadata.version("ultralytics")
        if ultralytics_version != EXPECTED_ULTRALYTICS_VERSION:
            failures.append(
                "ultralytics "
                f"{ultralytics_version} is installed; detector.pt requires "
                f"{EXPECTED_ULTRALYTICS_VERSION}"
            )
        block = importlib.import_module("ultralytics.nn.modules.block")
        if not hasattr(block, "C3k2"):
            failures.append("ultralytics does not provide C3k2 required by detector.pt")
    except Exception as error:
        failures.append(f"ultralytics checkpoint compatibility: {error}")

    if failures:
        print("Python environment validation failed:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1

    torch = importlib.import_module("torch")
    print("Python environment is ready")
    print(f"Python: {sys.version.split()[0]}")
    print(f"NumPy: {np.__version__}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Ultralytics: {metadata.version('ultralytics')}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
