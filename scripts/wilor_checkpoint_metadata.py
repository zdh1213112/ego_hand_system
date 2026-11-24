"""Record model identity without importing the GPU inference runtime."""

import hashlib
import os
from pathlib import Path


def checkpoint_metadata():
    root = Path(__file__).resolve().parents[1]
    checkpoint = Path(os.environ.get("EGO_WILOR_CHECKPOINT") or root / "models/wilor/wilor_final.ckpt").resolve()
    source = Path(os.environ.get("EGO_WILOR_ROOT") or root / "third_party/WiLoR").resolve()
    stat = checkpoint.stat()
    result = {
        "name": "wilor", "checkpoint": str(checkpoint), "wilor_root": str(source),
        "checkpoint_size_bytes": stat.st_size, "checkpoint_mtime_ns": stat.st_mtime_ns,
    }
    if os.environ.get("EGO_WILOR_ADAPTER"):
        adapter = Path(os.environ["EGO_WILOR_ADAPTER"]).resolve()
        result["adapter"] = str(adapter)
        result["adapter_sha256"] = hashlib.sha256(adapter.read_bytes()).hexdigest()
    return result
