#!/usr/bin/env python3
"""Read-only environment and optional real GEN MCAP H264 smoke check."""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path
import sys


REQUIRED = (
    "av", "cv2", "mediapipe", "numpy", "yaml", "mcap", "mcap_protobuf",
    "google.protobuf", "zstandard", "lz4.frame", "torch", "scipy", "trimesh",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mcap", type=Path, help="optionally decode one real camera frame")
    parser.add_argument("--camera", default="camera2")
    return parser.parse_args()


def main() -> int:
    modules = {}
    for name in REQUIRED:
        module = importlib.import_module(name)
        modules[name] = module
        print(f"OK {name}: {getattr(module, '__version__', 'available')}")
    av = modules["av"]
    cv2 = modules["cv2"]
    torch = modules["torch"]
    print(f"OK H264 decoder: {av.CodecContext.create('h264', 'r').name}")
    if not hasattr(cv2.fisheye, "stereoRectify") or not hasattr(cv2.fisheye, "initUndistortRectifyMap"):
        raise RuntimeError("OpenCV fisheye stereo APIs are missing")
    print("OK OpenCV fisheye APIs")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    value = float(torch.ones(16, device="cuda").sum().cpu())
    print(f"OK CUDA: {torch.cuda.get_device_name(0)} (smoke={value})")
    args = parse_args()
    if args.mcap:
        from ego_data.genrobot_mcap import decode_stereo_mcap
        decoded = []
        def receive(frame):
            decoded.append(frame)
        decode_stereo_mcap(args.mcap, (args.camera, "camera3" if args.camera != "camera3" else "camera2"), receive, 1)
        match = next((frame for frame in decoded if frame.camera_id == args.camera), None)
        if match is None:
            raise RuntimeError(f"did not decode a {args.camera} frame")
        print(f"OK real MCAP H264: {args.camera} {match.image.shape[1]}x{match.image.shape[0]}")
    print("GEN environment is ready")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(1)
