#!/usr/bin/env python3
"""Validate user-supplied licensed MANO model assets and the prepared input contract."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description="Check MANO assets before fitting.")
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument(
        "--mano-source", type=Path,
        default=Path(os.environ.get("MANO_SOURCE", "third_party/MANO")),
        help="external MANO source tree (default: MANO_SOURCE or third_party/MANO)",
    )
    parser.add_argument("--input", required=True, type=Path, help="mano_input.npz")
    parser.add_argument("--input-only", action="store_true", help="validate NPZ without MANO assets")
    args = parser.parse_args()

    with np.load(args.input) as data:
        required = {
            "positions_left_camera_m", "valid", "confidence", "track_ids", "handedness",
            "observed", "input_observed", "outlier_rejected", "interpolated",
            "bone_lengths_m", "pair_indices", "left_frame_indices", "right_frame_indices",
            "left_to_rectified_rotation", "projection_left_rectified",
            "projection_right_rectified", "rectified_size",
        }
        missing_keys = required - set(data.files)
        if missing_keys:
            raise RuntimeError(f"missing NPZ arrays: {sorted(missing_keys)}")
        positions = data["positions_left_camera_m"]
        valid = data["valid"]
        if positions.ndim != 4 or positions.shape[2:] != (21, 3) or valid.shape != positions.shape[:-1]:
            raise RuntimeError(f"unexpected MANO input shapes: positions={positions.shape}, valid={valid.shape}")
        for name in ("confidence", "observed", "input_observed", "outlier_rejected", "interpolated"):
            if data[name].shape != valid.shape:
                raise RuntimeError(f"unexpected {name} shape: {data[name].shape}")
        if data["bone_lengths_m"].shape != (positions.shape[0], 20):
            raise RuntimeError(f"unexpected bone length shape: {data['bone_lengths_m'].shape}")
        for name in ("pair_indices", "left_frame_indices", "right_frame_indices"):
            if data[name].shape != (positions.shape[1],):
                raise RuntimeError(f"unexpected {name} shape: {data[name].shape}")
        if data["left_to_rectified_rotation"].shape != (3, 3):
            raise RuntimeError("unexpected left rectification rotation shape")
        for name in ("projection_left_rectified", "projection_right_rectified"):
            if data[name].shape != (3, 4):
                raise RuntimeError(f"unexpected {name} shape: {data[name].shape}")
        finite = np.isfinite(positions).all(axis=-1)
        if not np.array_equal(finite, valid):
            raise RuntimeError("positions and valid mask disagree")
        if np.any(data["observed"] & data["interpolated"]):
            raise RuntimeError("observed and interpolated masks overlap")
        if np.any(data["observed"] & data["outlier_rejected"]):
            raise RuntimeError("accepted observations overlap rejected outliers")
        print(f"MANO input: tracks={positions.shape[0]} pairs={positions.shape[1]} valid_points={np.count_nonzero(valid)}")

    if args.input_only:
        print("MANO input contract is valid; licensed assets were not checked.")
        return 0
    if args.model_dir is None:
        raise ValueError("--model-dir is required unless --input-only is used")

    source = args.mano_source.resolve()
    if not (source / "mano" / "model.py").is_file():
        raise FileNotFoundError(f"invalid MANO source tree: {source}")
    revision = "unknown"
    try:
        revision = subprocess.run(
            ["git", "-C", str(source), "rev-parse", "HEAD"], check=True,
            capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        pass
    print(f"MANO source: {source} revision={revision}")

    missing = []
    for name in ("MANO_RIGHT.pkl",):
        path = args.model_dir / name
        if not path.is_file():
            missing.append(str(path))
        else:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            print(f"{name}: {path.stat().st_size} bytes sha256={digest}")
    if missing:
        raise FileNotFoundError("missing licensed MANO assets: " + ", ".join(missing))

    package_status = {
        name: importlib.util.find_spec(name) is not None
        for name in ("torch", "scipy", "trimesh", "chumpy")
    }
    print("Python packages:", package_status)
    if not all(package_status.values()):
        raise RuntimeError(
            "MANO fitting requires torch, scipy, trimesh and chumpy in the active Conda environment"
        )
    fit_script = Path(__file__).with_name("fit_mano_sequence.py")
    fit_spec = importlib.util.spec_from_file_location("fit_mano_sequence", fit_script)
    fit_module = importlib.util.module_from_spec(fit_spec)
    assert fit_spec.loader is not None
    fit_spec.loader.exec_module(fit_module)
    mano = fit_module.import_mano(source)
    import torch
    model = mano.load(
        str(args.model_dir.resolve()), is_rhand=True, num_pca_comps=15,
        batch_size=1, flat_hand_mean=False,
    )
    with torch.no_grad():
        output = model(return_tips=True)
    if tuple(output.vertices.shape) != (1, 778, 3) or tuple(output.joints.shape) != (1, 21, 3):
        raise RuntimeError("unexpected MANO_RIGHT output shapes")
    if not np.isfinite(output.vertices.detach().cpu().numpy()).all():
        raise RuntimeError("non-finite MANO_RIGHT vertices")
    print(f"MANO_RIGHT forward: vertices=778 joints=21 faces={len(model.faces)}")
    print("MANO assets and prepared input are ready for fitting.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(1)
