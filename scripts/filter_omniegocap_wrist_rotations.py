#!/usr/bin/env python3
"""Flag both observations around excessive adjacent wrist rotation steps.

This writes a separate input file; it never interpolates a replacement wrist.
The threshold is a chosen quality heuristic, not a measured accuracy bound.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def rotation_rejections(wrist, visible, threshold):
    rejected = np.zeros_like(visible, dtype=bool)
    pairs = []
    for side in (0, 1):
        R = wrist[:, side, :3, :3].astype(float)
        angles = np.degrees(Rotation.from_matrix(np.swapaxes(R[:-1], -1, -2) @ R[1:]).magnitude())
        ends = np.flatnonzero(visible[:-1, side] & visible[1:, side] & (angles > threshold)) + 1
        for end in ends:
            rejected[end - 1:end + 1, side] = True
            pairs.append({"side": side, "frames": [int(end - 1), int(end)], "rotation_step_deg": float(angles[end - 1])})
    return rejected, pairs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-step-deg", type=float, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not 0 < args.max_step_deg <= 180:
        p.error("max-step-deg must be in (0,180]")
    with np.load(args.input, allow_pickle=False) as source:
        data = {k: source[k].copy() for k in source.files}
    rejected, pairs = rotation_rejections(data["wrist"], data["visible"], args.max_step_deg)
    data["visible"][rejected] = False
    data["wrist"][rejected] = np.eye(4)
    meta = json.loads(str(data.get("meta", "{}")))
    meta["wrist_rotation_postcheck"] = {
        "source": str(args.input.resolve()), "source_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "max_adjacent_step_deg": args.max_step_deg, "policy": "mark BOTH endpoints invisible; no replacement",
        "pairs": pairs, "removed_observations": int(rejected.sum()),
    }
    data["meta"] = np.str_(json.dumps(meta))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **data)
    print(json.dumps(meta["wrist_rotation_postcheck"], indent=2))


if __name__ == "__main__":
    main()
