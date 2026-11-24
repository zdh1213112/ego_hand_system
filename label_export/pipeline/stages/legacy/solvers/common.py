# Source work/soft35_export_20261119/common.py, SHA-256 0ccb8438607541aac4e25b0c88899a3b5dcdce3e5fb2b2918ee7fd51d43f01c1

import case_settings as case

import json

import numpy as np

ROOT = case.stage('soft35_export_20261119')

WORK = ROOT.parent

from acceptance import HERE, OPT, EGO, TIPS, geometry, project, save, summary, load, write_json

HANDS = ("left", "right")

MOTION = WORK / "joint_motion_20261119"

def isolated_poses(times):
    """One valid pose per physical-hand sync, with no other valid sync at distance 1–3."""
    sync = np.asarray(times, dtype=np.int64)
    if sync.ndim != 1 or np.any(np.diff(sync) <= 0):
        raise ValueError("Expected a strictly increasing list of valid synchronized exposures")
    isolated = np.ones(len(sync), dtype=bool)
    near = np.diff(sync) <= 3
    isolated[:-1] &= ~near
    isolated[1:] &= ~near
    return isolated

def output(group):
    return WORK / (group + "_soft35_20261119")

def track_file(group, hand, variant="soft"):
    return output(group) / variant / (hand + ".npz")

def read_jsonl(path):
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]

def write_jsonl(path, rows):
    with path.open("w") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")

def gate_and_errors(track):
    error = np.linalg.norm(track["target24"] - track["predicted24"], axis=2) * 1000
    error = np.where(track["marker_valid"], error, np.nan)
    hands = error[:, 3:]
    complete = track["marker_valid"][:, 4:].reshape(-1, 5, 4).sum(2).min(1) >= case.params['min_markers_per_finger']
    return complete & (np.nanmedian(hands, 1) <= case.params['surface_median_mm']) & (np.nanmax(hands, 1) <= case.params['surface_max_mm']), error
