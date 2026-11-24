# Source work/soft35_export_20261119/assess.py, SHA-256 f5ea54af8d63b92f60b053021835e2a8001b5dce2c84b3b44859db3e2454e4a0

import case_settings as case

import csv

import json

import numpy as np

from common import output, project, summary

from optical import SETS

def held_times(group, hand):
    return json.loads((output(group) / "calibration" / (hand + "_split.json")).read_text())["held_sync"]

def paired_pixels(rows):
    values = np.array([[r["hard_px"], r["soft_px"]] for r in rows], dtype=float).reshape(-1, 2)
    finite = np.isfinite(values)
    paired = finite.all(axis=1)
    result = {mode: summary(values[paired, side]) for side, mode in enumerate(("hard", "soft"))}
    result["observed"] = len(rows)
    result["unprojectable"] = dict(zip(("hard", "soft"), (~finite).sum(axis=0).tolist()))
    # Bound the median change for all raw observations, including a prediction behind a camera.
    # Give missing hard predictions the smallest possible error and missing soft predictions the largest.
    if paired.any():
        lower = np.median(np.where(finite[:, 0], values[:, 0], 0.0))
        upper = np.median(np.where(finite[:, 1], values[:, 1], np.inf))
        result["worst_case_delta_px"] = float(upper - lower) if np.isfinite(upper) else None
    else:
        result["worst_case_delta_px"] = None
    return result

def score_optical(group, hand, tracks, cams, rt):
    root = output(group)
    rows = []
    mirror = np.array([-1 if hand == "left" else 1, 1, 1])
    for key in SETS:
        folder = root / "optical" / (hand + "_" + key)
        with np.load(folder / "transforms.npz") as z:
            transforms = {
                tuple(k): (r, t) for k, r, t in zip(z["sync_camera"], z["rotation"], z["translation_m"])
            }
        with (folder / "spots.csv").open() as f:
            for obs in csv.DictReader(f):
                t, c, m = (int(obs[k]) for k in ("sync", "camera", "marker"))
                r, tr = transforms[t, c]
                xy = np.array([float(obs["u"]), float(obs["v"])])
                assert int(obs["source_views"]) >= 2
                row = dict(
                    group=group,
                    hand=hand,
                    subset=key,
                    sync=t,
                    camera=c,
                    marker=m,
                    name=obs["name"],
                    u=float(xy[0]),
                    v=float(xy[1]),
                )
                for mode, track in tracks.items():
                    i = np.searchsorted(track["pair_indices"], t)
                    assert track["pair_indices"][i] == t and track["marker_valid"][i, m]
                    uv = project(
                        (track["predicted24"][i, m] * mirror @ r.T + tr)[None], rt[c], cams[c], 1
                    )[0]
                    row[mode + "_px"] = float(np.linalg.norm(uv - xy)) if np.isfinite(uv).all() else None
                rows.append(row)
    assert len(rows) > 0
    with (root / ("optical_" + hand + ".csv")).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0])
        w.writeheader()
        w.writerows(rows)
    sets = dict(
        wrist=[r for r in rows if r["marker"] < 3],
        nails=[r for r in rows if r["subset"] == "wrist_nail" and r["marker"] >= 3],
        four=[r for r in rows if r["subset"] != "wrist_nail"],
        all=[r for r in rows if r["subset"] == "wrist_nail" or r["marker"] % 4 != 3],
    )
    all_ids = [(r["sync"], r["camera"], r["marker"]) for r in sets["all"]]
    assert len(all_ids) == len(set(all_ids))
    return {k: paired_pixels(rows) for k, rows in sets.items()}
