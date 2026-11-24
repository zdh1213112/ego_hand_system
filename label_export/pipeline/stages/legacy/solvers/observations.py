# Source work/tip_pivot_20261119/observations.py, SHA-256 9d523e65b3b15791a0e222f7da7ae7d8b2789c7ef0cb77216b71d68831043bab

import case_settings as case

import csv

import json

from pathlib import Path

import cv2

import numpy as np

from scipy.spatial import cKDTree

from fit_pivot import HANDS, OPT, OUT

from image_alignment import blobs

from ego_data.calibration import CameraCalibration

from native_camera_geometry import reference_to_camera

from wilor_label_projection import project_label_points

def geometry(group):
    path = Path(json.loads((OPT[group] / "context.json").read_text())["dataset"])
    cameras = [CameraCalibration.load(path / "calibration" / f"camera{i}.json") for i in range(4)]
    transforms = [reference_to_camera(cameras[0], cam) for cam in cameras]
    with (path / "multiview_frames.csv").open() as f:
        rows = {int(r["sync_index"]): r for r in csv.DictReader(f)}
    return path, cameras, transforms, rows

def project(xyz, rt, camera, side, min_depth=1e-8):
    r, t = rt
    actual = xyz * np.array([1 if side else -1, 1, 1]) @ r.T + t
    good = np.isfinite(actual).all(-1) & (actual[..., 2] > min_depth)
    uv = np.full((*actual.shape[:-1], 2), np.nan)
    if good.any():
        uv[good] = project_label_points(actual[good], dict(K=camera.K, distortion=camera.distortion,
                                         camera_model="KB", image_space="native_raw"))
    return uv

def read_rgb(cap, index):
    pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    if pos <= index <= pos + 6:
        for _ in range(index - pos):
            assert cap.grab()
    else:
        assert cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, img = cap.read()
    if not ok or int(cap.get(cv2.CAP_PROP_POS_FRAMES)) != index + 1:
        raise IOError(f"Unable to decode original source frame {index}")
    return img

def optical(group):
    path, cams, transforms, rows = geometry(group)
    inputs = {}
    for side, hand in enumerate(HANDS):
        with np.load(OUT / group / hand / "before.npz") as z:
            inputs[side] = {int(t): (target[4:], valid[4:]) for t, target, valid in
                            zip(z["times"], z["target24"], z["marker_valid"])}
    selected = sorted(set(inputs[0]) | set(inputs[1]))
    spots = []
    for c, cal in enumerate(cams):
        cap = cv2.VideoCapture(str(path / "cameras" / f"camera{c}" / "video.mp4"), cv2.CAP_FFMPEG,
                               [cv2.CAP_PROP_N_THREADS, 2])
        try:
            for t in selected:
                img = read_rgb(cap, int(rows[t][f"camera{c}_frame_index"]))
                for side, events in inputs.items():
                    if t not in events:
                        continue
                    target, good = events[t]
                    uv = project(target, transforms[c], cal, side, min_depth=.005)
                    ids = np.flatnonzero(good & np.isfinite(uv).all(1) &
                                         (uv >= 0).all(1) & (uv < cal.image_size).all(1))
                    if len(ids) < 2:
                        continue
                    measured = uv[ids]
                    low = np.maximum(0, np.floor(measured.min(0) - 32).astype(int))
                    high = np.minimum(cal.image_size, np.ceil(measured.max(0) + 33).astype(int))
                    found = blobs(img[low[1]:high[1], low[0]:high[0]]) + low
                    if len(found) < 2:
                        continue
                    distance, match = cKDTree(found).query(measured, k=2)
                    gap = cKDTree(measured).query(measured, k=2)[0][:, 1]
                    reliable = (gap > 6) & (distance[:, 0] < 3) & ((distance[:, 1] - distance[:, 0]) > 2)
                    for i in np.flatnonzero(reliable):
                        u, v = found[match[i, 0]]
                        spots.append((t, side, c, int(ids[i]), float(u), float(v)))
        finally:
            cap.release()
    file = OUT / group / "rgb_spots.csv"
    with file.with_suffix(".partial").open("w") as stream:
        writer = csv.writer(stream)
        writer.writerow(["sync", "side", "camera", "finger_marker", "u", "v"])
        writer.writerows(spots)
    file.with_suffix(".partial").replace(file)
    print(group, "independent RGB spots", len(spots), flush=True)
