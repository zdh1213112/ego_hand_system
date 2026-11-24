# Source work/direction_acceptance_20261119/observed.py, SHA-256 496c0753187ec8a1aa3b92cce3e48420818fb3f3c839fa214fb0bf7919028632

import case_settings as case

import itertools

import cv2

import numpy as np

from fit_wide import HANDS, ROOT, WORK, source_track

from anatomy_common import OPT, geometry, paths, physical, project

from direction import TRIPLES, angles, angle_distance, saved_wilor, sdk_in_fit_space

def uv_all(xyz, cams, rt):
    return np.stack([project(xyz, x, c, 1) for x, c in zip(rt, cams)], axis=1)

def triangulated(uv, usable, cams, rt):
    n = len(uv)
    origins = np.stack([-r.T @ t for r, t in rt])
    rays = np.zeros((n, 4, 21, 3))
    available = np.zeros((n, 4, 21), bool)
    for c, (cam, (r, t)) in enumerate(zip(cams, rt)):
        pixels = uv[:, c]
        ok = (
            usable[:, c, None]
            & np.isfinite(pixels).all(-1)
            & (pixels >= 0).all(-1)
            & (pixels < cam.image_size).all(-1)
        )
        if not ok.any():
            continue
        xy = cv2.fisheye.undistortPoints(
            np.ascontiguousarray(pixels[ok], dtype=float).reshape(-1, 1, 2),
            cam.K,
            cam.distortion,
        ).reshape(-1, 2)
        d = np.c_[xy, np.ones(len(xy))] @ r
        rays[:, c][ok] = d / np.linalg.norm(d, axis=1)[:, None]
        available[:, c][ok] = np.isfinite(xy).all(1) & (
            np.linalg.norm(xy, axis=1) < 1e4
        )
    planes = np.eye(3) - rays[..., None] * rays[..., None, :]
    rhs = (planes @ origins[None, :, None, :, None])[..., 0]

    def solve(mask):
        lhs = (planes * mask[..., None, None]).sum(1)
        v = (rhs * mask[..., None]).sum(1)
        good = (mask.sum(1) >= 2) & (np.linalg.det(lhs) > 1e-6)
        out = np.full((n, 21, 3), np.nan)
        if good.any():
            out[good] = np.linalg.solve(lhs[good], v[good, ..., None])[..., 0]
        return out

    best = np.zeros_like(available)
    best_n = np.zeros((n, 21), int)
    best_e = np.full((n, 21), np.inf)
    for c, k in itertools.combinations(range(4), 2):
        separation = np.linalg.norm(np.cross(rays[:, c], rays[:, k]), axis=-1)
        q = available[:, c] & available[:, k] & (separation >= np.sin(np.deg2rad(5)))
        pair = np.zeros_like(available)
        pair[:, c] = q
        pair[:, k] = q
        xyz = solve(pair)
        err = np.linalg.norm(uv_all(xyz, cams, rt) - uv, axis=-1)
        inlier = available & (err <= 5.0)
        count = inlier.sum(1)
        avg = np.where(inlier, err, 0).sum(1) / np.maximum(count, 1)
        use = (
            q
            & inlier[:, c]
            & inlier[:, k]
            & ((count > best_n) | ((count == best_n) & (avg < best_e)))
        )
        best = np.where(use[:, None], inlier, best)
        best_n = np.where(use, count, best_n)
        best_e = np.where(use, avg, best_e)
    xyz = solve(best)
    for _ in range(2):
        err = np.linalg.norm(uv_all(xyz, cams, rt) - uv, axis=-1)
        best &= err <= 5.0
        cross_ok = np.zeros((n, 21), bool)
        for c, k in itertools.combinations(range(4), 2):
            cross_ok |= (
                best[:, c]
                & best[:, k]
                & (
                    np.linalg.norm(np.cross(rays[:, c], rays[:, k]), axis=-1)
                    >= np.sin(np.deg2rad(5))
                )
            )
        best &= cross_ok[:, None]
        xyz = solve(best)
    err = np.linalg.norm(uv_all(xyz, cams, rt) - uv, axis=-1)
    return xyz, best, np.where(best, err, np.nan)

def measure(uv, cams):
    a, length = angles(uv)
    size = np.array([c.image_size for c in cams])[None, :, None, :]
    visible = np.isfinite(uv).all(-1) & (uv >= 0).all(-1) & (uv < size).all(-1)
    return a, (length >= 8) & visible[..., TRIPLES].all(-1)

def prepare(group, hand):
    dest = ROOT / group
    dest.mkdir(exist_ok=True)
    side = HANDS.index(hand)
    _, cams, rt, frames = geometry(group)
    sync = np.array(sorted(frames))
    source = np.array(
        [[int(frames[t][f"camera{c}_frame_index"]) for c in range(4)] for t in sync]
    )
    cache = saved_wilor(group, source)
    with np.load(paths(group, hand)["soft35"]) as z:
        ts = z["pair_indices"]
        soft_full = physical(z["joints"].astype(float), hand)
    with np.load(OPT[group] / "skeleton/aligned_skeleton.npz") as z:
        np.testing.assert_equal(
            z["reference_timestamp_ns"],
            [int(frames[int(t)]["reference_timestamp_ns"]) for t in sync],
        )
        sdk = sdk_in_fit_space(
            group,
            side,
            z["positions_cam0_m"][:, :, z["provisional21_indices"]].astype(float),
            z["marker24_cam0_m"].astype(float),
            ts,
        )
    times = np.array(sorted({t for t, c, h in cache if h == side} & set(ts)))
    idx = np.searchsorted(ts, times)
    soft = soft_full[idx]
    sdk = sdk[idx]
    with np.load(source_track(group, hand)) as z:
        np.testing.assert_equal(z["pair_indices"], ts)
        wide = physical(z["joints"][idx].astype(float), hand)
        isol = z["isolated_pose"][idx]
    uv = np.full((len(times), 4, 21, 2), np.nan)
    conf = np.full((len(times), 4), np.nan)
    for i, t in enumerate(times):
        for c in range(4):
            if (int(t), c, side) in cache:
                uv[i, c], conf[i, c] = cache[t, c, side]
    su = uv_all(sdk, cams, rt)
    ou = uv_all(soft, cams, rt)
    wu = uv_all(wide, cams, rt)
    span = np.linalg.norm(np.ptp(ou, axis=-2), axis=-1)
    usable = (
        (conf >= 0.5)
        & np.isfinite(uv).all((-1, -2))
        & np.isfinite(su).all((-1, -2))
        & np.isfinite(ou).all((-1, -2))
    )
    usable &= np.linalg.norm(ou[:, :, 0] - uv[:, :, 0], axis=-1) <= 0.35 * span
    sa, so = measure(su, cams)
    wa, wo = measure(uv, cams)
    ok = (
        usable[..., None]
        & so
        & wo
        & measure(ou, cams)[1]
        & measure(wu, cams)[1]
        & (angle_distance(sa, wa) <= 12)
    )
    ok &= (ok.sum(1) >= 2)[:, None, :]
    reconstructed, inlier, errors = triangulated(uv, usable, cams, rt)
    totals = set()
    for path in (WORK / "joint_motion_20261119" / group).glob("*wilor.npz"):
        with np.load(path) as z:
            totals.update(z["times"])
    np.savez_compressed(
        dest / (hand + "_observations.npz"),
        times=times,
        track_index=idx,
        isolated=isol,
        sdk=sdk,
        soft=soft,
        wide=wide,
        sdk_uv=su,
        soft_uv=ou,
        wide_uv=wu,
        native_uv=uv,
        native3d=reconstructed,
        tri_views=inlier,
        tri_errors=errors,
        matched=usable,
        eligible=ok,
        confidence=conf,
        n_sync=len(sync),
        archive_sync=len(totals),
        detected_hands=len(cache),
    )
    print(
        group,
        hand,
        "sync",
        len(times),
        "qualifying camera-joints",
        ok.sum(),
        "triangulated joints",
        np.isfinite(reconstructed).all(-1).sum(),
        flush=True,
    )
