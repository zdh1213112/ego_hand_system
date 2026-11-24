"""Read real camera pixels around measured NOKOV guides. Match identities before scoring the candidate."""

import itertools, json
from concurrent.futures import ThreadPoolExecutor
import cv2, numpy as np


def peaks(gray, guide, cfg):
    parts = []
    for hand in guide:
        visible = hand[
            np.isfinite(hand).all(1) & (hand > 0).all(1) & (hand < np.array(gray.shape[::-1]) - 1).all(1)
        ]
        if not len(visible):
            continue
        xy = np.maximum(0, np.floor(visible.min(0) - cfg["roi_padding_px"]).astype(int))
        end = np.minimum(gray.shape[::-1], np.ceil(visible.max(0) + cfg["roi_padding_px"] + 1).astype(int))
        patch = gray[xy[1] : end[1], xy[0] : end[0]].astype(np.float32)
        core = cv2.GaussianBlur(patch, (0, 0), cfg["core_sigma_px"])
        dog = core - cv2.GaussianBlur(patch, (0, 0), cfg["surround_sigma_px"])
        yy, xx = np.nonzero((dog == cv2.dilate(dog, np.ones((cfg["dilation_size"],) * 2, np.uint8))) & (dog >= cfg["min_dog"]) & (core >= cfg["min_intensity"]))
        keep = (xx >= cfg["patch_border_px"]) & (yy >= cfg["patch_border_px"]) & (xx < patch.shape[1] - cfg["patch_border_px"]) & (yy < patch.shape[0] - cfg["patch_border_px"])
        xx, yy = xx[keep], yy[keep]
        if not len(xx):
            continue
        ring = np.array(
            [
                core[yy + dy, xx + dx]
                for dx, dy in cfg["ring_offsets_px"]
            ]
        )
        keep = ((core[yy, xx] - ring) > cfg["ring_contrast"]).sum(0) >= cfg["min_ring_samples"]
        parts.append(np.stack([xx[keep], yy[keep]], 1) + xy)
    return np.unique(np.concatenate(parts), axis=0) if parts else np.empty((0, 2))


def projected(x, cam, cfg):
    trans, K, D = cam
    xyz = x @ trans[:3, :3].T + trans[:3, 3]
    out = np.full(xyz.shape[:-1] + (2,), np.nan)
    good = np.isfinite(xyz).all(-1) & (xyz[..., 2] > cfg["min_depth_m"])
    if good.any():
        out[good] = cv2.fisheye.projectPoints(
            xyz[good].astype("d").reshape(-1, 1, 3), np.zeros(3), np.zeros(3), K, D
        )[0][:, 0]
    return out


def one_view(impath, guide, cfg):
    gp = guide[-1]
    valid = np.isfinite(guide).all(-1)
    if not valid[-1] or np.any(gp < cfg["image_margin_px"]):
        return None
    if np.min(np.linalg.norm(guide[:-1][valid[:-1]] - gp, axis=-1), initial=np.inf) < cfg["min_marker_distance_px"]:
        return None
    # Local image evidence for all named markers of this hand, independent of both predicted nail positions.
    im = cv2.imread(str(impath), cv2.IMREAD_GRAYSCALE)
    if im is None:
        raise FileNotFoundError(impath)
    if np.any(gp > np.asarray(im.shape[::-1]) - cfg["image_margin_px"] - 1):
        return None
    dots = peaks(im, guide[None], cfg)
    if len(dots) < cfg["min_identified_markers"]:
        return None
    ids = np.flatnonzero(valid)
    dist = np.linalg.norm(guide[ids, None] - dots, axis=-1)
    best = np.argmin(dist, axis=1)
    nearest = np.sort(dist, axis=1)
    acceptable = (
        (nearest[:, 0] <= cfg["match_radius_px"])
        & (nearest[:, 1] - nearest[:, 0] >= cfg["match_margin_px"])
        & (np.argmin(dist[:, best], axis=0) == np.arange(len(ids)))
    )
    if acceptable.sum() < cfg["min_identified_markers"] or not acceptable[-1]:
        return None
    return dots[best[-1]]


def verify(data, solution, caldir, images, cfg):
    with np.load(data) as track, np.load(solution) as sol:
        sync = sol["sync"]
        idx = np.searchsorted(track["pair_indices"], sync)
        np.testing.assert_array_equal(track["pair_indices"][idx], sync)
        true = track["target24"][idx].astype("d")
        raw = track["target24_uncorrected"][idx].astype("d")
        before, after = sol["nail_before"].astype("d"), sol["nail_after"].astype("d")
        declared = track["marker_valid"][idx].astype(bool)
    true[~declared] = np.nan
    raw[~declared] = np.nan
    cams = [json.loads((caldir / f"camera{i}.json").read_text()) for i in range(4)]
    T = [np.array(c["T_base_camera"]) for c in cams]
    camera = [(np.linalg.inv(t) @ T[0], np.array(c["K"]), np.array(c["distortion"])) for t, c in zip(T, cams)]
    guides = [projected(true, c, cfg) for c in camera]
    obs = np.full((len(sync), 4, 2), np.nan)
    tasks = [(i, c, im) for i, t in enumerate(sync) for c, im in images.get(int(t), {}).items()]

    def load(task):
        i, c, im = task
        return i, c, one_view(im, guides[c][i], cfg)

    with ThreadPoolExecutor(max_workers=cfg["workers"]) as pool:
        for i, c, val in pool.map(load, tasks):
            if val is not None:
                obs[i, c] = val
    rays = np.full((len(sync), 4, 3), np.nan)
    orig = []
    for c, (x, K, D) in enumerate(camera):
        origin = -x[:3, 3] @ x[:3, :3]
        orig.append(origin)
        good = np.isfinite(obs[:, c]).all(1)
        if good.any():
            uv = cv2.fisheye.undistortPoints(obs[good, c, None], K, D)[:, 0]
            xyz = np.c_[uv, np.ones(len(uv))] @ x[:3, :3]
            rays[good, c] = xyz / np.linalg.norm(xyz, axis=1, keepdims=True)
    supports = np.zeros((len(sync), 4), bool)
    rank = np.full(len(sync), -np.inf)
    for a, b in itertools.combinations(range(4), 2):
        da, db = rays[:, a], rays[:, b]
        w = orig[a] - orig[b]
        cos = (da * db).sum(1)
        den = 1 - cos**2
        la = (cos * (db @ w) - da @ w) / np.maximum(den, cfg["denominator_floor"])
        lb = (db @ w - cos * (da @ w)) / np.maximum(den, cfg["denominator_floor"])
        pos = 0.5 * (orig[a] + da * la[:, None] + orig[b] + db * lb[:, None])
        err = np.stack(
            [np.linalg.norm(projected(pos, cam, cfg) - obs[:, c], axis=1) for c, cam in enumerate(camera)], axis=1
        )
        sup = err <= cfg["support_radius_px"]
        deviation = np.linalg.norm(pos - true[:, -1], axis=1)
        valid = (
            (den > cfg["min_ray_angle_sine"]**2)
            & (la > 0)
            & (lb > 0)
            & (deviation <= cfg["max_registered_distance_m"])
            & (np.linalg.norm(pos - raw[:, -1], axis=1) <= cfg["max_raw_distance_m"])
            & (err[:, a] <= cfg["pair_reprojection_px"])
            & (err[:, b] <= cfg["pair_reprojection_px"])
        )
        score = sup.sum(1) - cfg["reprojection_penalty"] * np.where(sup, err, 0).sum(1) - deviation * cfg["deviation_penalty"]
        use = valid & (score > rank)
        supports[use] = sup[use]
        rank[use] = score[use]
    eb, ea = [
        np.stack(
            [np.linalg.norm(projected(pts, c, cfg) - obs[:, j], axis=1) for j, c in enumerate(camera)], axis=1
        )
        for pts in (before, after)
    ]
    d = np.where(supports, eb - ea, 0)
    gain = d.sum(1) / np.maximum(supports.sum(1), 1)
    use = (supports.sum(1) >= cfg["min_views"]) & (gain >= cfg["min_gain_px"]) & np.all(~supports | (d >= -cfg["max_view_loss_px"]), axis=1)
    # Each accepted label agrees with the measured raw NOKOV nail within the calibration uncertainty.
    raw_before = 1000 * np.linalg.norm(before - raw[:, -1], axis=1)
    raw_after = 1000 * np.linalg.norm(after - raw[:, -1], axis=1)
    use &= raw_after <= raw_before + cfg["max_raw_loss_mm"]
    return dict(
        sync=sync,
        accept=use,
        support=supports,
        observations=obs,
        before_px=eb,
        after_px=ea,
        raw_before_mm=raw_before,
        raw_after_mm=raw_after,
    )
