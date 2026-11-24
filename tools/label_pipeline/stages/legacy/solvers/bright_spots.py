# Source work/group_003_mount_acceptance_20261119/bright_spots.py, SHA-256 abe86a7c53372dd2f856aacc3ed75576747f239278113584105a070dd88d02a0

import case_settings as case

import csv

import cv2

import numpy as np

from acceptance import BASE, GROUP, HERE, ROOT, project, save, write_json

from observations import read_rgb

from optical_holdout import ALIGN, align_sources, assign, detected, inside, rigid, triangulate

GROUPS = {
    "right_mr": ("right", tuple(range(12, 20)) + (23,)),
    "left_im": ("left", tuple(range(8, 16))),
    "left_rp": ("left", tuple(range(16, 24))),
}

def detect_all(source, cals, rt, rows, times, recording=GROUP):
    raw = {}
    for side, name in enumerate(("left", "right")):
        mirror = np.array([1 if side else -1, 1, 1])
        with np.load(HERE / recording / "aligned" / (name + "_pilot.npz")) as z:
            ix = {int(t): i for i, t in enumerate(z["pair_indices"])}
            xyz = z["target24_uncorrected"].astype(float) * mirror
        for t in times:
            if t in ix:
                raw.setdefault(t, {})[name] = xyz[ix[t]]
    clouds = {}
    for cam, cal in enumerate(cals):
        cap = cv2.VideoCapture(
            str(source / "cameras" / f"camera{cam}" / "video.mp4"),
            cv2.CAP_FFMPEG,
            [cv2.CAP_PROP_N_THREADS, 2],
        )
        try:
            for t in times:
                image = read_rgb(cap, int(rows[t][f"camera{cam}_frame_index"]))
                guides = [project(x, rt[cam], cal, 1) for x in raw[t].values()]
                clouds[t, cam] = detected(image, guides, cal)
        finally:
            cap.release()
    return raw, clouds

def identify(group, times, raw, clouds, cals, rt, *, target=None, source_track=None, output_root=ROOT):
    hand, markers = GROUPS[group] if target is None else target
    other = "left" if hand == "right" else "right"
    mirror = np.array([1 if hand == "right" else -1, 1, 1])
    path = BASE / "tracks" / (hand + ".npz") if source_track is None else source_track
    with np.load(path) as z:
        timeline = z["pair_indices"]
        names = z["marker_names"]
        archive = z["target24"] * mirror
        valid = (z["marker_valid"] if source_track is None
                 else np.isfinite(z["target24_uncorrected"]).all(-1))
    found = []
    registrations = []
    transforms = []
    coverage = []
    for t in times:
        slot = int(np.searchsorted(timeline, t))
        assert timeline[slot] == t
        current = raw[t][hand]
        present = valid[slot] & np.isfinite(current).all(1)
        rr, tr = rigid(archive[slot, present], current[present])
        assert (
            np.max(np.linalg.norm(archive[slot, present] @ rr.T + tr - current[present], axis=1))
            < 0.000002
        )
        # align_sources uses key 1 for its centering hand. All coordinates here are physical camera0.
        guides = {1: current.copy()}
        guides[1][list(markers)] = np.nan
        assert not np.isfinite(guides[1][list(markers)]).any()
        if other in raw[t]:
            guides[0] = raw[t][other]
        cloud = {c: clouds[t, c] for c in range(4)}
        for held in (1, 2, 3):
            sources = [c for c in range(4) if c != held]
            fit = align_sources(guides, sources, cals, rt, cloud)
            counts = dict(
                sync=int(t),
                camera=held,
                registered=fit is not None,
                in_frame=0,
                triangulated=0,
                matched=0,
                matched_8px=0,
                matched_16px=0,
            )
            if fit is None:
                coverage.append(counts)
                continue
            r, v, n, med = fit
            counts.update(source_points=n, source_residual_median_px=med)
            registrations.append((t, held))
            transforms.append((r @ rr, r @ tr + v))
            coords = np.vstack([current, raw[t][other]]) if other in raw[t] else current
            xyz = coords @ r.T + v
            src_matches = {}
            for c in sources:
                p = project(xyz, rt[c], cals[c], 1)
                p[~inside(p, cals[c])] = np.nan
                src_matches[c] = assign(p, cloud[c], 5.5, 1.5, 5.5)
            search = project(xyz, rt[held], cals[held], 1)
            search[~inside(search, cals[held])] = np.nan
            identified = {}
            for m in markers:
                if not present[m] or not np.isfinite(search[m]).all():
                    continue
                counts["in_frame"] += 1
                src_uv = {c: src_matches[c][m] for c in sources if m in src_matches[c]}
                point = triangulate(src_uv, xyz[m], cals, rt)
                if point is None:
                    continue
                q = project(point[None], rt[held], cals[held], 1)[0]
                if not inside(q[None], cals[held])[0]:
                    continue
                search[m] = q
                identified[m] = len(src_uv)
                counts["triangulated"] += 1
            matches = assign(search, cloud[held], 16.0, 2.0, 6.0)
            for m, support in identified.items():
                if m not in matches:
                    continue
                uv = matches[m]
                dist = float(np.linalg.norm(uv - search[m]))
                counts["matched_16px"] += 1
                if dist >= 12:
                    continue
                counts["matched"] += 1
                counts["matched_8px"] += int(dist < 8)
                found.append(
                    dict(
                        group=group,
                        hand=hand,
                        sync=int(t),
                        camera=held,
                        marker=m,
                        name=str(names[m]),
                        u=float(uv[0]),
                        v=float(uv[1]),
                        source_views=support,
                        guide_error_px=dist,
                    )
                )
            coverage.append(counts)
    keys = [(row["sync"], row["camera"], row["marker"]) for row in found]
    assert len(keys) == len(set(keys))
    folder = output_root / "optical" / group
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "spots.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("group", "hand", "sync", "camera", "marker", "name",
                                                     "u", "v", "source_views", "guide_error_px"))
        writer.writeheader()
        writer.writerows(found)
    save(
        folder / "transforms.npz",
        dict(
            sync_camera=np.array(registrations),
            rotation=np.array([r for r, _ in transforms]).reshape(-1,3,3),
            translation_m=np.array([v for _, v in transforms]).reshape(-1,3),
        ),
    )
    write_json(
        folder / "coverage.json",
        dict(
            evaluated_markers=list(markers),
            subject_markers_used_to_register=sorted(set(ALIGN.tolist()) - set(markers)),
            other_hand_markers_used_to_register=ALIGN.tolist(),
            attempted_sync=len(times),
            registered_camera_views=len(registrations),
            matched_camera_marker_pairs=len(found),
            matched_sync=len(set(row["sync"] for row in found)),
            views=coverage,
        ),
    )
    print(
        group,
        "poses",
        len(times),
        "registered",
        len(registrations),
        "bright spots",
        len(found),
        flush=True,
    )
