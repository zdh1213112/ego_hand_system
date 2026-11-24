# Source work/anatomy_mounts_20261119/rounds.py, SHA-256 4f95958a2dc602485b8dfdc2523bb6fde36e121e7a4915f22c5d28b913e1bc35

import case_settings as case

import json

import numpy as np

from scipy.sparse import coo_matrix

from scipy.sparse.csgraph import dijkstra

from scipy.spatial.transform import Rotation

ROOT = case.stage('anatomy_mounts_20261119')

from experiment import CompactHand, Fit

from anatomy_common import HANDS, checkpoint, paths, write_json

from assess import held_times

from calibrate_mounts import patch_fit, units

from common import isolated_poses

from fit_cuff import rigid_wrists

WEIGHTS = case.params['prior_weight']

ROUNDS = 3

def track(group, hand, version):
    if version == "soft35":
        return paths(group, hand)["soft35"]
    if version == "C":
        return checkpoint(group, "relaxed", hand)
    return ROOT / group / version / (hand + ".npz")

def load(path):
    with np.load(path) as z:
        return {k: z[k] for k in z.files}

def make_split(group, hand, times):
    held = np.isin(times, held_times(group, hand))
    return ~held, held

def calibration_metadata(group, hand, times, train, seed):
    with np.load(paths(group, hand)["soft35"]) as source:
        full_times = source["pair_indices"]
    index = np.searchsorted(full_times, times)
    np.testing.assert_array_equal(full_times[index], times)
    return dict(
        mount_calibration_sync=times[train],
        cuff_local_calibration_sync=times[train],
        cuff_reference_calibration_sync=seed["cuff_local_calibration_sync"],
        isolated_pose=isolated_poses(full_times)[index],
    )

def cuff_reference(oldlocal):
    # The cuff has its own rigid geometry. Freeze three tangent patches relative to the
    # old cuff triangle, with the configured ball radius as the height datum.
    pts = oldlocal.astype(float) * 1000
    e1, e2 = pts[1:] - pts[0]
    normal = units(np.cross(e1, e2))
    cc = (
        np.linalg.solve(np.stack([e1, e2, normal]), [e1 @ e1 / 2, e2 @ e2 / 2, 0])
        + pts[0]
    )
    radius = (
        json.loads((case.placement).read_text())[
            "marker_diameter_mm"
        ]
        / 2
    )
    patches = []
    for pt in pts:
        radial = units(pt - cc)
        tangent = units(np.cross(normal, radial))
        center = pt - radius * radial
        sq = np.array(
            [
                center + 12 * (a * tangent + b * normal)
                for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))
            ]
        )
        patches.append(sq[[[0, 1, 2], [0, 2, 3]]])
    return np.stack(patches)

def calibrate(group, hand, z, seed, train, branch, stamps):
    faces = z["faces"].astype(int)
    v = z["vertices"].astype(np.float64) * 1000
    observed = z["marker_valid"]
    targets = z["target24"].astype(float) * 1000
    local = z["wrist_local"].copy()
    wrist_polygons = cuff_reference(seed["wrist_local"])
    rotation = Rotation.from_rotvec(z["cuff_global_orient"]).as_matrix()
    wrist_targets = np.einsum(
        "nji,nkj->nki",
        rotation,
        targets[:, :3] - z["cuff_pivot_camera0_canonical_m"][:, None] * 1000,
    )
    rows = {}
    wrist_b = []
    wrist_f = []
    wrist_h = []
    for m in range(3):
        data = wrist_targets[train & observed[:, m], m]
        candidates = []
        for face, tri in enumerate(wrist_polygons[m]):
            b, h, e = patch_fit(
                np.broadcast_to(tri, (len(data), 3, 3)), data, tuple(case.params['mount_bounds_mm'])
            )
            candidates.append((float(np.mean(e**2)), face, b, h))
        _, face, b, h = min(candidates, key=lambda c: c[0])
        tri = wrist_polygons[m, face]
        local[m] = (
            b @ tri + h * units(np.cross(tri[1] - tri[0], tri[2] - tri[0]))
        ) / 1000
        wrist_b.append(b)
        wrist_f.append(face)
        wrist_h.append(h)
        rows[str(z["marker_names"][m])] = dict(
            surface="rigid_cuff_tangent_patch",
            n=len(data),
            face=int(face),
            barycentric=b.tolist(),
            height_mm=float(h),
            bounds_mm=list(case.params['mount_bounds_mm']),
            cuff_local_mm=(local[m] * 1000).tolist(),
        )
    wrist_goals = np.where(observed[:, :3, None], z["target24"][:, :3], np.nan)
    cuff_r, cuff_pivot, seen = rigid_wrists(
        local,
        wrist_goals,
        z["cuff_global_orient"],
        z["cuff_pivot_camera0_canonical_m"],
        z["pair_indices"],
        stamps,
    )
    np.testing.assert_equal(seen, observed[:, :3])
    attach = z["attachment_faces"].copy()
    bary = z["barycentric"].copy()
    off = z["attachment_local_offsets_m"].copy()
    idx = np.flatnonzero(train)
    sample = idx[np.linspace(0, len(idx) - 1, min(120, len(idx))).round().astype(int)]
    mesh = v[sample]
    edges = np.unique(
        np.sort(
            np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [0, 2]]]),
            axis=1,
        ),
        axis=0,
    )
    lengths = np.median(
        np.linalg.norm(mesh[:, edges[:, 0]] - mesh[:, edges[:, 1]], axis=-1), axis=0
    )
    graph = coo_matrix(
        (lengths, (edges[:, 0], edges[:, 1])), shape=(v.shape[1],) * 2
    ).tocsr()
    normals = units(
        np.cross(
            mesh[:, faces[:, 1]] - mesh[:, faces[:, 0]],
            mesh[:, faces[:, 2]] - mesh[:, faces[:, 0]],
        )
    )
    for m in range(3, 24):
        j = m - 3
        anchor = seed["attachment_faces"][j]
        radius = 25 if m == 3 else (20 if m == 4 else 11 if m % 4 == 3 else 14)
        distance = dijkstra(graph, directed=False, indices=faces[anchor]).min(0)
        align = np.median((normals * normals[:, anchor : anchor + 1]).sum(2), axis=0)
        candidates = np.union1d(
            [anchor, attach[j]],
            np.flatnonzero((distance[faces].max(1) <= radius) & (align >= 0.65)),
        )
        window = tuple(case.params['thumb_mount_bounds_mm']) if branch == "wide" and m in (4, 5, 6) else tuple(case.params['mount_bounds_mm'])
        fast = sample[observed[sample, m]]
        full = idx[observed[idx, m]]
        if not len(full):
            raise ValueError(f"No calibration observations of {z['marker_names'][m]}")
        if not len(fast):
            fast = full[:: max(1, len(full) // 120)]

        def estimate(f, ids):
            return patch_fit(v[ids[:, None], faces[f]], targets[ids, m], window)

        short = sorted(
            (float(np.mean(estimate(f, fast)[2] ** 2)), int(f)) for f in candidates
        )[:6]
        best = None
        for f in sorted({int(attach[j]), *[f for _, f in short]}):
            b, h, e = estimate(f, full)
            cost = np.mean(e**2)
            if best is None or cost < best[0]:
                best = (cost, f, b, h)
        _, f, b, h = best
        attach[j] = f
        bary[j] = b
        off[j] = [0, 0, h / 1000]
        rows[str(z["marker_names"][m])] = dict(
            surface="MANO",
            n=len(full),
            face=int(f),
            old_face=int(z["attachment_faces"][j]),
            source_face=int(anchor),
            barycentric=b.tolist(),
            height_mm=float(h),
            bounds_mm=list(window),
            candidate_faces=len(candidates),
        )
    values = dict(
        attachment_faces=attach,
        barycentric=bary,
        attachment_local_offsets_m=off,
        normal_offsets_m=off[:, 2].copy(),
        wrist_local=local,
        cuff_global_orient=Rotation.from_matrix(cuff_r).as_rotvec().astype(np.float32),
        cuff_pivot_camera0_canonical_m=cuff_pivot.astype(np.float32),
        cuff_patch_triangles_local_m=(wrist_polygons / 1000).astype(np.float32),
        cuff_attachment_faces=np.array(wrist_f),
        cuff_barycentric=np.array(wrist_b),
        cuff_height_mm=np.array(wrist_h),
    )
    values.update(calibration_metadata(group, hand, z["pair_indices"], train, seed))
    return values, rows

def apply_mounts(fit, z):
    fit.kin = CompactHand(fit.model, fit.tensor(z["betas"]), fit.axes, z)
    fit.cuff = fit.tensor(z["cuff_pivot_camera0_canonical_m"]) * 1000

def run(group, hand, branch, *, folder=None, seed_path=None, fit=None):
    folder = ROOT / group if folder is None else folder
    fit = Fit(group, hand, bounds_folder=folder / (branch + "_config")) if fit is None else fit
    source = track(group, hand, "C") if seed_path is None else seed_path
    seed = load(source)
    train, held = make_split(group, hand, fit.ts)
    weight = WEIGHTS[hand]
    for rnd in range(1, ROUNDS + 1):
        name = branch + str(rnd)
        dst = folder / name / (hand + ".npz")
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            source = dst
            continue
        z = seed.copy() if rnd == 1 else load(source)
        values, rows = calibrate(group, hand, z, seed, train, branch, fit.stamps)
        write_json(
            dst.parent / (hand + "_mounts.json"),
            dict(
                hand=hand,
                round=rnd,
                branch=branch,
                fit_sync=fit.ts[train].tolist(),
                holdout_sync=fit.ts[held].tolist(),
                markers=rows,
            ),
        )
        updated = {**z, **values}
        apply_mounts(fit, updated)
        fit.start = fit.tensor(z["solution_q"])
        indices = np.arange(len(fit.ts))
        strict, _ = fit.solve(indices, weight)
        q, ii, relaxed = fit.run("relaxed", weight, indices, prior_q=strict)
        fit.store(
            name, q, ii, relaxed, weight, source=source, destination=dst, mounts=values
        )
        source = dst
