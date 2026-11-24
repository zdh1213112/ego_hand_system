# Source work/soft35_export_20261119/fit_recording.py, SHA-256 4ff07f702403ba068081223b13364d5040ea93024ba66dc8c80e6cc8c5990c11

import case_settings as case

import json

import numpy as np

from scipy.sparse import coo_matrix

from scipy.sparse.csgraph import dijkstra

from common import HERE, OPT, ROOT, TIPS, geometry, load, output, save, summary, write_json

from fit_sequence import source_hand, fit_from_reference

from fit_cuff import complete_rotations, fit

from lock_sequence import lock

from calibrate_mounts import patch_fit, units

from refit import run

def prepare(group, hand):
    dest = output(group)
    dest.mkdir(exist_ok=True)
    data = source_hand(hand, group)
    times = data["pair_indices"]
    _, cameras, rt, rows = geometry(group)
    selection = json.loads((OPT[group] / "selection/selection.json").read_text())["report"]["hands"][
        hand
    ]["calibration_sync"]
    with np.load(ROOT.parent / "tip_pivot_20261119" / group / hand / "geometry6_locked.npz") as z:
        np.testing.assert_equal(z["betas"], data["betas"])
        tt = z["times"]
        cand = np.flatnonzero(np.isin(tt, times) & ((tt // 120) % 4 == 0))
        ix = np.searchsorted(times, tt[cand])
        with np.load(HERE / group / "aligned" / (hand + "_pilot.npz")) as aligned:
            jj = np.searchsorted(aligned["pair_indices"], times)
            np.testing.assert_equal(aligned["pair_indices"][jj], times)
            cuffs = aligned["target24"][jj, :3]
        seen = np.isfinite(cuffs).all((1, 2))
        cand = cand[seen[ix]]
        if not len(cand):
            raise ValueError("Missing recorded outward anchor outside the validation blocks")
        reach = np.linalg.norm(z["joints"][cand, 0, None] - z["joints"][cand][:, TIPS[1:]], axis=2).mean(
            1
        )
        # Pick a relatively extended calibration pose without looking at any candidate fit.
        pick = cand[np.argmin(abs(reach - np.percentile(reach, 75)))]
        requested=case.config.get('wrist_sync',{}).get(hand)
        if requested is not None:
            explicit=np.flatnonzero(tt==requested)
            if len(explicit)!=1 or requested not in times:raise ValueError(f'腕部签收帧缺失 {hand} {requested}')
            pick=int(explicit[0])
        anchor = int(tt[pick])
        end = {
            key: z[key][pick]
            for key in (
                "joints",
                "vertices",
                "angles",
                "global_orient",
                "hand_pose_pca",
                "translation",
                "mounted",
            )
        }
    held = ((times // 120) % 4 == 2) & (np.abs(times - anchor) > 120)
    for t in selection:
        held &= abs(times - t) > 8
    ht = times[held]
    opt = ht[np.linspace(0, len(ht) - 1, min(120, len(ht))).round().astype(int)]
    assert len(ht) > 0 and not np.intersect1d(opt, [*selection, anchor]).size
    split = dict(
        anchor_sync=anchor,
        former_shape_calibration=list(map(int, selection)),
        fit_sync=times[~held].tolist(),
        held_sync=ht.tolist(),
        optical_sync=opt.tolist(),
        reference="own recorded 6 degree fingertip-pivot pose, transferred in the same hand's palm coordinates",
        anchor_selection="Complete cuff, block 0 of 4; upper-quartile finger reach; prior to evaluation.",
    )
    split_dir = dest / "calibration"
    split_dir.mkdir(exist_ok=True)
    write_json(split_dir / (hand + "_split.json"), split)
    stamps = np.array([int(rows[int(t)]["reference_timestamp_ns"]) for t in times])
    data["_timestamp_ns"] = stamps
    data["_max_dt_ns"] = 1.2 * np.median(np.diff(stamps)[np.diff(times) == 1])
    outward = dest / "outward"
    tracks = outward / "tracks"
    tracks.mkdir(parents=True, exist_ok=True)
    if not (tracks / (hand + ".npz")).is_file():
        ref = fit_from_reference(data, end, hand, pin_tips_3d=True, group=group, anchor=anchor)
        save(tracks / (hand + ".npz"), ref)
        del ref
    # Estimate a fixed cuff triangle in the same local coordinate gauge as the subject's initial calibration.
    complete = cuffs[seen & ~held]
    old = data["wrist_local"]
    center = old.mean(0)
    template = old.copy()
    for _ in range(4):
        rotations = complete_rotations(template, complete)
        template = np.median((complete - complete.mean(1, keepdims=True)) @ rotations, axis=0)
        template += center - template.mean(0)
    del data
    base = dest / "baseline"
    base.mkdir(exist_ok=True)
    if not (base / "initial_fit" / (hand + ".npz")).exists():
        fit(
            hand,
            group=group,
            prior_dir=outward,
            dest=base,
            anchor=anchor,
            calibration_mask=~held,
            cuff_template=template,
        )
    if not (base / "tracks" / (hand + ".npz")).exists():
        lock(hand, group=group, prior_dir=outward, dest=base)
    return base, outward, stamps, held, template

def calibrate(group, hand, base, held, template):
    root = output(group) / "calibration"
    z = load(base / "tracks" / (hand + ".npz"))
    faces = z["faces"].astype(int)
    vs = z["vertices"] * 1000
    mount = z["attachment_faces"].copy()
    bary = z["barycentric"].copy()
    off = z["attachment_local_offsets_m"].copy()
    training = np.flatnonzero(~held)
    sample = training[np.linspace(0, len(training) - 1, min(100, len(training))).round().astype(int)]
    mesh = vs[sample]
    edges = np.unique(
        np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [0, 2]]]), axis=1), axis=0
    )
    lengths = np.median(np.linalg.norm(mesh[:, edges[:, 0]] - mesh[:, edges[:, 1]], axis=-1), axis=0)
    graph = coo_matrix((lengths, (edges[:, 0], edges[:, 1])), shape=(vs.shape[1],) * 2).tocsr()
    normals = units(
        np.cross(
            mesh[:, faces[:, 1]] - mesh[:, faces[:, 0]], mesh[:, faces[:, 2]] - mesh[:, faces[:, 0]]
        )
    )
    report = {}
    for m in range(3, 24):
        j = m - 3
        face = mount[j]
        ok = z["marker_valid"][:, m]
        target = z["target24"][:, m].astype(float) * 1000
        radius = 25 if m == 3 else (20 if m == 4 else 11 if m % 4 == 3 else 14)
        distance = np.min(dijkstra(graph, directed=False, indices=faces[face]), axis=0)
        alignment = np.median((normals * normals[:, face : face + 1]).sum(-1), axis=0)
        candidates = np.flatnonzero((distance[faces].max(1) <= radius) & (alignment >= 0.65))
        assert face in candidates

        def bestfit(f, ids):
            return patch_fit(vs[ids[:, None], faces[f][None]].astype(float), target[ids], tuple(case.params['mount_bounds_mm']))

        ids = sample[ok[sample]]
        train = training[ok[training]]
        fast = sorted((np.mean(bestfit(f, ids)[2] ** 2), int(f)) for f in candidates)
        best = None
        for _, f in fast[:6]:
            b, h, e = bestfit(f, train)
            score = np.mean(e**2)
            if best is None or score < best[0]:
                best = (score, f, b, h)
        _, f, b, h = best
        tri = vs[:, faces[f]]
        pred = (tri * b[None, :, None]).sum(1) + h * units(
            np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        )
        err = np.linalg.norm(pred - target, axis=1)
        mount[j] = f
        bary[j] = b
        off[j] = [0, 0, h / 1000]
        report[str(z["marker_names"][m])] = dict(
            previous_face=int(face),
            face=int(f),
            height_mm=float(h),
            barycentric=b.tolist(),
            train=summary(err[(~held) & ok]),
            held=summary(err[held & ok]),
        )
    write_json(
        root / (hand + "_mounts.json"),
        dict(
            cuff_template_m=template.tolist(), cuff_trained_on_complete_nonholdout=True, markers=report
        ),
    )
    save(
        root / (hand + "_mounts.npz"),
        dict(attachment_faces=mount, barycentric=bary, attachment_local_offsets_m=off),
    )

def fit_hand(group, hand):
    dest = output(group)
    print(group, hand, "prepare", flush=True)
    base, prior, stamps, held, template = prepare(group, hand)
    mounts = dest / "calibration" / (hand + "_mounts.npz")
    if not mounts.exists():
        calibrate(group, hand, base, held, template)
    for mode in ("hard", "soft"):
        if not (dest / mode / (hand + ".npz")).exists():
            run(
                mode,
                hand=hand,
                output_dir=dest / mode,
                group=group,
                base=base,
                prior_dir=prior,
                mount_file=mounts,
                timestamps=stamps,
                cap_guard_mm=0.001,
            )
    print(group, hand, "done", flush=True)
