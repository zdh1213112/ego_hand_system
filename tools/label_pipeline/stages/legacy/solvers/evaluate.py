# Source work/group_010_anatomy_20261119/joint_anatomy_20261119/evaluate.py, SHA-256 06ed62a82a59bfec5f650fecaceff856c24384ccc9a22a04a8c730a46b7cc856

import case_settings as case

import csv

import numpy as np

from anatomy_common import DIGITS, JOINTS, ROOT, geometry, output, percentiles, physical, project

from diagnose import write_csv

def compare_optics(group, hand, tracks, destination=None):
    raw = output(group)
    _, cams, ext, _ = geometry(group)
    with (raw / ("optical_" + hand + ".csv")).open() as source:
        observations = list(csv.DictReader(source))
    for obs in observations:
        for key in ("sync", "camera", "marker"):
            obs[key] = int(obs[key])
        for key in ("u", "v"):
            obs[key] = float(obs[key])
    # These transforms were fitted on raw brightness measurements in the other cameras.
    for subset in sorted({r["subset"] for r in observations}):
        folder = raw / "optical" / (hand + "_" + subset)
        with np.load(folder / "transforms.npz") as calibration:
            transforms = {
                tuple(k): (r, t)
                for k, r, t in zip(
                    calibration["sync_camera"], calibration["rotation"], calibration["translation_m"]
                )
            }
        for c in range(4):
            rows = [r for r in observations if r["subset"] == subset and r["camera"] == c]
            if not rows:
                continue
            times = np.array([r["sync"] for r in rows])
            mounts = np.array([r["marker"] for r in rows])
            xy = np.array([[r["u"], r["v"]] for r in rows])
            rotations, translations = map(np.stack, zip(*(transforms[t, c] for t in times)))
            for name, z in tracks.items():
                ids = np.searchsorted(z["pair_indices"], times)
                np.testing.assert_equal(z["pair_indices"][ids], times)
                assert z["marker_valid"][ids, mounts].all()
                physical_mount = physical(z["predicted24"][ids, mounts].astype(float), hand)
                xyz = np.einsum("nij,nj->ni", rotations, physical_mount) + translations
                err = np.linalg.norm(project(xyz, ext[c], cams[c], 1) - xy, axis=1)
                for row, e in zip(rows, err):
                    row[name + "_px"] = float(e)
            old = [float(r["soft_px"]) if r["soft_px"] else np.nan for r in rows]
            np.testing.assert_allclose(
                [r["soft35_px"] for r in rows], old, atol=0.00001, rtol=0, equal_nan=True
            )
    sets = dict(
        all=[r for r in observations if r["subset"] == "wrist_nail" or r["marker"] % 4 != 3],
        wrist=[r for r in observations if r["marker"] < 3],
        nails=[r for r in observations if r["subset"] == "wrist_nail" and r["marker"] >= 3],
        four=[r for r in observations if r["subset"] != "wrist_nail"],
    )
    unique = [(r["sync"], r["camera"], r["marker"]) for r in sets["all"]]
    assert len(set(unique)) == len(unique)
    directory = ROOT / group if destination is None else destination
    write_csv(directory / ("held_spots_" + hand + ".csv"), observations)
    answer = {}
    for name in tracks:
        answer[name] = {}
        for subset, rows in sets.items():
            base = np.array([r["soft35_px"] for r in rows])
            test = np.array([r[name + "_px"] for r in rows])
            same = np.isfinite(base) & np.isfinite(test)
            med_delta = np.median(np.where(np.isfinite(test), test, np.inf)) - np.median(
                np.where(np.isfinite(base), base, 0)
            )
            answer[name][subset] = dict(
                observed=len(rows),
                paired=int(same.sum()),
                baseline=percentiles(base[same]),
                candidate=percentiles(test[same]),
                worst_case_delta_px=float(med_delta) if np.isfinite(med_delta) else None,
            )
    return answer

def gate_anatomy(stats, reference):
    failures = []
    for fi, digit in enumerate(DIGITS):
        items = [
            (JOINTS[fi][ji], stats["hyper_percent"][fi][ji], reference["hyper_percent"][fi][ji])
            for ji in range(3)
        ]
        items.append(("zig", stats["zig_percent"][fi], reference["zig_percent"][fi]))
        for joint, measured, ref in items:
            gate = max(2, 2 * ref)
            if measured > gate:
                failures.append(dict(digit=digit, joint=joint, measured=measured, limit=gate))
    return failures
