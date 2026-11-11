#!/usr/bin/env python3
"""Evaluate recorded PICO head VIO against quality-gated NOKOV, per session.

Frames: N=NOKOV world, P=PICO raw local world, B=NOKOV head rigid body,
H=PICO reported head frame. T_A_B maps B to A. Fit N_B(t) X = Y P_H(t),
where X=T_B_H and Y=T_N_P. All spatial translations use metres; scale is 1.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import itertools
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from check_pico_capture_quality import analyze_capture, number
from ego_data.fixed_relations import error_summary, pose_errors
from pico_sync_quality import (
    full_motion_quality,
    motion_window_quality,
    nearest_time_pairs,
)
from scipy.optimize import least_squares, minimize_scalar
from scipy.spatial.transform import Rotation, Slerp


@dataclass
class Series:
    rows: list
    t: np.ndarray
    host: np.ndarray
    poses: np.ndarray
    good: np.ndarray
    raw_origin: int
    reasons: list


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if any(None in row or any(v is None for v in row.values()) for row in rows):
        raise ValueError(f"malformed CSV: {path}")
    return rows


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inv(transform):
    result = np.zeros_like(transform)
    result[..., 3, 3] = 1
    result[..., :3, :3] = np.swapaxes(transform[..., :3, :3], -1, -2)
    result[..., :3, 3] = -np.einsum(
        "...ij,...j->...i", result[..., :3, :3], transform[..., :3, 3]
    )
    return result


def pack(matrix):
    return np.r_[Rotation.from_matrix(matrix[:3, :3]).as_rotvec(), matrix[:3, 3]]


def unpack(parameters):
    result = np.eye(4)
    result[:3, :3] = Rotation.from_rotvec(parameters[:3]).as_matrix()
    result[:3, 3] = parameters[3:6]
    return result


def average(matrices):
    result = np.eye(4)
    result[:3, :3] = Rotation.from_matrix(matrices[:, :3, :3]).mean().as_matrix()
    result[:3, 3] = matrices[:, :3, 3].mean(axis=0)
    return result


def load_series(path, pico, body="pico_head"):
    rows = read_csv(path)
    if not pico:
        rows = [r for r in rows if r["rigid_body_name"] == body]
    if len(rows) < 30:
        raise ValueError(f"too few poses in {path}: {len(rows)}")
    key = "head_timestamp_ns" if pico else "device_timestamp_raw"
    raw = np.array([int(r[key]) for r in rows], dtype=np.int64)
    host = np.array(
        [int(r["host_receive_perf_ns" if pico else "receive_perf_ns"]) for r in rows],
        dtype=np.int64,
    )
    if np.any(np.diff(raw) <= 0) or np.any(np.diff(host) <= 0):
        raise ValueError(f"timestamps must increase without sorting: {path}")
    if not pico and not 10 <= np.median(np.diff(raw)) <= 12:
        raise ValueError("NOKOV millisecond clock inference failed")
    t = (raw - raw[0]) * (1e-9 if pico else 0.001)
    if abs((host[-1] - host[0]) * 1e-9 / t[-1] - 1) > 0.02:
        raise ValueError(f"device/host elapsed times disagree: {path}")
    fields = ("x_m", "y_m", "z_m") if pico else ("x_mm", "y_mm", "z_mm")
    positions = np.array(
        [[number(r[k]) for k in fields] for r in rows], dtype=float
    ) * (1 if pico else 0.001)
    q = np.array(
        [[number(r[k]) for k in ("qx", "qy", "qz", "qw")] for r in rows], dtype=float
    )
    good = np.array([r["valid_numeric"] == "1" for r in rows])
    good &= np.isfinite(positions).all(axis=1) & (abs(positions) < 1000).all(axis=1)
    good &= np.isfinite(q).all(axis=1) & (abs(np.linalg.norm(q, axis=1) - 1) < 0.001)
    q[~good] = [0, 0, 0, 1]
    poses = np.tile(np.eye(4), (len(rows), 1, 1))
    poses[:, :3, :3] = Rotation.from_quat(q).as_matrix()
    poses[:, :3, 3] = np.where(good[:, None], positions, 0)
    return Series(
        rows,
        t,
        host,
        poses,
        good,
        int(raw[0]),
        [set() if g else {"invalid_numeric"} for g in good],
    )


def grow_bad(times, bad, guard):
    prefix = np.r_[0, np.cumsum(bad)]
    left = np.searchsorted(times, times - guard, side="left")
    right = np.searchsorted(times, times + guard, side="right")
    return prefix[right] - prefix[left] > 0


def gate_reference(
    series, marker_path, body="pico_head", rms_limit_mm=1.0, guard_s=0.1
):
    index = {int(r["frame_no"]): i for i, r in enumerate(series.rows)}
    records = [r for r in read_csv(marker_path) if r["rigid_body_name"] == body]
    ids = sorted({int(r["marker_id"]) for r in records})
    if len(ids) < 3:
        raise ValueError("not enough head rigid markers")
    observed = np.full((len(series.t), len(ids), 3), np.nan)
    seen = set()
    for row in records:
        i, j = index[int(row["frame_no"])], ids.index(int(row["marker_id"]))
        if (i, j) in seen:
            raise ValueError("duplicate rigid marker key")
        seen.add((i, j))
        point = np.array(
            [number(row[k]) for k in ("x_mm", "y_mm", "z_mm")], dtype=float
        )
        if (
            row["valid"] == "1"
            and np.isfinite(point).all()
            and (abs(point) < 1e6).all()
            and np.linalg.norm(point) > 0
        ):
            observed[i, j] = point * 0.001
    residual = np.full(observed.shape[:2], np.nan)
    template = {}
    rotations = Rotation.from_matrix(series.poses[:, :3, :3])
    for j in range(len(ids)):
        valid = np.isfinite(observed[:, j]).all(axis=1) & series.good
        local = (
            rotations[valid]
            .inv()
            .apply(observed[valid, j] - series.poses[valid, :3, 3])
        )
        if not len(local):
            raise ValueError("head marker never valid")
        template[str(ids[j])] = (np.median(local, axis=0) * 1000).tolist()
        residual[valid, j] = (
            np.linalg.norm(local - np.median(local, axis=0), axis=1) * 1000
        )
    count = np.isfinite(residual).sum(axis=1)
    rms = np.sqrt(
        np.divide(
            np.nansum(residual**2, axis=1),
            count,
            out=np.full(len(count), np.nan),
            where=count > 0,
        )
    )
    position_steps = (
        np.linalg.norm(np.diff(series.poses[:, :3, 3], axis=0), axis=1) * 1000
    )
    angles = np.degrees((rotations[:-1].inv() * rotations[1:]).magnitude())
    jumps = (position_steps > 50) | (angles > 5) | (np.diff(series.t) > 0.03)
    endpoints = np.r_[False, jumps] | np.r_[jumps, False]
    for i in range(len(series.t)):
        if count[i] != len(ids):
            series.reasons[i].add("incomplete_markers")
        if not np.isfinite(rms[i]) or rms[i] > rms_limit_mm:
            series.reasons[i].add("marker_rms")
        if endpoints[i]:
            series.reasons[i].add("reference_step_or_gap")
    initial = np.array([bool(r) for r in series.reasons])
    bad = grow_bad(series.t, initial, guard_s)
    for i in np.flatnonzero(bad & ~initial):
        series.reasons[i].add("reference_guard")
    series.good = ~bad
    return {
        "expected_markers": len(ids),
        "raw_rejected_frames": int(initial.sum()),
        "guarded_rejected_frames": int(bad.sum()),
        "total_frames": len(series.t),
        "marker_rms_limit_mm": rms_limit_mm,
        "guard_s": guard_s,
        "marker_frame_rms_mm": rms,
        "valid_marker_count": count,
        "marker_template_body_mm": template,
    }


def affine_fit(device_t, host_t):
    keep = np.ones(len(device_t), bool)
    for _ in range(8):
        a, b = np.linalg.lstsq(
            np.c_[device_t[keep], np.ones(keep.sum())], host_t[keep], rcond=None
        )[0]
        residual = host_t - (a * device_t + b)
        center = np.median(residual[keep])
        mad = np.median(abs(residual[keep] - center))
        updated = abs(residual - center) <= max(0.0001, 5 * 1.4826 * mad)
        if np.array_equal(updated, keep) or updated.sum() < 30:
            break
        keep = updated
    return (
        float(a),
        float(b),
        {
            "inliers": int(keep.sum()),
            "residual_p95_ms": float(np.percentile(abs(residual[keep]), 95) * 1000),
        },
    )


def host_clock(pico, nokov):
    origin = min(pico.host[0], nokov.host[0])
    ap, bp, sp = affine_fit(pico.t, (pico.host - origin) * 1e-9)
    an, bn, sn = affine_fit(nokov.t, (nokov.host - origin) * 1e-9)
    return (
        ap / an,
        (bp - bn) / an,
        {
            "pico_host_fit": sp,
            "nokov_host_fit": sn,
            "host_relative_rate_ppm": (ap / an - 1) * 1e6,
        },
    )


def support(series, times, radius=0, good=None):
    times = np.asarray(times)
    low, high = times - radius, times + radius
    i = np.clip(np.searchsorted(series.t, low, side="right") - 1, 0, len(series.t) - 1)
    j = np.clip(np.searchsorted(series.t, high, side="left"), 0, len(series.t) - 1)
    prefix = np.r_[0, np.cumsum(~(series.good if good is None else good))]
    gaps = np.r_[0, np.cumsum(np.diff(series.t) > 0.05)]
    return (
        (low >= series.t[0])
        & (high <= series.t[-1])
        & ((prefix[j + 1] - prefix[i]) == 0)
        & ((gaps[j] - gaps[i]) == 0)
    )


def interpolate(series, times):
    clipped = np.clip(times, series.t[0], series.t[-1])
    result = np.tile(np.eye(4), (len(times), 1, 1))
    result[:, :3, :3] = Slerp(series.t, Rotation.from_matrix(series.poses[:, :3, :3]))(
        clipped
    ).as_matrix()
    result[:, :3, 3] = np.column_stack(
        [np.interp(clipped, series.t, series.poses[:, k, 3]) for k in range(3)]
    )
    return result


def angular_speed(series, times, half_window=0.035):
    rotation = Slerp(series.t, Rotation.from_matrix(series.poses[:, :3, :3]))
    before = rotation(np.clip(times - half_window, series.t[0], series.t[-1]))
    after = rotation(np.clip(times + half_window, series.t[0], series.t[-1]))
    values = (before.inv() * after).magnitude() / (2 * half_window)
    values[~support(series, times, half_window)] = np.nan
    return values


def synchronize(pico, nokov, calibration_s=20.0, max_offset_s=0.5):
    a, b, clock = host_clock(pico, nokov)
    ng = np.arange(0.04, nokov.t[-1] - 0.04, 0.005)
    nw = angular_speed(nokov, ng)
    pg = np.arange(0.04, pico.t[-1] - 0.04, 0.02)
    pw = angular_speed(pico, pg)
    fit = (pg <= calibration_s) & np.isfinite(pw)
    fit &= support(nokov, a * pg + b, max_offset_s + 0.04)
    if fit.sum() < 100 or np.std(pw[fit]) < 0.05:
        raise ValueError("insufficient angular motion for clock estimation")

    def cost(delta):
        candidate = a * np.interp(a * pg[fit] + b + delta, ng, nw)
        return float(np.mean((candidate - pw[fit]) ** 2))

    offsets = np.linspace(-max_offset_s, max_offset_s, 501)
    costs = np.array([cost(d) for d in offsets])
    best = int(np.argmin(costs))
    lower, upper = offsets[max(best - 2, 0)], offsets[min(best + 2, len(offsets) - 1)]
    solution = minimize_scalar(
        cost, bounds=(lower, upper), method="bounded", options={"xatol": 1e-8}
    )
    delta = float(solution.x)
    mapped = a * pg + b + delta
    match = a * np.interp(mapped, ng, nw, left=np.nan, right=np.nan)
    valid = np.isfinite(pw) & np.isfinite(match) & support(nokov, mapped, 0.04)
    correlations = {}
    for label, mask in [
        ("calibration", fit),
        ("validation", valid & (pg > calibration_s + 0.2)),
    ]:
        keep = mask & valid
        correlations[label] = {
            "samples": int(keep.sum()),
            "correlation": float(np.corrcoef(pw[keep], match[keep])[0, 1])
            if keep.sum() > 10
            else None,
            "speed_rmse_rad_s": float(np.sqrt(np.mean((pw[keep] - match[keep]) ** 2)))
            if keep.any()
            else None,
        }
    clock.update(
        {
            "equation": "t_N_relative_s = scale * t_P_relative_s + host_offset_s + motion_offset_s",
            "scale": a,
            "host_offset_s": b,
            "motion_offset_s": delta,
            "total_offset_s": b + delta,
            "calibration_seconds": calibration_s,
            "search_limit_s": max_offset_s,
            "boundary_optimum": abs(delta) > 0.98 * max_offset_s,
            "motion_validation": correlations,
            "all_quality_supported": {
                "samples": int(valid.sum()),
                "correlation": float(np.corrcoef(pw[valid], match[valid])[0, 1])
                if valid.sum() > 10
                else None,
                "speed_rmse_rad_s": float(
                    np.sqrt(np.mean((pw[valid] - match[valid]) ** 2))
                )
                if valid.any()
                else None,
            },
        }
    )
    diagnostic = {
        "pico_time_s": pg,
        "pico_angular_speed": pw,
        "nokov_angular_speed": match,
        "offset_candidates_s": offsets,
        "offset_cost": costs,
        "fit_common_support": fit,
        "reference_supported": valid,
        "nokov_grid_relative_s": ng,
        "nokov_grid_angular_speed": nw,
    }
    return a, b + delta, clock, diagnostic


def fit_extrinsics(n, p, times):
    av, bv, ta, tb, ra = [], [], [], [], []
    for lag in (0.25, 0.5, 1.0, 2.0, 4.0):
        right = np.searchsorted(times, times + lag)
        left = np.flatnonzero(right < len(times))
        right = right[left]
        aa, bb = inv(n[left]) @ n[right], inv(p[left]) @ p[right]
        va, vb = (
            Rotation.from_matrix(aa[:, :3, :3]).as_rotvec(),
            Rotation.from_matrix(bb[:, :3, :3]).as_rotvec(),
        )
        select = (np.linalg.norm(va, axis=1) > 0.08) & (
            np.linalg.norm(vb, axis=1) > 0.08
        )
        select &= (np.linalg.norm(va, axis=1) < 2.8) & (
            np.linalg.norm(vb, axis=1) < 2.8
        )
        av.extend(va[select])
        bv.extend(vb[select])
        ta.extend(aa[select, :3, 3])
        tb.extend(bb[select, :3, 3])
        ra.extend(aa[select, :3, :3])
    if len(av) < 20:
        raise ValueError("insufficient hand-eye motion pairs")
    av, bv, ra = np.asarray(av), np.asarray(bv), np.asarray(ra)
    singular = np.linalg.svd(bv, compute_uv=False)
    if singular[-1] / singular[0] < 0.03:
        raise ValueError("head rotations do not constrain three-axis mount")
    rx, _ = Rotation.align_vectors(av, bv)
    design = (ra - np.eye(3)).reshape(-1, 3)
    translation = (rx.apply(tb) - ta).reshape(-1)
    tx, _, rank, st = np.linalg.lstsq(design, translation, rcond=None)
    if rank < 3:
        raise ValueError("mount translation is not observable")
    x = np.eye(4)
    x[:3, :3], x[:3, 3] = rx.as_matrix(), tx
    y = average(n @ x @ inv(p))

    def residual(parameters):
        xx, yy = unpack(parameters[:6]), unpack(parameters[6:12])
        left, right = n @ xx, yy @ p
        dr = Rotation.from_matrix(
            np.swapaxes(left[:, :3, :3], 1, 2) @ right[:, :3, :3]
        ).as_rotvec()
        dp = right[:, :3, 3] - left[:, :3, 3]
        return np.c_[dp / 0.01, dr / np.radians(1.0)].ravel()

    fit = least_squares(
        residual,
        np.r_[pack(x), pack(y)],
        loss="soft_l1",
        f_scale=1.0,
        max_nfev=200,
        ftol=1e-10,
        xtol=1e-10,
        gtol=1e-10,
    )
    jac_singular = np.linalg.svd(fit.jac, compute_uv=False)
    details = {
        "fit_poses": len(times),
        "motion_pairs": len(av),
        "rotation_excitation_singular_values": singular.tolist(),
        "rotation_excitation_ratio": float(singular[-1] / singular[0]),
        "translation_design_singular_values": st.tolist(),
        "jacobian_rank": int(np.linalg.matrix_rank(fit.jac)),
        "jacobian_condition": float(jac_singular[0] / jac_singular[-1]),
        "optimizer_success": bool(fit.success),
        "optimizer_message": str(fit.message),
        "scale_fitted": False,
        "loss": "soft_l1, position scale 10 mm, rotation scale 1 degree",
    }
    if not fit.success or details["jacobian_rank"] < 12:
        raise ValueError("joint mount/world calibration failed or is rank deficient")
    return unpack(fit.x[:6]), unpack(fit.x[6:12]), details


def metrics(estimate, reference, selection):
    pe, ae = pose_errors(
        estimate,
        reference[:, :3, 3],
        Rotation.from_matrix(reference[:, :3, :3]),
        selection,
    )
    result = error_summary(pe, ae, selection)
    for key in ("position_rmse", "position_median", "position_max"):
        value = result.pop(key + "_m")
        result[key + "_mm"] = value * 1000 if value is not None else None
    result["position_p95_mm"] = (
        float(np.percentile(pe[selection], 95) * 1000) if result["poses"] else None
    )
    result["orientation_p95_deg"] = (
        float(np.percentile(ae[selection], 95)) if result["poses"] else None
    )
    return result, pe, ae


def analyze_session(
    path, calibration_s=20.0, body="pico_head", marker_rms_mm=1.0, guard_s=0.1
):
    paths = [
        path / "pico/pico_head_poses.csv",
        path / "pico/pico_tracker_poses.csv",
        path / "pico/capture_metadata.json",
        path / "nokov/nokov_rigid_bodies.csv",
        path / "nokov/nokov_rigid_body_markers.csv",
        path / "nokov/capture_metadata.json",
    ]
    paths += [
        p for p in (path / "pico/events.csv", path / "nokov/events.csv") if p.is_file()
    ]
    hashes = {str(p): sha256(p) for p in paths}
    pico = load_series(paths[0], True)
    nokov = load_series(paths[3], False, body)
    numeric_reference = nokov.good.copy()
    pico_integrity, _, _ = analyze_capture(path)
    quality = gate_reference(nokov, paths[4], body, marker_rms_mm, guard_s)
    if pico.t[-1] < calibration_s + 5:
        raise ValueError("need at least five seconds of held-out data")
    a, b, clock, sync_data = synchronize(pico, nokov, calibration_s)
    target = a * pico.t + b
    n = interpolate(nokov, target)
    valid = pico.good & support(nokov, target)
    pairing_summary, pairing_data = nearest_time_pairs(pico.t, nokov.t, a, b, valid)
    motion_time_quality = motion_window_quality(
        sync_data["pico_time_s"],
        sync_data["pico_angular_speed"],
        sync_data["nokov_grid_relative_s"],
        sync_data["nokov_grid_angular_speed"],
        a,
        b,
        lambda t, radius: support(nokov, t, radius),
    )
    full_time_quality = full_motion_quality(
        sync_data["pico_time_s"],
        sync_data["pico_angular_speed"],
        sync_data["nokov_grid_relative_s"],
        sync_data["nokov_grid_angular_speed"],
        a,
        b,
        lambda t, radius: support(nokov, t, radius),
    )
    full_time_quality["nokov_source"] = (
        f"nokov/nokov_rigid_bodies.csv: rigid_body_name={body}"
    )
    train = valid & (pico.t <= calibration_s)
    validation = valid & (pico.t > calibration_s + 0.2)
    indices = np.flatnonzero(train)
    chosen = []
    for i in indices:
        if not chosen or pico.t[i] - pico.t[chosen[-1]] >= 0.1:
            chosen.append(i)
    chosen = np.array(chosen)
    if len(chosen) < 80 or validation.sum() < 100:
        raise ValueError("too few usable calibration/validation samples")
    x, y, fit = fit_extrinsics(n[chosen], pico.poses[chosen], pico.t[chosen])
    reference, estimate = n @ x, y @ pico.poses
    groups = {}
    for label, mask in [
        ("all_quality_supported", valid),
        ("calibration", train),
        ("validation", validation),
    ]:
        groups[label], _, _ = metrics(estimate, reference, mask)
    unfiltered = pico.good & support(nokov, target, good=numeric_reference)
    groups["ungated_reference_diagnostic"], _, _ = metrics(
        estimate, reference, unfiltered
    )
    _, pe, ae = metrics(estimate, reference, valid)
    # One-second RPE uses fixed calibration and fully supported intervals.
    right = np.searchsorted(pico.t, pico.t + 1.0)
    left = np.flatnonzero(right < len(pico.t))
    right = right[left]
    middle = (pico.t[left] + pico.t[right]) * 0.5
    half_span = (pico.t[right] - pico.t[left]) * 0.5
    select = (
        validation[left] & validation[right] & (pico.t[right] - pico.t[left] < 1.03)
    )
    select &= support(pico, middle, half_span) & support(
        nokov, a * middle + b, a * half_span
    )
    rpe_gt, rpe_est = (
        inv(reference[left]) @ reference[right],
        inv(estimate[left]) @ estimate[right],
    )
    groups["validation_rpe_1s"], _, _ = metrics(rpe_est, rpe_gt, select)
    correlations = clock["motion_validation"]
    status = (
        "ok"
        if all(
            v["correlation"] is not None and v["correlation"] > 0.9
            for v in correlations.values()
        )
        and not clock["boundary_optimum"]
        else "needs_review"
    )
    summary = {
        "session": path.name,
        "directory": str(path),
        "status": status,
        "head_rigid_body": body,
        "frames": {
            "N": "NOKOV world",
            "P": "PICO raw local world P0",
            "B": body,
            "H": "PICO reported head frame",
        },
        "equation": "T_N_B(t_N) @ T_B_H = T_N_P @ T_P_H(t_P)",
        "transform_convention": "T_A_B maps B into A; translations in m; quaternions xyzw",
        "T_B_H": x.tolist(),
        "T_H_B": inv(x).tolist(),
        "T_N_P": y.tolist(),
        "T_P_N": inv(y).tolist(),
        "clock": clock,
        "time_sync_quality": {
            "nearest_raw_pairing": pairing_summary,
            "motion_window_residuals": motion_time_quality,
            "full_sequence_motion_residual": full_time_quality,
        },
        "fit": fit,
        "calibration_interval_pico_s": [0, calibration_s],
        "validation_interval_pico_s": [calibration_s + 0.2, float(pico.t[-1])],
        "metrics": groups,
        "pico_raw_head_rows": len(pico.t),
        "usable_head_rows": int(valid.sum()),
        "reference_coverage_percent": float(valid.mean() * 100),
        "quality": {k: v for k, v in quality.items() if not isinstance(v, np.ndarray)},
        "pico_capture_integrity_status": pico_integrity["status"],
        "pico_device_origin_ns": pico.raw_origin,
        "nokov_device_origin_raw_ms": nokov.raw_origin,
        "input_sha256": hashes,
        "coordinate_remap_applied": False,
        "metric_scale": 1.0,
        "raw_inputs_unchanged": all(sha256(Path(p)) == h for p, h in hashes.items()),
    }
    data = {
        "pico": pico,
        "nokov": nokov,
        "reference": reference,
        "estimate": estimate,
        "valid": valid,
        "training": train,
        "validation": validation,
        "target": target,
        "time_pairing": pairing_data,
        "position_error_m": pe,
        "angle_error_deg": ae,
        "quality": quality,
        "sync": sync_data,
        "world_per_frame": n @ x @ inv(pico.poses),
        "T_P_B": inv(y) @ n,
    }
    data["pico_integrity"] = pico_integrity
    return summary, data


def compare_transforms(results):
    rows = []
    for first, second in itertools.combinations(results, 2):
        first_template = first.get("quality", {}).get("marker_template_body_mm", {})
        second_template = second.get("quality", {}).get("marker_template_body_mm", {})
        common_markers = sorted(set(first_template) & set(second_template))
        template_rms = (
            float(
                np.sqrt(
                    np.mean(
                        [
                            np.sum(
                                (np.asarray(first_template[k]) - second_template[k])
                                ** 2
                            )
                            for k in common_markers
                        ]
                    )
                )
            )
            if common_markers
            else None
        )
        for name in ("T_B_H", "T_N_P", "T_P_N"):
            relative = inv(np.asarray(first[name])) @ np.asarray(second[name])
            rows.append(
                {
                    "session_i": first["session"],
                    "session_j": second["session"],
                    "transform": name,
                    "translation_difference_mm": float(
                        np.linalg.norm(relative[:3, 3]) * 1000
                    ),
                    "rotation_difference_deg": float(
                        np.degrees(Rotation.from_matrix(relative[:3, :3]).magnitude())
                    ),
                    "relative_transform_i_inverse_j": relative.tolist(),
                    "translation_delta_xyz_mm": (relative[:3, 3] * 1000).tolist(),
                    "body_marker_template_rms_difference_mm": template_rms,
                }
            )
    return rows


def safe_json(value):
    if isinstance(value, dict):
        return {k: safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return safe_json(value.tolist())
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def save_json(path, value):
    path.write_text(
        json.dumps(safe_json(value), ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def save_csv(path, rows, fields=None):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def export_session(output, result, data):
    output.mkdir(parents=True, exist_ok=True)
    save_json(output / "calibration.json", result)
    save_json(output / "pico_capture_quality.json", data["pico_integrity"])
    save_json(output / "time_sync_quality.json", result["time_sync_quality"])
    save_csv(
        output / "motion_time_residuals.csv",
        result["time_sync_quality"]["motion_window_residuals"]["windows"],
    )
    np.savez_compressed(output / "synchronization_diagnostics.npz", **data["sync"])
    reference_rows, pico_rows = [], []
    n, p = data["nokov"], data["pico"]
    q = data["quality"]
    for i, row in enumerate(n.rows):
        reference_rows.append(
            {
                "frame_no": row["frame_no"],
                "device_timestamp_raw_ms": row["device_timestamp_raw"],
                "time_s": n.t[i],
                "valid_markers": q["valid_marker_count"][i],
                "marker_frame_rms_mm": q["marker_frame_rms_mm"][i],
                "reference_usable": int(n.good[i]),
                "reasons": ";".join(sorted(n.reasons[i])),
            }
        )
    estimate, reference = data["estimate"], data["reference"]
    qe = Rotation.from_matrix(estimate[:, :3, :3]).as_quat()
    qr = Rotation.from_matrix(reference[:, :3, :3]).as_quat()
    world_rows, body_rows = [], []
    for i, row in enumerate(p.rows):
        valid = bool(data["valid"][i])
        record = {
            "sample_index": row["sample_index"],
            "pico_timestamp_ns": row["head_timestamp_ns"],
            "pico_relative_s": p.t[i],
            "nokov_target_relative_s": data["target"][i],
            "reference_usable": int(valid),
            "split": "calibration"
            if data["training"][i]
            else "validation"
            if data["validation"][i]
            else "excluded_or_transition",
            "position_error_mm": data["position_error_m"][i] * 1000 if valid else "",
            "orientation_error_deg": data["angle_error_deg"][i] if valid else "",
        }
        for label, position, quat in [
            ("pico_estimate", estimate[i, :3, 3], qe[i]),
            ("nokov_reference", reference[i, :3, 3], qr[i]),
        ]:
            for k, v in zip(
                ("x_m", "y_m", "z_m", "qx", "qy", "qz", "qw"), np.r_[position, quat]
            ):
                record[label + "_" + k] = v if valid or label == "pico_estimate" else ""
        pico_rows.append(record)
        if valid:
            common = {"sample_index": row["sample_index"], "pico_relative_s": p.t[i]}
            world_rows.append(
                {
                    **common,
                    **{
                        f"m{r}{c}": data["world_per_frame"][i, r, c]
                        for r in range(4)
                        for c in range(4)
                    },
                }
            )
            body_rows.append(
                {
                    **common,
                    **{
                        f"m{r}{c}": data["T_P_B"][i, r, c]
                        for r in range(4)
                        for c in range(4)
                    },
                }
            )
    save_csv(output / "reference_quality.csv", reference_rows)
    save_csv(output / "head_errors.csv", pico_rows)
    save_csv(output / "T_N_P_per_frame.csv", world_rows)
    save_csv(output / "T_P_B_per_frame.csv", body_rows)
    pairing = data["time_pairing"]
    paired_rows = []
    for i, row in enumerate(p.rows):
        inside = bool(pairing["inside"][i])
        j, left, right = (int(pairing[key][i]) for key in ("nearest", "left", "right"))
        paired_rows.append(
            {
                "pico_sample_index": row["sample_index"],
                "pico_device_timestamp_ns": row["head_timestamp_ns"],
                "pico_relative_s": p.t[i],
                "mapped_nokov_relative_s": pairing["target_s"][i],
                "inside_nokov_recording": int(inside),
                "used_for_statistics": int(pairing["used"][i]),
                "nearest_nokov_frame_no": n.rows[j]["frame_no"] if inside else "",
                "nearest_nokov_device_timestamp_raw_ms": n.rows[j][
                    "device_timestamp_raw"
                ]
                if inside
                else "",
                "nearest_nokov_relative_s": n.t[j] if inside else "",
                "pair_delta_ms": pairing["delta_s"][i] * 1000 if inside else "",
                "absolute_pair_delta_ms": abs(pairing["delta_s"][i]) * 1000
                if inside
                else "",
                "left_nokov_frame_no": n.rows[left]["frame_no"] if inside else "",
                "right_nokov_frame_no": n.rows[right]["frame_no"] if inside else "",
                "bracket_gap_ms": (n.t[right] - n.t[left]) * 1000 if inside else "",
            }
        )
    save_csv(output / "time_pairing.csv", paired_rows)


def make_plots(output, results, datasets):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "Noto Sans CJK JP",
            "axes.unicode_minus": False,
            "font.size": 10,
        }
    )
    columns = len(results)
    fig, axes = plt.subplots(
        3, columns, figsize=(6 * columns, 10), squeeze=False, constrained_layout=True
    )
    for col, (result, d) in enumerate(zip(results, datasets)):
        t = d["pico"].t
        axes[0, col].plot(t, d["position_error_m"] * 1000, lw=0.8)
        axes[0, col].set_title(result["session"][-3:] + " · 头显 VIO")
        axes[1, col].plot(t, d["angle_error_deg"], lw=0.8)
        sync = d["sync"]
        supported = sync["reference_supported"]
        axes[2, col].plot(
            sync["pico_time_s"],
            np.where(supported, sync["pico_angular_speed"], np.nan),
            label="PICO",
            lw=0.8,
        )
        axes[2, col].plot(
            sync["pico_time_s"],
            np.where(supported, sync["nokov_angular_speed"], np.nan),
            label="NOKOV",
            lw=0.7,
            alpha=0.75,
        )
        axes[2, col].legend()
        for r in range(3):
            axes[r, col].grid(alpha=0.2)
            axes[r, col].set_xlabel("PICO 首样本后 / s")
    for r, label in enumerate(
        ("位置误差 / mm", "姿态误差 / °", "中心差分角速度模长 / rad/s")
    ):
        axes[r, 0].set_ylabel(label)
    fig.suptitle("整段参考有效样本的头显误差与角速度对比")
    fig.savefig(output / "errors_and_sync.png", dpi=145)
    plt.close(fig)
    fig = plt.figure(figsize=(6 * columns, 6), layout="constrained")
    for col, (result, d) in enumerate(zip(results, datasets)):
        ax = fig.add_subplot(1, columns, col + 1, projection="3d")
        valid = d["valid"]
        for label, poses, color in [
            ("PICO VIO", d["estimate"], "#2866b3"),
            ("NOKOV 参考", d["reference"], "#cf7725"),
        ]:
            points = poses[:, :3, 3].copy()
            points[~valid] = np.nan
            ax.plot(
                points[:, 0],
                points[:, 1],
                points[:, 2],
                lw=1,
                alpha=0.8,
                label=label,
                color=color,
            )
        cloud = d["reference"][valid, :3, 3]
        center, extent = np.mean(cloud, axis=0), np.ptp(cloud, axis=0).max() * 0.55
        ax.set_xlim(center[0] - extent, center[0] + extent)
        ax.set_ylim(center[1] - extent, center[1] + extent)
        ax.set_zlim(center[2] - extent, center[2] + extent)
        ax.set_box_aspect((1, 1, 1))
        ax.set_xlabel("N X / m")
        ax.set_ylabel("N Y / m")
        ax.set_zlabel("N Z / m")
        ax.set_title(result["session"][-3:])
        ax.legend(loc="upper left", fontsize=8)
    fig.suptitle("PICO 与 NOKOV 头显轨迹（整段参考有效样本）")
    fig.savefig(
        output / "trajectories.png", dpi=145, bbox_inches="tight", pad_inches=0.3
    )
    plt.close(fig)
    fig, axes = plt.subplots(
        1, columns, figsize=(6 * columns, 4.2), squeeze=False, constrained_layout=True
    )
    all_residuals = [
        abs(w["residual_offset_ms"])
        for r in results
        for w in r["time_sync_quality"]["motion_window_residuals"]["windows"]
        if w["status"] == "ok"
    ]
    extent = max(1.0, np.ceil(max(all_residuals, default=0)))
    for col, (result, d) in enumerate(zip(results, datasets)):
        windows = [
            w
            for w in result["time_sync_quality"]["motion_window_residuals"]["windows"]
            if w["status"] == "ok"
        ]
        axes[0, col].plot(
            [w["center_s"] for w in windows],
            [w["residual_offset_ms"] for w in windows],
            "o-",
            lw=1.1,
        )
        axes[0, col].axhline(0, color="#777", lw=0.9)
        axes[0, col].set_ylim(-extent - 0.5, extent + 0.5)
        axes[0, col].set_xlim(0, d["pico"].t[-1])
        axes[0, col].set_title(result["session"][-3:])
        axes[0, col].set_xlabel("时间段中心 / s")
        axes[0, col].grid(alpha=0.2)
    axes[0, 0].set_ylabel("还需调整的时间 / ms")
    fig.suptitle("对齐后剩余的时间错位（每个点代表约 8 秒的一段运动）")
    fig.savefig(output / "remaining_time_offset.png", dpi=145)
    plt.close(fig)


def html_table(headers, rows):
    return (
        "<table><tr>"
        + "".join("<th>" + html.escape(str(v)) + "</th>" for v in headers)
        + "</tr>"
        + "".join(
            "<tr>"
            + "".join("<td>" + html.escape(str(v)) + "</td>" for v in row)
            + "</tr>"
            for row in rows
        )
        + "</table>"
    )


def time_quality_report(results):
    def fmt(value):
        return f"{value:.3f}" if value is not None else "—"

    full_rows, motion_rows, links = [], [], []
    for result in results:
        quality = result["time_sync_quality"]
        full = quality["full_sequence_motion_residual"]
        reliable = full["status"] == "ok"
        full_rows.append(
            [
                result["session"][-3:],
                fmt(full["residual_offset_ms"]) if reliable else "不可可靠估计",
                fmt(full["absolute_residual_ms"]) if reliable else "—",
            ]
        )
        motion = quality["motion_window_residuals"]
        values = motion["residual_summary"]
        motion_rows.append(
            [
                result["session"][-3:],
                f"{motion['accepted_windows']}/{motion['total_windows']}",
                fmt(values["mean_abs_ms"]),
                fmt(values["max_abs_ms"]),
            ]
        )
        session = html.escape(result["session"], quote=True)
        links.append(
            f'<p>{session}：<a href="{session}/motion_time_residuals.csv">各时间段的错位明细</a> · '
            f'<a href="{session}/time_sync_quality.json">完整数值 JSON</a></p>'
        )
    return [
        "<h2>对齐后还剩多少时间错位</h2>",
        "<h3>整段录制的剩余时间差</h3>",
        "<p><strong>比较的是谁与谁：</strong>PICO 头显 VIO 记录的头部运动，与 NOKOV 的 <code>pico_head</code> 刚体记录的同一次头部运动。PICO 的采样时刻先按现有同步关系映射到 NOKOV 时间域，再通过两边的角速度变化估计还剩多少先后错位。</p>",
        "<p><strong>时间差的方向：</strong>按“NOKOV 同一次头部动作的时刻 − 同步映射后的 PICO 动作时刻”理解。实际数值由整段运动曲线的最佳对齐估计，并非逐个动作人工标注的时间差。</p>",
        "<p>下面的整段值，是使用整个录制中的可靠运动数据，求一个最合适的额外时间调整；不是将各段偏移简单求平均。正值表示还需匹配稍晚的 NOKOV 运动，负值表示稍早。例如 +2 ms，表示现有同步后，整段动作再匹配晚 2 ms 的 NOKOV 运动会更重合。该残余估计没有应用到同步参数中。</p>",
        html_table(
            ["Session", "整段剩余时间差 ms（带方向）", "整段错位大小 ms"], full_rows
        ),
        "<p><a href='full_sequence_time_offset.csv'>下载整段时间差 CSV（包含双方数据来源）</a>。一个整段值概括整体情况，下面的分段曲线用于查看局部变化。</p>",
        "<p>这里报告的是根据同一次头部运动估计的时间差，不是最近邻 PICO/NOKOV 原始帧采样时刻的间距；位姿插值后的查询时刻相等也不代表真实同步误差为零。</p>",
        "<h3>分段检查：整段之内是否有变化</h3>",
        "<p>两套记录已经完成时间对齐。这里把录制分成约 8 秒的一段段运动，再检查：每一段还需把匹配时刻前后调整多少，PICO 与 NOKOV 的动作才能更重合。得到的数值就是该段估计的剩余错位；这些数值没有再次用于修改同步参数。</p>",
        html_table(
            [
                "Session",
                "可分析 / 全部时间段",
                "平均错位大小 ms",
                "最大错位大小 ms",
            ],
            motion_rows,
        ),
        "<p>平均错位大小：各可分析时间段的错位取绝对值后求平均；最大错位大小：这些时间段中最大的错位绝对值。例如 5/6 表示总共 6 段，其中 5 段可用于估计；太短、运动不足或参考不可靠的段不计入统计。</p>",
        '<img src="remaining_time_offset.png" alt="三个 session 对齐后剩余时间错位">',
        "<p><strong>图的读法：</strong>横轴是录制时间，每个点代表约 8 秒的一段运动；纵轴是这段运动还需调整的毫秒数。点越靠近 0，说明该段动作越对齐；正值表示应匹配稍晚的 NOKOV 运动，负值表示应匹配稍早的 NOKOV 运动。例如 +3 ms，表示在现有同步基础上再匹配晚 3 ms 的 NOKOV 运动，两边会更重合。连线仅帮助观察变化，不代表逐帧估计。</p>",
        "<p>若点持续向同一个方向变化，可能有随时间积累的错位；上下波动则表示不同时间段的对齐情况有变化。</p>",
        "<p>这些是根据运动记录估计的剩余错位，受位姿预测和参考噪声影响。没有硬件共同时间真值，不能将它直接当作精确测得的硬件同步误差。</p>",
        *links,
    ]


def write_report(output, results, pairs):
    calibration_s = results[0]["calibration_interval_pico_s"][1]
    content = [
        "<h1>PICO 头显 VIO 与 NOKOV：逐 session 评价及转换关系偏差</h1>",
        f"<p>分析对象为 PICO 已输出的头显位姿。每个 session 用前 {calibration_s:g} 秒独立估计时间补偿及安装与世界关系，固定参数后，统计整段录制中所有参考有效样本的指标。尺度固定为 1，未补帧，未根据 PICO/NOKOV 残差剔除评价样本。</p>",
        "<h2>整段参考有效样本性能</h2>",
    ]
    rows = []
    for r in results:
        m = r["metrics"]["all_quality_supported"]
        rows.append(
            [
                r["session"][-3:],
                m["poses"],
                f"{r['reference_coverage_percent']:.2f}%",
                f"{m['position_rmse_mm']:.3f}",
                f"{m['position_p95_mm']:.3f}",
                f"{m['position_max_mm']:.3f}",
                f"{m['orientation_rmse_deg']:.3f}",
                f"{m['orientation_p95_deg']:.3f}",
                r["status"],
            ]
        )
    content.append(
        html_table(
            [
                "Session",
                "参考有效样本",
                "全段参考覆盖率",
                "位置 RMSE mm",
                "位置 p95 mm",
                "位置 max mm",
                "姿态 RMSE °",
                "姿态 p95 °",
                "同步/可观测性",
            ],
            rows,
        )
    )
    content += [
        '<img src="errors_and_sync.png" alt="误差和同步"><img src="trajectories.png" alt="轨迹">',
        "<h2>坐标关系与两两偏差</h2><p>N：NOKOV 世界，P：PICO 原始局部世界 P0，B：NOKOV 的 pico_head 刚体坐标，H：PICO 输出的头显坐标。T_A_B 把 B 中的坐标转换到 A，矩阵平移单位 m。</p>",
        "<pre>T_N_B(t_N) · X = Y · T_P_H(t_P)\nX = T_B_H        头显安装外参（H → B）\nY = T_N_P        局部世界配准（P → N）\nT_P_N = inverse(Y)\nT_P_B(t) = inverse(Y) · T_N_B(t)  （随头部运动变化）</pre>",
        "<p>头部运动时，刚体 B 到局部世界 P 的关系本来就随时间变化，不能用单个常量矩阵表示。各 session 同时输出 T_P_B_per_frame.csv。下面分别比较物理安装关系 X 和世界配准 Y，避免将两者混为一谈。</p>",
        "<p>偏差定义为 Δ_ij = inverse(T_i) · T_j；平移偏差为 Δ 平移向量模长，旋转偏差为最短旋转角。X 的偏差反映本次数据估计出的安装关系一致性；Y 的偏差还包括局部世界原点、朝向、漂移与标定误差，不能直接称为安装误差。没有独立测量安装真值，X 不是绝对精度认证。</p>",
    ]
    content.append(
        html_table(
            ["Session 对", "关系", "平移偏差 mm", "旋转偏差 °"],
            [
                [
                    p["session_i"][-3:] + " / " + p["session_j"][-3:],
                    p["transform"],
                    f"{p['translation_difference_mm']:.3f}",
                    f"{p['rotation_difference_deg']:.4f}",
                ]
                for p in pairs
            ],
        )
    )
    content.append("<h2>每个 session 的固定转换矩阵</h2>")
    for r in results:
        content.append("<h3>" + html.escape(r["session"]) + "</h3>")
        for name in ("T_B_H", "T_N_P", "T_P_N"):
            content.append(
                "<p>"
                + name
                + "</p><pre>"
                + html.escape(
                    np.array2string(
                        np.asarray(r[name]), precision=8, suppress_small=True
                    )
                )
                + "</pre>"
            )
        content.append(
            f'<p><a href="{r["session"]}/calibration.json">标定与全部指标 JSON</a> · <a href="{r["session"]}/head_errors.csv">逐样本误差</a> · <a href="{r["session"]}/reference_quality.csv">NOKOV 参考质量</a> · <a href="{r["session"]}/T_P_B_per_frame.csv">动态 T_P_B</a></p>'
        )
    content += [
        *time_quality_report(results),
        "<h2>证据范围</h2><ul>",
        "<li>NOKOV Marker 每帧局部 RMS >1 mm、预期点不全、相邻位移 >50 mm、相邻角差 >5° 或时间异常触发参考复核，两端点及前后 0.1 s 默认保护区不参与拟合和评价。这些阈值是经验筛查条件。</li>",
        "<li>同时保存 ungated_reference_diagnostic：固定相同标定，额外显示未使用 Marker 质量门时的参考误差，只用于判断参考异常影响。正式结论使用质量通过区间。</li>",
        "<li>本报告统计固定标定后的整段参考有效样本，包含用于标定的样本，反映本段配准后的整体表现。安装外参并非独立测量真值；时钟拟合和 NOKOV 姿态噪声会影响结果。</li>",
        "<li>PICO v1 位姿格式没有物理跟踪状态。原始采集的 Tracker 异常仍保留在 pico_capture_quality.json；本报告头显结果不会据此被解释成 Tracker 精度。</li>",
        "<li>左右 Tracker 主要近似静止，缺少求解各自完整六自由度安装外参所需的运动激励；本次转换比较针对有充分运动的头显 pico_head。未对静止 Tracker 强行求解安装矩阵。</li>",
        "<li>原始 PICO 四元数直接使用，未套用历史 FLU/CAD 外参。空间尺度固定为 1，未用相似变换吸收 VIO 尺度误差。全部输入保留 SHA-256，原始 CSV 不变。</li>",
        "</ul><p>机器可读汇总：<a href='summary.json'>summary.json</a> · <a href='metrics.csv'>metrics.csv</a> · <a href='transform_differences.csv'>transform_differences.csv</a> · <a href='transform_differences.json'>完整差分矩阵</a></p>",
    ]
    style = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>PICO VIO / NOKOV 分析</title><style>body{max-width:1500px;margin:32px auto;padding:0 20px;font:16px/1.65 system-ui;color:#182637}table{border-collapse:collapse;width:100%;font-size:14px}th,td{border:1px solid #ccd6e1;padding:8px;text-align:left}th{background:#edf2f7}img{max-width:100%;margin:20px 0}pre{background:#eef2f6;padding:14px;overflow:auto}h2{margin-top:36px}li{margin:10px 0}</style>'
    (output / "report.html").write_text(
        style + "\n".join(content) + "</html>", encoding="utf-8"
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sessions", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--calibration-seconds", type=float, default=20.0)
    parser.add_argument("--head-rigid-body", default="pico_head")
    parser.add_argument("--reference-marker-rms-mm", type=float, default=1.0)
    parser.add_argument("--reference-guard-s", type=float, default=0.1)
    args = parser.parse_args(argv)
    if (
        any(
            not np.isfinite(v) or v <= 0
            for v in (args.calibration_seconds, args.reference_marker_rms_mm)
        )
        or not np.isfinite(args.reference_guard_s)
        or args.reference_guard_s < 0
    ):
        parser.error(
            "calibration duration/RMS threshold must be positive; guard must be nonnegative"
        )
    sessions, output = [p.resolve() for p in args.sessions], args.output.resolve()
    if len({p.name for p in sessions}) != len(sessions):
        parser.error("session names must be unique")
    for destination in [output] + [output / p.name for p in sessions]:
        if any(destination == p or p in destination.parents for p in sessions):
            parser.error("output must be outside source sessions")
    results, datasets = [], []
    for session in sessions:
        result, data = analyze_session(
            session,
            args.calibration_seconds,
            args.head_rigid_body,
            args.reference_marker_rms_mm,
            args.reference_guard_s,
        )
        results.append(result)
        datasets.append(data)
        print(
            session.name,
            result["status"],
            json.dumps(result["metrics"]["all_quality_supported"]),
            flush=True,
        )
    output.mkdir(parents=True, exist_ok=True)
    pairs = compare_transforms(results)
    for result, data in zip(results, datasets):
        export_session(output / result["session"], result, data)
    make_plots(output, results, datasets)
    write_report(output, results, pairs)
    flat_metrics = [
        {
            "session": r["session"],
            "split": "all_quality_supported",
            **r["metrics"]["all_quality_supported"],
        }
        for r in results
    ]
    save_csv(output / "metrics.csv", flat_metrics)
    save_csv(
        output / "full_sequence_time_offset.csv",
        [
            {
                "session": r["session"],
                "pico_source": "pico/pico_head_poses.csv",
                "nokov_rigid_body": r["head_rigid_body"],
                **r["time_sync_quality"]["full_sequence_motion_residual"],
            }
            for r in results
        ],
    )
    save_csv(
        output / "time_pairing_stats.csv",
        [
            {
                "session": r["session"],
                **{
                    k: r["time_sync_quality"]["nearest_raw_pairing"][k]
                    for k in (
                        "count",
                        "mean_ms",
                        "std_ms",
                        "mean_abs_ms",
                        "p95_abs_ms",
                        "p99_abs_ms",
                        "max_abs_ms",
                        "reused_frame_pairs",
                    )
                },
            }
            for r in results
        ],
    )
    save_csv(
        output / "transform_differences.csv",
        [{k: v for k, v in p.items() if not isinstance(v, list)} for p in pairs],
    )
    save_json(output / "transform_differences.json", pairs)
    save_json(
        output / "summary.json",
        {
            "sessions": results,
            "transform_differences": pairs,
            "command": sys.argv,
            "program_sha256": sha256(Path(__file__)),
            "sync_quality_program_sha256": sha256(
                Path(__file__).with_name("pico_sync_quality.py")
            ),
            "evaluation": "fixed first-segment calibration; report metrics use all full-sequence quality-supported samples",
            "report_metrics_scope": "all_quality_supported",
        },
    )
    for result in results:
        if not all(sha256(Path(p)) == h for p, h in result["input_sha256"].items()):
            raise ValueError("raw inputs changed during processing")
    print("Report:", output / "report.html", flush=True)
    return int(any(r["status"] != "ok" for r in results))


if __name__ == "__main__":
    raise SystemExit(main())
