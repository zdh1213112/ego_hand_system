#!/usr/bin/env python3
"""Estimate NOKOV clock/IMU rigid transform from raw inertial data, never VIO.

Use the first 20 seconds for calibration and retain the later sequence as an
inertial validation split. This is a recording-derived calibration, not a
replacement for an independently measured rigid-body installation transform.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import least_squares, minimize_scalar
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation, Slerp
from ego_data.time_sync import require_zero_camera_offset


def skew(v):
    x, y, z = v
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])


def summarize(x):
    x = np.asarray(x)
    return {"rmse": float(np.sqrt(np.mean(x * x))), "median": float(np.median(x)),
            "p95": float(np.percentile(x, 95)), "max": float(np.max(x))}


def load_reference(path: Path, origin_ns: int):
    directory = path / "nokov" if (path / "nokov").is_dir() else path
    rows = [r for r in csv.DictReader((directory / "nokov_rigid_bodies.csv").open(encoding="utf-8-sig"))
            if r["rigid_body_name"] == "orbbec_ego4" and r["valid_numeric"] == "1"]
    raw = np.array([int(r["device_timestamp_raw"]) for r in rows], dtype=np.int64)
    # Infer milliseconds from 11/12 raw ticks at the recorded 90 Hz; verify.
    dt = np.median(np.diff(raw))
    if not 10 <= dt <= 12:
        raise ValueError("unexpected raw clock step; do not assume milliseconds")
    t = (raw * 1_000_000 - np.int64(origin_ns)) * 1e-9
    pos = np.array([[float(r[k]) for k in ("x_mm", "y_mm", "z_mm")] for r in rows]) * .001
    q = np.array([[float(r[k]) for k in ("qx", "qy", "qz", "qw")] for r in rows])
    rotation = Rotation.from_quat(q)
    recv = np.array([int(r["receive_unix_ns"]) for r in rows], dtype=np.int64)
    return directory, rows, t, pos, rotation, (recv - raw * 1_000_000) * 1e-9


def calibrate(output: Path, reference: Path, train_s: float):
    info = json.loads((output / "input_validation.json").read_text())
    require_zero_camera_offset(info,str(output))
    origin = int(info["first_timestamp_ns"])
    directory, rows, tm, pm, rm, recv_offset = load_reference(reference, origin)
    imu = list(csv.DictReader((output / "input_adapter/imu.csv").open()))
    ti = (np.array([int(r["timestamp_ns"]) for r in imu], dtype=np.int64) - np.int64(origin)) * 1e-9
    gyro = np.array([[float(r[k]) for k in ("wx", "wy", "wz")] for r in imu])
    accel = np.array([[float(r[k]) for k in ("ax", "ay", "az")] for r in imu])
    # Compare smoothed, body-coordinate angular velocities. R_MI maps IMU
    # vectors to the marker body, and t_imu = t_mocap_raw_ms + delta_s.
    h = .035
    slerp = Slerp(tm, rm)
    centers = np.arange(max(tm[0] + h, .6), min(tm[-1] - h, ti[-1] - .6), .02)
    rminus, rplus, rcenter = slerp(centers - h), slerp(centers + h), slerp(centers)
    wm = (rcenter.inv() * rminus).apply((rminus.inv() * rplus).as_rotvec()) / (2 * h)
    gyro_smooth = savgol_filter(gyro, 71, 3, axis=0)
    gi = interp1d(ti, gyro_smooth, axis=0, bounds_error=True)
    train = centers < train_s - .6
    def scalar_cost(delta):
        return float(np.mean((np.linalg.norm(wm[train], axis=1) - np.linalg.norm(gi(centers[train] + delta), axis=1)) ** 2))
    offsets = np.arange(-.5, .5001, .002)
    costs = np.array([scalar_cost(d) for d in offsets])
    delta0 = offsets[np.argmin(costs)]
    r0, _ = Rotation.align_vectors(wm[train], gi(centers[train] + delta0))
    def residual(x, select):
        return (Rotation.from_rotvec(x[1:4]).apply(gi(centers[select] + x[0])) + x[4:7] - wm[select]).ravel()
    fit = least_squares(lambda x: residual(x, train), np.r_[delta0, r0.as_rotvec(), np.zeros(3)],
                        bounds=(np.r_[-.5, [-np.inf] * 6], np.r_[.5, [np.inf] * 6]),
                        loss="soft_l1", f_scale=.035, max_nfev=200)
    delta, R_m_i = fit.x[0], Rotation.from_rotvec(fit.x[1:4])
    werror = (R_m_i.apply(gi(centers + delta)) + fit.x[4:7] - wm)
    gyro_stats = {"calibration_rad_s": summarize(np.linalg.norm(werror[train], axis=1)),
                  "validation_rad_s": summarize(np.linalg.norm(werror[~train], axis=1))}

    # Estimate the IMU lever arm with rigid-body acceleration kinematics:
    # R_MI f_I = R_NM^T (p_M'' - g_N) + ([alpha]x + [w]x^2) r_MI + b_M.
    # Translation calibration thus depends on raw IMU + independent mocap only.
    dtm = (tm[-1] - tm[0]) / (len(tm) - 1)
    window = 21  # ~0.23 s; suppress second-derivative amplification of marker noise.
    p_acc = savgol_filter(pm, window, 3, deriv=2, delta=dtm, axis=0)
    w_all = (rm[:-1].inv() * rm[1:]).as_rotvec() / np.diff(tm)[:, None]
    # R_NM is validated against the observed rigid markers below.
    w_all = np.vstack([w_all, w_all[-1]])
    wm_s = savgol_filter(w_all, window, 3, axis=0)
    alpha = savgol_filter(w_all, window, 3, deriv=1, delta=dtm, axis=0)
    acc_i = interp1d(ti, savgol_filter(accel, 231, 3, axis=0), axis=0, bounds_error=False, fill_value=np.nan)(tm + delta)
    valid = (tm + delta > .5) & (tm + delta < ti[-1] - .5)
    train_acc = valid & (tm + delta < train_s)
    matrices = rm.as_matrix()
    design = np.array([np.column_stack((skew(a) + skew(w) @ skew(w), -r.T, np.eye(3)))
                       for r, w, a in zip(matrices, wm_s, alpha)])
    target = R_m_i.apply(acc_i) - rm.inv().apply(p_acc)
    A, b = design[train_acc].reshape(-1, 9), target[train_acc].reshape(-1)
    x0 = np.linalg.lstsq(A, b, rcond=None)[0]
    def accel_residual(x):
        # Gravity magnitude is known; otherwise a limited rotation sequence can
        # trade a spurious gravity magnitude against accelerometer bias.
        constrained = x.copy()
        constrained[3:6] = x[3:6] * (9.81 / np.linalg.norm(x[3:6]))
        return np.r_[A @ constrained - b, (np.linalg.norm(x[3:6]) - 9.81) * 10]
    lever_fit = least_squares(accel_residual, x0, loss="soft_l1", f_scale=.08, max_nfev=200)
    lever_fit.x[3:6] *= 9.81 / np.linalg.norm(lever_fit.x[3:6])
    lever, gravity, accel_bias = np.split(lever_fit.x, 3)
    acc_residual = np.einsum("nij,j->ni", design, lever_fit.x) - target
    accel_stats = {"calibration_m_s2": summarize(np.linalg.norm(acc_residual[train_acc], axis=1)),
                   "validation_m_s2": summarize(np.linalg.norm(acc_residual[valid & ~train_acc], axis=1))}
    singular = np.linalg.svd(A, compute_uv=False)
    # Rigid-marker consistency independently checks the mocap quaternion direction.
    by_frame = {int(r["frame_no"]): j for j, r in enumerate(rows)}
    local = {}
    for row in csv.DictReader((directory / "nokov_rigid_body_markers.csv").open(encoding="utf-8-sig")):
        if row["valid"] != "1" or int(row["frame_no"]) not in by_frame:
            continue
        j = by_frame[int(row["frame_no"])]
        point = np.array([float(row[k]) for k in ("x_mm", "y_mm", "z_mm")]) * .001
        local.setdefault(int(row["marker_index"]), []).append(rm[j].inv().apply(point - pm[j]))
    rigidity = {}
    for k, points in local.items():
        points = np.asarray(points)
        center = np.median(points, axis=0)
        rigidity[str(k)] = {"local_position_m": center.tolist(), "residual_m": summarize(np.linalg.norm(points - center, axis=1))}
    transform = np.eye(4)
    transform[:3, :3], transform[:3, 3] = R_m_i.as_matrix(), lever
    result = {
        "reference_directory": str(directory), "rigid_body": "orbbec_ego4", "reference_rows": len(rows),
        "position_unit_input": "millimetres", "position_unit_output": "metres",
        "raw_clock_unit_inferred": "milliseconds; 11/12 ticks at approximately 90 Hz",
        "time_formula": "t_IMU_seconds = device_timestamp_raw * 1e-3 + mocap_to_imu_offset_s",
        "mocap_to_imu_offset_s": float(delta), "receive_minus_raw_s": summarize(recv_offset),
        "training_interval_imu_s": [.5, train_s], "validation_interval_imu_s": [train_s, ti[-1] - .5],
        "R_marker_imu": R_m_i.as_matrix().tolist(), "T_marker_imu": transform.tolist(),
        "lever_arm_marker_to_imu_m": lever.tolist(), "lever_arm_norm_m": float(np.linalg.norm(lever)),
        "estimated_gravity_N_m_s2": gravity.tolist(), "estimated_gravity_norm_m_s2": float(np.linalg.norm(gravity)),
        "estimated_accel_bias_marker_m_s2": accel_bias.tolist(), "estimated_gyro_bias_marker_rad_s": fit.x[4:7].tolist(),
        "angular_velocity_residual": gyro_stats, "acceleration_residual": accel_stats,
        "acceleration_design_condition": float(singular[0] / singular[-1]),
        "rigid_marker_consistency": rigidity,
        "method": "clock + rotation from raw gyro/mocap; lever arm from raw accelerometer/mocap acceleration; first 20 s calibration only; no VIO used",
        "limitation": "recording-derived inertial calibration, not an independently surveyed installation transform; acceleration smoothing and clock assumptions affect accuracy",
        "motion_fit_success": bool(fit.success and lever_fit.success),
    }
    (output / "nokov_calibration.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    np.savez_compressed(output / "nokov_sync_diagnostics.npz", time_mocap_s=centers, angular_speed_mocap=np.linalg.norm(wm, axis=1),
                        angular_speed_imu=np.linalg.norm(gi(centers + delta), axis=1), rotation_residual_rad_s=np.linalg.norm(werror, axis=1),
                        offset_candidates_s=offsets, offset_cost=costs)
    print(json.dumps({k: result[k] for k in ("mocap_to_imu_offset_s", "lever_arm_marker_to_imu_m", "estimated_gravity_norm_m_s2", "angular_velocity_residual", "acceleration_residual", "acceleration_design_condition")}, indent=2), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--calibration-seconds", type=float, default=20)
    a = p.parse_args()
    calibrate(a.output.resolve(), a.reference.resolve(), a.calibration_seconds)
