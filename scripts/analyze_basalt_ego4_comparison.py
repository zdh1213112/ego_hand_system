#!/usr/bin/env python3
"""Evaluate EGO-4 Basalt runs with synchronized, lever-arm-corrected NOKOV poses."""
from __future__ import annotations
import argparse
import csv
import json
import os
from pathlib import Path
import re

import cv2
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from calibrate_nokov_ego4 import load_reference, summarize
from compare_basalt_ego4 import dump
from fuse_basalt_hand_trajectory import load_basalt_trajectory
from ego_data.time_sync import require_zero_camera_offset

CASES = ("original_2cam", "monado_2cam", "monado_4cam")
LABELS = {"original_2cam": "Original / 2 cameras", "monado_2cam": "Monado / 2 cameras", "monado_4cam": "Monado / 4 cameras"}
COLORS = {"original_2cam": "#2563eb", "monado_2cam": "#e68a00", "monado_4cam": "#13956b"}


def map_directory(output, case):
    directory = output / "runs" / case / "repeat_1"
    return directory / ("map_lm" if (directory / "map_lm/summary.json").exists() else "map")


def interpolate_reference(output):
    info = json.loads((output / "input_validation.json").read_text())
    require_zero_camera_offset(info,str(output))
    calibration = json.loads((output / "nokov_calibration.json").read_text())
    origin = int(info["first_timestamp_ns"])
    _, _, times, positions, rotations, _ = load_reference(Path(calibration["reference_directory"]), origin)
    times = times + calibration["mocap_to_imu_offset_s"]
    r_m_i = Rotation.from_matrix(calibration["R_marker_imu"])
    lever = np.array(calibration["lever_arm_marker_to_imu_m"])
    slerp = Slerp(times, rotations)
    def query(t, lever_override=None):
        r = slerp(t)
        p = np.column_stack([np.interp(t, times, positions[:, axis]) for axis in range(3)])
        p += r.apply(lever if lever_override is None else lever_override)
        return p, r * r_m_i
    return info, calibration, origin, query


def aligned_poses(positions, rotations, gt_positions, gt_rotations, times):
    """Register worlds using first two seconds only; preserve metric scale."""
    select = times <= 2.0
    world_rotation = (gt_rotations[select] * rotations[select].inv()).mean()
    translation = np.mean(gt_positions[select] - world_rotation.apply(positions[select]), axis=0)
    return world_rotation.apply(positions) + translation, world_rotation * rotations, world_rotation, translation


def relative_pose_errors(times, pos, rot, gt_pos, gt_rot, interval_s, selection=None):
    j = np.searchsorted(times, times + interval_s)
    i = np.arange(len(times))
    # Select nearest samples, not always the later one: floating-point rounding
    # of nominal 0.1 s targets must not turn a 3-frame interval into 4 frames.
    high = np.minimum(j, len(times) - 1)
    low = np.maximum(high - 1, 0)
    target = times + interval_s
    j = np.where(np.abs(times[low] - target) < np.abs(times[high] - target), low, high)
    valid = j < len(times)
    j = np.minimum(j, len(times) - 1)
    valid &= np.abs(times[j] - times - interval_s) <= .02
    if selection is not None:
        valid &= selection & selection[j]
    i, j = i[valid], j[valid]
    if not len(i):
        empty={key:None for key in ['rmse','median','p95','max']}
        return {'count':0,'translation_m':empty.copy(),'rotation_deg':empty.copy()}
    # The translation of inv(delta_gt) @ delta_est has the same norm as
    # delta_est.translation - delta_gt.translation (rotation preserves norm).
    dp = rot[i].inv().apply(pos[j] - pos[i])
    dg = gt_rot[i].inv().apply(gt_pos[j] - gt_pos[i])
    dr = rot[i].inv() * rot[j]
    gr = gt_rot[i].inv() * gt_rot[j]
    return {"count": len(i), "translation_m": summarize(np.linalg.norm(dp - dg, axis=1)),
            "rotation_deg": summarize(np.rad2deg((gr.inv() * dr).magnitude()))}


def position_se3_alignment(p, target):
    """Kabsch SE(3) alignment, with no estimated scale."""
    pc, tc = p.mean(axis=0), target.mean(axis=0)
    u, _, vt = np.linalg.svd((p - pc).T @ (target - tc))
    correction = np.diag([1., 1., np.linalg.det(vt.T @ u.T)])
    rotation = vt.T @ correction @ u.T
    return (p - pc) @ rotation.T + tc


def load_stats(directory):
    log = (directory / "vio.log").read_text(errors="replace")
    result = {}
    for name in ("num_lms", "num_obs", "optimize", "marginalize", "measure"):
        match = re.search(r"^" + re.escape(name) + r"\s+\([^\n:]*\):\s*([+-]?[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)", log, re.M)
        if match:
            result[name + "_mean"] = float(match.group(1))
    match = re.search(r"Total runtime: ([0-9.]+)s", log)
    result["internal_runtime_s"] = float(match.group(1)) if match else None
    resource = (directory / "vio_resource.txt").read_text()
    for key, pattern in (("peak_rss_mib", r"Maximum resident set size \(kbytes\): (\d+)"),
                         ("user_cpu_s", r"User time \(seconds\): ([0-9.]+)"),
                         ("system_cpu_s", r"System time \(seconds\): ([0-9.]+)")):
        match = re.search(pattern, resource)
        if match:
            result[key] = float(match.group(1)) / (1024 if key == "peak_rss_mib" else 1)
    return result


def evaluate(output):
    info, calibration, origin, query = interpolate_reference(output)
    results, trajectories = {}, {}
    export = output / "evaluation"
    export.mkdir(exist_ok=True)
    for case in CASES:
        repeats = []
        for directory in sorted((output / "runs" / case).glob("repeat_*")):
            timing = json.loads((directory / "vio_timing.json").read_text())
            if timing["returncode"] != 0:
                raise RuntimeError(f"failed VIO run: {directory}")
            poses = load_basalt_trajectory(directory / "trajectory.csv")
            ts = np.array(sorted(poses), dtype=np.int64)
            times = (ts - np.int64(origin)) * 1e-9
            matrices = np.array([poses[int(t)] for t in ts])
            if len(ts) != info["prepared_groups"] or not np.isfinite(matrices).all():
                raise ValueError(f"incomplete or non-finite trajectory: {directory}")
            p, r = matrices[:, :3, 3], Rotation.from_matrix(matrices[:, :3, :3])
            gp, gr = query(times)
            ap, ar, registration_r, registration_t = aligned_poses(p, r, gp, gr, times)
            pe, re_deg = np.linalg.norm(ap - gp, axis=1), np.rad2deg((gr.inv() * ar).magnitude())
            holdout = times >= 20.0
            repeat = {"directory": str(directory), "wall_s": timing["wall_s"], "poses": len(ts),
                      "coverage": len(ts) / info["prepared_groups"], "duration_s": float(times[-1] - times[0]),
                      "translation_error_all_m": summarize(pe), "rotation_error_all_deg": summarize(re_deg),
                      "translation_error_holdout_m": summarize(pe[holdout]), "rotation_error_holdout_deg": summarize(re_deg[holdout]),
                      "global_se3_ate_all_m": summarize(np.linalg.norm(position_se3_alignment(p, gp) - gp, axis=1)),
                      "rpe_0.1s_holdout": relative_pose_errors(times, p, r, gp, gr, .1, holdout),
                      "rpe_1s_holdout": relative_pose_errors(times, p, r, gp, gr, 1.0, holdout),
                      "path_length_m": float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum()),
                      "end_displacement_m": float(np.linalg.norm(p[-1] - p[0])),
                      "final_position_error_m": float(pe[-1]), "final_rotation_error_deg": float(re_deg[-1]),
                      "world_registration_rotation": registration_r.as_matrix().tolist(), "world_registration_translation_m": registration_t.tolist(),
                      "stats": load_stats(directory)}
            # Show sensitivity to the missing surveyed lever arm. All variants
            # share the same gyro-only rotation/time calibration and origin fit.
            sensitivity = {}
            for lever_name, lever in (("zero_lever", np.zeros(3)),
                                      ("estimated_plus_x_1cm", np.array(calibration["lever_arm_marker_to_imu_m"]) + [.01, 0, 0]),
                                      ("estimated_minus_x_1cm", np.array(calibration["lever_arm_marker_to_imu_m"]) - [.01, 0, 0])):
                sp, sr = query(times, lever)
                pp, _, _, _ = aligned_poses(p, r, sp, sr, times)
                sensitivity[lever_name] = float(np.sqrt(np.mean(np.linalg.norm(pp[holdout] - sp[holdout], axis=1) ** 2)))
            repeat["lever_arm_sensitivity_holdout_rmse_m"] = sensitivity
            repeat["clock_offset_sensitivity_holdout_rmse_m"] = {}
            for offset in (-.005, .005):
                sp, sr = query(times + offset)
                pp, _, _, _ = aligned_poses(p, r, sp, sr, times)
                repeat["clock_offset_sensitivity_holdout_rmse_m"][str(offset)] = float(np.sqrt(np.mean(np.linalg.norm(pp[holdout] - sp[holdout], axis=1)**2)))
            repeats.append(repeat)
            if directory.name == "repeat_1":
                trajectories[case] = {"time": times, "timestamp": ts, "raw_position": p, "aligned_position": ap,
                                      "aligned_rotation": ar, "gt_position": gp, "gt_rotation": gr,
                                      "position_error": pe, "rotation_error_deg": re_deg,
                                      "world_rotation": registration_r, "world_translation": registration_t}
                with (export / f"{case}_aligned_vs_nokov.csv").open("w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["timestamp_ns", "time_s", "estimated_imu_x_m", "estimated_imu_y_m", "estimated_imu_z_m",
                                "nokov_imu_x_m", "nokov_imu_y_m", "nokov_imu_z_m", "position_error_m", "rotation_error_deg", "split"])
                    for k, t in enumerate(times):
                        w.writerow([int(ts[k]), t, *ap[k], *gp[k], pe[k], re_deg[k], "holdout" if t >= 20 else "calibration"])
                # Standard TUM exports retain both raw IMU poses and aligned
                # IMU poses; labels explicitly identify the tracked point.
                for label, pp, rr in (("imu_raw", p, r), ("imu_nokov_world", ap, ar)):
                    with (export / f"{case}_{label}.tum").open("w") as f:
                        for stamp, position, quat in zip(ts, pp, rr.as_quat()):
                            f.write(f"{stamp // 1000000000}.{stamp % 1000000000:09d} " + " ".join(f"{v:.12g}" for v in (*position, *quat)) + "\n")
        walls = np.array([x["wall_s"] for x in repeats])
        ate = np.array([x["translation_error_holdout_m"]["rmse"] for x in repeats])
        angle = np.array([x["rotation_error_holdout_deg"]["rmse"] for x in repeats])
        result = {"repeats": repeats, "wall_s_median": float(np.median(walls)), "wall_s_range": [float(walls.min()), float(walls.max())],
                  "fps_median": info["prepared_groups"] / float(np.median(walls)),
                  "mean_ms_per_input_frame": float(np.median(walls)) * 1000 / info["prepared_groups"],
                  "faster_than_recording_factor": info["duration_s"] / float(np.median(walls)),
                  "holdout_position_rmse_m_median": float(np.median(ate)), "holdout_position_rmse_m_range": [float(ate.min()), float(ate.max())],
                  "holdout_rotation_rmse_deg_median": float(np.median(angle))}
        result["global_se3_ate_all_m_median"] = float(np.median([x["global_se3_ate_all_m"]["rmse"] for x in repeats]))
        map_dir = map_directory(output, case)
        if (map_dir / "summary.json").exists():
            result["mapping"] = json.loads((map_dir / "summary.json").read_text())
            stem = "mapping_lm" if map_dir.name == "map_lm" else "mapping"
            result["mapping"]["timing"] = json.loads((map_dir.parent / f"{stem}_timing.json").read_text())
            result["mapping"]["solver"] = "LM" if map_dir.name == "map_lm" else "GN"
            gn = map_dir.parent / "map/summary.json"
            if map_dir.name == "map_lm" and gn.exists():
                result["mapping_gn_diagnostic"] = json.loads(gn.read_text())
            keyframes = list(csv.DictReader((map_dir / "keyframe_imu_poses.csv").open()))
            mt = (np.array([int(x["timestamp_ns"]) for x in keyframes], dtype=np.int64) - np.int64(origin)) * 1e-9
            mp = np.array([[float(x[k]) for k in ("x_m", "y_m", "z_m")] for x in keyframes])
            mq = np.array([[float(x[k]) for k in ("qx", "qy", "qz", "qw")] for x in keyframes])
            gp, gr = query(mt)
            tr = trajectories[case]
            ap = tr["world_rotation"].apply(mp) + tr["world_translation"]
            ar = tr["world_rotation"] * Rotation.from_quat(mq)
            result["mapping"]["keyframe_position_error_all_m"] = summarize(np.linalg.norm(ap - gp, axis=1))
            result["mapping"]["keyframe_position_error_holdout_m"] = summarize(np.linalg.norm(ap[mt >= 20] - gp[mt >= 20], axis=1))
            result["mapping"]["keyframe_rotation_error_holdout_deg"] = summarize(np.rad2deg((gr[mt >= 20].inv() * ar[mt >= 20]).magnitude()))
            indices = np.searchsorted(tr["timestamp"], np.array([int(x["timestamp_ns"]) for x in keyframes], dtype=np.int64))
            if not np.array_equal(tr["timestamp"][indices], np.array([int(x["timestamp_ns"]) for x in keyframes], dtype=np.int64)):
                raise ValueError("mapping keyframes not present in original online trajectory")
            result["mapping"]["online_vio_at_same_keyframes_holdout_error_m"] = summarize(np.linalg.norm(tr["aligned_position"][indices][mt >= 20] - gp[mt >= 20], axis=1))
        results[case] = result
    summary = {"input": {k: info[k] for k in ("session", "prepared_groups", "duration_s", "resolution", "camera_delta_ns_max", "frame_gap_ms_max")},
               "evaluation_protocol": {"registration": "orientation mean + position mean on first 2 seconds; SE(3), scale fixed to 1",
                                       "calibration": "raw IMU/mocap fit on first 20 seconds; no VIO used for marker-to-IMU or clock calibration",
                                       "holdout": "time >= 20 seconds, fixed calibration and world registration",
                                       "reference_point": "IMU origin (NOKOV marker pose corrected with estimated rigid transform)",
                                       "reference_limitation": calibration["limitation"],
                                       "timing": "serial full-process wall time including PNG reads, marginalization cache writes and trajectory export; excluding preparation/build/NFR mapping",
                                       "mapping": "same headless NFR 10+10 iterations and outlier filter, same two-thread limit; no independent surface truth"},
               "calibration": calibration, "results": results}
    dump(output / "comparison_summary.json", summary)
    with (output / "comparison_metrics.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case", "wall_s_median", "fps", "ms_per_group", "holdout_position_rmse_m", "holdout_rotation_rmse_deg", "map_points", "map_wall_s", "map_reprojection_rmse_px"])
        for name, result in results.items():
            m = result.get("mapping", {})
            w.writerow([name, result["wall_s_median"], result["fps_median"], result["mean_ms_per_input_frame"],
                        result["holdout_position_rmse_m_median"], result["holdout_rotation_rmse_deg_median"],
                        m.get("landmarks"), m.get("timing", {}).get("wall_s"), m.get("reprojection_rmse_px")])
    return summary, trajectories


def figures(output, summary, trajectories):
    os.environ["MPLCONFIGDIR"] = str(output / ".matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": .22})
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    first = trajectories[CASES[0]]
    gt = first["gt_position"]
    fig = plt.figure(figsize=(14, 10), constrained_layout=True)
    ax3 = fig.add_subplot(221, projection="3d")
    ax2 = fig.add_subplot(222)
    for ax in (ax3, ax2):
        if ax is ax3:
            ax.plot(*gt.T, c="black", lw=2, label="NOKOV / IMU reference")
        else:
            ax.plot(gt[:, 0], gt[:, 1], c="black", lw=2, label="NOKOV / IMU reference")
        for case, tr in trajectories.items():
            p = tr["aligned_position"]
            ax.plot(*(p.T if ax is ax3 else p[:, :2].T), c=COLORS[case], lw=1.2, label=LABELS[case])
        ax.set(xlabel="NOKOV X (m)", ylabel="NOKOV Y (m)")
        ax.legend(fontsize=8)
    ax3.set(zlabel="NOKOV Z (m)", title="3D IMU trajectories / first 2 s registration")
    ax2.set(title="Top view / metric scale retained", aspect="equal")
    for index, field, label in ((223, "position_error", "Position difference (m)"), (224, "rotation_error_deg", "Orientation difference (deg)")):
        ax = fig.add_subplot(index)
        for case, tr in trajectories.items():
            ax.plot(tr["time"], tr[field], c=COLORS[case], label=LABELS[case], lw=1)
        ax.axvspan(0, 20, color="#888888", alpha=.12, label="calibration interval")
        ax.axvline(20, color="#555555", ls="--", lw=1)
        ax.set(xlabel="Recording time (s)", ylabel=label, title="Fixed calibration; shaded first 20 s excluded from headline errors")
        ax.legend(fontsize=8)
    fig.suptitle("Basalt VIO vs NOKOV | 1608 common frames | estimated marker-to-IMU calibration", fontsize=14)
    fig.savefig(figures / "trajectory_and_errors.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for ax, key, title, unit in zip(axes, ("wall_s_median", "holdout_position_rmse_m_median", "holdout_rotation_rmse_deg_median"),
                                     ("VIO wall time (median of 3)", "Position RMSE / t >= 20 s", "Orientation RMSE / t >= 20 s"), ("s", "m", "deg")):
        values = [summary["results"][c][key] for c in CASES]
        bars = ax.bar(["Original 2", "Monado 2", "Monado 4"], values, color=[COLORS[c] for c in CASES])
        ax.bar_label(bars, fmt="%.3f", padding=3)
        ax.set(title=title, ylabel=unit, ylim=(0, max(values) * 1.22))
    fig.savefig(figures / "comparison_bars.png", dpi=160)
    plt.close(fig)
    sync = np.load(output / "nokov_sync_diagnostics.npz")
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), constrained_layout=True)
    t = sync["time_mocap_s"] + summary["calibration"]["mocap_to_imu_offset_s"]
    axes[0].plot(t, sync["angular_speed_mocap"], label="NOKOV angular speed", lw=1)
    axes[0].plot(t, sync["angular_speed_imu"], label="IMU after time offset", lw=1, alpha=.7)
    axes[0].set(xlabel="IMU recording time (s)", ylabel="rad/s", title="Synchronization from raw sensors, independent of VIO")
    axes[0].legend()
    axes[1].plot(sync["offset_candidates_s"] * 1000, sync["offset_cost"])
    axes[1].axvline(summary["calibration"]["mocap_to_imu_offset_s"] * 1000, c="red", ls="--")
    axes[1].set(xlabel="NOKOV raw-clock offset added (ms)", ylabel="Angular speed mismatch", title="Clock offset search on first 20 s")
    fig.savefig(figures / "synchronization.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    displayed = []
    for ax, case in zip(axes, CASES):
        mapdir = map_directory(output, case)
        if not (mapdir / "sparse_landmarks.csv").exists():
            ax.text(.5, .5, "Mapping unavailable", ha="center", transform=ax.transAxes)
            continue
        points = np.loadtxt(mapdir / "sparse_landmarks.csv", delimiter=",", skiprows=1, ndmin=2)[:, 1:4]
        tr = trajectories[case]
        points = tr["world_rotation"].apply(points) + tr["world_translation"]
        center = np.mean(tr["gt_position"], axis=0)
        keep = np.linalg.norm(points - center, axis=1) <= 5.0
        displayed.append(points[keep])
        ax.scatter(points[keep, 0], points[keep, 1], c=points[keep, 2], cmap="viridis", s=2, alpha=.4)
        ax.plot(gt[:, 0], gt[:, 1], c="black", lw=1)
        ax.set(title=f"{LABELS[case]}\n{len(points)} unique points; {keep.sum()} shown within 5 m", xlabel="NOKOV X (m)", ylabel="NOKOV Y (m)", aspect="equal")
    if displayed:
        all_points = np.vstack(displayed)
        low, high = all_points[:, :2].min(axis=0) - .15, all_points[:, :2].max(axis=0) + .15
        for ax in axes:
            ax.set(xlim=(low[0], high[0]), ylim=(low[1], high[1]))
    fig.suptitle("NFR sparse maps / display subset only; original PLY retains all points")
    fig.savefig(figures / "sparse_maps.png", dpi=160)
    plt.close(fig)
    # Contact sheet helps review actual texture, hands and occlusions.
    mapping = list(csv.DictReader((output / "dataset_4cam/frame_map.csv").open()))
    samples = np.linspace(0, len(mapping) - 1, 5).astype(int)
    sheet = np.zeros((5 * 220, 4 * 320, 3), np.uint8)
    for row_index, index in enumerate(samples):
        stamp = mapping[index]["timestamp_ns"]
        for camera in range(4):
            frame = cv2.imread(str(output / f"dataset_4cam/mav0/cam{camera}/data/{stamp}.png"))
            frame = cv2.resize(frame, (320, 200))
            y, x = row_index * 220, camera * 320
            sheet[y + 20:y + 220, x:x + 320] = frame
            cv2.putText(sheet, f"cam{camera}  t={(int(stamp)-int(mapping[0]['timestamp_ns']))*1e-9:.1f}s", (x + 8, y + 15), cv2.FONT_HERSHEY_SIMPLEX, .42, (255, 255, 255), 1)
    cv2.imwrite(str(figures / "recording_contact_sheet.jpg"), sheet)


def report_html(output, summary):
    from basalt_comparison_report import render_report
    render_report(output, summary)


def video(output, summary, trajectories):
    first = trajectories[CASES[0]]
    times, stamps = first["time"], first["timestamp"]
    gt = first["gt_position"]
    cloud = np.vstack([gt, *[tr["aligned_position"] for tr in trajectories.values()]])
    lo, hi = cloud[:, :2].min(axis=0), cloud[:, :2].max(axis=0)
    mid = (lo + hi) / 2
    half = max(hi - lo) * .58
    def points(p):
        x = 920 + (p[:, 0] - mid[0]) / half * 285
        y = 370 - (p[:, 1] - mid[1]) / half * 285
        return np.column_stack([x, y]).astype(np.int32)
    gtxy = points(gt)
    estimated = {name: points(tr["aligned_position"]) for name, tr in trajectories.items()}
    colors = {name: tuple(int(COLORS[name][i:i+2], 16) for i in (5, 3, 1)) for name in CASES}
    path = output / "comparison_video.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (1280, 720))
    if not writer.isOpened():
        raise RuntimeError("MP4 encoder unavailable")
    def text(canvas, value, xy, color=(30, 40, 55), size=.52):
        cv2.putText(canvas, value, xy, cv2.FONT_HERSHEY_SIMPLEX, size, color, 1, cv2.LINE_AA)
    for t in np.arange(0, times[-1] + .001, .1):
        i = int(np.argmin(np.abs(times - t)))
        canvas = np.full((720, 1280, 3), 248, np.uint8)
        text(canvas, f"EGO-4 / Basalt comparison / t={t:.1f}s", (18, 27), size=.65)
        for camera in range(4):
            image = cv2.imread(str(output / f"dataset_4cam/mav0/cam{camera}/data/{int(stamps[i])}.png"))
            image = cv2.resize(image, (280, 175))
            x, y = 18 + (camera % 2) * 292, 65 + (camera // 2) * 200
            canvas[y:y + 175, x:x + 280] = image
            text(canvas, f"cam{camera}", (x, y - 8), size=.45)
        text(canvas, "Top view in NOKOV world / IMU origin", (640, 58), size=.57)
        for x in range(640, 1231, 59):
            cv2.line(canvas, (x, 85), (x, 655), (222, 226, 231), 1)
        for y in range(85, 656, 57):
            cv2.line(canvas, (640, y), (1230, y), (222, 226, 231), 1)
        cv2.polylines(canvas, [gtxy[:i + 1]], False, (20, 20, 20), 2, cv2.LINE_AA)
        cv2.circle(canvas, tuple(gtxy[i]), 5, (20, 20, 20), -1)
        text(canvas, "NOKOV (estimated marker-to-IMU transform)", (640, 692), size=.45)
        for n, case in enumerate(CASES):
            xy = estimated[case]
            cv2.polylines(canvas, [xy[:i + 1]], False, colors[case], 2, cv2.LINE_AA)
            cv2.circle(canvas, tuple(xy[i]), 4, colors[case], -1)
            tr = trajectories[case]
            label = f"{LABELS[case]}: {tr['position_error'][i]*100:.1f} cm / {tr['rotation_error_deg'][i]:.2f} deg"
            text(canvas, label, (18, 500 + n * 36), colors[case], size=.52)
        text(canvas, "Calibration interval" if t < 20 else "Held-out evaluation interval", (18, 636), size=.57)
        text(canvas, "World registration: first 2 s; no scale fitting", (18, 668), size=.46)
        text(canvas, "Clock + installation estimate: first 20 s of raw IMU/mocap", (18, 695), size=.42)
        writer.write(canvas)
    writer.release()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--no-figures", action="store_true")
    p.add_argument("--video", action="store_true")
    a = p.parse_args()
    output = a.output.resolve()
    summary, trajectories = evaluate(output)
    if not a.no_figures:
        figures(output, summary, trajectories)
    if a.video:
        video(output, summary, trajectories)
    report_html(output, summary)
    print((output / "comparison_metrics.csv").read_text())


if __name__ == "__main__":
    main()
