#!/usr/bin/env python3
"""Compare full Basalt runs when NOKOV covers only part of the recording.

Use an IMU-only sensor calibration, a two-second overlapping world-registration
prefix, and a disjoint evaluation interval. Never extrapolate motion-capture
poses, estimate a trajectory scale, or register on the evaluation interval.
"""
from __future__ import annotations

import argparse
import base64
import csv
import html
import json
import os
from pathlib import Path
import platform

import numpy as np
from scipy.spatial.transform import Rotation

from analyze_basalt_ego4_comparison import (
    CASES, COLORS, load_stats, position_se3_alignment, relative_pose_errors,
)
from calibrate_nokov_ego4 import load_reference, summarize
from compare_basalt_ego4 import dump, sha256, source_info
from ego_data.fixed_relations import FixedReference
from fuse_basalt_hand_trajectory import load_basalt_trajectory
from ego_data.time_sync import require_zero_camera_offset

ROOT = Path(__file__).resolve().parents[1]
LABELS = {"original_2cam": "Basalt 双目", "monado_2cam": "Basalt Monado 双目",
          "monado_4cam": "Basalt Monado 四目"}
ENGLISH = {"original_2cam": "Basalt / stereo", "monado_2cam": "Monado / stereo",
           "monado_4cam": "Monado / 4 cameras"}


def overlap_mask(times, reference_times, margin_s=.005, max_gap_s=.05):
    """Reserve ±5 ms for equal-sample sensitivity tests; reject reference holes."""
    times, rt = np.asarray(times), np.asarray(reference_times)
    if len(rt) < 2 or np.any(np.diff(rt) <= 0):
        raise ValueError("reference timestamps must strictly increase")
    mask = (times >= rt[0] + margin_s) & (times <= rt[-1] - margin_s)
    for shift in (-margin_s, 0, margin_s):
        hi = np.clip(np.searchsorted(rt, times + shift), 1, len(rt)-1)
        mask &= (rt[hi] - rt[hi-1]) <= max_gap_s
    return mask


def register_prefix(p, r, gp, gr, times, seconds=2.):
    selected = times <= times[0] + seconds
    if selected.sum() < 3 or times[-1] < times[0] + seconds:
        raise ValueError("insufficient reference overlap for world registration")
    wr = (gr[selected] * r[selected].inv()).mean()
    wt = np.mean(gp[selected] - wr.apply(p[selected]), axis=0)
    return wr.apply(p) + wt, wr * r, wr, wt, selected


def pose_summary(pe, re):
    return {"poses": len(pe), "position_m": summarize(pe), "orientation_deg": summarize(re),
            "position_below_5cm_fraction": float(np.mean(pe < .05)),
            "position_below_10cm_fraction": float(np.mean(pe < .1)),
            "position_below_20cm_fraction": float(np.mean(pe < .2))}


def write_tum(path, stamps, p, r):
    with path.open("w") as f:
        for t, pos, q in zip(stamps, p, r.as_quat()):
            f.write(f"{int(t)//10**9}.{int(t)%10**9:09d} " +
                    " ".join(f"{v:.12g}" for v in (*pos, *q)) + "\n")


def evaluate(output):
    info = json.loads((output / "input_validation.json").read_text())
    require_zero_camera_offset(info,str(output))
    calibration = json.loads((output / "nokov_calibration.json").read_text())
    if not calibration["motion_fit_success"]:
        raise ValueError("sensor calibration did not converge")
    calibration_end = float(calibration["training_interval_imu_s"][1])
    origin = int(info["first_timestamp_ns"])
    reference = FixedReference({"nokov_directory": calibration["reference_directory"],
        "rigid_body": calibration["rigid_body"], "T_marker_imu": calibration["T_marker_imu"],
        "nokov_to_imu_offset_s": calibration["mocap_to_imu_offset_s"]}, origin)
    with (output / "dataset_4cam/frame_map.csv").open() as f:
        frame_rows = list(csv.DictReader(f))
    expected = np.array([int(r["timestamp_ns"]) for r in frame_rows], dtype=np.int64)
    all_times = (expected - origin) * 1e-9
    common = overlap_mask(all_times, reference.times)
    times, stamps = all_times[common], expected[common]
    if len(times) < 60:
        raise ValueError("insufficient synchronized reference coverage")
    gp, gr, valid = reference.query(times)
    if not valid.all():
        raise ValueError("reference extrapolation forbidden")
    registration_end = times[0] + 2.
    holdout = times >= max(calibration_end, registration_end)
    if holdout.sum() < 60:
        raise ValueError("insufficient evaluation data after calibration")
    reference_speed = np.linalg.norm(np.gradient(gp, times, axis=0), axis=1)
    reference_angular_speed = np.r_[np.rad2deg((gr[:-1].inv()*gr[1:]).magnitude()) / np.diff(times), 0.]
    export = output / "evaluation"
    export.mkdir(exist_ok=True)
    write_tum(export / "nokov_imu_reference.tum", stamps, gp, gr)
    results, trajectories, windows = {}, {}, []
    for case in CASES:
        runs = []
        for directory in sorted((output / "runs" / case).glob("repeat_*")):
            timing = json.loads((directory / "vio_timing.json").read_text())
            if timing["returncode"] != 0:
                raise ValueError(f"failed VIO run: {directory}")
            poses = load_basalt_trajectory(directory / "trajectory.csv")
            # Explicit row count catches duplicate timestamps hidden by a dict.
            with (directory / "trajectory.csv").open() as f:
                row_count = sum(bool(s.strip()) and not s.startswith("#") for s in f)
            ts = np.array(sorted(poses), dtype=np.int64)
            if row_count != len(ts) or not np.array_equal(ts, expected):
                raise ValueError(f"missing, extra or duplicate VIO timestamps: {directory}")
            matrices = np.array([poses[int(t)] for t in ts])
            if not np.isfinite(matrices).all():
                raise ValueError(f"non-finite trajectory: {directory}")
            p_all, r_all = matrices[:, :3, 3], Rotation.from_matrix(matrices[:, :3, :3])
            p, r = p_all[common], r_all[common]
            ap, ar, wr, wt, registered = register_prefix(p, r, gp, gr, times)
            pe = np.linalg.norm(ap - gp, axis=1)
            re = np.rad2deg((gr.inv() * ar).magnitude())
            transform = np.eye(4)
            transform[:3, :3], transform[:3, 3] = wr.as_matrix(), wt
            run = {"repeat": int(directory.name.split("_")[-1]), "directory": str(directory),
                "poses": len(ts), "input_coverage": 1., "scored_poses": len(times),
                "wall_s": timing["wall_s"], "fps": len(ts) / timing["wall_s"],
                "all_overlap": pose_summary(pe, re), "holdout": pose_summary(pe[holdout], re[holdout]),
                "global_se3_ate_m": summarize(np.linalg.norm(position_se3_alignment(p, gp)-gp, axis=1)),
                "rpe_0_1s_holdout": relative_pose_errors(times, p, r, gp, gr, .1, holdout),
                "rpe_1s_holdout": relative_pose_errors(times, p, r, gp, gr, 1., holdout),
                "end_position_error_m": float(pe[-1]), "end_orientation_error_deg": float(re[-1]),
                "T_nokov_vio": transform.tolist(), "registration_poses": int(registered.sum()),
                "vio_path_length_overlap_m": float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum()),
                "max_output_step_m": float(np.linalg.norm(np.diff(p_all, axis=0), axis=1).max()),
                "stats": load_stats(directory), "clock_sensitivity_holdout_rmse_m": {},
                "lever_sensitivity_holdout_rmse_m": {}}
            for offset in (-.005, .005):
                sp, sr, ok = reference.query(times + offset)
                if not ok.all():
                    raise ValueError("sensitivity query outside reference")
                pp, _, _, _, _ = register_prefix(p, r, sp, sr, times)
                run["clock_sensitivity_holdout_rmse_m"][str(offset)] = float(
                    np.sqrt(np.mean(np.sum((pp[holdout]-sp[holdout])**2, axis=1))))
            for axis in range(3):
                for delta in (-.01, .01):
                    shift = np.zeros(3)
                    shift[axis] = delta
                    # R_N_M = R_N_I @ R_M_I^-1.
                    sp = gp + (gr * Rotation.from_matrix(reference.mount[:3, :3]).inv()).apply(shift)
                    pp, _, _, _, _ = register_prefix(p, r, sp, gr, times)
                    run["lever_sensitivity_holdout_rmse_m"][f"{'xyz'[axis]}{delta:+.2f}m"] = float(
                        np.sqrt(np.mean(np.sum((pp[holdout]-sp[holdout])**2, axis=1))))
            suffix = directory.name
            with (export / f"{case}_{suffix}_errors.csv").open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp_ns", "recording_time_s", "vio_x_m", "vio_y_m", "vio_z_m",
                    "nokov_imu_x_m", "nokov_imu_y_m", "nokov_imu_z_m", "position_error_m", "orientation_error_deg", "split"])
                for i, t in enumerate(times):
                    writer.writerow([int(stamps[i]), t, *ap[i], *gp[i], pe[i], re[i],
                                     "holdout" if holdout[i] else "calibration"])
            write_tum(export / f"{case}_{suffix}_imu_raw_full.tum", ts, p_all, r_all)
            write_tum(export / f"{case}_{suffix}_imu_nokov_full.tum", ts, wr.apply(p_all)+wt, wr*r_all)
            runs.append(run)
            if run["repeat"] == 1:
                trajectories[case] = {"time": times, "position": ap, "orientation": ar,
                    "gt": gp, "gt_orientation": gr, "pe": pe, "re": re, "holdout": holdout}
                for start in np.arange(0, all_times[-1], 5.):
                    select = holdout & (times >= start) & (times < start+5)
                    if select.any():
                        windows.append({"case": case, "start_s": start, "end_s": min(start+5, times[-1]),
                                        "reference_speed_p95_m_s": float(np.percentile(reference_speed[select], 95)),
                                        "reference_angular_speed_p95_deg_s": float(np.percentile(reference_angular_speed[select], 95)),
                                        **pose_summary(pe[select], re[select])})
        if not runs:
            raise ValueError(f"no runs for {case}")
        aggregate = {}
        paths = {"wall_s": lambda x:x["wall_s"], "fps":lambda x:x["fps"],
            "holdout_position_rmse_m":lambda x:x["holdout"]["position_m"]["rmse"],
            "holdout_position_p95_m":lambda x:x["holdout"]["position_m"]["p95"],
            "holdout_orientation_rmse_deg":lambda x:x["holdout"]["orientation_deg"]["rmse"],
            "all_overlap_position_rmse_m":lambda x:x["all_overlap"]["position_m"]["rmse"],
            "global_se3_ate_m":lambda x:x["global_se3_ate_m"]["rmse"],
            "rpe_1s_translation_rmse_m":lambda x:x["rpe_1s_holdout"]["translation_m"]["rmse"],
            "rpe_1s_orientation_rmse_deg":lambda x:x["rpe_1s_holdout"]["rotation_deg"]["rmse"],
            "rpe_0_1s_translation_rmse_m":lambda x:x["rpe_0_1s_holdout"]["translation_m"]["rmse"],
            "peak_rss_mib":lambda x:x["stats"]["peak_rss_mib"],
            "end_position_error_m":lambda x:x["end_position_error_m"],
            "below_10cm_fraction":lambda x:x["holdout"]["position_below_10cm_fraction"]}
        for name, get in paths.items():
            v = [get(x) for x in runs]
            aggregate[name] = {"median": float(np.median(v)), "min": float(min(v)), "max": float(max(v))}
        results[case] = {"label": LABELS[case], "runs": runs, "aggregate": aggregate}
    _, rows, _, _, _, _ = load_reference(Path(calibration["reference_directory"]), origin)
    counts = {p:sum(r["params"] == p for r in rows) for p in sorted({r["params"] for r in rows})}
    summary = {"input": info, "calibration": calibration, "results": results,
        "protocol": {"ground_truth_overlap_s": [float(reference.times[0]), float(reference.times[-1])],
            "scored_interval_s": [float(times[0]), float(times[-1])], "scored_poses": len(times),
            "unscored_input_poses": int((~common).sum()), "holdout_poses": int(holdout.sum()),
            "world_registration_interval_s": [float(times[0]), float(registration_end)],
            "sensor_calibration_interval_s": calibration["training_interval_imu_s"],
            "holdout_interval_s": [float(times[holdout][0]), float(times[holdout][-1])],
            "no_reference_extrapolation": True, "reference_gap_limit_s": .05, "reference_edge_margin_s": .005,
            "scale": 1., "sensor_calibration_uses_vio": False,
            "world_registration": "orientation mean + position mean on first 2 s of reference overlap, per repeat; fixed afterward",
            "headline": "t >= max(sensor calibration end, world registration end); median of repeat-level RMSE",
            "ATE": "SE(3) position alignment on entire scored overlap, scale 1; secondary descriptive metric",
            "rpe": "body-frame relative SE(3) error; nearest interval within 20 ms; both endpoints in holdout",
            "timing": "serial process wall time, PNG input and trajectory output included; no marginalization cache, build, video conversion or mapping",
            "latency": "offline throughput only; online sensor-to-pose latency not measured",
            "reference_path_length_m": float(np.linalg.norm(np.diff(gp,axis=0),axis=1).sum()),
            "rigid_body_params_counts": counts,
            "tracking_validity": "valid_numeric only certifies finite values; SDK tracking bits are not interpreted; marker rigidity checked separately"}}
    dump(output / "comparison_summary.json", summary)
    rows = []
    for case, result in results.items():
        rows.append({"case":case, "repeats":len(result["runs"]), "poses_per_run":info["prepared_groups"],
                     "scored_poses":len(times), "holdout_poses":int(holdout.sum()),
                     **{k:v["median"] for k,v in result["aggregate"].items()}})
    with (output / "comparison_metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    dump(output / "evaluation/windows_5s_repeat_1.json", windows)
    return summary, trajectories


def figures(output, summary, trajectories):
    os.environ["MPLCONFIGDIR"] = str(output / ".matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"axes.grid":True,"grid.alpha":.2,"font.size":10})
    destination = output / "figures"; destination.mkdir(exist_ok=True)
    first = trajectories[CASES[0]]; gt = first["gt"]; protocol = summary["protocol"]
    fig = plt.figure(figsize=(14, 10), constrained_layout=True)
    ax3 = fig.add_subplot(221, projection="3d"); ax2 = fig.add_subplot(222)
    ax3.plot(*gt.T, c="black", lw=2, label="NOKOV / IMU reference")
    ax2.plot(*gt[:,:2].T, c="black", lw=2, label="NOKOV / IMU reference")
    for case, tr in trajectories.items():
        ax3.plot(*tr["position"].T, c=COLORS[case], label=ENGLISH[case], lw=1)
        ax2.plot(*tr["position"][:,:2].T, c=COLORS[case], label=ENGLISH[case], lw=1)
    cloud = np.vstack([gt]+[v["position"] for v in trajectories.values()])
    center = (cloud.max(axis=0)+cloud.min(axis=0))/2; radius = max(np.ptp(cloud,axis=0))*.55
    ax3.set(xlabel="NOKOV X [m]", ylabel="Y [m]", zlabel="Z [m]", title="Repeat 1 / frozen prefix registration",
        xlim=(center[0]-radius,center[0]+radius), ylim=(center[1]-radius,center[1]+radius),
        zlim=(center[2]-radius,center[2]+radius)); ax3.set_box_aspect((1,1,1))
    ax2.set(xlabel="NOKOV X [m]",ylabel="Y [m]",title="XY / equal metric axes",aspect="equal")
    ax3.legend(fontsize=8); ax2.legend(fontsize=8)
    for index, field, label in [(223,"pe","Position error [m]"),(224,"re","Orientation error [deg]")]:
        ax=fig.add_subplot(index)
        for case,tr in trajectories.items(): ax.plot(tr["time"],tr[field],c=COLORS[case],lw=1,label=ENGLISH[case])
        ax.axvspan(*protocol["world_registration_interval_s"],color="#2563eb",alpha=.12,label="World registration")
        ax.axvspan(first["time"][0],protocol["sensor_calibration_interval_s"][1],color="#777777",alpha=.1)
        ax.axvline(protocol["holdout_interval_s"][0],c="#555555",ls="--",label="Holdout starts")
        ax.set(xlabel="EGO recording time [s]",ylabel=label); ax.legend(fontsize=8)
    fig.suptitle(f"EGO 20261104_195732 / {summary['input']['prepared_groups']} input groups / {protocol['scored_poses']} scored poses")
    fig.savefig(destination / "trajectory_and_errors.png", dpi=150); plt.close(fig)
    fig,axes=plt.subplots(1,4,figsize=(16,4),constrained_layout=True)
    for ax,key,title,factor in zip(axes,["holdout_position_rmse_m","holdout_orientation_rmse_deg","rpe_1s_translation_rmse_m","fps"],
        ["Holdout position RMSE [cm]","Holdout orientation RMSE [deg]","1 s RPE translation [cm]","Offline throughput [groups/s]"],[100,1,100,1]):
        values=[summary["results"][c]["aggregate"][key] for c in CASES]
        v=np.array([x["median"] for x in values])*factor
        errors=np.array([[x["median"]-x["min"] for x in values],[x["max"]-x["median"] for x in values]])*factor
        bars=ax.bar(["Basalt 2","Monado 2","Monado 4"],v,yerr=errors,capsize=4,color=[COLORS[c] for c in CASES])
        ax.bar_label(bars,fmt="%.2f",padding=6); ax.set(title=title,ylim=(0,max(x["max"] for x in values)*factor*1.3))
    fig.suptitle("Median of 3 serial runs; error bars show min-max")
    fig.savefig(destination / "comparison_bars.png",dpi=150); plt.close(fig)
    sync=np.load(output/"nokov_sync_diagnostics.npz"); cal=summary["calibration"]
    fig,axes=plt.subplots(2,1,figsize=(12,6),constrained_layout=True)
    t=sync["time_mocap_s"]+cal["mocap_to_imu_offset_s"]
    axes[0].plot(t,sync["angular_speed_mocap"],label="NOKOV angular speed",lw=1)
    axes[0].plot(t,sync["angular_speed_imu"],label="IMU / calibrated clock",lw=1,alpha=.7)
    axes[0].axvline(cal["training_interval_imu_s"][1],c="black",ls="--"); axes[0].legend()
    axes[0].set(xlabel="EGO recording time [s]",ylabel="rad/s",title="Raw sensor synchronization; VIO excluded from calibration")
    axes[1].plot(sync["offset_candidates_s"]*1000,sync["offset_cost"])
    axes[1].axvline(cal["mocap_to_imu_offset_s"]*1000,c="red",ls="--")
    axes[1].set(xlabel="Offset added to NOKOV raw clock [ms]",ylabel="Angular-speed mismatch")
    fig.savefig(destination/"synchronization.png",dpi=150); plt.close(fig)


def write_manifest(output):
    inputs=json.loads((output/"input_validation.json").read_text())
    calibration=json.loads((output/"nokov_calibration.json").read_text())
    paths=[output/"build_original/basalt_vio",ROOT/"third_party/basalt_monado/build/basalt_vio",
        output/"config_original.json",output/"config_monado.json",output/"nokov_calibration.json",
        output/"dataset_2cam/calibration.json",output/"dataset_4cam/calibration.json",
        Path(calibration["reference_directory"])/"nokov_rigid_bodies.csv",
        Path(calibration["reference_directory"])/"nokov_rigid_body_markers.csv",
        Path(calibration["reference_directory"])/"capture_metadata.json",
        Path(calibration["reference_directory"])/"asset_descriptions.json",
        Path(calibration["reference_directory"])/"nokov_frames.csv",
        output/"build_original/CMakeCache.txt",ROOT/"third_party/basalt_monado/build/CMakeCache.txt",
        ROOT/"tools/basalt_local_comparison/CMakeLists.txt",
        ROOT/"scripts/compare_basalt_ego4.py",ROOT/"scripts/prepare_basalt_video_dataset.py",
        ROOT/"scripts/calibrate_nokov_ego4.py",Path(__file__).resolve()]
    paths += sorted((output/"runs").glob("*/repeat_*/trajectory.csv"))
    paths += sorted((output/"runs").glob("*/repeat_*/vio_command.json"))
    sources={name:source_info(ROOT/"third_party"/name) for name in ("basalt","basalt_monado")}
    for name, entry in sources.items():
        if entry["commit"] != inputs["sources"][name]["commit"] or entry["status"]:
            raise ValueError(f"algorithm source changed or is dirty: {name}")
    for item in inputs["files"]:
        if sha256(Path(item["path"])) != item["sha256"]:
            raise ValueError(f"source recording changed: {item['path']}")
    cpu=next((s.split(":",1)[1].strip() for s in Path("/proc/cpuinfo").read_text().splitlines() if s.startswith("model name")),"")
    dump(output/"run_manifest.json",{"platform":platform.platform(),"cpu":cpu,"logical_cpus":os.cpu_count(),
        "sources":sources, "source_recordings_rehashed_unchanged":True,
        "binary_provenance":"Reused existing Release builds of the same clean source commits; binary and config hashes below",
        "source_recording_files":inputs["files"],
        "files":[{"path":str(p.resolve()),"sha256":sha256(p),"bytes":p.stat().st_size} for p in paths]})


def report(output, summary):
    result=summary["results"]; protocol=summary["protocol"]; cal=summary["calibration"]; info=summary["input"]
    def value(case,key):return result[case]["aggregate"][key]["median"]
    best=min(CASES,key=lambda c:value(c,"holdout_position_rmse_m"))
    base=value("original_2cam","holdout_position_rmse_m")
    stereo_gain=(1-value("monado_2cam","holdout_position_rmse_m")/base)*100
    four_gain=(1-value("monado_4cam","holdout_position_rmse_m")/value("monado_2cam","holdout_position_rmse_m"))*100
    stereo_rotation=(value("monado_2cam","holdout_orientation_rmse_deg")/value("original_2cam","holdout_orientation_rmse_deg")-1)*100
    stereo_rpe=(1-value("monado_2cam","rpe_1s_translation_rmse_m")/value("original_2cam","rpe_1s_translation_rmse_m"))*100
    four_cost=value("monado_4cam","wall_s")/value("monado_2cam","wall_s")
    four_rpe=(1-value("monado_4cam","rpe_1s_translation_rmse_m")/value("monado_2cam","rpe_1s_translation_rmse_m"))*100
    four_rotation=(1-value("monado_4cam","holdout_orientation_rmse_deg")/value("monado_2cam","holdout_orientation_rmse_deg"))*100
    def change(x):return f"降低 {x:.1f}%" if x>=0 else f"升高 {-x:.1f}%"
    window_rows=json.loads((output/"evaluation/windows_5s_repeat_1.json").read_text())
    temporal=""
    for start in sorted({r["start_s"] for r in window_rows}):
        matching={r["case"]:r for r in window_rows if r["start_s"]==start}
        row=matching[CASES[0]]
        temporal+=f"<tr><td>{start:.0f}–{row['end_s']:.2f}</td><td>{row['reference_speed_p95_m_s']:.2f}</td><td>{row['reference_angular_speed_p95_deg_s']:.1f}</td>"+"".join(f"<td>{matching[c]['position_m']['rmse']*100:.2f}</td>" for c in CASES)+"</tr>"
    def cell(case,key,factor=1):
        a=result[case]["aggregate"][key]
        return f"{a['median']*factor:.3f}<small>[{a['min']*factor:.3f}, {a['max']*factor:.3f}]</small>"
    rows="".join(f"<tr><th>{LABELS[c]}</th><td>{cell(c,'holdout_position_rmse_m',100)}</td><td>{cell(c,'holdout_orientation_rmse_deg')}</td><td>{cell(c,'rpe_1s_translation_rmse_m',100)}</td><td>{cell(c,'global_se3_ate_m',100)}</td><td>{cell(c,'wall_s')}</td><td>{cell(c,'fps')}</td><td>{cell(c,'peak_rss_mib')}</td></tr>" for c in CASES)
    details="".join(f"<tr><th>{LABELS[c]}</th><td>{cell(c,'all_overlap_position_rmse_m',100)}</td><td>{cell(c,'holdout_position_p95_m',100)}</td><td>{cell(c,'rpe_0_1s_translation_rmse_m',100)}</td><td>{cell(c,'rpe_1s_orientation_rmse_deg')}</td><td>{cell(c,'end_position_error_m',100)}</td><td>{cell(c,'below_10cm_fraction',100)}%</td></tr>" for c in CASES)
    repeat_rows="".join(f"<tr><td>{LABELS[c]}</td><td>{r['repeat']}</td><td>{r['holdout']['position_m']['rmse']*100:.3f}</td><td>{r['holdout']['orientation_deg']['rmse']:.3f}</td><td>{r['wall_s']:.3f}</td><td>{r['poses']}</td></tr>" for c in CASES for r in result[c]["runs"])
    sensitivity="".join(f"<tr><td>{LABELS[c]} / 第 {r['repeat']} 次</td><td>{r['holdout']['position_m']['rmse']*100:.3f}</td><td>{min(r['clock_sensitivity_holdout_rmse_m'].values())*100:.3f}–{max(r['clock_sensitivity_holdout_rmse_m'].values())*100:.3f}</td><td>{min(r['lever_sensitivity_holdout_rmse_m'].values())*100:.3f}–{max(r['lever_sensitivity_holdout_rmse_m'].values())*100:.3f}</td></tr>" for c in CASES for r in result[c]["runs"])
    def embedded(name):return "data:image/png;base64,"+base64.b64encode((output/"figures"/name).read_bytes()).decode()
    def esc(s):return html.escape(str(s))
    gyro=cal["angular_velocity_residual"]; accel=cal["acceleration_residual"]
    calibration_end=cal["training_interval_imu_s"][1]
    text=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Basalt / Monado × NOKOV · 20261104 19:57:32</title><style>
body{{font:16px/1.7 system-ui,sans-serif;color:#172b43;background:#f3f6fa;margin:0}}main{{max-width:1250px;margin:auto;padding:36px 26px 80px}}h1{{font-size:32px;line-height:1.3}}h2{{margin-top:40px;font-size:23px}}.lead{{background:#e1edf8;padding:22px;border-radius:10px}}.table{{overflow:auto;background:white;border-radius:8px;padding:8px}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{padding:12px;text-align:right;border-bottom:1px solid #e2e8f0;white-space:nowrap}}th:first-child,td:first-child{{text-align:left}}small{{display:block;color:#65758a;font-size:11px}}img{{width:100%;background:white;border-radius:8px;margin:12px 0}}code{{overflow-wrap:anywhere;background:#e4eaf1;padding:2px 5px}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#e4eaf1;padding:16px}}a{{color:#2459a6}}li{{margin:8px 0}}.note{{color:#596b80}}details{{background:white;padding:14px;margin:15px 0;border-radius:8px}}</style><main>
<p class="note">实测日期 2026-11-06 · EGO 20261104_195732 · NOKOV 20261104_195743_694430</p>
<h1>Basalt 与 Basalt Monado<br>头环 VIO 定位跟踪对比</h1>
<p class="lead">本次后段位置 RMSE 最低的是 <b>{LABELS[best]}</b>（{value(best,'holdout_position_rmse_m')*100:.2f} cm）。相同双目输入下，Monado 的位置 RMSE 相比原版{change(stereo_gain)}；Monado 四目相比自身双目{change(four_gain)}。每种配置串行完整运行三次，主表为每次运行指标的中位数，方括号为最小值与最大值。该结论限于本次序列及标定条件。</p>
<h2>主要结果</h2><div class="table"><table><thead><tr><th>方法</th><th>后段位置 RMSE / cm ↓</th><th>后段姿态 RMSE / ° ↓</th><th>1 s RPE 平移 / cm ↓</th><th>全段 SE(3) ATE / cm ↓</th><th>完整 VIO 耗时 / s ↓</th><th>吞吐 / 帧组·s⁻¹ ↑</th><th>峰值 RSS / MiB</th></tr></thead><tbody>{rows}</tbody></table></div>
<p>“后段”指 EGO 首个准备帧后的 {protocol['holdout_interval_s'][0]:.3f}–{protocol['holdout_interval_s'][1]:.3f} 秒，共 {protocol['holdout_poses']} 个位姿。1 s RPE 在该区间比较相对运动。全段 ATE 另用整个真值重叠段的位置做 SE(3) 配准，尺度固定为 1，仅作为辅助指标；它会消除部分整体偏差，不能代替固定前段配准后的漂移指标。</p>
<img alt="三次运行指标与范围" src="{embedded('comparison_bars.png')}">
<h2>对比解读</h2><ul>
<li>相同双目输入：Monado 的后段位置 RMSE {change(stereo_gain)}，1 s RPE 平移 RMSE {change(stereo_rpe)}，姿态 RMSE {change(-stereo_rotation)}。位置累计误差、短时相对运动误差与姿态精度应分别判断。</li>
<li>增加侧向相机：Monado 四目相比双目，位置 RMSE {change(four_gain)}，1 s RPE 平移 RMSE {change(four_rpe)}，姿态 RMSE {change(four_rotation)}。完整处理耗时为双目的 {four_cost:.2f} 倍，离线吞吐 {value('monado_4cam','fps'):.1f} 帧组/秒。该结果同时包含额外图像与多相机前端的作用，不能归结为仅更换版本的效果。</li>
<li>源码和实际配置：原版前端从 cam0 检测并向 cam1 匹配；Monado 对其他相机循环匹配，并启用非重叠视区检测和基于重投影的跨相机匹配初值。所有共有配置项一致，Monado 专有项保留在配置 JSON 中。本实验没有逐项消融，因此这些机制仅解释实现差别，不构成单项性能归因。参见<a href="../../third_party/basalt/include/basalt/optical_flow/frame_to_frame_optical_flow.h">原版前端</a>与<a href="../../third_party/basalt_monado/include/basalt/optical_flow/frame_to_frame_optical_flow.h">Monado 前端</a>。</li></ul>
<h2>轨迹和逐时误差</h2><p>图展示第 1 次运行。轨迹、位置误差和姿态误差共用冻结的世界变换；蓝色阴影是世界配准窗口，灰色为传感器标定段，虚线之后为评价段。3D 三轴采用相同长度比例。</p>
<img alt="NOKOV 与三种配置的轨迹和误差" src="{embedded('trajectory_and_errors.png')}">
<div class="table"><table><tr><th>方法</th><th>重叠全段位置 RMSE / cm</th><th>后段位置 P95 / cm</th><th>0.1 s RPE 平移 / cm</th><th>1 s RPE 姿态 / °</th><th>重叠段末位置误差 / cm</th><th>后段位置 &lt;10 cm 比例</th></tr>{details}</table></div>
<details open><summary>第 1 次运行：各运动时段的位置 RMSE</summary><p>速度与角速度是 NOKOV 参考的窗口 P95，误差为窗口 RMSE；仅展示后段。两者用于定位问题时段，不作因果归因。</p><div class="table"><table><tr><th>EGO 时间 / s</th><th>参考速度 P95 / m·s⁻¹</th><th>参考角速度 P95 / °·s⁻¹</th><th>Basalt 双目 / cm</th><th>Monado 双目 / cm</th><th>Monado 四目 / cm</th></tr>{temporal}</table></div></details>
<h2>数据范围与公平性</h2><ul>
<li>EGO：<code>{esc(info['session'])}</code>；原始四路帧数为 {info['source_video_counts']}，按真实时间戳交集与 IMU 覆盖筛选得到 <b>{info['prepared_groups']} 帧组 / {info['duration_s']:.3f} 秒</b>。输入 {info['resolution'][0]}×{info['resolution'][1]} 灰度 KB4 鱼眼，图像缩放 {info['image_scale']}；IMU 共 {info['imu_samples']} 组配对采样，厂家校正一次。</li>
<li>NOKOV：<code>{esc(cal['reference_directory'])}</code>；使用 <code>{cal['rigid_body']}</code> 头环刚体，{cal['reference_rows']} 帧，约 90 Hz；毫米转换为米，四元数 xyzw。</li>
<li>校时后真值覆盖 {protocol['ground_truth_overlap_s'][0]:.3f}–{protocol['ground_truth_overlap_s'][1]:.3f} 秒；评分使用 {protocol['scored_interval_s'][0]:.3f}–{protocol['scored_interval_s'][1]:.3f} 秒的 {protocol['scored_poses']} 帧，边缘预留 ±5 ms 做同步敏感性验证。其余 {protocol['unscored_input_poses']} 帧有 VIO 输出但没有误差评分，绝不外推真值。重叠段 IMU 参考运动路程 {protocol['reference_path_length_m']:.3f} m。</li>
<li>三种配置、每次运行均输出全部 {info['prepared_groups']} 个位姿，输出时间戳逐一匹配输入。双目原版与 Monado 复用相同 PNG 文件、相同 IMU 与相同图像时刻；四目单列为额外输入对照。输出完整不等于已证明算法从未失跟，需结合误差与相对运动指标判断。</li>
<li>相机间最大时间差 {info['camera_delta_ns_max']/1000:.1f} μs，最大输入帧间隔 {info['frame_gap_ms_max']:.3f} ms。TBB 并行上限 2，float VIO，公共后端 ABS_QR、3 状态、7 关键帧、最多 7 次迭代、LM 开启，recall 关闭。并行上限不包含其他输入输出线程。</li>
<li>图像与 IMU 直接使用录制器导出的设备时间，附加时间修正固定为 0 ms；仅做单位转换和采样关联。</li>
<li>耗时包括进程启动、PNG 读取、VIO 和轨迹导出，不包含编译、数据转换、传感器标定或建图；未写边缘化缓存。三种配置依次执行，第二轮反转顺序。离线吞吐不是在线传感器到位姿延迟。</li></ul>
<p class="note">原版日志中的内置 GT 评估可能打印 <code>error -nan / number of associations 0</code>，因为 EuRoC 输入未内置真值。本文使用独立 NOKOV 评价，并逐一检查实际导出位姿均为有限值。</p>
<h2>NOKOV 时钟和头环安装关系</h2><p>没有独立测量的刚体到 IMU 外参，因此只用本次前 {calibration_end:g} 秒中两传感器重叠的原始 IMU/NOKOV 运动估计同步与安装关系，不使用 VIO 拟合它们。世界配准对每次运行只使用最早重叠的两秒：{protocol['world_registration_interval_s'][0]:.3f}–{protocol['world_registration_interval_s'][1]:.3f} 秒；随后冻结旋转、平移与尺度。</p>
<p><code>t_IMU = device_timestamp_raw × 10⁻³ + {cal['mocap_to_imu_offset_s']:.9f} s</code>。<code>T_N_I = T_N_M × T_M_I</code>，其中矩阵 T_A_B 把 B 系坐标映射到 A 系；marker→IMU 杠杆臂在 marker 坐标系为 <code>{esc(np.round(cal['lever_arm_marker_to_imu_m'],6).tolist())} m</code>，长度 {cal['lever_arm_norm_m']*100:.2f} cm。刚体原点与 IMU 原点不同，已补偿旋转造成的杠杆臂运动。</p>
<p>角速度拟合残差 RMSE：标定 {gyro['calibration_rad_s']['rmse']:.4f} / 验证 {gyro['validation_rad_s']['rmse']:.4f} rad/s；加速度残差 RMSE：标定 {accel['calibration_m_s2']['rmse']:.3f} / 验证 {accel['validation_m_s2']['rmse']:.3f} m/s²；加速度设计矩阵条件数 {cal['acceleration_design_condition']:.1f}。完整外参、验证残差和各反光球的刚体一致性保留于标定 JSON。</p>
<img alt="原始 IMU 和 NOKOV 的同步验证" src="{embedded('synchronization.png')}">
<p>将同步偏移扰动 ±5 ms、安装平移每轴扰动 ±1 cm，只重算前段世界变换后，后段位置 RMSE 如下（单位 cm）。这些是敏感性试验，不能解释为统计置信区间。</p>
<div class="table"><table><tr><th>运行</th><th>标称</th><th>时钟 ±5 ms 范围</th><th>安装各轴 ±1 cm 范围</th></tr>{sensitivity}</table></div>
<p class="note">NOKOV 元数据声明未硬件同步；valid_numeric 仅表示数值有效，SDK params 分布为 {esc(protocol['rigid_body_params_counts'])}，未把它解释为明确的追踪状态。重建外参来自同段原始传感器、并非独立安装测量；运动激励、平滑微分、镜头与 IMU 残余时差都会影响厘米级结果。这是一段录制上的条件性比较，不能据此推广到其他场景的最终精度或通用胜负。</p>
<h2>复现与产物</h2><p><a href="comparison_metrics.csv">汇总 CSV</a> · <a href="comparison_summary.json">全部指标 JSON</a> · <a href="nokov_calibration.json">时钟与安装标定</a> · <a href="run_manifest.json">源码版本、二进制及输入哈希</a> · <a href="evaluation/">逐帧误差与完整 TUM 轨迹</a> · <a href="evaluation/windows_5s_repeat_1.json">第 1 次运行的 5 秒窗口指标</a></p>
<pre>bash scripts/run_basalt_nokov_comparison.sh output/basalt_nokov_195732_repeat</pre>
<details><summary>三次运行逐次结果</summary><div class="table"><table><tr><th>方法</th><th>重复</th><th>后段位置 RMSE / cm</th><th>后段姿态 RMSE / °</th><th>耗时 / s</th><th>输出位姿</th></tr>{repeat_rows}</table></div></details>
<details><summary>输入与算法版本</summary><pre>{esc(json.dumps(info['sources'],indent=2,ensure_ascii=False))}</pre></details>
</main></html>'''
    (output/"comparison_report.html").write_text(text,encoding="utf-8")


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--output",type=Path,required=True)
    args=p.parse_args(); output=args.output.resolve()
    summary,trajectories=evaluate(output); figures(output,summary,trajectories); write_manifest(output); report(output,summary)
    print(json.dumps({"report":str(output/"comparison_report.html"),"scored_interval_s":summary['protocol']['scored_interval_s'],
        "median_holdout_rmse_cm":{c:r['aggregate']['holdout_position_rmse_m']['median']*100 for c,r in summary['results'].items()}},indent=2))


if __name__ == "__main__":
    main()
