#!/usr/bin/env python3
"""Present saved Basalt comparison results without rerunning the benchmark."""
from __future__ import annotations

import argparse
import base64
import csv
import html
import json
import os
from pathlib import Path
import shutil
from statistics import median
from string import Template
import subprocess

import numpy as np

CASES = ("original_2cam", "monado_2cam", "monado_4cam")
LABELS = dict(zip(CASES, ("原版双目", "Monado 双目", "Monado 四目")))
COLORS = dict(zip(CASES, ("#2563eb", "#e68a00", "#13956b")))


def video_assets(output):
    """Keep future report generations playable when the benchmark emits MPEG-4."""
    original, web = output / "comparison_video.mp4", output / "comparison_video_web.mp4"
    encoder = shutil.which("ffmpeg")
    if original.exists() and encoder and (not web.exists() or web.stat().st_mtime < original.stat().st_mtime):
        temporary = output / ".comparison_video_web.tmp.mp4"
        try:
            subprocess.run([encoder, "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i", str(original),
                            "-c:v", "libx264", "-threads", "2", "-preset", "fast", "-crf", "22", "-pix_fmt", "yuv420p",
                            "-movflags", "+faststart", "-an", str(temporary)], check=True)
            temporary.replace(web)
        finally:
            temporary.unlink(missing_ok=True)
    video = web if web.exists() else original if original.exists() else None
    poster = output / "video_preview.jpg"
    if video and encoder and not poster.exists():
        subprocess.run([encoder, "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i", str(video),
                        "-frames:v", "1", "-update", "1", str(poster)], check=True)
    return video, poster if poster.exists() else output / "figures/recording_contact_sheet.jpg"


def median_value(result, *keys):
    values = result["repeats"]
    for key in keys:
        values = [value[key] for value in values]
    return median(values)


def saved_geometry(output, summary):
    trajectories, maps = {}, {}
    for case in CASES:
        with (output / "evaluation" / f"{case}_aligned_vs_nokov.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        trajectories[case] = {
            "time": np.array([float(row["time_s"]) for row in rows]),
            "position": np.array([[float(row[f"estimated_imu_{axis}_m"]) for axis in "xyz"] for row in rows]),
            "reference": np.array([[float(row[f"nokov_imu_{axis}_m"]) for axis in "xyz"] for row in rows]),
            "position_error": np.array([float(row["position_error_m"]) for row in rows]),
            "rotation_error": np.array([float(row["rotation_error_deg"]) for row in rows]),
        }
        run = summary["results"][case]["repeats"][0]
        directory = output / "runs" / case / "repeat_1"
        directory = directory / ("map_lm" if (directory / "map_lm/summary.json").exists() else "map")
        path = directory / "sparse_landmarks.csv"
        points = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)[:, 1:4] if path.exists() else np.empty((0, 3))
        maps[case] = (points @ np.array(run["world_registration_rotation"]).T
                      + np.array(run["world_registration_translation_m"]))
    return trajectories, maps


def presentation_figures(output, summary, trajectories, maps):
    os.environ.setdefault("MPLCONFIGDIR", str(output / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Noto Sans CJK JP", "DejaVu Sans"],
                         "font.size": 10, "axes.unicode_minus": False, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True, "grid.alpha": .17,
                         "axes.axisbelow": True, "savefig.facecolor": "white"})
    directory = output / "figures"
    directory.mkdir(exist_ok=True)
    labels = [LABELS[case] for case in CASES]
    colors = [COLORS[case] for case in CASES]
    results = summary["results"]

    def save(fig, name):
        fig.savefig(directory / name, dpi=165, bbox_inches="tight")
        plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13.4, 3.6), constrained_layout=True)
    for ax, key, title, scale, unit in zip(
        axes, ("holdout_position_rmse_m_median", "holdout_rotation_rmse_deg_median", "wall_s_median"),
        ("后段位置误差 ↓", "后段姿态误差 ↓", "处理整段数据的耗时 ↓"), (100, 1, 1), ("厘米", "度", "秒"),
    ):
        values = [results[case][key] * scale for case in CASES]
        bars = ax.bar(labels, values, color=colors, width=.56)
        ax.bar_label(bars, fmt="%.2f", padding=5, fontsize=12)
        ax.set(title=title, ylabel=unit, ylim=(0, max(values) * 1.24))
    save(fig, "report_overview.png")

    first = trajectories[CASES[0]]
    reference = first["reference"]
    fig = plt.figure(figsize=(13.3, 9), constrained_layout=True)
    ax3, ax2 = fig.add_subplot(221, projection="3d"), fig.add_subplot(222)
    for ax in (ax3, ax2):
        ax.plot(*(reference.T if ax is ax3 else reference[:, :2].T), color="#172c40", lw=2, label="NOKOV 参考")
        for case in CASES:
            p = trajectories[case]["position"]
            ax.plot(*(p.T if ax is ax3 else p[:, :2].T), color=COLORS[case], lw=1.1, label=LABELS[case])
        ax.set(xlabel="X / 米", ylabel="Y / 米")
        ax.legend(fontsize=8)
    ax3.set(title="三维轨迹", zlabel="Z / 米")
    ax2.set(title="轨迹俯视图", aspect="equal")
    for slot, key, scale, title, ylabel in (
        (223, "position_error", 100, "位置误差随时间变化", "位置误差 / 厘米"),
        (224, "rotation_error", 1, "姿态误差随时间变化", "姿态误差 / 度"),
    ):
        ax = fig.add_subplot(slot)
        for case in CASES:
            tr = trajectories[case]
            ax.plot(tr["time"], tr[key] * scale, color=COLORS[case], lw=1, label=LABELS[case])
        ax.axvspan(0, 20, color="#9baaba", alpha=.16, label="标定区间")
        ax.axvline(20, color="#7a8a9b", ls="--", lw=1)
        ax.set(title=title, xlabel="录制时间 / 秒", ylabel=ylabel)
        ax.legend(fontsize=8)
    save(fig, "report_trajectory.png")

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 3.7), constrained_layout=True)
    x = np.arange(3)
    for case_index, case in enumerate(CASES):
        run = results[case]["repeats"]
        axes[0].plot(x + 1, [r["wall_s"] for r in run], "o-", color=COLORS[case], label=LABELS[case], lw=1.5)
        axes[1].scatter(np.full(3, case_index) + np.array([-.12, 0, .12]),
                        [r["stats"]["peak_rss_mib"] for r in run], color=COLORS[case], s=40)
        value = median_value(results[case], "stats", "peak_rss_mib")
        axes[1].hlines(value, case_index-.25, case_index+.25, color=COLORS[case], lw=3)
        axes[1].annotate(f"中位数 {value:.1f}", (case_index, value), xytext=(0, 10), textcoords="offset points", ha="center")
    axes[0].set(title="三次串行运行的整段耗时", xlabel="运行次数", ylabel="秒", xticks=[1, 2, 3], ylim=(0, 44))
    axes[0].legend()
    axes[1].set(title="峰值内存：各次测量与中位数", ylabel="MiB", xticks=x, xticklabels=labels, ylim=(0, 250))
    save(fig, "report_runtime.png")

    center = reference.mean(axis=0)
    visible = {case: points[np.linalg.norm(points-center, axis=1) <= 5] for case, points in maps.items()}
    merged = np.vstack([p for p in visible.values() if len(p)])
    low, high = merged.min(axis=0) - .12, merged.max(axis=0) + .12
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3), constrained_layout=True)
    for ax, case in zip(axes, CASES):
        p = visible[case]
        ax.scatter(p[:, 0], p[:, 1], c=p[:, 2], cmap="viridis", vmin=low[2], vmax=high[2], s=4, alpha=.65)
        ax.plot(reference[:, 0], reference[:, 1], color="#182d40", lw=1)
        ax.set(title=f"{LABELS[case]} · {len(maps[case]):,} 个地图点\n显示 {len(p):,} 个（5 米内）", xlabel="X / 米", ylabel="Y / 米",
               xlim=(low[0], high[0]), ylim=(low[1], high[1]), aspect="equal")
    save(fig, "report_sparse_maps.png")

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.1), constrained_layout=True)
    for index, (key, label, color) in enumerate((
        ("reprojection_median_px", "残差中位数", "#8fa7be"),
        ("reprojection_p95_px", "残差 P95", "#4c769d"),
        ("reprojection_rmse_px", "残差 RMSE", "#263e59"),
    )):
        values = [results[case]["mapping"][key] for case in CASES]
        bars = axes[0].bar(x + (index-1)*.23, values, width=.22, label=label, color=color)
        axes[0].bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
    axes[0].set(title="重投影残差：双目 Monado 的长尾异常", ylabel="像素（对数刻度）", yscale="log", ylim=(.35, 100), xticks=x, xticklabels=labels)
    axes[0].legend(fontsize=9, loc="upper left")
    for index, (key, label, color) in enumerate((
        ("online_vio_at_same_keyframes_holdout_error_m", "优化前 · 同一组关键帧", "#9cb0c4"),
        ("keyframe_position_error_holdout_m", "NFR 优化后", "#2b7595"),
    )):
        values = [results[case]["mapping"][key]["rmse"]*100 for case in CASES]
        bars = axes[1].bar(x + (index-.5)*.3, values, width=.28, label=label, color=color)
        axes[1].bar_label(bars, fmt="%.2f", padding=4)
    axes[1].set(title="建图优化前后：后段关键帧位置误差", ylabel="位置 RMSE / 厘米", xticks=x, xticklabels=labels, ylim=(0, 25))
    axes[1].legend(fontsize=9)
    save(fig, "report_mapping_quality.png")

    sync = np.load(output / "nokov_sync_diagnostics.npz")
    calibration = summary["calibration"]
    offset = calibration["mocap_to_imu_offset_s"]
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 3.5), constrained_layout=True, gridspec_kw={"width_ratios": [1.8, 1]})
    t = sync["time_mocap_s"] + offset
    axes[0].plot(t, sync["angular_speed_mocap"], color="#263e59", lw=1.2, label="NOKOV 角速度")
    axes[0].plot(t, sync["angular_speed_imu"], color="#13956b", lw=1, alpha=.8, label="对齐后 IMU 角速度")
    axes[0].axvspan(0, 20, color="#a0afbe", alpha=.13)
    axes[0].set(title="传感器角运动对齐（独立于 VIO）", xlabel="IMU 录制时间 / 秒", ylabel="角速度 / (rad/s)")
    axes[0].legend(fontsize=9)
    axes[1].plot(sync["offset_candidates_s"]*1000, sync["offset_cost"], color="#315d88")
    axes[1].axvline(offset*1000, color="#db8732", ls="--", label=f"估计偏移 {offset*1000:.2f} ms")
    axes[1].set(title="时钟偏移搜索", xlabel="NOKOV 时钟偏移 / 毫秒", ylabel="角速度不一致代价")
    axes[1].legend(fontsize=9)
    save(fig, "report_sync.png")


def render_report(output, summary):
    output = Path(output)
    results, info, calibration = summary["results"], summary["input"], summary["calibration"]
    trajectories, maps = saved_geometry(output, summary)
    presentation_figures(output, summary, trajectories, maps)

    def image_source(path):
        mime = "image/jpeg" if path.suffix == ".jpg" else "image/png"
        return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"

    def figure(name, caption, alt):
        path = output / "figures" / name
        return (f"<figure><a href='#' data-zoom aria-label='放大图片'>"
                f"<img src='{image_source(path)}' alt='{html.escape(alt)}' loading='lazy'></a>"
                f"<figcaption>{html.escape(caption)}</figcaption></figure>")

    def rows(values):
        return "".join(f"<tr class='{case}'><th scope='row'>{LABELS[case]}</th>" +
                       "".join(f"<td>{value}</td>" for value in values(results[case])) + "</tr>" for case in CASES)

    accuracy_rows = rows(lambda r: [f"{r['holdout_position_rmse_m_median']*100:.2f}", f"{r['holdout_rotation_rmse_deg_median']:.2f}",
                                  f"{r['global_se3_ate_all_m_median']*100:.2f}", f"{median_value(r, 'final_position_error_m')*100:.2f}"])
    rpe_rows = rows(lambda r: [f"{median_value(r, 'rpe_0.1s_holdout', 'translation_m', 'rmse')*1000:.2f}",
                             f"{median_value(r, 'rpe_0.1s_holdout', 'rotation_deg', 'rmse'):.3f}",
                             f"{median_value(r, 'rpe_1s_holdout', 'translation_m', 'rmse')*100:.2f}",
                             f"{median_value(r, 'rpe_1s_holdout', 'rotation_deg', 'rmse'):.3f}"])
    performance_rows = rows(lambda r: [f"{r['wall_s_median']:.2f}", f"{r['mean_ms_per_input_frame']:.2f}",
                                     f"{r['fps_median']:.1f}", f"{r['faster_than_recording_factor']:.2f}×",
                                     f"{median_value(r, 'stats', 'peak_rss_mib'):.1f}",
                                     f"{r['wall_s_range'][0]:.2f}–{r['wall_s_range'][1]:.2f}"])
    map_rows = rows(lambda r: [f"{r['mapping']['landmarks']:,}", f"{r['mapping']['valid_reprojection_observations']:,}",
                             f"{r['mapping']['timing']['wall_s']:.2f}", f"{r['mapping']['reprojection_rmse_px']:.3f}",
                             f"{r['mapping']['reprojection_median_px']:.3f}", f"{r['mapping']['reprojection_p95_px']:.3f}"])
    original, stereo, multi = (results[case] for case in CASES)
    position_drop = (1-multi['holdout_position_rmse_m_median']/original['holdout_position_rmse_m_median'])*100
    rpe_drop = (1-median_value(multi, 'rpe_1s_holdout', 'translation_m', 'rmse')/median_value(original, 'rpe_1s_holdout', 'translation_m', 'rmse'))*100
    reference = trajectories[CASES[0]]["reference"]
    center = reference.mean(axis=0)
    map_data = {"reference": reference[::3].round(6).tolist(), "center": center.round(6).tolist(), "cases": {}}
    for case in CASES:
        points = maps[case]
        map_data["cases"][case] = {"label": LABELS[case], "color": COLORS[case], "points": points.round(6).tolist()}
    video, poster_path = video_assets(output)
    video_source = "data:video/mp4;base64," + base64.b64encode(video.read_bytes()).decode() if video else ""
    poster = image_source(poster_path)
    template_dir = Path(__file__).parent / "templates"
    values = dict(
        session=html.escape(Path(info["session"]).name), groups=f"{info['prepared_groups']:,}",
        duration=f"{info['duration_s']:.2f}", resolution=" × ".join(map(str, info["resolution"])),
        accuracy_rows=accuracy_rows, rpe_rows=rpe_rows, performance_rows=performance_rows, map_rows=map_rows,
        video_file=video.name if video else "", video_source=video_source, video_poster=poster,
        video_hidden="" if video else "hidden",
        map_data=json.dumps(map_data, ensure_ascii=False, separators=(",", ":")),
        map_script=(template_dir / "basalt_comparison_maps.js").read_text(),
        contact_figure=figure("recording_contact_sheet.jpg", "四路原始画面 × 五个时刻。双目方案使用前两列；四目方案同时使用全部四列。", "cam0 至 cam3 在 0、13.4、26.8、40.2、53.8 秒的真实鱼眼画面"),
        overview_figure=figure("report_overview.png", "三次运行中位数。位置与姿态误差统计录制 20 秒之后的区间。", "位置误差、姿态误差与整段处理耗时的三方案对比"),
        trajectory_figure=figure("report_trajectory.png", "代表运行：各方案第 1 次。黑线为 NOKOV；灰色区域为标定区间，后段误差保留初始化后的漂移。", "三维轨迹、轨迹俯视图、位置误差及姿态误差时序"),
        runtime_figure=figure("report_runtime.png", "每个点对应一次真实测量；内存图的横线标示三次运行中位数。", "三次重复运行的总耗时与峰值内存占用"),
        sparse_figure=figure("report_sparse_maps.png", "统一坐标与显示范围；颜色表示高度，黑线为 NOKOV 轨迹。仅显示轨迹中心 5 米内的地图点。", "三种方案的稀疏地图俯视对照"),
        mapping_figure=figure("report_mapping_quality.png", "左：重投影残差分布摘要。右：同一次运行、同一批后段关键帧，保留相同世界坐标配准，不重新拟合。", "重投影残差的中位数、P95、RMSE，以及关键帧位置误差优化前后对照"),
        sync_figure=figure("report_sync.png", "同步与安装外参仅由前 20 秒原始 IMU / NOKOV 估计，未使用 VIO 结果拟合。", "NOKOV 与 IMU 的角运动对齐，以及时钟偏移搜索曲线"),
        position_drop=f"{position_drop:.1f}", rpe_drop=f"{rpe_drop:.1f}",
        stereo_drop=f"{(1-stereo['holdout_position_rmse_m_median']/original['holdout_position_rmse_m_median'])*100:.1f}",
        four_vs_stereo=f"{(1-multi['holdout_position_rmse_m_median']/stereo['holdout_position_rmse_m_median'])*100:.1f}",
        multi_position=f"{multi['holdout_position_rmse_m_median']*100:.2f}", multi_ate=f"{multi['global_se3_ate_all_m_median']*100:.2f}",
        multi_fps=f"{multi['fps_median']:.1f}", multi_points=f"{multi['mapping']['landmarks']:,}",
        vio_cost=f"{multi['wall_s_median']/original['wall_s_median']:.2f}",
        stereo_cost_delta=f"{(stereo['wall_s_median']/original['wall_s_median']-1)*100:.1f}",
        map_point_gain=f"{(multi['mapping']['landmarks']/original['mapping']['landmarks']-1)*100:.1f}",
        original_angle=f"{original['holdout_rotation_rmse_deg_median']:.2f}", stereo_angle=f"{stereo['holdout_rotation_rmse_deg_median']:.2f}", multi_angle=f"{multi['holdout_rotation_rmse_deg_median']:.2f}",
        stereo_map_rmse=f"{stereo['mapping']['reprojection_rmse_px']:.2f}", stereo_map_median=f"{stereo['mapping']['reprojection_median_px']:.2f}",
        multi_keyframe_before=f"{multi['mapping']['online_vio_at_same_keyframes_holdout_error_m']['rmse']*100:.2f}",
        multi_keyframe_after=f"{multi['mapping']['keyframe_position_error_holdout_m']['rmse']*100:.2f}",
        offset_ms=f"{calibration['mocap_to_imu_offset_s']*1000:.2f}", lever_mm=f"{calibration['lever_arm_norm_m']*1000:.2f}",
        gyro_residual=f"{calibration['angular_velocity_residual']['validation_rad_s']['rmse']:.3f}",
        accel_residual=f"{calibration['acceleration_residual']['validation_m_s2']['rmse']:.3f}",
        camera_delta_us=f"{info['camera_delta_ns_max']/1000:.0f}", frame_gap_ms=f"{info['frame_gap_ms_max']:.3f}",
    )
    document = Template((template_dir / "basalt_comparison_report.html").read_text()).substitute(values)
    (output / "comparison_report.html").write_text(document, encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    render_report(args.output, json.loads((args.output / "comparison_summary.json").read_text()))
    print(args.output / "comparison_report.html")
