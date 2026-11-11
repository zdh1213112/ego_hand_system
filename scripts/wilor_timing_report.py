"""Small, readable averages built from the independent timing recordings."""
from __future__ import annotations

import csv
import json
from pathlib import Path


FRAME_STEPS = [
    ("video.decode", "图像解码"),
    ("detector", "YOLO 手部检测"),
    ("preprocess.crops", "手部裁剪与预处理"),
    ("transfer.h2d", "输入数据传输"),
    ("model.total", "WiLoR 前向推理（分摊到每帧）"),
    ("postprocess", "预测后处理"),
    ("output.jsonl", "结果写出"),
    ("cleanup", "内存与缓存清理"),
]
NETWORK_STEPS = [
    ("vit.features", "ViT 骨干网络（特征提取）"),
    ("mano.parameter_decoder", "MANO 参数解码器"),
    ("mano.coarse", "初始 MANO 网格计算"),
    ("refinenet.total", "RefineNet 精修网络"),
    ("mano.final", "最终 MANO 网格与关节点计算"),
]
PIPELINE_STEPS = {
    "normalize_orbbec_ego4_session": "录制整理与同步（摊销）",
    "normalize_multiview_recording": "录制整理与同步（摊销）",
    "assist_wilor_with_glove_markers": "手套标记辅助（摊销）",
    "sam3_glove_assist": "SAM3 辅助（摊销）",
    "fuse_multiview_wilor_guided": "多视角融合与时序处理（摊销）",
    "render_multiview_wilor": "诊断视频渲染（摊销）",
    "render_hybrid_multiview": "交付视频渲染（摊销）",
}


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def build_averages(directory, stages, components, command):
    output = directory.parent
    if "--output" in command:
        output = Path(command[command.index("--output") + 1]).resolve()
    inference = load_json(output / "wilor_multiview/summary.json") or {}
    frame_count = int(inference.get("frame_count", 0))
    cameras = inference.get("camera_ids", [])
    camera_frames = sum(row["frame_count"] for row in inference.get("cameras", {}).values())
    if not camera_frames:
        camera_frames = frame_count * len(cameras)
    components = components or {}
    entries = components.get("components", {})
    samples_path = directory / "samples.csv"
    samples = []
    if samples_path.exists():
        with samples_path.open(newline="", encoding="utf-8") as stream:
            samples = list(csv.DictReader(stream))
    forwards = [row for row in samples if row["name"] == "model.total" and row["status"] == "ok"]
    forward_keys = {(row["camera"], row["batch_index"]) for row in forwards}

    def totals(name, metric="elapsed_ms", model_only=False):
        if samples:
            rows = [row for row in samples if row["name"] == name and row["status"] == "ok"]
            if model_only:
                rows = [row for row in rows if (row["camera"], row["batch_index"]) in forward_keys]
            # Old CPU recordings have host_wall_ms only; they remain usable.
            values = [row.get(metric) or (row.get("host_wall_ms") if metric == "elapsed_ms" else None)
                      for row in rows]
            return sum(float(value) for value in values if value not in (None, "")), sum(value not in (None, "") for value in values)
        entry = entries.get(name, {})
        values = entry.get(metric, entry.get("host_wall_ms", {}) if metric == "elapsed_ms" else {}).get("all", {})
        return values.get("total_ms", 0), values.get("calls", 0)

    frame_rows = []
    if camera_frames:
        for key, label in FRAME_STEPS:
            total, count = totals(key)
            if key == "postprocess":
                extra_total, extra_count = totals("postprocess.camera")
                total, count = total + extra_total, count + extra_count
            if key == "cleanup":
                extra_total, extra_count = totals("cleanup.frame")
                total, count = total + extra_total, count + extra_count
            # An absent forward (no detections in the entire recording) costs zero per image.
            if count or key == "model.total":
                frame_rows.append({"step": label, "mean_ms": total / camera_frames, "unit": "单路图像"})
        if "inference_seconds" in inference:
            frame_rows.append({"step": "推理流程合计（实测，含解码、检测及后处理）",
                               "mean_ms": inference["inference_seconds"] * 1000 / camera_frames,
                               "unit": "单路图像"})
    pipeline_rows = []
    for stage in stages:
        if stage["stage"] in PIPELINE_STEPS and stage["exit_code"] == 0 and frame_count:
            pipeline_rows.append({"step": PIPELINE_STEPS[stage["stage"]],
                                  "mean_ms": stage["wall_seconds"] * 1000 / frame_count,
                                  "unit": "同步帧组"})
    # Put normalization before per-image inference, followed by fusion and rendering.
    normalize = [row for row in pipeline_rows if row["step"].startswith("录制")]
    frame_rows = normalize + frame_rows + [row for row in pipeline_rows if row not in normalize]

    metric = "cuda_ms" if totals("model.total", "cuda_ms")[1] else "host_wall_ms"
    total_ms, forward_count = totals("model.total", metric)
    network_rows = []
    if forward_count:
        specs = NETWORK_STEPS
        if not totals("vit.features", metric, True)[1] and totals("backbone.total", metric, True)[1]:
            specs = [("backbone.total", "ViT 与 MANO 参数解码器（编译整体）"), *NETWORK_STEPS[2:]]
        for key, label in specs:
            value, count = totals(key, metric, True)
            network_rows.append({"module": label, "mean_ms": value / forward_count if count == forward_count else None})
        if all(row["mean_ms"] is not None for row in network_rows):
            remainder = total_ms / forward_count - sum(row["mean_ms"] for row in network_rows)
            if remainder >= -0.001:
                network_rows.append({"module": "其他张量与相机运算（含计时开销）", "mean_ms": max(0, remainder)})
        network_rows.append({"module": "一次 WiLoR 前向推理总耗时", "mean_ms": total_ms / forward_count})
    batch_sizes = [int(row["hypotheses"]) for row in forwards if row.get("hypotheses")]
    return {
        "device": components.get("device"), "synchronized_frames": frame_count,
        "camera_count": len(cameras), "image_count": camera_frames,
        "frame_averages": frame_rows,
        "wilor_forward": {
            "successful_calls": forward_count,
            "mean_ms": total_ms / forward_count if forward_count else None,
            "mean_hand_crops_per_call": sum(batch_sizes) / len(batch_sizes) if batch_sizes else None,
            "timing_method": "CUDA Event" if metric == "cuda_ms" else "CPU 墙钟",
            "modules": network_rows,
        },
    }


def write_simple_report(directory, averages, elapsed, returncode):
    count = averages["image_count"]
    forward = averages["wilor_forward"]
    lines = ["# 每帧处理与 WiLoR 前向平均耗时", "",
             f"设备：{averages['device'] or '未知'}；处理 {averages['synchronized_frames']} 组同步帧 × "
             f"{averages['camera_count']} 路相机 = {count} 张图像。", ""]
    if forward["mean_ms"] is not None:
        lines += [f"**平均一次 WiLoR 前向推理：{forward['mean_ms']:.3f} ms**（{forward['successful_calls']} 次成功调用）。", ""]
    if returncode:
        lines += [f"本次运行未完成（退出码 {returncode}），仅展示已采集的数据；错误见 run.log。", ""]
    lines += ["## 每帧各环节", "",
              "单路图像 = 一台相机的一张图；同步帧组 = 同一时刻所有相机的图像。", "",
              "| 处理环节 | 平均耗时 (ms) | 每次对应 |", "| --- | ---: | --- |"]
    for row in averages["frame_averages"]:
        lines.append(f"| {row['step']} | {row['mean_ms']:.3f} | {row['unit']} |")
    if not count:
        lines += ["", "未采集到完整帧数，无法计算每帧平均耗时。"]
    lines += ["", "每帧均值 = 环节累计耗时 ÷ 图像数（无手帧也计入）。解码可与推理重叠，合计使用实测总时间。",
              "同步、融合、渲染按同步帧组摊销，包含其进程启动和文件读写。模型加载不属于逐帧推理。", "",
              "## 一次 WiLoR 前向推理", ""]
    if forward["successful_calls"]:
        crops = forward["mean_hand_crops_per_call"]
        lines += [f"按完整 forward 调用统计，包含首次调用，计时方式为 {forward['timing_method']}。"]
        if crops is not None:
            lines += [f"每次前向平均处理 {crops:.2f} 个手部裁剪；一次前向是一个批次，并不等于一张原始图像。"]
        lines += ["", "| 主要模块（执行顺序） | 平均耗时 (ms/次前向) |", "| --- | ---: |"]
        for row in forward["modules"]:
            value = "未采集" if row["mean_ms"] is None else f"{row['mean_ms']:.3f}"
            lines.append(f"| {row['module']} | {value} |")
        lines += ["", "ViT 特征提取与 MANO 参数解码分别列出；两次 MANO 网格计算分别位于 RefineNet 前后。"]
    else:
        lines += ["未采集到成功的 WiLoR 前向调用；查看 run.log 确认是否无手部检测或发生错误。"]
    lines += ["", f"本次程序总耗时（含启动、加载及全部阶段）：{elapsed:.3f} s。",
              "原始计时保存在 samples.csv 和 components.json；详细记录不在主表展开。"]
    (directory / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for filename, rows, fields in [
        ("frame_average.csv", averages["frame_averages"], ["step", "mean_ms", "unit"]),
        ("wilor_forward_average.csv", forward["modules"], ["module", "mean_ms"]),
    ]:
        with (directory / filename).open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fields)
            writer.writeheader()
            writer.writerows(rows)
