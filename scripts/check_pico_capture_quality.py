#!/usr/bin/env python3
"""Audit PICO raw head/Tracker CSVs, without NOKOV or an SDK (standard library only)."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import statistics
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field
from itertools import pairwise
from pathlib import Path

POSE = ("x_m", "y_m", "z_m", "qx", "qy", "qz", "qw")
HOST = ("host_receive_perf_ns", "host_receive_unix_ns")
DERIVATIVES = (
    "vx_m_s",
    "vy_m_s",
    "vz_m_s",
    "wx_rad_s",
    "wy_rad_s",
    "wz_rad_s",
    "ax_m_s2",
    "ay_m_s2",
    "az_m_s2",
    "alphax_rad_s2",
    "alphay_rad_s2",
    "alphaz_rad_s2",
)
ISSUE_FIELDS = (
    "severity",
    "stream",
    "csv_line",
    "sample_index",
    "timestamp_ns",
    "code",
    "detail",
)
SAMPLE_FIELDS = (
    "stream",
    "csv_line",
    "sample_index",
    "timestamp_ns",
    "time_s",
    "pose_valid",
    "quaternion_norm",
    "device_interval_ms",
    "host_interval_ms",
    "position_step_mm",
    "rotation_step_deg",
    "speed_m_s",
    "angular_speed_deg_s",
    "review",
    "reasons",
)
LIMITATIONS = (
    "本报告只分析 PICO 自身的 pico_head_poses.csv / pico_tracker_poses.csv，不读取 NOKOV 刚体数据。",
    "valid_numeric 仅表示数值有效；该 v1 格式没有物理跟踪状态，无法据此证明跟踪正常或绝对精度。",
    "突跳与长间隔阈值用于定位复核样本。静止不等于冻结故障；速度/加速度全零不等于设备没有运动。",
    "poll_hz 是主机轮询频率；设备更新率来自原始设备时间戳。去重轮询计数不作为录制丢帧数。",
    "头显 sample_index 与 Tracker 的 motion packet 序号分别计数；Tracker 按序列号分组，不推断左右身份。",
    "不排序、补帧、平滑或修复原始样本；不跨无效样本、序号缺口、时间倒退或长间隔计算轨迹变化。",
    "设备时间与主机时间时钟域不同，不直接相减估计单向延迟；未进行 PICO/NOKOV 同步、外参或精度评估。",
)


@dataclass(frozen=True)
class Thresholds:
    position_step_mm: float = 50.0
    rotation_step_deg: float = 5.0
    quaternion_norm_error: float = 0.001
    gap_factor: float = 2.5
    expected_hz: float | None = None


@dataclass
class Sample:
    row: dict
    line: int
    index: int | None
    timestamp: int | None
    host: int | None
    unix: int | None
    pose: tuple | None
    norm: float | None
    reasons: set = field(default_factory=set)
    interval: float | None = None
    host_interval: float | None = None
    step: float | None = None
    angle: float | None = None
    speed: float | None = None
    angular_speed: float | None = None


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def integer(value, minimum=0):
    try:
        result = int(value)
        return result if result >= minimum else None
    except (ValueError, TypeError):
        return None


def number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (ValueError, TypeError):
        return None


def stats(values):
    ordered = sorted(v for v in values if v is not None and math.isfinite(v))
    if not ordered:
        return {
            "count": 0,
            "min": None,
            "median": None,
            "p95": None,
            "max": None,
            "mean": None,
        }

    def percentile(p):
        index = (len(ordered) - 1) * p
        low = math.floor(index)
        return ordered[low] + (ordered[math.ceil(index)] - ordered[low]) * (index - low)

    return {
        "count": len(ordered),
        "min": ordered[0],
        "median": percentile(0.5),
        "p95": percentile(0.95),
        "max": ordered[-1],
        "mean": statistics.fmean(ordered),
    }


def add_issue(issues, code, detail, stream="capture", sample=None, severity="review"):
    issues.append(
        dict(
            zip(
                ISSUE_FIELDS,
                (
                    severity,
                    stream,
                    sample.line if sample else "",
                    sample.index if sample else "",
                    sample.timestamp if sample else "",
                    code,
                    detail,
                ),
            )
        )
    )
    if sample is not None and severity == "review":
        sample.reasons.add(code)


def read_csv(path, required, issues):
    if not path.is_file():
        add_issue(issues, "missing_file", str(path))
        return [], []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        if len(fields) != len(set(fields)) or set(required) - set(fields):
            add_issue(
                issues,
                "invalid_columns",
                f"{path.name}: missing {sorted(set(required) - set(fields))}, or duplicate headers",
            )
            return [], fields
        rows = []
        for row in reader:
            line = reader.line_num
            if None in row or any(v is None for v in row.values()):
                add_issue(
                    issues,
                    "malformed_csv_row",
                    f"{path.name}:{line}: inconsistent column count",
                )
                # Retain the row: otherwise a bad pose could disappear before continuity checks.
            rows.append((line, {k: v for k, v in row.items() if k is not None}))
        return rows, fields


def parse_sample(row, line, timestamp_field, thresholds, issues, stream):
    values = tuple(number(row.get(k)) for k in POSE)
    norm = math.hypot(*values[3:]) if all(v is not None for v in values[3:]) else None
    valid = (
        all(v is not None for v in values)
        and norm is not None
        and abs(norm - 1) <= thresholds.quaternion_norm_error
    )
    sample = Sample(
        row,
        line,
        integer(row.get("sample_index")),
        integer(row.get(timestamp_field), 1),
        integer(row.get(HOST[0]), 1),
        integer(row.get(HOST[1]), 1),
        values if valid and row.get("valid_numeric") == "1" else None,
        norm,
    )
    if sample.index is None:
        add_issue(
            issues,
            "invalid_sample_index",
            "expected a nonnegative integer",
            stream,
            sample,
        )
    if sample.timestamp is None or sample.host is None or sample.unix is None:
        add_issue(
            issues,
            "invalid_timestamp",
            "device and host timestamps must be positive integer nanoseconds",
            stream,
            sample,
        )
    if row.get("valid_numeric") != "1":
        add_issue(
            issues,
            "numeric_flag_invalid",
            "valid_numeric is not 1 (includes missing Tracker placeholder rows)",
            stream,
            sample,
        )
    if not valid:
        add_issue(
            issues,
            "invalid_pose",
            "nonfinite/blank pose or quaternion norm outside tolerance",
            stream,
            sample,
        )
    return sample


def analyze_stream(name, samples, thresholds, issues):
    if not samples:
        add_issue(issues, "empty_stream", "no recorded samples", name)
    elif len(samples) < 2:
        add_issue(
            issues,
            "insufficient_stream_samples",
            "need at least two samples to check continuity",
            name,
        )
    if samples and samples[0].index is not None and samples[0].index != 0:
        add_issue(
            issues,
            "nonzero_first_sample_index",
            "capture indices must start at zero",
            name,
            samples[0],
        )
    times = [s.timestamp for s in samples]
    positive_dt = [
        (b - a) * 1e-9
        for a, b in pairwise(times)
        if a is not None and b is not None and b > a
    ]
    period = (
        1 / thresholds.expected_hz
        if thresholds.expected_hz
        else (statistics.median(positive_dt) if positive_dt else None)
    )
    gap_limit = period * thresholds.gap_factor if period else None
    edge_counts = Counter()
    repeats, longest_repeat = 0.0, 0.0
    for previous, sample in pairwise(samples):
        contiguous_index = (
            previous.index is not None
            and sample.index is not None
            and sample.index == previous.index + 1
        )
        if (
            previous.index is not None
            and sample.index is not None
            and not contiguous_index
        ):
            code = (
                "sample_index_gap"
                if sample.index > previous.index
                else "non_increasing_sample_index"
            )
            add_issue(
                issues, code, f"previous sample_index={previous.index}", name, sample
            )
            edge_counts[code] += 1
        positive_time = (
            previous.timestamp is not None
            and sample.timestamp is not None
            and sample.timestamp > previous.timestamp
        )
        if previous.timestamp is not None and sample.timestamp is not None:
            sample.interval = (sample.timestamp - previous.timestamp) * 1e-6
            if not positive_time:
                add_issue(
                    issues,
                    "non_increasing_device_time",
                    f"previous timestamp_ns={previous.timestamp}",
                    name,
                    sample,
                )
                edge_counts["non_increasing_device_time"] += 1
        for field_name, old, new in (
            ("perf", previous.host, sample.host),
            ("unix", previous.unix, sample.unix),
        ):
            if old is not None and new is not None and new <= old:
                add_issue(
                    issues,
                    "non_increasing_host_" + field_name,
                    f"previous={old}, current={new}",
                    name,
                    sample,
                )
        if previous.host is not None and sample.host is not None:
            sample.host_interval = (sample.host - previous.host) * 1e-6
        gap = (
            positive_time
            and gap_limit is not None
            and sample.interval * 1e-3 > gap_limit
        )
        if gap:
            add_issue(
                issues,
                "device_time_gap",
                f"interval_ms={sample.interval:.6f}, limit_ms={gap_limit * 1000:.6f}",
                name,
                sample,
            )
            edge_counts["device_time_gap"] += 1
        broken_time = {
            "invalid_timestamp",
            "non_increasing_device_time",
            "packet_timestamp_conflict",
        }
        if (
            previous.pose is None
            or sample.pose is None
            or not positive_time
            or gap
            or not contiguous_index
            or previous.reasons & broken_time
            or sample.reasons & broken_time
        ):
            repeats = 0
            continue
        sample.step = math.dist(previous.pose[:3], sample.pose[:3]) * 1000
        q0 = [v / previous.norm for v in previous.pose[3:]]
        q1 = [v / sample.norm for v in sample.pose[3:]]
        dot = min(1.0, abs(sum(a * b for a, b in zip(q0, q1))))
        sample.angle = math.degrees(2 * math.acos(dot)) if dot < 1 - 1e-15 else 0.0
        seconds = (sample.timestamp - previous.timestamp) * 1e-9
        sample.speed, sample.angular_speed = (
            sample.step * 0.001 / seconds,
            sample.angle / seconds,
        )
        if sample.step > thresholds.position_step_mm:
            add_issue(
                issues,
                "position_step",
                f"previous sample_index={previous.index}, step_mm={sample.step:.6f}",
                name,
                sample,
            )
            edge_counts["position_step"] += 1
        if sample.angle > thresholds.rotation_step_deg:
            add_issue(
                issues,
                "rotation_step",
                f"previous sample_index={previous.index}, step_deg={sample.angle:.6f}",
                name,
                sample,
            )
            edge_counts["rotation_step"] += 1
        if previous.pose[:3] == sample.pose[:3] and sample.angle == 0:
            repeats += seconds
            longest_repeat = max(longest_repeat, repeats)
        else:
            repeats = 0
    valid_times = [t for t in times if t is not None]
    ordered = (
        len(valid_times) == len(samples)
        and bool(valid_times)
        and all(b > a for a, b in pairwise(valid_times))
    )
    duration = (
        (valid_times[-1] - valid_times[0]) * 1e-9
        if len(valid_times) > 1 and ordered
        else None
    )
    poses = [s.pose for s in samples if s.pose is not None]
    auxiliary = {}
    for key in DERIVATIVES:
        if any(key in s.row for s in samples):
            values = [number(s.row.get(key)) for s in samples]
            auxiliary[key] = {
                "finite_rows": sum(v is not None for v in values),
                "all_finite_values_zero": bool(values) and all(v == 0 for v in values),
            }
    return {
        "stream": name,
        "rows": len(samples),
        "numeric_valid_rows": len(poses),
        "review_samples": sum(bool(s.reasons) for s in samples),
        "device_duration_s": duration,
        "average_hz": (len(samples) - 1) / duration if duration else None,
        "device_time_strictly_increasing": ordered,
        "device_interval_ms": stats(s.interval for s in samples),
        "host_interval_ms": stats(s.host_interval for s in samples),
        "gap_limit_ms": gap_limit * 1000 if gap_limit else None,
        "gap_period_source": "explicit_expected_hz"
        if thresholds.expected_hz
        else "median_positive_device_interval",
        "edge_counts": dict(edge_counts),
        "position_step_mm": stats(s.step for s in samples),
        "rotation_step_deg": stats(s.angle for s in samples),
        "speed_m_s": stats(s.speed for s in samples),
        "angular_speed_deg_s": stats(s.angular_speed for s in samples),
        "position_range_xyz_mm": [
            (max(p[k] for p in poses) - min(p[k] for p in poses)) * 1000
            for k in range(3)
        ]
        if poses
        else None,
        "quaternion_norm_error": stats(
            abs(s.norm - 1) for s in samples if s.norm is not None
        ),
        "longest_identical_pose_duration_s": longest_repeat,
        "auxiliary_fields": auxiliary,
        "tracking_state_assessment": "not_available_in_v1_pose_format",
    }


def missing_ranges(indices, expected):
    """Compress absent packet IDs without allocating range(expected)."""
    present = sorted({i for i in indices if i is not None and 0 <= i < expected})
    missing, cursor = [], 0
    for i in present:
        if cursor < i:
            missing.append([cursor, i - 1])
        cursor = i + 1
    if cursor < expected:
        missing.append([cursor, expected - 1])
    return expected - len(present), missing


def analyze_capture(path, thresholds=None):
    thresholds = thresholds if thresholds is not None else Thresholds()
    directory = path.resolve()
    if (directory / "pico").is_dir():
        directory /= "pico"
    if not directory.is_dir():
        raise ValueError(f"not a session/PICO directory: {path}")
    name = directory.parent.name if directory.name == "pico" else directory.name
    source_paths = [
        directory / n
        for n in (
            "capture_metadata.json",
            "pico_head_poses.csv",
            "pico_tracker_poses.csv",
            "events.csv",
        )
        if (directory / n).is_file()
    ]
    hashes = {str(p): sha256(p) for p in source_paths}
    issues, metadata = [], {}
    meta_path = directory / "capture_metadata.json"
    if meta_path.is_file():
        try:
            metadata = json.loads(meta_path.read_text(encoding="utf-8-sig"))
            if not isinstance(metadata, dict):
                raise TypeError("metadata must be an object")
        except (ValueError, TypeError) as exc:
            metadata = {}
            add_issue(issues, "invalid_metadata", str(exc))
    else:
        add_issue(issues, "missing_metadata", str(meta_path))
    if metadata and metadata.get("schema") != "pico_raw_tracking_capture_v1":
        add_issue(issues, "unsupported_metadata_schema", str(metadata.get("schema")))
    if metadata.get("status") != "complete" or metadata.get("error"):
        add_issue(
            issues,
            "capture_not_complete",
            f"status={metadata.get('status')}, error={metadata.get('error')}",
        )
    base = ("sample_index", *HOST, "valid_numeric", *POSE)
    head, _ = read_csv(
        directory / "pico_head_poses.csv", (*base, "head_timestamp_ns"), issues
    )
    tracker, _ = read_csv(
        directory / "pico_tracker_poses.csv",
        (*base, "motion_timestamp_ns", "tracker_serial", "tracker_index", "selected"),
        issues,
    )
    raw_groups = {"head": head}
    for line, row in tracker:
        serial = row.get("tracker_serial") or "<missing_serial>"
        raw_groups.setdefault("tracker:" + serial, []).append((line, row))
        if serial == "<missing_serial>":
            add_issue(issues, "missing_tracker_serial", f"CSV line {line}")
    selected = metadata.get("selected_tracker_serials", [])
    selected = selected if isinstance(selected, list) else []
    observed = metadata.get("observed_tracker_serials", [])
    observed = observed if isinstance(observed, list) else []
    for serial in sorted(set(selected + observed)):
        raw_groups.setdefault("tracker:" + serial, [])
    if not any(k.startswith("tracker:") for k in raw_groups):
        add_issue(issues, "no_tracker_streams", "no Tracker serials in CSV or metadata")
    samples = {
        key: [
            parse_sample(
                row,
                line,
                "head_timestamp_ns" if key == "head" else "motion_timestamp_ns",
                thresholds,
                issues,
                key,
            )
            for line, row in rows
        ]
        for key, rows in raw_groups.items()
    }
    packet_timestamps = {}
    for key, rows in samples.items():
        if key == "head":
            continue
        for sample in rows:
            if sample.index is None or sample.timestamp is None:
                continue
            previous = packet_timestamps.setdefault(sample.index, sample.timestamp)
            if previous != sample.timestamp:
                add_issue(
                    issues,
                    "packet_timestamp_conflict",
                    f"same motion packet has timestamps {previous} and {sample.timestamp}",
                    key,
                    sample,
                )
    declared = metadata.get("rows", {})
    declared = declared if isinstance(declared, dict) else {}
    expected = integer(declared.get("motion_packets"))
    if expected is None:
        expected = max(packet_timestamps, default=-1) + 1
    for key, rows in samples.items():
        if key == "head":
            continue
        count, ranges = missing_ranges([s.index for s in rows], expected)
        if count:
            add_issue(
                issues,
                "missing_tracker_packets",
                f"missing_rows={count}, packet_index_ranges={ranges}",
                key,
            )
        if any(s.index is not None and s.index >= expected for s in rows):
            add_issue(
                issues,
                "packet_index_outside_metadata",
                f"motion_packets={expected}",
                key,
            )
    actual = {
        "head": len(head),
        "tracker": len(tracker),
        "motion_packets": len(packet_timestamps),
    }
    for key, value in actual.items():
        if key in declared and declared[key] != value:
            add_issue(
                issues,
                "metadata_row_mismatch",
                f"{key}: metadata={declared[key]}, CSV={value}",
            )
    for key in (
        "missing_selected_tracker_serials",
        "selected_tracker_missing_frame_rows",
    ):
        if metadata.get(key):
            add_issue(issues, "capture_missing_trackers", f"{key}={metadata[key]}")
    summaries = [
        analyze_stream(key, rows, thresholds, issues) for key, rows in samples.items()
    ]
    unchanged = all(sha256(Path(p)) == h for p, h in hashes.items())
    if not unchanged:
        raise ValueError("input files changed during analysis")
    report = {
        "schema": "pico_capture_quality_v1",
        "session": name,
        "source": "PICO_raw_tracking",
        "pico_directory": str(directory),
        "status": "needs_review"
        if any(i["severity"] == "review" for i in issues)
        else "ok",
        "absolute_accuracy_evaluated": False,
        "thresholds": asdict(thresholds),
        "recorded_rows": actual,
        "poll_hz_metadata": metadata.get("poll_hz"),
        "duplicate_polls_not_written": metadata.get(
            "duplicate_or_non_increasing_polls_not_written",
            metadata.get("duplicate_polls_not_written"),
        ),
        "streams": summaries,
        "issue_counts": dict(Counter(i["code"] for i in issues)),
        "input_sha256": hashes,
        "checker_sha256": sha256(Path(__file__)),
        "raw_inputs_unchanged": unchanged,
        "limitations": LIMITATIONS,
    }
    return report, samples, issues


def write_json(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def table(headers, rows):
    escape = lambda x: html.escape(str(x))
    return (
        "<table><tr>"
        + "".join(f"<th>{escape(x)}</th>" for x in headers)
        + "</tr>"
        + "".join(
            "<tr>" + "".join(f"<td>{escape(x)}</td>" for x in row) + "</tr>"
            for row in rows
        )
        + "</table>"
    )


def fmt(value):
    return f"{value:.3f}" if isinstance(value, (float, int)) else "—"


def plot_svg(samples, attribute, limit):
    """Keep bin extrema so isolated spikes survive report downsampling."""
    origin = next((s.timestamp for s in samples if s.timestamp is not None), 0)
    points = [
        ((s.timestamp - origin) * 1e-9, getattr(s, attribute))
        if s.timestamp is not None
        else (None, None)
        for s in samples
    ]
    finite = [
        (t, v)
        for t, v in points
        if t is not None and v is not None and math.isfinite(v)
    ]
    if not finite:
        return "<p>没有可计算的连续有效样本。</p>"
    xmin, xmax = min(t for t, _ in finite), max(t for t, _ in finite)
    ymax = max(limit, max(v for _, v in finite), 1e-9) * 1.08

    def xy(t, v):
        return f"{55 + (t - xmin) / max(xmax - xmin, 1e-9) * 800:.2f},{175 - v / ymax * 150:.2f}"

    lines, run = [], []

    def flush():
        if not run:
            return
        chosen = set()
        bin_size = max(1, len(run) // 400)
        for start in range(0, len(run), bin_size):
            indices = list(range(start, min(start + bin_size, len(run))))
            chosen.update(
                (
                    indices[0],
                    indices[-1],
                    min(indices, key=lambda i: run[i][1]),
                    max(indices, key=lambda i: run[i][1]),
                )
            )
        lines.append(
            '<polyline fill="none" stroke="#235ca7" stroke-width="1" points="'
            + " ".join(xy(*run[i]) for i in sorted(chosen))
            + '"/>'
        )
        run.clear()

    for t, v in points:
        if t is None or v is None or not math.isfinite(v):
            flush()
        else:
            if run and t <= run[-1][0]:
                flush()
            run.append((t, v))
    flush()
    ylimit = 175 - limit / ymax * 150
    return (
        f'<svg viewBox="0 0 900 210" role="img"><path d="M55 20V175H855" fill="none" stroke="#888"/><line x1="55" x2="855" y1="{ylimit}" y2="{ylimit}" stroke="#c33" stroke-dasharray="5 4"/><text x="5" y="25">{ymax:.2f}</text><text x="35" y="178">0</text><text x="55" y="200">{xmin:.2f} s</text><text x="780" y="200">{xmax:.2f} s</text>'
        + "".join(lines)
        + "</svg>"
    )


def document(title, content):
    return (
        '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>'
        + html.escape(title)
        + "</title><style>body{max-width:1150px;margin:32px auto;padding:0 20px;font:16px/1.6 system-ui;color:#172435}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:7px;border:1px solid #ccd6e2;text-align:left}th{background:#edf2f8}svg{max-width:100%;width:100%;background:#f9fbfe}h2{margin-top:30px}code{word-break:break-all}li{margin:8px 0}</style><h1>"
        + html.escape(title)
        + "</h1>"
        + content
        + "</html>"
    )


def write_capture(output, report, samples, issues):
    output.mkdir(parents=True, exist_ok=True)
    sample_rows = []
    for key, rows in samples.items():
        origin = next((s.timestamp for s in rows if s.timestamp is not None), 0)
        for s in rows:
            sample_rows.append(
                dict(
                    zip(
                        SAMPLE_FIELDS,
                        (
                            key,
                            s.line,
                            s.index,
                            s.timestamp,
                            (s.timestamp - origin) * 1e-9
                            if s.timestamp is not None
                            else None,
                            int(s.pose is not None),
                            s.norm,
                            s.interval,
                            s.host_interval,
                            s.step,
                            s.angle,
                            s.speed,
                            s.angular_speed,
                            int(bool(s.reasons)),
                            ";".join(sorted(s.reasons)),
                        ),
                    )
                )
            )
    write_csv(output / "sample_quality.csv", sample_rows, SAMPLE_FIELDS)
    write_csv(output / "issues.csv", issues, ISSUE_FIELDS)
    write_json(output / "summary.json", report)
    content = f"<p>数据来源：<strong>PICO 自身原始位姿</strong>；结果：{report['status']}。目录：<code>{html.escape(report['pico_directory'])}</code></p>"
    content += (
        "<ul>"
        + "".join("<li>" + html.escape(v) + "</li>" for v in LIMITATIONS)
        + "</ul>"
    )
    content += '<p><a href="summary.json">完整 JSON</a> · <a href="sample_quality.csv">逐样本清单</a> · <a href="issues.csv">异常事件</a></p>'
    content += (
        "<p>本次经验筛查阈值："
        + html.escape(json.dumps(report["thresholds"], ensure_ascii=False))
        + "。边事件标记后一个样本，issues.csv 中给出前一个序号。低于阈值只表示本轮未检出对应异常。</p>"
    )
    content += table(
        [
            "数据流（Tracker 序列号）",
            "样本",
            "数值有效",
            "需复核",
            "平均 Hz",
            "位置单步 max mm",
            "角差单步 max °",
            "长间隔",
        ],
        [
            [
                s["stream"],
                s["rows"],
                s["numeric_valid_rows"],
                s["review_samples"],
                fmt(s["average_hz"]),
                fmt(s["position_step_mm"]["max"]),
                fmt(s["rotation_step_deg"]["max"]),
                s["edge_counts"].get("device_time_gap", 0),
            ]
            for s in report["streams"]
        ],
    )
    content += "<h2>异常汇总</h2>" + table(
        ["类型", "数量"], sorted(report["issue_counts"].items())
    )
    content += "<h2>事件预览（前 80 条，完整结果见 CSV）</h2>" + table(
        ISSUE_FIELDS, [[r[k] for k in ISSUE_FIELDS] for r in issues[:80]]
    )
    for key, rows in samples.items():
        content += (
            "<h2>"
            + html.escape(key)
            + "</h2><p>位置单步变化 / mm；红虚线为复核阈值。横轴相对该流首个设备时间。</p>"
        )
        content += plot_svg(rows, "step", report["thresholds"]["position_step_mm"])
        content += "<p>相邻姿态角差 / °；q 与 −q 视为同一姿态。</p>" + plot_svg(
            rows, "angle", report["thresholds"]["rotation_step_deg"]
        )
        summary = next(s for s in report["streams"] if s["stream"] == key)
        zeros = [
            k
            for k, v in summary["auxiliary_fields"].items()
            if v["all_finite_values_zero"]
        ]
        if zeros:
            content += (
                "<p>以下附加字段全零，仅作信息记录，不按静止或坏帧判断："
                + html.escape(", ".join(zeros))
                + "。</p>"
            )
    (output / "report.html").write_text(
        document(report["session"] + " · PICO 数据质量", content), encoding="utf-8"
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "sessions",
        nargs="+",
        type=Path,
        help="session directories or their pico/ subdirectories",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="report directory outside all source session directories",
    )
    parser.add_argument("--max-position-step-mm", type=float, default=50.0)
    parser.add_argument("--max-rotation-step-deg", type=float, default=5.0)
    parser.add_argument("--quaternion-norm-tolerance", type=float, default=0.001)
    parser.add_argument("--gap-factor", type=float, default=2.5)
    parser.add_argument(
        "--expected-hz",
        type=float,
        help="optional known device update rate; never use poll_hz here",
    )
    args = parser.parse_args(argv)
    values = [
        args.max_position_step_mm,
        args.max_rotation_step_deg,
        args.quaternion_norm_tolerance,
        args.gap_factor,
    ]
    if args.expected_hz is not None:
        values.append(args.expected_hz)
    if any(not math.isfinite(v) or v <= 0 for v in values) or args.gap_factor <= 1:
        parser.error("thresholds must be finite and positive; gap-factor must exceed 1")
    limits = Thresholds(
        args.max_position_step_mm,
        args.max_rotation_step_deg,
        args.quaternion_norm_tolerance,
        args.gap_factor,
        args.expected_hz,
    )
    try:
        results = [analyze_capture(p, limits) for p in args.sessions]
        names = [r[0]["session"] for r in results]
        if len(names) != len(set(names)):
            raise ValueError("session names must be unique in one batch")
        output = args.output.resolve()
        source_roots = []
        for report, _, _ in results:
            source = Path(report["pico_directory"])
            source_roots.append(source.parent if source.name == "pico" else source)
        for destination in [output] + [output / name for name in names]:
            destination = destination.resolve()
            if any(
                destination == source or source in destination.parents
                for source in source_roots
            ):
                raise ValueError("output must be outside source session directories")
        output.mkdir(parents=True, exist_ok=True)
        aggregate = []
        for report, samples, issues in results:
            write_capture(output / report["session"], report, samples, issues)
            for stream in report["streams"]:
                aggregate.append(
                    {
                        "session": report["session"],
                        "source": "PICO_raw_tracking",
                        "status": report["status"],
                        "stream": stream["stream"],
                        "rows": stream["rows"],
                        "numeric_valid_rows": stream["numeric_valid_rows"],
                        "review_samples": stream["review_samples"],
                        "average_hz": stream["average_hz"],
                        "max_position_step_mm": stream["position_step_mm"]["max"],
                        "max_rotation_step_deg": stream["rotation_step_deg"]["max"],
                    }
                )
            print(
                f"{report['session']}: {report['status']}; "
                + ", ".join(
                    f"{s['stream']} {s['review_samples']}/{s['rows']} review"
                    for s in report["streams"]
                )
            )
        write_json(output / "summary.json", [r[0] for r in results])
        write_csv(output / "summary.csv", aggregate, list(aggregate[0]))
        links = (
            "<ul>"
            + "".join(
                '<li><a href="'
                + html.escape(r[0]["session"], quote=True)
                + '/report.html">'
                + html.escape(r[0]["session"])
                + "</a> · "
                + r[0]["status"]
                + "</li>"
                for r in results
            )
            + "</ul>"
        )
        (output / "index.html").write_text(
            document(
                "PICO 原始录制质量检测",
                links
                + "<p>数据来源为 PICO，不包含 NOKOV 精度评价。ok 表示本次完整性与阈值筛查未检出异常。</p>",
            ),
            encoding="utf-8",
        )
        return int(any(r[0]["status"] != "ok" for r in results))
    except (OSError, ValueError, csv.Error) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
