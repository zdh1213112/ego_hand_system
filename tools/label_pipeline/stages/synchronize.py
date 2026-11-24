# Source nokov_ego_hand_validation/tools/synchronize_ego_imu_nokov.py, SHA-256 47f6d12624d9e08266836d50e1cd29dae6b10c1c63f6fe3f9496280cbbc16b3d
#!/usr/bin/env python3
"""Estimate EGO-to-NOKOV time offset from IMU and head rigid-body rotation.

The first-stage mapping fixes clock scale a=1 and estimates b in:

    nokov_relative_s = ego_relative_s + b

EGO angular speed is read from /robot0/sensor/imu or an Orbbec EGO4 IMU CSV.
NOKOV angular speed is
computed from consecutive rigid-body quaternions. Their norms are invariant to
the unknown fixed rotation between the raw IMU and NOKOV rigid-body frames.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    ego_source = parser.add_mutually_exclusive_group(required=True)
    ego_source.add_argument("--ego-mcap", type=Path)
    ego_source.add_argument(
        "--ego-imu-csv", type=Path,
        help="Orbbec EGO4 CSV with timestamp_us,x,y,z,type columns",
    )
    parser.add_argument("--nokov-csv", type=Path, required=True)
    parser.add_argument("--rigid-body", default="head_rigidbody")
    parser.add_argument("--imu-topic", default="/robot0/sensor/imu")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resample-hz", type=float, default=100.0)
    parser.add_argument("--smooth-ms", type=float, default=80.0)
    parser.add_argument("--max-offset-s", type=float, default=30.0)
    parser.add_argument("--min-overlap-s", type=float, default=8.0)
    parser.add_argument("--max-angular-speed-rad-s", type=float, default=15.0)
    parser.add_argument(
        "--max-interpolation-gap-s",
        type=float,
        default=0.05,
        help=(
            "maximum NOKOV pose bracket allowed when interpolating its 90 Hz "
            "rigid pose onto exact EGO IMU timestamps"
        ),
    )
    parser.add_argument(
        "--nokov-time-field", default="receive_perf_ns",
        choices=("receive_perf_ns", "receive_unix_ns", "device_timestamp_raw"),
    )
    parser.add_argument(
        "--nokov-time-scale", type=float, default=None,
        help="seconds per NOKOV timestamp unit; defaults to 1e-9 for receive_*_ns",
    )
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def scalar_timestamp_ns(value: Any, fallback: int) -> int:
    if value is None:
        return fallback
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, float):
        return int(value)
    if hasattr(value, "seconds") and hasattr(value, "nanos"):
        return int(value.seconds) * 1_000_000_000 + int(value.nanos)
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def read_ego_gyro(path: Path, topic: str) -> tuple[np.ndarray, np.ndarray]:
    try:
        from mcap.reader import make_reader
        from mcap_protobuf.decoder import DecoderFactory
    except ImportError as exc:
        raise RuntimeError(
            "missing MCAP packages; install tools/requirements-sync.txt"
        ) from exc

    timestamps: list[int] = []
    gyro: list[tuple[float, float, float]] = []
    with path.open("rb") as stream:
        reader = make_reader(stream, decoder_factories=[DecoderFactory()])
        for _schema, _channel, message, decoded in reader.iter_decoded_messages(
            topics=[topic]
        ):
            header = getattr(decoded, "header", None)
            stamp = scalar_timestamp_ns(
                getattr(header, "timestamp", None), int(message.log_time)
            )
            angular = getattr(decoded, "angular_velocity", None)
            if angular is None:
                continue
            values = (float(angular.x), float(angular.y), float(angular.z))
            if not all(math.isfinite(value) for value in values):
                continue
            timestamps.append(stamp)
            gyro.append(values)
    if len(timestamps) < 20:
        raise RuntimeError(f"only {len(timestamps)} valid IMU rows found on {topic}")

    ts = np.asarray(timestamps, dtype=np.int64)
    values = np.asarray(gyro, dtype=np.float64)
    order = np.argsort(ts, kind="stable")
    ts, values = ts[order], values[order]
    keep = np.r_[True, np.diff(ts) > 0]
    return ts[keep], values[keep]


def read_ego_gyro_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read Orbbec EGO4 gyro samples and convert Unix microseconds to ns."""
    samples: list[tuple[int, tuple[float, float, float]]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"timestamp_us", "x", "y", "z", "type"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise RuntimeError(f"Orbbec IMU CSV is missing fields: {sorted(missing)}")
        for row in reader:
            if row["type"].strip().lower() != "gyro":
                continue
            try:
                timestamp_ns = int(row["timestamp_us"]) * 1000
                values = tuple(float(row[axis]) for axis in ("x", "y", "z"))
            except (TypeError, ValueError):
                continue
            if timestamp_ns > 0 and all(math.isfinite(value) for value in values):
                samples.append((timestamp_ns, values))
    if len(samples) < 20:
        raise RuntimeError(f"only {len(samples)} valid gyro rows found in {path}")
    samples.sort(key=lambda item: item[0])
    timestamps = np.asarray([item[0] for item in samples], dtype=np.int64)
    gyro = np.asarray([item[1] for item in samples], dtype=np.float64)
    keep = np.r_[True, np.diff(timestamps) > 0]
    return timestamps[keep], gyro[keep]


def parse_int(row: dict[str, str], key: str, default: int = 0) -> int:
    value = row.get(key, "")
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError:
        return int(float(value))


def read_nokov_quaternions(
    path: Path, rigid_body: str, time_field: str
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    rows: list[tuple[int, np.ndarray, int]] = []
    recording_timestamps: list[int] = []
    seen_names: set[str] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {time_field, "rigid_body_name", "qx", "qy", "qz", "qw"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise RuntimeError(f"NOKOV CSV is missing fields: {sorted(missing)}")
        for row in reader:
            name = row.get("rigid_body_name", "")
            seen_names.add(name)
            if name != rigid_body:
                continue
            recording_stamp = parse_int(row, time_field)
            if recording_stamp:
                recording_timestamps.append(recording_stamp)
            if parse_int(row, "valid_numeric", 1) == 0:
                continue
            try:
                stamp = parse_int(row, time_field)
                quat = np.asarray(
                    [float(row[key]) for key in ("qx", "qy", "qz", "qw")],
                    dtype=np.float64,
                )
                params = parse_int(row, "params", 1)
            except (TypeError, ValueError):
                continue
            norm = float(np.linalg.norm(quat))
            if (
                stamp
                and np.isfinite(quat).all()
                and np.max(np.abs(quat)) <= 2.0
                and 0.5 < norm < 1.5
            ):
                rows.append((stamp, quat / norm, params))
    if len(rows) < 10:
        raise RuntimeError(
            f"only {len(rows)} valid rows for rigid body {rigid_body!r}; "
            f"available names={sorted(seen_names)!r}"
        )

    rows.sort(key=lambda item: item[0])
    has_tracking_bits = any(params & 1 for _, _, params in rows)
    if has_tracking_bits:
        tracked = [item for item in rows if item[2] & 1]
        if len(tracked) >= 10:
            rows = tracked
    timestamps = np.asarray([item[0] for item in rows], dtype=np.int64)
    quaternions = np.asarray([item[1] for item in rows], dtype=np.float64)
    keep = np.r_[True, np.diff(timestamps) > 0]
    return timestamps[keep], quaternions[keep], {
        "available_rigid_body_names": sorted(seen_names),
        "tracking_valid_bit_used": bool(has_tracking_bits),
        "recording_origin_timestamp_raw": int(min(recording_timestamps)),
    }


def read_nokov_poses(
    path: Path, rigid_body: str, time_field: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read valid rigid poses without trusting NOKOV's validity bit alone."""
    rows: list[tuple[int, int, np.ndarray, np.ndarray]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {
            "frame_no", time_field, "rigid_body_name", "x_mm", "y_mm", "z_mm",
            "qx", "qy", "qz", "qw",
        }
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise RuntimeError(f"NOKOV CSV is missing pose fields: {sorted(missing)}")
        for row in reader:
            if row.get("rigid_body_name", "") != rigid_body:
                continue
            try:
                stamp = parse_int(row, time_field)
                frame = parse_int(row, "frame_no", -1)
                position = np.asarray(
                    [float(row[key]) for key in ("x_mm", "y_mm", "z_mm")],
                    dtype=np.float64,
                )
                quaternion = np.asarray(
                    [float(row[key]) for key in ("qx", "qy", "qz", "qw")],
                    dtype=np.float64,
                )
            except (TypeError, ValueError):
                continue
            quaternion_norm = float(np.linalg.norm(quaternion))
            if (
                stamp
                and np.isfinite(position).all()
                and np.isfinite(quaternion).all()
                and np.max(np.abs(position)) < 1_000_000.0
                and np.max(np.abs(quaternion)) <= 2.0
                and 0.5 < quaternion_norm < 1.5
            ):
                rows.append((stamp, frame, position, quaternion / quaternion_norm))
    if len(rows) < 10:
        raise RuntimeError(f"only {len(rows)} valid poses for rigid body {rigid_body!r}")
    rows.sort(key=lambda item: item[0])
    timestamps = np.asarray([item[0] for item in rows], dtype=np.int64)
    frames = np.asarray([item[1] for item in rows], dtype=np.int64)
    positions = np.asarray([item[2] for item in rows], dtype=np.float64)
    quaternions = np.asarray([item[3] for item in rows], dtype=np.float64)
    keep = np.r_[True, np.diff(timestamps) > 0]
    return timestamps[keep], frames[keep], positions[keep], quaternions[keep]


def slerp_quaternions(
    first: np.ndarray, second: np.ndarray, alpha: np.ndarray
) -> np.ndarray:
    """Shortest-path SLERP for batches of xyzw quaternions."""
    q0 = np.asarray(first, dtype=np.float64)
    q1 = np.asarray(second, dtype=np.float64).copy()
    fraction = np.asarray(alpha, dtype=np.float64).reshape(-1, 1)
    dots = np.sum(q0 * q1, axis=1)
    negative = dots < 0.0
    q1[negative] *= -1.0
    dots = np.clip(np.abs(dots), 0.0, 1.0)
    result = np.empty_like(q0)
    linear = dots > 0.9995
    if np.any(linear):
        result[linear] = (
            (1.0 - fraction[linear]) * q0[linear]
            + fraction[linear] * q1[linear]
        )
    curved = ~linear
    if np.any(curved):
        theta = np.arccos(dots[curved])
        sine = np.sin(theta)
        result[curved] = (
            np.sin((1.0 - fraction[curved]) * theta[:, None]) / sine[:, None]
            * q0[curved]
            + np.sin(fraction[curved] * theta[:, None]) / sine[:, None]
            * q1[curved]
        )
    result /= np.linalg.norm(result, axis=1, keepdims=True)
    return result


def interpolate_rigid_poses(
    pose_time_s: np.ndarray,
    frames: np.ndarray,
    positions_mm: np.ndarray,
    quaternions_xyzw: np.ndarray,
    target_time_s: np.ndarray,
    max_gap_s: float,
) -> dict[str, np.ndarray]:
    """Interpolate NOKOV poses at arbitrary mapped EGO timestamps."""
    count = len(target_time_s)
    right = np.searchsorted(pose_time_s, target_time_s, side="right")
    exact_last = target_time_s == pose_time_s[-1]
    right[exact_last] = len(pose_time_s) - 1
    left = right - 1
    in_range = (left >= 0) & (right < len(pose_time_s))
    safe_left = np.clip(left, 0, len(pose_time_s) - 2)
    safe_right = np.clip(right, 1, len(pose_time_s) - 1)
    bracket_s = pose_time_s[safe_right] - pose_time_s[safe_left]
    alpha = np.divide(
        target_time_s - pose_time_s[safe_left],
        bracket_s,
        out=np.full(count, np.nan, dtype=np.float64),
        where=bracket_s > 0,
    )
    valid = (
        in_range
        & (bracket_s > 0)
        & (bracket_s <= max_gap_s)
        & (alpha >= -1e-9)
        & (alpha <= 1.0 + 1e-9)
    )
    position = np.full((count, 3), np.nan, dtype=np.float64)
    quaternion = np.full((count, 4), np.nan, dtype=np.float64)
    if np.any(valid):
        weights = alpha[valid, None]
        position[valid] = (
            (1.0 - weights) * positions_mm[safe_left[valid]]
            + weights * positions_mm[safe_right[valid]]
        )
        quaternion[valid] = slerp_quaternions(
            quaternions_xyzw[safe_left[valid]],
            quaternions_xyzw[safe_right[valid]],
            alpha[valid],
        )
    return {
        "valid": valid,
        "in_range": in_range,
        "left_frame": frames[safe_left],
        "right_frame": frames[safe_right],
        "left_time_s": pose_time_s[safe_left],
        "right_time_s": pose_time_s[safe_right],
        "bracket_s": bracket_s,
        "alpha": alpha,
        "position_mm": position,
        "quaternion_xyzw": quaternion,
    }


def quaternion_angular_speed(
    timestamps_raw: np.ndarray,
    quaternions: np.ndarray,
    seconds_per_unit: float,
    max_speed: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    time_s = (timestamps_raw - timestamps_raw[0]).astype(np.float64) * seconds_per_unit
    dt = np.diff(time_s)
    positive = dt > 0
    if not np.any(positive):
        raise RuntimeError("NOKOV timestamps are not increasing")
    median_dt = float(np.median(dt[positive]))
    total_duration = float(time_s[-1] - time_s[0])
    mean_pose_rate = (
        float(len(time_s) - 1) / total_duration if total_duration > 0 else float("nan")
    )
    dots = np.abs(np.sum(quaternions[:-1] * quaternions[1:], axis=1))
    dots = np.clip(dots, 0.0, 1.0)
    angle = 2.0 * np.arccos(dots)
    speed = np.divide(angle, dt, out=np.full_like(angle, np.nan), where=dt > 0)
    midpoint = 0.5 * (time_s[:-1] + time_s[1:])
    keep = (
        np.isfinite(speed)
        & (dt > 0)
        & (dt <= max(0.1, 5.0 * median_dt))
        & (speed <= max_speed)
    )
    if int(np.count_nonzero(keep)) < 10:
        raise RuntimeError("too few valid NOKOV angular-speed samples")
    return midpoint[keep], speed[keep], {
        "pose_rate_hz": mean_pose_rate,
        "median_timestamp_step_s": median_dt,
        "filtered_angular_speed_rows": int(np.count_nonzero(~keep)),
    }


def uniform_signal(
    time_s: np.ndarray, values: np.ndarray, rate_hz: float, smooth_ms: float
) -> tuple[np.ndarray, np.ndarray]:
    duration = float(time_s[-1])
    if duration <= 0:
        raise RuntimeError("signal duration is zero")
    grid = np.arange(0.0, duration, 1.0 / rate_hz, dtype=np.float64)
    signal = np.interp(grid, time_s, values)
    window = max(1, int(round(smooth_ms * 1e-3 * rate_hz)))
    if window > 1:
        if window % 2 == 0:
            window += 1
        half = window // 2
        padded = np.pad(signal, (half, half), mode="edge")
        signal = np.convolve(padded, np.ones(window) / window, mode="valid")
    return grid, signal


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3:
        return float("nan")
    aa = a - float(np.mean(a))
    bb = b - float(np.mean(b))
    denom = float(np.linalg.norm(aa) * np.linalg.norm(bb))
    return float(np.dot(aa, bb) / denom) if denom > 1e-12 else float("nan")


def estimate_offset(
    ego: np.ndarray,
    nokov: np.ndarray,
    rate_hz: float,
    max_offset_s: float,
    min_overlap_s: float,
) -> tuple[float, float, float, float, np.ndarray, np.ndarray]:
    max_shift = int(round(max_offset_s * rate_hz))
    minimum = int(round(min_overlap_s * rate_hz))
    shifts: list[int] = []
    correlations: list[float] = []
    for shift in range(-max_shift, max_shift + 1):
        ego_start = max(0, -shift)
        nokov_start = max(0, shift)
        count = min(len(ego) - ego_start, len(nokov) - nokov_start)
        if count < minimum:
            continue
        corr = pearson(
            ego[ego_start:ego_start + count],
            nokov[nokov_start:nokov_start + count],
        )
        if math.isfinite(corr):
            shifts.append(shift)
            correlations.append(corr)
    if not correlations:
        raise RuntimeError("no candidate offset has enough overlapping data")

    shifts_array = np.asarray(shifts, dtype=np.int64)
    corr_array = np.asarray(correlations, dtype=np.float64)
    best_index = int(np.argmax(corr_array))
    best_shift = float(shifts_array[best_index])
    if 0 < best_index < len(corr_array) - 1:
        left, center, right = corr_array[best_index - 1:best_index + 2]
        curvature = left - 2.0 * center + right
        if abs(curvature) > 1e-12:
            delta = 0.5 * (left - right) / curvature
            if abs(delta) <= 1.0:
                best_shift += float(delta)
    offset_s = best_shift / rate_hz

    exclusion = max(1, int(round(1.0 * rate_hz)))
    far = np.abs(shifts_array - shifts_array[best_index]) > exclusion
    second = float(np.max(corr_array[far])) if np.any(far) else float("nan")
    margin = float(corr_array[best_index] - second) if math.isfinite(second) else float("nan")
    return (
        offset_s,
        float(corr_array[best_index]),
        second,
        margin,
        shifts_array / rate_hz,
        corr_array,
    )


def write_aligned_csv(
    path: Path,
    ego_grid: np.ndarray,
    ego_signal: np.ndarray,
    nokov_grid: np.ndarray,
    nokov_signal: np.ndarray,
    offset_s: float,
    ego_origin_ns: int,
    nokov_origin_raw: int,
    nokov_scale: float,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    target_nokov_s = ego_grid + offset_s
    valid = (target_nokov_s >= nokov_grid[0]) & (target_nokov_s <= nokov_grid[-1])
    aligned_nokov = np.full_like(ego_signal, np.nan)
    aligned_nokov[valid] = np.interp(
        target_nokov_s[valid], nokov_grid, nokov_signal
    )
    with path.open("w", encoding="utf-8", newline="") as stream:
        fields = (
            "ego_relative_s", "ego_timestamp_ns", "ego_gyro_norm_rad_s",
            "nokov_target_relative_s", "nokov_target_timestamp_raw",
            "nokov_angular_speed_rad_s", "valid_overlap",
        )
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index in range(len(ego_grid)):
            writer.writerow({
                "ego_relative_s": f"{ego_grid[index]:.9f}",
                "ego_timestamp_ns": ego_origin_ns + int(round(ego_grid[index] * 1e9)),
                "ego_gyro_norm_rad_s": f"{ego_signal[index]:.9f}",
                "nokov_target_relative_s": f"{target_nokov_s[index]:.9f}",
                "nokov_target_timestamp_raw": (
                    f"{nokov_origin_raw + target_nokov_s[index] / nokov_scale:.3f}"
                ),
                "nokov_angular_speed_rad_s": (
                    f"{aligned_nokov[index]:.9f}" if valid[index] else ""
                ),
                "valid_overlap": int(valid[index]),
            })
    overlap_s = float(np.count_nonzero(valid)) * float(np.median(np.diff(ego_grid)))
    return overlap_s, target_nokov_s, aligned_nokov, valid


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.copy()
    if window % 2 == 0:
        window += 1
    half = window // 2
    padded = np.pad(values, (half, half), mode="edge")
    return np.convolve(padded, np.ones(window) / window, mode="valid")


def write_pose_at_ego_timestamps(
    path: Path,
    ego_ts: np.ndarray,
    ego_gyro: np.ndarray,
    nokov_pose_ts: np.ndarray,
    nokov_frames: np.ndarray,
    nokov_positions_mm: np.ndarray,
    nokov_pose_quaternions: np.ndarray,
    nokov_speed_time_s: np.ndarray,
    nokov_speed: np.ndarray,
    ego_origin_ns: int,
    nokov_origin_raw: int,
    nokov_scale: float,
    clock_scale: float,
    offset_s: float,
    max_gap_s: float,
    smooth_ms: float,
) -> dict[str, Any]:
    """Write one interpolated NOKOV rigid pose for each actual EGO IMU row."""
    ego_relative_s = (ego_ts - ego_origin_ns).astype(np.float64) * 1e-9
    target_s = clock_scale * ego_relative_s + offset_s
    pose_time_s = (
        nokov_pose_ts - nokov_origin_raw
    ).astype(np.float64) * nokov_scale
    interpolation = interpolate_rigid_poses(
        pose_time_s,
        nokov_frames,
        nokov_positions_mm,
        nokov_pose_quaternions,
        target_s,
        max_gap_s,
    )
    valid = interpolation["valid"]
    in_speed_range = (
        (target_s >= nokov_speed_time_s[0])
        & (target_s <= nokov_speed_time_s[-1])
        & valid
    )
    aligned_speed = np.full(len(ego_ts), np.nan, dtype=np.float64)
    aligned_speed[in_speed_range] = np.interp(
        target_s[in_speed_range], nokov_speed_time_s, nokov_speed
    )
    ego_speed = np.linalg.norm(ego_gyro, axis=1)
    rate_hz = float(1.0 / np.median(np.diff(ego_relative_s)))
    smoothing_window = max(1, int(round(smooth_ms * 1e-3 * rate_hz)))
    if int(np.count_nonzero(in_speed_range)) >= 3:
        correlation_raw = pearson(
            ego_speed[in_speed_range], aligned_speed[in_speed_range]
        )
        correlation_smoothed = pearson(
            moving_average(ego_speed[in_speed_range], smoothing_window),
            moving_average(aligned_speed[in_speed_range], smoothing_window),
        )
    else:
        correlation_raw = float("nan")
        correlation_smoothed = float("nan")

    fields = (
        "ego_timestamp_ns", "ego_relative_s", "ego_gyro_x_rad_s",
        "ego_gyro_y_rad_s", "ego_gyro_z_rad_s", "ego_gyro_norm_rad_s",
        "nokov_target_timestamp_raw", "nokov_target_relative_s",
        "nokov_left_frame", "nokov_right_frame", "interpolation_alpha",
        "bracket_gap_ms", "x_mm", "y_mm", "z_mm", "qx", "qy", "qz", "qw",
        "valid_interpolation",
    )
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        positions = interpolation["position_mm"]
        quaternions = interpolation["quaternion_xyzw"]
        for index in range(len(ego_ts)):
            pose_values = (
                [f"{value:.9f}" for value in positions[index]]
                + [f"{value:.12f}" for value in quaternions[index]]
                if valid[index]
                else [""] * 7
            )
            writer.writerow({
                "ego_timestamp_ns": int(ego_ts[index]),
                "ego_relative_s": f"{ego_relative_s[index]:.9f}",
                "ego_gyro_x_rad_s": f"{ego_gyro[index, 0]:.9f}",
                "ego_gyro_y_rad_s": f"{ego_gyro[index, 1]:.9f}",
                "ego_gyro_z_rad_s": f"{ego_gyro[index, 2]:.9f}",
                "ego_gyro_norm_rad_s": f"{ego_speed[index]:.9f}",
                "nokov_target_timestamp_raw": (
                    f"{nokov_origin_raw + target_s[index] / nokov_scale:.6f}"
                ),
                "nokov_target_relative_s": f"{target_s[index]:.9f}",
                "nokov_left_frame": int(interpolation["left_frame"][index]),
                "nokov_right_frame": int(interpolation["right_frame"][index]),
                "interpolation_alpha": (
                    f"{interpolation['alpha'][index]:.9f}"
                    if interpolation["in_range"][index] else ""
                ),
                "bracket_gap_ms": (
                    f"{interpolation['bracket_s'][index] * 1000.0:.6f}"
                    if interpolation["in_range"][index] else ""
                ),
                "x_mm": pose_values[0],
                "y_mm": pose_values[1],
                "z_mm": pose_values[2],
                "qx": pose_values[3],
                "qy": pose_values[4],
                "qz": pose_values[5],
                "qw": pose_values[6],
                "valid_interpolation": int(valid[index]),
            })

    overlap_count = int(np.count_nonzero(interpolation["in_range"]))
    valid_count = int(np.count_nonzero(valid))
    gaps_ms = interpolation["bracket_s"][valid] * 1000.0
    quaternion_norms = np.linalg.norm(
        interpolation["quaternion_xyzw"][valid], axis=1
    )
    coverage_ok = bool(overlap_count and valid_count / overlap_count >= 0.98)
    correlation_ok = bool(
        math.isfinite(correlation_smoothed) and correlation_smoothed >= 0.75
    )
    return {
        "schema": "ego_nokov_timestamp_interpolation_v1",
        "status": "ok" if coverage_ok and correlation_ok else "needs_review",
        "time_mapping": {
            "equation": "nokov_relative_s = a * ego_relative_s + b",
            "a": clock_scale,
            "b_s": offset_s,
        },
        "sampling": {
            "ego_imu_observed_rate_hz": rate_hz,
            "nokov_pose_observed_rate_hz": (
                float(len(pose_time_s) - 1) / float(pose_time_s[-1] - pose_time_s[0])
            ),
            "method": {
                "position": "linear interpolation",
                "orientation": "shortest-path quaternion SLERP",
            },
            "note": (
                "output rows follow exact EGO IMU timestamps; interpolation does not "
                "increase the physical information bandwidth of the 90 Hz NOKOV data"
            ),
        },
        "coverage": {
            "ego_imu_rows": int(len(ego_ts)),
            "rows_inside_nokov_time_range": overlap_count,
            "valid_interpolated_rows": valid_count,
            "rows_rejected_for_large_pose_gap": int(
                np.count_nonzero(interpolation["in_range"] & ~valid)
            ),
            "overlap_valid_ratio": valid_count / overlap_count if overlap_count else 0.0,
            "whole_ego_recording_valid_ratio": valid_count / len(ego_ts),
        },
        "quality": {
            "maximum_allowed_pose_gap_ms": max_gap_s * 1000.0,
            "pose_bracket_gap_median_ms": float(np.median(gaps_ms)),
            "pose_bracket_gap_p95_ms": float(np.percentile(gaps_ms, 95)),
            "pose_bracket_gap_max_ms": float(np.max(gaps_ms)),
            "quaternion_norm_max_abs_error": float(
                np.max(np.abs(quaternion_norms - 1.0))
            ),
            "angular_speed_correlation_at_exact_ego_timestamps_raw": correlation_raw,
            "angular_speed_correlation_at_exact_ego_timestamps_smoothed": correlation_smoothed,
            "smoothing_ms": smooth_ms,
        },
        "files": {"interpolated_pose_csv": path.name},
    }


def save_plot(
    path: Path,
    ego_grid: np.ndarray,
    ego_signal: np.ndarray,
    aligned_nokov: np.ndarray,
    valid: np.ndarray,
    offsets: np.ndarray,
    correlations: np.ndarray,
    best_offset: float,
) -> str | None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return "matplotlib is unavailable; plot was not written"
    figure, axes = plt.subplots(2, 1, figsize=(12, 7), constrained_layout=True)
    axes[0].plot(ego_grid, ego_signal, label="EGO IMU |gyro|", linewidth=1.0)
    axes[0].plot(
        ego_grid[valid], aligned_nokov[valid],
        label="NOKOV rigid |omega| aligned", linewidth=1.0,
    )
    axes[0].set_xlabel("EGO relative time (s)")
    axes[0].set_ylabel("angular speed (rad/s)")
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    axes[1].plot(offsets, correlations, linewidth=1.0)
    axes[1].axvline(best_offset, color="red", linestyle="--", label=f"b={best_offset:.4f}s")
    axes[1].set_xlabel("candidate b: NOKOV relative = EGO relative + b (s)")
    axes[1].set_ylabel("Pearson correlation")
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return None


def main() -> int:
    args = parse_args()
    if (
        args.resample_hz <= 0
        or args.max_offset_s <= 0
        or args.min_overlap_s <= 0
        or args.max_interpolation_gap_s <= 0
    ):
        print("[FAIL] rates and durations must be positive", file=sys.stderr)
        return 2
    ego_path = (args.ego_mcap or args.ego_imu_csv).resolve()
    nokov_path = args.nokov_csv.resolve()
    if not ego_path.is_file() or not nokov_path.is_file():
        print(f"[FAIL] missing input: EGO={ego_path} NOKOV={nokov_path}", file=sys.stderr)
        return 2
    nokov_scale = args.nokov_time_scale
    if nokov_scale is None:
        if args.nokov_time_field in ("receive_perf_ns", "receive_unix_ns"):
            nokov_scale = 1e-9
        else:
            print(
                "[FAIL] --nokov-time-scale is required with device_timestamp_raw",
                file=sys.stderr,
            )
            return 2

    try:
        ego_ts, ego_gyro = (
            read_ego_gyro_csv(ego_path) if args.ego_imu_csv
            else read_ego_gyro(ego_path, args.imu_topic)
        )
        nokov_ts, nokov_quat, nokov_info = read_nokov_quaternions(
            nokov_path, args.rigid_body, args.nokov_time_field
        )
        (
            nokov_pose_ts,
            nokov_frames,
            nokov_positions_mm,
            nokov_pose_quaternions,
        ) = read_nokov_poses(nokov_path, args.rigid_body, args.nokov_time_field)
        ego_time = (ego_ts - ego_ts[0]).astype(np.float64) * 1e-9
        ego_speed = np.linalg.norm(ego_gyro, axis=1)
        keep_ego = np.isfinite(ego_speed) & (ego_speed <= args.max_angular_speed_rad_s)
        ego_time, ego_speed = ego_time[keep_ego], ego_speed[keep_ego]
        nokov_time, nokov_speed, quaternion_info = quaternion_angular_speed(
            nokov_ts, nokov_quat, nokov_scale, args.max_angular_speed_rad_s
        )
        nokov_origin_raw = int(nokov_info["recording_origin_timestamp_raw"])
        nokov_time += float(int(nokov_ts[0]) - nokov_origin_raw) * nokov_scale
        ego_grid, ego_uniform = uniform_signal(
            ego_time, ego_speed, args.resample_hz, args.smooth_ms
        )
        nokov_grid, nokov_uniform = uniform_signal(
            nokov_time, nokov_speed, args.resample_hz, args.smooth_ms
        )
        offset_s, corr, second_corr, margin, offsets, correlations = estimate_offset(
            ego_uniform, nokov_uniform, args.resample_hz,
            args.max_offset_s, args.min_overlap_s,
        )
    except Exception as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 2

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    aligned_csv = output / "imu_nokov_aligned_signals.csv"
    overlap_s, _target_nokov, aligned_nokov, valid = write_aligned_csv(
        aligned_csv, ego_grid, ego_uniform, nokov_grid, nokov_uniform,
        offset_s, int(ego_ts[0]), nokov_origin_raw, nokov_scale,
    )
    aligned_corr = pearson(ego_uniform[valid], aligned_nokov[valid])
    pose_csv = output / "nokov_pose_at_ego_imu_timestamps.csv"
    interpolation_result = write_pose_at_ego_timestamps(
        pose_csv,
        ego_ts,
        ego_gyro,
        nokov_pose_ts,
        nokov_frames,
        nokov_positions_mm,
        nokov_pose_quaternions,
        nokov_time,
        nokov_speed,
        int(ego_ts[0]),
        nokov_origin_raw,
        nokov_scale,
        1.0,
        offset_s,
        args.max_interpolation_gap_s,
        args.smooth_ms,
    )
    interpolation_json = output / "ego_nokov_interpolation_validation.json"
    interpolation_json.write_text(
        json.dumps(interpolation_result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    ego_activity = float(np.percentile(ego_uniform, 95) - np.percentile(ego_uniform, 20))
    nokov_activity = float(np.percentile(nokov_uniform, 95) - np.percentile(nokov_uniform, 20))
    if aligned_corr >= 0.75 and (not math.isfinite(margin) or margin >= 0.05):
        confidence = "strong"
    elif aligned_corr >= 0.55 and (not math.isfinite(margin) or margin >= 0.02):
        confidence = "usable"
    else:
        confidence = "weak"
    warnings: list[str] = []
    if confidence == "weak":
        warnings.append("sync peak is weak or ambiguous; repeat a more distinctive head-motion sequence")
    if ego_activity < 0.2 or nokov_activity < 0.2:
        warnings.append("angular activity is low; add clear yaw, pitch and roll pulses")
    if abs(offset_s) >= args.max_offset_s - 1.0 / args.resample_hz:
        warnings.append("best offset reached the search boundary; increase --max-offset-s")
    if min(float(ego_grid[-1]), float(nokov_grid[-1])) > 120.0:
        warnings.append("long recording detected; first-stage a=1 ignores clock drift")
    if interpolation_result["status"] != "ok":
        warnings.append(
            "less than 98% of overlapping EGO timestamps received a safe NOKOV pose "
            "interpolation; inspect ego_nokov_interpolation_validation.json"
        )

    result = {
        "schema": "ego_nokov_imu_sync_v1",
        "status": "ok" if confidence in ("strong", "usable") else "needs_review",
        "confidence": confidence,
        "method": "angular_speed_norm_cross_correlation",
        "time_mapping": {
            "equation": "nokov_relative_s = a * ego_relative_s + b",
            "a": 1.0,
            "b_s": offset_s,
            "ego_origin_timestamp_ns": int(ego_ts[0]),
            "nokov_origin_field": args.nokov_time_field,
            "nokov_origin_timestamp_raw": nokov_origin_raw,
            "nokov_seconds_per_timestamp_unit": nokov_scale,
            "ego_to_nokov_raw_equation": (
                "nokov_raw = nokov_origin_raw + "
                "(((ego_timestamp_ns-ego_origin_timestamp_ns)*1e-9)+b_s) / "
                "nokov_seconds_per_timestamp_unit"
            ),
        },
        "quality": {
            "peak_correlation": aligned_corr,
            "second_peak_correlation_outside_1s": second_corr,
            "peak_margin": margin,
            "overlap_s": overlap_s,
            "resample_hz": args.resample_hz,
            "smooth_ms": args.smooth_ms,
            "offset_grid_resolution_s": 1.0 / args.resample_hz,
            "ego_activity_p95_minus_p20_rad_s": ego_activity,
            "nokov_activity_p95_minus_p20_rad_s": nokov_activity,
        },
        "inputs": {
            "ego_mcap": str(ego_path) if args.ego_mcap else None,
            "ego_imu_csv": str(ego_path) if args.ego_imu_csv else None,
            "ego_imu_topic": args.imu_topic if args.ego_mcap else None,
            "ego_imu_rows": int(len(ego_ts)),
            "ego_duration_s": float(ego_grid[-1]),
            "nokov_csv": str(nokov_path),
            "nokov_rigid_body": args.rigid_body,
            "nokov_pose_rows": int(len(nokov_ts)),
            "nokov_duration_s": float(nokov_grid[-1]),
            **nokov_info,
            **quaternion_info,
        },
        "coordinate_note": (
            "angular-speed norm is rotation invariant; raw IMU axis conversion is not "
            "required for this offset estimate"
        ),
        "clock_drift_note": (
            "this first-stage run fixes a=1; estimate a from a longer recording with "
            "distinctive motions near the start, middle and end"
        ),
        "files": {
            "aligned_signals": aligned_csv.name,
            "pose_at_ego_imu_timestamps": pose_csv.name,
            "interpolation_validation": interpolation_json.name,
            "diagnostic_plot": "imu_nokov_sync.png" if not args.no_plot else None,
        },
        "warnings": warnings,
    }
    json_path = output / "imu_nokov_sync.json"
    json_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    plot_warning = None
    if not args.no_plot:
        plot_warning = save_plot(
            output / "imu_nokov_sync.png", ego_grid, ego_uniform,
            aligned_nokov, valid, offsets, correlations, offset_s,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if plot_warning:
        print(f"[WARN] {plot_warning}")
    print(f"Sync result: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
