#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import time

import cv2
import numpy as np
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert an Orbbec EGO stereo+IMU recording to Basalt's EuRoC input format."
    )
    parser.add_argument("--session", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--image-scale", type=float, default=0.5)
    parser.add_argument("--max-delta-us", type=int, default=1500)
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument("--png-compression", type=int, default=1)
    parser.add_argument(
        "--no-precalibrate-imu", action="store_true",
        help="keep raw IMU values; intended only for calibration experiments",
    )
    return parser.parse_args()


def unique_file(directory: Path, suffix: str) -> Path:
    matches = sorted(directory.glob(f"*{suffix}"))
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one '*{suffix}' in {directory}, found {len(matches)}"
        )
    return matches[0]


def read_timestamps(path: Path) -> list[int]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        timestamps = [int(row["timestamp_us"]) for row in csv.DictReader(stream)]
    if not timestamps:
        raise RuntimeError(f"empty timestamp file: {path}")
    if any(current <= previous for previous, current in zip(timestamps, timestamps[1:])):
        raise RuntimeError(f"timestamps are not strictly increasing: {path}")
    return timestamps


def pair_timestamps(
    left: list[int], right: list[int], maximum_delta_us: int
) -> list[tuple[int, int]]:
    if maximum_delta_us < 0:
        raise ValueError("maximum_delta_us must be non-negative")
    pairs: list[tuple[int, int]] = []
    left_index = 0
    right_index = 0
    while left_index < len(left) and right_index < len(right):
        delta = right[right_index] - left[left_index]
        if abs(delta) <= maximum_delta_us:
            pairs.append((left_index, right_index))
            left_index += 1
            right_index += 1
        elif delta > 0:
            left_index += 1
        else:
            right_index += 1
    return pairs


def make_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.asarray(rotation, dtype=np.float64)
    transform[:3, 3] = np.asarray(translation, dtype=np.float64)
    return transform


def invert_transform(transform: np.ndarray) -> np.ndarray:
    transform = np.asarray(transform, dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError("transform must be (4, 4)")
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rotation.T
    result[:3, 3] = -(rotation.T @ translation)
    return result


def quaternion_xyzw_from_matrix(rotation: np.ndarray) -> np.ndarray:
    rotation = np.asarray(rotation, dtype=np.float64)
    if rotation.shape != (3, 3):
        raise ValueError("rotation must be (3, 3)")
    quaternion = np.empty(4, dtype=np.float64)
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quaternion[3] = 0.25 * scale
        quaternion[0] = (rotation[2, 1] - rotation[1, 2]) / scale
        quaternion[1] = (rotation[0, 2] - rotation[2, 0]) / scale
        quaternion[2] = (rotation[1, 0] - rotation[0, 1]) / scale
    else:
        index = int(np.argmax(np.diag(rotation)))
        if index == 0:
            scale = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
            quaternion[3] = (rotation[2, 1] - rotation[1, 2]) / scale
            quaternion[0] = 0.25 * scale
            quaternion[1] = (rotation[0, 1] + rotation[1, 0]) / scale
            quaternion[2] = (rotation[0, 2] + rotation[2, 0]) / scale
        elif index == 1:
            scale = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
            quaternion[3] = (rotation[0, 2] - rotation[2, 0]) / scale
            quaternion[0] = (rotation[0, 1] + rotation[1, 0]) / scale
            quaternion[1] = 0.25 * scale
            quaternion[2] = (rotation[1, 2] + rotation[2, 1]) / scale
        else:
            scale = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
            quaternion[3] = (rotation[1, 0] - rotation[0, 1]) / scale
            quaternion[0] = (rotation[0, 2] + rotation[2, 0]) / scale
            quaternion[1] = (rotation[1, 2] + rotation[2, 1]) / scale
            quaternion[2] = 0.25 * scale
    quaternion /= np.linalg.norm(quaternion)
    if quaternion[3] < 0.0:
        quaternion *= -1.0
    return quaternion


def basalt_pose(transform: np.ndarray) -> dict[str, float]:
    quaternion = quaternion_xyzw_from_matrix(transform[:3, :3])
    translation = transform[:3, 3]
    return {
        "px": float(translation[0]), "py": float(translation[1]),
        "pz": float(translation[2]), "qx": float(quaternion[0]),
        "qy": float(quaternion[1]), "qz": float(quaternion[2]),
        "qw": float(quaternion[3]),
    }


def identity_pose() -> dict[str, float]:
    return basalt_pose(np.eye(4, dtype=np.float64))


def flat_vignette() -> dict:
    return {"value0": 0, "value1": 10_000_000_000, "value2": [[1.0]] * 16}


def scaled_intrinsics(camera: dict, scale: float) -> dict:
    intrinsics = camera["intrinsics"]
    distortion = camera["distortion"]
    return {
        "camera_type": "kb4",
        "intrinsics": {
            "fx": float(intrinsics["fx"]) * scale,
            "fy": float(intrinsics["fy"]) * scale,
            "cx": float(intrinsics["cx"]) * scale,
            "cy": float(intrinsics["cy"]) * scale,
            "k1": float(distortion["k1"]),
            "k2": float(distortion["k2"]),
            "k3": float(distortion["k3"]),
            "k4": float(distortion["k4"]),
        },
    }


def validate_rotation(rotation: np.ndarray, label: str) -> None:
    if not np.isfinite(rotation).all():
        raise RuntimeError(f"non-finite rotation: {label}")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-4):
        raise RuntimeError(f"rotation is not orthonormal: {label}")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=2e-4):
        raise RuntimeError(f"rotation determinant is not +1: {label}")


def build_basalt_calibration(
    camera_yaml: dict, imu_yaml: dict, scale: float, precalibrated_imu: bool
) -> tuple[dict, dict]:
    cameras = camera_yaml["cameras"]
    if len(cameras) != 2 or any(camera["distortion_model"] != "KB" for camera in cameras):
        raise RuntimeError("expected two KB cameras")
    left, right = cameras
    original_resolution = (int(left["image_width"]), int(left["image_height"]))
    if original_resolution != (int(right["image_width"]), int(right["image_height"])):
        raise RuntimeError("left/right resolutions differ")
    output_resolution = (
        int(round(original_resolution[0] * scale)),
        int(round(original_resolution[1] * scale)),
    )

    camera_reference_transforms = []
    for index, camera in enumerate(cameras):
        rotation = np.asarray(camera["extrinsics"]["rotation"], dtype=np.float64)
        validate_rotation(rotation, f"camera {index}")
        translation_m = np.asarray(
            camera["extrinsics"]["translation"], dtype=np.float64
        ) * 1e-3
        camera_reference_transforms.append(make_transform(rotation, translation_m))
    t_c1_c0 = camera_reference_transforms[1] @ invert_transform(camera_reference_transforms[0])

    imu0 = imu_yaml["imu0"]
    t_c0_i = np.asarray(imu_yaml["cam0"]["T_cam_imu"], dtype=np.float64)
    if t_c0_i.shape != (4, 4):
        raise RuntimeError("T_cam_imu must be 4x4")
    t_c0_i = t_c0_i.copy()
    t_c0_i[:3, 3] *= 1e-3
    validate_rotation(t_c0_i[:3, :3], "T_cam_imu")
    t_i_c0 = invert_transform(t_c0_i)
    t_i_c1 = t_i_c0 @ invert_transform(t_c1_c0)

    baseline_m = float(np.linalg.norm((invert_transform(t_i_c0) @ t_i_c1)[:3, 3]))
    expected_baseline_m = float(np.linalg.norm(t_c1_c0[:3, 3]))
    if not np.isclose(baseline_m, expected_baseline_m, atol=1e-9):
        raise RuntimeError("camera/IMU transform composition changed the stereo baseline")

    zeros_accel = [0.0] * 9
    zeros_gyro = [0.0] * 12
    calibration = {
        "value0": {
            "T_imu_cam": [basalt_pose(t_i_c0), basalt_pose(t_i_c1)],
            "intrinsics": [scaled_intrinsics(left, scale), scaled_intrinsics(right, scale)],
            "resolution": [list(output_resolution), list(output_resolution)],
            "vignette": [flat_vignette(), flat_vignette()],
            "calib_accel_bias": zeros_accel,
            "calib_gyro_bias": zeros_gyro,
            "imu_update_rate": float(imu0["update_rate"]),
            "accel_noise_std": [float(imu0["accelerometer"]["noise_density"])] * 3,
            "gyro_noise_std": [float(imu0["gyroscope"]["noise_density"])] * 3,
            "accel_bias_std": [float(imu0["accelerometer"]["random_walk"])] * 3,
            "gyro_bias_std": [float(imu0["gyroscope"]["random_walk"])] * 3,
            "T_mocap_world": identity_pose(),
            "T_imu_marker": identity_pose(),
            "mocap_time_offset_ns": 0,
            "mocap_to_imu_offset_ns": 0,
            "cam_time_offset_ns": 0,
        }
    }
    metadata = {
        "original_resolution": list(original_resolution),
        "output_resolution": list(output_resolution),
        "stereo_baseline_m": expected_baseline_m,
        "T_cam0_imu_input_translation_unit": "millimetres converted to metres",
        "T_imu_cam0": t_i_c0.tolist(),
        "T_imu_cam1": t_i_c1.tolist(),
        "imu_values_precalibrated": precalibrated_imu,
        "camera_time_shift_s": float(imu_yaml["cam0"]["timeshift_cam_imu"]),
        "camera_time_shift_baked_into_image_timestamps": True,
    }
    return calibration, metadata


def read_imu_samples(path: Path) -> list[tuple[int, np.ndarray, np.ndarray]]:
    samples: dict[int, dict[str, np.ndarray]] = {}
    with path.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            timestamp_us = int(row["timestamp_us"])
            sample_type = row["type"]
            if sample_type not in ("accel", "gyro"):
                raise RuntimeError(f"unknown IMU sample type: {sample_type}")
            vector = np.asarray(
                (float(row["x"]), float(row["y"]), float(row["z"])), dtype=np.float64
            )
            entry = samples.setdefault(timestamp_us, {})
            if sample_type in entry:
                raise RuntimeError(f"duplicate {sample_type} sample at {timestamp_us}")
            entry[sample_type] = vector
    complete = []
    for timestamp_us in sorted(samples):
        entry = samples[timestamp_us]
        if "accel" in entry and "gyro" in entry:
            complete.append((timestamp_us, entry["accel"], entry["gyro"]))
    if len(complete) < 2:
        raise RuntimeError("not enough paired IMU samples")
    return complete


def calibrate_imu_sample(
    accel: np.ndarray, gyro: np.ndarray, imu0: dict
) -> tuple[np.ndarray, np.ndarray]:
    m_acc = np.asarray(imu0["M_acc"], dtype=np.float64)
    m_gyr = np.asarray(imu0["M_gyr"], dtype=np.float64)
    accel_bias = np.asarray(imu0["AccBias"], dtype=np.float64)
    gyro_bias = np.asarray(imu0["GyrBias"], dtype=np.float64)
    calibrated_accel = m_acc @ accel - accel_bias
    calibrated_gyro = m_gyr @ gyro - gyro_bias
    return calibrated_accel, calibrated_gyro


def read_frame_at(capture: cv2.VideoCapture, target: int, state: list[int]) -> np.ndarray:
    frame = None
    while state[0] <= target:
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError(f"video ended before frame {target}")
        state[0] += 1
    if frame is None:
        raise RuntimeError(f"failed to decode frame {target}")
    return frame


def ensure_empty_output(output: Path) -> None:
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)


def main() -> int:
    args = parse_args()
    if not 0.1 <= args.image_scale <= 1.0:
        raise ValueError("--image-scale must be in [0.1, 1.0]")
    if args.max_delta_us < 0 or args.max_pairs < 0:
        raise ValueError("--max-delta-us and --max-pairs must be non-negative")
    if not 0 <= args.png_compression <= 9:
        raise ValueError("--png-compression must be in [0, 9]")

    session = args.session.resolve()
    output = args.output.resolve()
    if not session.is_dir():
        raise FileNotFoundError(session)
    ensure_empty_output(output)

    left_video = unique_file(session, "_camera_left.mp4")
    right_video = unique_file(session, "_camera_right.mp4")
    left_pts = unique_file(session, "_camera_left_pts.csv")
    right_pts = unique_file(session, "_camera_right_pts.csv")
    camera_calibration_path = unique_file(session, "_calibration_camera.yaml")
    imu_calibration_path = unique_file(session, "_calibration_imu.yaml")
    imu_csv_path = unique_file(session, "_imu.csv")

    with camera_calibration_path.open("r", encoding="utf-8") as stream:
        camera_yaml = yaml.safe_load(stream)
    with imu_calibration_path.open("r", encoding="utf-8") as stream:
        imu_yaml = yaml.safe_load(stream)

    left_timestamps = read_timestamps(left_pts)
    right_timestamps = read_timestamps(right_pts)
    pairs = pair_timestamps(left_timestamps, right_timestamps, args.max_delta_us)
    if args.max_pairs > 0:
        pairs = pairs[:args.max_pairs]
    if not pairs:
        raise RuntimeError("no synchronized stereo pairs")

    precalibrated_imu = not args.no_precalibrate_imu
    calibration, calibration_metadata = build_basalt_calibration(
        camera_yaml, imu_yaml, args.image_scale, precalibrated_imu
    )
    resolution = tuple(calibration["value0"]["resolution"][0])
    time_shift_ns = int(round(float(imu_yaml["cam0"]["timeshift_cam_imu"]) * 1e9))

    mav0 = output / "mav0"
    cam0_data = mav0 / "cam0" / "data"
    cam1_data = mav0 / "cam1" / "data"
    imu0_dir = mav0 / "imu0"
    cam0_data.mkdir(parents=True)
    cam1_data.mkdir(parents=True)
    imu0_dir.mkdir(parents=True)

    left_capture = cv2.VideoCapture(str(left_video))
    right_capture = cv2.VideoCapture(str(right_video))
    if not left_capture.isOpened() or not right_capture.isOpened():
        raise RuntimeError("failed to open stereo videos")
    left_state = [0]
    right_state = [0]
    timestamp_rows = []
    started = time.perf_counter()
    for pair_index, (left_index, right_index) in enumerate(pairs):
        left_frame = read_frame_at(left_capture, left_index, left_state)
        right_frame = read_frame_at(right_capture, right_index, right_state)
        if left_frame.shape[:2][::-1] != tuple(calibration_metadata["original_resolution"]):
            raise RuntimeError("left decoded resolution differs from calibration")
        if right_frame.shape[:2][::-1] != tuple(calibration_metadata["original_resolution"]):
            raise RuntimeError("right decoded resolution differs from calibration")
        left_gray = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)
        if args.image_scale != 1.0:
            left_gray = cv2.resize(left_gray, resolution, interpolation=cv2.INTER_AREA)
            right_gray = cv2.resize(right_gray, resolution, interpolation=cv2.INTER_AREA)
        basalt_timestamp_ns = left_timestamps[left_index] * 1000 + time_shift_ns
        filename = f"{basalt_timestamp_ns}.png"
        parameters = [cv2.IMWRITE_PNG_COMPRESSION, args.png_compression]
        if not cv2.imwrite(str(cam0_data / filename), left_gray, parameters):
            raise RuntimeError(f"failed to write cam0/{filename}")
        if not cv2.imwrite(str(cam1_data / filename), right_gray, parameters):
            raise RuntimeError(f"failed to write cam1/{filename}")
        timestamp_rows.append({
            "pair_index": pair_index,
            "left_index": left_index,
            "right_index": right_index,
            "left_timestamp_us": left_timestamps[left_index],
            "right_timestamp_us": right_timestamps[right_index],
            "stereo_delta_us": right_timestamps[right_index] - left_timestamps[left_index],
            "camera_to_imu_shift_ns": time_shift_ns,
            "basalt_timestamp_ns": basalt_timestamp_ns,
            "filename": filename,
        })
        if (pair_index + 1) % 100 == 0 or pair_index + 1 == len(pairs):
            elapsed = time.perf_counter() - started
            print(f"images {pair_index + 1}/{len(pairs)} ({(pair_index + 1) / elapsed:.1f} pairs/s)")
    left_capture.release()
    right_capture.release()

    for camera_id in (0, 1):
        data_csv = mav0 / f"cam{camera_id}" / "data.csv"
        with data_csv.open("w", encoding="utf-8", newline="") as stream:
            stream.write("#timestamp [ns],filename\n")
            for row in timestamp_rows:
                stream.write(f"{row['basalt_timestamp_ns']},{row['filename']}\n")

    imu_samples = read_imu_samples(imu_csv_path)
    imu0 = imu_yaml["imu0"]
    with (imu0_dir / "data.csv").open("w", encoding="utf-8", newline="") as stream:
        stream.write(
            "#timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],"
            "w_RS_S_z [rad s^-1],a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],"
            "a_RS_S_z [m s^-2]\n"
        )
        for timestamp_us, accel, gyro in imu_samples:
            if precalibrated_imu:
                accel, gyro = calibrate_imu_sample(accel, gyro, imu0)
            values = (*gyro.tolist(), *accel.tolist())
            stream.write(
                f"{timestamp_us * 1000}," + ",".join(f"{value:.12g}" for value in values) + "\n"
            )

    with (output / "calibration.json").open("w", encoding="utf-8") as stream:
        json.dump(calibration, stream, indent=2)
        stream.write("\n")
    config_source = Path(__file__).resolve().parents[1] / "third_party" / "basalt" / "data" / "euroc_config.json"
    if not config_source.is_file():
        config_source = Path("/home/zdh/ego_hand_system/third_party/basalt/data/euroc_config.json")
    with config_source.open("r", encoding="utf-8") as stream:
        vio_config = json.load(stream)
    with (output / "vio_config.json").open("w", encoding="utf-8") as stream:
        json.dump(vio_config, stream, indent=2)
        stream.write("\n")

    map_fields = list(timestamp_rows[0])
    with (output / "ego_timestamp_map.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=map_fields)
        writer.writeheader()
        writer.writerows(timestamp_rows)

    stereo_deltas = np.asarray([row["stereo_delta_us"] for row in timestamp_rows])
    imu_timestamps = np.asarray([sample[0] for sample in imu_samples], dtype=np.int64)
    elapsed = time.perf_counter() - started
    summary = {
        "stage": "ego_to_basalt_euroc",
        "session": str(session),
        "output": str(output),
        "basalt_dataset_type": "euroc",
        "stereo_pairs": len(timestamp_rows),
        "left_frames_skipped_before_first_pair": timestamp_rows[0]["left_index"],
        "right_frames_skipped_before_first_pair": timestamp_rows[0]["right_index"],
        "stereo_delta_us_median": float(np.median(np.abs(stereo_deltas))),
        "stereo_delta_us_max_abs": int(np.max(np.abs(stereo_deltas))),
        "image_scale": args.image_scale,
        "resolution": list(resolution),
        "imu_samples": len(imu_samples),
        "imu_duration_s": float((imu_timestamps[-1] - imu_timestamps[0]) * 1e-6),
        "imu_rate_hz": float((len(imu_samples) - 1) / ((imu_timestamps[-1] - imu_timestamps[0]) * 1e-6)),
        "elapsed_seconds": elapsed,
        "processing_pairs_per_second": len(timestamp_rows) / elapsed,
        "calibration": calibration_metadata,
        "imu_correction": (
            "a_cal=M_acc*a_raw-AccBias; w_cal=M_gyr*w_raw-GyrBias; "
            "Basalt static calibration arrays are zero because samples are pre-calibrated."
            if precalibrated_imu else "raw EGO IMU samples; Basalt static calibration arrays are zero"
        ),
        "outputs": {
            "calibration": "calibration.json",
            "config": "vio_config.json",
            "timestamp_map": "ego_timestamp_map.csv",
            "cam0": "mav0/cam0",
            "cam1": "mav0/cam1",
            "imu0": "mav0/imu0/data.csv",
        },
    }
    with (output / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
        stream.write("\n")

    print(f"Basalt dataset: {output}")
    print(f"Stereo pairs: {len(timestamp_rows)} at {resolution[0]}x{resolution[1]}")
    print(f"IMU samples: {len(imu_samples)} ({summary['imu_rate_hz']:.1f} Hz)")
    print(f"Stereo baseline: {calibration_metadata['stereo_baseline_m']:.6f} m")
    print(f"Camera/IMU time shift baked into images: {time_shift_ns} ns")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
