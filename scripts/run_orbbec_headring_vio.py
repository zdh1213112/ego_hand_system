#!/usr/bin/env python3
"""Run calibrated Orbbec multi-camera + IMU VIO without a hand-pose dependency."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import yaml

from prepare_basalt_dataset import (build_basalt_calibration, calibrate_imu_sample,
                                    read_imu_samples, read_timestamps, unique_file)
from prepare_basalt_video_dataset import prepare
from associate_vio_imu import associate
from ego_data.time_sync import camera_offset_policy
from fuse_basalt_hand_trajectory import (load_basalt_trajectory, pose_from_basalt_json,
                                        quaternion_xyzw_from_matrix, rpy_zyx_from_matrix)


def load_vendor_yaml(path: Path) -> dict:
    # OpenCV's %YAML:1.0 directive is not valid standard YAML syntax.
    text = path.read_text(encoding="utf-8-sig")
    return yaml.safe_load("\n".join(line for line in text.splitlines() if not line.startswith("%YAML:")))


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--basalt-source", type=Path, default=root / "third_party/basalt_monado")
    parser.add_argument("--image-scale", type=float, default=0.5)
    parser.add_argument("--num-threads", type=int, default=2)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--save-marg-data", action="store_true")
    args = parser.parse_args()
    session, output = args.session.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError(f"choose a new output: {output}")
    source = args.basalt_source.resolve()
    binary = source / "build/basalt_vio"
    if not binary.is_file():
        raise FileNotFoundError(binary)
    cameras = load_vendor_yaml(unique_file(session, "_calibration_camera.yaml"))
    imu_yaml = load_vendor_yaml(unique_file(session, "_calibration_imu.yaml"))
    camera_order = {"cam_0": "left", "cam_1": "right", "cam_2": "right_side", "cam_3": "left_side"}
    cameras["cameras"].sort(key=lambda item: int(item["id"].split("_")[-1]))
    if any(camera["id"] not in camera_order for camera in cameras["cameras"]):
        raise ValueError("unsupported Orbbec camera IDs")
    calibration, metadata = build_basalt_calibration(cameras, imu_yaml, 1.0, True)
    samples = read_imu_samples(unique_file(session, "_imu.csv"))
    time_policy = camera_offset_policy(args, imu_yaml)
    print("Camera timing policy: " + json.dumps(time_policy), flush=True)
    output.mkdir(parents=True)
    adapter = output / "input_adapter"
    adapter.mkdir()
    (adapter / "calibration.json").write_text(json.dumps(calibration, indent=2) + "\n")
    videos, timestamp_files, timing = [], [], {}
    for camera in cameras["cameras"]:
        stream = camera_order[camera["id"]]
        values = read_timestamps(unique_file(session, f"_camera_{stream}_pts.csv"))
        kept = [value for value in values if samples[0][0] <= value < samples[-1][0]]
        if len(kept) < 2:
            raise ValueError("insufficient video/IMU overlap")
        path = adapter / f"{camera['id']}_timestamps.csv"
        path.write_text("timestamp_ns\n" + "".join(f"{value * 1000}\n" for value in values))
        timestamp_files.append(path)
        videos.append(unique_file(session, f"_camera_{stream}.mp4"))
        timing[camera["id"]] = {"original_frames": len(values), "frames_with_imu_coverage": len(kept)}
    imu_path = adapter / "imu.csv"
    with imu_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("timestamp_ns", "wx", "wy", "wz", "ax", "ay", "az"))
        for timestamp, accel, gyro in samples:
            accel, gyro = calibrate_imu_sample(accel, gyro, imu_yaml["imu0"])
            writer.writerow((timestamp * 1000, *gyro, *accel))
    metadata.update({"camera_order": [camera["id"] for camera in cameras["cameras"]],
                     "timing": timing, "camera_to_imu_shift_ns": 0,
                     "camera_time_shift_s": 0.0,
                     "time_sync_policy": time_policy,
                     "source_session": str(session), "image_scale": args.image_scale})
    (adapter / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    dataset = prepare(argparse.Namespace(
        videos=videos, timestamps=timestamp_files, calibration=adapter / "calibration.json",
        config=source / "data/default_config.json", imu=imu_path, output=output / "dataset",
        max_delta_us=100, max_frames=args.max_frames, image_scale=args.image_scale,
        first_timestamp_ns=samples[0][0]*1000,last_timestamp_ns=samples[-1][0]*1000))
    command = [str(binary), "--dataset-path", str(dataset), "--dataset-type", "euroc",
               "--cam-calib", str(dataset / "calibration.json"),
               "--config-path", str(dataset / "config.json"), "--show-gui", "0",
               "--use-imu", "1", "--num-threads", str(args.num_threads),
               "--save-trajectory", "euroc"]
    if args.save_marg_data:
        # Basalt clears this directory internally, so always use a fresh output root.
        command.extend(("--marg-data", str(output / "marg_data")))
    (output / "vio_command.json").write_text(json.dumps(command, indent=2) + "\n")
    print("Running Basalt VIO", flush=True)
    started = time.perf_counter()
    with (output / "vio.log").open("w") as log:
        subprocess.run(command, cwd=output, stdout=log, stderr=subprocess.STDOUT, check=True)
    elapsed = time.perf_counter() - started
    imu_association = associate(output / "trajectory.csv", imu_path, output)
    poses = load_basalt_trajectory(output / "trajectory.csv")
    transform = pose_from_basalt_json(calibration["value0"]["T_imu_cam"][0])
    timestamps, matrices = zip(*sorted(poses.items()))
    head_poses = [pose @ transform for pose in matrices]
    columns = ["timestamp_ns", "source_cam0_timestamp_us", "time_s", "x_m", "y_m", "z_m",
               "qw", "qx", "qy", "qz", "roll_deg", "pitch_deg", "yaw_deg"]
    with (output / "headring_poses.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        for timestamp, pose in zip(timestamps, head_poses):
            q = quaternion_xyzw_from_matrix(pose[:3, :3])
            rpy = np.rad2deg(rpy_zyx_from_matrix(pose[:3, :3]))
            writer.writerow((timestamp, timestamp // 1000, (timestamp - timestamps[0]) * 1e-9,
                             *pose[:3, 3], q[3], *q[:3], *rpy))
    with (output / "headring_poses.tum").open("w") as stream:
        for timestamp, pose in zip(timestamps, head_poses):
            q = quaternion_xyzw_from_matrix(pose[:3, :3])
            stream.write(f"{timestamp * 1e-9:.9f} " + " ".join(f"{v:.12g}" for v in (*pose[:3, 3], *q)) + "\n")
    positions = np.asarray([pose[:3, 3] for pose in head_poses])
    distances = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    intervals = np.diff(np.asarray(timestamps, dtype=np.int64)) * 1e-9
    summary = {"source": str(session), "mode": "four_camera_inertial" if len(videos) == 4 else "multi_camera_inertial",
               "pose_count": len(poses), "duration_s": (timestamps[-1] - timestamps[0]) * 1e-9,
               "processing_s": elapsed, "path_length_m": float(distances.sum()),
               "time_sync_policy": time_policy, "vio_imu_timing": imu_association,
               "first_last_displacement_m": float(np.linalg.norm(positions[-1] - positions[0])),
               "frame_translation_p95_m": float(np.percentile(distances, 95)),
               "max_speed_m_s": float(np.max(distances / intervals)),
               "headring_reference": "factory cam_0 optical frame; geometric ring center is not calibrated",
               "world_frame": "Basalt initial IMU local world, gravity aligned, metres",
               "quaternion_csv_order": "qw,qx,qy,qz", "quaternion_tum_order": "qx,qy,qz,qw",
               "transform": "T_world_cam0 = T_world_imu @ T_imu_cam0",
               "accuracy": "no external ground truth; trajectory continuity is not an accuracy measurement",
               "position_min_m": positions.min(axis=0).tolist(), "position_max_m": positions.max(axis=0).tolist()}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
