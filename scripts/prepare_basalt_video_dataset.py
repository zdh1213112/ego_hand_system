#!/usr/bin/env python3
"""Convert synchronized camera videos and native Basalt calibration to EuRoC."""

from __future__ import annotations

import argparse
from bisect import bisect_left
import csv
import json
import math
from pathlib import Path
import shutil
import subprocess
import tempfile

import cv2


def read_timestamps(path: Path) -> list[int]:
    """One timestamp_ns row per decoded frame, in video presentation order."""
    with path.open(newline="", encoding="utf-8") as stream:
        values = [int(row["timestamp_ns"]) for row in csv.DictReader(stream)]
    if not values or values[0] < 0:
        raise ValueError(f"empty or negative timestamps: {path}")
    if any(b <= a for a, b in zip(values, values[1:])):
        raise ValueError(f"timestamps must strictly increase: {path}")
    return values


def synchronize(streams: list[list[int]], tolerance_ns: int) -> list[tuple[int, ...]]:
    """Use camera 0 as reference; never repeat or reverse another camera's frames."""
    next_indices = [0] * len(streams)
    pairs = []
    for reference_index, timestamp in enumerate(streams[0]):
        selected = [reference_index]
        for camera, values in enumerate(streams[1:], 1):
            position = bisect_left(values, timestamp, lo=next_indices[camera])
            candidates = [i for i in (position - 1, position)
                          if next_indices[camera] <= i < len(values)]
            if not candidates:
                break
            best = min(candidates, key=lambda i: abs(values[i] - timestamp))
            if abs(values[best] - timestamp) > tolerance_ns:
                break
            selected.append(best)
        if len(selected) == len(streams):
            pairs.append(tuple(selected))
            for camera, index in enumerate(selected):
                next_indices[camera] = index + 1
    return pairs


def validate_calibration(calibration: dict, camera_count: int) -> list[list[int]]:
    value = calibration["value0"]
    for key in ("intrinsics", "resolution", "T_imu_cam", "vignette"):
        if len(value[key]) != camera_count:
            raise ValueError(f"calibration {key} must have {camera_count} entries")
    sizes = value["resolution"]
    if any(len(size) != 2 or any(type(v) is not int or v <= 0 for v in size) for size in sizes):
        raise ValueError("calibration resolution must contain positive integer [width, height]")
    if any(size != sizes[0] for size in sizes):
        raise ValueError("this entry point requires equal camera resolutions")
    for pose in value["T_imu_cam"]:
        if not all(math.isfinite(pose[k]) for k in ("px", "py", "pz", "qx", "qy", "qz", "qw")):
            raise ValueError("non-finite camera transform")
        if not math.isclose(sum(pose[k] ** 2 for k in ("qx", "qy", "qz", "qw")), 1, abs_tol=1e-4):
            raise ValueError("camera transform quaternion is not normalized")
    # The inspected upstream reader does not apply this value to image timestamps.
    if value.get("cam_time_offset_ns", 0) != 0:
        raise ValueError("Camera/IMU time correction is disabled: preserve recorded timestamps and set cam_time_offset_ns=0")
    return sizes


def read_imu(path: Path, first_ns: int, last_ns: int) -> list[tuple[int, list[float]]]:
    """Require explicit SI units and ordering, rather than guessing vendor formats."""
    with path.open(newline="", encoding="utf-8") as stream:
        rows = [(int(row["timestamp_ns"]), [float(row[k]) for k in
                 ("wx", "wy", "wz", "ax", "ay", "az")]) for row in csv.DictReader(stream)]
    if len(rows) < 2 or rows[0][0] < 0 or any(b[0] <= a[0] for a, b in zip(rows, rows[1:])):
        raise ValueError("IMU timestamps must be nonnegative and strictly increase")
    if any(not all(math.isfinite(v) for v in values) for _, values in rows):
        raise ValueError("non-finite IMU measurement")
    if rows[0][0] > first_ns or rows[-1][0] <= last_ns:
        raise ValueError("IMU must cover the first image and extend beyond the last image")
    return rows


def prepare(args: argparse.Namespace) -> Path:
    videos = [path.resolve() for path in args.videos]
    timestamps = [path.resolve() for path in args.timestamps]
    if len(videos) < 2 or len(videos) != len(timestamps):
        raise ValueError("provide at least two videos and one timestamp CSV per video")
    if len(set(videos)) != len(videos):
        raise ValueError("camera videos must be different files")
    if args.max_delta_us < 0 or args.max_frames < 0:
        raise ValueError("max-delta-us/max-frames must be nonnegative")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"choose a new output directory: {output}")
    for path in [*videos, *timestamps, args.calibration, args.config]:
        if not path.is_file():
            raise FileNotFoundError(path)
    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
    sizes = validate_calibration(calibration, len(videos))
    scale = getattr(args, "image_scale", 1.0)
    if not math.isfinite(scale) or not 0.1 <= scale <= 1.0:
        raise ValueError("image-scale must be in [0.1, 1.0]")
    output_sizes = [[round(w * scale), round(h * scale)] for w, h in sizes]
    for intrinsics, size, output_size in zip(calibration["value0"]["intrinsics"], sizes, output_sizes):
        for name, factor in (("fx", output_size[0] / size[0]), ("cx", output_size[0] / size[0]),
                             ("fy", output_size[1] / size[1]), ("cy", output_size[1] / size[1])):
            intrinsics["intrinsics"][name] *= factor
    calibration["value0"]["resolution"] = output_sizes
    config = json.loads(args.config.read_text(encoding="utf-8"))
    options = config["value0"]
    streams = [read_timestamps(path) for path in timestamps]
    pairs = synchronize(streams, args.max_delta_us * 1000)
    # Filter groups, never timestamp rows: source frame indices must continue
    # to address the original videos when the IMU starts after a camera.
    first_ns = getattr(args, "first_timestamp_ns", None)
    last_ns = getattr(args, "last_timestamp_ns", None)
    pairs = [indices for indices in pairs
             if (first_ns is None or streams[0][indices[0]] >= first_ns)
             and (last_ns is None or streams[0][indices[0]] < last_ns)]
    if args.max_frames:
        pairs = pairs[:args.max_frames]
    if len(pairs) < 2:
        raise ValueError("fewer than two synchronized image groups")
    imu = read_imu(args.imu, streams[0][pairs[0][0]], streams[0][pairs[-1][0]]) if args.imu else []
    options["config.vio_enforce_realtime"] = False
    if not imu:
        # Fixed-template tracking avoids an IMU-based prediction in visual-only runs.
        options["config.optical_flow_type"] = "patch"
        options["config.optical_flow_recall_enable"] = False

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    readers = []
    files = []
    try:
        for camera, video in enumerate(videos):
            capture = cv2.VideoCapture(str(video))
            readers.append(capture)
            if not capture.isOpened():
                raise RuntimeError(f"cannot open {video}")
            directory = temporary / "mav0" / f"cam{camera}"
            (directory / "data").mkdir(parents=True)
            stream = (directory / "data.csv").open("w", encoding="utf-8")
            stream.write("#timestamp [ns],filename\n")
            files.append(stream)
        next_indices = [0] * len(videos)
        mapping = []
        for group, indices in enumerate(pairs):
            timestamp = streams[0][indices[0]]
            filename = f"{timestamp}.png"
            row = {"group_index": group, "timestamp_ns": timestamp}
            for camera, index in enumerate(indices):
                frame = None
                while next_indices[camera] <= index:
                    ok, frame = readers[camera].read()
                    if not ok:
                        raise RuntimeError(f"camera {camera} ended before frame {index}")
                    next_indices[camera] += 1
                if frame is None or list(frame.shape[1::-1]) != sizes[camera]:
                    raise ValueError(f"camera {camera} video resolution differs from calibration")
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if output_sizes[camera] != sizes[camera]:
                    gray = cv2.resize(gray, tuple(output_sizes[camera]), interpolation=cv2.INTER_AREA)
                destination = temporary / "mav0" / f"cam{camera}" / "data" / filename
                if not cv2.imwrite(str(destination), gray, [cv2.IMWRITE_PNG_COMPRESSION, 1]):
                    raise RuntimeError(f"cannot write {destination}")
                files[camera].write(f"{timestamp},{filename}\n")
                row[f"cam{camera}_frame_index"] = index
                row[f"cam{camera}_source_timestamp_ns"] = streams[camera][index]
                row[f"cam{camera}_delta_ns"] = streams[camera][index] - timestamp
            mapping.append(row)
            if (group + 1) % 100 == 0:
                print(f"Converted {group + 1}/{len(pairs)} image groups", flush=True)
        imu_dir = temporary / "mav0" / "imu0"
        imu_dir.mkdir()
        with (imu_dir / "data.csv").open("w", encoding="utf-8") as stream:
            stream.write("#timestamp [ns],wx,wy,wz,ax,ay,az\n")
            for timestamp, values in imu:
                stream.write(f"{timestamp}," + ",".join(str(v) for v in values) + "\n")
        with (temporary / "frame_map.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(mapping[0]))
            writer.writeheader()
            writer.writerows(mapping)
        summary = {
            "mode": "vio" if imu else "vo", "camera_count": len(videos),
            "image_scale": scale, "output_resolution": output_sizes,
            "image_groups": len(pairs), "imu_samples": len(imu),
            "videos": [str(p) for p in videos], "timestamps": [str(p) for p in timestamps],
            "max_abs_camera_delta_ns": max(abs(row[f"cam{i}_delta_ns"])
                                            for row in mapping for i in range(len(videos))),
            "calibration_source": str(args.calibration.resolve()),
            "config_source": str(args.config.resolve()),
            "pose_convention": "T_world_body; calibration T_imu_cam maps camera to body",
            "note": "Pairing does not correct physical exposure synchronization or clock drift.",
        }
        for name, value in (("calibration.json", calibration), ("config.json", config), ("summary.json", summary)):
            (temporary / name).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
        for stream in files:
            stream.close()
        temporary.rename(output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    finally:
        for capture in readers:
            capture.release()
        for stream in files:
            stream.close()
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos", nargs="+", type=Path, required=True)
    parser.add_argument("--timestamps", nargs="+", type=Path, required=True,
                        help="CSV per video, timestamp_ns column, one row per decoded frame")
    parser.add_argument("--calibration", type=Path, required=True, help="native Basalt calibration JSON")
    parser.add_argument("--config", type=Path, required=True, help="config JSON from the matching Basalt checkout")
    parser.add_argument("--imu", type=Path, help="CSV: timestamp_ns,wx,wy,wz,ax,ay,az; rad/s and m/s^2")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-delta-us", type=int, default=0, help="maximum offset from camera 0; default exact sync")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--image-scale", type=float, default=1.0)
    parser.add_argument("--basalt-bin", type=Path, help="optionally run this basalt_vio after conversion")
    args = parser.parse_args()
    if args.basalt_bin and not args.basalt_bin.is_file():
        raise FileNotFoundError(args.basalt_bin)
    dataset = prepare(args)
    print(f"Prepared dataset: {dataset}")
    if args.basalt_bin:
        command = [str(args.basalt_bin.resolve()), "--dataset-path", str(dataset),
                   "--dataset-type", "euroc", "--cam-calib", str(dataset / "calibration.json"),
                   "--config-path", str(dataset / "config.json"), "--show-gui", "0",
                   "--use-imu", str(int(args.imu is not None)), "--save-trajectory", "euroc"]
        with (dataset / "basalt.log").open("w", encoding="utf-8") as stream:
            subprocess.run(command, cwd=dataset, stdout=stream, stderr=subprocess.STDOUT, check=True)
        trajectory = dataset / "trajectory.csv"
        if not trajectory.is_file() or trajectory.stat().st_size == 0:
            raise RuntimeError(f"Basalt did not write a trajectory; inspect {dataset / 'basalt.log'}")
        print(f"Trajectory: {trajectory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
