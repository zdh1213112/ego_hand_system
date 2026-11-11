#!/usr/bin/env python3
"""Prepare and benchmark original/Monado Basalt on identical EGO-4 frames.

Keep the original recording read-only. All new datasets, commands and logs are
written under --output. Run VIO cases serially to avoid CPU contention.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import cv2
import numpy as np

from prepare_basalt_dataset import build_basalt_calibration, calibrate_imu_sample, read_imu_samples, read_timestamps
from prepare_basalt_video_dataset import prepare
from run_orbbec_headring_vio import load_vendor_yaml
from associate_vio_imu import associate
from ego_data.time_sync import camera_offset_policy, require_zero_camera_offset

ROOT = Path(__file__).resolve().parents[1]
STREAMS = ("left", "right", "right_side", "left_side")


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def source_info(source: Path) -> dict:
    def git(*args):
        return subprocess.check_output(["git", "-C", str(source), *args], text=True).strip()
    return {"path": str(source), "commit": git("rev-parse", "HEAD"),
            "status": git("status", "--short"), "remotes": git("remote", "-v")}


def prepare_data(args) -> None:
    output, session = args.output.resolve(), args.session.resolve()
    meta_path = next(session.glob("meta_*.json"))
    meta = json.loads(meta_path.read_text())
    files = meta["files"]
    camera_path = session / files["calibration_camera"]
    imu_calib_path = session / files["calibration_imu"]
    cameras, imu_yaml = load_vendor_yaml(camera_path), load_vendor_yaml(imu_calib_path)
    cameras["cameras"].sort(key=lambda c: int(c["id"].split("_")[-1]))
    assert [c["id"] for c in cameras["cameras"]] == [f"cam_{i}" for i in range(4)]
    calibration, calibration_info = build_basalt_calibration(cameras, imu_yaml, 1.0, True)
    videos = [session / files[f"video_{stream}"] for stream in STREAMS]
    pts_paths = [session / files[f"video_{stream}_pts"] for stream in STREAMS]
    streams = [read_timestamps(p) for p in pts_paths]
    counts = []
    for video, timestamps in zip(videos, streams):
        capture = cv2.VideoCapture(str(video))
        counts.append(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        capture.release()
        if counts[-1] != len(timestamps):
            raise ValueError(f"video/PTS mismatch: {video}")
    samples = read_imu_samples(session / files["imu"])
    time_policy = camera_offset_policy(args, imu_yaml)
    shift_ns = time_policy["camera_to_imu_offset_ns"]
    calibration_info["camera_time_shift_s"] = shift_ns * 1e-9
    print("Camera timing policy: " + json.dumps(time_policy), flush=True)
    # Preserve epoch timestamps as int64, not float (the input is ~1.8e18 ns).
    shifted = [[t * 1000 for t in times] for times in streams]
    adapter = output / "input_adapter"
    adapter.mkdir(parents=True, exist_ok=False)
    dump(adapter / "calibration.json", calibration)
    timing_paths = []
    for i, times in enumerate(shifted):
        path = adapter / f"cam{i}_timestamps.csv"
        # Keep one timestamp per original video frame. The group filter below
        # handles both leading and trailing images outside IMU coverage.
        path.write_text("timestamp_ns\n" + "".join(f"{t}\n" for t in times))
        timing_paths.append(path)
    imu_path = adapter / "imu.csv"
    with imu_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp_ns", "wx", "wy", "wz", "ax", "ay", "az"])
        for ts, accel, gyro in samples:
            accel, gyro = calibrate_imu_sample(accel, gyro, imu_yaml["imu0"])
            w.writerow([ts * 1000, *gyro, *accel])
    start = time.perf_counter()
    four = prepare(argparse.Namespace(
        videos=videos, timestamps=timing_paths, calibration=adapter / "calibration.json",
        config=ROOT / "third_party/basalt_monado/data/default_config.json", imu=imu_path,
        output=output / "dataset_4cam", max_delta_us=100, max_frames=args.max_frames,
        image_scale=args.image_scale, first_timestamp_ns=samples[0][0] * 1000,
        last_timestamp_ns=samples[-1][0] * 1000))
    # Stereo uses the exact same times and cam0/cam1 PNG bytes, without duplicating
    # storage. It does not select a different, more favorable set of stereo pairs.
    stereo = output / "dataset_2cam"
    (stereo / "mav0").mkdir(parents=True)
    for name in ("cam0", "cam1", "imu0"):
        (stereo / "mav0" / name).symlink_to(four / "mav0" / name, target_is_directory=True)
    c2 = json.loads((four / "calibration.json").read_text())
    for key in ("T_imu_cam", "intrinsics", "resolution", "vignette"):
        c2["value0"][key] = c2["value0"][key][:2]
    dump(stereo / "calibration.json", c2)
    (stereo / "frame_map.csv").symlink_to(four / "frame_map.csv")
    common = json.loads((ROOT / "third_party/basalt/data/tumvi_512_config.json").read_text())["value0"]
    common.update({"config.optical_flow_detection_grid_size": 50,
                   "config.vio_use_lm": True, "config.vio_lm_lambda_initial": 1e-4,
                   "config.vio_enforce_realtime": False, "config.vio_extended_logging": False})
    original = {"value0": common}
    monado = json.loads((four / "config.json").read_text())
    monado["value0"].update(common)
    dump(output / "config_original.json", original)
    dump(output / "config_monado.json", monado)
    frame_rows = list(csv.DictReader((four / "frame_map.csv").open()))
    ts = np.array([int(r["timestamp_ns"]) for r in frame_rows], dtype=np.int64)
    deltas = [abs(int(r[f"cam{i}_delta_ns"])) for r in frame_rows for i in range(4)]
    source_paths = [meta_path, camera_path, imu_calib_path, session / files["imu"], *videos, *pts_paths]
    summary = {"session": str(session), "source_meta": meta, "source_video_counts": counts,
               "source_camera_order": list(STREAMS), "prepared_groups": len(ts),
               "duration_s": float((ts[-1] - ts[0]) * 1e-9),
               "first_timestamp_ns": int(ts[0]), "last_timestamp_ns": int(ts[-1]),
               "image_scale": args.image_scale, "resolution": json.loads((four / "calibration.json").read_text())["value0"]["resolution"][0],
               "camera_delta_ns_max": max(deltas), "camera_delta_ns_p95": float(np.percentile(deltas, 95)),
               "frame_gap_ms_max": float(np.max(np.diff(ts)) * 1e-6),
               "imu_samples": len(samples), "camera_time_shift_ns": shift_ns,
               "time_sync_policy": time_policy,
               "calibration": calibration_info, "preparation_wall_s": time.perf_counter() - start,
               "ground_truth_status": "external reference not bundled in recording; evaluate separately",
               "files": [{"path": str(p), "bytes": p.stat().st_size, "sha256": sha256(p)} for p in source_paths],
               "sources": {name: source_info(ROOT / "third_party" / name) for name in ("basalt", "basalt_monado")}}
    dump(output / "input_validation.json", summary)
    print(json.dumps({k: summary[k] for k in ("prepared_groups", "duration_s", "resolution", "camera_delta_ns_max", "frame_gap_ms_max", "imu_samples")}, indent=2), flush=True)


def timed_run(command, directory: Path, stem: str, timeout: int = 1800) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    dump(directory / f"{stem}_command.json", command)
    environment = os.environ.copy()
    for key in ("LD_LIBRARY_PATH", "CMAKE_PREFIX_PATH", "CONDA_PREFIX", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        environment.pop(key, None)
    environment.update({"PATH": "/usr/bin:/bin", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"})
    wrapped = ["/usr/bin/time", "-v", "-o", str(directory / f"{stem}_resource.txt"), *command]
    started = time.perf_counter()
    with (directory / f"{stem}.log").open("w") as log:
        process = subprocess.Popen(wrapped, cwd=directory, env=environment, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            import signal
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            code = -999
    result = {"wall_s": time.perf_counter() - started, "returncode": code}
    dump(directory / f"{stem}_timing.json", result)
    return result


def run_cases(args) -> None:
    output = args.output.resolve()
    require_zero_camera_offset(json.loads((output/'input_validation.json').read_text()),str(output))
    binaries = {"original": output / "build_original/basalt_vio",
                "monado": ROOT / "third_party/basalt_monado/build/basalt_vio"}
    cases = {"original_2cam": ("original", 2), "monado_2cam": ("monado", 2), "monado_4cam": ("monado", 4)}
    selected = args.cases or list(cases)
    all_results = []
    # Alternate the case order across repeats to reduce ordering/cache bias.
    for repeat in range(1, args.repeats + 1):
        order = selected if repeat % 2 else list(reversed(selected))
        for name in order:
            family, cameras = cases[name]
            result_dir = output / "runs" / name / f"repeat_{repeat}"
            if (result_dir / "vio_timing.json").exists():
                print(f"Retaining existing run: {result_dir}", flush=True)
                continue
            dataset = output / f"dataset_{cameras}cam"
            require_zero_camera_offset(json.loads((dataset/'calibration.json').read_text()),str(dataset))
            command = [str(binaries[family]), "--dataset-path", str(dataset), "--dataset-type", "euroc",
                       "--cam-calib", str(dataset / "calibration.json"), "--config-path", str(output / f"config_{family}.json"),
                       "--show-gui", "0", "--use-imu", "1", "--use-double", "0",
                       "--num-threads", str(args.threads), "--save-trajectory", "euroc"]
            if args.max_frames:
                command += ["--max-frames", str(args.max_frames)]
            # Save marginalization for all repetitions: same output overhead for
            # every timed VIO run. Mapping is separately timed on repeat 1 only.
            if args.save_marg:
                command += ["--marg-data", str(result_dir / "marg_data")]
            print(f"Starting {name} repeat {repeat}", flush=True)
            timing = timed_run(command, result_dir, "vio", args.timeout)
            timing.update({"case": name, "repeat": repeat})
            all_results.append(timing)
            print(json.dumps(timing), flush=True)
            if timing["returncode"] != 0:
                print(f"Run failed; inspect {result_dir / 'vio.log'}", flush=True)
            else:
                associate(result_dir / "trajectory.csv", output / "input_adapter/imu.csv", result_dir)
    dump(output / "last_batch.json", all_results)


def run_mapping(args) -> None:
    output = args.output.resolve()
    require_zero_camera_offset(json.loads((output/'input_validation.json').read_text()),str(output))
    cases = {"original_2cam": ("original", 2), "monado_2cam": ("monado", 2), "monado_4cam": ("monado", 4)}
    for name in args.cases or list(cases):
        family, cameras = cases[name]
        directory = output / "runs" / name / "repeat_1"
        if not (directory / "marg_data").is_dir():
            raise FileNotFoundError(f"VIO must run with --save-marg: {directory}")
        suffix = "_lm" if args.mapping_solver == "lm" else ""
        map_directory = directory / ("map" + suffix)
        stem = "mapping" + suffix
        if map_directory.exists() or (directory / f"{stem}_timing.json").exists():
            print(f"Retaining existing mapping result: {directory}", flush=True)
            continue
        binary = (output / "build_original/basalt_mapper_export" if family == "original" else
                  output / "basalt_monado_mapper_export")
        config_path = output / f"config_{family}.json"
        if args.mapping_solver == "lm":
            config = json.loads(config_path.read_text())
            config["value0"].update({"config.mapper_use_lm": True, "config.mapper_lm_lambda_min": 1e-6})
            config_path = output / f"config_map_lm_{family}.json"
            dump(config_path, config)
        command = [str(binary), str(output / f"dataset_{cameras}cam/calibration.json"),
                   str(config_path), str(directory / "marg_data"), str(map_directory)]
        print(f"Starting NFR mapping: {name}", flush=True)
        result = timed_run(command, directory, stem, args.timeout)
        print(name, json.dumps(result), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "run", "map"))
    parser.add_argument("--session", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-scale", type=float, default=0.5)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--cases", nargs="+", choices=("original_2cam", "monado_2cam", "monado_4cam"))
    parser.add_argument("--save-marg", action="store_true")
    parser.add_argument("--mapping-solver", choices=("gn", "lm"), default="gn")
    args = parser.parse_args()
    if args.phase == "prepare":
        if not args.session:
            parser.error("prepare requires --session")
        prepare_data(args)
    elif args.phase == "run":
        run_cases(args)
    else:
        run_mapping(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
