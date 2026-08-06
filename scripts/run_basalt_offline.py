#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Run the complete EGO stereo-inertial Basalt pipeline and fuse the "
            "camera world trajectory with an existing hand end-effector 6D CSV."
        )
    )
    parser.add_argument("--session", required=True, type=Path)
    parser.add_argument("--hand-pose-csv", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument(
        "--world-output", type=Path,
        help="world-hand output directory; allows reusing one Basalt dataset with a new MANO fit",
    )
    parser.add_argument("--image-scale", type=float, default=0.5)
    parser.add_argument("--max-delta-us", type=int, default=1500)
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument("--num-threads", type=int, default=8)
    parser.add_argument("--trajectory-length", type=int, default=300)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument(
        "--rebuild-dataset", action="store_true",
        help="refuse reuse of an existing dataset and ask for a fresh output root",
    )
    parser.set_defaults(project_root=project_root)
    return parser.parse_args()


def session_output_name(session: Path) -> str:
    marker = "Orbbec_Ego_"
    if marker in session.name:
        suffix = session.name.split(marker, 1)[1]
        parts = suffix.rsplit("_", 2)
        if len(parts) == 3:
            return f"recording_{parts[1]}_{parts[2]}"
    return session.name


def default_hand_pose(project_root: Path, session: Path) -> Path:
    session_output = project_root / "output" / session_output_name(session)
    candidates = [
        session_output / "mano_overlay_trajectory_tuned" / "hand_end_effector_6d.csv",
        session_output / "mano_overlay_trajectory" / "hand_end_effector_6d.csv",
    ]
    return next((path for path in candidates if path.is_file()), candidates[0])


def run(command: list[str], *, cwd: Path | None = None, env: dict | None = None) -> None:
    print("+", " ".join(str(value) for value in command), flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def require_files(paths: list[Path], label: str) -> None:
    missing = [path for path in paths if not path.is_file()]
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"missing {label}:\n{formatted}")


def main() -> int:
    args = parse_args()
    if not 0.1 <= args.image_scale <= 1.0:
        raise ValueError("--image-scale must be in [0.1, 1.0]")
    if args.max_delta_us < 0 or args.max_pairs < 0 or args.num_threads < 1:
        raise ValueError("invalid non-positive pipeline option")

    project_root = args.project_root.resolve()
    session = args.session.resolve()
    if not session.is_dir():
        raise FileNotFoundError(session)
    output_root = (
        args.output_root.resolve()
        if args.output_root
        else project_root / "output" / session_output_name(session) / "basalt_world"
    )
    hand_pose_csv = (
        args.hand_pose_csv.resolve()
        if args.hand_pose_csv
        else default_hand_pose(project_root, session)
    )

    runtime = project_root / "third_party" / "basalt_runtime"
    basalt_vio = runtime / "bin" / "basalt_vio"
    basalt_library = runtime / "lib" / "libbasalt.so"
    prepare_script = project_root / "scripts" / "prepare_basalt_dataset.py"
    fuse_script = project_root / "scripts" / "fuse_basalt_hand_trajectory.py"
    require_files(
        [basalt_vio, basalt_library, prepare_script, fuse_script, hand_pose_csv],
        "Basalt pipeline input",
    )

    dataset = output_root / "dataset"
    world_output = (
        args.world_output.resolve()
        if args.world_output
        else output_root / "world_hand_trajectory"
    )
    output_root.mkdir(parents=True, exist_ok=True)

    dataset_outputs = [
        dataset / "calibration.json",
        dataset / "vio_config.json",
        dataset / "ego_timestamp_map.csv",
        dataset / "summary.json",
        dataset / "mav0" / "imu0" / "data.csv",
    ]
    dataset_ready = all(path.is_file() for path in dataset_outputs)
    if args.rebuild_dataset and dataset.exists():
        raise RuntimeError(
            f"--rebuild-dataset never deletes data. Choose a fresh --output-root: {output_root}"
        )
    if not dataset_ready:
        if dataset.exists() and any(dataset.iterdir()):
            raise RuntimeError(
                f"partial/non-empty Basalt dataset: {dataset}. Choose a fresh --output-root."
            )
        command = [
            sys.executable, str(prepare_script),
            "--session", str(session),
            "--output", str(dataset),
            "--image-scale", str(args.image_scale),
            "--max-delta-us", str(args.max_delta_us),
        ]
        if args.max_pairs:
            command += ["--max-pairs", str(args.max_pairs)]
        run(command)
    else:
        print(f"Reuse prepared Basalt dataset: {dataset}")

    trajectory = dataset / "trajectory.csv"
    if not trajectory.is_file():
        basalt_env = os.environ.copy()
        current_library_path = basalt_env.get("LD_LIBRARY_PATH", "")
        basalt_env["LD_LIBRARY_PATH"] = (
            str(runtime / "lib")
            if not current_library_path
            else f"{runtime / 'lib'}:{current_library_path}"
        )
        run([
            str(basalt_vio),
            "--dataset-path", str(dataset),
            "--dataset-type", "euroc",
            "--cam-calib", str(dataset / "calibration.json"),
            "--config-path", str(dataset / "vio_config.json"),
            "--show-gui", "0",
            "--save-trajectory", "euroc",
            "--use-imu", "1",
            "--num-threads", str(args.num_threads),
        ], cwd=dataset, env=basalt_env)
    else:
        print(f"Reuse Basalt trajectory: {trajectory}")

    if not trajectory.is_file():
        raise RuntimeError(f"Basalt did not create {trajectory}")
    if world_output.exists() and any(world_output.iterdir()):
        summary_path = world_output / "summary.json"
        if summary_path.is_file():
            with summary_path.open("r", encoding="utf-8") as stream:
                summary = json.load(stream)
            recorded_hand_pose = Path(summary.get("hand_pose_csv", "")).resolve()
            if recorded_hand_pose == hand_pose_csv:
                print(f"Reuse world-hand result: {world_output} ({summary.get('frames', '?')} frames)")
                return 0
            raise RuntimeError(
                "existing world-hand output was generated from a different MANO 6D CSV:\n"
                f"  existing: {recorded_hand_pose}\n"
                f"  requested: {hand_pose_csv}\n"
                "Choose a fresh --output-root so the tuned pose is not silently mixed "
                "with an older result."
            )
        raise RuntimeError(
            f"partial/non-empty world output: {world_output}. Choose a fresh --output-root."
        )

    command = [
        sys.executable, str(fuse_script),
        "--session", str(session),
        "--basalt-dataset", str(dataset),
        "--hand-pose-csv", str(hand_pose_csv),
        "--output", str(world_output),
        "--trajectory-length", str(args.trajectory_length),
    ]
    if args.no_video:
        command.append("--no-video")
    run(command)
    print(f"\nBasalt VIO result: {trajectory}")
    print(f"Camera world trajectory: {world_output / 'camera_trajectory_world.csv'}")
    print(f"Hand world trajectory: {world_output / 'hand_trajectory_world.csv'}")
    if not args.no_video:
        print(f"Preview video: {world_output / 'world_hand_trajectory_overlay.mp4'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
