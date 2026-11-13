"""Infer a shared hand model from calibrated Orbbec EGO four-camera recordings."""

from __future__ import annotations
import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
CAMERAS = tuple(f"camera{i}" for i in range(4))


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        epilog="Environment: --conda-env NAME (shell launcher, default ego-hand), EGO_HAND_PYTHON=/path/to/python. The WiLoR entry point always renders the final video.",
    )
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--input",
        type=Path,
        help="MP4/MCAP recording root, selected part directory, or Orbbec .mcap file",
    )
    source.add_argument(
        "--session", type=Path, help="four-MP4 recording or selected part directory"
    )
    source.add_argument(
        "--mcap",
        type=Path,
        help="Orbbec four-camera MCAP file or recording/part directory",
    )
    p.add_argument(
        "--calibration",
        type=Path,
        help="matching Orbbec camera YAML or directory with camera0..3.json; auto-detect sidecar YAML when omitted",
    )
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--device", default="cuda")
    p.add_argument("--detector", type=Path, default=ROOT / "models/wilor/detector.pt")
    p.add_argument("--checkpoint", type=Path, help="WiLoR checkpoint")
    p.add_argument(
        "--model-config", type=Path, default=ROOT / "models/wilor/model_config.yaml"
    )
    p.add_argument("--mano-model-dir", type=Path, default=ROOT / "models/mano")
    p.add_argument(
        "--gpu-profile", choices=("compatible", "rtx5090d"), default="compatible"
    )
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--frame-batch-size", type=int, default=1)
    p.add_argument("--preprocess-workers", type=int, default=1)
    p.add_argument("--max-detections-per-class", type=int, default=0)
    p.add_argument("--compile-backbone", type=int, choices=(0, 1), default=0)
    p.add_argument(
        "--camera-confidence", action="append", default=[], metavar="CAMERA=VALUE"
    )
    p.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="0 processes the complete recording (default)",
    )
    p.add_argument("--max-delta-us", type=int, default=1500)
    p.add_argument("--shared-steps", type=int, default=100)
    p.add_argument("--shared-batch-size", type=int, default=32)
    p.add_argument("--tile-width", type=int, default=960)
    p.add_argument(
        "--video-codec", choices=("auto", "h264_nvenc", "mp4v"), default="auto"
    )
    a = p.parse_args(argv)
    if (
        min(
            a.batch_size,
            a.frame_batch_size,
            a.preprocess_workers,
            a.shared_steps,
            a.shared_batch_size,
        )
        < 1
    ):
        p.error("batch sizes, workers and fitting steps must be positive")
    if min(a.max_frames, a.max_delta_us, a.max_detections_per_class) < 0:
        p.error(
            "frame/detection limits and synchronization tolerance must be non-negative"
        )
    if a.tile_width < 32 or a.tile_width % 2:
        p.error("tile width must be an even number >= 32")
    for value in a.camera_confidence:
        try:
            camera, score = value.split("=")
            if camera not in CAMERAS or not 0 < float(score) <= 1:
                raise ValueError
        except ValueError:
            p.error(
                "--camera-confidence requires camera0..3=VALUE with VALUE in (0, 1]"
            )
    for name in ("detector", "model_config", "mano_model_dir", "calibration"):
        value = getattr(a, name)
        if value is not None:
            setattr(a, name, value.expanduser().resolve())
    from ego_hand_inference.orbbec_io import resolve_recording_input

    try:
        a.input, a.kind = resolve_recording_input(
            a.input or a.session or a.mcap,
            "mp4" if a.session else "mcap" if a.mcap else None,
        )
    except (ValueError, FileNotFoundError) as error:
        p.error(str(error))
    a.output = a.output.expanduser().resolve()
    a.checkpoint = (
        (a.checkpoint or ROOT / "models/wilor/wilor_final.ckpt").expanduser().resolve()
    )
    return a


def calibration_source(a):
    if a.calibration:
        path = a.calibration.expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"calibration does not exist: {path}")
        return path
    from ego_hand_inference.orbbec_io import discover_calibration

    return discover_calibration(a.input)


def load_calibrations(source):
    from ego_hand_inference.ego_data.calibration import load_orbbec_four_calibrations

    return load_orbbec_four_calibrations(source)


def signature(path):
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    stat = path.stat()
    return {"path": str(path), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def run_stage(script, argv):
    print(f"[ego-hand] {script}", flush=True)
    environment = os.environ.copy()
    paths = [str(ROOT / "scripts"), environment.get("PYTHONPATH", "")]
    environment["PYTHONPATH"] = os.pathsep.join(path for path in paths if path)
    subprocess.run(
        [sys.executable, "-m", "ego_hand_inference." + Path(script).stem, *map(str, argv)],
        check=True,
        cwd=ROOT,
        env=environment,
    )


def prepare_dataset(a, calibration, normalized):
    if (normalized / "manifest.json").is_file():
        return
    if normalized.exists():
        raise RuntimeError(
            f"incomplete normalized dataset: {normalized}; use a new output directory"
        )
    if a.kind == "mp4":
        from ego_hand_inference.normalize_orbbec_ego4_session import normalize

        normalize(
            SimpleNamespace(
                session=a.input,
                output=normalized,
                calibration=calibration_source(a),
                reference_camera="camera0",
                max_delta_us=a.max_delta_us,
                max_frames=a.max_frames,
            )
        )
    else:
        from ego_hand_inference.normalize_orbbec_multiview_mcap import normalize

        staging = normalized.parent / "input_calibration"
        staging.mkdir(exist_ok=True)
        for camera, value in calibration.items():
            value.save(staging / f"{camera}.json")
        normalize(
            SimpleNamespace(
                input=a.input,
                output=normalized,
                cameras=list(CAMERAS),
                camera_topic=[],
                reference_camera="camera0",
                max_delta_us=a.max_delta_us,
                max_frames=a.max_frames,
                width=calibration["camera0"].image_size[0],
                height=calibration["camera0"].image_size[1],
                calibration_dir=staging,
                allow_truncated=False,
            )
        )
    manifest = json.loads((normalized / "manifest.json").read_text())
    if (
        manifest["camera_ids"] != list(CAMERAS)
        or not manifest["calibration"]["available"]
    ):
        raise ValueError("expected four calibrated Orbbec camera streams")
    import cv2

    for camera, value in calibration.items():
        saved = json.loads((normalized / "calibration" / f"{camera}.json").read_text())
        if saved != value.to_dict():
            raise ValueError(f"normalized calibration differs for {camera}")
        capture = cv2.VideoCapture(
            str(normalized / "cameras" / camera / manifest["storage"]["video_filename"])
        )
        try:
            ok, image = capture.read()
            if not ok or image.shape[1::-1] != value.image_size:
                raise ValueError(
                    f"{camera}: decoded video size differs from calibration"
                )
        finally:
            capture.release()


def main(argv=None):
    a = parse_args(argv)
    cal_source = calibration_source(a)
    calibration = load_calibrations(cal_source)
    if a.kind == "mp4":
        from ego_hand_inference.normalize_orbbec_ego4_session import CAMERA_SOURCES, session_stream_file

        source_files = [
            session_stream_file(a.input, stream, timestamps=pts)
            for stream, _ in CAMERA_SOURCES.values()
            for pts in (False, True)
        ]
    else:
        source_files = [a.input]
    from ego_hand_inference.orbbec_io import recording_metadata_root

    source_files += sorted(recording_metadata_root(a.input).glob("meta_*.json"))
    cal_files = (
        [cal_source / f"{c}.json" for c in CAMERAS]
        if cal_source.is_dir()
        else [cal_source]
    )
    model_files = [a.detector, a.checkpoint]
    model_files += [a.model_config, a.mano_model_dir / "MANO_RIGHT.pkl"]
    # Source identities prevent silently reusing a result produced by different code.
    code = sorted((ROOT / "scripts").rglob("*.py"))
    options = {
        k: str(v) if isinstance(v, Path) else v
        for k, v in vars(a).items()
        if k not in ("session", "mcap", "output")
    }
    config = {
        "schema_version": 1,
        "pipeline": "orbbec_ego4_shared_wilor_v1",
        "options": options,
        "input_files": [signature(p) for p in source_files],
        "calibration_files": [signature(p) for p in cal_files],
        "model_files": [signature(p) for p in model_files],
        "code_files": [signature(p) for p in code],
        "camera_ids": list(CAMERAS),
        "saved_categories": ["A", "B", "C", "D", "E", "G", "H"],
    }
    a.output.mkdir(parents=True, exist_ok=True)
    config_path = a.output / "run_config.json"
    if config_path.exists():
        if json.loads(config_path.read_text()) != config:
            raise RuntimeError(
                "input, model, code or options changed; use a new --output directory"
            )
    elif any(a.output.iterdir()):
        raise RuntimeError(
            "output is nonempty and has no matching run_config.json; use a new output directory"
        )
    else:
        config_path.write_text(json.dumps(config, indent=2) + "\n")
    cache = a.output / ".cache"
    summary_path = a.output / "summary.json"
    # The summary is written only after delivery files pass integrity checks.
    if summary_path.is_file():
        from ego_hand_inference.export_hand_results import validate_delivery

        validate_delivery(a.output)
        if cache.exists():
            shutil.rmtree(cache)
        print(
            f"[ego-hand] complete result already exists: {a.output / 'final_shared_4view.mp4'}"
        )
        return 0
    cache.mkdir(exist_ok=True)
    started = time.perf_counter()
    normalized, predictions, shared = [
        cache / name for name in ("normalized", "predictions", "shared")
    ]
    prepare_dataset(a, calibration, normalized)
    if not (predictions / "summary.json").is_file():
        if predictions.exists():
            # Retry only the owned, incomplete prediction stage.
            shutil.rmtree(predictions)
        infer = [
            "--dataset",
            normalized,
            "--output",
            predictions,
            "--detector",
            a.detector,
            "--checkpoint",
            a.checkpoint,
            "--device",
            a.device,
            "--max-frames",
            a.max_frames,
            "--gpu-profile",
            a.gpu_profile,
            "--batch-size",
            a.batch_size,
            "--frame-batch-size",
            a.frame_batch_size,
            "--preprocess-workers",
            a.preprocess_workers,
            "--max-detections-per-class",
            a.max_detections_per_class,
            "--compile-backbone",
            a.compile_backbone,
        ]
        overrides = {
            "camera0": "0.2",
            "camera1": "0.3",
            "camera2": "0.3",
            "camera3": "0.3",
        }
        overrides.update(dict((x.split("=") for x in a.camera_confidence)))
        for camera, value in overrides.items():
            infer += ["--camera-confidence", f"{camera}={value}"]
        infer += [
            "--model-config",
            a.model_config,
            "--mano-model-dir",
            a.mano_model_dir,
            "--save-mano-params",
        ]
        run_stage("wilor_multiview_inference.py", infer)
    if shared.exists() and (not (shared / "summary.json").is_file()):
        shutil.rmtree(shared)
    fit = [
        "--dataset",
        normalized,
        "--predictions",
        predictions,
        "--native-predictions",
        predictions,
        "--output",
        shared,
        "--cameras",
        *CAMERAS,
        "--device",
        a.device,
        "--max-frames",
        a.max_frames,
        "--steps",
        a.shared_steps,
        "--batch-size",
        a.shared_batch_size,
    ]
    fit += ["--mano-model-dir", a.mano_model_dir]
    run_stage("fit_shared_wilor_multiview.py", fit)
    video = shared / "final_shared_4view.mp4"
    render = [
        "--dataset",
        normalized,
        "--shared",
        shared,
        "--output",
        video,
        "--cameras",
        *CAMERAS,
        "--max-frames",
        a.max_frames,
        "--tile-width",
        a.tile_width,
        "--video-codec",
        a.video_codec,
    ]
    render += ["--detector-predictions", predictions]
    run_stage("render_shared_wilor_multiview.py", render)
    from ego_hand_inference.export_hand_results import export_results

    export_results(
        normalized, predictions, shared, video, a.output, time.perf_counter() - started
    )
    shutil.rmtree(cache)
    print(f"[ego-hand] saved A+B+C+D+E+G+H: {a.output}", flush=True)
    print(f"[ego-hand] final video: {a.output / 'final_shared_4view.mp4'}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        ValueError,
        RuntimeError,
        FileNotFoundError,
        subprocess.CalledProcessError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
