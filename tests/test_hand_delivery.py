from pathlib import Path
import csv
import json
import sys
import cv2
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from ego_hand_inference.export_hand_results import export_results, validate_delivery
from ego_hand_inference.run_hand_inference import CAMERAS, load_calibrations, parse_args
from ego_hand_inference.ego_data.calibration import CameraCalibration
from ego_hand_inference.orbbec_io import discover_calibration


def test_delivery_keeps_selected_data_and_empty_frames_without_diagnostics(tmp_path):
    dataset, predictions, shared, output = [
        tmp_path / p for p in ("dataset", "predictions", "shared", "output")
    ]
    for p in (dataset, predictions, shared, output):
        p.mkdir()
    (dataset / "calibration").mkdir()
    manifest = {
        "camera_ids": list(CAMERAS),
        "coordinate_system": "test reference",
        "image_size": [64, 48],
    }
    (dataset / "manifest.json").write_text(json.dumps(manifest))
    fields = ["sync_index", "reference_timestamp_ns"] + [
        f"{c}_{key}"
        for c in CAMERAS
        for key in ("frame_index", "timestamp_ns", "delta_us")
    ]
    rows = []
    for index in range(2):
        row = {
            "sync_index": index,
            "reference_timestamp_ns": 1000000000 + index * 33333333,
        }
        for c in CAMERAS:
            row.update(
                {
                    f"{c}_frame_index": index,
                    f"{c}_timestamp_ns": row["reference_timestamp_ns"],
                    f"{c}_delta_us": 0,
                }
            )
        rows.append(row)
    with (dataset / "multiview_frames.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    for c in CAMERAS:
        CameraCalibration(
            c,
            c,
            "KB",
            (64, 48),
            [[40, 0, 32], [0, 40, 24], [0, 0, 1]],
            [0] * 4,
            np.eye(4),
        ).save(dataset / "calibration" / f"{c}.json")
        (dataset / "cameras" / c).mkdir(parents=True)
        (dataset / "cameras" / c / "timestamps.csv").write_text(
            "frame_index,timestamp_ns\n0,1000000000\n1,1033333333\n"
        )
        (predictions / c).mkdir()
        with (predictions / c / "predictions.jsonl").open("w") as f:
            for row in rows:
                f.write(
                    json.dumps(
                        {
                            "sync_index": row["sync_index"],
                            "source_frame_index": row["sync_index"],
                            "timestamp_ns": row["reference_timestamp_ns"],
                            "hands": [
                                {
                                    "detection_index": 0,
                                    "hypothesis_index": 1,
                                    "bbox_xyxy": [1, 2, 20, 30],
                                    "confidence": 0.9,
                                    "detector_is_right": 1,
                                    "is_right": 1,
                                    "joints_2d": np.ones((21, 2)).tolist(),
                                    "mano": {"raw": "excluded"},
                                    "camera_translation": [0, 0, 1],
                                }
                            ],
                        }
                    )
                    + "\n"
                )
    hand = {
        "side": 1,
        "vertices_base_m": torch.ones(778, 3),
        "parameters": {"hand_pose": torch.eye(3).repeat(15, 1, 1)},
        "rotation_base_from_model": torch.eye(3),
        "translation_base_m": torch.ones(3),
        "model_scale": 1.0,
        "views": {"camera0": {"median_px": 1.0}},
        "supporting_cameras": list(CAMERAS),
    }
    hand["joints_base_m"] = torch.ones(21, 3)
    torch.save(
        {
            "format": "shared_wilor_mano_v1",
            "coordinate_system": "test reference",
            "topology": {"right": torch.tensor([[0, 1, 2]])},
            "frames": [
                {"sync_index": 0, "hands": [hand]},
                {"sync_index": 1, "hands": []},
            ],
        },
        shared / "shared_pose.pt",
    )
    video = shared / "final_shared_4view.mp4"
    writer = cv2.VideoWriter(
        str(video), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (64, 48)
    )
    for _ in rows:
        writer.write(np.zeros((48, 64, 3), np.uint8))
    writer.release()
    metadata = {
        "frame_count": 2,
        "fps": 30.0,
        "codec": "mp4v",
        "video_size": [64, 48],
        "render_seconds": 0.1,
        "skeleton_and_mesh_share_model": True,
        "independent_view_fallback": False,
        "raw_images_without_undistortion": True,
    }
    video.with_suffix(".json").write_text(json.dumps(metadata))
    with video.with_suffix(".projections.jsonl").open("w") as f:
        for row in rows:
            f.write(
                json.dumps(
                    {
                        "sync_index": row["sync_index"],
                        "cameras": {
                            c: [
                                {
                                    "side": 1,
                                    "shared_model_id": "0:1",
                                    "joints_2d": [[1, 2]] * 21,
                                    "supporting_cameras": list(CAMERAS),
                                    "mesh_projection_sha256": "discarded",
                                }
                            ]
                            if row["sync_index"] == 0
                            else []
                            for c in CAMERAS
                        },
                    }
                )
                + "\n"
            )
    (output / "run_config.json").write_text("{}")
    export_results(dataset, predictions, shared, video, output, 0.5)
    summary = validate_delivery(output)
    assert summary["saved_categories"] == ["A", "B", "C", "D", "E", "G", "H"]
    assert summary["frame_count"] == 2 and summary["frames_with_hands"] == 1
    assert not (output / ".delivery").exists()
    assert not any(
        ("accepted" in p.name or "rejected" in p.name for p in output.rglob("*"))
    )
    joints = [
        json.loads(line)
        for line in (output / "joints3d.jsonl").read_text().splitlines()
    ]
    assert (
        joints[1]["hands"] == []
        and joints[1]["timestamp_ns"] == rows[1]["reference_timestamp_ns"]
    )
    model = torch.load(output / "hand_model.pt", weights_only=False)
    saved = model["frames"][0]["hands"][0]
    assert "views" not in saved and "supporting_cameras" not in saved
    assert saved["vertices_base_m"].shape == hand["vertices_base_m"].shape
    assert torch.equal(
        saved["parameters"]["hand_pose"], hand["parameters"]["hand_pose"]
    )
    observation = json.loads(
        (output / "observations_2d/camera0.jsonl").read_text().splitlines()[0]
    )
    assert (
        "mano" not in observation["hands"][0]
        and len(observation["hands"][0]["joints_2d"]) == 21
    )
    (output / "joints3d.jsonl").write_text("modified")
    with pytest.raises(ValueError, match="changed"):
        validate_delivery(output)


def test_recording_metadata_selects_new_calibration_and_missing_files_fail(tmp_path):
    root = tmp_path / "recording"
    part = root / "part0001"
    part.mkdir(parents=True)
    source = part / "recording_part0001.mcap"
    source.touch()
    cal = root / "recording_params_cam_1920x1200.yaml"
    cal.write_text("cameras: []")
    (root / "meta_recording.json").write_text(
        json.dumps({"files": {"calibration_camera": cal.name}})
    )
    assert discover_calibration(source) == cal
    cal.unlink()
    with pytest.raises(ValueError, match="missing"):
        discover_calibration(source)


def test_json_calibration_requires_explicit_extrinsics_and_baseline(tmp_path):
    for i, c in enumerate(CAMERAS):
        transform = np.eye(4)
        transform[0, 3] = i * 0.03
        CameraCalibration(
            c,
            c,
            "KB",
            (64, 48),
            [[40, 0, 32], [0, 40, 24], [0, 0, 1]],
            [0] * 4,
            transform,
        ).save(tmp_path / f"{c}.json")
    assert len(load_calibrations(tmp_path)) == 4
    path = tmp_path / "camera0.json"
    payload = json.loads(path.read_text())
    payload.pop("T_base_camera")
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="extrinsics"):
        load_calibrations(tmp_path)


@pytest.mark.parametrize(
    "flag,value",
    [
        ("--wilor-mode", "both"),
        ("--gem-mode", "independent"),
        ("--gem-root", "/unused"),
        ("--gem-python", "/unused/python"),
        ("--soma-assets", "/unused/assets"),
        ("--backend", "gem"),
        ("--final-renderer", "hybrid"),
        ("--no-video", None),
        ("--final-video", None),
    ],
)
def test_removed_modes_are_rejected(tmp_path, flag, value):
    argv = ["--session", str(tmp_path), "--output", str(tmp_path / "out"), flag]
    if value:
        argv.append(value)
    with pytest.raises(SystemExit) as error:
        parse_args(argv)
    assert error.value.code == 2


def test_relative_asset_paths_are_absolute(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    args = parse_args(
        ["--session", ".", "--output", "out", "--detector", "detector.pt"]
    )
    assert args.detector == tmp_path / "detector.pt"


def test_pipeline_retries_failed_stage_cleans_cache_and_reuses_delivery(
    tmp_path, monkeypatch
):
    import subprocess
    from ego_hand_inference import run_hand_inference as pipeline

    session = tmp_path / "recording"
    session.mkdir()
    calibration = tmp_path / "calibration"
    calibration.mkdir()
    models = tmp_path / "models"
    models.mkdir()
    for name in ("detector.pt", "checkpoint.pt", "config.yaml", "MANO_RIGHT.pkl"):
        (models / name).touch()
    for i, camera in enumerate(CAMERAS):
        stream = ("left", "right", "right_side", "left_side")[i]
        writer = cv2.VideoWriter(
            str(session / f"recording_camera_{stream}.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            30.0,
            (64, 48),
        )
        for _ in range(2):
            writer.write(np.zeros((48, 64, 3), np.uint8))
        writer.release()
        (session / f"recording_camera_{stream}_pts.csv").write_text(
            "timestamp_ns\n1000000000\n1033333333\n"
        )
        transform = np.eye(4)
        transform[0, 3] = i * 0.03
        CameraCalibration(
            camera,
            camera,
            "KB",
            (64, 48),
            [[40, 0, 32], [0, 40, 24], [0, 0, 1]],
            [0] * 4,
            transform,
        ).save(calibration / f"{camera}.json")
    argv = [
        "--session",
        str(session),
        "--calibration",
        str(calibration),
        "--output",
        str(tmp_path / "result"),
        "--detector",
        str(models / "detector.pt"),
        "--checkpoint",
        str(models / "checkpoint.pt"),
        "--model-config",
        str(models / "config.yaml"),
        "--mano-model-dir",
        str(models),
        "--device",
        "cpu",
        "--video-codec",
        "mp4v",
        "--tile-width",
        "64",
    ]
    original_run = pipeline.run_stage
    attempts = []
    fail_once = [True]

    def stages(script, argv, python=None):
        args = list(map(str, argv))
        output = Path(args[args.index("--output") + 1])
        dataset = Path(args[args.index("--dataset") + 1])
        attempts.append(script)
        if script == "wilor_multiview_inference.py":
            output.mkdir()
            if fail_once[0]:
                fail_once[0] = False
                raise subprocess.CalledProcessError(1, ["fake-inference"])
            with (dataset / "multiview_frames.csv").open() as f:
                rows = list(csv.DictReader(f))
            for c in CAMERAS:
                (output / c).mkdir()
                with (output / c / "predictions.jsonl").open("w") as f:
                    for row in rows:
                        f.write(
                            json.dumps(
                                {
                                    "sync_index": int(row["sync_index"]),
                                    "source_frame_index": int(row[f"{c}_frame_index"]),
                                    "timestamp_ns": int(row[f"{c}_timestamp_ns"]),
                                    "hands": [],
                                }
                            )
                            + "\n"
                        )
            (output / "summary.json").write_text(
                json.dumps(
                    {
                        "dataset": str(dataset),
                        "camera_ids": list(CAMERAS),
                        "frame_count": 2,
                        "cameras": {c: {"physical_box_count": 0} for c in CAMERAS},
                    }
                )
            )
        elif script == "fit_shared_wilor_multiview.py":
            output.mkdir()
            (output / "summary.json").write_text(
                json.dumps({"camera_ids": list(CAMERAS)})
            )
            torch.save(
                {
                    "format": "shared_wilor_mano_v1",
                    "coordinate_system": "test",
                    "topology": {},
                    "frames": [
                        {"sync_index": 0, "hands": []},
                        {"sync_index": 1, "hands": []},
                    ],
                },
                output / "shared_pose.pt",
            )
        else:
            original_run(script, argv)

    monkeypatch.setattr(pipeline, "run_stage", stages)
    with pytest.raises(subprocess.CalledProcessError):
        pipeline.main(argv)
    output = tmp_path / "result"
    assert (output / ".cache/normalized/manifest.json").exists()
    assert not (output / "summary.json").exists()
    assert pipeline.main(argv) == 0
    assert not (output / ".cache").exists()
    assert validate_delivery(output)["frame_count"] == 2
    count = len(attempts)
    assert pipeline.main(argv) == 0 and len(attempts) == count
    with pytest.raises(RuntimeError, match="changed"):
        pipeline.main(argv + ["--max-frames", "1"])


def test_recording_root_detects_formats_and_rejects_ambiguous_parts(tmp_path):
    from ego_hand_inference.orbbec_io import resolve_recording_input

    part = tmp_path / "part0001"
    part.mkdir()
    mcap = part / "recording_part0001.mcap"
    mcap.touch()
    assert resolve_recording_input(tmp_path) == (mcap, "mcap")
    assert resolve_recording_input(part) == (mcap, "mcap")
    (part / "recording_camera_left_part0001.mp4").touch()
    with pytest.raises(ValueError, match="MP4 and MCAP"):
        resolve_recording_input(tmp_path)
    assert resolve_recording_input(tmp_path, "mp4") == (tmp_path, "mp4")
    assert resolve_recording_input(tmp_path, "mcap") == (mcap, "mcap")
    (part / "recording_camera_left_part0001.mp4").unlink()
    other = tmp_path / "part0002"
    other.mkdir()
    (other / "other.mcap").touch()
    with pytest.raises(ValueError, match="specific part"):
        resolve_recording_input(tmp_path)
