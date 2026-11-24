"""Export exactly A+B+C+D+E+G+H, excluding diagnostic reports and caches."""

from __future__ import annotations
import csv
import json
from pathlib import Path
import shutil
import torch
from ego_hand_inference.orbbec_io import sha256_file

CAMERAS = tuple((f"camera{i}" for i in range(4)))
OBSERVATION_KEYS = (
    "detection_index",
    "hypothesis_index",
    "bbox_xyxy",
    "confidence",
    "detector_is_right",
    "is_right",
    "joints_2d",
)
MODEL_KEYS = (
    "side",
    "joints_base_m",
    "vertices_base_m",
    "parameters",
    "rotation_base_from_model",
    "translation_base_m",
    "model_scale",
    "mano_convention",
)


def write_json(path, value):
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def validate_delivery(output):
    """Reject incomplete or modified deliveries before treating a rerun as complete."""
    output = Path(output).resolve()
    summary = json.loads((output / "summary.json").read_text())
    if summary.get("status") != "complete" or summary.get("saved_categories") != [
        "A",
        "B",
        "C",
        "D",
        "E",
        "G",
        "H",
    ]:
        raise ValueError("incomplete or incompatible delivery summary")
    required = {
        "final_shared_4view.mp4",
        "joints3d.jsonl",
        "hand_model.pt",
        "projections_2d.jsonl",
        "manifest.json",
        "multiview_frames.csv",
        "run_config.json",
        "video.json",
        *(f"calibration/{c}.json" for c in CAMERAS),
        *(f"observations_2d/{c}.jsonl" for c in CAMERAS),
        *(f"timestamps/{c}.csv" for c in CAMERAS),
    }
    if not required <= set(summary.get("artifacts", {})):
        raise ValueError("delivery summary is missing required artifacts")
    for name, recorded in summary["artifacts"].items():
        path = (output / name).resolve()
        if not path.is_relative_to(output) or not path.is_file():
            raise ValueError(f"missing delivery artifact: {name}")
        if (
            path.stat().st_size != recorded["size_bytes"]
            or sha256_file(path) != recorded["sha256"]
        ):
            raise ValueError(f"delivery artifact changed: {name}")
    return summary


def export_results(dataset, predictions, shared, video, output, elapsed_seconds):
    dataset, predictions, shared, video, output = map(
        Path, (dataset, predictions, shared, video, output)
    )
    manifest = json.loads((dataset / "manifest.json").read_text())
    with (dataset / "multiview_frames.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if manifest["camera_ids"] != list(CAMERAS) or not rows:
        raise ValueError("expected nonempty synchronized Orbbec four-camera dataset")
    data = torch.load(
        shared / "shared_pose.pt", weights_only=False, map_location="cpu", mmap=True
    )
    expected_format = "shared_wilor_mano_v1"
    if data["format"] != expected_format:
        raise ValueError("expected shared WiLoR MANO model")
    frames = {int(frame["sync_index"]): frame for frame in data["frames"]}
    if len(frames) != len(rows) or len(frames) != len(data["frames"]):
        raise ValueError("shared model frame coverage differs from synchronized input")
    stage = output / ".delivery"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir()
    for folder in ("observations_2d", "calibration", "timestamps"):
        (stage / folder).mkdir()
    shutil.copy2(output / "run_config.json", stage / "run_config.json")
    shutil.copy2(dataset / "multiview_frames.csv", stage / "multiview_frames.csv")
    write_json(stage / "manifest.json", manifest)
    if (dataset / "source_meta.json").is_file():
        shutil.copy2(dataset / "source_meta.json", stage / "source_meta.json")
    model_frames = []
    accepted_frames = accepted_hands = 0
    with (stage / "joints3d.jsonl").open("w") as stream:
        for row in rows:
            sync = int(row["sync_index"])
            model_frame = frames[sync]
            timestamp = int(row["reference_timestamp_ns"])
            clean_hands, joints_hands = ([], [])
            sides = set()
            for hand in model_frame["hands"]:
                side = int(hand["side"])
                if side not in (0, 1) or side in sides:
                    raise ValueError(f"duplicate or invalid hand side at frame {sync}")
                sides.add(side)
                clean = {key: hand[key] for key in MODEL_KEYS if key in hand}
                points = hand["joints_base_m"]
                points = torch.as_tensor(points)
                if points.shape != (21, 3) or not torch.isfinite(points).all():
                    raise ValueError(f"invalid shared joints at frame {sync}")
                clean_hands.append(clean)
                joints_hand = {
                    "side": side,
                    "handedness": "right" if side else "left",
                    "joints_base_m": points.tolist(),
                }
                joints_hands.append(joints_hand)
            model_frames.append(
                {"sync_index": sync, "timestamp_ns": timestamp, "hands": clean_hands}
            )
            accepted_frames += bool(joints_hands)
            accepted_hands += len(joints_hands)
            stream.write(
                json.dumps(
                    {
                        "sync_index": sync,
                        "timestamp_ns": timestamp,
                        "hands": joints_hands,
                    },
                    allow_nan=False,
                )
                + "\n"
            )
    model = {
        key: value
        for key, value in data.items()
        if key in ("format", "coordinate_system", "mano_convention", "topology")
    }
    model.update(
        {
            "export_schema": "ego_hand_model_v1",
            "backend": "wilor",
            "units": "metres",
            "frames": model_frames,
        }
    )
    torch.save(model, stage / "hand_model.pt")
    for camera in CAMERAS:
        shutil.copy2(
            dataset / "calibration" / f"{camera}.json",
            stage / "calibration" / f"{camera}.json",
        )
        shutil.copy2(
            dataset / "cameras" / camera / "timestamps.csv",
            stage / "timestamps" / f"{camera}.csv",
        )
        with (
            (predictions / camera / "predictions.jsonl").open() as source,
            (stage / "observations_2d" / f"{camera}.jsonl").open("w") as dest,
        ):
            for row in rows:
                record = json.loads(next(source))
                if (
                    int(record["sync_index"]) != int(row["sync_index"])
                    or int(record["timestamp_ns"]) != int(row[f"{camera}_timestamp_ns"])
                    or int(record["source_frame_index"])
                    != int(row[f"{camera}_frame_index"])
                ):
                    raise ValueError(f"{camera}: observation frame/timestamp mismatch")
                clean = {
                    key: record[key]
                    for key in ("sync_index", "source_frame_index", "timestamp_ns")
                }
                clean["hands"] = [
                    {key: hand[key] for key in OBSERVATION_KEYS if key in hand}
                    for hand in record["hands"]
                ]
                dest.write(json.dumps(clean, allow_nan=False) + "\n")
            if source.read().strip():
                raise ValueError(f"{camera}: unexpected extra observation frames")
    with (
        video.with_suffix(".projections.jsonl").open() as source,
        (stage / "projections_2d.jsonl").open("w") as dest,
    ):
        for row in rows:
            record = json.loads(next(source))
            if int(record["sync_index"]) != int(row["sync_index"]):
                raise ValueError("projection frame mismatch")
            clean = {
                "sync_index": int(row["sync_index"]),
                "timestamp_ns": int(row["reference_timestamp_ns"]),
                "cameras": {},
            }
            for camera in CAMERAS:
                clean["cameras"][camera] = [
                    {key: hand[key] for key in ("side", "shared_model_id", "joints_2d")}
                    for hand in record["cameras"][camera]
                ]
            dest.write(json.dumps(clean, allow_nan=False) + "\n")
        if source.read().strip():
            raise ValueError("unexpected extra projection frames")
    video_meta = json.loads(video.with_suffix(".json").read_text())
    if int(video_meta["frame_count"]) != len(rows):
        raise ValueError("rendered video frame count differs from synchronized input")
    import cv2

    capture = cv2.VideoCapture(str(video))
    try:
        ok, image = capture.read()
        if (
            not ok
            or int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) != len(rows)
            or list(image.shape[1::-1]) != video_meta["video_size"]
        ):
            raise ValueError(
                "final video is unreadable or has an unexpected frame count/size"
            )
    finally:
        capture.release()
    write_json(
        stage / "video.json",
        {
            key: video_meta[key]
            for key in (
                "frame_count",
                "fps",
                "codec",
                "video_size",
                "render_seconds",
                "skeleton_and_mesh_share_model",
                "independent_view_fallback",
                "raw_images_without_undistortion",
            )
        },
    )
    shutil.copy2(video, stage / "final_shared_4view.mp4")
    artifacts = {
        str(p.relative_to(stage)): {
            "size_bytes": p.stat().st_size,
            "sha256": sha256_file(p),
        }
        for p in sorted(stage.rglob("*"))
        if p.is_file()
    }
    summary = {
        "schema_version": 1,
        "status": "complete",
        "backend": "wilor",
        "saved_categories": ["A", "B", "C", "D", "E", "G", "H"],
        "camera_ids": list(CAMERAS),
        "frame_count": len(rows),
        "frames_with_hands": accepted_frames,
        "hand_count": accepted_hands,
        "coordinate_system": data["coordinate_system"],
        "units": "metres",
        "joint_order": "wrist; thumb, index, middle, ring, little (base to tip), 21 joints",
        "hand_sides": {"0": "left", "1": "right"},
        "elapsed_seconds": elapsed_seconds,
        "skeleton_and_mesh_share_model": True,
        "independent_view_fallback": False,
        "artifacts": artifacts,
    }
    write_json(stage / "summary.json", summary)
    validate_delivery(stage)
    for path in list(stage.iterdir()):
        if path.name == "summary.json":
            continue
        target = output / path.name
        if path.is_dir() and target.exists():
            shutil.rmtree(target)
        path.replace(target)
    (stage / "summary.json").replace(output / "summary.json")
    stage.rmdir()
    validate_delivery(output)
