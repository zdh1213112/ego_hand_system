#!/usr/bin/env python3
"""Adapt local FAH pose/MANO inference to the existing synchronized video pipeline.

The detector, original image pixels and synchronization table are retained. Each
box gets both handedness hypotheses, as in the WiLoR experiment. FAH is used
framewise so the existing multiview temporal processing remains unchanged.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import csv
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


def training_root():
    default = Path(__file__).resolve().parents[1] / "training"
    return Path(os.environ.get("WILOR_TRAINING_ROOT", default)).resolve()


def geometry_sources():
    directory = training_root() / "tools/hand_finetune"
    return [directory / name for name in ("crop_geometry.py", "fah_views.py")]


def predict_camera_hands(method, image, boxes, scores, side_groups, K, kb_camera, export_parameters):
    """Run and fit in the network camera, then return joints and fitted mesh in the physical camera."""
    import torch

    views = None
    with torch.no_grad():
        if kb_camera is None:
            counts = [len(group) for group in side_groups]
            hands = method.est.process_image(image, boxes=np.repeat(boxes, counts, axis=0),
                is_right=np.asarray([s for group in side_groups for s in group], bool),
                scores=np.repeat(scores, counts), cam_int=np.asarray(K, np.float32)) if len(boxes) else []
        else:
            if image.shape[1::-1] != kb_camera.image_size:
                raise ValueError(f"{kb_camera.camera_id}: video frame and KB calibration sizes differ")
            path = str(training_root())
            if path not in sys.path:
                sys.path.insert(0, path)
            try:
                from tools.hand_finetune.fah_views import native_prediction, predict_views
            except ModuleNotFoundError as exc:
                if exc.name not in ("tools", "tools.hand_finetune", "tools.hand_finetune.fah_views"):
                    raise
                raise ImportError(f"Check the integrated training module or set WILOR_TRAINING_ROOT to a compatible checkout (tried {path})") from exc
            hands, views = predict_views(method.est, image, boxes, scores, side_groups,
                                         kb_camera.K, kb_camera.distortion)

    if export_parameters:
        parameters = fit_meshes(method, hands)
    else:
        method._fit_meshes(hands)
        parameters = [None] * len(hands)
    if views is not None:
        for i, (hand, view) in enumerate(zip(hands, views, strict=True)):
            hands[i], parameters[i] = native_prediction(hand, view, parameters[i])
    return hands, parameters


def backend_config(root, model, weights, mano_dir, sam3d):
    """Fingerprint inference assets without importing FAH or loading weights."""
    cfg = json.loads((weights / model / "config.json").read_text())
    paths = [weights / model / "config.json", weights / model / cfg["model"],
             weights / model / cfg["mesh_fitter"], mano_dir / "MANO_RIGHT.pkl",
             sam3d / "model_config.yaml", sam3d / "assets/mhr_model.pt",
             Path(__file__), root / "bench/methods/fah.py"]
    paths.extend(sorted((root / "fah").glob("*.py")))
    paths.extend(path for path in geometry_sources() if path.is_file())
    signatures = []
    for path in paths:
        path = path.resolve()
        stat = path.stat()
        signatures.append({"path": str(path), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    return {"name": "fah", "model": model, "fah_root": str(root.resolve()),
            "mode": "framewise_dual_hypothesis", "mesh": "mano", "assets": signatures,
            "camera_model_for_pose": "KB_local_perspective_other_native_FAH"}


def load_external_boxes(root, cameras, rows):
    result = {}
    for camera in cameras:
        path = root / camera / "predictions.jsonl"
        records = {}
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            index = int(record["sync_index"])
            if index in records:
                raise ValueError(f"{path}: duplicate sync_index {index}")
            records[index] = record
        for sync in rows:
            record = records[int(sync["sync_index"])]
            for key, expected in (("source_frame_index", sync[f"{camera}_frame_index"]),
                                  ("timestamp_ns", sync[f"{camera}_timestamp_ns"])):
                if key in record and int(record[key]) != int(expected):
                    raise ValueError(f"{camera}: external boxes {key} mismatch")
        result[camera] = records
    return result


def external_detections(record):
    boxes, scores, sides, sources = [], [], [], []
    for item in record.get("boxes", []):
        box = np.asarray(item.get("bbox_xyxy", item.get("box", [])), dtype=np.float32)
        score = float(item.get("confidence", item.get("score", 0)))
        if (box.shape != (4,) or not np.isfinite(box).all() or not np.isfinite(score)
                or np.any(box[2:] <= box[:2])):
            raise ValueError("invalid external detector box/score")
        side = item.get("detector_is_right", item.get("is_right", -1))
        boxes.append(box)
        scores.append(np.clip(score, 0, 1))
        sides.append(int(side) if side in (0, 1) else -1)
        sources.append(str(item.get("source", "external")))
    return (np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
            np.asarray(scores, dtype=np.float32), np.asarray(sides, dtype=np.int8), sources)


def select_detections(boxes, scores, sides, sources, limit):
    selected = np.arange(len(boxes))
    if limit:
        selected = np.asarray(sorted(i for side in np.unique(sides)
            for i in np.flatnonzero(sides == side)[np.argsort(-scores[sides == side], kind="stable")][:limit]),
            dtype=np.int64)
    return boxes[selected], scores[selected], sides[selected], [sources[i] for i in selected]


def fit_meshes(method, hands):
    """Run FAH's own MANO fit and retain its rotations/shape for shared fitting.

    MeshStage exposes the GN result in graphs[B].fs. Fit at most two hands at
    once, matching FAH's internal chunk size, and copy those buffers before
    the next call reuses them. Pixel-aligned native vertices are saved separately;
    the rigid MANO parameters are only the shared optimizer's initialization.
    """
    import cv2

    parameters = []
    for start in range(0, len(hands), 2):
        chunk = hands[start:start + 2]
        if any(np.shape(h.get("vertices_mano")) != (778, 3) or h.get("mesh_param") for h in chunk):
            raise ValueError("expected FAH dense 778-vertex predictions before MANO fitting")
        method._fit_meshes(chunk)
        stage = getattr(method, "_stage", None)
        if stage is None or len(chunk) not in stage.graphs:
            raise RuntimeError("unsupported FAH MeshStage API: cannot export fitted MANO parameters")
        state = stage.graphs[len(chunk)][0]["fs"]
        root = state["out_R"].detach().float().cpu().numpy().copy()
        fitted = state["out_x"].detach().float().cpu().numpy().copy()
        for index in range(len(chunk)):
            pose = np.stack([cv2.Rodrigues(v)[0] for v in fitted[index, :45].reshape(15, 3)])
            parameters.append({"global_orient": root[index:index + 1].tolist(),
                               "hand_pose": pose.tolist(), "betas": fitted[index, 45:55].tolist()})
    return parameters


def adapt_hands(hands, parameters, boxes, scores, sides, sources):
    if len(hands) != 2 * len(boxes) or len(parameters) != len(hands):
        raise ValueError("FAH must return both hypotheses for every detector box")
    records, meshes = [], []
    for index, (hand, mano) in enumerate(zip(hands, parameters)):
        detection, side = index // 2, index % 2
        if int(hand["is_right"]) != side:
            raise ValueError("FAH changed the handedness hypothesis order")
        points = np.asarray(hand.get("keypoints_2d_mano", hand["keypoints_2d"]), dtype=np.float32)
        joints = np.asarray(hand.get("keypoints_3d_mano", hand["keypoints_3d"]), dtype=np.float32)
        mesh = np.asarray(hand["vertices_mano"], dtype=np.float32)
        for name, value, shape in (("joints2d", points, (21, 2)), ("joints3d", joints, (21, 3)),
                                   ("verts3d", mesh, (778, 3))):
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f"invalid FAH {name}: {value.shape}")
        for name, value in mano.items():
            if not np.isfinite(value).all():
                raise ValueError(f"invalid FAH MANO {name}")
        records.append({"detection_index": detection, "hypothesis_index": index,
            "bbox_xyxy": boxes[detection].tolist(), "confidence": float(scores[detection]),
            "detector_is_right": int(sides[detection]) if sides[detection] in (0, 1) else None,
            "proposal_source": sources[detection], "is_right": side,
            "camera_translation": joints[0].tolist(), "joints_2d": points.tolist(),
            "joints_3d_camera_relative": (joints - joints[0]).tolist(), "mano": mano,
            "mano_parameter_source": "fah_native_fit_before_pixel_alignment",
            "pose_backend": "fah"})
        meshes.append(mesh.copy())
    return records, meshes


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset", "output", "fah-root", "weights", "mano-model-dir", "sam3d", "detector"):
        p.add_argument("--" + name, required=True, type=Path)
    p.add_argument("--model", choices=("fah-s", "fah-b", "fah", "fah-h"), default="fah-s")
    p.add_argument("--external-boxes", type=Path)
    p.add_argument("--device", default="cuda")
    p.add_argument("--gpu-profile", choices=("compatible", "rtx5090d"), default="compatible")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--frame-batch-size", type=int, default=1)
    p.add_argument("--preprocess-workers", type=int, default=1)
    p.add_argument("--max-detections-per-class", type=int, default=0)
    p.add_argument("--compile-backbone", type=int, choices=(0, 1), default=0)
    p.add_argument("--camera-confidence", action="append", default=[])
    p.add_argument("--max-frames", type=int, default=0)
    args = p.parse_args()
    if (min(args.batch_size, args.frame_batch_size, args.preprocess_workers) < 1
            or min(args.max_detections_per_class, args.max_frames) < 0):
        p.error("invalid batch, worker, detection limit or frame count")
    return args


def main():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    args = parse_args()
    dataset, output = args.dataset.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    manifest = json.loads((dataset / "manifest.json").read_text())
    if manifest.get("dataset_type") != "normalized_multiview":
        raise ValueError("expected normalized multiview video dataset")
    with (dataset / "multiview_frames.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))[:args.max_frames or None]
    if not rows:
        raise ValueError("no synchronized frames")
    cameras = manifest["camera_ids"]
    confidences = {camera: .3 for camera in cameras}
    for setting in args.camera_confidence:
        camera, value = setting.split("=", 1)
        if camera not in confidences or not 0 < float(value) <= 1:
            raise ValueError(f"invalid camera confidence: {setting}")
        confidences[camera] = float(value)
    external = load_external_boxes(args.external_boxes.resolve(), cameras, rows) if args.external_boxes else None
    backend = backend_config(args.fah_root, args.model, args.weights, args.mano_model_dir, args.sam3d)

    sys.path.insert(0, str(args.fah_root.resolve()))
    import cv2
    import torch
    import fah
    from ego_hand_inference.ego_data.calibration import CameraCalibration
    from ego_data.dataset import SequentialVideoReader

    device = "cuda" if args.device == "auto" else args.device
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("FAH requested CUDA, but it is unavailable; no CPU fallback")
    if args.gpu_profile == "rtx5090d" and not str(device).startswith("cuda"):
        raise RuntimeError("the rtx5090d profile requires CUDA")
    cv2.setNumThreads(args.preprocess_workers)
    # FAH normally constructs YOLO even when the estimator is given boxes.
    # Suppress that constructor for recorded proposals, as the original runner does.
    detector_context = patch("bench.methods.fah.build_hand_detector",
        return_value=SimpleNamespace(names={0: "left", 1: "right"}, conf=.3)) if external is not None else nullcontext()
    with detector_context:
        model = fah.FAH(args.model, mode="image", weights=str(args.weights.resolve()),
            mano_dir=str(args.mano_model_dir.resolve()), sam3d=str(args.sam3d.resolve()),
            device=device, mesh="mano", tips=False, parts=False, verify=False,
            detector_weights=str(args.detector.resolve()), fast_det=False, det_iou=.5,
            compile=bool(args.compile_backbone), parts_in_mask=False)
    method = model.method
    method.est.max_batch_size = args.batch_size
    detector = method.detector
    if external is None and detector.names != {0: "left", 1: "right"}:
        raise ValueError(f"detector classes must be 0=left, 1=right; got {detector.names}")
    faces = np.asarray(model.faces, dtype=np.int32)
    output.mkdir(parents=True)
    started, summaries = time.perf_counter(), {}
    for camera in cameras:
        camera_start = time.perf_counter()
        calibration = json.loads((dataset / "calibration" / f"{camera}.json").read_text())
        K = np.asarray(calibration["K"], dtype=np.float32)
        if K.shape != (3, 3) or not np.isfinite(K).all() or np.any(np.diag(K)[:2] <= 0):
            raise ValueError(f"{camera}: invalid intrinsics")
        kb_camera = None
        mode = "native_pinhole" if calibration.get("model", "PINHOLE").upper() == "PINHOLE" else "pinhole_approximation"
        if calibration.get("model", "").upper() == "KB":
            kb_camera = CameraCalibration.from_dict(calibration)
            if kb_camera.image_size != tuple(manifest["image_size"]):
                raise ValueError(f"{camera}: manifest and KB calibration image sizes differ")
            mode = "KB_local_perspective"
        model.reset()
        detector.conf = confidences[camera]
        root = output / camera
        root.mkdir()
        video = dataset / "cameras" / camera / manifest["storage"]["video_filename"]
        reader = SequentialVideoReader(video, tuple(manifest["image_size"]))
        meshes, mesh_frames, mesh_hypotheses, mesh_sides = [], [], [], []
        physical_boxes = detected_frames = 0
        try:
            with (root / "predictions.jsonl").open("w") as stream:
                for ordinal, row in enumerate(rows):
                    frame_id = int(row[f"{camera}_frame_index"])
                    image = reader.read(frame_id)
                    if external is None:
                        boxes, scores, sides = detector(image)
                        sources = ["detector"] * len(boxes)
                    else:
                        boxes, scores, sides, sources = external_detections(external[camera][int(row["sync_index"])])
                    boxes, scores, sides, sources = select_detections(
                        np.asarray(boxes, np.float32).reshape(-1, 4), np.asarray(scores, np.float32),
                        np.asarray(sides, np.int8), sources, args.max_detections_per_class)
                    hands, parameters = predict_camera_hands(method, image, boxes, scores,
                        [(False, True)] * len(boxes), K, kb_camera, export_parameters=True)
                    converted, vertices = adapt_hands(hands, parameters, boxes, scores, sides, sources)
                    record = {"sync_index": int(row["sync_index"]), "camera_id": camera,
                        "source_frame_index": frame_id, "timestamp_ns": int(row[f"{camera}_timestamp_ns"]),
                        "hands": converted}
                    stream.write(json.dumps(record, allow_nan=False, separators=(",", ":")) + "\n")
                    meshes.extend(vertices)
                    mesh_frames.extend([frame_id] * len(vertices))
                    mesh_hypotheses.extend(h["hypothesis_index"] for h in converted)
                    mesh_sides.extend(h["is_right"] for h in converted)
                    physical_boxes += len(boxes)
                    detected_frames += bool(len(boxes))
                    if (ordinal + 1) % 10 == 0:
                        print(f"[fah] {camera}: {ordinal + 1}/{len(rows)}", flush=True)
        finally:
            reader.close()
        np.savez_compressed(root / "meshes.npz", vertices_camera_m=np.asarray(meshes, np.float32).reshape(-1, 778, 3),
            source_frame_indices=np.asarray(mesh_frames, np.int64), hypothesis_indices=np.asarray(mesh_hypotheses, np.int32),
            is_right=np.asarray(mesh_sides, np.int8), faces_right=faces, K=K,
            **({"kb_calibration_json": json.dumps(kb_camera.to_dict())} if kb_camera else {}))
        summary = {"frame_count": len(rows), "detected_frame_count": detected_frames,
            "physical_box_count": physical_boxes, "hypothesis_count": len(meshes),
            "detector_confidence": confidences[camera], "intrinsics": K.tolist(),
            "camera_model_for_pose": mode, "inference_seconds": time.perf_counter() - camera_start}
        (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        summaries[camera] = summary
    summary = {"schema_version": 1, "stage": "fah_multiview_dual_hypothesis", "dataset": str(dataset),
        "backend": backend, "camera_ids": cameras, "frame_count": len(rows), "save_mano_params": True,
        "native_mesh": "camera*/meshes.npz: camera coordinates in metres; right-hand triangle winding",
        "image_preprocessing": "detector on source video; KB hand boxes rectified to local perspective", "gpu_profile": args.gpu_profile,
        "batch_size": args.batch_size, "frame_batch_size": 1,
        "requested_frame_batch_size": args.frame_batch_size,
        "camera_model_for_pose": {camera: s["camera_model_for_pose"] for camera, s in summaries.items()},
        "inference_seconds": time.perf_counter() - started, "cameras": summaries}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
