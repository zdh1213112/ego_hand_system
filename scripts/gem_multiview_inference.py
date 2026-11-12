#!/usr/bin/env python3
"""Run GEM-X on original multiview frames using WiLoR detector hypotheses.

The native .pt files retain SOMA mesh, topology, parameters, and all 25 joints.
Raw fisheye images are passed directly to GEM with a pinhole approximation;
the fusion stage still uses each camera's original calibrated lens model.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time
import sys

import numpy as np

# SOMA has an extra metacarpal joint (Index1 etc.) for each non-thumb finger.
# WiLoR's 21-point layout starts those fingers at MCP, followed by PIP, DIP, tip.
SOMA_TO_21 = (0, 1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14,
              16, 17, 18, 19, 21, 22, 23, 24)


def adapt_frame(frame: dict, sync: dict, camera: str, proposal_source: str = "gem_rtdetr") -> dict:
    hands = []
    for index, instance in enumerate(frame["instances"]):
        side = instance["side"]
        if side not in ("left", "right"):
            raise ValueError(f"invalid GEM hand side: {side!r}")
        points = np.asarray(instance["keypoints_2d"], dtype=float)
        if points.shape != (25, 2):
            raise ValueError(f"expected 25 SOMA image joints, got {points.shape}")
        selected = points[list(SOMA_TO_21)]
        if not np.isfinite(selected).all():
            continue
        if "projection_valid" in instance:
            valid = np.asarray(instance["projection_valid"], dtype=bool)
            if valid.shape != (25,):
                raise ValueError(f"expected 25 projection-valid flags, got {valid.shape}")
            if not valid[list(SOMA_TO_21)].all():
                continue
        bbox = np.asarray(instance["bbox"], dtype=float)
        if bbox.shape != (3,) or not np.isfinite(bbox).all() or bbox[2] <= 0:
            raise ValueError("GEM bbox must be finite [center_x, center_y, size]")
        x, y, size = bbox
        right = int(side == "right")
        score = float(instance.get("detection_score", 1.0))
        if not np.isfinite(score):
            raise ValueError("non-finite detector confidence")
        hands.append({
            "detection_index": index, "hypothesis_index": index,
            "bbox_xyxy": [x-size/2, y-size/2, x+size/2, y+size/2],
            "confidence": score, "detector_is_right": right,
            "is_right": right, "proposal_source": proposal_source,
            "joints_2d": selected.tolist(),
        })
    return {"sync_index": int(sync["sync_index"]),
            "source_frame_index": int(sync[f"{camera}_frame_index"]),
            "timestamp_ns": int(sync[f"{camera}_timestamp_ns"]), "hands": hands}



def make_hypotheses(boxes, confidences, sides, rescale_factor=2.0):
    """Keep detector boxes unchanged while making the baseline's two side hypotheses."""
    instances = []
    for index, (box, score, side) in enumerate(zip(boxes, confidences, sides)):
        x1, y1, x2, y2 = map(float, box)
        # WiLoR ViTDetDataset expands to BBOX_SHAPE=(192,256), then takes max.
        crop_size = rescale_factor * max(y2 - y1, (x2 - x1) / 0.75)
        if not np.isfinite([x1, y1, x2, y2, score, crop_size]).all() or crop_size <= 0:
            raise ValueError(f"invalid detector box {index}: {box}")
        for is_right in (0, 1):
            instances.append({"track_id": 2 * index + is_right,
                              "side": "right" if is_right else "left",
                              "bbox": [(x1 + x2) / 2, (y1 + y2) / 2, crop_size],
                              "detection_score": float(score)})
    return instances


def serialize_hypotheses(frame, sync, camera, boxes, confidences, sides):
    hands = []
    for instance in frame["instances"]:
        hypothesis = int(instance["track_id"])
        detection = hypothesis // 2
        right = hypothesis % 2
        if instance["side"] != ("right" if right else "left"):
            raise ValueError("GEM side does not match hypothesis index")
        points = np.asarray(instance["keypoints_2d"], dtype=float)
        valid = np.asarray(instance["projection_valid"], dtype=bool)
        if points.shape != (25, 2) or valid.shape != (25,):
            raise ValueError("GEM must produce 25 projected SOMA joints")
        selected = points[list(SOMA_TO_21)]
        if not np.isfinite(selected).all() or not valid[list(SOMA_TO_21)].all():
            continue
        detector_side = int(sides[detection])
        hands.append({"detection_index": detection, "hypothesis_index": hypothesis,
                      "bbox_xyxy": np.asarray(boxes[detection], dtype=float).tolist(),
                      "confidence": float(confidences[detection]),
                      "detector_is_right": detector_side if detector_side in (0, 1) else None,
                      "proposal_source": "detector", "is_right": right,
                      "camera_translation": np.asarray(instance["camera_translation"], dtype=float).tolist(),
                      "joints_2d": selected.tolist()})
    return {"sync_index": int(sync["sync_index"]),
            "source_frame_index": int(sync[f"{camera}_frame_index"]),
            "timestamp_ns": int(sync[f"{camera}_timestamp_ns"]), "hands": hands}


def predict_in_batches(predictor, image, frame_id, instances, intrinsics, topology, batch_size, *, amp=False):
    """Run every hypothesis, reducing batch size on CUDA memory pressure."""
    import torch
    from gem.inference.pose import predict_frame
    result = {"frame_id": frame_id, "image_size": [image.shape[1], image.shape[0]],
              "intrinsics": torch.as_tensor(intrinsics, dtype=torch.float32), "instances": []}
    position = 0
    while position < len(instances):
        current = min(batch_size, len(instances) - position)
        while True:
            try:
                # The accelerated GPU profile opts into tensor-core inference.
                # GEM's predictor/geometry explicitly disable autocast for
                # camera decoding and SOMA FK, retaining physical FP32 output.
                with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                    part = predict_frame(predictor, image, frame_id,
                        {"instances": instances[position:position + current], "intrinsics": intrinsics},
                        task="hand", topology=topology)
                break
            except torch.cuda.OutOfMemoryError:
                if current == 1:
                    raise
                torch.cuda.empty_cache()
                current = max(1, current // 2)
                batch_size = current
                print(f"[gem] CUDA memory pressure: reducing pose batch to {current}", flush=True)
        result["instances"].extend(part["instances"])
        position += current
    return result


def iter_detected_frames(reader, rows, camera, detector, confidence, accelerated, frame_batch_size):
    """Use the baseline's detector options and frame batching for each profile."""
    import torch
    step = frame_batch_size if accelerated else 1
    for start in range(0, len(rows), step):
        chunk = rows[start:start + step]
        images = [reader.read(int(row[f"{camera}_frame_index"])) for row in chunk]
        with torch.inference_mode():
            if accelerated:
                detections = detector(images, conf=confidence, iou=0.5, verbose=False, half=True)
            else:
                detections = [detector(images[0], conf=confidence, iou=0.5, verbose=False)[0]]
        if len(detections) != len(images):
            raise RuntimeError(f"{camera}: detector returned an unexpected frame count")
        yield from zip(chunk, images, detections)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset", "output", "gem-root", "checkpoint", "detector", "soma-assets"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gpu-profile", choices=("compatible", "rtx5090d"), default="compatible")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--frame-batch-size", type=int, default=1)
    parser.add_argument("--preprocess-workers", type=int, default=1)
    parser.add_argument("--max-detections-per-class", type=int, default=0)
    parser.add_argument("--compile-backbone", type=int, choices=(0, 1), default=0)
    parser.add_argument("--camera-confidence", action="append", default=[])
    parser.add_argument("--max-frames", type=int, default=0)
    args = parser.parse_args()
    if (args.batch_size < 1 or args.frame_batch_size < 1 or args.preprocess_workers < 1
            or args.max_detections_per_class < 0 or args.max_frames < 0):
        parser.error("invalid batch, worker, detection limit or frame count")
    dataset, output = args.dataset.resolve(), args.output.resolve()
    manifest = json.loads((dataset / "manifest.json").read_text())
    if manifest.get("dataset_type") != "normalized_multiview":
        raise ValueError("expected normalized multiview dataset")
    with (dataset / "multiview_frames.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))[:args.max_frames or None]
    if not rows:
        raise ValueError("no synchronized frames")
    cameras = manifest["camera_ids"]
    confidences_by_camera = {camera: 0.3 for camera in cameras}
    for value in args.camera_confidence:
        camera, value = value.split("=", 1)
        if camera not in confidences_by_camera or not 0 < float(value) <= 1:
            raise ValueError(f"invalid camera confidence: {camera}={value}")
        confidences_by_camera[camera] = float(value)
    sys.path.insert(0, str(args.gem_root.resolve()))
    import cv2
    import torch
    import wilor_inference as base
    from ego_data.dataset import SequentialVideoReader
    from gem.inference.pose import load_predictor
    from gem.inference.serialization import save_predictions
    from wilor_multiview_inference import _top_detections_per_class
    device = base.choose_device(args.device)
    accelerated = args.gpu_profile == "rtx5090d"
    if accelerated and device.type != "cuda":
        raise RuntimeError("the rtx5090d profile requires CUDA")
    print("[gem] original pixels: pinhole intrinsics approximation; fusion uses original lens calibration", flush=True)
    if accelerated:
        print("[gem] FP16 autocast backbone/head; camera and SOMA geometry remain FP32", flush=True)
    detector = base.load_detector(args.detector, device)
    predictor, release = load_predictor(args.checkpoint, task="hand", release="hand",
                                         soma_assets=args.soma_assets, device=device)
    if args.compile_backbone:
        predictor.model.backbone = torch.compile(predictor.model.backbone, mode="reduce-overhead", dynamic=True)
        print("[gem] backbone compilation enabled; the first batch may take longer", flush=True)
    output.mkdir(parents=True, exist_ok=False)
    summaries = {}
    started = time.perf_counter()
    for camera in cameras:
        camera_start = time.perf_counter()
        calibration = json.loads((dataset / "calibration" / f"{camera}.json").read_text())
        intrinsics = np.asarray(calibration["K"], dtype=np.float32)
        if intrinsics.shape != (3, 3) or not np.isfinite(intrinsics).all():
            raise ValueError(f"{camera}: invalid camera intrinsics")
        root = output / camera
        root.mkdir()
        video = dataset / "cameras" / camera / manifest["storage"]["video_filename"]
        reader = SequentialVideoReader(video, tuple(manifest["image_size"]))
        source_frame_count = int(reader.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        topology, native_frames = {}, []
        detected_frames = physical_boxes = 0
        try:
            with (root / "predictions.jsonl").open("w", encoding="utf-8") as stream:
                for ordinal, (row, image, detection) in enumerate(iter_detected_frames(
                    reader, rows, camera, detector, confidences_by_camera[camera],
                    accelerated, args.frame_batch_size,
                )):
                    frame_id = int(row[f"{camera}_frame_index"])
                    if detection.boxes is None or len(detection.boxes) == 0:
                        boxes = np.empty((0, 4), dtype=np.float32)
                        scores = np.empty(0, dtype=np.float32)
                        sides = np.empty(0, dtype=np.int8)
                    else:
                        boxes = detection.boxes.xyxy.detach().cpu().numpy().astype(np.float32)
                        scores = detection.boxes.conf.detach().cpu().numpy().astype(np.float32)
                        sides = detection.boxes.cls.detach().cpu().numpy().astype(np.int8)
                        if accelerated:
                            selected = _top_detections_per_class(scores, sides, args.max_detections_per_class)
                            boxes, scores, sides = boxes[selected], scores[selected], sides[selected]
                    instances = make_hypotheses(boxes, scores, sides)
                    native = predict_in_batches(predictor, cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                                                frame_id, instances, intrinsics, topology, args.batch_size,
                                                amp=accelerated)
                    converted = serialize_hypotheses(native, row, camera, boxes, scores, sides)
                    stream.write(json.dumps(converted, allow_nan=False, separators=(",", ":")) + "\n")
                    native_frames.append(native)
                    physical_boxes += len(boxes)
                    detected_frames += bool(len(boxes))
                    if (ordinal + 1) % 10 == 0:
                        print(f"[gem] {camera}: {ordinal + 1}/{len(rows)}", flush=True)
        finally:
            reader.close()
        native_path = root / "gem_pose.pt"
        save_predictions({"format": "gem_pose_predictions_v1", "task": "hand", "mode": "framewise_video",
            "source": str(video), "source_fps": manifest.get("fps"), "source_frame_count": source_frame_count,
            "release": release, "coordinates": {"geometry": "camera_meters", "keypoints_2d": "full_image_pixels"},
            "camera_adapter": {"raw_fisheye_pinhole_approximation": True, "source_camera_model": calibration["model"]},
            "topology": topology, "frames": native_frames}, native_path)
        summary = {"frame_count": len(rows), "detected_frame_count": detected_frames,
                   "physical_box_count": physical_boxes, "detector_confidence": confidences_by_camera[camera],
                   "inference_seconds": time.perf_counter() - camera_start, "native_result": str(native_path)}
        (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        summaries[camera] = summary
        del native_frames
    (output / "summary.json").write_text(json.dumps({
        "schema_version": 1, "stage": "gem_multiview_dual_hypothesis", "dataset": str(dataset),
        "camera_ids": cameras, "frame_count": len(rows), "hypotheses_per_detection": 2,
        "gem_root": str(args.gem_root.resolve()), "checkpoint": str(args.checkpoint.resolve()),
        "detector": str(args.detector.resolve()), "soma_assets": str(args.soma_assets.resolve()),
        "device": str(device), "gpu_profile": args.gpu_profile,
        "pose_precision": "fp16_backbone_head_fp32_geometry" if accelerated else "fp32",
        "compiled_backbone": bool(args.compile_backbone),
        "camera_model_for_pose": "pinhole_approximation_on_original_pixels",
        "soma_to_21_indices": SOMA_TO_21, "inference_seconds": time.perf_counter() - started,
        "cameras": summaries}, indent=2) + "\n")


if __name__ == "__main__":
    main()
