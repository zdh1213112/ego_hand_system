#!/usr/bin/env python3
"""Run dual-handedness WiLoR hypotheses on every camera in a multiview dataset."""

from __future__ import annotations

import argparse
import csv
import gc
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
import wilor_inference as base  # noqa: E402
from ego_data.dataset import SequentialVideoReader  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT / "models/wilor/wilor_final.ckpt")
    parser.add_argument("--model-config", type=Path, default=PROJECT_ROOT / "models/wilor/model_config.yaml")
    parser.add_argument("--detector", type=Path, default=PROJECT_ROOT / "models/wilor/detector.pt")
    parser.add_argument("--mano-model-dir", type=Path, default=PROJECT_ROOT / "models/mano")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--confidence", type=float, default=0.3)
    parser.add_argument(
        "--camera-confidence", action="append", default=[], metavar="CAMERA=VALUE",
        help="override detector confidence for one camera; may be repeated",
    )
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--rescale-factor", type=float, default=2.0)
    parser.add_argument("--max-frames", type=int, default=0)
    return parser.parse_args()


def infer_dual(
    frame, detector, model, model_cfg, device, confidence: float,
    iou: float, rescale_factor: float, batch_size: int,
) -> list[dict[str, Any]]:
    import torch

    detection = detector(frame, conf=confidence, iou=iou, verbose=False)[0]
    if detection.boxes is None or len(detection.boxes) == 0:
        return []
    boxes = detection.boxes.xyxy.detach().cpu().numpy().astype(np.float32)
    confidences = detection.boxes.conf.detach().cpu().numpy().astype(np.float32)
    detector_sides = detection.boxes.cls.detach().cpu().numpy().astype(np.int8)
    hypothesis_boxes = np.repeat(boxes, 2, axis=0)
    hypothesis_sides = np.tile(np.asarray([0.0, 1.0], dtype=np.float32), len(boxes))
    source_indices = np.repeat(np.arange(len(boxes), dtype=np.int32), 2)
    dataset = base.ViTDetDataset(
        model_cfg, frame, hypothesis_boxes, hypothesis_sides,
        rescale_factor=rescale_factor, fp16=False,
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    records: list[dict[str, Any] | None] = [None] * len(dataset)
    for batch in loader:
        batch = base.recursive_to(batch, device)
        with torch.inference_mode():
            result = model(batch)
        multiplier = 2 * batch["right"] - 1
        pred_cam = result["pred_cam"].clone()
        pred_cam[:, 1] = multiplier * pred_cam[:, 1]
        image_size = batch["img_size"].float()
        focal_length = (
            model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * image_size.max()
        ).item()
        camera_full = base.cam_crop_to_full(
            pred_cam, batch["box_center"].float(), batch["box_size"].float(),
            image_size, focal_length,
        ).detach().cpu().numpy()
        for index in range(batch["img"].shape[0]):
            hypothesis_index = int(batch["personid"][index].item())
            detection_index = int(source_indices[hypothesis_index])
            is_right = int(round(float(batch["right"][index].item())))
            mirror = 2 * is_right - 1
            joints3d = result["pred_keypoints_3d"][index].detach().float().cpu().numpy()
            joints3d[:, 0] *= mirror
            camera = camera_full[index].astype(np.float32)
            joints2d = base.project_points(
                joints3d, camera, focal_length,
                image_size[index].detach().cpu().numpy(),
            )
            records[hypothesis_index] = {
                "detection_index": detection_index,
                "hypothesis_index": hypothesis_index,
                "bbox_xyxy": boxes[detection_index].tolist(),
                "confidence": float(confidences[detection_index]),
                "detector_is_right": int(detector_sides[detection_index]),
                "is_right": is_right,
                "camera_translation": camera.tolist(),
                "joints_2d": joints2d.tolist(),
            }
    return [record for record in records if record is not None]


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"output already exists: {args.output}")
    if args.batch_size < 1 or args.max_frames < 0:
        raise ValueError("batch-size must be positive and max-frames non-negative")
    dataset = args.dataset.resolve()
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("dataset_type") != "normalized_multiview":
        raise ValueError(f"not a normalized multiview dataset: {dataset}")
    with (dataset / "multiview_frames.csv").open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    rows = rows[: args.max_frames or None]
    if not rows:
        raise ValueError("multiview dataset contains no selected frames")
    cameras = tuple(manifest["camera_ids"])
    camera_confidences = {camera: args.confidence for camera in cameras}
    for value in args.camera_confidence:
        try:
            camera, confidence_text = value.split("=", 1)
            confidence = float(confidence_text)
        except ValueError as error:
            raise ValueError(f"invalid --camera-confidence {value!r}; expected CAMERA=VALUE") from error
        if camera not in camera_confidences:
            raise ValueError(f"unknown camera in --camera-confidence: {camera}")
        if not 0.0 < confidence <= 1.0:
            raise ValueError(f"camera confidence must be in (0, 1], got {confidence}")
        camera_confidences[camera] = confidence
    image_size = tuple(manifest["image_size"])
    output = args.output.resolve()
    output.mkdir(parents=True)
    device = base.choose_device(args.device)
    model, model_cfg = base.load_wilor_model(
        args.checkpoint, args.model_config, args.mano_model_dir
    )
    model = model.to(device).eval()
    detector = base.load_detector(args.detector, device)
    camera_summaries = {}
    for camera in cameras:
        camera_root = output / camera
        camera_root.mkdir()
        prediction_path = camera_root / "predictions.jsonl"
        reader = SequentialVideoReader(
            dataset / "cameras" / camera / manifest["storage"]["video_filename"],
            image_size,
        )
        physical_boxes = 0
        detected_frames = 0
        try:
            with prediction_path.open("w", encoding="utf-8") as stream:
                for ordinal, row in enumerate(rows):
                    frame_index = int(row[f"{camera}_frame_index"])
                    frame = reader.read(frame_index)
                    hands = infer_dual(
                        frame, detector, model, model_cfg, device,
                        camera_confidences[camera], args.iou,
                        args.rescale_factor, args.batch_size,
                    )
                    detections = {int(hand["detection_index"]) for hand in hands}
                    physical_boxes += len(detections)
                    detected_frames += bool(detections)
                    record = {
                        "sync_index": int(row["sync_index"]),
                        "source_frame_index": frame_index,
                        "timestamp_ns": int(row[f"{camera}_timestamp_ns"]),
                        "hands": hands,
                    }
                    stream.write(json.dumps(record, separators=(",", ":")) + "\n")
                    del hands, record, frame
                    gc.collect()
                    if device.type == "cuda":
                        torch = __import__("torch")
                        torch.cuda.empty_cache()
                    if (ordinal + 1) % 10 == 0:
                        print(f"{camera}: {ordinal + 1}/{len(rows)}", flush=True)
        finally:
            reader.close()
        camera_summaries[camera] = {
            "frame_count": len(rows), "detected_frame_count": detected_frames,
            "physical_box_count": physical_boxes,
            "detector_confidence": camera_confidences[camera],
        }
        (camera_root / "summary.json").write_text(
            json.dumps(camera_summaries[camera], indent=2) + "\n", encoding="utf-8"
        )
    summary = {
        "schema_version": 1,
        "stage": "wilor_multiview_dual_hypothesis",
        "dataset": str(dataset),
        "camera_ids": list(cameras),
        "frame_count": len(rows),
        "hypotheses_per_detection": 2,
        "cameras": camera_summaries,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
