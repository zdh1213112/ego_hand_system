#!/usr/bin/env python3
"""Run dual-handedness WiLoR hypotheses on every camera in a multiview dataset."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import gc
import json
from pathlib import Path
import sys
import time
from typing import Any

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
import wilor_inference as base  # noqa: E402
from ego_data.dataset import SequentialVideoReader  # noqa: E402
from wilor.datasets.utils import (  # noqa: E402
    convert_cvimg_to_tensor,
    expand_to_aspect_ratio,
    generate_image_patch_cv2,
)


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
    parser.add_argument(
        "--gpu-profile", choices=("compatible", "rtx5090d"), default="compatible",
        help="compatible keeps the original FP32 path; rtx5090d batches frames and uses CUDA AMP",
    )
    parser.add_argument(
        "--frame-batch-size", type=int, default=0,
        help="frames sent to the batched detector at once (0 selects the profile default)",
    )
    parser.add_argument(
        "--preprocess-workers", type=int, default=0,
        help="parallel hand-crop workers (0 selects the profile default)",
    )
    parser.add_argument(
        "--max-detections-per-class", type=int, default=0,
        help="keep the top N detector boxes for each left/right class (0 keeps all)",
    )
    parser.add_argument(
        "--compile-backbone", type=int, choices=(0, 1), default=0,
        help="compile the WiLoR backbone with torch.compile (0 or 1)",
    )
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


class _FrameHypothesisDataset:
    """Attach the source frame number to one WiLoR crop dataset."""

    def __init__(self, dataset, frame_batch_index: int):
        self.dataset = dataset
        self.frame_batch_index = frame_batch_index

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        item = self.dataset[index]
        item["frame_batch_index"] = self.frame_batch_index
        return item


class _FastViTDetDataset(base.ViTDetDataset):
    """WiLoR crop preprocessing optimized for the batched GPU profile.

    The upstream dataset copies and filters the entire source image through
    scikit-image once per hypothesis. OpenCV's Gaussian implementation serves
    the same anti-aliasing purpose without the full-image float64 pipeline.
    """

    def __getitem__(self, index: int):
        center = self.center[index].copy()
        scale = self.scale[index]
        bbox_size = float(expand_to_aspect_ratio(
            scale * 200,
            target_aspect_ratio=self.cfg.MODEL.get("BBOX_SHAPE", None),
        ).max())
        image = self.img_cv2
        downsampling_factor = (bbox_size / self.img_size) / 2.0
        if downsampling_factor > 1.1:
            sigma = (downsampling_factor - 1.0) / 2.0
            image = cv2.GaussianBlur(
                image, (0, 0), sigmaX=sigma, sigmaY=sigma,
                borderType=cv2.BORDER_REPLICATE,
            )
        right = self.right[index].copy()
        image_patch, _ = generate_image_patch_cv2(
            image, float(center[0]), float(center[1]), bbox_size, bbox_size,
            self.img_size, self.img_size, bool(right == 0), 1.0, 0,
            border_mode=cv2.BORDER_CONSTANT,
        )
        image_patch = convert_cvimg_to_tensor(image_patch[:, :, ::-1])
        for channel in range(min(self.img_cv2.shape[2], 3)):
            image_patch[channel] = (
                image_patch[channel] - self.mean[channel]
            ) / self.std[channel]
        if self.fp16:
            import torch
            image_patch = torch.from_numpy(image_patch).half()
        return {
            "img": image_patch,
            "personid": int(self.personid[index]),
            "box_center": center,
            "box_size": bbox_size,
            "img_size": np.asarray(
                [self.img_cv2.shape[1], self.img_cv2.shape[0]], dtype=np.float32,
            ),
            "right": right,
        }


def _top_detections_per_class(
    confidences: np.ndarray, detector_sides: np.ndarray, limit: int,
) -> np.ndarray:
    if limit == 0:
        return np.arange(len(confidences), dtype=np.int64)
    selected = []
    for side in np.unique(detector_sides):
        candidates = np.flatnonzero(detector_sides == side)
        ranked = candidates[np.argsort(-confidences[candidates], kind="stable")]
        selected.extend(ranked[:limit].tolist())
    return np.asarray(sorted(selected), dtype=np.int64)


def infer_dual_batched(
    frames, detector, model, model_cfg, device, confidence: float,
    iou: float, rescale_factor: float, batch_size: int, preprocess_workers: int,
    max_detections_per_class: int,
) -> list[list[dict[str, Any]]]:
    """Run one detector batch and one cross-frame WiLoR hypothesis stream."""
    import torch

    detections = detector(
        frames, conf=confidence, iou=iou, verbose=False, half=True,
    )
    states: list[dict[str, Any]] = []
    datasets = []
    for frame_batch_index, (frame, detection) in enumerate(zip(frames, detections)):
        if detection.boxes is None or len(detection.boxes) == 0:
            states.append({"records": []})
            continue
        boxes = detection.boxes.xyxy.detach().cpu().numpy().astype(np.float32)
        confidences = detection.boxes.conf.detach().cpu().numpy().astype(np.float32)
        detector_sides = detection.boxes.cls.detach().cpu().numpy().astype(np.int8)
        selected = _top_detections_per_class(
            confidences, detector_sides, max_detections_per_class,
        )
        boxes = boxes[selected]
        confidences = confidences[selected]
        detector_sides = detector_sides[selected]
        hypothesis_boxes = np.repeat(boxes, 2, axis=0)
        hypothesis_sides = np.tile(
            np.asarray([0.0, 1.0], dtype=np.float32), len(boxes)
        )
        source_indices = np.repeat(np.arange(len(boxes), dtype=np.int32), 2)
        crop_dataset = _FastViTDetDataset(
            model_cfg, frame, hypothesis_boxes, hypothesis_sides,
            rescale_factor=rescale_factor, fp16=True,
        )
        states.append({
            "boxes": boxes,
            "confidences": confidences,
            "detector_sides": detector_sides,
            "source_indices": source_indices,
            "records": [None] * len(crop_dataset),
        })
        datasets.append(_FrameHypothesisDataset(crop_dataset, frame_batch_index))

    if not datasets:
        return [[] for _ in frames]

    combined = torch.utils.data.ConcatDataset(datasets)
    with ThreadPoolExecutor(max_workers=preprocess_workers) as executor:
        samples = list(executor.map(combined.__getitem__, range(len(combined))))
    for batch_start in range(0, len(samples), batch_size):
        batch = torch.utils.data.default_collate(
            samples[batch_start : batch_start + batch_size]
        )
        frame_indices = batch.pop("frame_batch_index").numpy()
        hypothesis_indices = batch["personid"].numpy()
        batch = base.recursive_to(batch, device)
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.float16,
        ):
            result = model(batch)
            multiplier = 2 * batch["right"] - 1
            pred_cam = result["pred_cam"].clone()
            pred_cam[:, 1] = multiplier * pred_cam[:, 1]
            image_size = batch["img_size"].float()
            focal_length = (
                model_cfg.EXTRA.FOCAL_LENGTH
                / model_cfg.MODEL.IMAGE_SIZE
                * image_size.max()
            ).item()
            camera_full = base.cam_crop_to_full(
                pred_cam, batch["box_center"].float(), batch["box_size"].float(),
                image_size, focal_length,
            )

        rights = batch["right"].detach().float().cpu().numpy()
        image_sizes = image_size.detach().cpu().numpy()
        cameras = camera_full.detach().float().cpu().numpy()
        joints = result["pred_keypoints_3d"].detach().float().cpu().numpy()
        joints[:, :, 0] *= (2 * rights - 1)[:, None]
        for index, (frame_batch_index, hypothesis_index) in enumerate(
            zip(frame_indices, hypothesis_indices)
        ):
            state = states[int(frame_batch_index)]
            detection_index = int(state["source_indices"][hypothesis_index])
            is_right = int(round(float(rights[index])))
            joints2d = base.project_points(
                joints[index], cameras[index], focal_length, image_sizes[index],
            )
            state["records"][int(hypothesis_index)] = {
                "detection_index": detection_index,
                "hypothesis_index": int(hypothesis_index),
                "bbox_xyxy": state["boxes"][detection_index].tolist(),
                "confidence": float(state["confidences"][detection_index]),
                "detector_is_right": int(state["detector_sides"][detection_index]),
                "is_right": is_right,
                "camera_translation": cameras[index].astype(np.float32).tolist(),
                "joints_2d": joints2d.tolist(),
            }

    return [
        [record for record in state["records"] if record is not None]
        for state in states
    ]


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"output already exists: {args.output}")
    if (
        args.batch_size < 1
        or args.frame_batch_size < 0
        or args.preprocess_workers < 0
        or args.max_detections_per_class < 0
        or args.max_frames < 0
    ):
        raise ValueError(
            "batch-size must be positive; frame-batch-size/preprocess-workers/"
            "max-detections-per-class/max-frames must be non-negative"
        )
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
    accelerated = args.gpu_profile == "rtx5090d"
    if accelerated and device.type != "cuda":
        raise RuntimeError("the rtx5090d profile requires a CUDA device")
    frame_batch_size = args.frame_batch_size or (4 if accelerated else 1)
    preprocess_workers = args.preprocess_workers or (8 if accelerated else 1)
    if accelerated:
        import torch
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    print(
        f"GPU profile: {args.gpu_profile} | frame batch: {frame_batch_size} | "
        f"hypothesis batch: {args.batch_size} | crop workers: {preprocess_workers} | "
        f"detections/class: {args.max_detections_per_class or 'all'} | "
        f"compiled backbone: {bool(args.compile_backbone)}",
        flush=True,
    )
    model, model_cfg = base.load_wilor_model(
        args.checkpoint, args.model_config, args.mano_model_dir
    )
    model = model.to(device).eval()
    if accelerated:
        model = model.half()
        if args.compile_backbone:
            import torch
            model.backbone = torch.compile(
                model.backbone, mode="reduce-overhead", dynamic=True,
            )
            print(
                "The first WiLoR batch compiles the backbone and can pause for "
                "about one minute; later batches reuse the compiled graph.",
                flush=True,
            )
    detector = base.load_detector(args.detector, device)
    camera_summaries = {}
    inference_started = time.perf_counter()
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
        camera_started = time.perf_counter()
        try:
            with prediction_path.open("w", encoding="utf-8") as stream:
                for chunk_start in range(0, len(rows), frame_batch_size):
                    chunk = rows[chunk_start : chunk_start + frame_batch_size]
                    frames = [
                        reader.read(int(row[f"{camera}_frame_index"])) for row in chunk
                    ]
                    if accelerated:
                        hands_by_frame = infer_dual_batched(
                            frames, detector, model, model_cfg, device,
                            camera_confidences[camera], args.iou,
                            args.rescale_factor, args.batch_size,
                            preprocess_workers, args.max_detections_per_class,
                        )
                    else:
                        hands_by_frame = [
                            infer_dual(
                                frame, detector, model, model_cfg, device,
                                camera_confidences[camera], args.iou,
                                args.rescale_factor, args.batch_size,
                            )
                            for frame in frames
                        ]
                    for offset, (row, hands) in enumerate(zip(chunk, hands_by_frame)):
                        ordinal = chunk_start + offset
                        frame_index = int(row[f"{camera}_frame_index"])
                        detection_ids = {
                            int(hand["detection_index"]) for hand in hands
                        }
                        physical_boxes += len(detection_ids)
                        detected_frames += bool(detection_ids)
                        record = {
                            "sync_index": int(row["sync_index"]),
                            "source_frame_index": frame_index,
                            "timestamp_ns": int(row[f"{camera}_timestamp_ns"]),
                            "hands": hands,
                        }
                        stream.write(
                            json.dumps(record, separators=(",", ":")) + "\n"
                        )
                        if not accelerated:
                            gc.collect()
                            if device.type == "cuda":
                                torch = __import__("torch")
                                torch.cuda.empty_cache()
                        if (ordinal + 1) % 10 == 0:
                            elapsed = time.perf_counter() - camera_started
                            print(
                                f"{camera}: {ordinal + 1}/{len(rows)} | "
                                f"{(ordinal + 1) / elapsed:.2f} frames/s",
                                flush=True,
                            )
                    del hands_by_frame, frames
        finally:
            reader.close()
        if accelerated:
            gc.collect()
            torch = __import__("torch")
            torch.cuda.empty_cache()
        camera_seconds = time.perf_counter() - camera_started
        camera_summaries[camera] = {
            "frame_count": len(rows), "detected_frame_count": detected_frames,
            "physical_box_count": physical_boxes,
            "detector_confidence": camera_confidences[camera],
            "inference_seconds": camera_seconds,
            "camera_frames_per_second": len(rows) / camera_seconds,
        }
        (camera_root / "summary.json").write_text(
            json.dumps(camera_summaries[camera], indent=2) + "\n", encoding="utf-8"
        )
    inference_seconds = time.perf_counter() - inference_started
    processed_camera_frames = len(rows) * len(cameras)
    summary = {
        "schema_version": 1,
        "stage": "wilor_multiview_dual_hypothesis",
        "dataset": str(dataset),
        "camera_ids": list(cameras),
        "frame_count": len(rows),
        "hypotheses_per_detection": 2,
        "gpu_profile": args.gpu_profile,
        "frame_batch_size": frame_batch_size,
        "hypothesis_batch_size": args.batch_size,
        "preprocess_workers": preprocess_workers,
        "max_detections_per_class": args.max_detections_per_class,
        "compiled_backbone": bool(args.compile_backbone),
        "inference_seconds": inference_seconds,
        "camera_frames_per_second": processed_camera_frames / inference_seconds,
        "cameras": camera_summaries,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
