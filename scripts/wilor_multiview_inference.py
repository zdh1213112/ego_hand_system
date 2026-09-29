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
from types import SimpleNamespace
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
    parser.add_argument(
        "--external-boxes", type=Path, default=None,
        help=(
            "optional proposal tree (root/camera/predictions.jsonl). When set, "
            "these boxes are sent to WiLoR instead of calling YOLO; each row "
            "is keyed by sync_index and contains a boxes list"
        ),
    )
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


def _external_detection(row: dict[str, Any] | None):
    """Build the tiny Ultralytics-like object needed by the crop path.

    SAM3 proposals intentionally have no handedness class.  They are marked
    ``-1`` and WiLoR still evaluates both handedness hypotheses for every
    proposal, exactly like the normal dual-hypothesis detector path.
    """
    import torch

    boxes = [] if row is None else row.get("boxes", [])
    xyxy: list[list[float]] = []
    confidence: list[float] = []
    sides: list[float] = []
    sources: list[str] = []
    for item in boxes:
        if not isinstance(item, dict):
            continue
        box = item.get("bbox_xyxy", item.get("box", []))
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            continue
        try:
            values = [float(value) for value in box]
            score = float(item.get("confidence", item.get("score", 0.0)))
            side = item.get("detector_is_right", item.get("is_right", -1))
            side = float(side) if side in (0, 1, 0.0, 1.0) else -1.0
        except (TypeError, ValueError):
            continue
        if not np.isfinite(values).all() or not np.isfinite(score):
            continue
        if values[2] <= values[0] or values[3] <= values[1]:
            continue
        xyxy.append(values)
        confidence.append(max(0.0, min(1.0, score)))
        sides.append(side)
        sources.append(str(item.get("source", "external")))
    class _Boxes(SimpleNamespace):
        def __len__(self) -> int:
            return int(self.xyxy.shape[0])

    return SimpleNamespace(
        proposal_sources=sources,
        boxes=_Boxes(
            xyxy=torch.as_tensor(xyxy, dtype=torch.float32).reshape(-1, 4),
            conf=torch.as_tensor(confidence, dtype=torch.float32),
            cls=torch.as_tensor(sides, dtype=torch.float32),
        )
    )


def _load_external_boxes(
    root: Path, cameras: tuple[str, ...], sync_rows: list[dict[str, str]],
) -> dict[str, list[dict[str, Any]]]:
    """Load proposal rows and align them to the normalized sync table."""
    result: dict[str, list[dict[str, Any]]] = {}
    for camera in cameras:
        path = root / camera / "predictions.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"external proposal file is missing: {path}")
        by_sync: dict[int, dict[str, Any]] = {}
        with path.open("r", encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                if "sync_index" not in row:
                    continue
                by_sync[int(row["sync_index"])] = row
        missing = [
            int(row["sync_index"]) for row in sync_rows
            if int(row["sync_index"]) not in by_sync
        ]
        if missing:
            raise ValueError(
                f"external proposals for {camera} do not cover "
                f"{len(missing)} synchronized frames (first={missing[:3]})"
            )
        result[camera] = [by_sync[int(row["sync_index"])] for row in sync_rows]
    return result


def _json_detector_side(value: Any) -> int | None:
    """Keep unknown proposal handedness explicit in the JSON contract."""
    try:
        numeric = int(round(float(value)))
    except (TypeError, ValueError):
        return None
    return numeric if numeric in (0, 1) else None


def infer_dual(
    frame, detector, model, model_cfg, device, confidence: float,
    iou: float, rescale_factor: float, batch_size: int,
    external_boxes: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    import torch

    detection = (
        _external_detection(external_boxes)
        if external_boxes is not None
        else detector(frame, conf=confidence, iou=iou, verbose=False)[0]
    )
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
                "detector_is_right": _json_detector_side(
                    detector_sides[detection_index]
                ),
                "proposal_source": getattr(detection, "proposal_sources", ["detector"] * len(boxes))[
                    detection_index
                ],
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


def _prepare_dual_batched(
    frames, detector, model_cfg, confidence: float,
    iou: float, rescale_factor: float, preprocess_workers: int,
    max_detections_per_class: int,
    preprocess_executor: ThreadPoolExecutor | None = None,
    external_boxes: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run detection and CPU crop preparation for one frame chunk."""
    import torch

    if external_boxes is None:
        with torch.inference_mode():
            detections = detector(
                frames, conf=confidence, iou=iou, verbose=False, half=True,
            )
    else:
        if len(external_boxes) != len(frames):
            raise ValueError(
                "external proposal chunk length does not match decoded frames: "
                f"{len(external_boxes)} != {len(frames)}"
            )
        detections = [_external_detection(row) for row in external_boxes]
    states: list[dict[str, Any]] = []
    datasets = []
    for frame_batch_index, (frame, detection) in enumerate(zip(frames, detections)):
        if detection.boxes is None or len(detection.boxes) == 0:
            states.append({"records": []})
            continue
        boxes = detection.boxes.xyxy.detach().cpu().numpy().astype(np.float32)
        confidences = detection.boxes.conf.detach().cpu().numpy().astype(np.float32)
        detector_sides = detection.boxes.cls.detach().cpu().numpy().astype(np.int8)
        proposal_sources = list(
            getattr(detection, "proposal_sources", ["detector"] * len(boxes))
        )
        selected = _top_detections_per_class(
            confidences, detector_sides, max_detections_per_class,
        )
        boxes = boxes[selected]
        confidences = confidences[selected]
        detector_sides = detector_sides[selected]
        proposal_sources = [proposal_sources[int(index)] for index in selected]
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
            "proposal_sources": proposal_sources,
            "source_indices": source_indices,
            "records": [None] * len(crop_dataset),
        })
        datasets.append(_FrameHypothesisDataset(crop_dataset, frame_batch_index))

    if datasets:
        combined = torch.utils.data.ConcatDataset(datasets)
        if preprocess_executor is None:
            with ThreadPoolExecutor(max_workers=preprocess_workers) as executor:
                samples = list(
                    executor.map(combined.__getitem__, range(len(combined)))
                )
        else:
            samples = list(preprocess_executor.map(
                combined.__getitem__, range(len(combined))
            ))
    else:
        samples = []
    return {"states": states, "samples": samples}


def _infer_prepared_batched(
    prepared: dict[str, Any], model, model_cfg, device, batch_size: int,
    runtime_stats: dict[str, int] | None = None,
) -> list[list[dict[str, Any]]]:
    """Run WiLoR on crops that may have been prepared beside a prior batch."""
    import torch

    states = prepared["states"]
    samples = prepared["samples"]
    if not samples:
        return [[] for _ in states]
    effective_batch_size = (
        int(runtime_stats.get("effective_hypothesis_batch_size", batch_size))
        if runtime_stats is not None else batch_size
    )
    batch_start = 0
    while batch_start < len(samples):
        batch_end = min(batch_start + effective_batch_size, len(samples))
        batch = torch.utils.data.default_collate(
            samples[batch_start:batch_end]
        )
        frame_indices = batch.pop("frame_batch_index").numpy()
        hypothesis_indices = batch["personid"].numpy()
        try:
            # Pinned host tensors let CUDA use its asynchronous copy engine.
            batch = {
                key: (
                    value.pin_memory().to(device, non_blocking=True)
                    if isinstance(value, torch.Tensor) else value
                )
                for key, value in batch.items()
            }
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
                    pred_cam, batch["box_center"].float(),
                    batch["box_size"].float(), image_size, focal_length,
                )
        except torch.cuda.OutOfMemoryError:
            if effective_batch_size <= 1:
                raise
            effective_batch_size = max(1, effective_batch_size // 2)
            if runtime_stats is not None:
                runtime_stats["effective_hypothesis_batch_size"] = (
                    effective_batch_size
                )
                runtime_stats["cuda_oom_retries"] = (
                    int(runtime_stats.get("cuda_oom_retries", 0)) + 1
                )
            del batch
            gc.collect()
            torch.cuda.empty_cache()
            print(
                "CUDA memory pressure: retrying WiLoR with hypothesis batch "
                f"{effective_batch_size}",
                flush=True,
            )
            continue

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
                "detector_is_right": _json_detector_side(
                    state["detector_sides"][detection_index]
                ),
                "proposal_source": state.get("proposal_sources", ["detector"] * len(state["boxes"]))[
                    detection_index
                ],
                "is_right": is_right,
                "camera_translation": cameras[index].astype(np.float32).tolist(),
                "joints_2d": joints2d.tolist(),
            }
        batch_start = batch_end

    return [
        [record for record in state["records"] if record is not None]
        for state in states
    ]


def infer_dual_batched(
    frames, detector, model, model_cfg, device, confidence: float,
    iou: float, rescale_factor: float, batch_size: int, preprocess_workers: int,
    max_detections_per_class: int,
    preprocess_executor: ThreadPoolExecutor | None = None,
    runtime_stats: dict[str, int] | None = None,
    external_boxes: list[dict[str, Any]] | None = None,
) -> list[list[dict[str, Any]]]:
    """Run a complete detector/crop/WiLoR batch without pipeline scheduling."""
    prepared = _prepare_dual_batched(
        frames, detector, model_cfg, confidence, iou, rescale_factor,
        preprocess_workers, max_detections_per_class, preprocess_executor,
        external_boxes,
    )
    return _infer_prepared_batched(
        prepared, model, model_cfg, device, batch_size, runtime_stats
    )


def _prefetched_frame_chunks(
    reader: SequentialVideoReader,
    rows: list[dict[str, str]],
    camera: str,
    frame_batch_size: int,
):
    """Decode the next H.264 chunk while CUDA processes the current chunk."""
    def load(chunk_start: int):
        chunk = rows[chunk_start:chunk_start + frame_batch_size]
        frames = [
            reader.read(int(row[f"{camera}_frame_index"])) for row in chunk
        ]
        return chunk_start, chunk, frames

    with ThreadPoolExecutor(max_workers=1) as decode_executor:
        next_start = 0
        pending = decode_executor.submit(load, next_start)
        next_start += frame_batch_size
        while pending is not None:
            chunk_start, chunk, frames = pending.result()
            pending = (
                decode_executor.submit(load, next_start)
                if next_start < len(rows) else None
            )
            next_start += frame_batch_size
            yield chunk_start, chunk, frames


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
    external_by_camera = (
        _load_external_boxes(args.external_boxes.resolve(), cameras, rows)
        if args.external_boxes is not None else None
    )
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
        cv2.setNumThreads(1)
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
    detector = (
        None if external_by_camera is not None
        else base.load_detector(args.detector, device)
    )
    if external_by_camera is not None:
        print(
            f"Using external hand proposals from {args.external_boxes.resolve()} "
            "(YOLO detector bypassed)",
            flush=True,
        )
    camera_summaries = {}
    runtime_stats = {
        "effective_hypothesis_batch_size": args.batch_size,
        "cuda_oom_retries": 0,
    }
    inference_started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=preprocess_workers) as preprocess_executor:
        for camera in cameras:
            camera_root = output / camera
            camera_root.mkdir()
            prediction_path = camera_root / "predictions.jsonl"
            reader = SequentialVideoReader(
                dataset / "cameras" / camera
                / manifest["storage"]["video_filename"],
                image_size,
            )
            physical_boxes = 0
            detected_frames = 0
            camera_started = time.perf_counter()
            try:
                with prediction_path.open("w", encoding="utf-8") as stream:
                    if accelerated:
                        result_batches = (
                            (
                                chunk_start,
                                chunk,
                                infer_dual_batched(
                                    frames, detector, model, model_cfg, device,
                                    camera_confidences[camera], args.iou,
                                    args.rescale_factor, args.batch_size,
                                    preprocess_workers,
                                    args.max_detections_per_class,
                                    preprocess_executor, runtime_stats,
                                    external_boxes=external_by_camera[camera][
                                        chunk_start:chunk_start + len(frames)
                                    ] if external_by_camera is not None else None,
                                ),
                            )
                            for chunk_start, chunk, frames
                            in _prefetched_frame_chunks(
                                reader, rows, camera, frame_batch_size
                            )
                        )
                    else:
                        result_batches = (
                            (
                                chunk_start,
                                chunk,
                                [
                                    infer_dual(
                                        frame, detector, model, model_cfg,
                                        device, camera_confidences[camera],
                                        args.iou, args.rescale_factor,
                                        args.batch_size,
                                        external_boxes=external_by_camera[camera][ordinal]
                                        if external_by_camera is not None else None,
                                    )
                                    for ordinal, frame in enumerate(
                                        frames, start=chunk_start
                                    )
                                ],
                            )
                            for chunk_start, chunk, frames
                            in _prefetched_frame_chunks(
                                reader, rows, camera, frame_batch_size
                            )
                        )
                    for chunk_start, chunk, hands_by_frame in result_batches:
                        for offset, (row, hands) in enumerate(
                            zip(chunk, hands_by_frame)
                        ):
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
                                "timestamp_ns": int(
                                    row[f"{camera}_timestamp_ns"]
                                ),
                                "hands": hands,
                            }
                            stream.write(
                                json.dumps(record, separators=(",", ":"))
                                + "\n"
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
                        del hands_by_frame
            finally:
                reader.close()
            if accelerated:
                gc.collect()
                torch = __import__("torch")
                torch.cuda.empty_cache()
            camera_seconds = time.perf_counter() - camera_started
            camera_summaries[camera] = {
                "frame_count": len(rows),
                "detected_frame_count": detected_frames,
                "physical_box_count": physical_boxes,
                "detector_confidence": camera_confidences[camera],
                "inference_seconds": camera_seconds,
                "camera_frames_per_second": len(rows) / camera_seconds,
            }
            (camera_root / "summary.json").write_text(
                json.dumps(camera_summaries[camera], indent=2) + "\n",
                encoding="utf-8",
            )
    inference_seconds = time.perf_counter() - inference_started
    processed_camera_frames = len(rows) * len(cameras)
    summary = {
        "schema_version": 1,
        "stage": "wilor_multiview_dual_hypothesis",
        "dataset": str(dataset),
        "external_boxes": (
            str(args.external_boxes.resolve())
            if args.external_boxes is not None else None
        ),
        "camera_ids": list(cameras),
        "frame_count": len(rows),
        "hypotheses_per_detection": 2,
        "gpu_profile": args.gpu_profile,
        "frame_batch_size": frame_batch_size,
        "hypothesis_batch_size": args.batch_size,
        "effective_hypothesis_batch_size": runtime_stats[
            "effective_hypothesis_batch_size"
        ],
        "cuda_oom_retries": runtime_stats["cuda_oom_retries"],
        "prefetched_video_decode": accelerated,
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
