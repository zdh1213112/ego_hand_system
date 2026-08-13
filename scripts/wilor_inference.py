#!/usr/bin/env python3
"""Run WiLoR on one camera of an EGO rectified stereo dataset.

This is the project adapter around the pinned WiLoR research checkout.  The
process handles one camera so the offline runner can launch left/right workers
with isolated CUDA model lifetimes.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WILOR_ROOT = PROJECT_ROOT / "third_party" / "WiLoR"
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(WILOR_ROOT))

from ego_data.dataset import RectifiedStereoDataset, SequentialVideoReader  # noqa: E402
from wilor.datasets.vitdet_dataset import ViTDetDataset  # noqa: E402
from wilor.configs import get_config  # noqa: E402
from wilor.models import WiLoR  # noqa: E402
from wilor.utils import recursive_to  # noqa: E402
from wilor.utils.renderer import cam_crop_to_full  # noqa: E402


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false")
    return device


def load_detector(checkpoint_path: Path, device: torch.device):
    from ultralytics import YOLO
    original_torch_load = torch.load

    def trusted_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_torch_load(*args, **kwargs)

    torch.load = trusted_load
    try:
        return YOLO(str(checkpoint_path)).to(device)
    finally:
        torch.load = original_torch_load


def load_wilor_model(checkpoint: Path, config: Path, mano_model_dir: Path):
    model_cfg = get_config(str(config), update_cachedir=True)
    model_cfg.defrost()
    if "vit" in model_cfg.MODEL.BACKBONE.TYPE and "BBOX_SHAPE" not in model_cfg.MODEL:
        model_cfg.MODEL.BBOX_SHAPE = [192, 256]
    if "PRETRAINED_WEIGHTS" in model_cfg.MODEL.BACKBONE:
        model_cfg.MODEL.BACKBONE.pop("PRETRAINED_WEIGHTS")
    model_cfg.MANO.DATA_DIR = str(mano_model_dir.resolve())
    model_cfg.MANO.MODEL_PATH = str(mano_model_dir.resolve())
    model_cfg.MANO.MEAN_PARAMS = str(
        (WILOR_ROOT / "mano_data" / "mano_mean_params.npz").resolve()
    )
    model_cfg.freeze()
    model = WiLoR.load_from_checkpoint(
        str(checkpoint), strict=False, cfg=model_cfg, map_location="cpu"
    )
    return model, model_cfg


def project_points(points: np.ndarray, camera_translation: np.ndarray,
                   focal_length: float, image_size: np.ndarray) -> np.ndarray:
    points = points + camera_translation
    points = points / points[..., -1:]
    camera_center = np.asarray([image_size[0] / 2.0, image_size[1] / 2.0])
    return np.column_stack((focal_length * points[:, 0] + camera_center[0],
                            focal_length * points[:, 1] + camera_center[1]))


def infer_frame(frame, detector, model, model_cfg, device, confidence, iou,
                rescale_factor, batch_size, fp16):
    detection = detector(frame, conf=confidence, iou=iou, verbose=False)[0]
    if detection.boxes is None or len(detection.boxes) == 0:
        return [], []
    boxes = detection.boxes.xyxy.detach().cpu().numpy().astype(np.float32)
    confidences = detection.boxes.conf.detach().cpu().numpy().astype(np.float32)
    handedness = detection.boxes.cls.detach().cpu().numpy().astype(np.float32)
    dataset = ViTDetDataset(model_cfg, frame, boxes, handedness,
                            rescale_factor=rescale_factor, fp16=fp16)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=False, num_workers=0)
    records = [None] * len(dataset)
    vertices_out = [None] * len(dataset)
    for batch in loader:
        batch = recursive_to(batch, device)
        with torch.inference_mode():
            output = model(batch)
        multiplier = 2 * batch["right"] - 1
        pred_cam = output["pred_cam"].clone()
        pred_cam[:, 1] = multiplier * pred_cam[:, 1]
        image_size = batch["img_size"].float()
        focal_length = (model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE
                        * image_size.max()).item()
        camera_full = cam_crop_to_full(
            pred_cam, batch["box_center"].float(), batch["box_size"].float(),
            image_size, focal_length).detach().cpu().numpy()
        for index in range(batch["img"].shape[0]):
            detection_index = int(batch["personid"][index].item())
            is_right = int(round(float(batch["right"][index].item())))
            side = 2 * is_right - 1
            joints3d = output["pred_keypoints_3d"][index].detach().float().cpu().numpy()
            vertices = output["pred_vertices"][index].detach().float().cpu().numpy()
            joints3d[:, 0] *= side
            vertices[:, 0] *= side
            camera = camera_full[index].astype(np.float32)
            joints2d = project_points(joints3d, camera, focal_length,
                                       image_size[index].detach().cpu().numpy())
            mano_params = {
                key: value[index].detach().float().cpu().numpy().tolist()
                for key, value in output["pred_mano_params"].items()
            }
            records[detection_index] = {
                "detection_index": detection_index,
                "bbox_xyxy": boxes[detection_index].tolist(),
                "confidence": float(confidences[detection_index]),
                "handedness": "right" if is_right else "left",
                "is_right": is_right,
                "camera_translation": camera.tolist(),
                "joints_2d": joints2d.tolist(),
                "joints_3d": joints3d.tolist(),
                "mano": mano_params,
            }
            vertices_out[detection_index] = vertices
    return [record for record in records if record is not None], \
        [value for value in vertices_out if value is not None]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rectified-dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--camera", required=True, choices=("left", "right"))
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--model-config", required=True, type=Path)
    parser.add_argument("--detector", required=True, type=Path)
    parser.add_argument("--mano-model-dir", required=True, type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--confidence", type=float, default=0.3)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--rescale-factor", type=float, default=2.0)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--save-vertices", action="store_true")
    return parser.parse_args()


def _json_record(record: dict[str, Any], save_vertices: bool) -> dict[str, Any]:
    if save_vertices:
        return record
    result = dict(record)
    for hand in result.get("hands", []):
        hand.pop("vertices", None)
    return result


def _coerce_mano_param(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    """Convert one WiLoR MANO parameter to the stable NPZ shape.

    WiLoR emits ``global_orient`` with a singleton joint dimension, i.e.
    ``(1, 3, 3)``, while the per-hand archive contract stores one matrix as
    ``(3, 3)``.  Keep accepting that native form, but fail with a useful error
    if a checkpoint produces a genuinely incompatible tensor.
    """
    result = np.asarray(value, dtype=np.float32)
    if result.shape != shape and result.ndim == len(shape) + 1 and result.shape[0] == 1:
        result = result[0]
    if result.shape != shape:
        raise ValueError(
            f"WiLoR MANO parameter {name} has shape {result.shape}; expected {shape}"
        )
    return result


def _write_npz(path: Path, rows: list[dict[str, Any]]) -> None:
    max_hands = max((len(row["hands"]) for row in rows), default=0)
    frames = len(rows)
    bboxes = np.full((frames, max_hands, 4), np.nan, dtype=np.float32)
    confidences = np.full((frames, max_hands), np.nan, dtype=np.float32)
    is_right = np.full((frames, max_hands), -1, dtype=np.int8)
    joints2d = np.full((frames, max_hands, 21, 2), np.nan, dtype=np.float32)
    joints3d = np.full((frames, max_hands, 21, 3), np.nan, dtype=np.float32)
    camera_translation = np.full((frames, max_hands, 3), np.nan, dtype=np.float32)
    global_orient = np.full((frames, max_hands, 3, 3), np.nan, dtype=np.float32)
    hand_pose = np.full((frames, max_hands, 15, 3, 3), np.nan, dtype=np.float32)
    betas = np.full((frames, max_hands, 10), np.nan, dtype=np.float32)
    for frame_index, row in enumerate(rows):
        for hand_index, hand in enumerate(row["hands"]):
            bboxes[frame_index, hand_index] = hand["bbox_xyxy"]
            confidences[frame_index, hand_index] = hand["confidence"]
            is_right[frame_index, hand_index] = hand["is_right"]
            joints2d[frame_index, hand_index] = hand["joints_2d"]
            joints3d[frame_index, hand_index] = hand["joints_3d"]
            camera_translation[frame_index, hand_index] = hand["camera_translation"]
            mano = hand.get("mano", {})
            if "global_orient" in mano:
                global_orient[frame_index, hand_index] = _coerce_mano_param(
                    mano["global_orient"], (3, 3), "global_orient"
                )
            if "hand_pose" in mano:
                hand_pose[frame_index, hand_index] = _coerce_mano_param(
                    mano["hand_pose"], (15, 3, 3), "hand_pose"
                )
            if "betas" in mano:
                betas[frame_index, hand_index] = _coerce_mano_param(
                    mano["betas"], (10,), "betas"
                )
    np.savez_compressed(
        path,
        frame_index=np.asarray([row["frame_index"] for row in rows], dtype=np.int64),
        pair_index=np.asarray([row["pair_index"] for row in rows], dtype=np.int64),
        source_frame_index=np.asarray([row["source_frame_index"] for row in rows], dtype=np.int64),
        timestamp_us=np.asarray([row["timestamp_us"] for row in rows], dtype=np.int64),
        hand_count=np.asarray([len(row["hands"]) for row in rows], dtype=np.int32),
        bboxes_xyxy=bboxes,
        confidences=confidences,
        is_right=is_right,
        joints_2d_px=joints2d,
        joints_3d=joints3d,
        camera_translation=camera_translation,
        global_orient=global_orient,
        hand_pose=hand_pose,
        betas=betas,
    )


def run(args: argparse.Namespace) -> None:
    if args.batch_size < 1 or args.frame_stride < 1 or args.max_pairs < 0:
        raise ValueError("batch-size/frame-stride must be positive and max-pairs non-negative")
    dataset = RectifiedStereoDataset(args.rectified_dataset)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    output.mkdir(parents=True)
    device = choose_device(args.device)
    model, model_cfg = load_wilor_model(
        args.checkpoint, args.model_config, args.mano_model_dir
    )
    if args.fast:
        if device.type != "cuda":
            raise RuntimeError("--fast requires CUDA")
        import torch
        torch.set_float32_matmul_precision("high")
        model = model.half()
        model.backbone = torch.compile(model.backbone)
        model.backbone.skip_blocks = True
    model = model.to(device).eval()
    detector = load_detector(args.detector, device)

    video_path = dataset.root / ("left.mkv" if args.camera == "left" else "right.mkv")
    reader = SequentialVideoReader(video_path, dataset.image_size)
    rows: list[dict[str, Any]] = []
    try:
        selected = dataset.pairs[: args.max_pairs or None]
        for frame_index, pair in enumerate(selected):
            frame = reader.read(frame_index)
            if frame_index % args.frame_stride:
                continue
            hands, vertices = infer_frame(
                frame, detector, model, model_cfg, device,
                args.confidence, args.iou, args.rescale_factor, args.batch_size, args.fast,
            )
            if args.save_vertices:
                for hand, hand_vertices in zip(hands, vertices):
                    hand["vertices"] = hand_vertices.tolist()
            timestamp_key = "left_timestamp_ns" if args.camera == "left" else "right_timestamp_ns"
            rows.append({
                "frame_index": len(rows),
                "pair_index": int(pair["pair_index"]),
                "source_frame_index": int(pair["left_frame_index"] if args.camera == "left" else pair["right_frame_index"]),
                "timestamp_us": int(pair[timestamp_key]) // 1000,
                "hands": hands,
            })
    finally:
        reader.close()
    with (output / "predictions.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(_json_record(row, args.save_vertices), separators=(",", ":")) + "\n")
    _write_npz(output / "predictions.npz", rows)
    summary = {
        "schema_version": 1,
        "stage": "wilor_stereo",
        "camera": args.camera,
        "pair_count": len(rows),
        "input_dataset": str(dataset.root),
        "image_size": list(dataset.image_size),
        "frame_stride": args.frame_stride,
        "coordinate_system": "rectified image pixels; WiLoR camera-relative 3D prediction",
        "model": {"checkpoint": str(args.checkpoint), "detector": str(args.detector)},
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    run(parse_args())
