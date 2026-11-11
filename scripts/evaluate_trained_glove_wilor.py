#!/usr/bin/env python3
"""Run a glove WiLoR adapter through the EGO hand crop/detector path.

The input manifest contains rectified cam0/cam1 images.  This diagnostic keeps
the EGO system's crop and legacy full-image projection, and also evaluates the
trained physical-camera projection on exactly the same crops.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch


EGO_ROOT = Path(__file__).resolve().parents[1]
GLOVE_ROOT = EGO_ROOT.parent
TRAINING_ROOT = GLOVE_ROOT / "wilor_training"
sys.path.insert(0, str(TRAINING_ROOT))
sys.path.insert(0, str(EGO_ROOT / "scripts"))

# Preload the fine-tuning fork before importing the EGO inference module, which
# otherwise selects its pinned, unmodified WiLoR checkout.
from wilor.datasets.vitdet_dataset import ViTDetDataset  # noqa: E402,F401
from wilor.datasets.utils import gen_trans_from_patch_cv  # noqa: E402
from wilor.models.adapter import load_adapter_weights  # noqa: E402
from wilor.training.evaluation import crop_points_to_image  # noqa: E402
from wilor.training.glove import GloveTrainingConfig, load_glove_model  # noqa: E402
from compare_marker_projection import mapped_points  # noqa: E402
import wilor_multiview_inference as ego_multi  # noqa: E402


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    upper_left = np.maximum(a[:2], b[:2])
    lower_right = np.minimum(a[2:], b[2:])
    area = float(np.prod(np.maximum(lower_right - upper_left, 0)))
    area_a = float(np.prod(np.maximum(a[2:] - a[:2], 0)))
    area_b = float(np.prod(np.maximum(b[2:] - b[:2], 0)))
    return area / max(area_a + area_b - area, 1e-9)


def _metrics(rows: list[dict], mode: str, camera: str,
             model_projection: str) -> dict:
    selected = [r for r in rows if r["mode"] == mode and r["camera"] == camera]
    if not selected:
        return {"samples": 0}
    result = {"samples": len(selected)}
    for field in (f"{model_projection}_error_px", "ego_legacy_error_px",
                  "mano_error_px"):
        values = np.concatenate([np.asarray(r[field + "_points"], dtype=np.float64) for r in selected])
        result[field] = {
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "p95": float(np.percentile(values, 95)),
            "points": int(values.size),
        }
    return result


def _draw(image: np.ndarray, marker: np.ndarray, physical: np.ndarray,
          legacy: np.ndarray, box: np.ndarray, title: str) -> np.ndarray:
    canvas = image.copy()
    for points, color in ((marker, (220, 0, 220)), (legacy, (0, 220, 220)),
                          (physical, (0, 0, 255))):
        for x, y in points:
            if np.isfinite((x, y)).all():
                cv2.circle(canvas, (round(float(x)), round(float(y))), 4, color, -1)
    x1, y1, x2, y2 = np.rint(box).astype(int)
    cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 160, 0), 2)
    cv2.putText(canvas, title, (30, 50), cv2.FONT_HERSHEY_SIMPLEX,
                1.1, (255, 255, 255), 3, cv2.LINE_AA)
    return canvas


def _raw_marker_report(results: list[dict], observations_path: Path,
                       rectification_path: Path,
                       model_projection: str = "physical") -> dict:
    observations = {
        int(row["sync_index"]): row
        for row in (json.loads(line) for line in observations_path.open())
    }
    with np.load(rectification_path) as archive:
        maps = {key: archive[key] for key in archive.files if key.startswith("map_")}
    errors = {mode: {camera: {kind: [] for kind in (model_projection, "ego_legacy")}
                     for camera in ("camera0", "camera1")}
              for mode in ("label", "detector")}
    frames = {mode: {camera: 0 for camera in ("camera0", "camera1")}
              for mode in errors}
    for row in results:
        camera, mode, sync = row["camera"], row["mode"], row["sync_index"]
        observation = observations.get(sync)
        if observation is None:
            continue
        hand = next((hand for hand in observation["hands"]
                     if hand["side"] == 0), None)
        if hand is None:
            continue
        source = hand["views"][camera]["joints_2d"]
        target = np.asarray([
            xy if xy is not None and len(xy) == 2
            and all(value is not None for value in xy) else [np.nan, np.nan]
            for xy in source
        ], dtype=np.float64)
        valid = np.isfinite(target).all(axis=1)
        prefix = "left" if camera == "camera0" else "right"
        projected = {}
        for kind, key in ((model_projection, f"pred_{model_projection}_2d"),
                          ("ego_legacy", "pred_ego_legacy_2d")):
            xy, in_bounds = mapped_points(
                np.asarray(row[key], dtype=np.float64),
                maps[f"map_{prefix}_x"], maps[f"map_{prefix}_y"])
            projected[kind] = xy
            valid &= in_bounds
        if not valid.any():
            continue
        frames[mode][camera] += 1
        for kind, xy in projected.items():
            errors[mode][camera][kind].extend(
                np.linalg.norm(xy[valid] - target[valid], axis=1).tolist())
    report = {}
    for mode, by_camera in errors.items():
        report[mode] = {}
        for camera, by_kind in by_camera.items():
            report[mode][camera] = {"frames": frames[mode][camera]}
            for kind, values in by_kind.items():
                data = np.asarray(values, dtype=np.float64)
                report[mode][camera][kind] = {
                    "n": int(data.size),
                    "mean": float(data.mean()) if data.size else None,
                    "median": float(np.median(data)) if data.size else None,
                    "p95": float(np.percentile(data, 95)) if data.size else None,
                }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--marker-input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path,
                        default=TRAINING_ROOT / "pretrained_models/wilor_final.ckpt")
    parser.add_argument("--model-config", type=Path,
                        default=TRAINING_ROOT / "pretrained_models/model_config.yaml")
    parser.add_argument("--mano-path", type=Path,
                        default=TRAINING_ROOT / "mano_data/mano_v1_2")
    parser.add_argument("--detector", type=Path,
                        default=EGO_ROOT / "models/wilor/detector_1009.pt")
    parser.add_argument("--raw-observations", type=Path, default=None)
    parser.add_argument("--rectification", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--confidence", type=float, default=0.1)
    parser.add_argument("--rescale-factor", type=float, default=2.0)
    parser.add_argument("--overlay-every", type=int, default=40)
    args = parser.parse_args()
    root = args.root.resolve()
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in (root / "val.jsonl").open()]
    if args.max_samples:
        rows = rows[:args.max_samples]
    with np.load(args.marker_input) as source:
        hand_index = next(i for i, side in enumerate(source["handedness"])
                          if str(side).lower() == "left")
        marker = {camera: source[key][hand_index].copy() for camera, key in (
            ("camera0", "left_rectified_px"), ("camera1", "right_rectified_px"))}
        marker_valid = {camera: source[key][hand_index].copy() for camera, key in (
            ("camera0", "left_rectified_valid"),
            ("camera1", "right_rectified_valid"))}
        pair_index = {int(value): i for i, value in enumerate(source["pair_indices"])}

    device = torch.device(args.device)
    config = GloveTrainingConfig(root=root, checkpoint=args.checkpoint,
                                 model_config=args.model_config,
                                 mano_path=args.mano_path, output=output_dir)
    model = load_glove_model(config, device)
    load_adapter_weights(args.adapter, model)
    model.eval()
    model_cfg = model.cfg
    detector = ego_multi.base.load_detector(args.detector, device)
    model_projection = "physical" if model.physical_camera_enabled else "virtual_crop"
    print(f"loaded model and detector; model_projection={model_projection}", flush=True)

    all_results: list[dict] = []
    detection_stats = {camera: {"total": 0, "found": 0, "iou_50": 0,
                                "ious": [], "left_class": 0} for camera in marker}
    overlay_dir = output_dir / "overlays"
    overlay_dir.mkdir(exist_ok=True)
    with torch.inference_mode():
        for ordinal, row in enumerate(rows):
            camera = str(row["camera"])
            if camera not in marker or int(row["side"]) != 0:
                continue
            sync = int(row["sync_index"])
            if sync not in pair_index:
                continue
            marker_xy = marker[camera][pair_index[sync]]
            valid = marker_valid[camera][pair_index[sync]] & np.isfinite(marker_xy).all(axis=1)
            if not valid.any():
                continue
            frame = cv2.imread(str(root / row["image"]), cv2.IMREAD_COLOR)
            label = np.load(root / row["label"], allow_pickle=True).item()
            gt_box = np.asarray(label["bbox"], dtype=np.float32)
            K = np.asarray(label["K"], dtype=np.float32)
            height, width = frame.shape[:2]

            stat = detection_stats[camera]
            stat["total"] += 1
            detection = detector(frame, conf=args.confidence, iou=0.5,
                                 verbose=False)[0]
            proposals: list[tuple[str, np.ndarray, float | None]] = [("label", gt_box, None)]
            if detection.boxes is not None and len(detection.boxes):
                boxes = detection.boxes.xyxy.detach().cpu().numpy().astype(np.float32)
                sides = detection.boxes.cls.detach().cpu().numpy().astype(np.int32)
                overlaps = np.asarray([_iou(box, gt_box) for box in boxes])
                chosen = int(np.argmax(overlaps))
                best_iou = float(overlaps[chosen])
                stat["found"] += 1
                stat["iou_50"] += int(best_iou >= 0.5)
                stat["ious"].append(best_iou)
                stat["left_class"] += int(sides[chosen] == 0)
                # Matching by label IoU isolates detector localization.  The
                # IoU and handedness are reported so this is not mistaken for
                # fully automatic hand association.
                proposals.append(("detector", boxes[chosen], best_iou))

            boxes = np.stack([item[1] for item in proposals])
            sides = np.zeros(len(proposals), dtype=np.float32)
            crops = ego_multi._FastViTDetDataset(
                model_cfg, frame, boxes, sides,
                rescale_factor=args.rescale_factor, fp16=False)
            crop_samples = [crops[i] for i in range(len(crops))]
            transforms = []
            for sample in crop_samples:
                center = sample["box_center"]
                size = float(sample["box_size"])
                mirrored_center_x = width - float(center[0]) - 1
                transforms.append(gen_trans_from_patch_cv(
                    mirrored_center_x, float(center[1]), size, size,
                    256, 256, 1.0, 0))
            model_batch = {
                "img": torch.as_tensor(np.stack([s["img"] for s in crop_samples]),
                                       device=device, dtype=torch.float32),
                "side": torch.zeros(len(crop_samples), device=device, dtype=torch.long),
                "camera_K": torch.as_tensor(np.repeat(K[None], len(crop_samples), axis=0),
                                            device=device),
                "crop_transform": torch.as_tensor(np.stack(transforms), device=device,
                                                  dtype=torch.float32),
                "crop_box_size": torch.as_tensor([s["box_size"] for s in crop_samples],
                                                 device=device, dtype=torch.float32),
                "image_size": torch.as_tensor([[width, height]] * len(crop_samples),
                                               device=device, dtype=torch.float32),
            }
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                enabled=device.type == "cuda"):
                prediction = model(model_batch)
            for index, (mode, box, iou) in enumerate(proposals):
                transform = np.asarray(transforms[index], dtype=np.float32)
                model_points = crop_points_to_image(
                    prediction["pred_keypoints_2d"][index].float().cpu().numpy(),
                    transform, np.array([width, height]), True)
                # The actual EGO path uses a virtual full-image focal length.
                virtual = prediction["pred_cam"][index].detach().float().clone()
                virtual[1] *= -1
                focal = float(model_cfg.EXTRA.FOCAL_LENGTH / 256 * max(width, height))
                camera_full = ego_multi.base.cam_crop_to_full(
                    virtual[None],
                    torch.as_tensor(crop_samples[index]["box_center"],
                                    device=device, dtype=torch.float32)[None],
                    torch.tensor([crop_samples[index]["box_size"]], device=device),
                    torch.tensor([[width, height]], device=device, dtype=torch.float32),
                    focal)[0].cpu().numpy()
                joints_3d = prediction["pred_keypoints_3d"][index].float().cpu().numpy()
                joints_3d[:, 0] *= -1
                legacy = ego_multi.base.project_points(
                    joints_3d, camera_full, focal, np.array([width, height]))
                mano_3d = np.asarray(label["joints_3d"], dtype=np.float32)
                trans = np.asarray(label["trans"], dtype=np.float32)
                points = mano_3d + trans
                mano = (points @ K.T)[:, :2] / points[:, 2:3]
                errors = {name: np.linalg.norm(xy[valid] - marker_xy[valid], axis=1)
                          for name, xy in ((model_projection, model_points), ("ego_legacy", legacy),
                                           ("mano", mano))}
                record = {"sync_index": sync, "camera": camera, "mode": mode,
                          "detector_iou": iou, "bbox": box.tolist(),
                          "marker_valid_count": int(valid.sum()),
                          f"pred_{model_projection}_2d": model_points.tolist(),
                          "pred_ego_legacy_2d": legacy.tolist()}
                for name, values in errors.items():
                    record[name + "_error_px_points"] = values.tolist()
                    record[name + "_error_px"] = float(values.mean())
                all_results.append(record)
                if args.overlay_every > 0 and ordinal % args.overlay_every == 0:
                    title = (f"{camera} sync={sync} {mode}: {model_projection} "
                             f"{record[f'{model_projection}_error_px']:.1f}px, EGO legacy "
                             f"{record['ego_legacy_error_px']:.1f}px")
                    canvas = _draw(frame, marker_xy[valid], model_points, legacy, box, title)
                    cv2.imwrite(str(overlay_dir / f"{camera}_{sync:06d}_{mode}.jpg"), canvas)
            if (ordinal + 1) % 50 == 0:
                print(f"processed {ordinal + 1}/{len(rows)}", flush=True)

    metrics = {mode: {camera: _metrics(all_results, mode, camera, model_projection)
                      for camera in marker} for mode in ("label", "detector")}
    for camera, stat in detection_stats.items():
        stat["mean_best_iou"] = float(np.mean(stat["ious"])) if stat["ious"] else None
        del stat["ious"]
    report = {"adapter": str(args.adapter.resolve()),
              "input": str(root), "marker_input": str(args.marker_input.resolve()),
              "detector": str(args.detector.resolve()),
              "confidence": args.confidence,
              "box_match": "best IoU to label box; diagnostic association",
              "rescale_factor": args.rescale_factor,
              "model_projection": model_projection,
              "metrics": metrics, "detection": detection_stats}
    (output_dir / "predictions.jsonl").write_text(
        "".join(json.dumps(r, separators=(",", ":")) + "\n" for r in all_results))
    (output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    observations_path = (args.raw_observations or
                         args.marker_input.parent / "nokov_observations_cam01/accepted.jsonl")
    rectification_path = (args.rectification or
                          args.marker_input.parent / "rectification_cam01.npz")
    if observations_path.is_file() and rectification_path.is_file():
        raw_report = _raw_marker_report(
            all_results, observations_path, rectification_path, model_projection)
        (output_dir / "raw_marker_report.json").write_text(
            json.dumps(raw_report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
