"""Evaluation and full-frame visualization for glove fine-tuning."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from wilor.datasets.glove_dataset import FrameGroupKey, frame_group_key
from wilor.datasets.utils import crop_points_to_image
from wilor.utils import recursive_to
from wilor.utils.visualization import draw_hand_overlay

from .reporting import write_json, write_jsonl


def group_prediction_records(
    records: Iterable[dict[str, Any]],
) -> dict[FrameGroupKey, list[dict[str, Any]]]:
    """Group left/right predictions from the same camera frame."""
    groups: dict[FrameGroupKey, list[dict[str, Any]]] = {}
    for record in records:
        key = frame_group_key(record)
        groups.setdefault(key, []).append(record)

    for key, values in groups.items():
        sides = [int(record["side"]) for record in values]
        if len(sides) != len(set(sides)):
            raise ValueError(f"Duplicate hand side in visualization group {key}")
        source_frames = {
            int(record.get("source_frame_index", -1))
            for record in values
            if int(record.get("source_frame_index", -1)) >= 0
        }
        if len(source_frames) > 1:
            raise ValueError(
                f"Conflicting source frames in visualization group {key}: "
                f"{sorted(source_frames)}"
            )
    return groups


def _aggregate_metrics(values: list[dict[str, Any]]) -> dict[str, Any]:
    if not values:
        return {
            "samples": 0,
            "mean_mpjpe_mm": None,
            "mean_reprojection_error_px": None,
            "mean_visible_reprojection_error_px": None,
            "mean_all_reprojection_error_px": None,
            "visible_joint_count": 0,
            "mean_pred_cam": None,
            "mean_pred_cam_t": None,
            "mean_camera_root_error_mm": None,
        }
    visible_count = sum(int(record["visible_joint_count"]) for record in values)
    visible_error = (
        sum(
            float(record["visible_reprojection_error_px"])
            * int(record["visible_joint_count"])
            for record in values
            if record["visible_reprojection_error_px"] is not None
        ) / visible_count
        if visible_count else None
    )
    return {
        "samples": len(values),
        "mean_mpjpe_mm": float(np.mean([
            record["mpjpe_mm"] for record in values
        ])),
        "mean_reprojection_error_px": visible_error,
        "mean_visible_reprojection_error_px": visible_error,
        "mean_all_reprojection_error_px": float(np.mean([
            record["all_reprojection_error_px"] for record in values
        ])),
        "visible_joint_count": visible_count,
        "mean_pred_cam": np.mean(np.asarray([
            record["pred_cam"] for record in values
        ], dtype=np.float64), axis=0).tolist(),
        "mean_pred_cam_t": np.mean(np.asarray([
            record["pred_cam_t"] for record in values
        ], dtype=np.float64), axis=0).tolist(),
        "mean_camera_root_error_mm": (
            float(np.mean([
                record["camera_root_error_mm"] for record in values
            ])) if "camera_root_error_mm" in values[0] else None
        ),
    }


def _safe_name(value: Any) -> str:
    return "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in str(value)
    )


@torch.no_grad()
def evaluate_glove_set(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    output_dir: str | Path,
    use_amp: bool,
    save_overlays: bool = True,
) -> dict[str, Any]:
    """Run known-bbox inference and write camera-aware metrics and predictions."""
    destination = Path(output_dir).expanduser().resolve()
    overlay_dir = destination / "overlays"
    if save_overlays:
        overlay_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    for batch in loader:
        device_batch = recursive_to(batch, device)
        amp_context = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if use_amp else nullcontext()
        )
        with amp_context:
            output = model.forward_step(device_batch, train=False)

        use_marker = (
            "pred_marker_keypoints_2d" in output and "marker_keypoints_2d" in batch
        )
        pred_crop = output[
            "pred_marker_keypoints_2d" if use_marker else "pred_keypoints_2d"
        ].float().cpu().numpy()
        gt_crop = batch[
            "marker_keypoints_2d" if use_marker else "keypoints_2d"
        ].float().cpu().numpy()
        pred_3d = output[
            "pred_marker_keypoints_3d" if use_marker else "pred_keypoints_3d"
        ].float().cpu()
        gt_3d = batch[
            "marker_keypoints_3d" if use_marker else "keypoints_3d"
        ].float()
        transforms = batch["crop_transform"].numpy()
        image_sizes = batch["image_size"].numpy()
        bboxes = batch["bbox_xyxy"].numpy()
        sides = batch["side"].numpy().astype(np.int32)
        pred_cams = output["pred_cam"].float().cpu().numpy()
        pred_cam_ts = output["pred_cam_t"].float().cpu().numpy()
        focal_lengths = output["focal_length"].float().cpu().numpy()
        physical_roots = (
            output[
                "pred_marker_root_camera_m" if use_marker else "pred_root_camera_m"
            ].float().cpu().numpy()
            if "pred_root_camera_m" in output else None
        )
        target_roots = (
            batch[
                "marker_root_camera_m" if use_marker else "camera_root_m"
            ].float().cpu().numpy()
            if physical_roots is not None else None
        )

        for index in range(int(batch["img"].shape[0])):
            gt_image = crop_points_to_image(
                gt_crop[index],
                transforms[index],
                image_sizes[index],
                bool(sides[index] == 0),
            )
            pred_image = crop_points_to_image(
                pred_crop[index],
                transforms[index],
                image_sizes[index],
                bool(sides[index] == 0),
            )
            pred_root = pred_3d[index] - pred_3d[index, :1]
            gt_root = gt_3d[index, :, :3] - gt_3d[index, :1, :3]
            if "view_rotation" in batch:
                from tools.hand_finetune.crop_geometry import PerspectiveCrop
                rotation = batch["view_rotation"][index].numpy()
                view = PerspectiveCrop(rotation, batch["camera_K"][index].numpy(),
                                       batch["raw_K"][index].numpy(), batch["camera_distortion"][index].numpy(),
                                       int(batch["img"].shape[-1]))
                pred_image, gt_image = view.native_pixels(pred_image), view.native_pixels(gt_image)
                bboxes[index] = batch["raw_bbox"][index].numpy()
                canonical_rotation = view.physical_mano_orientation(np.eye(3), bool(sides[index])).T
                pred_root = pred_root @ pred_root.new_tensor(canonical_rotation)
                gt_root = gt_root @ gt_root.new_tensor(canonical_rotation)
                if physical_roots is not None:
                    physical_roots[index] = physical_roots[index] @ rotation
                    target_roots[index] = target_roots[index] @ rotation
            visibility = gt_crop[index, :, 2] > 0
            point_errors = np.linalg.norm(pred_image - gt_image, axis=1)
            visible_count = int(visibility.sum())
            visible_error = (
                float(point_errors[visibility].mean())
                if visible_count else None
            )
            record = {
                "sample_id": str(batch["sample_id"][index]),
                "image_path": str(batch["image_path"][index]),
                "source_group": (
                    str(batch["source_group"][index])
                    if "source_group" in batch else ""
                ),
                "sync_index": int(batch["sync_index"][index]),
                "source_frame_index": int(batch["source_frame_index"][index]),
                "camera": str(batch["camera"][index]),
                "side": int(sides[index]),
                "point_type": "marker21" if use_marker else "mano21",
                "bbox": bboxes[index].tolist(),
                "visibility_2d": visibility.astype(np.int32).tolist(),
                "visible_joint_count": visible_count,
                "mpjpe_mm": float(
                    torch.linalg.vector_norm(
                        pred_root - gt_root, dim=-1
                    ).mean() * 1000.0
                ),
                "reprojection_error_px": visible_error,
                "visible_reprojection_error_px": visible_error,
                "all_reprojection_error_px": float(point_errors.mean()),
                "pred_cam": pred_cams[index].tolist(),
                "pred_cam_t": pred_cam_ts[index].tolist(),
                "focal_length": focal_lengths[index].tolist(),
                "pred_joints_2d": pred_image.tolist(),
                "gt_joints_2d": gt_image.tolist(),
                "pred_joints_3d_root_relative": pred_root.numpy().tolist(),
                "gt_joints_3d_root_relative": gt_root.numpy().tolist(),
            }
            if physical_roots is not None:
                record["pred_root_camera_m"] = physical_roots[index].tolist()
                record["gt_root_camera_m"] = target_roots[index].tolist()
                record["camera_root_error_mm"] = float(
                    np.linalg.norm(physical_roots[index] - target_roots[index]) * 1000
                )
            records.append(record)

    if not records:
        raise RuntimeError("Evaluation loader produced no samples")

    groups = group_prediction_records(records)
    if save_overlays:
        for group_index, (key, group_records) in enumerate(groups.items()):
            source_group, frame_id, camera = key
            source_path = Path(group_records[0]["image_path"])
            source = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
            if source is None:
                raise IOError(f"Could not read test image: {source_path}")
            hands = [{
                "bbox": record["bbox"],
                "gt_joints_2d": record["gt_joints_2d"],
                "pred_joints_2d": record["pred_joints_2d"],
                "is_right": record["side"] == 1,
                "label": (
                    f"{'R' if record['side'] == 1 else 'L'} "
                    f"{record['mpjpe_mm']:.1f}mm"
                ),
            } for record in group_records]
            group_mpjpe = float(np.mean([
                record["mpjpe_mm"] for record in group_records
            ]))
            group_visible = [
                record["visible_reprojection_error_px"] for record in group_records
                if record["visible_reprojection_error_px"] is not None
            ]
            group_error = float(np.mean(group_visible)) if group_visible else float("nan")
            sync_name = (
                f"sync_{int(frame_id):06d}"
                if isinstance(frame_id, int) else f"group_{group_index:06d}"
            )
            source_name = f"{_safe_name(source_group)}_" if source_group else ""
            camera_name = _safe_name(camera)
            output_path = overlay_dir / f"{source_name}{sync_name}_{camera_name}.jpg"
            overlay = draw_hand_overlay(
                source,
                hands,
                footer=(
                    f"source={source_group or '-'} sync={frame_id} camera={camera} "
                    f"hands={len(group_records)} "
                    f"mean MPJPE={group_mpjpe:.1f}mm 2D={group_error:.1f}px"
                ),
            )
            if not cv2.imwrite(str(output_path), overlay):
                raise IOError(f"Could not write visualization: {output_path}")

    metrics = {
        **_aggregate_metrics(records),
        "visualization_frames": len(groups),
        "by_side": {
            name: _aggregate_metrics([
                record for record in records if record["side"] == side
            ])
            for name, side in (("left", 0), ("right", 1))
        },
        "by_camera": {
            camera: _aggregate_metrics([
                record for record in records if record["camera"] == camera
            ])
            for camera in sorted({record["camera"] for record in records})
        },
    }
    write_json(destination / "metrics.json", metrics)
    write_jsonl(destination / "predictions.jsonl", records)
    return metrics


# Compatibility for callers written before train/val replaced train/val/test.
evaluate_glove_test_set = evaluate_glove_set
