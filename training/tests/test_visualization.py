from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from wilor.training.evaluation import evaluate_glove_set, group_prediction_records
from wilor.utils.visualization import draw_hand_overlay


def _record(
    side: int,
    camera: str = "camera2",
    source_group: str = "",
) -> dict:
    return {
        "source_group": source_group,
        "sync_index": 12,
        "source_frame_index": 12,
        "camera": camera,
        "side": side,
        "image_path": f"{camera}_{side}.jpg",
    }


def test_group_predictions_combines_left_and_right_per_camera() -> None:
    records = [
        _record(0, "camera2"),
        _record(0, "camera3"),
        _record(1, "camera2"),
        _record(1, "camera3"),
    ]
    groups = group_prediction_records(records)
    assert len(groups) == 2
    assert [len(values) for values in groups.values()] == [2, 2]
    assert all({record["side"] for record in values} == {0, 1}
               for values in groups.values())


def test_group_predictions_rejects_duplicate_side() -> None:
    with pytest.raises(ValueError, match="Duplicate hand side"):
        group_prediction_records([_record(0), _record(0)])


def test_group_predictions_separates_recordings_with_reused_sync_index() -> None:
    groups = group_prediction_records([
        _record(0, source_group="source_dataset_id:0"),
        _record(0, source_group="source_dataset_id:1"),
    ])
    assert len(groups) == 2


def test_draw_hand_overlay_does_not_mutate_input() -> None:
    image = np.zeros((160, 240, 3), dtype=np.uint8)
    original = image.copy()
    left_points = np.stack([
        np.linspace(30, 90, 21),
        np.linspace(40, 120, 21),
    ], axis=1)
    right_points = left_points + np.array([100, 0])
    output = draw_hand_overlay(image, [
        {
            "bbox": [20, 20, 100, 140],
            "joints_2d": left_points,
            "is_right": False,
        },
        {
            "bbox": [120, 20, 210, 140],
            "joints_2d": right_points,
            "is_right": True,
        },
    ])
    np.testing.assert_array_equal(image, original)
    assert output.shape == image.shape
    assert np.any(output != image)


class _EvaluationModel(torch.nn.Module):
    def forward_step(self, batch: dict, train: bool = False) -> dict:
        batch_size = int(batch["img"].shape[0])
        pred_2d = batch["keypoints_2d"][..., :2].clone()
        pred_2d[..., 0] += 1.0 / 256.0
        return {
            "pred_keypoints_2d": pred_2d,
            "pred_keypoints_3d": batch["keypoints_3d"][..., :3].clone(),
            "pred_cam": torch.tensor(
                [1.0, 0.1, -0.1], device=pred_2d.device
            ).repeat(batch_size, 1),
            "pred_cam_t": torch.tensor(
                [0.1, -0.1, 3.0], device=pred_2d.device
            ).repeat(batch_size, 1),
            "focal_length": torch.full(
                (batch_size, 2), 5000.0, device=pred_2d.device
            ),
        }


def test_evaluation_reports_visible_and_camera_metrics(tmp_path: Path) -> None:
    image_path = tmp_path / "frame.jpg"
    assert cv2.imwrite(str(image_path), np.zeros((256, 256, 3), dtype=np.uint8))

    samples = []
    for side in (0, 1):
        keypoints_2d = torch.zeros((21, 3), dtype=torch.float32)
        keypoints_2d[:20, 2] = 1.0
        samples.append({
            "img": torch.zeros((3, 256, 256), dtype=torch.float32),
            "keypoints_2d": keypoints_2d,
            "keypoints_3d": torch.zeros((21, 4), dtype=torch.float32),
            "crop_transform": torch.tensor(
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32
            ),
            "bbox_xyxy": torch.tensor([64.0, 64.0, 192.0, 192.0]),
            "image_size": torch.tensor([256.0, 256.0]),
            "side": side,
            "sample_id": str(side),
            "image_path": str(image_path),
            "sync_index": 10,
            "source_frame_index": 10,
            "camera": "camera2",
        })

    metrics = evaluate_glove_set(
        _EvaluationModel(),
        DataLoader(samples, batch_size=2, shuffle=False),
        torch.device("cpu"),
        tmp_path / "evaluation",
        use_amp=False,
    )
    assert metrics["samples"] == 2
    assert metrics["visible_joint_count"] == 40
    assert metrics["mean_visible_reprojection_error_px"] == pytest.approx(1.0)
    assert metrics["mean_all_reprojection_error_px"] == pytest.approx(1.0)
    assert metrics["mean_pred_cam"] == pytest.approx([1.0, 0.1, -0.1])
    assert metrics["by_side"]["left"]["samples"] == 1
    records = (tmp_path / "evaluation" / "predictions.jsonl").read_text(
        encoding="utf-8"
    )
    assert '"pred_cam"' in records and '"visibility_2d"' in records
