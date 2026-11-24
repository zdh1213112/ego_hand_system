from __future__ import annotations

from pathlib import Path

import pytest
import torch

from wilor.models.losses import Keypoint2DLoss, Keypoint3DLoss, ParameterLoss
from wilor.training.losses import GloveLossComputer, load_glove_loss_config


REPO_ROOT = Path(__file__).resolve().parents[1]


class _LossModel:
    keypoint_2d_loss = Keypoint2DLoss("l1")
    keypoint_3d_loss = Keypoint3DLoss("l1")
    mano_parameter_loss = ParameterLoss()


def _profile(parameters: bool = False) -> dict:
    return {
        "supervision": {
            "keypoints_2d": {"available": True, "reason": "ok"},
            "keypoints_3d": {"available": True, "reason": "ok"},
            "global_orient": {"available": parameters, "reason": "native"},
            "hand_pose": {"available": parameters, "reason": "native"},
            "betas": {"available": parameters, "reason": "native"},
        }
    }


def test_stage1_weights_are_profile_gated() -> None:
    config = load_glove_loss_config(
        REPO_ROOT / "configs" / "glove" / "stage1_keypoints.yaml",
        _profile(),
    )
    assert config.effective == {
        "keypoints_2d": 0.01,
        "keypoints_3d": 0.05,
        "coarse_keypoints_2d": 0.005,
        "coarse_keypoints_3d": 0.025,
        "global_orient": 0.0,
        "hand_pose": 0.0,
        "betas": 0.0,
    }


def test_loss_computer_reuses_final_and_coarse_keypoint_primitives() -> None:
    config = load_glove_loss_config(
        REPO_ROOT / "configs" / "glove" / "stage1_keypoints.yaml",
        _profile(),
    )
    target_2d = torch.zeros((1, 21, 3))
    target_2d[..., 2] = 1
    target_3d = torch.zeros((1, 21, 4))
    target_3d[..., 3] = 1
    prediction_2d = torch.ones((1, 21, 2), requires_grad=True)
    prediction_3d = torch.ones((1, 21, 3), requires_grad=True)
    output = {
        "pred_keypoints_2d": prediction_2d,
        "pred_keypoints_3d": prediction_3d,
        "coarse_keypoints_2d": prediction_2d * 2,
        "coarse_keypoints_3d": prediction_3d * 2,
        "pred_mano_params": {},
    }
    loss, components = GloveLossComputer(_LossModel(), config)(
        {
            "keypoints_2d": target_2d,
            "keypoints_3d": target_3d,
            "mano_params": {},
            "has_mano_params": {},
        },
        output,
    )
    expected = sum(
        components[name]
        for name in (
            "weighted_keypoints_2d", "weighted_keypoints_3d",
            "weighted_coarse_keypoints_2d", "weighted_coarse_keypoints_3d",
        )
    )
    torch.testing.assert_close(loss.detach(), expected)
    loss.backward()
    assert prediction_2d.grad is not None
    assert prediction_3d.grad is not None


def test_loss_config_rejects_unknown_keys(tmp_path: Path) -> None:
    path = tmp_path / "loss.yaml"
    path.write_text("schema_version: 1\nlosses:\n  vertex: 1\n")
    with pytest.raises(ValueError, match="exactly"):
        load_glove_loss_config(path, _profile())


def test_loss_primitives_average_only_over_batch() -> None:
    loss_2d = Keypoint2DLoss("l1")
    gt_2d = torch.zeros((1, 21, 3))
    gt_2d[..., 2] = 1
    pred_2d = torch.ones((1, 21, 2), requires_grad=True)
    single_2d = loss_2d(pred_2d, gt_2d)
    repeated_2d = loss_2d(
        pred_2d.detach().repeat(4, 1, 1),
        gt_2d.repeat(4, 1, 1),
    )
    torch.testing.assert_close(single_2d, repeated_2d)
    torch.testing.assert_close(single_2d, torch.tensor(42.0))

    loss_3d = Keypoint3DLoss("l1")
    gt_3d = torch.zeros((1, 21, 4))
    gt_3d[..., 3] = 1
    pred_3d = torch.zeros((1, 21, 3))
    pred_3d[:, 1:, 0] = 1
    single_3d = loss_3d(pred_3d, gt_3d)
    repeated_3d = loss_3d(pred_3d.repeat(4, 1, 1), gt_3d.repeat(4, 1, 1))
    torch.testing.assert_close(single_3d, repeated_3d)
    torch.testing.assert_close(single_3d, torch.tensor(20.0))

    parameter_loss = ParameterLoss()
    pred_param = torch.ones((1, 10))
    gt_param = torch.zeros((1, 10))
    has_param = torch.ones((1,), dtype=torch.bool)
    single_param = parameter_loss(pred_param, gt_param, has_param)
    repeated_param = parameter_loss(
        pred_param.repeat(4, 1),
        gt_param.repeat(4, 1),
        has_param.repeat(4),
    )
    torch.testing.assert_close(single_param, repeated_param)
    torch.testing.assert_close(single_param, torch.tensor(10.0))
