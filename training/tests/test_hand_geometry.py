from __future__ import annotations

import pytest
import torch

from wilor.utils.hand_geometry import palm_scale_normalize


def _hand() -> torch.Tensor:
    generator = torch.Generator().manual_seed(4)
    joints = torch.randn((2, 21, 3), generator=generator) * 0.1
    joints[:, 0] = 0
    joints[:, 5] = torch.tensor([1.0, 0.1, 0.0])
    joints[:, 9] = torch.tensor([0.1, 1.2, 0.2])
    joints[:, 17] = torch.tensor([-0.8, -0.1, 0.1])
    return joints


def test_palm_frame_removes_translation_rotation_and_scale() -> None:
    joints = _hand()
    angle = torch.tensor(0.73)
    rotation = torch.tensor([
        [torch.cos(angle), -torch.sin(angle), 0.0],
        [torch.sin(angle), torch.cos(angle), 0.0],
        [0.0, 0.0, 1.0],
    ])
    transformed = joints @ rotation.T * 3.7 + torch.tensor([5.0, -2.0, 9.0])
    torch.testing.assert_close(
        palm_scale_normalize(joints),
        palm_scale_normalize(transformed),
        atol=2e-6,
        rtol=2e-6,
    )


def test_palm_frame_rejects_degenerate_landmarks() -> None:
    with pytest.raises(ValueError, match="non-degenerate"):
        palm_scale_normalize(torch.zeros((1, 21, 3)))
