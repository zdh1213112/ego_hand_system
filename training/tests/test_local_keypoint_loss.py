from __future__ import annotations

import torch

from wilor.training.losses import local_keypoint_3d_loss


def test_local_loss_is_batch_mean_and_honors_validity() -> None:
    target = torch.zeros((2, 21, 3))
    prediction = target.clone()
    prediction[0] = 0.02
    prediction[1] = 100.0
    valid = torch.tensor([True, False])
    single = local_keypoint_3d_loss(prediction, target, valid, beta=0.01)
    repeated = local_keypoint_3d_loss(
        prediction[:1].repeat(4, 1, 1), target[:1].repeat(4, 1, 1), beta=0.01
    )
    torch.testing.assert_close(single, repeated)
    torch.testing.assert_close(single, torch.tensor(0.015))


def test_local_loss_keeps_gradient_for_empty_valid_batch() -> None:
    prediction = torch.ones((2, 21, 3), requires_grad=True)
    loss = local_keypoint_3d_loss(
        prediction, torch.zeros_like(prediction), torch.tensor([False, False])
    )
    loss.backward()
    assert prediction.grad is not None
    assert not bool(prediction.grad.any())
