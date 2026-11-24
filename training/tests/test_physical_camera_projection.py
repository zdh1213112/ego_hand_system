"""The calibrated projection must agree with the image and crop convention."""

import torch

from wilor.models.physical_camera import project_rectified_joints


def test_rectified_projection_and_left_hand_flip():
    joints = torch.zeros(2, 21, 3)
    joints[0, 1] = torch.tensor([0.1, 0.02, 0.03])
    joints[1, 1] = torch.tensor([-0.1, 0.02, 0.03])
    root = torch.tensor([[0.2, 0.1, 0.5], [0.2, 0.1, 0.5]])
    K = torch.tensor([[[300., 0., 960.], [0., 300., 600.], [0., 0., 1.]]]).expand(2, -1, -1)
    affine = torch.tensor([[[0.5, 0., -300.], [0., 0.5, -200.]]]).expand(2, -1, -1)
    width = torch.full((2,), 1920.)
    left = torch.tensor([False, True])

    projected = project_rectified_joints(joints, root, K, affine, width, left, 256)
    camera_joint = torch.tensor([0.3, 0.12, 0.53])
    source = torch.tensor([
        300 * camera_joint[0] / camera_joint[2] + 960,
        300 * camera_joint[1] / camera_joint[2] + 600,
    ])
    expected_right = (source * 0.5 + torch.tensor([-300., -200.])) / 256 - 0.5
    mirrored = torch.tensor([1919. - source[0], source[1]])
    expected_left = (mirrored * 0.5 + torch.tensor([-300., -200.])) / 256 - 0.5
    torch.testing.assert_close(projected[0, 1], expected_right, atol=1e-5, rtol=0)
    torch.testing.assert_close(projected[1, 1], expected_left, atol=1e-5, rtol=0)
