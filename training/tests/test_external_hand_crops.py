"""Catch incorrect KB rectification, root translation and side conventions."""
import json

import cv2
import numpy as np
import pytest
import torch

from tools.hand_finetune.data import HandCrops, project


@pytest.mark.parametrize("side", [0, 1])
@pytest.mark.parametrize("canonical_side,size", [(1, 320), (0, 256)])
def test_native_pixel_matches_projected_label(tmp_path, side, canonical_side, size):
    K = np.array([[200., 0., 300.], [0., 210., 200.], [0., 0., 1.]])
    distortion = np.array([0.1, -0.01, 0.003, 0.0002])
    wrist = np.array([0.44, -0.17, 0.38])
    fingertip = np.array([0.48, -0.15, 0.37])
    xy, _ = cv2.fisheye.projectPoints(np.array([[wrist, fingertip]]), np.zeros(3), np.zeros(3), K, distortion)
    xy = xy[0]
    box = np.r_[xy.min(axis=0) - 12, xy.max(axis=0) + 12]
    image = np.zeros((400, 600, 3), dtype=np.uint8)
    for pixel, color in zip(xy, [(0, 0, 255), (0, 255, 0)]):
        cv2.circle(image, tuple(np.round(pixel).astype(int)), 2, color, -1)
    assert cv2.imwrite(str(tmp_path / "frame.png"), image)
    trans = np.array([0.21, -0.15, 0.24])
    joints = np.tile(fingertip - trans, (21, 1))
    joints[0] = wrist - trans
    np.save(tmp_path / "hand.npy", dict(camera_model="KB", image_space="native_raw", side=side,
            K=K, distortion=distortion, bbox=box, trans=trans, joints_3d=joints,
            vertices=np.tile(wrist-trans, (778, 1))))
    row = dict(image="frame.png", label="hand.npy", sync_index=1, source_group="example")
    (tmp_path / "val.jsonl").write_text(json.dumps(row) + "\n")

    sample = HandCrops(tmp_path, "val", size, canonical_side)[0]
    xyz = sample["points"][:2]
    torch.testing.assert_close(xyz.norm(dim=1), torch.from_numpy(np.linalg.norm([wrist, fingertip], axis=1)).float())
    pixels = project(xyz[None], sample["K"][None])[0].round().long()
    for channel, (x, y) in enumerate(pixels):
        assert sample["rgb"][channel, y, x] > 150
    assert bool(pixels[1, 0] > pixels[0, 0]) == (side == canonical_side)
