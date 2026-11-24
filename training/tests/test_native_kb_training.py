import json
import cv2
import numpy as np
import torch

from wilor.datasets.glove_dataset import GloveDataset
from wilor.models.kb_camera import (project_kb, pixel_to_z_ray,
    physical_points_to_crop, crop_points_to_feature_pixels, native_box_depth_prior)
from wilor.models.physical_camera import PhysicalCameraHead
from wilor.models.adapter import save_trainable_checkpoint, load_trainable_weights


def test_kb_projection_ray_roundtrip_and_gradients():
    points = torch.tensor([[[0., 0., .5], [.2, -.1, .4], [-.3, .1, .3]]], requires_grad=True)
    K = torch.tensor([[[505., 0, 939], [0, 506., 592], [0, 0, 1.]]])
    D = torch.tensor([[.076, -.014, .0019, .00007]])
    pixels = project_kb(points, K, D)
    expected = cv2.fisheye.projectPoints(points.detach().numpy().astype(float).reshape(-1, 1, 3),
        np.zeros(3), np.zeros(3), K[0].numpy().astype(float), D[0].numpy().astype(float))[0][:, 0]
    np.testing.assert_allclose(pixels.detach().numpy()[0], expected, atol=1e-4)
    for i in range(points.shape[1]):
        ray = pixel_to_z_ray(pixels[:, i], K, D)
        torch.testing.assert_close(ray, points[:, i]/points[:, i, 2:], atol=1e-5, rtol=1e-5)
    pixels.sum().backward()
    assert torch.isfinite(points.grad).all()


def test_native_two_hand_crop_targets_match_model_projection(tmp_path):
    (tmp_path/'images').mkdir(); (tmp_path/'labels').mkdir()
    cv2.imwrite(str(tmp_path/'images/frame.jpg'), np.full((240, 320, 3), 100, np.uint8))
    K = np.array([[150., 0, 160], [0, 152., 120], [0, 0, 1]], np.float32)
    D = np.array([.07, -.01, .001, 0.], float)
    rows = []
    for side in (0, 1):
        joints = np.zeros((21, 3), np.float32)
        joints[:, 0] = np.linspace(-.04, .04, 21)
        joints[:, 1] = np.linspace(-.03, .03, 21)
        trans = np.array([-.08 if side == 0 else .08, 0., .5], np.float32)
        label = dict(bbox=np.array([70, 50, 250, 190]), joints_3d=joints, trans=trans,
            K=K, side=side, image_space='native_raw', camera_model='KB', distortion=D)
        np.save(tmp_path/f'labels/{side}.npy', label)
        rows.append(dict(index=side, image='images/frame.jpg', label=f'labels/{side}.npy',
            side=side, sync_index=0, camera='camera0'))
    (tmp_path/'index.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    data = GloveDataset(tmp_path/'index.jsonl', camera_model='physical_kb')
    for side in (0, 1):
        sample = data[side]
        batch = {key: value[None] for key, value in sample.items() if isinstance(value, torch.Tensor)}
        batch['side'] = torch.tensor([side])
        joints = batch['keypoints_3d'][..., :3]
        predicted = physical_points_to_crop(joints, batch['camera_root_m'], joints[:, :1], batch)
        torch.testing.assert_close(predicted[0], sample['keypoints_2d'][:, :2], atol=1e-5, rtol=1e-5)
    assert data.rows[0]['image'] == data.rows[1]['image']


def test_feature_sampling_maps_center_slice_correctly():
    points = torch.tensor([[[-.375, -.5], [0., 0.], [.375, .5]]])
    expected = torch.tensor([[[0., 0.], [11.5, 15.5], [23., 31.]]])
    torch.testing.assert_close(crop_points_to_feature_pixels(points, 32, 24), expected)


class CameraModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torch.nn.Linear(1, 1)
        self.physical_camera_head = PhysicalCameraHead()
        self.physical_camera_enabled = True
        self.marker_head_enabled = False


def test_native_checkpoint_restores_camera_mode(tmp_path):
    model = CameraModel()
    model.physical_camera_head.camera_model = 'physical_kb'
    optimizer = torch.optim.AdamW(model.parameters())
    path = tmp_path/'checkpoint.pt'
    save_trainable_checkpoint(path, model=model, optimizer=optimizer, step=1,
        base_checkpoint=tmp_path/'base.ckpt', recipe='joint_lora_finetune')
    restored = CameraModel()
    load_trainable_weights(path, restored)
    assert restored.physical_camera_head.camera_model == 'physical_kb'
    assert restored.physical_camera_head.native_root_prior == 'bbox_geometry'


def test_native_depth_prior_uses_predicted_geometry_and_box_only():
    joints = torch.zeros(1, 21, 3)
    joints[0, :, 0] = torch.linspace(0, .08, 21)
    joints[0, :, 1] = torch.linspace(0, .06, 21)
    joints.requires_grad_()
    K = torch.tensor([[[505., 0., 939.], [0., 506., 592.], [0., 0., 1.]]])
    D = torch.tensor([[.076, -.014, .0019, .00007]])
    pixels = project_kb(joints.detach() + torch.tensor([0., 0., .5]), K, D)
    extent = (pixels.amax(1) - pixels.amin(1)) / .85
    center = (pixels.amax(1) + pixels.amin(1)) / 2
    batch = dict(camera_K=K, camera_distortion=D, side=torch.tensor([1]),
                 bbox_xyxy=torch.cat((center-extent/2, center+extent/2), -1))
    wrist = K[:, :2, 2]
    depth = native_box_depth_prior(joints, wrist, torch.tensor([.8]), batch)
    torch.testing.assert_close(depth, torch.tensor([.5]), atol=.005, rtol=.01)
    batch['camera_root_m'] = torch.full((1, 3), float('nan'))
    batch['keypoints_3d'] = torch.full((1, 21, 4), float('nan'))
    torch.testing.assert_close(depth, native_box_depth_prior(joints, wrist, torch.tensor([.8]), batch))
    batch['bbox_xyxy'] = torch.cat((center-extent, center+extent), -1)
    nearer = native_box_depth_prior(joints, wrist, torch.tensor([.8]), batch)
    assert nearer.item() < depth.item() * .7
    depth.sum().backward()
    assert torch.isfinite(joints.grad).all()
