"""Run FAH on KB hand views with the same pixels and camera conditions as training."""
import numpy as np
import torch

from .crop_geometry import PerspectiveCrop, rgb_chw


def view_batch(images, views, device):
    """FAH's full-image metadata for already rectified square crops (identity affine)."""
    n, size = len(images), views[0].size

    def repeated(value):
        tensor = torch.as_tensor(value, dtype=torch.float32, device=device)
        return tensor.expand(n, *tensor.shape)

    extent = repeated([[size, size]])
    return dict(img=torch.from_numpy(np.stack(images)).to(device).float().div_(255).unsqueeze(1),
                img_unit_range=True, img_size=extent, ori_img_size=extent, bbox_scale=extent,
                bbox_center=extent / 2, bbox=repeated([[0, 0, size, size]]),
                cam_int=torch.as_tensor(np.stack([view.K for view in views]), dtype=torch.float32, device=device),
                affine_trans=repeated([[[1, 0, 0], [0, 1, 0]]]), mask_prompt=False,
                mask_score=torch.zeros((n, 1), device=device))


@torch.no_grad()
def predict_views(est, image, boxes, scores, side_groups, K, distortion):
    """Decode in each unmirrored virtual camera. Fit MANO here before restoring its frame."""
    size, height = est.input_size
    if height != size:
        raise ValueError("FAH perspective hand crops require square input")
    views, images, sides, confidences = [], [], [], []
    for box, score, group in zip(boxes, scores, side_groups, strict=True):
        view = PerspectiveCrop.from_box(box, K, distortion, size)
        rectified = view.rectify(image)
        for right in group:
            views.append(view)
            images.append(rgb_chw(rectified, mirror=not right))
            sides.append(right)
            confidences.append(score)
    hands = []
    for start in range(0, len(views), est.max_batch_size):
        end = start + est.max_batch_size
        batch = view_batch(images[start:end], views[start:end], est.device)
        est.model.keypoint_head.parts_on = False
        est.model.decoder_hand.active_depth = est.decoder_depth
        est.model.head_pose_hand.use_fast_hand = est.fast_mhr
        outputs = est._forward(batch)
        arrays = {k: v.detach().float().cpu().numpy() for k, v in outputs.items() if torch.is_tensor(v)}
        for key in ("affine_trans", "bbox_center", "bbox_scale"):
            arrays[key] = batch[key][:, 0].cpu().numpy()
        full_box = np.array([[0., 0., size, size]], np.float32)
        for i, view in enumerate(views[start:end]):
            single = {k: v[i:i + 1] for k, v in arrays.items()}
            hands.extend(est._results(single, full_box, full_box, [sides[start + i]],
                                      [confidences[start + i]], view.K.astype(np.float32), size, size))
    return hands, views


def native_prediction(hand, view, mano):
    """Keep FAH's independent 2D observations and the fitted physical-camera mesh."""
    xy = view.native_pixels(hand.get("keypoints_2d_mano", hand["keypoints_2d"]))
    joints = np.asarray(hand.get("keypoints_3d_mano", hand["keypoints_3d"])) @ view.rotation
    if not np.isfinite(xy).all():
        raise ValueError("FAH joint direction outside calibrated KB projection domain")
    hand = dict(is_right=hand["is_right"], keypoints_2d=xy.astype(np.float32),
                keypoints_3d=joints.astype(np.float32),
                vertices_mano=(hand["vertices_mano"] @ view.rotation).astype(np.float32))
    if mano is not None:
        root = view.physical_mano_orientation(np.asarray(mano["global_orient"]), hand["is_right"])
        mano = dict(mano, global_orient=root.tolist())
    return hand, mano
