"""Differentiable native KB camera geometry; no image undistortion/remapping."""
import torch


def project_kb(points, K, distortion):
    xy = points[..., :2]
    radius = torch.sqrt(xy.square().sum(-1) + 1e-14)
    theta = torch.atan2(radius, points[..., 2].clamp_min(0.01))
    theta2 = theta.square()
    d = distortion[:, None, :]
    distorted = theta * (1 + d[..., 0]*theta2 + d[..., 1]*theta2**2
                        + d[..., 2]*theta2**3 + d[..., 3]*theta2**4)
    normalized = xy * (distorted / radius)[..., None]
    return torch.stack((K[:, None, 0, 0]*normalized[..., 0]+K[:, None, 0, 2],
                        K[:, None, 1, 1]*normalized[..., 1]+K[:, None, 1, 2]), -1)


def pixel_to_z_ray(pixels, K, distortion):
    """Invert the KB angular polynomial for a ray with z=1, without resampling."""
    xy = (pixels - K[:, :2, 2]) / torch.stack((K[:, 0, 0], K[:, 1, 1]), -1)
    radius = torch.sqrt(xy.square().sum(-1) + 1e-14)
    theta = radius.clamp(0, 1.55)
    for _ in range(10):
        t2 = theta.square()
        value = theta * (1 + distortion[:, 0]*t2 + distortion[:, 1]*t2**2
                         + distortion[:, 2]*t2**3 + distortion[:, 3]*t2**4)
        derivative = (1 + 3*distortion[:, 0]*t2 + 5*distortion[:, 1]*t2**2
                      + 7*distortion[:, 2]*t2**3 + 9*distortion[:, 3]*t2**4)
        theta = (theta - (value-radius)/derivative.clamp_min(0.05)).clamp(0, 1.55)
    normalized = xy * (torch.tan(theta)/radius)[:, None]
    return torch.cat((normalized, torch.ones_like(normalized[:, :1])), -1)


def physical_points_to_crop(points, root, root_joint, batch, kb=True):
    """Canonical joints/vertices -> native physical pixels -> training crop."""
    left = batch['side'] == 0
    sign = torch.ones((points.shape[0], 1, 3), device=points.device, dtype=torch.float32)
    sign[:, :, 0] = torch.where(left[:, None], -1., 1.)
    camera = (points.float() - root_joint.float()) * sign + root.float()[:, None]
    K = batch['camera_K'].float()
    if kb:
        pixels = project_kb(camera, K, batch['camera_distortion'].float())
    else:
        projected = torch.bmm(camera, K.transpose(1, 2))
        pixels = projected[..., :2] / projected[..., 2:].clamp_min(0.01)
    x = torch.where(left[:, None], batch['image_size'][:, :1]-pixels[..., 0]-1, pixels[..., 0])
    pixels = torch.stack((x, pixels[..., 1]), -1)
    affine = batch['crop_transform'].float()
    # Full-resolution pixel coordinates need float32 even during BF16 training.
    with torch.autocast(device_type=points.device.type, enabled=False):
        crop = torch.bmm(pixels.float(), affine[:, :, :2].transpose(1, 2)) + affine[:, None, :, 2]
    return crop / float(batch['img'].shape[-1]) - .5


def crop_points_to_feature_pixels(points, feature_height, feature_width):
    """Full 256-square crop coordinates -> feature map of the center 192 crop."""
    x = ((points[..., 0] + .5) * 256. - 32.) / 192. * (feature_width-1)
    y = (points[..., 1] + .5) * (feature_height-1)
    return torch.stack((x, y), -1)


def bounded_wrist_pixel(pixel, bbox):
    center = (bbox[:, :2] + bbox[:, 2:]) * .5
    radius = ((bbox[:, 2:] - bbox[:, :2]) * .5).clamp_min(2.)
    return center + radius * torch.tanh((pixel-center)/radius)


def native_box_depth_prior(joints, wrist_pixel, initial_depth, batch):
    """Estimate metric scale from predicted MANO size and the observed box.

    Uses no ground-truth root or joint target. The factor accounts for box
    padding and the difference between a mesh outline and joint extent.
    """
    K = batch['camera_K'].float()
    D = batch['camera_distortion'].float()
    ray = pixel_to_z_ray(wrist_pixel, K, D)
    relative = joints.float() - joints[:, :1].float()
    relative = relative.clone()
    relative[..., 0] *= torch.where(batch['side'][:, None] == 0, -1., 1.)
    box_wh = (batch['bbox_xyxy'][:, 2:] - batch['bbox_xyxy'][:, :2]).float().clamp_min(2.)
    depth = initial_depth
    for _ in range(2):
        pixels = project_kb(relative + (ray*depth[:, None])[:, None], K, D)
        extent = pixels.amax(1) - pixels.amin(1)
        scale = (extent / (box_wh * .85)).amax(-1).clamp(.2, 5.)
        depth = (depth * scale).clamp(.05, 3.)
    return depth
