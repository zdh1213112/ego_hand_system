"""Differentiable full mesh silhouettes on original KB camera images.

The only image resampling is a fixed training ROI; calibration and exported geometry
remain in the real camera's distorted image coordinates.
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'output/dependencies/nvdiffrast_runtime'))
import nvdiffrast.torch as dr


def load_hand_mask(path):
    rgba = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if rgba is None:
        raise FileNotFoundError(path)
    mask = (rgba[:, :, 3] > 0).astype(np.uint8)
    # White-glove color segmentation leaves black marker/seam speckles as holes.
    # Fill only enclosed <=64-pixel, <=16px-wide holes. Large pinch holes and all
    # gaps connected to the image exterior are preserved.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(1-mask, 8)
    for k in range(1, count):
        x, y, w, h, area = stats[k]
        if x > 0 and y > 0 and x+w < mask.shape[1] and y+h < mask.shape[0] and area <= 64 and max(w, h) <= 16:
            mask[labels == k] = 1
    return mask


def project_native(vertices, records, side):
    device, dtype = vertices.device, vertices.dtype
    to = lambda x: torch.as_tensor(np.asarray(x), dtype=dtype, device=device)
    t0 = to([r['T_base_cam0'] for r in records])
    tc = to([r['T_base_camera'] for r in records])
    mirror = to([-1 if side == 0 else 1, 1, 1])
    base = torch.einsum('bvi,bji->bvj', vertices * mirror, t0[:, :3, :3]) + t0[:, None, :3, 3]
    camera = torch.einsum('bvi,bij->bvj', base - tc[:, None, :3, 3], tc[:, :3, :3])
    xy = camera[..., :2] / camera[..., 2:].clamp_min(1e-5)
    radius = xy.norm(dim=-1).clamp_min(1e-9)
    theta = torch.atan(radius)
    d = to([np.asarray(r['D']).reshape(4) for r in records])
    td = theta * (1 + d[:, 0, None] * theta**2 + d[:, 1, None] * theta**4
                  + d[:, 2, None] * theta**6 + d[:, 3, None] * theta**8)
    k = to([r['K'] for r in records])
    uv = xy * (td / radius)[..., None]
    uv = uv * torch.stack([k[:, 0, 0], k[:, 1, 1]], -1)[:, None] + k[:, None, :2, 2]
    return uv, camera[..., 2]


def raster_mask(context, uv, depth, faces, boxes, resolution):
    # OpenCV pixel centers are integer; nvdiffrast follows bottom-up OpenGL images.
    center = boxes[:, None, :2]
    size = boxes[:, None, 2:3]
    xy = (uv + .5 - center) / size * 2 - 1
    clip = torch.cat([xy[..., :1], -xy[..., 1:], depth[..., None] / 3 - .9,
                      torch.ones_like(depth[..., None])], -1).contiguous()
    rast, _ = dr.rasterize(context, clip, faces, resolution=[resolution, resolution], grad_db=False)
    hard = (rast[..., 3:] > 0).to(clip.dtype)
    alpha = dr.antialias(hard.contiguous(), rast, clip, faces)
    return alpha[..., 0].flip(1)


class FullMaskBatch:
    def __init__(self, records, lookup, source_joints, source_vertices, faces, side, resolution=256, device='cuda'):
        self.records, self.side, self.resolution = records, side, resolution
        self.device = device
        self.frame = torch.tensor([lookup[r['sync']] for r in records], device=device)
        self.faces = torch.as_tensor(faces.astype(np.int32), device=device).contiguous()
        self.context = dr.RasterizeCudaContext(device=device)
        self.boxes, targets, domains, outside, inside = [], [], [], [], []
        counts = {sync: sum(r['sync'] == sync for r in records) for sync in {r['sync'] for r in records}}
        for r in records:
            i = lookup[r['sync']]
            joints, _ = project_native(torch.as_tensor(source_joints[i:i+1], dtype=torch.float64, device=device), [r], side)
            vertices, _ = project_native(torch.as_tensor(source_vertices[i:i+1], dtype=torch.float64, device=device), [r], side)
            j, v = joints[0].cpu().numpy(), vertices[0].cpu().numpy()
            axis = j[[5, 9, 13, 17]].mean(0) - j[0]
            axis /= max(np.linalg.norm(axis), 1e-9)
            mask = load_hand_mask(r['mask']).astype(np.float32)
            ys, xs = np.nonzero(mask)
            distal = (np.c_[xs, ys] - j[0]) @ axis > 8
            points = np.r_[np.c_[xs[distal], ys[distal]], v]
            low, high = points.min(0) - 45, points.max(0) + 45
            size = float(max(high-low))
            low = (low+high)/2-size/2
            box = np.r_[low, size]
            self.boxes.append(box)
            row, col = np.mgrid[:resolution, :resolution].astype(np.float32)
            x, y = low[0]+(col+.5)*size/resolution-.5, low[1]+(row+.5)*size/resolution-.5
            target = cv2.remap(mask, x.astype(np.float32), y.astype(np.float32), cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            # Exclude cuff/forearm and the artificial wrist cut; retain all distal hand pixels.
            domain = (((x-j[0, 0])*axis[0]+(y-j[0, 1])*axis[1]) > 8).astype(np.float32)
            domain *= (x >= 0) & (x < mask.shape[1]-1) & (y >= 0) & (y < mask.shape[0]-1)
            binary = (target >= .5).astype(np.uint8)
            outside.append(cv2.distanceTransform(1-binary, cv2.DIST_L2, 5)*size/resolution)
            inside.append(cv2.distanceTransform(binary, cv2.DIST_L2, 5)*size/resolution)
            targets.append(target)
            domains.append(domain)
        to = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float32, device=device)
        self.boxes = to(self.boxes)
        self.target, self.domain = to(targets), to(domains)
        self.outside, self.inside = to(outside).clamp_max(30), to(inside).clamp_max(30)
        self.weights = to([1/counts[r['sync']] for r in records])
        self.weights /= self.weights.sum()
        self.target_area = (self.target*self.domain).sum((1, 2)).clamp_min(1)
        self.last_prediction = None

    def render(self, vertices):
        uv, depth = project_native(vertices[self.frame], self.records, self.side)
        return raster_mask(self.context, uv, depth, self.faces, self.boxes, self.resolution)

    def terms(self, vertices):
        pred = self.render(vertices).clamp(0, 1)
        intersection = (pred*self.target*self.domain).sum((1, 2))
        union = ((pred+self.target-pred*self.target)*self.domain).sum((1, 2)).clamp_min(1)
        iou = intersection/union
        sdf = ((pred*(1-self.target)*self.outside + (1-pred)*self.target*self.inside)*self.domain).sum((1, 2))/self.target_area/5
        return ((1-iou)*self.weights).sum(), (sdf*self.weights).sum(), (iou*self.weights).sum()
