"""Detector-box perspective views of a native Kannala–Brandt (KB4) camera."""
from dataclasses import dataclass
from functools import lru_cache

import cv2
import numpy as np


def _radius(theta, distortion):
    k1, k2, k3, k4 = distortion
    t = theta * theta
    return theta * (1 + t * (k1 + t * (k2 + t * (k3 + t * k4))))


@lru_cache(maxsize=64)
def _theta_limit(distortion):
    # The angular polynomial is invertible up to its first stationary radius.
    roots = np.polynomial.polynomial.polyroots([1., *(np.array(distortion) * [3, 5, 7, 9])])
    return min([np.pi] + [float(np.sqrt(r.real)) for r in roots
                          if abs(r.imag) < 1e-10 and 0 < r.real < np.pi**2])


def _native_pixels(rays, K, distortion):
    r = np.hypot(rays[..., 0], rays[..., 1])
    theta = np.arctan2(r, rays[..., 2])
    scale = np.divide(_radius(theta, distortion), r, out=np.zeros_like(r), where=r > 0)
    xy = rays[..., :2] * scale[..., None] * K.diagonal()[:2] + K[:2, 2]
    return np.where((theta < _theta_limit(tuple(distortion)))[..., None], xy, np.nan)


def crop_camera(box, K, distortion, size, padding=1.2):
    """Return physical-to-view rotation and intrinsics, using only the box and calibration."""
    x1, y1, x2, y2 = np.asarray(box, dtype=np.float64)
    if not (x2 > x1 and y2 > y1 and size > 1 and padding > 0):
        raise ValueError(f"Invalid KB crop geometry for box {box}")
    x, y = (x1 + x2) / 2, (y1 + y2) / 2
    pixels = np.array([(x, y), (x1, y1), (x2, y1), (x2, y2), (x1, y2),
                       (x1, y), (x2, y), (x, y1), (x, y2)])
    xy = (pixels - K[:2, 2]) / K.diagonal()[:2]
    radius = np.linalg.norm(xy, axis=1)
    limit = _theta_limit(tuple(distortion))
    if not np.all(radius < _radius(limit, distortion)):
        raise ValueError(f"Box outside invertible KB calibration: {box}")
    lo, hi = np.zeros_like(radius), np.full_like(radius, limit)
    for _ in range(44):
        mid = (lo + hi) / 2
        below = _radius(mid, distortion) < radius
        lo, hi = np.where(below, mid, lo), np.where(below, hi, mid)
    theta = (lo + hi) / 2
    rays = np.column_stack((xy * np.divide(np.sin(theta), radius, out=np.ones_like(radius),
                                           where=radius > 0)[:, None], np.cos(theta)))
    z = rays[0]
    up = [0., 1., 0.] if abs(z[1]) < 1 - 1e-12 else [0., 0., 1.]
    horizontal = np.cross(up, z)
    horizontal /= np.linalg.norm(horizontal)
    rotation = np.stack((horizontal, np.cross(z, horizontal), z))
    aligned = rays[1:] @ rotation.T
    if not np.isfinite(aligned).all() or np.any(aligned[:, 2] <= 0):
        raise ValueError(f"Invalid KB crop geometry for box {box}")
    extent = np.abs(aligned[:, :2] / aligned[:, 2:]).max()
    center = (size - 1) / 2
    focal = center / (padding * extent)
    intrinsics = np.array([[focal, 0., center], [0., focal, center], [0., 0., 1.]])
    return rotation, intrinsics


@lru_cache(maxsize=8)
def _grid(size):
    v, u = np.indices((size, size), dtype=np.float64)
    return np.stack((u, v), axis=-1)


@dataclass
class PerspectiveCrop:
    rotation: np.ndarray
    K: np.ndarray
    source_K: np.ndarray
    distortion: np.ndarray
    size: int

    @classmethod
    def from_box(cls, box, K, distortion, size):
        K = np.asarray(K, dtype=np.float64)
        distortion = np.asarray(distortion, dtype=np.float64).reshape(4)
        if (K.shape != (3, 3) or not np.isfinite(K).all() or not np.isfinite(distortion).all()
                or min(K[0, 0], K[1, 1]) <= 0 or not np.allclose(K[2], [0, 0, 1])
                or not np.allclose([K[0, 1], K[1, 0]], 0)):
            raise ValueError("Expected finite KB4 calibration and zero-skew intrinsics")
        rotation, intrinsics = crop_camera(box, K, distortion, size)
        return cls(rotation, intrinsics, K, distortion, size)

    def native_pixels(self, view_pixels):
        """Unmirrored perspective pixels -> physical rays -> native fisheye pixels."""
        xy = (np.asarray(view_pixels) - self.K[:2, 2]) / self.K.diagonal()[:2]
        rays = np.concatenate((xy, np.ones_like(xy[..., :1])), axis=-1)
        return _native_pixels(rays @ self.rotation, self.source_K, self.distortion)

    def rectify(self, image):
        # Physical z is affine in view x,y, so its minimum is attained at a corner.
        extent = self.K[0, 2] / self.K[0, 0]
        min_z = self.rotation[2, 2] - extent * np.abs(self.rotation[:2, 2]).sum()
        if min_z > 0 and _theta_limit(tuple(self.distortion)) >= np.pi / 2:
            maps = cv2.fisheye.initUndistortRectifyMap(
                self.source_K, self.distortion, self.rotation, self.K,
                (self.size, self.size), cv2.CV_32FC1)
            return cv2.remap(image, *maps, cv2.INTER_LINEAR)
        # Spherical projection also covers valid rays past 90 degrees near the image edge.
        coords = self.native_pixels(_grid(self.size))
        maps = np.where(np.isfinite(coords), coords, -1).astype(np.float32)
        return cv2.remap(image, maps, None, cv2.INTER_LINEAR)

    def view_points(self, points, mirror):
        view = np.asarray(points) @ self.rotation.T
        if mirror:
            view[..., 0] *= -1
        return view

    def physical_mano_orientation(self, root, right):
        # MANO parameters are stored in the canonical right-hand convention for both sides.
        inverse = self.rotation.T
        if not right:
            reflection = np.diag([-1., 1., 1.])
            inverse = reflection @ inverse @ reflection
        return inverse @ root


def rgb_chw(view_bgr, mirror):
    view = view_bgr[:, ::-1] if mirror else view_bgr
    return np.ascontiguousarray(view[:, :, ::-1].transpose(2, 0, 1))
