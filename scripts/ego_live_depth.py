"""Bounded, calibrated cam0/cam1 stereo depth for the live viewer.

This is estimated stereo depth, not a hardware depth-sensor measurement.
All expensive rectification and matching runs on a separate CPU thread.
"""

from __future__ import annotations

from collections import deque
import threading

import cv2
import numpy as np


class LiveStereoDepth:
    def __init__(self, cameras, size=(640, 480)):
        left, right = cameras["cam_0"], cameras["cam_1"]
        if left.model != "KB" or right.model != "KB":
            raise ValueError("live stereo depth requires KB4 camera calibration")
        T = np.linalg.inv(right.T_base_camera) @ left.T_base_camera
        baseline = np.linalg.norm(T[:3, 3])
        if not 0.005 < baseline < 1:
            raise ValueError("invalid stereo camera baseline")
        R0, R1, P0, P1, _ = cv2.fisheye.stereoRectify(
            left.K, left.distortion, right.K, right.distortion, left.image_size,
            T[:3, :3], T[:3, 3], flags=cv2.CALIB_ZERO_DISPARITY,
            newImageSize=size, balance=0.0, fov_scale=1 / 1.5,
        )
        if P1[0, 3] >= 0:
            raise ValueError("live stereo calibration must produce positive disparity")
        self.focal_baseline = float(-P1[0, 3])
        self.K_rectified = P0[:3, :3].tolist()
        self.R_rectified_from_cam0 = R0.tolist()
        self.maps = []
        self.masks = []
        for camera, R, P in ((left, R0, P0), (right, R1, P1)):
            maps = cv2.fisheye.initUndistortRectifyMap(
                camera.K, camera.distortion, R, P, size, cv2.CV_32FC1)
            self.maps.append(maps)
            width, height = camera.image_size
            self.masks.append((maps[0] >= 0) & (maps[0] < width - 1)
                              & (maps[1] >= 0) & (maps[1] < height - 1))
        n, block = 96, 5
        common = dict(numDisparities=n, blockSize=block, P1=8 * block * block,
                      P2=32 * block * block, disp12MaxDiff=1, uniquenessRatio=10,
                      speckleWindowSize=100, speckleRange=2, preFilterCap=31,
                      mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
        self.left_matcher = cv2.StereoSGBM_create(minDisparity=0, **common)
        self.right_matcher = cv2.StereoSGBM_create(minDisparity=1 - n, **common)
        self.pending = deque(maxlen=1)
        self.condition = threading.Condition()
        self.latest = None
        self.last_sent = None
        self.closing = False
        self.error = None
        self.thread = threading.Thread(target=self._run, name="ego-live-depth", daemon=True)
        self.thread.start()

    def submit(self, timestamp_us, frames):
        with self.condition:
            self.pending.append((int(timestamp_us), frames["cam_0"], frames["cam_1"]))
            self.condition.notify()

    def take(self, current_timestamp_us):
        with self.condition:
            result = self.latest
            if result is None or result[0] == self.last_sent or result[0] > current_timestamp_us:
                return None
            self.last_sent = result[0]
            return result

    def _run(self):
        try:
            while True:
                with self.condition:
                    while not self.pending and not self.closing:
                        self.condition.wait()
                    if self.closing:
                        return
                    timestamp_us, left, right = self.pending.pop()
                rectified = [cv2.remap(frame, *maps, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                             for frame, maps in zip((left, right), self.maps)]
                gray = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in rectified]
                left_disp = self.left_matcher.compute(*gray).astype(np.float32) / 16
                right_disp = self.right_matcher.compute(gray[1], gray[0]).astype(np.float32) / 16
                h, w = left_disp.shape
                yy, xx = np.indices((h, w))
                xr = np.rint(xx - left_disp).astype(np.int32)
                inside = (xr >= 0) & (xr < w)
                safe = np.clip(xr, 0, w - 1)
                opposite = right_disp[yy, safe]
                depth_m = self.focal_baseline / np.maximum(left_disp, 1e-6)
                valid = (inside & (left_disp > 0) & (opposite < 0) & (opposite > -96)
                         & (np.abs(left_disp + opposite) <= 1.5)
                         & self.masks[0] & self.masks[1][yy, safe]
                         & (depth_m >= .2) & (depth_m <= 8.0))
                depth_mm = np.zeros((h, w), dtype=np.uint16)
                depth_mm[valid] = np.rint(depth_m[valid] * 1000).astype(np.uint16)
                ok, png = cv2.imencode(".png", depth_mm)
                if not ok:
                    raise ValueError("failed to encode live stereo depth")
                with self.condition:
                    self.latest = (timestamp_us, png.tobytes())
        except Exception as error:
            self.error = error

    def close(self):
        with self.condition:
            self.closing = True
            self.condition.notify_all()
        self.thread.join(timeout=2)
        if self.error is not None:
            raise self.error
