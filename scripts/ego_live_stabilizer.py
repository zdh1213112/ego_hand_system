"""Causal hand-joint smoothing for live output; raw measurements remain intact."""

from __future__ import annotations

from collections import deque

import numpy as np


class CausalJointFilter:
    """Three-sample median followed by EMA, without filling invalid joints."""

    def __init__(self, alpha: float, max_gap_us: int = 200_000,
                 reset_jump: float | None = None):
        self.alpha = alpha
        self.max_gap_us = max_gap_us
        self.reset_jump = reset_jump
        self.history = deque(maxlen=3)
        self.previous = None
        self.last_timestamp_us = None

    def update(self, joints, timestamp_us: int):
        if joints is None:
            return None
        current = np.asarray(joints, dtype=np.float64)
        if current.ndim != 2 or current.shape[0] != 21 or current.shape[1] not in (2, 3):
            raise ValueError(f"expected 21 x 2/3 joints, got {current.shape}")
        valid = np.isfinite(current).all(axis=1)
        gap = (timestamp_us - self.last_timestamp_us
               if self.last_timestamp_us is not None else None)
        reset = gap is None or gap <= 0 or gap > self.max_gap_us
        if (not reset and self.reset_jump is not None and valid[0]
                and self.previous is not None and np.isfinite(self.previous[0]).all()
                and np.linalg.norm(current[0] - self.previous[0]) > self.reset_jump):
            reset = True
        if reset:
            self.history.clear()
            self.previous = None
        self.history.append(current)
        stack = np.stack(self.history)
        history_valid = np.isfinite(stack).all(axis=2)
        stack = np.where(history_valid[:, :, None], stack, np.nan)
        # Every currently valid joint has at least its newest observation.
        # Invalid current joints are masked after the median, so fill those
        # columns here to avoid all-NaN reduction warnings.
        stack[:, ~valid] = 0.0
        median = np.nanmedian(stack, axis=0)
        median[~valid] = np.nan
        result = median.copy()
        if self.previous is not None:
            blend = valid & np.isfinite(self.previous).all(axis=1)
            result[blend] = (self.alpha * median[blend]
                             + (1.0 - self.alpha) * self.previous[blend])
        self.previous = result
        self.last_timestamp_us = timestamp_us
        return result.tolist()


class BoundedDisplayFilter:
    """Suppress small 3D measurement noise with a bounded spatial lag.

    Each displayed joint stays within 2.5 mm (wrist) or 3.5 mm (finger)
    of the latest valid triangulation, even during fast movement.
    """

    def __init__(self, alpha: float = 0.2, max_gap_us: int = 200_000):
        self.alpha = alpha
        self.max_gap_us = max_gap_us
        self.max_lag_m = np.asarray([0.0025] + [0.0035] * 20)
        self.previous = None
        self.last_timestamp_us = None

    def update(self, joints, timestamp_us: int):
        if joints is None:
            self.previous = None
            self.last_timestamp_us = None
            return None
        current = np.asarray(joints, dtype=np.float64)
        if current.shape != (21, 3):
            raise ValueError(f"expected 21 x 3 joints, got {current.shape}")
        result = current.copy()
        gap = (timestamp_us - self.last_timestamp_us
               if self.last_timestamp_us is not None else None)
        if self.previous is not None and gap is not None and 0 < gap <= self.max_gap_us:
            valid = np.isfinite(current).all(axis=1)
            usable = valid & np.isfinite(self.previous).all(axis=1)
            delta = current[usable] - self.previous[usable]
            distance = np.linalg.norm(delta, axis=1)
            gain = np.maximum(
                self.alpha,
                1.0 - self.max_lag_m[usable] / np.maximum(distance, 1e-12),
            )
            result[usable] = self.previous[usable] + gain[:, None] * delta
        self.previous = result
        self.last_timestamp_us = timestamp_us
        return result.tolist()


class LiveHandStabilizer:
    """Attach stabilized fields to each live row; keep original fields untouched."""

    def __init__(self):
        self.filters = {}
        self.display_filters = {}

    def _filter(self, key, alpha, joints, timestamp_us, reset_jump=None):
        if key not in self.filters:
            self.filters[key] = CausalJointFilter(alpha, reset_jump=reset_jump)
        return self.filters[key].update(joints, timestamp_us)

    def update_row(self, row):
        timestamp_us = int(row["timestamp_us"]["cam_0"])
        for camera, hands in row["camera_hands"].items():
            for hand_key, hand in hands.items():
                side = int(hand_key)
                hand["joints_2d_stabilized"] = self._filter(
                    (camera, side, "2d"), 0.65, hand.get("joints_2d"),
                    timestamp_us, reset_jump=200.0,
                )
                hand["joints_3d_camera_relative_stabilized"] = self._filter(
                    (camera, side, "relative_3d"), 0.55,
                    hand.get("joints_3d_camera_relative"), timestamp_us,
                )
        for hand in row["hands"]:
            side = int(hand["is_right"])
            hand["joints_3d_base_m_stabilized"] = self._filter(
                ("fused", side), 0.55, hand.get("joints_3d_base_m"),
                timestamp_us, reset_jump=0.15,
            )
            if side not in self.display_filters:
                self.display_filters[side] = BoundedDisplayFilter()
            hand["joints_3d_base_m_display"] = self.display_filters[side].update(
                hand.get("joints_3d_base_m"), timestamp_us,
            )
        return row
