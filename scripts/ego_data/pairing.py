"""Timestamp pairing utilities using nanoseconds internally."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class TimestampPair:
    pair_index: int
    left_frame_index: int
    right_frame_index: int
    left_timestamp_ns: int
    right_timestamp_ns: int

    @property
    def delta_ns(self) -> int:
        return self.right_timestamp_ns - self.left_timestamp_ns


def validate_timestamps(values: list[int], label: str) -> None:
    if not values:
        raise ValueError(f"{label} timestamps are empty")
    if any(current <= previous for previous, current in zip(values, values[1:])):
        raise ValueError(f"{label} timestamps are not strictly increasing")


def pair_timestamps_ns(left: list[int], right: list[int], max_delta_ns: int) -> list[TimestampPair]:
    validate_timestamps(left, "left")
    validate_timestamps(right, "right")
    if max_delta_ns < 0:
        raise ValueError("max_delta_ns must be non-negative")
    pairs = []
    left_index = right_index = 0
    while left_index < len(left) and right_index < len(right):
        delta = right[right_index] - left[left_index]
        if abs(delta) <= max_delta_ns:
            pairs.append(TimestampPair(
                len(pairs), left_index, right_index,
                int(left[left_index]), int(right[right_index]),
            ))
            left_index += 1
            right_index += 1
        elif delta > 0:
            left_index += 1
        else:
            right_index += 1
    return pairs


def pairing_statistics(left_count: int, right_count: int, pairs: list[TimestampPair]) -> dict:
    absolute = np.abs([pair.delta_ns for pair in pairs]).astype(np.float64)
    percentile = lambda level: int(round(np.percentile(absolute, level))) if absolute.size else None
    return {
        "left_frame_count": int(left_count),
        "right_frame_count": int(right_count),
        "pair_count": len(pairs),
        "unpaired_left": int(left_count - len(pairs)),
        "unpaired_right": int(right_count - len(pairs)),
        "abs_delta_ns_median": percentile(50),
        "abs_delta_ns_p95": percentile(95),
        "abs_delta_ns_max": percentile(100),
    }
