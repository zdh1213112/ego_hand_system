"""Preserve recorded camera/IMU times; keep external NOKOV clock mapping separate."""
from __future__ import annotations

from bisect import bisect_left
import statistics


def camera_offset_policy(args=None, imu_yaml=None):
    # Reject obsolete programmatic overrides too; CLI switches are removed.
    if getattr(args, 'camera_imu_offset_ns', 0) not in (None, 0) or getattr(args, 'use_factory_camera_imu_offset', False):
        raise ValueError('Camera/IMU time correction is disabled; use recorded timestamps with 0 ms offset')
    return {"camera_to_imu_offset_ns": 0, "source": "recorded_timestamps_zero_offset",
            "formula": "t_camera_ns = t_exported_camera_us * 1000; t_imu_ns = t_exported_imu_us * 1000",
            "association_method": "recorded_time_then_nearest_with_tolerance",
            "vio_imu_fusion": "all_samples_between_visual_updates; no nearest-only IMU reduction",
            "zero_default_does_not_certify_physical_synchronization": True,
            "camera_timestamps_modified": False, "imu_timestamps_modified": False, "application_count": 0}


def require_zero_camera_offset(metadata, label='EGO VIO input'):
    """Never relabel a shifted historical run as a zero-offset experiment."""
    keys={'camera_time_shift_ns','camera_to_imu_offset_ns','camera_to_imu_shift_ns',
          'camera_time_shift_s','cam_time_offset_ns','camera_time_shift_ns_used_by_original_run',
          'camera_shift_ns'}
    def check(value):
        if not isinstance(value,dict):return
        for key,item in value.items():
            if key in keys and item not in (None,0):
                raise ValueError(f'{label}: {key}={item}; camera/IMU offset must be 0 ms. Reprepare and rerun VIO; do not edit timestamps of an existing trajectory.')
            if key in {'time_sync_policy','camera_time_policy','calibration','value0'}:check(item)
    check(metadata)


def validate_times(values, label):
    if not values or any(b <= a for a, b in zip(values, values[1:])):
        raise ValueError(f"{label} timestamps must be nonempty and strictly increasing")


def nearest_indices(reference, samples, max_delta_ns, *, sample_offset_ns=0):
    """Match each reference to corrected samples, allowing repeated associations.

    Return original sample indices or None. Ties choose the earlier sample.
    Association does not move a reference timestamp or create a new observation.
    Python integers preserve epoch-nanosecond precision.
    """
    validate_times(reference, "reference")
    validate_times(samples, "sample")
    if max_delta_ns < 0:
        raise ValueError("max_delta_ns must be non-negative")
    corrected = [value + sample_offset_ns for value in samples]
    result = []
    for timestamp in reference:
        high = bisect_left(corrected, timestamp)
        candidates = [i for i in (high - 1, high) if 0 <= i < len(corrected)]
        best = min(candidates, key=lambda i: (abs(corrected[i] - timestamp), i))
        result.append(best if abs(corrected[best] - timestamp) <= max_delta_ns else None)
    return result


def sampling_summary(times):
    validate_times(times, "stream")
    intervals = [b - a for a, b in zip(times, times[1:])]
    return {"count": len(times), "first_timestamp_ns": times[0], "last_timestamp_ns": times[-1],
            "duration_s": (times[-1] - times[0]) * 1e-9,
            "average_hz": (len(times)-1) * 1e9 / (times[-1]-times[0]) if intervals else None,
            "median_interval_ms": statistics.median(intervals) / 1e6 if intervals else None}
