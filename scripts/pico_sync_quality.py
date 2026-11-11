"""Diagnostics for a frozen PICO-to-NOKOV clock mapping, without refitting it."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar


def distribution_ms(values_s):
    values = np.asarray(values_s, dtype=float)
    values = values[np.isfinite(values)] * 1000
    if not len(values):
        return {
            "count": 0,
            "mean_ms": None,
            "median_ms": None,
            "std_ms": None,
            "rmse_ms": None,
            "mean_abs_ms": None,
            "p95_abs_ms": None,
            "p99_abs_ms": None,
            "max_abs_ms": None,
            "min_ms": None,
            "max_ms": None,
        }
    return {
        "count": len(values),
        "mean_ms": float(values.mean()),
        "median_ms": float(np.median(values)),
        "std_ms": float(values.std()),
        "rmse_ms": float(np.sqrt(np.mean(values**2))),
        "mean_abs_ms": float(np.mean(abs(values))),
        "p95_abs_ms": float(np.percentile(abs(values), 95)),
        "p99_abs_ms": float(np.percentile(abs(values), 99)),
        "max_abs_ms": float(np.max(abs(values))),
        "min_ms": float(values.min()),
        "max_ms": float(values.max()),
    }


def nearest_time_pairs(pico_t, nokov_t, scale, offset_s, eligible):
    """Associate raw frames, allowing reuse; positive delta means NOKOV is later.

    This measures discrete sampling distance, NOT physical synchronization error.
    Ties within 1 ps choose the earlier frame. Never subtract epoch-scale floats.
    """
    pico_t, nokov_t = np.asarray(pico_t), np.asarray(nokov_t)
    eligible = np.asarray(eligible, dtype=bool)
    if (
        not len(nokov_t)
        or np.any(np.diff(nokov_t) <= 0)
        or np.any(np.diff(pico_t) <= 0)
    ):
        raise ValueError("nonempty NOKOV and strictly increasing device times required")
    if (
        eligible.shape != pico_t.shape
        or not np.isfinite(scale)
        or scale <= 0
        or not np.isfinite(offset_s)
    ):
        raise ValueError("invalid eligibility mask or clock mapping")
    target = scale * pico_t + offset_s
    inside = (target >= nokov_t[0]) & (target <= nokov_t[-1])
    high = np.clip(np.searchsorted(nokov_t, target, side="left"), 0, len(nokov_t) - 1)
    low = np.maximum(high - 1, 0)
    low = np.where(abs(nokov_t[high] - target) <= 1e-12, high, low)
    chosen = np.where(
        abs(target - nokov_t[low]) <= abs(nokov_t[high] - target) + 1e-12, low, high
    )
    delta = nokov_t[chosen] - target
    delta[~inside] = np.nan
    used = inside & eligible
    summary = distribution_ms(delta[used])
    summary.update(
        {
            "equation": "delta_s = nearest_raw_NOKOV_relative_s - (scale * PICO_relative_s + offset_s)",
            "meaning": "sampling_association_distance_not_physical_sync_accuracy",
            "raw_nokov_median_period_ms": float(np.median(np.diff(nokov_t)) * 1000)
            if len(nokov_t) > 1
            else None,
            "unique_nokov_frames": len(np.unique(chosen[used])),
            "reused_frame_pairs": int(used.sum() - len(np.unique(chosen[used]))),
            "within_1ms_percent": float(np.mean(abs(delta[used]) <= 0.001) * 100)
            if used.any()
            else None,
            "within_5ms_percent": float(np.mean(abs(delta[used]) <= 0.005) * 100)
            if used.any()
            else None,
        }
    )
    return summary, {
        "target_s": target,
        "inside": inside,
        "used": used,
        "nearest": chosen,
        "left": low,
        "right": high,
        "delta_s": delta,
    }


def motion_window_quality(
    pg, pw, ng, nw, scale, offset_s, reference_support, window_s=8.0, search_s=0.05
):
    """Estimate extra motion delay per independent window, keeping global a,b fixed.

    The same reference-supported samples are used for EVERY candidate in a
    window. A weakly excited or boundary-optimal window is not an accepted
    residual estimate. These pose-derived estimates have no hardware time truth.
    """
    pg, pw, ng, nw = (np.asarray(a, dtype=float) for a in (pg, pw, ng, nw))
    if window_s <= 0 or search_s <= 0 or not np.isfinite([window_s, search_s]).all():
        raise ValueError("window and search widths must be positive")
    supported = np.isfinite(pw) & reference_support(
        scale * pg + offset_s, search_s + 0.04
    )
    rows = []
    for start in np.arange(0, pg[-1], window_s):
        end = min(start + window_s, float(pg[-1]))
        keep = supported & (pg >= start) & (pg < end)
        row = {
            "start_s": float(start),
            "end_s": float(end),
            "center_s": float((start + end) / 2),
            "samples": int(keep.sum()),
            "motion_std_rad_s": float(np.std(pw[keep])) if keep.any() else None,
            "status": "insufficient_reference_support",
            "residual_offset_ms": None,
            "correlation_before": None,
            "correlation_after": None,
            "speed_rmse_before_rad_s": None,
            "speed_rmse_after_rad_s": None,
        }
        rows.append(row)
        if end - start < 4:
            row["status"] = "window_too_short"
            continue
        if keep.sum() < 80 or np.ptp(pg[keep]) < 3:
            continue
        if row["motion_std_rad_s"] < 0.1:
            row["status"] = "insufficient_motion"
            continue
        query, measured = scale * pg[keep] + offset_s, pw[keep]

        def cost(extra, query=query, measured=measured):
            predicted = scale * np.interp(query + extra, ng, nw)
            return float(np.mean((predicted - measured) ** 2))

        candidates = np.linspace(-search_s, search_s, 201)
        costs = np.array([cost(d) for d in candidates])
        if not np.isfinite(costs).all():
            row["status"] = "nonfinite_reference_support"
            continue
        best = int(np.argmin(costs))
        lower, upper = (
            candidates[max(best - 2, 0)],
            candidates[min(best + 2, len(candidates) - 1)],
        )
        fit = minimize_scalar(
            cost, bounds=(lower, upper), method="bounded", options={"xatol": 1e-8}
        )
        delta = float(fit.x)
        before = scale * np.interp(query, ng, nw)
        after = scale * np.interp(query + delta, ng, nw)
        corr0, corr1 = (
            float(np.corrcoef(measured, before)[0, 1]),
            float(np.corrcoef(measured, after)[0, 1]),
        )
        row.update(
            {
                "residual_offset_ms": delta * 1000,
                "correlation_before": corr0,
                "correlation_after": corr1,
                "speed_rmse_before_rad_s": float(np.sqrt(cost(0))),
                "speed_rmse_after_rad_s": float(np.sqrt(cost(delta))),
                "status": "search_boundary"
                if abs(delta) > 0.98 * search_s
                else "ok"
                if corr1 > 0.9
                else "low_correlation",
            }
        )
    accepted = [r for r in rows if r["status"] == "ok"]
    summary = distribution_ms([r["residual_offset_ms"] * 0.001 for r in accepted])
    trend = None
    if len(accepted) >= 3:
        trend = float(
            np.polyfit(
                [r["center_s"] for r in accepted],
                [r["residual_offset_ms"] for r in accepted],
                1,
            )[0]
        )
    return {
        "method": "fixed-global-clock extra angular-speed-norm delay in nonoverlapping windows",
        "window_s": window_s,
        "search_limit_ms": search_s * 1000,
        "global_clock_refitted": False,
        "accepted_windows": len(accepted),
        "total_windows": len(rows),
        "residual_summary": summary,
        "residual_trend_ms_per_s": trend,
        "residual_trend_ppm": trend * 1000 if trend is not None else None,
        "limitation": "motion-consistency estimate; pose prediction/filtering and reference noise contribute; no hardware synchronization ground truth",
        "windows": rows,
    }


def full_motion_quality(
    pg, pw, ng, nw, scale, offset_s, reference_support, search_s=0.05,
    estimate_source=None,reference_source=None,
):
    """Estimate one extra lag from the entire recording, not a mean of window lags.

    Compare PICO head angular-speed changes, already mapped to NOKOV time,
    with NOKOV pico_head rigid-body angular-speed changes for the same motion.
    The frozen clock is NOT changed by this diagnostic.
    """
    diagnostic = motion_window_quality(
        pg,
        pw,
        ng,
        nw,
        scale,
        offset_s,
        reference_support,
        window_s=max(8.0, float(pg[-1]) + 1.0),
        search_s=search_s,
    )
    result = dict(diagnostic["windows"][0])
    delta = result["residual_offset_ms"]
    result.update(
        {
            "scope": "entire_recording_reference_supported_motion",
            "pico_source": estimate_source or "pico/pico_head_poses.csv: PICO head VIO motion, mapped to NOKOV time",
            "nokov_source": reference_source or "nokov/nokov_rigid_bodies.csv: selected head rigid-body motion",
            "compared_signal": "angular-speed magnitude derived from head orientation poses",
            "definition": "extra NOKOV-side time shift needed for best whole-recording motion alignment under the frozen existing clock",
            "absolute_residual_ms": abs(delta) if delta is not None else None,
            "global_clock_refitted": False,
            "residual_applied": False,
            "search_limit_ms": search_s * 1000,
            "limitation": "motion-derived residual estimate, not raw-frame sampling distance or hardware-measured time error",
        }
    )
    return result
