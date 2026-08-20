#!/usr/bin/env python3
"""Fit licensed MANO hand models to prepared EGO stereo observations."""

from __future__ import annotations

import argparse
import builtins
import csv
import inspect
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import cv2
import numpy as np

from mano_conventions import (
    WILOR_RIGHT_CANONICAL,
    canonical_projection_rotation,
    mirror_left_points,
    physicalize_geometry,
)


# otaheri/MANO output order -> MediaPipe semantic order.
MANO_TO_MEDIAPIPE = np.asarray([
    0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20,
], dtype=np.int64)

MEDIAPIPE_NAMES = (
    "wrist", "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
    "index_mcp", "index_pip", "index_dip", "index_tip",
    "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
    "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit MANO to stabilized EGO hand tracks.")
    parser.add_argument("--input", required=True, type=Path, help="mano_input.npz")
    parser.add_argument("--mano-source", required=True, type=Path)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--mano-convention", choices=(WILOR_RIGHT_CANONICAL,),
        default=WILOR_RIGHT_CANONICAL,
        help="MANO representation; WiLoR uses MANO_RIGHT for both physical hands",
    )
    parser.add_argument("--initial-output", type=Path, help="warm-start from track_*.npz files")
    parser.add_argument("--track-id", type=int, action="append", help="fit only selected track(s)")
    parser.add_argument("--start-pair", type=int, default=0)
    parser.add_argument("--max-pairs", type=int)
    parser.add_argument("--pca-components", type=int, default=15)
    parser.add_argument("--shape-frames", type=int, default=64)
    parser.add_argument("--shape-iterations", type=int, default=350)
    parser.add_argument("--pose-iterations", type=int, default=220)
    parser.add_argument("--pose-window", type=int, default=0, help="pose window size; 0 uses the full track")
    parser.add_argument("--pose-overlap", type=int, default=4, help="overlap between pose windows")
    parser.add_argument("--learning-rate", type=float, default=0.025)
    parser.add_argument("--w-3d", type=float, default=1.0)
    parser.add_argument("--w-2d", type=float, default=0.12)
    parser.add_argument(
        "--w-pinch", type=float, default=0.0,
        help="thumb-index distance loss weight, active only for observed near-contact frames",
    )
    parser.add_argument(
        "--pinch-threshold-m", type=float, default=0.025,
        help="maximum observed thumb-index tip distance that activates the pinch loss",
    )
    parser.add_argument(
        "--w-contact-tips", type=float, default=0.0,
        help="near-contact thumb/index absolute 3D and 2D tip alignment weight",
    )
    parser.add_argument(
        "--contact-tip-threshold-m", type=float, default=0.035,
        help="observed thumb-index distance that activates absolute tip alignment",
    )
    parser.add_argument(
        "--min-fit-observed-points", type=int, default=12,
        help="minimum real 3D landmarks required for a frame to influence MANO fitting",
    )
    parser.add_argument(
        "--max-unobserved-gap", type=int, default=5,
        help="maximum unsupported gap kept inside one fitted/rendered motion segment",
    )
    parser.add_argument("--w-pose", type=float, default=0.003)
    parser.add_argument("--w-shape", type=float, default=0.015)
    parser.add_argument("--w-temporal", type=float, default=0.08)
    parser.add_argument("--w-rigid-temporal", type=float, default=0.01,
                        help="separate velocity penalty for global rotation/translation")
    parser.add_argument("--w-acceleration", type=float, default=0.004,
                        help="pose and rigid acceleration penalty")
    parser.add_argument("--boundary-weight", type=float, default=0.10,
                        help="anchor the overlap shared with the previous pose window")
    parser.add_argument("--max-orient-step-deg", type=float, default=75.0,
                        help="maximum base global rotation update before image-motion scaling")
    parser.add_argument("--max-translation-step-m", type=float, default=0.08,
                        help="maximum base translation update before image-motion scaling")
    parser.add_argument("--max-pose-step", type=float, default=0.0,
                        help="optional PCA pose-vector cap; 0 relies on velocity/acceleration losses")
    parser.add_argument("--image-rigid-alignment", action=argparse.BooleanOptionalAction,
                        default=False, help="experimental palm-only image-space translation correction")
    parser.add_argument("--image-alignment-max-m", type=float, default=0.025)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    parser.add_argument(
        "--rigid-initialization", action=argparse.BooleanOptionalAction, default=True,
        help="initialize each supported frame with Kabsch before nonlinear fitting",
    )
    parser.add_argument("--no-video", action="store_true")
    return parser.parse_args()


def validate_source_and_assets(
    source: Path, model_dir: Path, mano_convention: str = WILOR_RIGHT_CANONICAL,
) -> str:
    if mano_convention != WILOR_RIGHT_CANONICAL:
        raise ValueError(f"unsupported MANO convention: {mano_convention}")
    missing = [model_dir / "MANO_RIGHT.pkl"] if not (
        model_dir / "MANO_RIGHT.pkl"
    ).is_file() else []
    if missing:
        raise FileNotFoundError(
            "missing licensed MANO model data: " + ", ".join(str(path) for path in missing)
        )
    if not (source / "mano" / "model.py").is_file():
        raise FileNotFoundError(f"invalid MANO source: {source}")
    revision = "unknown"
    try:
        revision = subprocess.run(
            ["git", "-C", str(source), "rev-parse", "HEAD"], check=True,
            capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        pass
    return revision


def canonicalize_observations(observations: dict, handedness: str) -> dict:
    """Put a physical track into WiLoR's MANO_RIGHT optimization space."""
    canonical = dict(observations)
    canonical["positions"] = mirror_left_points(observations["positions"], handedness)
    canonical["rotation"] = canonical_projection_rotation(
        observations["rotation"], handedness
    )
    return canonical


def physical_result(result: dict, handedness: str, faces: np.ndarray) -> tuple[dict, np.ndarray]:
    """Return a shallow result copy suitable for physical-space CSV/video output."""
    vertices, joints, physical_faces = physicalize_geometry(
        result["vertices"], result["joints"], faces, handedness
    )
    converted = dict(result)
    converted["vertices"] = vertices
    converted["joints"] = joints
    converted["translation"] = mirror_left_points(result["translation"], handedness)
    return converted, physical_faces


def import_mano(source: Path):
    # Official MANO v1.2 pickles contain chumpy objects.  chumpy 0.70 still
    # references aliases removed by NumPy 2 and inspect.getargspec removed by
    # Python 3.11.  Restore only the names needed for safe legacy unpickling;
    # neither the model data nor the external MANO/chumpy sources are modified.
    legacy_numpy_aliases = {
        "bool": np.bool_, "int": builtins.int, "float": builtins.float,
        "complex": builtins.complex, "object": builtins.object,
        "unicode": builtins.str, "str": builtins.str,
    }
    for name, value in legacy_numpy_aliases.items():
        if name not in np.__dict__:
            setattr(np, name, value)
    if not hasattr(inspect, "getargspec"):
        inspect.getargspec = inspect.getfullargspec  # type: ignore[attr-defined]
    sys.path.insert(0, str(source))
    try:
        import mano  # type: ignore
        return mano
    except Exception:
        sys.path.pop(0)
        raise


def load_input(path: Path) -> dict[str, np.ndarray]:
    required = {
        "positions_left_camera_m", "valid", "confidence", "track_ids", "handedness",
        "left_rectified_px", "right_rectified_px", "left_to_rectified_rotation",
        "projection_left_rectified", "projection_right_rectified", "pair_indices",
    }
    with np.load(path) as archive:
        missing = required - set(archive.files)
        if missing:
            raise RuntimeError(f"missing MANO input arrays: {sorted(missing)}")
        data = {name: archive[name].copy() for name in archive.files}
    positions = data["positions_left_camera_m"]
    valid = data["valid"]
    if positions.ndim != 4 or positions.shape[2:] != (21, 3):
        raise RuntimeError(f"unexpected positions shape: {positions.shape}")
    if valid.shape != positions.shape[:-1]:
        raise RuntimeError("valid mask shape disagrees with positions")
    return data


def select_pair_range(data: dict[str, np.ndarray], args: argparse.Namespace) -> np.ndarray:
    pairs = data["pair_indices"].astype(np.int64)
    keep = pairs >= args.start_pair
    if args.max_pairs is not None:
        keep &= pairs < args.start_pair + args.max_pairs
    return np.flatnonzero(keep)


def project_rectified(points_left, rotation, projection):
    import torch
    rectified = torch.einsum("ij,bkj->bki", rotation, points_left)
    homogeneous = torch.cat((rectified, torch.ones_like(rectified[..., :1])), dim=-1)
    projected = torch.einsum("ij,bkj->bki", projection, homogeneous)
    return projected[..., :2] / projected[..., 2:].clamp_min(1e-6)


def robust_weighted_loss(residual, weight, scale: float):
    import torch
    finite = torch.isfinite(residual).all(dim=-1)
    weight = torch.where(finite, weight, 0.0)
    safe_residual = torch.where(finite[..., None], residual, 0.0)
    norm = torch.linalg.vector_norm(safe_residual, dim=-1)
    robust = torch.sqrt(norm.square() + scale * scale) - scale
    return (robust * weight).sum() / weight.sum().clamp_min(1.0)


def image_observation_weights(confidence, pixels, valid_mask):
    import torch
    finite = torch.isfinite(pixels).all(dim=-1)
    valid = valid_mask & finite
    minimum = torch.tensor(0.1, dtype=confidence.dtype, device=confidence.device)
    return torch.where(valid, torch.maximum(confidence, minimum), 0.0)


def pinch_distance_loss(joints, target, valid, confidence, threshold_m: float):
    import torch
    observed_distance = torch.linalg.vector_norm(target[:, 4] - target[:, 8], dim=-1)
    predicted_distance = torch.linalg.vector_norm(joints[:, 4] - joints[:, 8], dim=-1)
    active = valid[:, 4] & valid[:, 8] & torch.isfinite(observed_distance)
    active &= observed_distance < threshold_m
    tip_confidence = torch.minimum(confidence[:, 4], confidence[:, 8])
    contact_strength = torch.clamp(
        (threshold_m - observed_distance) / max(threshold_m, 1e-6), 0.0, 1.0
    )
    weights = torch.where(active, tip_confidence * contact_strength, 0.0)
    residual = torch.where(active, predicted_distance - observed_distance, 0.0)
    robust = torch.sqrt(residual.square() + 0.003 ** 2) - 0.003
    return (robust * weights).sum() / weights.sum().clamp_min(1e-6)


def contact_tip_alignment_loss(
        joints, target, valid, confidence,
        left_prediction, right_prediction, left_px, right_px,
        left_valid, right_valid, threshold_m: float):
    import torch

    tip_ids = torch.as_tensor((4, 8), dtype=torch.long, device=joints.device)
    observed_distance = torch.linalg.vector_norm(target[:, 4] - target[:, 8], dim=-1)
    active = valid[:, 4] & valid[:, 8] & torch.isfinite(observed_distance)
    active &= observed_distance < threshold_m
    strength = torch.clamp(
        (threshold_m - observed_distance) / max(threshold_m, 1e-6), 0.0, 1.0
    )
    tip_valid = valid.index_select(1, tip_ids) & active[:, None]
    tip_confidence = confidence.index_select(1, tip_ids) * strength[:, None]
    weights3d = torch.where(tip_valid, tip_confidence, 0.0)
    loss = robust_weighted_loss(
        joints.index_select(1, tip_ids) - target.index_select(1, tip_ids),
        weights3d, 0.004,
    )

    for prediction, pixels, pixel_valid in (
        (left_prediction, left_px, left_valid),
        (right_prediction, right_px, right_valid),
    ):
        selected_prediction = prediction.index_select(1, tip_ids)
        selected_pixels = pixels.index_select(1, tip_ids)
        selected_valid = pixel_valid.index_select(1, tip_ids) & tip_valid
        weights2d = image_observation_weights(
            tip_confidence, selected_pixels, selected_valid
        )
        loss = loss + 0.25 * robust_weighted_loss(
            (selected_prediction - selected_pixels) / 100.0, weights2d, 0.015
        )
    return loss


def run_model(model, betas, orient, pose, transl):
    output = model(
        betas=betas, global_orient=orient, hand_pose=pose, transl=transl,
        return_verts=True, return_tips=True, return_full_pose=True,
    )
    order = model.faces_tensor.new_tensor(MANO_TO_MEDIAPIPE)
    return output.vertices, output.joints.index_select(1, order), output


def weighted_kabsch(source: np.ndarray, target: np.ndarray, weights: np.ndarray) -> np.ndarray:
    weights = np.maximum(np.asarray(weights, dtype=np.float64), 1e-6)
    weights /= weights.sum()
    source_centre = np.sum(source * weights[:, None], axis=0)
    target_centre = np.sum(target * weights[:, None], axis=0)
    covariance = (source - source_centre).T @ ((target - target_centre) * weights[:, None])
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T
    return rotation


def observation_transition_profile(
        valid: np.ndarray, confidence: np.ndarray, left_px: np.ndarray,
        right_px: np.ndarray, max_orient_step_deg: float = 75.0,
        max_translation_step_m: float = 0.08,
        max_pose_step: float = 0.0) -> dict[str, np.ndarray]:
    """Build quality-aware temporal weights and physically plausible step limits.

    Image motion is deliberately used only to decide how much a state is allowed
    to change.  It cannot directly rotate the hand, which prevents a bad 2D
    landmark from causing a large MANO solution switch.
    """
    valid = np.asarray(valid, dtype=bool)
    confidence = np.nan_to_num(np.asarray(confidence, dtype=np.float32), nan=0.0)
    frame_count = valid.shape[0]
    observed_count = valid.sum(axis=1).astype(np.float32)
    mean_confidence = (confidence * valid).sum(axis=1) / np.maximum(observed_count, 1.0)
    coverage = np.sqrt(np.clip(observed_count / max(valid.shape[1], 1), 0.0, 1.0))
    quality = np.clip(mean_confidence * coverage, 0.0, 1.0)

    image_motion = np.zeros(frame_count, dtype=np.float32)
    for frame in range(1, frame_count):
        displacements = []
        for pixels in (left_px, right_px):
            previous = np.asarray(pixels[frame - 1])
            current = np.asarray(pixels[frame])
            finite = np.isfinite(previous).all(axis=-1) & np.isfinite(current).all(axis=-1)
            if np.any(finite):
                displacements.extend(
                    np.linalg.norm(current[finite] - previous[finite], axis=-1).tolist()
                )
        if displacements:
            image_motion[frame] = float(np.median(displacements))

    transition_quality = np.minimum(quality, np.roll(quality, 1))
    transition_quality[0] = quality[0]
    temporal_strength = np.clip(
        0.45 + 2.4 * (1.0 - transition_quality)
        + 0.45 * np.exp(-image_motion / 12.0),
        0.45, 3.0,
    ).astype(np.float32)

    # A reliable, visibly fast hand may move farther.  Low-quality observations
    # instead receive a smaller trust region, even if their detector position jumps.
    motion_factor = 0.75 + 0.50 * np.clip(image_motion / 35.0, 0.0, 1.0)
    quality_factor = 0.80 + 0.20 * transition_quality
    limit_factor = np.clip(motion_factor * quality_factor, 0.60, 1.25)
    orient_limit = np.deg2rad(max_orient_step_deg) * limit_factor
    translation_limit = max_translation_step_m * limit_factor
    if max_pose_step > 0.0:
        pose_limit = max_pose_step * np.clip(
            (0.60 + 0.40 * transition_quality)
            * (0.80 + 0.40 * np.clip(image_motion / 35.0, 0.0, 1.0)),
            0.50, 1.20,
        )
    else:
        pose_limit = np.full(frame_count, np.inf, dtype=np.float32)
    return {
        "quality": quality.astype(np.float32),
        "image_motion_px": image_motion,
        "temporal_strength": temporal_strength,
        "orient_limit_rad": orient_limit.astype(np.float32),
        "translation_limit_m": translation_limit.astype(np.float32),
        "pose_limit": pose_limit.astype(np.float32),
    }


def limit_parameter_transitions_(pose, orient, transl, frame_ids, profile) -> None:
    """Clamp frame-to-frame updates in-place without changing the first state."""
    import torch

    ids = [int(value) for value in frame_ids.detach().cpu().tolist()]
    with torch.no_grad():
        for frame in ids:
            if frame <= 0:
                continue
            for parameter, limit_key in ((pose, "pose_limit"), (transl, "translation_limit_m")):
                delta = parameter[frame] - parameter[frame - 1]
                norm = torch.linalg.vector_norm(delta)
                limit = profile[limit_key][frame]
                if bool(torch.isfinite(limit) & (norm > limit)):
                    parameter[frame].copy_(
                        parameter[frame - 1] + delta * (limit / norm.clamp_min(1e-8))
                    )

            # Axis-angle vectors are not Euclidean coordinates: equivalent
            # rotations can lie far apart around +/-pi.  Clamp the true
            # relative SO(3) rotation and convert it back to axis-angle.
            previous_vector = orient[frame - 1].detach().cpu().numpy().astype(np.float64)
            current_vector = orient[frame].detach().cpu().numpy().astype(np.float64)
            previous_rotation, _ = cv2.Rodrigues(previous_vector)
            current_rotation, _ = cv2.Rodrigues(current_vector)
            relative_rotation = current_rotation @ previous_rotation.T
            relative_vector, _ = cv2.Rodrigues(relative_rotation)
            relative_vector = relative_vector[:, 0]
            relative_angle = float(np.linalg.norm(relative_vector))
            limit = float(profile["orient_limit_rad"][frame])
            if np.isfinite(limit) and relative_angle > limit:
                limited_relative, _ = cv2.Rodrigues(relative_vector * (limit / relative_angle))
                limited_rotation = limited_relative @ previous_rotation
                limited_vector, _ = cv2.Rodrigues(limited_rotation)
                orient[frame].copy_(torch.as_tensor(
                    limited_vector[:, 0], dtype=orient.dtype, device=orient.device
                ))


def repair_low_support_initial(initial: dict, supported: np.ndarray) -> dict:
    """Replace underconstrained warm-start frames by interpolation from supported frames."""
    supported = np.asarray(supported, dtype=bool)
    repaired = {key: np.asarray(value).copy() for key, value in initial.items()}
    anchors = np.flatnonzero(supported)
    if len(anchors) == 0:
        return repaired

    for frame in np.flatnonzero(~supported):
        insertion = int(np.searchsorted(anchors, frame))
        previous = int(anchors[max(0, insertion - 1)])
        following = int(anchors[min(insertion, len(anchors) - 1)])
        if previous == following:
            alpha = 0.0
        else:
            alpha = (frame - previous) / (following - previous)
        for key in ("hand_pose_pca", "translation"):
            repaired[key][frame] = (
                (1.0 - alpha) * repaired[key][previous]
                + alpha * repaired[key][following]
            )

        previous_rotation, _ = cv2.Rodrigues(repaired["global_orient"][previous])
        following_rotation, _ = cv2.Rodrigues(repaired["global_orient"][following])
        relative_vector, _ = cv2.Rodrigues(following_rotation @ previous_rotation.T)
        partial_rotation, _ = cv2.Rodrigues(relative_vector[:, 0] * alpha)
        repaired_rotation = partial_rotation @ previous_rotation
        repaired_vector, _ = cv2.Rodrigues(repaired_rotation)
        repaired["global_orient"][frame] = repaired_vector[:, 0]
    return repaired


def support_segments(supported: np.ndarray, max_gap: int) -> list[tuple[int, int]]:
    """Return inclusive motion segments, splitting at long unsupported gaps."""
    supported = np.asarray(supported, dtype=bool).reshape(-1)
    if max_gap < 0:
        raise ValueError("max_gap must be non-negative")
    indices = np.flatnonzero(supported)
    if not len(indices):
        return []
    segments: list[tuple[int, int]] = []
    start = int(indices[0])
    previous = int(indices[0])
    for current_value in indices[1:]:
        current = int(current_value)
        if current - previous - 1 > max_gap:
            segments.append((start, previous))
            start = current
        previous = current
    segments.append((start, previous))
    return segments


def bridge_short_false_gaps(mask: np.ndarray, max_gap: int) -> np.ndarray:
    """Fill only bounded short gaps; never extrapolate across track ends."""
    result = np.asarray(mask, dtype=bool).reshape(-1).copy()
    if max_gap < 0:
        raise ValueError("max_gap must be non-negative")
    true_indices = np.flatnonzero(result)
    for left, right in zip(true_indices[:-1], true_indices[1:]):
        if 0 < right - left - 1 <= max_gap:
            result[left + 1:right] = True
    return result


def image_space_translation_alignment(
        joints, left_px, rotation, projection, quality, maximum_m: float):
    """Return a small robust translation that aligns the MANO palm in the image."""
    import torch

    frame_count = joints.shape[0]
    correction = torch.zeros((frame_count, 3), dtype=joints.dtype, device=joints.device)
    predicted = project_rectified(joints, rotation, projection)
    rectified = torch.einsum("ij,bkj->bki", rotation, joints)
    palm = (0, 5, 9, 13, 17)
    fx = projection[0, 0].abs().clamp_min(1e-6)
    fy = projection[1, 1].abs().clamp_min(1e-6)
    rotation_inverse = rotation.T
    for frame in range(frame_count):
        finite = torch.isfinite(left_px[frame]).all(dim=-1)
        selected = [joint for joint in palm if bool(finite[joint])]
        if len(selected) < 3 or float(quality[frame]) < 0.08:
            continue
        indices = torch.as_tensor(selected, dtype=torch.long, device=joints.device)
        residual = left_px[frame, indices] - predicted[frame, indices]
        pixel_offset = residual.median(dim=0).values
        depth = rectified[frame, indices, 2].median().clamp_min(0.05)
        delta_rectified = torch.stack((
            pixel_offset[0] * depth / fx,
            pixel_offset[1] * depth / fy,
            torch.zeros((), dtype=joints.dtype, device=joints.device),
        ))
        delta = rotation_inverse @ delta_rectified
        norm = torch.linalg.vector_norm(delta)
        if bool(norm > maximum_m):
            delta = delta * (maximum_m / norm.clamp_min(1e-8))
        correction[frame] = delta

    # A centred three-frame median removes isolated landmark spikes without
    # introducing the visible phase delay of a causal low-pass filter.
    smoothed = correction.clone()
    for frame in range(1, frame_count - 1):
        smoothed[frame] = correction[frame - 1:frame + 2].median(dim=0).values
    return smoothed


def initialize_rigid_parameters(model, target, valid, confidence, pca_components: int, device):
    import torch

    frame_count = target.shape[0]
    zeros_betas = torch.zeros((frame_count, 10), dtype=torch.float32, device=device)
    zeros_orient = torch.zeros((frame_count, 3), dtype=torch.float32, device=device)
    zeros_pose = torch.zeros((frame_count, pca_components), dtype=torch.float32, device=device)
    zeros_transl = torch.zeros((frame_count, 3), dtype=torch.float32, device=device)
    with torch.no_grad():
        _, template_joints_tensor, _ = run_model(
            model, zeros_betas, zeros_orient, zeros_pose, zeros_transl
        )
    template = template_joints_tensor.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    valid_np = valid.detach().cpu().numpy()
    confidence_np = confidence.detach().cpu().numpy()
    orientation = np.zeros((frame_count, 3), dtype=np.float32)
    palm_joints = np.asarray((0, 1, 5, 9, 13, 17), dtype=np.int64)
    previous_orientation = np.zeros(3, dtype=np.float32)

    for frame in range(frame_count):
        palm_valid = palm_joints[valid_np[frame, palm_joints]]
        selected = palm_valid if len(palm_valid) >= 3 else np.flatnonzero(valid_np[frame])
        if len(selected) >= 3:
            rotation = weighted_kabsch(
                template[frame, selected], target_np[frame, selected],
                confidence_np[frame, selected],
            )
            rotation_vector, _ = cv2.Rodrigues(rotation)
            previous_orientation = rotation_vector[:, 0].astype(np.float32)
        orientation[frame] = previous_orientation

    orientation_tensor = torch.as_tensor(orientation, dtype=torch.float32, device=device)
    with torch.no_grad():
        _, rotated_joints, _ = run_model(
            model, zeros_betas, orientation_tensor, zeros_pose, zeros_transl
        )
    translation = torch.zeros((frame_count, 3), dtype=torch.float32, device=device)
    for frame in range(frame_count):
        mask = valid[frame]
        if torch.any(mask):
            weights = confidence[frame, mask].clamp_min(0.05)
            translation[frame] = torch.sum(
                (target[frame, mask] - rotated_joints[frame, mask]) * weights[:, None], dim=0
            ) / weights.sum()
        elif frame:
            translation[frame] = translation[frame - 1]
        else:
            translation[frame] = torch.tensor((0.0, 0.0, 0.25), device=device)
    return orientation_tensor, translation


def optimize_track(model, observations: dict, args: argparse.Namespace, device):
    import torch

    target = torch.as_tensor(observations["positions"], dtype=torch.float32, device=device)
    valid = torch.as_tensor(observations["valid"], dtype=torch.bool, device=device)
    observed = torch.as_tensor(
        observations.get("observed", observations["valid"]),
        dtype=torch.bool,
        device=device,
    )
    min_fit_points = max(1, int(getattr(args, "min_fit_observed_points", 1)))
    max_unobserved_gap = max(0, int(getattr(args, "max_unobserved_gap", 5)))
    frame_support = observed.sum(dim=1) >= min_fit_points
    fit_valid = valid & observed & frame_support[:, None]
    confidence = torch.as_tensor(observations["confidence"], dtype=torch.float32, device=device)
    left_px = torch.as_tensor(observations["left_px"], dtype=torch.float32, device=device)
    right_px = torch.as_tensor(observations["right_px"], dtype=torch.float32, device=device)
    left_px_valid = torch.as_tensor(observations.get(
        "left_px_valid", np.isfinite(observations["left_px"]).all(axis=-1)
    ), dtype=torch.bool, device=device)
    right_px_valid = torch.as_tensor(observations.get(
        "right_px_valid", np.isfinite(observations["right_px"]).all(axis=-1)
    ), dtype=torch.bool, device=device)
    rotation = torch.as_tensor(observations["rotation"], dtype=torch.float32, device=device)
    p1 = torch.as_tensor(observations["p1"], dtype=torch.float32, device=device)
    p2 = torch.as_tensor(observations["p2"], dtype=torch.float32, device=device)
    frame_count = target.shape[0]

    fit_valid_np = fit_valid.detach().cpu().numpy()
    transition_profile_np = observation_transition_profile(
        fit_valid_np, observations["confidence"], observations["left_px"],
        observations["right_px"],
        max_orient_step_deg=getattr(args, "max_orient_step_deg", 75.0),
        max_translation_step_m=getattr(args, "max_translation_step_m", 0.08),
        max_pose_step=getattr(args, "max_pose_step", 0.0),
    )
    transition_profile = {
        key: torch.as_tensor(value, dtype=torch.float32, device=device)
        for key, value in transition_profile_np.items()
    }

    initial = observations.get("initial")
    if initial is not None:
        initial = repair_low_support_initial(
            initial, frame_support.detach().cpu().numpy()
        )
        pose = torch.as_tensor(
            initial["hand_pose_pca"], dtype=torch.float32, device=device
        ).clone().requires_grad_(True)
        orient = torch.as_tensor(
            initial["global_orient"], dtype=torch.float32, device=device
        ).clone().requires_grad_(True)
        transl = torch.as_tensor(
            initial["translation"], dtype=torch.float32, device=device
        ).clone().requires_grad_(True)
        betas = torch.as_tensor(
            initial["betas"], dtype=torch.float32, device=device
        ).reshape(1, 10).clone().requires_grad_(True)
    else:
        pose = torch.zeros((frame_count, args.pca_components), device=device, requires_grad=True)
    if initial is None and args.rigid_initialization:
        orient_initial, translation_initial = initialize_rigid_parameters(
            model, target, fit_valid, confidence, args.pca_components, device
        )
        orient = orient_initial.detach().clone().requires_grad_(True)
        transl = translation_initial.detach().clone().requires_grad_(True)
    elif initial is None:
        orient = torch.zeros((frame_count, 3), device=device, requires_grad=True)
        wrist = torch.where(fit_valid[:, 0, None], target[:, 0], torch.nan).clone()
        for frame in range(frame_count):
            if not torch.isfinite(wrist[frame]).all():
                available = target[frame, fit_valid[frame]]
                wrist[frame] = available.median(dim=0).values if len(available) else torch.tensor(
                    [0.0, 0.0, 0.25], device=device
                )
        transl = wrist.detach().clone().requires_grad_(True)
    if initial is None:
        betas = torch.zeros((1, 10), device=device, requires_grad=True)

    quality = (fit_valid.float() * confidence).sum(dim=1)
    shape_count = min(args.shape_frames, frame_count)
    shape_ids = torch.topk(quality, k=shape_count).indices.sort().values

    def weighted_transition_mean(delta, weights):
        per_transition = delta.square().mean(dim=-1)
        return (per_transition * weights).sum() / weights.sum().clamp_min(1.0)

    def objective(ids, include_temporal: bool, boundary=None):
        batch_betas = betas.expand(len(ids), -1)
        vertices, joints, _ = run_model(
            model, batch_betas, orient[ids], pose[ids], transl[ids]
        )
        mask = fit_valid[ids]
        weights3d = confidence[ids] * mask
        loss3d = robust_weighted_loss(joints - target[ids], weights3d, 0.006)

        left_prediction = project_rectified(joints, rotation, p1)
        right_prediction = project_rectified(joints, rotation, p2)
        weights2d_left = image_observation_weights(
            confidence[ids], left_px[ids], left_px_valid[ids] & fit_valid[ids]
        )
        weights2d_right = image_observation_weights(
            confidence[ids], right_px[ids], right_px_valid[ids] & fit_valid[ids]
        )
        loss2d = robust_weighted_loss((left_prediction - left_px[ids]) / 100.0, weights2d_left, 0.02)
        loss2d += robust_weighted_loss((right_prediction - right_px[ids]) / 100.0, weights2d_right, 0.02)
        pinch_loss = pinch_distance_loss(
            joints, target[ids], mask, confidence[ids],
            getattr(args, "pinch_threshold_m", 0.025),
        )
        contact_tip_loss = contact_tip_alignment_loss(
            joints, target[ids], mask, confidence[ids],
            left_prediction, right_prediction, left_px[ids], right_px[ids],
            left_px_valid[ids] & fit_valid[ids], right_px_valid[ids] & fit_valid[ids],
            getattr(args, "contact_tip_threshold_m", 0.035),
        )
        pose_prior = pose[ids].square().mean()
        shape_prior = betas.square().mean()
        pose_temporal = torch.tensor(0.0, device=device)
        rigid_temporal = torch.tensor(0.0, device=device)
        acceleration = torch.tensor(0.0, device=device)
        if include_temporal and len(ids) > 1:
            transition_weights = transition_profile["temporal_strength"][ids[1:]]
            pose_temporal = weighted_transition_mean(
                pose[ids[1:]] - pose[ids[:-1]], transition_weights
            )
            rigid_temporal = 0.25 * weighted_transition_mean(
                orient[ids[1:]] - orient[ids[:-1]], transition_weights
            )
            rigid_temporal += 5.0 * weighted_transition_mean(
                transl[ids[1:]] - transl[ids[:-1]], transition_weights
            )
            if len(ids) > 2:
                acceleration_weights = torch.maximum(
                    transition_weights[1:], transition_weights[:-1]
                )
                acceleration = weighted_transition_mean(
                    pose[ids[2:]] - 2.0 * pose[ids[1:-1]] + pose[ids[:-2]],
                    acceleration_weights,
                )
                acceleration += 0.25 * weighted_transition_mean(
                    orient[ids[2:]] - 2.0 * orient[ids[1:-1]] + orient[ids[:-2]],
                    acceleration_weights,
                )
                acceleration += 5.0 * weighted_transition_mean(
                    transl[ids[2:]] - 2.0 * transl[ids[1:-1]] + transl[ids[:-2]],
                    acceleration_weights,
                )
        boundary_loss = torch.tensor(0.0, device=device)
        if boundary is not None and len(boundary["ids"]):
            anchor_ids = boundary["ids"]
            boundary_loss = (pose[anchor_ids] - boundary["pose"]).square().mean()
            boundary_loss += 0.25 * (orient[anchor_ids] - boundary["orient"]).square().mean()
            boundary_loss += 5.0 * (transl[anchor_ids] - boundary["transl"]).square().mean()
        temporal = (
            getattr(args, "w_temporal", 0.08) * pose_temporal
            + getattr(args, "w_rigid_temporal", 0.01) * rigid_temporal
            + getattr(args, "w_acceleration", 0.004) * acceleration
            + getattr(args, "boundary_weight", 0.10) * boundary_loss
        )
        total = (
            args.w_3d * loss3d + args.w_2d * loss2d
            + getattr(args, "w_pinch", 0.0) * pinch_loss
            + getattr(args, "w_contact_tips", 0.0) * contact_tip_loss
            + args.w_pose * pose_prior + args.w_shape * shape_prior + temporal
        )
        return total, (loss3d, loss2d, pinch_loss, contact_tip_loss, pose_prior, shape_prior, temporal), vertices, joints

    if args.shape_iterations > 0:
        shape_parameters = [betas, pose, orient, transl]
        optimizer = torch.optim.Adam(shape_parameters, lr=args.learning_rate)
        for _ in range(args.shape_iterations):
            optimizer.zero_grad(set_to_none=True)
            loss, _, _, _ = objective(shape_ids, False)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(shape_parameters, 5.0)
            optimizer.step()

    betas.requires_grad_(False)
    all_ids = torch.arange(frame_count, device=device)
    window_size = min(args.pose_window, frame_count) if args.pose_window > 0 else frame_count
    effective_overlap = min(args.pose_overlap, max(window_size - 1, 0))
    window_step = window_size - effective_overlap
    if window_step <= 0:
        raise ValueError("--pose-overlap must be smaller than --pose-window")
    segments = support_segments(
        frame_support.detach().cpu().numpy(), max_unobserved_gap
    )
    for segment_start, segment_end in segments:
        segment_count = segment_end - segment_start + 1
        segment_window_size = min(window_size, segment_count)
        segment_overlap = min(effective_overlap, max(segment_window_size - 1, 0))
        segment_step = segment_window_size - segment_overlap
        window_starts = list(range(segment_start, segment_end + 1, segment_step))
        if window_starts and window_starts[-1] + segment_window_size - 1 < segment_end:
            window_starts.append(segment_end - segment_window_size + 1)
        for window_start in window_starts:
            window_end = min(segment_end + 1, window_start + segment_window_size)
            window_ids = all_ids[window_start:window_end]
            boundary = None
            if window_start > segment_start and segment_overlap > 0:
                anchor_ids = window_ids[:min(segment_overlap, len(window_ids))]
                boundary = {
                    "ids": anchor_ids,
                    "pose": pose[anchor_ids].detach().clone(),
                    "orient": orient[anchor_ids].detach().clone(),
                    "transl": transl[anchor_ids].detach().clone(),
                }
            # Adam state must be local to this window. Reusing one optimizer for the
            # full tensors lets stale momentum move frames that no longer participate
            # in the current loss, producing large unexplained solution jumps.
            optimizer = torch.optim.Adam(
                [pose, orient, transl], lr=args.learning_rate * 0.55
            )
            for _ in range(args.pose_iterations):
                optimizer.zero_grad(set_to_none=True)
                loss, _, _, _ = objective(window_ids, True, boundary)
                loss.backward()
                torch.nn.utils.clip_grad_norm_([pose, orient, transl], 5.0)
                optimizer.step()
            limit_parameter_transitions_(pose, orient, transl, window_ids, transition_profile)

    # Overlapping windows can modify their shared frames after an earlier local
    # clamp. Enforce the trust region once more on each complete motion segment so
    # the exported sequence cannot violate the final static/dynamic step limits.
    for segment_start, segment_end in segments:
        segment_ids = all_ids[segment_start:segment_end + 1]
        limit_parameter_transitions_(pose, orient, transl, segment_ids, transition_profile)

    # Frames without enough real observations are gap states, not optimization
    # targets. Windowed Adam may still move them through temporal gradients after
    # the warm-start repair. Reconstruct those states from their fitted neighbours
    # once more before export so detector dropouts cannot reappear as mesh jumps.
    repaired_final = repair_low_support_initial({
        "betas": betas.detach().cpu().numpy(),
        "hand_pose_pca": pose.detach().cpu().numpy(),
        "global_orient": orient.detach().cpu().numpy(),
        "translation": transl.detach().cpu().numpy(),
    }, frame_support.detach().cpu().numpy())
    with torch.no_grad():
        pose.copy_(torch.as_tensor(
            repaired_final["hand_pose_pca"], dtype=pose.dtype, device=device
        ))
        orient.copy_(torch.as_tensor(
            repaired_final["global_orient"], dtype=orient.dtype, device=device
        ))
        transl.copy_(torch.as_tensor(
            repaired_final["translation"], dtype=transl.dtype, device=device
        ))

    with torch.no_grad():
        vertices, joints, model_output = run_model(
            model, betas.expand(frame_count, -1), orient, pose, transl
        )
        image_alignment = torch.zeros_like(transl)
        if getattr(args, "image_rigid_alignment", False):
            image_alignment = image_space_translation_alignment(
                joints, left_px, rotation, p1, transition_profile["quality"],
                getattr(args, "image_alignment_max_m", 0.025),
            )
            transl.add_(image_alignment)
            vertices, joints, model_output = run_model(
                model, betas.expand(frame_count, -1), orient, pose, transl
            )
        total, terms, _, _ = objective(all_ids, True)
        errors = torch.linalg.vector_norm(joints - target, dim=-1)
        errors = torch.where(fit_valid, errors, torch.nan)
        left_prediction = project_rectified(joints, rotation, p1)
        right_prediction = project_rectified(joints, rotation, p2)
        left_pixel_error = torch.linalg.vector_norm(left_prediction - left_px, dim=-1)
        right_pixel_error = torch.linalg.vector_norm(right_prediction - right_px, dim=-1)
        left_pixel_error = torch.where(torch.isfinite(left_px).all(dim=-1), left_pixel_error, torch.nan)
        right_pixel_error = torch.where(torch.isfinite(right_px).all(dim=-1), right_pixel_error, torch.nan)
        expanded_hand_pose = getattr(model_output, "hand_pose", None)
        full_pose = getattr(model_output, "full_pose", None)
        result = {
            "vertices": vertices.cpu().numpy(),
            "joints": joints.cpu().numpy(),
            "betas": betas.cpu().numpy()[0],
            "global_orient": orient.cpu().numpy(),
            "hand_pose_pca": pose.cpu().numpy(),
            "translation": transl.cpu().numpy(),
            "joint_error_m": errors.cpu().numpy(),
            "left_reprojection_error_px": left_pixel_error.cpu().numpy(),
            "right_reprojection_error_px": right_pixel_error.cpu().numpy(),
            "loss": float(total.cpu()),
            "loss_terms": [float(term.cpu()) for term in terms],
            "observation_quality": transition_profile_np["quality"],
            "image_motion_px": transition_profile_np["image_motion_px"],
            "temporal_strength": transition_profile_np["temporal_strength"],
            "image_alignment_m": image_alignment.cpu().numpy(),
            "render_valid": bridge_short_false_gaps(
                observations["valid"].sum(axis=1) >= min_fit_points,
                max_unobserved_gap,
            ),
            "fit_segment_id": np.full(frame_count, -1, dtype=np.int32),
        }
        for segment_id, (segment_start, segment_end) in enumerate(segments):
            result["fit_segment_id"][segment_start:segment_end + 1] = segment_id
        if expanded_hand_pose is not None:
            result["hand_pose_axis_angle"] = expanded_hand_pose.reshape(frame_count, 15, 3).cpu().numpy()
        if full_pose is not None:
            result["full_pose_axis_angle"] = full_pose.reshape(frame_count, 16, 3).cpu().numpy()
    return result


def write_track_csv(path: Path, pair_indices: np.ndarray, track_id: int, result: dict) -> None:
    fields = ["pair_index", "track_id", "landmark_index", "joint_name", "x_m", "y_m", "z_m", "fit_error_m"]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for frame, pair in enumerate(pair_indices):
            for joint in range(21):
                point = result["joints"][frame, joint]
                error = result["joint_error_m"][frame, joint]
                writer.writerow({
                    "pair_index": int(pair), "track_id": int(track_id), "landmark_index": joint,
                    "joint_name": MEDIAPIPE_NAMES[joint], "x_m": f"{point[0]:.9f}",
                    "y_m": f"{point[1]:.9f}", "z_m": f"{point[2]:.9f}",
                    "fit_error_m": f"{error:.9f}" if np.isfinite(error) else "nan",
                })


def write_parameter_csv(path: Path, pair_indices: np.ndarray, track_id: int,
                        handedness: str, result: dict) -> None:
    pca_count = result["hand_pose_pca"].shape[1]
    fields = ["pair_index", "track_id", "handedness", "translation_x_m", "translation_y_m",
              "translation_z_m", "global_orient_x_rad", "global_orient_y_rad",
              "global_orient_z_rad"]
    fields += [f"pose_pca_{index}" for index in range(pca_count)]
    fields += [f"beta_{index}" for index in range(10)]
    fields += [f"joint_{joint}_{axis}_rad" for joint in range(15) for axis in ("x", "y", "z")]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for frame, pair in enumerate(pair_indices):
            row = {
                "pair_index": int(pair), "track_id": track_id, "handedness": handedness,
                "translation_x_m": f"{result['translation'][frame, 0]:.9f}",
                "translation_y_m": f"{result['translation'][frame, 1]:.9f}",
                "translation_z_m": f"{result['translation'][frame, 2]:.9f}",
                "global_orient_x_rad": f"{result['global_orient'][frame, 0]:.9f}",
                "global_orient_y_rad": f"{result['global_orient'][frame, 1]:.9f}",
                "global_orient_z_rad": f"{result['global_orient'][frame, 2]:.9f}",
            }
            row.update({f"pose_pca_{index}": f"{result['hand_pose_pca'][frame, index]:.9f}"
                        for index in range(pca_count)})
            row.update({f"beta_{index}": f"{result['betas'][index]:.9f}" for index in range(10)})
            axis_angle = result.get("hand_pose_axis_angle")
            if axis_angle is not None:
                row.update({
                    f"joint_{joint}_{axis}_rad": f"{axis_angle[frame, joint, axis_index]:.9f}"
                    for joint in range(15)
                    for axis_index, axis in enumerate(("x", "y", "z"))
                })
            writer.writerow(row)


def render_track_video(path: Path, pair_indices: np.ndarray, track_id: int, handedness: str,
                       observations: np.ndarray, observed_valid: np.ndarray, result: dict,
                       faces: np.ndarray, fps: float = 30.0) -> None:
    vertices = result["vertices"]
    joints = result["joints"]
    finite = np.concatenate((vertices.reshape(-1, 3), joints.reshape(-1, 3)), axis=0)
    finite = finite[np.isfinite(finite).all(axis=1)]
    bounds = np.percentile(finite, [1, 99], axis=0)
    padding = np.maximum((bounds[1] - bounds[0]) * 0.15, 0.025)
    bounds[0] -= padding
    bounds[1] += padding
    width, height = 1280, 720
    panels = ((55, 90, 550, 550), (675, 90, 550, 550))
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"cannot create {path}")

    def map_points(points: np.ndarray, axes: tuple[int, int], panel) -> np.ndarray:
        x0, y0, panel_width, panel_height = panel
        a, b = axes
        x = x0 + np.clip((points[:, a] - bounds[0, a]) / max(bounds[1, a] - bounds[0, a], 1e-6), 0, 1) * panel_width
        y = y0 + panel_height - np.clip((points[:, b] - bounds[0, b]) / max(bounds[1, b] - bounds[0, b], 1e-6), 0, 1) * panel_height
        return np.column_stack((x, y)).astype(np.int32)

    try:
        for frame, pair in enumerate(pair_indices):
            canvas = np.full((height, width, 3), 20, dtype=np.uint8)
            cv2.putText(canvas, f"MANO fit | T{track_id} {handedness} | pair {int(pair)}", (35, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (235, 235, 235), 2, cv2.LINE_AA)
            for title, axes, panel in (("Front: X / Y", (0, 1), panels[0]),
                                       ("Top: X / Z", (0, 2), panels[1])):
                cv2.putText(canvas, title, (panel[0], 78), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (210, 210, 210), 2, cv2.LINE_AA)
                cv2.rectangle(canvas, (panel[0], panel[1]),
                              (panel[0] + panel[2], panel[1] + panel[3]), (70, 70, 70), 1)
                vertex_pixels = map_points(vertices[frame], axes, panel)
                joint_pixels = map_points(joints[frame], axes, panel)
                depth_axis = ({0, 1, 2} - set(axes)).pop()
                face_depth = vertices[frame, faces, depth_axis].mean(axis=1)
                for face_index in np.argsort(face_depth)[::2]:
                    polygon = vertex_pixels[faces[face_index]]
                    cv2.polylines(canvas, [polygon], True, (105, 120, 145), 1, cv2.LINE_AA)
                for parent, child in (
                    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
                    (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15),
                    (15, 16), (0, 17), (17, 18), (18, 19), (19, 20),
                ):
                    cv2.line(canvas, tuple(joint_pixels[parent]), tuple(joint_pixels[child]),
                             (40, 210, 255), 2, cv2.LINE_AA)
                for joint in range(21):
                    cv2.circle(canvas, tuple(joint_pixels[joint]), 4, (40, 210, 255), -1, cv2.LINE_AA)
                    if observed_valid[frame, joint] and np.isfinite(observations[frame, joint]).all():
                        raw_pixel = map_points(observations[frame, joint][None], axes, panel)[0]
                        cv2.circle(canvas, tuple(raw_pixel), 3, (90, 240, 90), -1, cv2.LINE_AA)
            errors = result["joint_error_m"][frame]
            errors = errors[np.isfinite(errors)] * 1000.0
            label = f"observed joint error median: {np.median(errors):.2f} mm" if len(errors) else "no 3D observations"
            cv2.putText(canvas, label, (35, 690), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (210, 210, 210), 2, cv2.LINE_AA)
            writer.write(canvas)
    finally:
        writer.release()


def main() -> int:
    args = parse_args()
    if (args.w_pinch < 0.0 or args.w_contact_tips < 0.0
            or args.pinch_threshold_m <= 0.0 or args.contact_tip_threshold_m <= 0.0):
        raise ValueError("pinch weight must be non-negative and threshold must be positive")
    if args.min_fit_observed_points < 1 or args.min_fit_observed_points > 21:
        raise ValueError("--min-fit-observed-points must be between 1 and 21")
    if args.max_unobserved_gap < 0:
        raise ValueError("--max-unobserved-gap must be non-negative")
    source = args.mano_source.resolve()
    model_dir = args.model_dir.resolve()
    revision = validate_source_and_assets(source, model_dir, args.mano_convention)
    mano = import_mano(source)
    import torch

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    data = load_input(args.input.resolve())
    selected_pairs = select_pair_range(data, args)
    if not len(selected_pairs):
        raise RuntimeError("selected pair range is empty")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    selected_track_ids = set(args.track_id or data["track_ids"].astype(int).tolist())
    initial_output = args.initial_output.resolve() if args.initial_output else None
    summaries = []
    start = time.perf_counter()
    left_pixel_key = (
        "left_rectified_px_filtered"
        if "left_rectified_px_filtered" in data else "left_rectified_px"
    )
    right_pixel_key = (
        "right_rectified_px_filtered"
        if "right_rectified_px_filtered" in data else "right_rectified_px"
    )

    for track_slot, track_id_value in enumerate(data["track_ids"]):
        track_id = int(track_id_value)
        if track_id not in selected_track_ids:
            continue
        handedness = str(data["handedness"][track_slot])
        track_has_data = (
            np.any(data["valid"][track_slot, selected_pairs], axis=1)
            | np.any(np.isfinite(data["left_rectified_px"][track_slot, selected_pairs]).all(axis=-1), axis=1)
            | np.any(np.isfinite(data["right_rectified_px"][track_slot, selected_pairs]).all(axis=-1), axis=1)
        )
        active = np.flatnonzero(track_has_data)
        if not len(active):
            continue
        track_pairs = selected_pairs[active[0]:active[-1] + 1]
        model = mano.load(
            model_path=str(model_dir), is_rhand=True, use_pca=True,
            num_pca_comps=args.pca_components, batch_size=len(track_pairs),
            flat_hand_mean=False,
        ).to(device)
        observations = {
            "positions": data["positions_left_camera_m"][track_slot, track_pairs],
            "valid": data["valid"][track_slot, track_pairs],
            "observed": data.get(
                "observed", data["valid"]
            )[track_slot, track_pairs],
            "confidence": data["confidence"][track_slot, track_pairs],
            "left_px": data[left_pixel_key][track_slot, track_pairs],
            "right_px": data[right_pixel_key][track_slot, track_pairs],
            "left_px_valid": data.get(
                "left_rectified_valid",
                np.isfinite(data["left_rectified_px"]).all(axis=-1),
            )[track_slot, track_pairs],
            "right_px_valid": data.get(
                "right_rectified_valid",
                np.isfinite(data["right_rectified_px"]).all(axis=-1),
            )[track_slot, track_pairs],
            "rotation": data["left_to_rectified_rotation"],
            "p1": data["projection_left_rectified"],
            "p2": data["projection_right_rectified"],
        }
        observations = canonicalize_observations(observations, handedness)
        if initial_output is not None:
            initial_path = initial_output / f"track_{track_id}.npz"
            if not initial_path.is_file():
                raise FileNotFoundError(f"warm-start track is missing: {initial_path}")
            with np.load(initial_path) as initial_archive:
                initial_convention = (
                    str(initial_archive["mano_convention"])
                    if "mano_convention" in initial_archive.files else None
                )
                if initial_convention != args.mano_convention:
                    raise RuntimeError(
                        f"warm-start {initial_path} uses {initial_convention!r}, "
                        f"expected {args.mano_convention!r}"
                    )
                initial_pairs = initial_archive["pair_indices"].astype(np.int64)
                requested_pairs = data["pair_indices"][track_pairs].astype(np.int64)
                lookup = {int(pair): index for index, pair in enumerate(initial_pairs)}
                try:
                    initial_indices = np.asarray([lookup[int(pair)] for pair in requested_pairs])
                except KeyError as error:
                    raise RuntimeError(f"warm-start is missing pair {error.args[0]}") from error
                observations["initial"] = {
                    "betas": initial_archive["betas"].copy(),
                    "global_orient": initial_archive["global_orient"][initial_indices].copy(),
                    "hand_pose_pca": initial_archive["hand_pose_pca"][initial_indices].copy(),
                    "translation": initial_archive["translation"][initial_indices].copy(),
                }
        if np.count_nonzero(observations["valid"]) < 21:
            continue
        result = optimize_track(model, observations, args, device)
        rendered_result, rendered_faces = physical_result(
            result, handedness, np.asarray(model.faces)
        )
        prefix = output / f"track_{track_id}"
        np.savez_compressed(
            prefix.with_suffix(".npz"), pair_indices=data["pair_indices"][track_pairs],
            track_id=np.asarray(track_id), handedness=np.asarray(handedness), faces=model.faces,
            mano_convention=np.asarray(args.mano_convention),
            mano_model=np.asarray("MANO_RIGHT.pkl"),
            geometry_space=np.asarray("right_canonical"),
            **result,
        )
        write_track_csv(
            prefix.with_name(prefix.name + "_joints.csv"), data["pair_indices"][track_pairs],
            track_id, rendered_result,
        )
        write_parameter_csv(
            prefix.with_name(prefix.name + "_parameters.csv"),
            data["pair_indices"][track_pairs], track_id, handedness, result,
        )
        if not args.no_video:
            render_track_video(
                prefix.with_name(prefix.name + "_fit.mp4"), data["pair_indices"][track_pairs],
                track_id, handedness,
                mirror_left_points(observations["positions"], handedness),
                observations["valid"], rendered_result, rendered_faces,
                float(data.get("fps", np.asarray(30.0))),
            )
        finite_errors = result["joint_error_m"][np.isfinite(result["joint_error_m"])] * 1000.0
        left_errors = result["left_reprojection_error_px"][
            np.isfinite(result["left_reprojection_error_px"])
        ]
        right_errors = result["right_reprojection_error_px"][
            np.isfinite(result["right_reprojection_error_px"])
        ]
        summaries.append({
            "track_id": track_id, "handedness": handedness,
            "mano_model": "MANO_RIGHT.pkl",
            "geometry_space": "right_canonical",
            "frames": int(len(track_pairs)),
            "render_visible_frames": int(np.count_nonzero(result["render_valid"])),
            "render_hidden_frames": int(np.count_nonzero(~result["render_valid"])),
            "fit_segments": int(np.max(result["fit_segment_id"]) + 1)
            if np.any(result["fit_segment_id"] >= 0) else 0,
            "pair_range": [int(data["pair_indices"][track_pairs[0]]), int(data["pair_indices"][track_pairs[-1]])],
            "loss": result["loss"],
            "joint_error_median_mm": float(np.median(finite_errors)),
            "joint_error_p95_mm": float(np.percentile(finite_errors, 95)),
            "left_reprojection_error_median_px": float(np.median(left_errors)),
            "left_reprojection_error_p95_px": float(np.percentile(left_errors, 95)),
            "right_reprojection_error_median_px": float(np.median(right_errors)),
            "right_reprojection_error_p95_px": float(np.percentile(right_errors, 95)),
            "betas": result["betas"].tolist(),
        })

    if not summaries:
        raise RuntimeError("no tracks were fitted")
    summary = {
        "stage": "mano_sequence_fitting", "input": str(args.input.resolve()),
        "mano_source": str(source), "mano_revision": revision, "model_dir": str(model_dir),
        "mano_convention": args.mano_convention,
        "mano_model": "MANO_RIGHT.pkl",
        "mano_model_by_side": {
            "left": "MANO_RIGHT.pkl", "right": "MANO_RIGHT.pkl",
        },
        "physical_left_transform": "mirror canonical X and reverse triangle winding",
        "torch_version": torch.__version__, "device": str(device),
        "pair_range": [int(data["pair_indices"][selected_pairs[0]]), int(data["pair_indices"][selected_pairs[-1]])],
        "tracks": summaries, "elapsed_seconds": time.perf_counter() - start,
        "parameters": vars(args) | {},
    }
    summary["parameters"] = {key: str(value) if isinstance(value, Path) else value
                             for key, value in summary["parameters"].items()}
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(1)
