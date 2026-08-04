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
    parser.add_argument("--w-pose", type=float, default=0.003)
    parser.add_argument("--w-shape", type=float, default=0.015)
    parser.add_argument("--w-temporal", type=float, default=0.08)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    parser.add_argument("--rigid-initialization", action="store_true", help="initialize each frame with Kabsch")
    parser.add_argument("--no-video", action="store_true")
    return parser.parse_args()


def validate_source_and_assets(source: Path, model_dir: Path) -> str:
    if not (source / "mano" / "model.py").is_file():
        raise FileNotFoundError(f"invalid MANO source: {source}")
    missing = [model_dir / name for name in ("MANO_LEFT.pkl", "MANO_RIGHT.pkl")
               if not (model_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            "missing licensed MANO model data: " + ", ".join(str(path) for path in missing)
        )
    revision = "unknown"
    try:
        revision = subprocess.run(
            ["git", "-C", str(source), "rev-parse", "HEAD"], check=True,
            capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        pass
    return revision


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
    confidence = torch.as_tensor(observations["confidence"], dtype=torch.float32, device=device)
    left_px = torch.as_tensor(observations["left_px"], dtype=torch.float32, device=device)
    right_px = torch.as_tensor(observations["right_px"], dtype=torch.float32, device=device)
    rotation = torch.as_tensor(observations["rotation"], dtype=torch.float32, device=device)
    p1 = torch.as_tensor(observations["p1"], dtype=torch.float32, device=device)
    p2 = torch.as_tensor(observations["p2"], dtype=torch.float32, device=device)
    frame_count = target.shape[0]

    initial = observations.get("initial")
    if initial is not None:
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
            model, target, valid, confidence, args.pca_components, device
        )
        orient = orient_initial.detach().clone().requires_grad_(True)
        transl = translation_initial.detach().clone().requires_grad_(True)
    elif initial is None:
        orient = torch.zeros((frame_count, 3), device=device, requires_grad=True)
        wrist = torch.where(valid[:, 0, None], target[:, 0], torch.nan).clone()
        for frame in range(frame_count):
            if not torch.isfinite(wrist[frame]).all():
                available = target[frame, valid[frame]]
                wrist[frame] = available.median(dim=0).values if len(available) else torch.tensor(
                    [0.0, 0.0, 0.25], device=device
                )
        transl = wrist.detach().clone().requires_grad_(True)
    if initial is None:
        betas = torch.zeros((1, 10), device=device, requires_grad=True)

    quality = (valid.float() * confidence).sum(dim=1)
    shape_count = min(args.shape_frames, frame_count)
    shape_ids = torch.topk(quality, k=shape_count).indices.sort().values

    def objective(ids, include_temporal: bool):
        batch_betas = betas.expand(len(ids), -1)
        vertices, joints, _ = run_model(
            model, batch_betas, orient[ids], pose[ids], transl[ids]
        )
        mask = valid[ids]
        weights3d = confidence[ids] * mask
        loss3d = robust_weighted_loss(joints - target[ids], weights3d, 0.006)

        left_prediction = project_rectified(joints, rotation, p1)
        right_prediction = project_rectified(joints, rotation, p2)
        left_valid = torch.isfinite(left_px[ids]).all(dim=-1)
        right_valid = torch.isfinite(right_px[ids]).all(dim=-1)
        weights2d_left = torch.where(left_valid, torch.maximum(confidence[ids], torch.tensor(0.1, device=device)), 0.0)
        weights2d_right = torch.where(right_valid, torch.maximum(confidence[ids], torch.tensor(0.1, device=device)), 0.0)
        loss2d = robust_weighted_loss((left_prediction - left_px[ids]) / 100.0, weights2d_left, 0.02)
        loss2d += robust_weighted_loss((right_prediction - right_px[ids]) / 100.0, weights2d_right, 0.02)
        pose_prior = pose[ids].square().mean()
        shape_prior = betas.square().mean()
        temporal = torch.tensor(0.0, device=device)
        if include_temporal and len(ids) > 1:
            temporal = (pose[ids[1:]] - pose[ids[:-1]]).square().mean()
            temporal += 0.25 * (orient[ids[1:]] - orient[ids[:-1]]).square().mean()
            temporal += 5.0 * (transl[ids[1:]] - transl[ids[:-1]]).square().mean()
        total = (args.w_3d * loss3d + args.w_2d * loss2d + args.w_pose * pose_prior
                 + args.w_shape * shape_prior + args.w_temporal * temporal)
        return total, (loss3d, loss2d, pose_prior, shape_prior, temporal), vertices, joints

    if args.shape_iterations > 0:
        optimizer = torch.optim.Adam([betas, pose, orient, transl], lr=args.learning_rate)
        for _ in range(args.shape_iterations):
            optimizer.zero_grad(set_to_none=True)
            loss, _, _, _ = objective(shape_ids, False)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([betas, pose, orient, transl], 5.0)
            optimizer.step()

    betas.requires_grad_(False)
    optimizer = torch.optim.Adam([pose, orient, transl], lr=args.learning_rate * 0.55)
    all_ids = torch.arange(frame_count, device=device)
    window_size = min(args.pose_window, frame_count) if args.pose_window > 0 else frame_count
    effective_overlap = min(args.pose_overlap, max(window_size - 1, 0))
    window_step = window_size - effective_overlap
    if window_step <= 0:
        raise ValueError("--pose-overlap must be smaller than --pose-window")
    window_starts = list(range(0, frame_count, window_step))
    if window_starts and window_starts[-1] + window_size < frame_count:
        window_starts.append(frame_count - window_size)
    for window_start in window_starts:
        window_ids = all_ids[window_start:min(frame_count, window_start + window_size)]
        for _ in range(args.pose_iterations):
            optimizer.zero_grad(set_to_none=True)
            loss, _, _, _ = objective(window_ids, True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([pose, orient, transl], 5.0)
            optimizer.step()

    with torch.no_grad():
        total, terms, vertices, joints = objective(all_ids, True)
        vertices, joints, model_output = run_model(
            model, betas.expand(frame_count, -1), orient, pose, transl
        )
        errors = torch.linalg.vector_norm(joints - target, dim=-1)
        errors = torch.where(valid, errors, torch.nan)
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
        }
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
    source = args.mano_source.resolve()
    model_dir = args.model_dir.resolve()
    revision = validate_source_and_assets(source, model_dir)
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

    for track_slot, track_id_value in enumerate(data["track_ids"]):
        track_id = int(track_id_value)
        if track_id not in selected_track_ids:
            continue
        handedness = str(data["handedness"][track_slot])
        is_right = handedness.lower() == "right"
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
            model_path=str(model_dir), is_rhand=is_right, use_pca=True,
            num_pca_comps=args.pca_components, batch_size=len(track_pairs),
            flat_hand_mean=False,
        ).to(device)
        observations = {
            "positions": data["positions_left_camera_m"][track_slot, track_pairs],
            "valid": data["valid"][track_slot, track_pairs],
            "confidence": data["confidence"][track_slot, track_pairs],
            "left_px": data["left_rectified_px"][track_slot, track_pairs],
            "right_px": data["right_rectified_px"][track_slot, track_pairs],
            "rotation": data["left_to_rectified_rotation"],
            "p1": data["projection_left_rectified"],
            "p2": data["projection_right_rectified"],
        }
        if initial_output is not None:
            initial_path = initial_output / f"track_{track_id}.npz"
            if not initial_path.is_file():
                raise FileNotFoundError(f"warm-start track is missing: {initial_path}")
            with np.load(initial_path) as initial_archive:
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
        prefix = output / f"track_{track_id}"
        np.savez_compressed(
            prefix.with_suffix(".npz"), pair_indices=data["pair_indices"][track_pairs],
            track_id=np.asarray(track_id), handedness=np.asarray(handedness), faces=model.faces,
            **result,
        )
        write_track_csv(
            prefix.with_name(prefix.name + "_joints.csv"), data["pair_indices"][track_pairs],
            track_id, result,
        )
        write_parameter_csv(
            prefix.with_name(prefix.name + "_parameters.csv"),
            data["pair_indices"][track_pairs], track_id, handedness, result,
        )
        if not args.no_video:
            render_track_video(
                prefix.with_name(prefix.name + "_fit.mp4"), data["pair_indices"][track_pairs],
                track_id, handedness, observations["positions"], observations["valid"], result,
                np.asarray(model.faces), float(data.get("fps", np.asarray(30.0))),
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
            "frames": int(len(track_pairs)),
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
