#!/usr/bin/env python3
"""Overlay fitted MANO meshes and geometric finger angles on EGO video."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from mediapipe_left_baseline import (  # noqa: E402
    HAND_CONNECTIONS,
    camera_matrices,
    create_stereo_rectification,
    unique_file,
)
from fit_mano_sequence import MANO_TO_MEDIAPIPE, import_mano  # noqa: E402
from mano_conventions import (  # noqa: E402
    WILOR_RIGHT_CANONICAL,
    physicalize_geometry,
)


FINGER_CHAINS = {
    "thumb": ((0, 1, 2, "cmc"), (1, 2, 3, "mcp"), (2, 3, 4, "ip")),
    "index": ((0, 5, 6, "mcp"), (5, 6, 7, "pip"), (6, 7, 8, "dip")),
    "middle": ((0, 9, 10, "mcp"), (9, 10, 11, "pip"), (10, 11, 12, "dip")),
    "ring": ((0, 13, 14, "mcp"), (13, 14, 15, "pip"), (14, 15, 16, "dip")),
    "pinky": ((0, 17, 18, "mcp"), (17, 18, 19, "pip"), (18, 19, 20, "dip")),
}

ANGLE_KEYS = tuple(
    [f"{finger}_{joint}_bend_deg" for finger, chains in FINGER_CHAINS.items() for *_, joint in chains]
    + [f"{finger}_spread_deg" for finger in FINGER_CHAINS]
)

DISPLAY_JOINTS = {
    "thumb": ("cmc", "mcp", "ip"),
    "index": ("mcp", "pip", "dip"),
    "middle": ("mcp", "pip", "dip"),
    "ring": ("mcp", "pip", "dip"),
    "pinky": ("mcp", "pip", "dip"),
}

# MANO hand-pose order (global orientation is not included here).
MANO_POSE_INDICES = {
    "index": (0, 1, 2),
    "middle": (3, 4, 5),
    "pinky": (6, 7, 8),
    "ring": (9, 10, 11),
    "thumb": (12, 13, 14),
}

MANO_POSE_NAMES = (
    "index_mcp", "index_pip", "index_dip",
    "middle_mcp", "middle_pip", "middle_dip",
    "pinky_mcp", "pinky_pip", "pinky_dip",
    "ring_mcp", "ring_pip", "ring_dip",
    "thumb_cmc", "thumb_mcp", "thumb_ip",
)

# Five thumb DOFs plus four DOFs for every other finger = 21.
KINEMATIC_LAYOUT = {
    "thumb": (
        ("CF", "thumb_cmc_flex_rad"),
        ("CA", "thumb_cmc_abduction_rad"),
        ("OP", "thumb_cmc_opposition_rad"),
        ("MF", "thumb_mcp_flex_rad"),
        ("IF", "thumb_ip_flex_rad"),
    ),
    "index": (
        ("MF", "index_mcp_flex_rad"),
        ("MA", "index_mcp_abduction_rad"),
        ("PF", "index_pip_flex_rad"),
        ("DF", "index_dip_flex_rad"),
    ),
    "middle": (
        ("MF", "middle_mcp_flex_rad"),
        ("MA", "middle_mcp_abduction_rad"),
        ("PF", "middle_pip_flex_rad"),
        ("DF", "middle_dip_flex_rad"),
    ),
    "ring": (
        ("MF", "ring_mcp_flex_rad"),
        ("MA", "ring_mcp_abduction_rad"),
        ("PF", "ring_pip_flex_rad"),
        ("DF", "ring_dip_flex_rad"),
    ),
    "pinky": (
        ("MF", "pinky_mcp_flex_rad"),
        ("MA", "pinky_mcp_abduction_rad"),
        ("PF", "pinky_pip_flex_rad"),
        ("DF", "pinky_dip_flex_rad"),
    ),
}

KINEMATIC_KEYS = tuple(key for entries in KINEMATIC_LAYOUT.values() for _, key in entries)

KINEMATIC_LIMITS_RAD = {
    key: (0.8 if "abduction" in key else 0.5 if "opposition" in key else 3.2)
    for key in KINEMATIC_KEYS
}

TRACK_COLORS = {
    "Right": (255, 115, 35),
    "Left": (35, 135, 255),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render fitted MANO meshes on the original EGO left camera with angle gauges."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--session", type=Path, help="legacy Orbbec EGO KB session")
    source.add_argument(
        "--normalized-dataset", type=Path,
        help="normalized raw KB/DS dataset used for original-image overlay",
    )
    parser.add_argument(
        "--rectified-dataset", type=Path,
        help="required with --normalized-dataset; supplies R1/P1",
    )
    parser.add_argument("--mano-fit", required=True, type=Path, help="directory containing track_*.npz")
    parser.add_argument("--mano-source", type=Path, default=Path("third_party/MANO"))
    parser.add_argument("--model-dir", type=Path, default=Path("models/mano"))
    parser.add_argument("--stereo-frames", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--balance", type=float, default=0.0)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--panel-width", type=int, default=650)
    parser.add_argument("--mesh-alpha", type=float, default=0.38)
    parser.add_argument("--angle-radius", type=int, default=2)
    parser.add_argument("--trajectory-length", type=int, default=120,
                        help="number of recent frames drawn for each palm trajectory")
    parser.add_argument("--trajectory-max-jump-m", type=float, default=0.12,
                        help="break a trajectory segment when the palm jumps farther than this")
    parser.add_argument("--start-pair", type=int, default=0)
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument("--no-video", action="store_true")
    return parser.parse_args()


def unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm < 1e-9:
        return np.full(3, np.nan, dtype=np.float64)
    return np.asarray(vector, dtype=np.float64) / norm


def rotation_matrix_to_rpy(rotation: np.ndarray) -> np.ndarray:
    """Return ZYX roll/pitch/yaw where R = Rz(yaw) Ry(pitch) Rx(roll)."""
    rotation = np.asarray(rotation, dtype=np.float64)
    pitch = math.asin(float(np.clip(-rotation[2, 0], -1.0, 1.0)))
    cosine = math.cos(pitch)
    if abs(cosine) > 1e-7:
        roll = math.atan2(rotation[2, 1], rotation[2, 2])
        yaw = math.atan2(rotation[1, 0], rotation[0, 0])
    else:
        roll = math.atan2(-rotation[1, 2], rotation[1, 1])
        yaw = 0.0
    return np.asarray((roll, pitch, yaw), dtype=np.float64)


def rotation_matrix_to_quaternion(rotation: np.ndarray) -> np.ndarray:
    """Return quaternion in x,y,z,w order."""
    rotation = np.asarray(rotation, dtype=np.float64)
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * scale
        qx = (rotation[2, 1] - rotation[1, 2]) / scale
        qy = (rotation[0, 2] - rotation[2, 0]) / scale
        qz = (rotation[1, 0] - rotation[0, 1]) / scale
    else:
        diagonal = np.diag(rotation)
        index = int(np.argmax(diagonal))
        if index == 0:
            scale = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
            qw = (rotation[2, 1] - rotation[1, 2]) / scale
            qx = 0.25 * scale
            qy = (rotation[0, 1] + rotation[1, 0]) / scale
            qz = (rotation[0, 2] + rotation[2, 0]) / scale
        elif index == 1:
            scale = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
            qw = (rotation[0, 2] - rotation[2, 0]) / scale
            qx = (rotation[0, 1] + rotation[1, 0]) / scale
            qy = 0.25 * scale
            qz = (rotation[1, 2] + rotation[2, 1]) / scale
        else:
            scale = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
            qw = (rotation[1, 0] - rotation[0, 1]) / scale
            qx = (rotation[0, 2] + rotation[2, 0]) / scale
            qy = (rotation[1, 2] + rotation[2, 1]) / scale
            qz = 0.25 * scale
    quaternion = np.asarray((qx, qy, qz, qw), dtype=np.float64)
    return quaternion / max(float(np.linalg.norm(quaternion)), 1e-12)


def compute_hand_end_effector_pose(joints: np.ndarray, handedness: str) -> dict[str, np.ndarray]:
    """Construct a stable palm-centred 6D pose in the left camera optical frame.

    Local +Y points from wrist toward the middle MCP. Local +Z is the
    handedness-normalized palm normal, and +X completes the right-handed frame.
    """
    joints = np.asarray(joints, dtype=np.float64)
    if joints.shape != (21, 3) or not np.isfinite(joints).all():
        raise ValueError("expected finite (21, 3) hand joints")
    wrist = joints[0]
    index_mcp, middle_mcp, ring_mcp, pinky_mcp = joints[[5, 9, 13, 17]]
    origin = np.mean(joints[[0, 5, 9, 13, 17]], axis=0)
    mirror = 1.0 if handedness == "Right" else -1.0
    z_axis = unit(np.cross(index_mcp - wrist, pinky_mcp - wrist) * -mirror)
    y_hint = unit(middle_mcp - wrist)
    x_axis = unit(np.cross(y_hint, z_axis))
    y_axis = unit(np.cross(z_axis, x_axis))
    if not all(np.isfinite(axis).all() for axis in (x_axis, y_axis, z_axis)):
        raise RuntimeError("cannot construct hand end-effector frame")
    rotation = np.column_stack((x_axis, y_axis, z_axis))
    return {
        "position_m": origin,
        "rotation_matrix": rotation,
        "rpy_rad": rotation_matrix_to_rpy(rotation),
        "quaternion_xyzw": rotation_matrix_to_quaternion(rotation),
    }


def bend_angle(parent: np.ndarray, joint: np.ndarray, child: np.ndarray) -> float:
    """Return geometric bend: 0 degrees straight, increasing while flexed."""
    incoming = unit(joint - parent)
    outgoing = unit(child - joint)
    if not np.isfinite(incoming).all() or not np.isfinite(outgoing).all():
        return float("nan")
    cosine = float(np.clip(np.dot(incoming, outgoing), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def signed_angle_on_plane(reference: np.ndarray, vector: np.ndarray, normal: np.ndarray) -> float:
    normal = unit(normal)
    if not np.isfinite(normal).all():
        return float("nan")
    reference = unit(reference - normal * np.dot(reference, normal))
    vector = unit(vector - normal * np.dot(vector, normal))
    if not np.isfinite(reference).all() or not np.isfinite(vector).all():
        return float("nan")
    sine = float(np.dot(np.cross(reference, vector), normal))
    cosine = float(np.clip(np.dot(reference, vector), -1.0, 1.0))
    return float(np.degrees(np.arctan2(sine, cosine)))


def spread_angle_on_plane(reference: np.ndarray, vector: np.ndarray, normal: np.ndarray) -> float:
    """Return finger spread in [-90, 90], treating projected bone direction as an axis."""
    angle = signed_angle_on_plane(reference, vector, normal)
    if not np.isfinite(angle):
        return angle
    if angle > 90.0:
        angle -= 180.0
    elif angle < -90.0:
        angle += 180.0
    return angle


def compute_joint_angles(joints: np.ndarray) -> dict[str, float]:
    joints = np.asarray(joints, dtype=np.float64)
    if joints.shape != (21, 3):
        raise ValueError(f"expected (21, 3) joints, got {joints.shape}")
    result: dict[str, float] = {}
    for finger, chains in FINGER_CHAINS.items():
        for parent, joint, child, joint_name in chains:
            result[f"{finger}_{joint_name}_bend_deg"] = bend_angle(
                joints[parent], joints[joint], joints[child]
            )

    palm_normal = np.cross(joints[5] - joints[0], joints[17] - joints[0])
    reference = joints[10] - joints[9]
    spread_vectors = {
        "thumb": joints[2] - joints[1],
        "index": joints[6] - joints[5],
        "middle": joints[10] - joints[9],
        "ring": joints[14] - joints[13],
        "pinky": joints[18] - joints[17],
    }
    for finger, vector in spread_vectors.items():
        result[f"{finger}_spread_deg"] = spread_angle_on_plane(reference, vector, palm_normal)
    return result


def build_kinematic_axes(rest_joints: np.ndarray) -> dict[str, np.ndarray]:
    """Build MANO rest-pose axes used to reduce 45 pose components to 21 DOFs."""
    joints = np.asarray(rest_joints, dtype=np.float64)
    if joints.shape != (21, 3):
        raise ValueError(f"expected (21, 3) rest joints, got {joints.shape}")
    palm_normal = unit(np.cross(joints[5] - joints[0], joints[17] - joints[0]))
    if not np.isfinite(palm_normal).all():
        raise RuntimeError("cannot construct MANO palm normal")
    semantic = {
        "thumb": (("cmc", 1, 2), ("mcp", 2, 3), ("ip", 3, 4)),
        "index": (("mcp", 5, 6), ("pip", 6, 7), ("dip", 7, 8)),
        "middle": (("mcp", 9, 10), ("pip", 10, 11), ("dip", 11, 12)),
        "ring": (("mcp", 13, 14), ("pip", 14, 15), ("dip", 15, 16)),
        "pinky": (("mcp", 17, 18), ("pip", 18, 19), ("dip", 19, 20)),
    }
    axes: dict[str, np.ndarray] = {"palm_normal": palm_normal}
    for finger, segments in semantic.items():
        for joint_name, start, end in segments:
            direction = unit(joints[end] - joints[start])
            flex_axis = unit(np.cross(direction, palm_normal))
            if not np.isfinite(flex_axis).all():
                raise RuntimeError(f"cannot construct {finger} {joint_name} flexion axis")
            axes[f"{finger}_{joint_name}_flex"] = flex_axis
    axes["thumb_cmc_opposition"] = unit(joints[2] - joints[1])
    return axes


def extract_kinematic_sequence(
    hand_pose_axis_angle: np.ndarray,
    axes: dict[str, np.ndarray],
    handedness: str,
) -> np.ndarray:
    """Project MANO mean-relative local rotation vectors onto 21 anatomical axes."""
    pose = np.asarray(hand_pose_axis_angle, dtype=np.float64)
    if pose.ndim != 3 or pose.shape[1:] != (15, 3):
        raise ValueError(f"expected (frames, 15, 3) MANO pose, got {pose.shape}")
    mirror = 1.0 if handedness == "Right" else -1.0
    values: dict[str, np.ndarray] = {}

    thumb_cmc, thumb_mcp, thumb_ip = MANO_POSE_INDICES["thumb"]
    values["thumb_cmc_flex_rad"] = (
        pose[:, thumb_cmc] @ axes["thumb_cmc_flex"] * -mirror
    )
    values["thumb_cmc_abduction_rad"] = pose[:, thumb_cmc] @ axes["palm_normal"]
    values["thumb_cmc_opposition_rad"] = (
        pose[:, thumb_cmc] @ axes["thumb_cmc_opposition"] * mirror
    )
    values["thumb_mcp_flex_rad"] = (
        pose[:, thumb_mcp] @ axes["thumb_mcp_flex"] * mirror
    )
    values["thumb_ip_flex_rad"] = pose[:, thumb_ip] @ axes["thumb_ip_flex"] * -mirror

    for finger in ("index", "middle", "ring", "pinky"):
        mcp, pip, dip = MANO_POSE_INDICES[finger]
        values[f"{finger}_mcp_flex_rad"] = (
            pose[:, mcp] @ axes[f"{finger}_mcp_flex"] * mirror
        )
        values[f"{finger}_mcp_abduction_rad"] = pose[:, mcp] @ axes["palm_normal"]
        values[f"{finger}_pip_flex_rad"] = (
            pose[:, pip] @ axes[f"{finger}_pip_flex"] * mirror
        )
        values[f"{finger}_dip_flex_rad"] = (
            pose[:, dip] @ axes[f"{finger}_dip_flex"] * mirror
        )
    return np.column_stack([values[key] for key in KINEMATIC_KEYS])


def load_rest_joints(mano, model_dir: Path, track: dict) -> np.ndarray:
    import torch

    model = mano.load(
        str(model_dir), is_rhand=True,
        num_pca_comps=45,
        batch_size=1, flat_hand_mean=True,
    )
    betas = torch.as_tensor(track["betas"][None], dtype=torch.float32)
    with torch.no_grad():
        output = model(betas=betas, return_tips=True)
    joints = output.joints[0].detach().cpu().numpy()[MANO_TO_MEDIAPIPE]
    if joints.shape != (21, 3) or not np.isfinite(joints).all():
        raise RuntimeError("invalid flat-hand MANO joints")
    return joints


def median_filter(values: np.ndarray, radius: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or radius < 0:
        raise ValueError("values must be 2D and radius non-negative")
    output = np.full_like(values, np.nan)
    for frame in range(len(values)):
        window = values[max(0, frame - radius):min(len(values), frame + radius + 1)]
        for column in range(values.shape[1]):
            finite = window[:, column][np.isfinite(window[:, column])]
            if len(finite):
                output[frame, column] = np.median(finite)
    return output


def trajectory_displacements(
    positions_m: np.ndarray,
    initial_rotation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    positions = np.asarray(positions_m, dtype=np.float64)
    rotation = np.asarray(initial_rotation, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3 or len(positions) == 0:
        raise ValueError("positions must be a non-empty (frames, 3) array")
    if rotation.shape != (3, 3):
        raise ValueError("initial_rotation must be (3, 3)")
    delta_camera = positions - positions[0]
    delta_hand0 = (rotation.T @ delta_camera.T).T
    return delta_camera, delta_hand0


def load_tracks(
    directory: Path,
    angle_radius: int,
    mano_source: Path,
    model_dir: Path,
) -> list[dict]:
    mano = import_mano(mano_source)
    tracks = []
    for path in sorted(directory.glob("track_*.npz")):
        with np.load(path) as archive:
            required = {
                "pair_indices", "track_id", "handedness", "faces", "vertices", "joints",
                "betas", "hand_pose_axis_angle", "full_pose_axis_angle",
            }
            missing = required - set(archive.files)
            if missing:
                raise RuntimeError(f"{path} missing arrays: {sorted(missing)}")
            track = {name: archive[name].copy() for name in required}
            track["mano_convention"] = (
                str(archive["mano_convention"])
                if "mano_convention" in archive.files else None
            )
            track["render_valid"] = (
                archive["render_valid"].astype(bool).copy()
                if "render_valid" in archive.files
                else np.ones(len(track["pair_indices"]), dtype=bool)
            )
        pairs = track["pair_indices"].astype(np.int64)
        if len(np.unique(pairs)) != len(pairs):
            raise RuntimeError(f"duplicate pair index in {path}")
        track["lookup"] = {int(pair): index for index, pair in enumerate(pairs)}
        if track["render_valid"].shape != (len(pairs),):
            raise RuntimeError(f"invalid render_valid shape in {path}")
        track["track_id"] = int(track["track_id"])
        track["handedness"] = str(track["handedness"])
        if track["mano_convention"] != WILOR_RIGHT_CANONICAL:
            raise RuntimeError(
                f"{path} must use {WILOR_RIGHT_CANONICAL}; "
                f"got {track['mano_convention']!r}"
            )
        track["vertices"], track["joints"], track["faces"] = physicalize_geometry(
            track["vertices"], track["joints"], track["faces"], track["handedness"]
        )
        raw = np.asarray([
            [angles[key] for key in ANGLE_KEYS]
            for angles in (compute_joint_angles(joints) for joints in track["joints"])
        ])
        track["angles_raw"] = raw
        track["angles"] = median_filter(raw, angle_radius)
        track["rest_joints"] = load_rest_joints(mano, model_dir, track)
        track["kinematic_axes"] = build_kinematic_axes(track["rest_joints"])
        track["kinematic_raw"] = extract_kinematic_sequence(
            track["hand_pose_axis_angle"], track["kinematic_axes"], "Right"
        )
        track["kinematic"] = median_filter(track["kinematic_raw"], angle_radius)
        end_poses = [
            compute_hand_end_effector_pose(joints, track["handedness"])
            for joints in track["joints"]
        ]
        track["end_effector_position_m"] = np.stack([
            pose["position_m"] for pose in end_poses
        ])
        track["trajectory_position_m"] = track["end_effector_position_m"].copy()
        track["trajectory_position_m"][~track["render_valid"]] = np.nan
        track["end_effector_rotation_matrix"] = np.stack([
            pose["rotation_matrix"] for pose in end_poses
        ])
        track["end_effector_rpy_rad"] = np.stack([pose["rpy_rad"] for pose in end_poses])
        track["end_effector_quaternion_xyzw"] = np.stack([
            pose["quaternion_xyzw"] for pose in end_poses
        ])
        origin_rotation = track["end_effector_rotation_matrix"][0]
        (
            track["end_effector_delta_camera_m"],
            track["end_effector_delta_hand0_m"],
        ) = trajectory_displacements(
            track["end_effector_position_m"], origin_rotation
        )
        tracks.append(track)
    if not tracks:
        raise FileNotFoundError(f"no track_*.npz files in {directory}")
    return tracks


def load_frame_rows(path: Path, start_pair: int, max_pairs: int) -> list[dict[str, int]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = [
            {"pair_index": int(row["pair_index"]), "left_index": int(row["left_index"])}
            for row in csv.DictReader(stream)
        ]
    rows = [row for row in rows if row["pair_index"] >= start_pair]
    if max_pairs > 0:
        rows = rows[:max_pairs]
    if not rows:
        raise RuntimeError("no stereo frame rows selected")
    if any(b["left_index"] <= a["left_index"] for a, b in zip(rows, rows[1:])):
        raise RuntimeError("left frame indices must be strictly increasing")
    return rows


def read_frame_at(capture: cv2.VideoCapture, target: int, state: list[int]) -> np.ndarray:
    frame = None
    while state[0] <= target:
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError(f"video ended before left frame {target}")
        state[0] += 1
    if frame is None:
        raise RuntimeError(f"failed to decode left frame {target}")
    return frame


def project_fisheye(points: np.ndarray, camera_matrix: np.ndarray, distortion: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    projected, _ = cv2.fisheye.projectPoints(
        points.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), camera_matrix, distortion
    )
    return projected[:, 0]


def project_camera_model(points: np.ndarray, camera) -> np.ndarray:
    """Project points through a unified CameraCalibration object."""
    from camera_models import project_points
    pixels, _ = project_points(camera, points)
    return pixels


def project_for_render(
    points: np.ndarray,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    camera=None,
) -> np.ndarray:
    """Project renderer geometry through either legacy KB or unified KB/DS calibration."""
    if camera is None:
        return project_fisheye(points, camera_matrix, distortion)
    return project_camera_model(points, camera)


def raw_rectified_projection_residual(
    points: np.ndarray,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    rotation: np.ndarray,
    projection: np.ndarray,
) -> np.ndarray:
    raw = project_fisheye(points, camera_matrix, distortion)
    via_undistort = cv2.fisheye.undistortPoints(
        raw.reshape(-1, 1, 2), camera_matrix, distortion, R=rotation, P=projection
    )[:, 0]
    rectified = (rotation @ np.asarray(points, dtype=np.float64).T).T
    homogeneous = np.column_stack((rectified, np.ones(len(rectified)))) @ projection.T
    direct = homogeneous[:, :2] / homogeneous[:, 2:]
    return np.linalg.norm(via_undistort - direct, axis=1)


def shade_color(color: tuple[int, int, int], scale: float) -> tuple[int, int, int]:
    return tuple(int(np.clip(channel * scale, 0, 255)) for channel in color)


def draw_end_effector_trajectory(
    image: np.ndarray,
    positions_m: np.ndarray,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    color: tuple[int, int, int],
    maximum_frames: int = 120,
    maximum_jump_m: float = 0.12,
    camera=None,
) -> None:
    positions = np.asarray(positions_m, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3 or maximum_frames <= 0:
        return
    positions = positions[-maximum_frames:]
    finite = np.isfinite(positions).all(axis=1) & (positions[:, 2] > 0.03)
    if not np.any(finite):
        return
    safe_positions = positions.copy()
    safe_positions[~finite] = np.asarray((0.0, 0.0, 1.0))
    pixels = project_for_render(safe_positions, camera_matrix, distortion, camera)
    count = len(positions)
    for index in range(1, count):
        if not (finite[index - 1] and finite[index]):
            continue
        if np.linalg.norm(positions[index] - positions[index - 1]) > maximum_jump_m:
            continue
        progress = index / max(count - 1, 1)
        segment_color = shade_color(color, 0.22 + 0.78 * progress)
        thickness = max(1, int(round(1.0 + 3.0 * progress)))
        cv2.line(
            image, tuple(np.rint(pixels[index - 1]).astype(int)),
            tuple(np.rint(pixels[index]).astype(int)), segment_color,
            thickness, cv2.LINE_AA,
        )
    current_index = int(np.flatnonzero(finite)[-1])
    current_pixel = tuple(np.rint(pixels[current_index]).astype(int))
    cv2.circle(image, current_pixel, 8, color, -1, cv2.LINE_AA)
    position_mm = positions[current_index] * 1000.0
    text = f"x {position_mm[0]:+.0f}  y {position_mm[1]:+.0f}  z {position_mm[2]:+.0f} mm"
    (text_width, text_height), _ = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1
    )
    x = int(np.clip(current_pixel[0] + 12, 4, max(image.shape[1] - text_width - 12, 4)))
    y = int(np.clip(current_pixel[1] - 12, text_height + 10, image.shape[0] - 8))
    overlay = image.copy()
    cv2.rectangle(
        overlay, (x - 5, y - text_height - 6), (x + text_width + 6, y + 5),
        (12, 15, 20), -1, cv2.LINE_AA,
    )
    cv2.addWeighted(overlay, 0.78, image, 0.22, 0.0, image)
    cv2.putText(
        image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
        (245, 245, 245), 1, cv2.LINE_AA,
    )


def draw_mesh(
    image: np.ndarray,
    vertices: np.ndarray,
    joints: np.ndarray,
    faces: np.ndarray,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    color: tuple[int, int, int],
    alpha: float,
    label: str,
    handedness: str | None = None,
    camera=None,
) -> None:
    height, width = image.shape[:2]
    vertex_px = project_for_render(vertices, camera_matrix, distortion, camera)
    joint_px = project_for_render(joints, camera_matrix, distortion, camera)
    face_vertices = vertices[faces]
    normals = np.cross(face_vertices[:, 1] - face_vertices[:, 0], face_vertices[:, 2] - face_vertices[:, 0])
    normal_norm = np.linalg.norm(normals, axis=1)
    lighting = np.divide(
        np.abs(normals[:, 2]), normal_norm, out=np.zeros_like(normal_norm), where=normal_norm > 1e-9
    )
    buckets = np.clip((lighting * 4.999).astype(np.int32), 0, 4)
    on_camera = np.all(face_vertices[:, :, 2] > 0.03, axis=1)
    polygons = np.rint(vertex_px[faces]).astype(np.int32)
    intersects = (
        (polygons[:, :, 0].max(axis=1) >= 0)
        & (polygons[:, :, 0].min(axis=1) < width)
        & (polygons[:, :, 1].max(axis=1) >= 0)
        & (polygons[:, :, 1].min(axis=1) < height)
    )
    keep = on_camera & intersects
    overlay = image.copy()
    visible_polygons = []
    for bucket in range(5):
        selected = polygons[keep & (buckets == bucket)]
        if len(selected):
            cv2.fillPoly(overlay, list(selected), shade_color(color, 0.66 + bucket * 0.105), cv2.LINE_AA)
            visible_polygons.extend(selected[::2])
    cv2.addWeighted(overlay, alpha, image, 1.0 - alpha, 0.0, image)
    if visible_polygons:
        cv2.polylines(image, visible_polygons, True, shade_color(color, 0.60), 1, cv2.LINE_AA)
    for start, end in HAND_CONNECTIONS:
        a = tuple(np.rint(joint_px[start]).astype(int))
        b = tuple(np.rint(joint_px[end]).astype(int))
        cv2.line(image, a, b, shade_color(color, 1.15), 2, cv2.LINE_AA)
    for point in joint_px:
        cv2.circle(image, tuple(np.rint(point).astype(int)), 3, (245, 245, 245), -1, cv2.LINE_AA)
    wrist = np.rint(joint_px[0]).astype(int)
    cv2.putText(
        image, label, (int(wrist[0]) + 12, int(wrist[1]) - 12),
        cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2, cv2.LINE_AA,
    )
    if handedness in ("Left", "Right"):
        end_pose = compute_hand_end_effector_pose(joints, handedness)
        origin = end_pose["position_m"]
        rotation = end_pose["rotation_matrix"]
        axis_length = 0.045
        axis_points = np.vstack((origin, origin[None, :] + rotation.T * axis_length))
        axis_pixels = project_for_render(axis_points, camera_matrix, distortion, camera)
        origin_pixel = tuple(np.rint(axis_pixels[0]).astype(int))
        for axis_index, (axis_name, axis_color) in enumerate(
            (("X", (0, 0, 255)), ("Y", (0, 220, 0)), ("Z", (255, 80, 40)))
        ):
            endpoint = tuple(np.rint(axis_pixels[axis_index + 1]).astype(int))
            cv2.arrowedLine(image, origin_pixel, endpoint, axis_color, 3, cv2.LINE_AA, tipLength=0.18)
            cv2.putText(image, axis_name, endpoint, cv2.FONT_HERSHEY_SIMPLEX,
                        0.48, axis_color, 2, cv2.LINE_AA)


def draw_bar(
    panel: np.ndarray,
    origin: tuple[int, int],
    width: int,
    value: float,
    color: tuple[int, int, int],
    maximum: float = 120.0,
) -> None:
    x, y = origin
    cv2.rectangle(panel, (x, y), (x + width, y + 8), (57, 65, 78), -1, cv2.LINE_AA)
    if np.isfinite(value):
        filled = int(round(width * np.clip(value / maximum, 0.0, 1.0)))
        cv2.rectangle(panel, (x, y), (x + filled, y + 8), color, -1, cv2.LINE_AA)


def draw_signed_bar(
    panel: np.ndarray,
    origin: tuple[int, int],
    width: int,
    value: float,
    color: tuple[int, int, int],
    maximum_abs: float,
) -> None:
    x, y = origin
    centre = x + width // 2
    cv2.rectangle(panel, (x, y), (x + width, y + 6), (57, 65, 78), -1, cv2.LINE_AA)
    cv2.line(panel, (centre, y - 1), (centre, y + 7), (116, 124, 137), 1, cv2.LINE_AA)
    if not np.isfinite(value) or maximum_abs <= 0:
        return
    delta = int(round((width // 2) * np.clip(value / maximum_abs, -1.0, 1.0)))
    cv2.rectangle(
        panel, (min(centre, centre + delta), y), (max(centre, centre + delta), y + 6),
        color, -1, cv2.LINE_AA,
    )


def draw_hand_card(
    panel: np.ndarray,
    rect: tuple[int, int, int, int],
    track: dict,
    frame_index: int | None,
) -> None:
    x, y, width, height = rect
    handedness = track["handedness"]
    color = TRACK_COLORS.get(handedness, (120, 220, 120))
    cv2.rectangle(panel, (x, y), (x + width, y + height), (29, 35, 45), -1, cv2.LINE_AA)
    cv2.rectangle(panel, (x, y), (x + width, y + height), (70, 80, 95), 1, cv2.LINE_AA)
    title = f"{handedness.upper()} HAND   21-DOF   T{track['track_id']}"
    cv2.putText(panel, title, (x + 16, y + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2, cv2.LINE_AA)
    cv2.putText(
        panel, "MANO mean-relative local rotation (rad)", (x + 16, y + 47),
        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (170, 178, 190), 1, cv2.LINE_AA,
    )
    if frame_index is None:
        cv2.putText(
            panel, "NOT VISIBLE", (x + width // 2 - 82, y + height // 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.72, (105, 112, 123), 2, cv2.LINE_AA,
        )
        return

    angles = {
        key: track["kinematic"][frame_index, index] for index, key in enumerate(KINEMATIC_KEYS)
    }
    row_height = 50
    row_top = y + 55
    label_width = 66
    column_gap = 5
    usable = width - 32 - label_width
    for row, (finger, entries) in enumerate(KINEMATIC_LAYOUT.items()):
        top = row_top + row * row_height
        cv2.putText(
            panel, finger.upper(), (x + 16, top + 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (224, 227, 232), 1, cv2.LINE_AA,
        )
        column_width = (usable - (len(entries) - 1) * column_gap) // len(entries)
        for column, (short_name, key) in enumerate(entries):
            value = angles[key]
            bar_x = x + 16 + label_width + column * (column_width + column_gap)
            cv2.putText(
                panel, short_name, (bar_x, top + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.27, (137, 149, 166), 1, cv2.LINE_AA,
            )
            value_text = f"{value:+.2f}" if np.isfinite(value) else " nan"
            cv2.putText(
                panel, value_text, (bar_x, top + 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (220, 224, 230), 1, cv2.LINE_AA,
            )
            draw_signed_bar(
                panel, (bar_x, top + 33), column_width, value, color, KINEMATIC_LIMITS_RAD[key]
            )
    cv2.putText(
        panel, "CF/CA/OP: thumb CMC   MF/MA/PF/DF: finger joints", (x + 16, y + height - 29),
        cv2.FONT_HERSHEY_SIMPLEX, 0.31, (145, 155, 170), 1, cv2.LINE_AA,
    )
    position = track["end_effector_position_m"][frame_index]
    rpy_deg = np.degrees(track["end_effector_rpy_rad"][frame_index])
    pose_text = (
        f"6D P[m] {position[0]:+.3f} {position[1]:+.3f} {position[2]:+.3f}   "
        f"RPY[deg] {rpy_deg[0]:+.1f} {rpy_deg[1]:+.1f} {rpy_deg[2]:+.1f}"
    )
    cv2.putText(
        panel, pose_text, (x + 16, y + height - 11),
        cv2.FONT_HERSHEY_SIMPLEX, 0.29, (185, 205, 190), 1, cv2.LINE_AA,
    )


def draw_preview_mesh(
    panel: np.ndarray,
    rect: tuple[int, int, int, int],
    vertices: np.ndarray,
    joints: np.ndarray,
    faces: np.ndarray,
    color: tuple[int, int, int],
) -> None:
    x, y, width, height = rect
    centred_vertices = np.asarray(vertices, dtype=np.float64) - joints[0]
    centred_joints = np.asarray(joints, dtype=np.float64) - joints[0]
    xy = centred_vertices[:, :2]
    extent = np.ptp(xy, axis=0)
    if np.any(extent < 1e-8):
        return
    scale = 0.82 * min(width / extent[0], height / extent[1])
    centre_xy = 0.5 * (xy.min(axis=0) + xy.max(axis=0))
    vertex_px = (xy - centre_xy) * scale + np.asarray([x + width / 2, y + height / 2])
    joint_px = (centred_joints[:, :2] - centre_xy) * scale + np.asarray(
        [x + width / 2, y + height / 2]
    )
    polygons = np.rint(vertex_px[faces]).astype(np.int32)
    face_vertices = centred_vertices[faces]
    normals = np.cross(face_vertices[:, 1] - face_vertices[:, 0], face_vertices[:, 2] - face_vertices[:, 0])
    norm = np.linalg.norm(normals, axis=1)
    lighting = np.divide(np.abs(normals[:, 2]), norm, out=np.zeros_like(norm), where=norm > 1e-9)
    buckets = np.clip((lighting * 4.999).astype(np.int32), 0, 4)
    for bucket in range(5):
        selected = polygons[buckets == bucket]
        if len(selected):
            cv2.fillPoly(panel, list(selected), shade_color(color, 0.58 + bucket * 0.11), cv2.LINE_AA)
    cv2.polylines(panel, list(polygons[::3]), True, shade_color(color, 0.50), 1, cv2.LINE_AA)
    for start, end in HAND_CONNECTIONS:
        cv2.line(
            panel, tuple(np.rint(joint_px[start]).astype(int)), tuple(np.rint(joint_px[end]).astype(int)),
            (235, 238, 243), 1, cv2.LINE_AA,
        )


def draw_preview_card(
    panel: np.ndarray,
    rect: tuple[int, int, int, int],
    tracks: list[dict],
    frame_indices: dict[int, int],
) -> None:
    x, y, width, height = rect
    cv2.rectangle(panel, (x, y), (x + width, y + height), (25, 30, 39), -1, cv2.LINE_AA)
    cv2.rectangle(panel, (x, y), (x + width, y + height), (70, 80, 95), 1, cv2.LINE_AA)
    cv2.putText(
        panel, "MANO 3D PREVIEW", (x + 15, y + 25),
        cv2.FONT_HERSHEY_SIMPLEX, 0.54, (229, 233, 239), 1, cv2.LINE_AA,
    )
    cell_width = (width - 24) // 2
    for index, track in enumerate(tracks[:2]):
        cell_x = x + 8 + index * (cell_width + 8)
        color = TRACK_COLORS.get(track["handedness"], (120, 220, 120))
        cv2.putText(
            panel, track["handedness"].upper(), (cell_x + 5, y + 46),
            cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA,
        )
        frame_index = frame_indices.get(track["track_id"])
        if frame_index is None:
            cv2.putText(
                panel, "NOT VISIBLE", (cell_x + 48, y + height // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (100, 108, 120), 1, cv2.LINE_AA,
            )
            continue
        draw_preview_mesh(
            panel, (cell_x + 4, y + 52, cell_width - 8, height - 60),
            track["vertices"][frame_index], track["joints"][frame_index],
            track["faces"].astype(np.int32), color,
        )


def compose_canvas(
    frame: np.ndarray,
    tracks: list[dict],
    track_frame_indices: dict[int, int],
    output_size: tuple[int, int],
    panel_width: int,
    pair_index: int,
    left_index: int,
) -> np.ndarray:
    width, height = output_size
    view_width = width - panel_width
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    scale = min(view_width / frame.shape[1], height / frame.shape[0])
    resized_size = (int(round(frame.shape[1] * scale)), int(round(frame.shape[0] * scale)))
    resized = cv2.resize(frame, resized_size, interpolation=cv2.INTER_AREA)
    offset_x = (view_width - resized_size[0]) // 2
    offset_y = (height - resized_size[1]) // 2
    canvas[offset_y:offset_y + resized_size[1], offset_x:offset_x + resized_size[0]] = resized
    cv2.rectangle(canvas, (0, 0), (view_width - 1, height - 1), (75, 82, 92), 1)

    title_layer = canvas.copy()
    cv2.rectangle(title_layer, (18, 16), (580, 72), (9, 13, 18), -1, cv2.LINE_AA)
    cv2.addWeighted(title_layer, 0.72, canvas, 0.28, 0.0, canvas)
    cv2.putText(
        canvas, "EGO MANO 21-DOF OVERLAY", (35, 47),
        cv2.FONT_HERSHEY_SIMPLEX, 0.80, (242, 244, 247), 2, cv2.LINE_AA,
    )
    cv2.putText(
        canvas, f"pair {pair_index:03d}   left frame {left_index:03d}", (35, 66),
        cv2.FONT_HERSHEY_SIMPLEX, 0.43, (174, 182, 194), 1, cv2.LINE_AA,
    )

    panel = canvas[:, view_width:]
    panel[:] = (16, 20, 27)
    cv2.putText(
        panel, "DUAL-HAND KINEMATICS", (18, 34),
        cv2.FONT_HERSHEY_SIMPLEX, 0.70, (238, 241, 246), 2, cv2.LINE_AA,
    )
    cv2.putText(
        panel, "MANO: ON | 21 DOF RAD | metric fisheye overlay", (18, 55),
        cv2.FONT_HERSHEY_SIMPLEX, 0.39, (147, 158, 174), 1, cv2.LINE_AA,
    )
    card_gap = 8
    margin = 10
    top = 68
    preview_height = 260
    card_area_height = height - top - margin - preview_height - 2 * card_gap
    card_height = card_area_height // 2
    for card_index, track in enumerate(tracks[:2]):
        frame_index = track_frame_indices.get(track["track_id"])
        draw_hand_card(
            panel,
            (margin, top + card_index * (card_height + card_gap), panel_width - 2 * margin, card_height),
            track,
            frame_index,
        )
    preview_y = top + 2 * card_height + 2 * card_gap
    draw_preview_card(
        panel, (margin, preview_y, panel_width - 2 * margin, preview_height - margin),
        tracks, track_frame_indices,
    )
    return canvas


def write_montage(frames: list[np.ndarray], path: Path) -> None:
    if not frames:
        return
    thumbs = [cv2.resize(frame, (640, 360), interpolation=cv2.INTER_AREA) for frame in frames]
    while len(thumbs) < 6:
        thumbs.append(thumbs[-1].copy())
    montage = cv2.vconcat([cv2.hconcat(thumbs[:3]), cv2.hconcat(thumbs[3:6])])
    if not cv2.imwrite(str(path), montage):
        raise RuntimeError(f"cannot write {path}")


def main() -> int:
    args = parse_args()
    if args.width <= 0 or args.height <= 0 or not 250 <= args.panel_width < args.width:
        raise ValueError("invalid output dimensions/panel width")
    if (
        not 0.0 < args.mesh_alpha <= 1.0 or args.angle_radius < 0
        or args.trajectory_length < 0 or args.trajectory_max_jump_m <= 0.0
    ):
        raise ValueError("mesh alpha must be in (0,1] and angle radius non-negative")
    fit_dir = args.mano_fit.resolve()
    mano_source = args.mano_source.resolve()
    model_dir = args.model_dir.resolve()
    frames_path = args.stereo_frames.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    normalized_dataset = None
    unified_camera = None
    if args.normalized_dataset is not None:
        if args.rectified_dataset is None:
            raise ValueError("--rectified-dataset is required with --normalized-dataset")
        from ego_data.dataset import NormalizedStereoDataset, RectifiedStereoDataset
        normalized_dataset = NormalizedStereoDataset(args.normalized_dataset)
        rectified_dataset = RectifiedStereoDataset(args.rectified_dataset)
        normalized_sha = normalized_dataset.manifest.get("source", {}).get("sha256")
        rectified_sha = rectified_dataset.manifest.get("source_mcap_sha256")
        if rectified_sha is not None and rectified_sha != normalized_sha:
            raise ValueError("rectified and normalized datasets originate from different MCAP files")
        unified_camera = normalized_dataset.left
        expected_size = unified_camera.image_size
        rectification = rectified_dataset.rectification
        camera_matrix = unified_camera.K
        distortion = unified_camera.distortion
        source_description = str(normalized_dataset.root)
        source_video_description = str(normalized_dataset.video_path(normalized_dataset.left_id))
        normalized_iterator = iter(normalized_dataset)
        source_fps = 0.0
    else:
        if args.rectified_dataset is not None:
            raise ValueError("--rectified-dataset is only valid with --normalized-dataset")
        session = args.session.resolve()
        calibration_path = unique_file(session, "_calibration_camera.yaml")
        left_video_path = unique_file(session, "_camera_left.mp4")
        with calibration_path.open("r", encoding="utf-8") as stream:
            calibration = yaml.safe_load(stream)
        camera = calibration["cameras"][0]
        camera_matrix, distortion = camera_matrices(camera)
        expected_size = (int(camera["image_width"]), int(camera["image_height"]))
        rectification = create_stereo_rectification(calibration_path, args.balance)
        source_description = str(session)
        source_video_description = str(left_video_path)
    if not (mano_source / "mano" / "model.py").is_file():
        raise FileNotFoundError(f"invalid MANO source: {mano_source}")
    if not (model_dir / "MANO_RIGHT.pkl").is_file():
        raise FileNotFoundError(f"missing licensed MANO_RIGHT.pkl in {model_dir}")
    tracks = load_tracks(fit_dir, args.angle_radius, mano_source, model_dir)
    frame_rows = load_frame_rows(frames_path, args.start_pair, args.max_pairs)

    validation_points = tracks[0]["vertices"][0].astype(np.float64)
    if unified_camera is None:
        projection_residual = raw_rectified_projection_residual(
            validation_points, camera_matrix, distortion, rectification["r1"], rectification["p1"]
        )
    else:
        raw = project_camera_model(validation_points, unified_camera)
        # The DS renderer is validated by model unit tests.  Here verify finite projection
        # for the fitted vertices rather than applying OpenCV's KB-only undistortPoints.
        projection_residual = np.zeros(len(raw), dtype=np.float64)
        if not np.all(np.isfinite(raw)):
            raise RuntimeError("camera model produced invalid MANO projections")
    if float(np.max(projection_residual)) > 1e-6:
        raise RuntimeError("raw fisheye projection disagrees with rectified projection")

    capture = None
    if normalized_dataset is None:
        capture = cv2.VideoCapture(str(left_video_path))
        if not capture.isOpened():
            raise RuntimeError(f"cannot open {left_video_path}")
        source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    else:
        timestamps = [int(row["left_timestamp_ns"]) for row in normalized_dataset.pairs]
        source_fps = 1e9 / float(np.median(np.diff(timestamps))) if len(timestamps) > 1 else 30.0
    fps = source_fps if source_fps > 0 else 30.0
    video_path = output / "mano_overlay_21dof.mp4"
    writer = None
    if not args.no_video:
        writer = cv2.VideoWriter(
            str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (args.width, args.height)
        )
        if not writer.isOpened():
            raise RuntimeError(f"cannot create {video_path}")

    geometric_csv_path = output / "mano_joint_angles.csv"
    kinematic_csv_path = output / "mano_joint_angles_21dof.csv"
    pose_csv_path = output / "mano_pose_axis_angle.csv"
    pose6d_csv_path = output / "hand_end_effector_6d.csv"
    base_fields = ["pair_index", "left_frame_index", "track_id", "handedness"]
    geometric_fields = base_fields + [f"{key}_raw" for key in ANGLE_KEYS] + list(ANGLE_KEYS)
    kinematic_fields = list(base_fields)
    for key in KINEMATIC_KEYS:
        kinematic_fields.extend([f"{key}_raw", key, key.replace("_rad", "_deg")])
    pose_fields = list(base_fields) + [
        f"{joint}_{axis}_rad" for joint in MANO_POSE_NAMES for axis in ("x", "y", "z")
    ]
    pose6d_fields = list(base_fields) + [
        "x_m", "y_m", "z_m",
        "roll_rad", "pitch_rad", "yaw_rad",
        "roll_deg", "pitch_deg", "yaw_deg",
        "dx_camera_m", "dy_camera_m", "dz_camera_m",
        "dx_hand0_m", "dy_hand0_m", "dz_hand0_m",
        "qx", "qy", "qz", "qw",
    ] + [f"r{row}{column}" for row in range(3) for column in range(3)]
    preview_ordinals = set(np.linspace(0, len(frame_rows) - 1, min(6, len(frame_rows)), dtype=int).tolist())
    previews: list[np.ndarray] = []
    frame_state = [0]
    processed = 0
    visible_instances = 0
    start_time = time.perf_counter()

    with geometric_csv_path.open("w", encoding="utf-8", newline="") as geometric_stream, \
         kinematic_csv_path.open("w", encoding="utf-8", newline="") as kinematic_stream, \
         pose_csv_path.open("w", encoding="utf-8", newline="") as pose_stream, \
         pose6d_csv_path.open("w", encoding="utf-8", newline="") as pose6d_stream:
        geometric_writer = csv.DictWriter(geometric_stream, fieldnames=geometric_fields)
        kinematic_writer = csv.DictWriter(kinematic_stream, fieldnames=kinematic_fields)
        pose_writer = csv.DictWriter(pose_stream, fieldnames=pose_fields)
        pose6d_writer = csv.DictWriter(pose6d_stream, fieldnames=pose6d_fields)
        geometric_writer.writeheader()
        kinematic_writer.writeheader()
        pose_writer.writeheader()
        pose6d_writer.writeheader()
        for ordinal, row in enumerate(frame_rows):
            pair_index = row["pair_index"]
            left_index = row["left_index"]
            if normalized_dataset is None:
                frame = read_frame_at(capture, left_index, frame_state)
            else:
                while True:
                    row, (frame, _) = next(normalized_iterator)
                    decoded_left_index = int(row["left_frame_index"])
                    if decoded_left_index >= left_index:
                        break
                if decoded_left_index != left_index:
                    raise RuntimeError("normalized video and requested left frame are inconsistent")
            if frame.shape[1::-1] != expected_size:
                raise RuntimeError(f"decoded size {frame.shape[1::-1]} differs from calibration {expected_size}")

            visible = []
            frame_indices: dict[int, int] = {}
            for track in tracks:
                track_frame = track["lookup"].get(pair_index)
                if track_frame is None or not track["render_valid"][track_frame]:
                    continue
                frame_indices[track["track_id"]] = track_frame
                visible.append((float(np.median(track["vertices"][track_frame, :, 2])), track, track_frame))
                base_row = {
                    "pair_index": pair_index,
                    "left_frame_index": left_index,
                    "track_id": track["track_id"],
                    "handedness": track["handedness"],
                }
                geometric_row = dict(base_row)
                for index, key in enumerate(ANGLE_KEYS):
                    geometric_row[f"{key}_raw"] = f"{track['angles_raw'][track_frame, index]:.6f}"
                    geometric_row[key] = f"{track['angles'][track_frame, index]:.6f}"
                geometric_writer.writerow(geometric_row)

                kinematic_row = dict(base_row)
                for index, key in enumerate(KINEMATIC_KEYS):
                    raw_value = track["kinematic_raw"][track_frame, index]
                    value = track["kinematic"][track_frame, index]
                    kinematic_row[f"{key}_raw"] = f"{raw_value:.9f}"
                    kinematic_row[key] = f"{value:.9f}"
                    kinematic_row[key.replace("_rad", "_deg")] = f"{np.degrees(value):.6f}"
                kinematic_writer.writerow(kinematic_row)

                pose_row = dict(base_row)
                pose = track["hand_pose_axis_angle"][track_frame]
                for joint_index, joint_name in enumerate(MANO_POSE_NAMES):
                    for axis_index, axis_name in enumerate(("x", "y", "z")):
                        pose_row[f"{joint_name}_{axis_name}_rad"] = f"{pose[joint_index, axis_index]:.9f}"
                pose_writer.writerow(pose_row)

                position = track["end_effector_position_m"][track_frame]
                rpy = track["end_effector_rpy_rad"][track_frame]
                quaternion = track["end_effector_quaternion_xyzw"][track_frame]
                rotation_matrix = track["end_effector_rotation_matrix"][track_frame]
                delta_camera = track["end_effector_delta_camera_m"][track_frame]
                delta_hand0 = track["end_effector_delta_hand0_m"][track_frame]
                pose6d_row = dict(base_row)
                pose6d_row.update({
                    "x_m": f"{position[0]:.9f}", "y_m": f"{position[1]:.9f}",
                    "z_m": f"{position[2]:.9f}",
                    "roll_rad": f"{rpy[0]:.9f}", "pitch_rad": f"{rpy[1]:.9f}",
                    "yaw_rad": f"{rpy[2]:.9f}",
                    "roll_deg": f"{np.degrees(rpy[0]):.6f}",
                    "pitch_deg": f"{np.degrees(rpy[1]):.6f}",
                    "yaw_deg": f"{np.degrees(rpy[2]):.6f}",
                    "dx_camera_m": f"{delta_camera[0]:.9f}",
                    "dy_camera_m": f"{delta_camera[1]:.9f}",
                    "dz_camera_m": f"{delta_camera[2]:.9f}",
                    "dx_hand0_m": f"{delta_hand0[0]:.9f}",
                    "dy_hand0_m": f"{delta_hand0[1]:.9f}",
                    "dz_hand0_m": f"{delta_hand0[2]:.9f}",
                    "qx": f"{quaternion[0]:.9f}", "qy": f"{quaternion[1]:.9f}",
                    "qz": f"{quaternion[2]:.9f}", "qw": f"{quaternion[3]:.9f}",
                })
                pose6d_row.update({
                    f"r{row_index}{column_index}": f"{rotation_matrix[row_index, column_index]:.9f}"
                    for row_index in range(3) for column_index in range(3)
                })
                pose6d_writer.writerow(pose6d_row)

            for _, track, track_frame in sorted(visible, reverse=True, key=lambda item: item[0]):
                color = TRACK_COLORS.get(track["handedness"], (120, 220, 120))
                draw_end_effector_trajectory(
                    frame,
                    track["trajectory_position_m"][:track_frame + 1],
                    camera_matrix,
                    distortion,
                    color,
                    args.trajectory_length,
                    args.trajectory_max_jump_m,
                    camera=unified_camera,
                )
                draw_mesh(
                    frame,
                    track["vertices"][track_frame],
                    track["joints"][track_frame],
                    track["faces"].astype(np.int32),
                    camera_matrix,
                    distortion,
                    color,
                    args.mesh_alpha,
                    f"{track['handedness']} T{track['track_id']}",
                    track["handedness"],
                    camera=unified_camera,
                )
            canvas = compose_canvas(
                frame, tracks, frame_indices, (args.width, args.height), args.panel_width,
                pair_index, left_index,
            )
            if writer is not None:
                writer.write(canvas)
            if ordinal in preview_ordinals:
                previews.append(canvas.copy())
            visible_instances += len(visible)
            processed += 1

    if capture is not None:
        capture.release()
    if writer is not None:
        writer.release()
    elapsed = time.perf_counter() - start_time
    preview_path = output / "preview_montage.jpg"
    write_montage(previews, preview_path)

    angle_summary = {}
    for track in tracks:
        values = track["angles"]
        kinematic_values = track["kinematic"]
        bend_values = values[:, :15]
        spread_values = values[:, 15:]
        angle_summary[str(track["track_id"])] = {
            "handedness": track["handedness"],
            "frames": int(len(track["pair_indices"])),
            "bend_observations": int(np.count_nonzero(np.isfinite(bend_values))),
            "bend_over_130_deg": int(np.count_nonzero(bend_values > 130.0)),
            "bend_over_130_rate": float(np.nanmean(bend_values > 130.0)),
            "bend_deg_median": float(np.nanmedian(bend_values)),
            "bend_deg_p95": float(np.nanpercentile(bend_values, 95)),
            "spread_abs_deg_median": float(np.nanmedian(np.abs(spread_values))),
            "spread_deg_min": float(np.nanmin(spread_values)),
            "spread_deg_max": float(np.nanmax(spread_values)),
            "kinematic_dof_count": len(KINEMATIC_KEYS),
            "kinematic_abs_rad_median": float(np.nanmedian(np.abs(kinematic_values))),
            "kinematic_abs_rad_p95": float(np.nanpercentile(np.abs(kinematic_values), 95)),
            "kinematic_rad_range_by_dof": {
                key: [
                    float(np.nanmin(kinematic_values[:, index])),
                    float(np.nanmax(kinematic_values[:, index])),
                ]
                for index, key in enumerate(KINEMATIC_KEYS)
            },
        }
    summary = {
        "stage": "mano_camera_overlay_21dof_and_dual_3d_preview",
        "source": source_description,
        "session": source_description if normalized_dataset is None else None,
        "mano_fit": str(fit_dir),
        "mano_source": str(mano_source),
        "mano_model_dir": str(model_dir),
        "source_video": source_video_description,
        "source_fps": fps,
        "source_size": list(expected_size),
        "output_size": [args.width, args.height],
        "processed_pairs": processed,
        "visible_hand_instances": visible_instances,
        "track_count": len(tracks),
        "projection_validation_max_px": float(np.max(projection_residual)),
        "projection_validation_median_px": float(np.median(projection_residual)),
        "mesh_alpha": args.mesh_alpha,
        "angle_smoothing_radius_frames": args.angle_radius,
        "trajectory_length_frames": args.trajectory_length,
        "trajectory_max_jump_m": args.trajectory_max_jump_m,
        "processing_fps": processed / elapsed if elapsed > 0 else 0.0,
        "kinematic_dof_count_per_hand": len(KINEMATIC_KEYS),
        "angle_definitions": {
            "geometric": (
                "Bend is the angle between consecutive 3D bone directions; spread is measured in the fitted palm plane."
            ),
            "kinematic_21dof": (
                "MANO mean-relative local rotation vectors projected on fitted flat-hand flexion, palm-normal "
                "abduction, and thumb metacarpal opposition axes. Five thumb plus four per other finger."
            ),
            "raw_mano_pose": "All 15 x 3 mean-relative MANO hand-pose axis-angle components in radians.",
        },
        "hand_end_effector_6d_definition": {
            "reference_frame": "left camera optical frame: +X right, +Y down, +Z forward",
            "origin": "mean of wrist and index/middle/ring/pinky MCP joints",
            "local_y": "wrist toward middle MCP",
            "local_z": "handedness-normalized palm normal",
            "local_x": "completes the right-handed frame",
            "rpy_convention": "ZYX: R = Rz(yaw) Ry(pitch) Rx(roll)",
            "quaternion_order": "x,y,z,w",
        },
        "tracks": angle_summary,
        "outputs": {
            "video": video_path.name if writer is not None else None,
            "geometric_angles_csv": geometric_csv_path.name,
            "kinematic_21dof_csv": kinematic_csv_path.name,
            "mano_pose_axis_angle_csv": pose_csv_path.name,
            "hand_end_effector_6d_csv": pose6d_csv_path.name,
            "preview_montage": preview_path.name,
        },
    }
    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("EGO MANO 21-DOF camera overlay and dual-hand 3D dashboard")
    print(f"Processed pairs: {processed}")
    print(f"Visible hand instances: {visible_instances}")
    print(f"Raw/rectified projection agreement max: {summary['projection_validation_max_px']:.3e} px")
    print(f"Rendering speed: {summary['processing_fps']:.2f} fps")
    if writer is not None:
        print(f"Overlay video: {video_path}")
    print(f"Geometric angles CSV: {geometric_csv_path}")
    print(f"21-DOF kinematics CSV: {kinematic_csv_path}")
    print(f"Raw MANO pose CSV: {pose_csv_path}")
    print(f"Hand end-effector 6D CSV: {pose6d_csv_path}")
    print(f"Preview montage: {preview_path}")
    print(f"Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
