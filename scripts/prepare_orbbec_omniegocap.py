#!/usr/bin/env python3
"""Join cam_0 Basalt poses and calibrated multiview hands for OmniEgoCap.

Export intermediate camera/wrist files unless supplied with a calibrated head
transform or an explicitly requested IMU/neutral-pose approximation and floor.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


PALM = [0, 5, 9, 17]  # wrist, index MCP, middle MCP, little MCP


def rigid(T: np.ndarray) -> bool:
    return (T.shape == (4, 4) and np.isfinite(T).all()
            and np.allclose(T[3], [0, 0, 0, 1], atol=1e-6)
            and np.allclose(T[:3, :3].T @ T[:3, :3], np.eye(3), atol=1e-5)
            and np.isclose(np.linalg.det(T[:3, :3]), 1, atol=1e-5))


def wrist_from_joints(joints: np.ndarray, side: int) -> np.ndarray | None:
    """Geometric SMPL-H-compatible palm frame, not a fitted MANO rotation.

    y points out of the back of the hand. x follows the fingers on the left
    hand and opposes them on the right. z=x cross y. The sign is fixed by
    index/little MCP ordering and handedness, not by the viewing direction.
    """
    if joints.shape != (21, 3) or not np.isfinite(joints[PALM]).all():
        return None
    if side not in (0, 1):
        raise ValueError(f"invalid side: {side}")
    p = joints - joints[0]
    y = np.cross(p[5], p[17]) * (1 if side == 0 else -1)
    if np.linalg.norm(y) < 1e-6:
        return None
    y /= np.linalg.norm(y)
    x = p[9] * (1 if side == 0 else -1)
    x -= y * np.dot(x, y)
    if np.linalg.norm(x) < 1e-3:
        return None
    x /= np.linalg.norm(x)
    T = np.eye(4)
    T[:3, :3] = np.column_stack((x, y, np.cross(x, y)))
    T[:3, 3] = joints[0]
    return T if rigid(T) else None


def interpolate(T, times, query):
    if len(times) < 2 or np.any(np.diff(times) <= 0):
        raise ValueError("trajectory times must strictly increase, at least two rows")
    if np.min(query) < times[0] - 1e-8 or np.max(query) > times[-1] + 1e-8:
        raise ValueError("refusing to extrapolate camera trajectory")
    query = np.clip(query, times[0], times[-1])
    result = np.tile(np.eye(4), (len(query), 1, 1))
    result[:, :3, :3] = Slerp(times, Rotation.from_matrix(T[:, :3, :3]))(query).as_matrix()
    result[:, :3, 3] = np.column_stack([np.interp(query, times, T[:, k, 3]) for k in range(3)])
    return result


def nearest(times, query):
    hi = np.clip(np.searchsorted(times, query), 1, len(times) - 1)
    lo = hi - 1
    return np.where(np.abs(times[lo] - query) <= np.abs(times[hi] - query), lo, hi)


def load_camera(vio: Path, origin_ns: int):
    with (vio / "headring_poses.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    metadata = json.loads((vio / "input_adapter/metadata.json").read_text())
    from ego_data.time_sync import require_zero_camera_offset
    require_zero_camera_offset(metadata, str(vio))
    shift = 0
    # Subtract integers before converting epoch nanoseconds to floating point.
    t = np.array([(int(r["timestamp_ns"]) - origin_ns) / 1e9 for r in rows])
    raw_t = np.array([(int(r["source_cam0_timestamp_us"]) * 1000 - origin_ns) / 1e9 for r in rows])
    if not np.allclose(t, raw_t, atol=1e-8, rtol=0):
        raise ValueError("VIO and original camera timestamps disagree")
    T = np.tile(np.eye(4), (len(rows), 1, 1))
    T[:, :3, :3] = Rotation.from_quat([[float(r[k]) for k in ("qx", "qy", "qz", "qw")] for r in rows]).as_matrix()
    T[:, :3, 3] = [[float(r[k]) for k in ("x_m", "y_m", "z_m")] for r in rows]
    if not all(rigid(pose) for pose in T):
        raise ValueError("invalid camera poses")
    return T, t, shift


def load_imu(vio: Path, origin_ns: int, shift: int):
    if shift != 0:
        raise ValueError('Camera/IMU time correction is disabled; offset must be 0 ms')
    with (vio / "trajectory.csv").open() as stream:
        rows = np.asarray(list(csv.reader(line for line in stream if line.strip() and not line.lstrip().startswith("#"))))
    if rows.ndim != 2 or rows.shape[1] != 8:
        raise ValueError("expected EuRoC IMU trajectory: timestamp, xyz, qw qx qy qz")
    times = np.array([(int(r[0]) - origin_ns) / 1e9 for r in rows])
    poses = np.tile(np.eye(4), (len(rows), 1, 1))
    poses[:, :3, :3] = Rotation.from_quat(rows[:, [5, 6, 7, 4]].astype(float)).as_matrix()
    poses[:, :3, 3] = rows[:, 1:4].astype(float)
    if not all(rigid(T) for T in poses):
        raise ValueError("invalid IMU trajectory")
    return poses, times


def neutral_head_rotation(T_world_imu0, T_world_cam0):
    """Initial upright head, heading from cam_0 optical forward projected onto XY.

    Returns constant R_imu_head. Later poses retain all relative IMU rotation.
    This is a user-authorized neutral-pose approximation, not measured anatomy.
    """
    up = np.array([0., 0., 1.])
    forward = T_world_cam0[:3, 2].copy()
    forward[2] = 0
    if np.linalg.norm(forward) < 0.1:
        raise ValueError("initial camera forward too vertical to determine head heading")
    forward /= np.linalg.norm(forward)
    left = np.cross(up, forward)
    R_world_head0 = np.column_stack((left, up, forward))
    return T_world_imu0[:3, :3].T @ R_world_head0


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def prepare(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    head_at_imu = getattr(args, "head_at_imu", False)
    neutral = getattr(args, "neutral_first_frame", False)
    if neutral != head_at_imu:
        raise ValueError("--head-at-imu requires --neutral-first-frame and vice versa")
    if head_at_imu and args.head_to_camera is not None:
        raise ValueError("choose IMU neutral approximation or calibrated head-to-camera")
    final_requested = head_at_imu or args.head_to_camera is not None
    if final_requested != (args.floor_z is not None):
        raise ValueError("provide a head convention and --floor-z together, or neither")
    manifest = json.loads((args.dataset / "manifest.json").read_text())
    if manifest.get("source", {}).get("kind") != "orbbec_ego4_export":
        raise ValueError("requires normalized Orbbec EGO4 input")
    cal = json.loads((args.dataset / "calibration/camera0.json").read_text())
    if not np.allclose(cal["T_base_camera"], np.eye(4), atol=1e-6):
        raise ValueError("fusion base must be physical cam_0 optical frame")
    with (args.dataset / "multiview_frames.csv").open() as stream:
        frames = list(csv.DictReader(stream))
    ids = [int(r["sync_index"]) for r in frames]
    if ids != list(range(len(frames))) or len(frames) < 2:
        raise ValueError("expected contiguous sync indices")
    origin_ns = int(frames[0]["camera0_timestamp_ns"])
    hand_times = np.array([(int(r["camera0_timestamp_ns"]) - origin_ns) / 1e9 for r in frames])
    if np.any(np.diff(hand_times) <= 0):
        raise ValueError("hand timestamps must strictly increase")
    T_camera, camera_times, shift = load_camera(args.vio, origin_ns)
    T_imu = None
    if head_at_imu:
        T_imu, imu_times = load_imu(args.vio, origin_ns, shift)
        if len(imu_times) != len(camera_times) or not np.allclose(imu_times, camera_times, atol=1e-8, rtol=0):
            raise ValueError("IMU trajectory and derived camera pose timestamps disagree")
    wrist = np.tile(np.eye(4), (len(frames), 2, 1, 1))
    visible = np.zeros((len(frames), 2), dtype=bool)
    joints = np.full((len(frames), 2, 21, 3), np.nan, dtype=np.float32)
    seen = set()
    rejected_palm = 0
    fusion_path = args.fusion / "accepted.jsonl"
    for line in fusion_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        i = int(row["sync_index"])
        if not 0 <= i < len(frames):
            raise ValueError(f"fusion frame out of range: {i}")
        for hand in row["hands"]:
            side = int(hand["side"])
            if side not in (0, 1) or (i, side) in seen:
                raise ValueError(f"invalid or duplicate hand: {i}, {side}")
            seen.add((i, side))
            p = np.asarray(hand["joints_base_m"], dtype=float)
            count = np.asarray(hand["inlier_view_counts"], dtype=int)
            if p.shape != (21, 3) or count.shape != (21,):
                raise ValueError(f"invalid hand dimensions: {i}, {side}")
            joints[i, side] = p
            pose = wrist_from_joints(p, side)
            # A recovered full hand is not necessarily a current wrist observation.
            if pose is None or np.any(count[PALM] < args.min_palm_views):
                rejected_palm += 1
                continue
            wrist[i, side], visible[i, side] = pose, True

    within = (hand_times >= camera_times[0]) & (hand_times <= camera_times[-1])
    wrists_outside = int(visible[~within].sum())
    visible[~within] = False
    wrist[~visible] = np.eye(4)
    world_wrist = np.tile(np.eye(4), wrist.shape[:2] + (1, 1))
    cam_at_hand = interpolate(T_camera, camera_times, hand_times[within])
    world_wrist[within] = cam_at_hand[:, None] @ wrist[within]
    world_wrist[~visible] = np.eye(4)
    grid = camera_times[0] + np.arange(int(np.floor((camera_times[-1] - camera_times[0]) * 30)) + 1) / 30
    camera_30 = interpolate(T_camera, camera_times, grid)
    j = nearest(hand_times, grid)
    visible_30 = visible[j] & (np.abs(hand_times[j] - grid) <= 1 / 60)[:, None]
    wrist_30 = world_wrist[j].copy()
    wrist_30[~visible_30] = np.eye(4)
    info = {
        "adapter": "orbbec_multiview_basalt", "dataset": str(args.dataset.resolve()),
        "fusion": str(args.fusion.resolve()), "vio": str(args.vio.resolve()),
        "time_origin_camera0_ns": origin_ns, "camera_to_imu_shift_ns": shift,
        "world": "Basalt metric z-up world; floor not aligned in intermediate files",
        "wrist_orientation": "geometric palm frame; y dorsal, x left wrist-to-middle / right reversed, z=x cross y; approximate anatomical orientation",
        "palm_joint_indices": PALM, "min_palm_inlier_views": args.min_palm_views,
        "hand_order": ["left", "right"], "head_calibration_provided": args.head_to_camera is not None,
        "head_position_reference": "IMU origin" if head_at_imu else "calibrated head joint or pending",
        "frames_source": len(frames), "frames_30hz": len(grid),
        "visible_left_30hz": int(visible_30[:, 0].sum()), "visible_right_30hz": int(visible_30[:, 1].sum()),
        "hands_rejected_for_palm": rejected_palm, "wrists_outside_vio": wrists_outside,
        "source_frames_outside_vio": int((~within).sum()),
        "sha256": {str(p.resolve()): sha256(p) for p in [fusion_path, args.dataset / "manifest.json", args.vio / "headring_poses.csv"]},
    }
    head_T = None
    head_30 = None
    if head_at_imu:
        head_T = np.eye(4)
        head_T[:3, :3] = neutral_head_rotation(T_imu[0], T_camera[0])
        head_30 = interpolate(T_imu, camera_times, grid) @ head_T
        info.update(head_to_imu=head_T.tolist(), floor_z=float(args.floor_z),
                    calibration_note=args.calibration_note,
                    head_orientation="first frame upright, facing horizontal projection of cam_0 optical forward; subsequent relative rotation from IMU",
                    anatomical_calibration="approximation authorized by user; head position equals IMU origin; no measured head mounting extrinsic")
        info["sha256"][str((args.vio / "trajectory.csv").resolve())] = sha256(args.vio / "trajectory.csv")
    elif args.head_to_camera is not None:
        path = args.head_to_camera
        head_T = np.load(path, allow_pickle=False) if path.suffix == ".npy" else np.loadtxt(path)
        if not rigid(head_T) or not np.isfinite(args.floor_z):
            raise ValueError("invalid anatomical calibration or floor height")
        info.update(head_to_camera=head_T.tolist(), floor_z=float(args.floor_z),
                    calibration_note=args.calibration_note,
                    calibration_sha256=sha256(path))
        head_30 = camera_30 @ head_T
    if final_requested and not np.isfinite(args.floor_z):
        raise ValueError("floor height must be finite")
    args.output.mkdir(parents=True)
    np.savez_compressed(args.output / "camera_trajectory.npz", T_world_cam=T_camera, time_s=camera_times)
    if T_imu is not None:
        np.savez_compressed(args.output / "imu_trajectory.npz", T_world_imu=T_imu, time_s=camera_times)
        np.save(args.output / "head_to_imu_approx.npy", head_T)
    np.savez_compressed(args.output / "wrists_camera.npz", wrist=wrist.astype(np.float32), visible=visible,
                        time_s=hand_times, joints_camera_m=joints)
    np.savez_compressed(args.output / "wrists_world.npz", wrist=world_wrist.astype(np.float32), visible=visible, time_s=hand_times)
    if head_30 is not None:
        device = camera_30.copy()
        device[:, 2, 3] -= args.floor_z
        wrist_30[visible_30, 2, 3] -= args.floor_z
        head_30[:, 2, 3] -= args.floor_z
        info["input_world"] = "Basalt world shifted in Z so estimated floor is zero; metric z-up"
        np.savez_compressed(args.output / "omniegocap_input.npz", head=head_30.astype(np.float32),
                            wrist=wrist_30.astype(np.float32), visible=visible_30,
                            device_pose=device.astype(np.float32), raw_time_s=grid,
                            device=np.str_("orbbec_ego4"), meta=np.str_(json.dumps(info)))
    else:
        info["pending"] = ["T_C0_H anatomical head-to-camera calibration", "floor_z in Basalt world"]
    (args.output / "summary.json").write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(info, indent=2, ensure_ascii=False))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--fusion", type=Path, required=True)
    p.add_argument("--vio", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--min-palm-views", type=int, default=3)
    p.add_argument("--head-to-camera", type=Path, help="T_C0_H: head point -> physical cam_0 optical coordinates, metres")
    p.add_argument("--head-at-imu", action="store_true", help="approximate head joint position by IMU origin")
    p.add_argument("--neutral-first-frame", action="store_true", help="explicitly assume first head pose upright, facing along cam_0 horizontal forward")
    p.add_argument("--floor-z", type=float, help="floor height in the original Basalt world")
    p.add_argument("--calibration-note", default="user-supplied calibration")
    args = p.parse_args()
    if args.min_palm_views < 2:
        p.error("min-palm-views must be >=2")
    prepare(args)


if __name__ == "__main__":
    main()
