# Source work/soft35_export_20261119/package.py, SHA-256 16b4bc0574a265809efe430942126e56c82e01caa9dbe9063605080fd9f723c8

import case_settings as case

from collections import Counter

import json

import os

import numpy as np

from common import HANDS, MOTION, TIPS, gate_and_errors, geometry, isolated_poses, load, output, project, read_jsonl, save, summary, track_file, write_json, write_jsonl

from export_dataset import camera_label, reconstructed_poses

from export_multiview_wilor_training_dataset import _all_points_inside

from wilor_label_projection import project_label_points

from wilor_frame_annotations import write_frame_index

def timeline(group, tracks=None):
    _, cameras, transforms, frames = geometry(group)
    sync = np.array(sorted(frames), np.int32)
    n = len(sync)

    def blank(*shape):
        return np.full((n, *shape), np.nan, np.float32)

    line = dict(
        sync_index=sync,
        reference_timestamp_ns=np.array(
            [int(frames[t]["reference_timestamp_ns"]) for t in sync], np.int64
        ),
        source_camera_frames=np.array(
            [[int(frames[t][f"camera{c}_frame_index"]) for c in range(4)] for t in sync], np.int32
        ),
        camera_ids=np.array([f"camera{c}" for c in range(4)]),
        physical_hands=np.array(HANDS),
        cuff_marker_names=np.array(["WristM", "WristIn", "WristOut"]),
        pose_valid=np.zeros((n, 2), bool),
        isolated_pose=np.zeros((n, 2), bool),
        tip_relaxed=np.zeros((n, 2), bool),
        cuff_points_observed=np.zeros((n, 2), np.uint8),
        poses_pass_surface_limits=np.zeros((n, 2), bool),
        median_marker_residual_mm=blank(2),
        joints_camera0_m=blank(2, 21, 3),
        joints_2d_native=blank(4, 2, 21, 2),
        source_joints_2d_native=blank(4, 2, 21, 2),
        nail_markers_2d_native=blank(4, 2, 5, 2),
        tip_reference_camera0_m=blank(2, 5, 3),
        tip_drift_mm=blank(2, 5),
        cuff_observed=np.zeros((n, 2, 3), bool),
        cuff_observed_camera0_m=blank(2, 3, 3),
        cuff_model_camera0_m=blank(2, 3, 3),
        cuff_observed_2d_native=blank(4, 2, 3, 2),
        cuff_model_2d_native=blank(4, 2, 3, 2),
    )
    details = []
    for side, hand in enumerate(HANDS):
        source = track_file(group, hand) if tracks is None else tracks[hand]
        track = load(source)
        times = track["pair_indices"]
        ii = np.searchsorted(sync, times)
        np.testing.assert_equal(sync[ii], times)
        marker = np.where(track["marker_valid"][:, :, None], track["target24"], np.nan)
        gate, err = gate_and_errors(track)
        mirror = np.array([1 if side else -1, 1, 1], np.float32)
        prior_path=output(group)/'outward/tracks'/(hand+'.npz')
        with np.load(prior_path) as pb:
            prior = pb["joints"]
            np.testing.assert_equal(pb["pair_indices"], times)
        np.testing.assert_equal(prior[:, TIPS], track["tip_reference_m"])
        line["pose_valid"][ii, side] = True
        line["isolated_pose"][ii, side] = isolated_poses(times)
        line["tip_relaxed"][ii, side] = track.get("tip_relaxed", np.zeros(len(times), bool))
        line["cuff_points_observed"][ii, side] = track["marker_valid"][:, :3].sum(1)
        line["poses_pass_surface_limits"][ii, side] = gate
        line["median_marker_residual_mm"][ii, side] = np.nanmedian(err[:, 3:], axis=1)
        line["joints_camera0_m"][ii, side] = track["joints"] * mirror
        line["tip_reference_camera0_m"][ii, side] = prior[:, TIPS] * mirror
        line["tip_drift_mm"][ii, side] = (
            np.linalg.norm(track["joints"][:, TIPS] - prior[:, TIPS], axis=-1) * 1000
        )
        line["cuff_observed"][ii, side] = track["marker_valid"][:, :3]
        line["cuff_observed_camera0_m"][ii, side] = marker[:, :3] * mirror
        line["cuff_model_camera0_m"][ii, side] = track["predicted24"][:, :3] * mirror
        depths = []
        for c, (cam, rt) in enumerate(zip(cameras, transforms)):
            line["joints_2d_native"][ii, c, side] = project(track["joints"], rt, cam, side)
            line["source_joints_2d_native"][ii, c, side] = project(prior, rt, cam, side)
            line["nail_markers_2d_native"][ii, c, side] = project(marker[:, 7::4], rt, cam, side)
            line["cuff_observed_2d_native"][ii, c, side] = project(marker[:, :3], rt, cam, side)
            line["cuff_model_2d_native"][ii, c, side] = project(
                track["predicted24"][:, :3], rt, cam, side
            )
            depths.append((track["vertices"] @ (rt[0][2] * mirror) + rt[1][2]).min(1))
        full = reconstructed_poses(track)
        details.append(
            (track, full, gate, err, np.stack(depths), {int(t): j for j, t in enumerate(times)})
        )
    line["passes_source_marker_limits"] = line["poses_pass_surface_limits"]
    line["pass_surface"] = line["poses_pass_surface_limits"]
    return line, details, cameras, transforms

def write_dataset(group, stage, *, tracks=None, metadata=None):
    line, details, cameras, transforms = timeline(group, tracks)
    save(stage / "sequence.npz", line)
    source = MOTION / group / "final/export/merged"
    base = json.loads((source / "summary.json").read_text())
    clocks = dict(zip(line["sync_index"].tolist(), range(len(line["sync_index"]))))
    index = []
    omitted = []
    counts = Counter()
    max_px = 0.0
    for frame in read_jsonl(source / "frames.jsonl"):
        t = frame["sync_index"]
        cam_id = frame["camera"]
        c = int(cam_id[-1])
        k = clocks[t]
        cam = cameras[c]
        valid = []
        reasons = []
        for h in frame["hands"]:
            side = h["side"]
            hand = HANDS[side]
            track, full, gate, errors, depth, slots = details[side]
            i = slots[t]
            uv = line["joints_2d_native"][k, c, side]
            if depth[c, i] <= 1e-8 or not _all_points_inside(uv, cam.image_size):
                reasons.append(
                    hand
                    + (
                        "_mesh_crosses_camera_plane"
                        if depth[c, i] <= 1e-8
                        else "_joints_outside_native_image"
                    )
                )
                continue
            try:
                label = camera_label(track, i, full, transforms[c], cam, side)
            except ValueError as exc:
                if str(exc) != "No visible mesh in a source-qualified view":
                    raise
                reasons.append(hand + "_mesh_visibility_below_threshold")
                continue
            pixels = project_label_points(label["joints_3d"] + label["trans"], label)
            delta = np.max(np.abs(pixels - uv))
            assert delta < 0.002, (group, t, cam_id, hand, delta)
            max_px = max(max_px, float(delta))
            flag = bool(gate[i])
            label["pass_surface"] = flag
            label["isolated_pose"] = bool(line["isolated_pose"][k, side])
            label["tip_relaxed"] = bool(line["tip_relaxed"][k, side])
            label["cuff_points_observed"] = int(line["cuff_points_observed"][k, side])
            pc = line["cuff_observed_2d_native"][k, c, side]
            pm = line["cuff_model_2d_native"][k, c, side]
            seen = np.isfinite(pc).all(1) & (pc >= 0).all(1) & (pc < cam.image_size).all(1)
            r = dict(
                sync_index=t,
                side=side,
                camera=cam_id,
                physical_side=side,
                handedness=hand,
                image=frame["image"],
                label=h["label"],
                reference_timestamp_ns=frame["reference_timestamp_ns"],
                source_frame_index=frame["source_frame_index"],
                source_image=str(source / frame["image"]),
                source_group=group,
                reference_camera="camera0",
                stored_image_space="native_raw",
                stored_image_horizontally_flipped=False,
                mano_parameter_space="right_canonical",
                source_optimized_fit=str(track_file(group, hand) if tracks is None else tracks[hand]),
                source_track_index=i,
                cuff_fitted=True,
                fingertip_reference="prior_bilateral_outward_same_hand_same_sync_3d",
                maximum_tip_displacement_mm=float(line["tip_drift_mm"][k, side].max()),
                cuff_observed_count=int(track["marker_valid"][i, :3].sum()),
                cuff_fit_median_mm=summary(errors[i, :3])["median"],
                cuff_visible_count=int(seen.sum()),
                cuff_median_px=float(np.median(np.linalg.norm(pm[seen] - pc[seen], axis=1)))
                if seen.any()
                else None,
                fit_nokov_marker_median_mm=float(line["median_marker_residual_mm"][k, side]),
                fit_nokov_marker_p95_mm=float(np.nanpercentile(errors[i, 3:], 95)),
                fit_error_point_semantics="HandOffset and 20 named finger surface mounts; cuff measured separately",
                passes_source_marker_limits=flag,
                pass_surface=flag,
                isolated_pose=label["isolated_pose"],
                tip_relaxed=label["tip_relaxed"],
                cuff_points_observed=label["cuff_points_observed"],
                approved_for_training=False,
            )
            valid.append((r, label))
        if reasons:
            omitted.append(dict(sync_index=t, camera=cam_id, reason=reasons[0], reasons=reasons))
            continue
        assert sorted(r["side"] for r, _ in valid) == [0, 1]
        os.link(source / frame["image"], stage / frame["image"])
        for row, label in valid:
            row["index"] = len(index)
            np.save(stage / row["label"], label)
            counts[cam_id] += 1
            index.append(row)
    write_jsonl(stage / "index.jsonl", index)
    write_jsonl(stage / "rejected.jsonl", omitted)
    write_frame_index(stage / "frames.jsonl", index)
    n = len(index) // 2
    assert n + len(omitted) == base["image_count"]
    keys = (
        "schema_version",
        "image_layout",
        "annotation_policy",
        "image_size",
        "image_space",
        "geometry_space",
        "mano_parameter_space",
        "side_semantics",
        "mano_model_by_side",
        "mano_pose_representation",
        "mano_model",
        "mano_source_revision",
        "mano_assets",
        "camera_calibrations",
        "source_dataset",
        "reference_camera",
        "jpeg_quality",
    )
    result = {k: base[k] for k in keys}
    decision_path = output(group) / "decision.json"
    limits = (
        json.loads(decision_path.read_text())["limitations"] if decision_path.exists() else []
    )
    result.update(
        stage="native_cuff_soft35",
        source_group=group,
        side_counts={h: n for h in HANDS},
        sample_count=n * 2,
        image_count=n,
        label_count=n * 2,
        camera_counts=dict(counts),
        rejected_camera_frames=len(omitted),
        native_vs_sequence_max_abs_px=max_px,
        sequence="sequence.npz",
        sync_count=len(clocks),
        wrist_marker_sequence="sequence.npz",
        soft_tip_kink_mm=3.0,
        soft_tip_cap_mm=3.5,
        fit_tracks={h: str(track_file(group, h) if tracks is None else tracks[h]) for h in HANDS},
        approved_for_training=False,
        pass_surface="Per physical hand and sync; retained for both passing and failing fits.",
        known_limitations=limits,
    )
    coverage_path = output(group) / "cuff_coverage.json"
    if coverage_path.exists():
        coverage = json.loads(coverage_path.read_text())
        result["cuff_missing_full_timeline"] = dict(
            recording_syncs=coverage["recording_syncs"],
            both_hands_without_any_point=coverage["both_hands_without_any_point"],
            hands={
                h: dict(missing=v["missing"], percent=v["missing_percent"])
                for h, v in coverage["hands"].items()
            },
        )
    if metadata is not None:
        result.pop("soft_tip_cap_mm", None)
        result.pop("soft_tip_kink_mm", None)
        result.update(metadata)
    write_json(stage / "summary.json", result)
    print(
        group,
        "images",
        n,
        "labels",
        n * 2,
        "all syncs",
        len(clocks),
        "excluded views",
        len(omitted),
        flush=True,
    )
