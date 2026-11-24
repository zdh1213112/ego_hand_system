"""Build one v4 training dataset from the finished, time-split recording candidate."""

from collections import Counter, defaultdict
import csv
import fcntl
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np
import torch

PIPE = Path(__file__).resolve().parents[1]
REPO = PIPE.parents[1]
sys.path[:0] = [str(PIPE), str(REPO / "scripts"), str(REPO / "scripts/joint_refit")]
from stages.fingerprint import digest
from stages.raw_release import lines
from stages.signed import rows, write_json
from stages.legacy.raw_export import setup
from stages.merge_verified import verify_loaders
from fit_mano_sequence import import_mano
from wilor_label_projection import project_label_points


def load(path):
    return json.loads(path.read_text())


def corrections(root, work, train, index, cfg):
    from v4.fit import propose
    from v4.optics import verify

    group = root.name
    times = {r["sync_index"] for r in train if r["side"] == 1}
    if not times:
        return {}, {}
    mano = import_mano(REPO / "third_party/MANO")
    model = (
        mano.load(
            model_path=str(REPO / "models/mano"),
            is_rhand=True,
            use_pca=True,
            num_pca_comps=45,
            flat_hand_mean=False,
            batch_size=1,
        )
        .cuda()
        .eval()
        .requires_grad_(False)
    )
    track = root / "geometry_input" / group / "tracks/right.npz"
    proposal = work / "pinky_candidates.npz"
    with torch.no_grad():
        stats = propose(track, model, times, proposal, cfg["fit"])
    if not stats["corrected"]:
        return {}, {}
    images = defaultdict(dict)
    for row in index:
        images[row["sync_index"]][int(row["camera"][-1])] = root / row["image"]
    cameras = root / "native/processed/ego/normalized_multiview/calibration"
    evidence = verify(track, proposal, cameras, images, cfg["optics"])
    np.savez_compressed(work / "pinky_evidence.npz", **evidence)
    optical = evidence["accept"]
    fallback = (
        (evidence["support"].sum(1) < cfg["optics"]["min_views"])
        & (evidence["raw_before_mm"] - evidence["raw_after_mm"] >= cfg["selection"]["min_raw_gain_mm"])
        & (evidence["raw_before_mm"] < cfg["selection"]["max_raw_error_mm"])
        & np.all(
            ~np.isfinite(evidence["after_px"]) | (evidence["after_px"] <= evidence["before_px"] + cfg["selection"]["max_unconfirmed_loss_px"]),
            axis=1,
        )
    )
    if "_".join(group.split("_")[:2]) in cfg["selection"]["require_rgb_recordings"]:
        fallback[:] = False
    with np.load(proposal) as z:
        data = {key: z[key] for key in z.files}
    np.testing.assert_array_equal(data["sync"], evidence["sync"])
    use = {
        int(data["sync"][i]): (i, "rgb_and_nokov" if optical[i] else "nokov_two_registrations")
        for i in np.flatnonzero(optical | fallback)
    }
    return data, use


def update_label(label, fit, i, image_size, cfg):
    rot = label["mano"]["global_orient"][0] @ fit["ori"][i].T
    delta = fit["dj"][i] @ rot.T
    if np.max(np.linalg.norm(delta[:18], axis=1)) >= cfg["fixed_joint_tolerance_m"]:
        raise ValueError("Right pinky correction moved the wrist, another finger or pinky MCP")
    label["joints_3d"] += delta
    label["vertices"] += fit["dv"][i] @ rot.T
    label["mano"]["hand_pose"] = fit["pose"][i].copy()
    pixels = project_label_points(label["vertices"] + label["trans"], label)
    label["joints_2d"] = torch.from_numpy(pixels.astype("f"))
    inside = (pixels >= 0).all(1) & (pixels < image_size).all(1)
    box = label["bbox"]
    enclosing = np.r_[np.minimum(box[:2], pixels[inside].min(0)), np.maximum(box[2:], pixels[inside].max(0))]
    if np.max(np.abs(box - enclosing)) > cfg["bbox_tolerance_px"]:
        label["bbox"] = enclosing


def build(root, dst, final, cfg):
    setup(root / "records/raw_case.json")
    (dst / "labels").mkdir()
    (dst / "images").mkdir()
    review = dst / "review"
    review.mkdir()
    summary = load(root / "summary.json")
    index = list(rows(root / "index.jsonl"))
    train = list(rows(root / "train.jsonl"))
    train_ids = {r["index"] for r in train}
    val_ids = {r["index"] for r in rows(root / "val.jsonl")}
    if train_ids & val_ids or train_ids | val_ids != {r["index"] for r in index}:
        raise ValueError("Incomplete or overlapping source splits")
    fit, selected = corrections(root, review, train, index, cfg)
    unsupported = set()
    target = load(root / "review/dataset_targets.json")
    if any(target[h]["failed_19"] for h in ("left", "right")):
        from recordings.anatomy_audit import run

        run(root, review)
        for side, hand in enumerate(("left", "right")):
            for issue in load(review / "unsupported_exposures.json")["hands"][hand]:
                unsupported.add((issue["sync"], side))
    accepted, changed, removed = [], [], []
    positions = {"train": [], "val": []}
    by_image = defaultdict(list)
    sides, cameras = Counter(), Counter()
    for original in index:
        row = original.copy()
        train_sample = row["index"] in train_ids
        time, side = row["sync_index"], row["side"]
        if train_sample and (time, side) in unsupported:
            removed.append(
                dict(
                    sync_index=time,
                    side=side,
                    camera=row["camera"],
                    index=row["index"],
                    reason="unsupported_anatomy",
                )
            )
            continue
        row.pop("glove_label", None)
        row["approved_for_training"] = False
        src, dest = root / row["label"], dst / row["label"]
        if train_sample and side == 1 and time in selected:
            i, mode = selected[time]
            label = np.load(src, allow_pickle=True).item()
            update_label(label, fit, i, summary["image_size"], cfg["selection"])
            np.save(dest, label)
            changed.append(row["label"])
            row["label_refinement"] = "right_pinky_" + mode
        else:
            dest.symlink_to(os.path.relpath(src, dst / "labels"))
        if row["image"] not in by_image:
            (dst / row["image"]).symlink_to(os.path.relpath(root / row["image"], dst / "images"))
        by_image[row["image"]].append({k: row[k] for k in ("index", "label", "side")})
        sides[row["handedness"]] += 1
        cameras[row["camera"]] += 1
        positions["train" if train_sample else "val"].append(len(accepted))
        accepted.append(row)
    if not all(positions.values()):
        raise ValueError("Filtering left an empty training or validation split")
    lines(dst / "index.jsonl", accepted)
    for split, ids in positions.items():
        lines(dst / (split + ".jsonl"), (accepted[i] for i in ids))
    lines(
        dst / "frames.jsonl",
        (
            dict(r, hand_count=len(by_image[r["image"]]), hands=by_image[r["image"]])
            for r in rows(root / "frames.jsonl")
            if r["image"] in by_image
        ),
    )
    lines(dst / "rejected.jsonl", [*rows(root / "rejected.jsonl"), *removed])
    splits = load(root / "splits.json")
    write_json(
        dst / "splits.json", dict(splits, source=str(final / "index.jsonl"), source_positions=positions)
    )
    summary.update(
        sample_count=len(accepted),
        label_count=len(accepted),
        image_count=len(by_image),
        side_counts=dict(sides),
        camera_counts=dict(cameras),
        fit_method="adaptive_wrist_v4",
        label_refinement_version="v4_right_pinky",
        rejected_count=summary["rejected_count"] + len(removed),
        approved_for_training=False,
    )
    write_json(dst / "summary.json", summary)
    verify_loaders(dst, REPO)
    profile = load(dst / "dataset_profile.json")
    profile["dataset_root"] = str(final)
    if "manifest" in profile["geometry_validation"]:
        profile["geometry_validation"]["manifest"] = str(final / "index.jsonl")
    write_json(dst / "dataset_profile.json", profile)
    from recordings.preview import write_video
    from render_wilor_training_dataset import _draw_sample

    def draw(frame, item):
        sample = np.load(dst / item["label"], allow_pickle=True).item()
        _draw_sample(frame, sample, item)
        return frame

    native = root / "native/processed/ego/normalized_multiview"
    with (native / "multiview_frames.csv").open() as f:
        clock = list(csv.DictReader(f))
    paths = {(r["sync_index"], 0, r["side"]): r for r in accepted if r["camera"] == "camera0"}
    video = write_video(native, review, clock, paths, draw_label=draw)
    with (dst / "export_sha256.csv").open("w") as f:
        writer = csv.writer(f)
        for folder in ("images", "labels"):
            for p in sorted((dst / folder).iterdir()):
                writer.writerow([digest(p), str(p.relative_to(dst))])
    files = [
        "index.jsonl",
        "frames.jsonl",
        "train.jsonl",
        "val.jsonl",
        "splits.json",
        "summary.json",
        "rejected.jsonl",
        "export_sha256.csv",
        "loader_check.json",
        "dataset_profile.json",
        "review/camera0_full.mp4",
    ]
    proof = dict(
        version="v4",
        source=str(root),
        source_target_sha256=digest(root / "review/dataset_targets.json"),
        corrected_poses=len(selected),
        corrected_labels=len(changed),
        excluded_train_views=len(removed),
        train=len(positions["train"]),
        val=len(positions["val"]),
        video=video,
        files={f: digest(dst / f) for f in files},
    )
    write_json(review / "dataset_targets.json", proof)
    write_json(
        review / "dataset_signoff.json", dict(approved=False, sha256=digest(review / "dataset_targets.json"))
    )
    (review / "DATASET_REVIEW.md").write_text(
        f"# {root.name} v4 标签\n\n训练 {proof['train']} 份，验证 {proof['val']} 份。右小指修正 {len(changed)} 份，剔除训练样本 {len(removed)} 份。视频从最终训练格式 NPY 绘制，覆盖相机 0 的原始时间轴。\n\n[整段视频](camera0_full.mp4) · [逐步校验](dataset_targets.json) · [训练读取检查](../loader_check.json)\n"
    )
    print(json.dumps({k: v for k, v in proof.items() if k != "files"}), flush=True)


def publish(dataset, proof):
    sign = load(dataset / "review/dataset_signoff.json")
    target_sha = digest(dataset / "review/dataset_targets.json")
    if sign.get("approved") is not True or sign.get("sha256") != target_sha:
        raise ValueError("请先核对 optimized/review/DATASET_REVIEW.md，并签收该版本的 dataset_targets.json")
    release = dataset.parent / "published"
    if release.exists():
        if load(release / "release.json")["target_sha256"] != target_sha:
            raise FileExistsError(release)
        return
    with tempfile.TemporaryDirectory(prefix=".publish-", dir=release.parent) as temp:
        staging = Path(temp)
        for entry in (
            "labels",
            "images",
            "review",
            "rejected.jsonl",
            "frames.jsonl",
            "export_sha256.csv",
            "loader_check.json",
        ):
            (staging / entry).symlink_to(os.path.relpath(dataset / entry, staging))
        for name in ("train", "val", "index"):
            lines(
                staging / (name + ".jsonl"),
                (dict(r, approved_for_training=True) for r in rows(dataset / (name + ".jsonl"))),
            )
        write_json(staging / "summary.json", dict(load(dataset / "summary.json"), approved_for_training=True))
        splits = load(dataset / "splits.json")
        write_json(staging / "splits.json", dict(splits, source=str(release / "index.jsonl")))
        profile = load(dataset / "dataset_profile.json")
        profile["dataset_root"] = str(release)
        profile["geometry_validation"]["manifest"] = str(release / "index.jsonl")
        write_json(staging / "dataset_profile.json", profile)
        write_json(
            staging / "release.json",
            dict(
                target_sha256=target_sha,
                source=str(dataset),
                train=proof["train"],
                val=proof["val"],
                approved_for_training=True,
            ),
        )
        staging.chmod(release.parent.stat().st_mode & 0o777)
        staging.rename(release)


def main(root, dest, requested_publish=False, settings=None):
    import yaml
    cfg = (load(settings)["effective_parameters"] if settings else yaml.safe_load((PIPE/"defaults.yaml").read_text()))["v4"]
    root, dest = root.resolve(strict=True), dest.resolve()
    with (dest.parent / ".v4.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if dest.exists():
            proof = load(dest / "review/dataset_targets.json")
            if proof["source_target_sha256"] != digest(root / "review/dataset_targets.json") or proof[
                "source"
            ] != str(root):
                raise ValueError("Existing v4 dataset belongs to a different candidate")
            for name, sha in proof["files"].items():
                if digest(dest / name) != sha:
                    raise ValueError(f"Changed v4 output: {name}")
            with (dest / "export_sha256.csv").open() as f:
                for sha, name in csv.reader(f):
                    if digest(dest / name) != sha:
                        raise ValueError(f"Changed v4 sample: {name}")
            if requested_publish:
                publish(dest, proof)
            return
        if requested_publish:
            raise FileNotFoundError(dest / "review/dataset_targets.json")
        with tempfile.TemporaryDirectory(prefix=".v4-", dir=dest.parent) as tmp:
            # Keep staging and final dataset at the same depth for relative symlinks.
            staging = Path(tmp)
            build(root, staging, dest, cfg)
            staging.chmod(dest.parent.stat().st_mode & 0o777)
            staging.rename(dest)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording_candidate", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--settings",type=Path)
    args = parser.parse_args()
    main(args.recording_candidate, args.output, args.publish, args.settings)
