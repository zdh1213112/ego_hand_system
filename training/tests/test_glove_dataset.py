from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from torch.utils.data import DataLoader

from wilor.datasets.glove_dataset import (
    GloveDataset,
    frame_group_key,
    load_manifest_rows,
    make_splits,
    write_manifest_rows,
)
from wilor.training.evaluation import crop_points_to_image


def test_legacy_loader_rejects_native_label_before_perspective_projection(
    tmp_path: Path,
) -> None:
    path = tmp_path / "native.npy"
    np.save(path, {"image_space": "native_raw", "camera_model": "KB"})
    with pytest.raises(ValueError, match="camera-aware training adapter"):
        GloveDataset._load_label(path)


def _write_sample(
    root: Path,
    index: int,
    side: int,
    sync_index: int,
    camera: str,
) -> dict:
    image_dir = root / "images"
    label_dir = root / "labels"
    image_dir.mkdir(exist_ok=True)
    label_dir.mkdir(exist_ok=True)
    image_path = image_dir / f"{index:06d}.jpg"
    label_path = label_dir / f"{index:06d}.npy"
    image = np.full((240, 320, 3), 90 + index, dtype=np.uint8)
    assert cv2.imwrite(str(image_path), image)
    joints = np.zeros((21, 3), dtype=np.float32)
    joints[:, 0] = np.linspace(-0.06, 0.06, 21)
    joints[:, 1] = np.linspace(-0.04, 0.04, 21)
    label = {
        "bbox": np.array([110, 65, 210, 185], dtype=np.float32),
        "vertices": np.zeros((778, 3), dtype=np.float32),
        "joints_3d": joints,
        "trans": np.array([0, 0, 1], dtype=np.float32),
        "K": np.array([
            [300, 0, 160],
            [0, 300, 120],
            [0, 0, 1],
        ], dtype=np.float32),
        "side": side,
    }
    label["vertices"][:, 2] = 0.05
    camera_vertices = label["vertices"] + label["trans"][None]
    homogeneous = camera_vertices @ label["K"].T
    label["joints_2d"] = (
        homogeneous[:, :2] / homogeneous[:, 2:3]
    ).astype(np.float32)
    np.save(label_path, label, allow_pickle=True)
    return {
        "index": index,
        "sync_index": sync_index,
        "source_frame_index": sync_index,
        "camera": camera,
        "side": side,
        "image": str(image_path.relative_to(root)),
        "label": str(label_path.relative_to(root)),
    }


def test_left_and_right_crop_coordinates_round_trip(tmp_path: Path) -> None:
    rows = [
        _write_sample(tmp_path, 0, 0, 10, "camera2"),
        _write_sample(tmp_path, 1, 1, 10, "camera2"),
    ]
    manifest = tmp_path / "manifest.jsonl"
    write_manifest_rows(manifest, rows)
    dataset = GloveDataset(manifest)

    assert set(dataset[0]["mano_params"]) == {
        "global_orient", "hand_pose", "betas",
    }
    assert not any(bool(value) for value in dataset[0]["has_mano_params"].values())

    for item, row in zip((dataset[0], dataset[1]), rows):
        label = np.load(tmp_path / row["label"], allow_pickle=True).item()
        expected = GloveDataset.project_joints(
            label["joints_3d"], label["trans"], label["K"]
        )
        restored = crop_points_to_image(
            item["keypoints_2d"].numpy(),
            item["crop_transform"].numpy(),
            item["image_size"].numpy(),
            is_left=row["side"] == 0,
        )
        np.testing.assert_allclose(restored, expected, atol=1e-4)


@pytest.mark.parametrize("side", [0, 1])
def test_bbox_pinhole_loader_maps_to_raw_kb_and_preserves_mano(tmp_path: Path, side: int) -> None:
    from tools.hand_finetune.crop_geometry import PerspectiveCrop

    row = _write_sample(tmp_path, 0, side, 0, "camera2")
    path = tmp_path / row["label"]
    label = np.load(path, allow_pickle=True).item()
    label.update(image_space="native_raw", camera_model="KB",
                 distortion=np.array([.12, -.04, .005, 0.]),
                 bbox=np.array([187., 73., 272., 177.]))
    label["trans"][0] += .24
    np.save(path, label)
    manifest = tmp_path / "val.jsonl"
    write_manifest_rows(manifest, [row])
    sample = GloveDataset(manifest, camera_model="kb_bbox_pinhole")[0]
    view = PerspectiveCrop(sample["view_rotation"].numpy(), sample["camera_K"].numpy(),
                           sample["raw_K"].numpy(), sample["camera_distortion"].numpy(), 256)
    pixels = crop_points_to_image(sample["keypoints_2d"].numpy(), sample["crop_transform"].numpy(),
                                 sample["image_size"].numpy(), is_left=side == 0)
    xyz = (label["joints_3d"] + label["trans"]).astype(np.float64)
    expected = cv2.fisheye.projectPoints(xyz[:, None], np.zeros(3), np.zeros(3),
                                         label["K"].astype(float), label["distortion"])[0][:, 0]
    np.testing.assert_allclose(view.native_pixels(pixels), expected, atol=.001)
    np.testing.assert_allclose(sample["camera_root_m"].numpy() @ view.rotation, xyz[0], atol=1e-6)
    rect_relative = sample["keypoints_3d"].numpy()[:, :3].copy()
    if side == 0:
        rect_relative[:, 0] *= -1
    np.testing.assert_allclose((rect_relative - rect_relative[0]) @ view.rotation, xyz - xyz[0], atol=1e-6)


def test_max_frames_keeps_both_hands_in_an_interleaved_manifest(
    tmp_path: Path,
) -> None:
    rows = [
        _write_sample(tmp_path, 0, 0, 10, "camera2"),
        _write_sample(tmp_path, 1, 0, 10, "camera3"),
        _write_sample(tmp_path, 2, 1, 10, "camera2"),
        _write_sample(tmp_path, 3, 1, 10, "camera3"),
    ]
    manifest = tmp_path / "manifest.jsonl"
    write_manifest_rows(manifest, rows)
    dataset = GloveDataset(manifest, max_frames=1)
    assert len(dataset) == 2
    assert {(row["camera"], row["side"]) for row in dataset.rows} == {
        ("camera2", 0),
        ("camera2", 1),
    }


def test_measured_only_rows_never_supply_fitted_joint_targets(tmp_path: Path) -> None:
    measured = _write_sample(tmp_path, 0, 0, 10, "camera2")
    trusted = _write_sample(tmp_path, 1, 1, 10, "camera2")
    manifest = tmp_path / "manifest.jsonl"
    write_manifest_rows(manifest, [dict(measured, measured_surface_only=True), trusted])

    batch = next(iter(DataLoader(GloveDataset(manifest), batch_size=2)))
    assert batch["geometry_supervised"].tolist() == [False, True]
    for key in ("keypoints_2d", "keypoints_3d"):
        assert batch[key][0, :, -1].count_nonzero() == 0
        assert batch[key][1, :, -1].count_nonzero() == 21


def test_frame_groups_namespace_reused_sync_indices_by_source_dataset() -> None:
    first = {
        "source_dataset_id": 0,
        "sync_index": 40,
        "camera": "camera2",
        "side": 0,
    }
    other_hand = {**first, "side": 1}
    other_recording = {**first, "source_dataset_id": 1}

    assert frame_group_key(first) == frame_group_key(other_hand)
    assert frame_group_key(first) != frame_group_key(other_recording)


def test_detector_box_drives_both_image_crop_and_depth_prior(tmp_path: Path) -> None:
    row = _write_sample(tmp_path, 0, 1, 4, "camera1")
    proposal = [80., 40., 220., 190.]
    manifest = tmp_path / "manifest.jsonl"
    write_manifest_rows(manifest, [dict(row, input_bbox_xyxy=proposal)])
    dataset = GloveDataset(manifest)
    plain = dataset[0]
    np.testing.assert_allclose(plain["bbox_xyxy"], proposal)
    jittered = GloveDataset(manifest, crop_jitter=.15)[0]
    for item in (plain, jittered):
        box = item["bbox_xyxy"].numpy()
        center = (box[:2] + box[2:]) / 2
        restored = item["crop_transform"].numpy() @ np.r_[center, 1]
        np.testing.assert_allclose(restored, [128, 128], atol=1e-3)


def test_random_splits_overwrite_old_manifests_and_are_reproducible(
    tmp_path: Path,
) -> None:
    rows = [
        {
            "index": index,
            "sync_index": index,
            "side": index % 2,
            "camera": f"camera{index % 3}",
        }
        for index in range(60)
    ]
    (tmp_path / "index.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    for name in ("train.jsonl", "val.jsonl", "test.jsonl"):
        (tmp_path / name).write_text('{"old":true}\n', encoding="utf-8")

    metadata = make_splits(tmp_path, train_ratio=0.9, split_seed=42)
    train = load_manifest_rows(tmp_path / "train.jsonl")
    val = load_manifest_rows(tmp_path / "val.jsonl")
    assert metadata["samples"] == {"train": 54, "val": 6}
    assert not (tmp_path / "test.jsonl").exists()
    assert all("old" not in row for row in train + val)
    assert {row["index"] for row in train}.isdisjoint(
        {row["index"] for row in val}
    )
    assert {row["index"] for row in train + val} == set(range(60))

    first_membership = [row["index"] for row in train]
    repeated = make_splits(tmp_path, train_ratio=0.9, split_seed=42)
    assert repeated["source_positions"] == metadata["source_positions"]
    assert [
        row["index"] for row in load_manifest_rows(tmp_path / "train.jsonl")
    ] == first_membership

    make_splits(tmp_path, train_ratio=0.9, split_seed=43)
    assert [
        row["index"] for row in load_manifest_rows(tmp_path / "train.jsonl")
    ] != first_membership


def test_source_group_split_keeps_recordings_intact(tmp_path: Path) -> None:
    rows = [
        {"index": index, "side": index % 2, "camera": "camera2",
         "sync_index": index // 2, "source_group": f"session_{index // 8}"}
        for index in range(24)
    ]
    (tmp_path / "index.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    metadata = make_splits(tmp_path, group_by_source=True, split_seed=42)
    train = load_manifest_rows(tmp_path / "train.jsonl")
    val = load_manifest_rows(tmp_path / "val.jsonl")
    assert metadata["strategy"] == "source_group"
    assert {row["source_group"] for row in train}.isdisjoint(
        {row["source_group"] for row in val}
    )
    assert len(train) + len(val) == 24


def test_2d_visibility_requires_source_and_crop_visibility(tmp_path: Path) -> None:
    row = _write_sample(tmp_path, 0, 1, 10, "camera2")
    label_path = tmp_path / row["label"]
    label = np.load(label_path, allow_pickle=True).item()
    # u=400 is outside the 320px source image; u=300 is inside the source but
    # outside the expanded crop whose horizontal extent ends near x=280.
    label["joints_3d"][0] = np.array([0.8, 0.0, 0.0], dtype=np.float32)
    label["joints_3d"][1] = np.array([140.0 / 300.0, 0.0, 0.0], dtype=np.float32)
    np.save(label_path, label, allow_pickle=True)
    manifest = tmp_path / "manifest.jsonl"
    write_manifest_rows(manifest, [row])

    keypoints = GloveDataset(manifest)[0]["keypoints_2d"].numpy()
    assert keypoints[0, 2] == 0.0
    assert keypoints[1, 2] == 0.0
    assert np.all(keypoints[2:, 2] == 1.0)
