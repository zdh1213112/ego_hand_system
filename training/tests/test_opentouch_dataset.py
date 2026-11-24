from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from wilor.datasets.glove_dataset import load_manifest_rows, write_manifest_rows
from wilor.datasets.opentouch_dataset import (
    OpenTouchDataset,
    opentouch_frame_key,
    prepare_opentouch_dataset,
)


def _write_fixture(root: Path) -> tuple[Path, Path]:
    decoded = root / "decoded"
    rows = []
    for clip_index, clip_id in enumerate(("clip_a", "clip_b")):
        image_relative = Path("images") / "recording" / clip_id / "000000.jpg"
        image_path = decoded / image_relative
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image = np.full((480, 640, 3), 40 + clip_index * 20, dtype=np.uint8)
        assert cv2.imwrite(str(image_path), image)
        landmarks = np.zeros((1, 21, 3), dtype=np.float32)
        landmarks[0, :, 2] = np.linspace(0, 0.2, 21)
        landmarks[0, 5] = [1.0, 0.0, 0.0]
        landmarks[0, 9] = [0.0, 1.0, 0.0]
        landmarks[0, 17] = [-1.0, 0.0, 0.0]
        clip_relative = Path("clips") / "recording" / f"{clip_id}.npz"
        clip_path = decoded / clip_relative
        clip_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            clip_path,
            right_hand_landmarks=landmarks,
            source_frame_indices=np.asarray([0], dtype=np.int64),
        )
        rows.append({
            "sample_id": f"recording::{clip_id}::000000",
            "source_file": "recording.hdf5",
            "clip_id": clip_id,
            "source_frame_index": 0,
            "clip_data_index": 0,
            "sync_index": 0,
            "image": str(image_relative),
            "clip_data": str(clip_relative),
            "handedness": "right",
        })
    write_manifest_rows(decoded / "frames.jsonl", rows)
    annotations = root / "bboxes.jsonl"
    # Reverse annotation order and repeat sync_index across clips: only the
    # composite immutable key is allowed to control this join.
    write_manifest_rows(annotations, [{
        "sample_id": row["sample_id"],
        "source_file": row["source_file"],
        "clip_id": row["clip_id"],
        "source_frame_index": row["source_frame_index"],
        "sync_index": 0,
        "bbox_xyxy": [200, 100, 440, 420],
    } for row in reversed(rows)])
    return decoded, annotations


def test_prepare_and_load_opentouch_with_composite_frame_key(tmp_path: Path) -> None:
    decoded, annotations = _write_fixture(tmp_path)
    prepared = tmp_path / "prepared"
    result = prepare_opentouch_dataset(
        decoded,
        annotations,
        prepared,
        sample_fps=30,
        source_fps=30,
        val_ratio=0.5,
        split_seed=7,
    )
    assert result["profile"]["samples"] == {"train": 1, "val": 1}
    train_rows = load_manifest_rows(prepared / "train.jsonl")
    val_rows = load_manifest_rows(prepared / "val.jsonl")
    assert opentouch_frame_key(train_rows[0]) != opentouch_frame_key(val_rows[0])
    assert train_rows[0]["sync_index"] == val_rows[0]["sync_index"] == 0
    sample = OpenTouchDataset(prepared / "train.jsonl")[0]
    assert sample["img"].shape == (3, 256, 256)
    assert sample["local_keypoints_3d"].shape == (21, 3)
    assert bool(sample["has_local_keypoints_3d"])


def test_prepare_requires_exact_bbox_coverage(tmp_path: Path) -> None:
    decoded, annotations = _write_fixture(tmp_path)
    rows = load_manifest_rows(annotations)[:-1]
    write_manifest_rows(annotations, rows)
    try:
        prepare_opentouch_dataset(decoded, annotations, tmp_path / "prepared")
    except ValueError as exc:
        assert "exactly cover" in str(exc)
    else:
        raise AssertionError("missing bbox annotation was accepted")


def test_sample_id_is_a_checksum_not_the_join_key(tmp_path: Path) -> None:
    decoded, annotations = _write_fixture(tmp_path)
    rows = load_manifest_rows(annotations)
    rows[0]["sample_id"] = "wrong"
    annotations.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    try:
        prepare_opentouch_dataset(decoded, annotations, tmp_path / "prepared")
    except ValueError as exc:
        assert "does not match frame key" in str(exc)
    else:
        raise AssertionError("incorrect sample_id was accepted")
