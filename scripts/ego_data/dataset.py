"""Readers for normalized raw and rectified stereo frame datasets."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np

from .calibration import CameraCalibration, StereoCalibration


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def read_image(path: Path, expected_size: tuple[int, int]) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"cannot read image: {path}")
    if image.shape[1::-1] != expected_size:
        raise RuntimeError(f"image {path} has size {image.shape[1::-1]}, expected {expected_size}")
    return image


class SequentialVideoReader:
    """Read strictly increasing frame indices without random video seeks."""

    def __init__(self, path: Path, expected_size: tuple[int, int]):
        self.path = path
        self.expected_size = expected_size
        self.capture = cv2.VideoCapture(str(path))
        if not self.capture.isOpened():
            raise RuntimeError(f"cannot open video: {path}")
        self.next_index = 0

    def read(self, target_index: int) -> np.ndarray:
        if target_index < self.next_index:
            raise RuntimeError(
                f"video frame indices must be read in increasing order: {target_index} < {self.next_index}"
            )
        frame = None
        while self.next_index <= target_index:
            ok, frame = self.capture.read()
            if not ok:
                raise RuntimeError(f"video {self.path} ended before frame {target_index}")
            self.next_index += 1
        if frame is None or frame.shape[1::-1] != self.expected_size:
            raise RuntimeError(f"decoded frame from {self.path} has an unexpected size")
        return frame

    def close(self) -> None:
        self.capture.release()


class NormalizedStereoDataset:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        if self.manifest.get("dataset_type") != "normalized_stereo":
            raise ValueError(f"not a normalized stereo dataset: {root}")
        self.left_id = self.manifest["left_camera"]
        self.right_id = self.manifest["right_camera"]
        self.left = CameraCalibration.load(self.root / "calibration" / f"{self.left_id}.json")
        self.right = CameraCalibration.load(self.root / "calibration" / f"{self.right_id}.json")
        self.stereo = StereoCalibration.from_cameras(self.left, self.right)
        self.pairs = read_csv(self.root / "stereo_pairs.csv")
        if not self.pairs:
            raise ValueError("normalized dataset contains no stereo pairs")

    def video_path(self, camera_id: str) -> Path:
        return self.root / "cameras" / camera_id / self.manifest["storage"]["video_filename"]

    def open_readers(self) -> tuple[SequentialVideoReader, SequentialVideoReader]:
        return (
            SequentialVideoReader(self.video_path(self.left_id), self.left.image_size),
            SequentialVideoReader(self.video_path(self.right_id), self.right.image_size),
        )

    def __iter__(self):
        left_reader, right_reader = self.open_readers()
        try:
            for row in self.pairs:
                yield row, (
                    left_reader.read(int(row["left_frame_index"])),
                    right_reader.read(int(row["right_frame_index"])),
                )
        finally:
            left_reader.close()
            right_reader.close()


class RectifiedStereoDataset:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        if self.manifest.get("dataset_type") != "rectified_stereo":
            raise ValueError(f"not a rectified stereo dataset: {root}")
        self.image_size = tuple(self.manifest["image_size"])
        self.pairs = read_csv(self.root / "stereo_pairs.csv")
        if not self.pairs:
            raise ValueError("rectified dataset contains no stereo pairs")
        with np.load(self.root / "rectification.npz") as data:
            self.rectification = {
                "image_size": self.image_size,
                "r1": np.asarray(data["R1"], dtype=np.float64),
                "r2": np.asarray(data["R2"], dtype=np.float64),
                "p1": np.asarray(data["P1"], dtype=np.float64),
                "p2": np.asarray(data["P2"], dtype=np.float64),
                "q": np.asarray(data["Q"], dtype=np.float64),
                "calibration_serial": str(self.manifest.get("calibration_hash", "normalized-dataset")),
            }

    def __iter__(self):
        left_reader = SequentialVideoReader(self.root / "left.mkv", self.image_size)
        right_reader = SequentialVideoReader(self.root / "right.mkv", self.image_size)
        try:
            for row in self.pairs:
                pair_index = int(row["pair_index"])
                yield {
                    "pair_index": pair_index,
                    "left_index": int(row["left_frame_index"]),
                    "right_index": int(row["right_frame_index"]),
                    "left_timestamp_us": int(row["left_timestamp_ns"]) // 1000,
                    "right_timestamp_us": int(row["right_timestamp_ns"]) // 1000,
                    "left_image": left_reader.read(pair_index),
                    "right_image": right_reader.read(pair_index),
                }
        finally:
            left_reader.close()
            right_reader.close()
