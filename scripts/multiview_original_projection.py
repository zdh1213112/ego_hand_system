"""Display independent-pinhole predictions on their original KB/DS images.

Tracking, smoothing and trust decisions belong to the inference image domain.
This adapter only maps their selected display results; it never edits fusion.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from camera_models import project_points, unproject_points
from ego_data.calibration import CameraCalibration
from ego_data.dataset import read_csv


def inside_image(pixels: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return (np.isfinite(pixels).all(axis=-1)
            & (pixels[..., 0] >= 0) & (pixels[..., 0] <= size[0] - 1)
            & (pixels[..., 1] >= 0) & (pixels[..., 1] <= size[1] - 1))


def pixels_to_original(
    pixels: np.ndarray, pinhole: CameraCalibration, original: CameraCalibration,
) -> np.ndarray:
    """Map only pixels visible in both images, without reviving cropped joints."""
    pixels = np.asarray(pixels, dtype=np.float64)
    rays, valid = unproject_points(pinhole, pixels)
    mapped, projectable = project_points(original, rays)
    valid &= (projectable & inside_image(pixels, pinhole.image_size)
              & inside_image(mapped, original.image_size))
    mapped[~valid] = np.nan
    return mapped


def box_to_original(box, pinhole: CameraCalibration, original: CameraCalibration) -> list:
    """Axis-aligned envelope of the warped, clipped box perimeter.

    Four transformed corners miss extrema on curved fisheye boundaries. Sample
    every edge at intervals of at most 4 input pixels before taking its bounds.
    """
    box = np.asarray(box, dtype=np.float64)
    if box.shape != (4,) or not np.isfinite(box).all():
        return []
    x0, y0, x1, y1 = box
    width, height = pinhole.image_size
    x0, x1 = max(0., x0), min(width - 1., x1)
    y0, y1 = max(0., y0), min(height - 1., y1)
    if x0 >= x1 or y0 >= y1:
        return []
    xs = np.linspace(x0, x1, max(3, int(np.ceil((x1 - x0) / 4)) + 1))
    ys = np.linspace(y0, y1, max(3, int(np.ceil((y1 - y0) / 4)) + 1))
    border = np.concatenate((
        np.column_stack((xs, np.full_like(xs, y0))),
        np.column_stack((xs, np.full_like(xs, y1))),
        np.column_stack((np.full_like(ys, x0), ys)),
        np.column_stack((np.full_like(ys, x1), ys)),
    ))
    mapped = pixels_to_original(border, pinhole, original)
    valid = mapped[np.isfinite(mapped).all(axis=1)]
    if len(valid) < 2:
        return []
    lower, upper = valid.min(axis=0), valid.max(axis=0)
    if np.any(upper <= lower):
        return []
    return np.concatenate((lower, upper)).tolist()


class OriginalVideoProjection:
    def __init__(self, dataset: Path, original: Path, cameras: tuple[str, ...]):
        self.root = original.resolve()
        source = json.loads((dataset / "manifest.json").read_text())
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        if (source.get("dataset_type") != "normalized_multiview"
                or self.manifest.get("dataset_type") != "normalized_multiview"
                or source.get("undistortion", {}).get("config", {}).get("options", {}).get("mode") != "independent_pinhole"
                or self.manifest.get("undistortion")):
            raise ValueError("original display requires an independent-pinhole dataset and its raw normalized dataset")
        for field in ("source", "camera_ids", "reference_camera"):
            if source.get(field) != self.manifest.get(field):
                raise ValueError(f"original display: dataset {field} differs")
        if read_csv(dataset / "multiview_frames.csv") != read_csv(self.root / "multiview_frames.csv"):
            raise ValueError("original display: synchronized frame indices/timestamps differ")
        self.pinhole, self.original = {}, {}
        for camera in cameras:
            pinhole = CameraCalibration.load(dataset / "calibration" / f"{camera}.json")
            raw = CameraCalibration.load(self.root / "calibration" / f"{camera}.json")
            saved = CameraCalibration.load(dataset / "source_calibration" / f"{camera}.json")
            if (pinhole.model != "PINHOLE" or raw.model not in ("KB", "DS")
                    or pinhole.camera_id != camera or raw.camera_id != camera
                    or raw.to_dict() != saved.to_dict()
                    or not np.array_equal(pinhole.T_base_camera, raw.T_base_camera)
                    or pinhole.image_size != tuple(source["image_size"])
                    or raw.image_size != tuple(self.manifest["image_size"])):
                raise ValueError(f"original display: {camera} calibration/optical frame differs")
            video = self.root / "cameras" / camera / self.manifest["storage"]["video_filename"]
            if not video.is_file():
                raise FileNotFoundError(video)
            self.pinhole[camera], self.original[camera] = pinhole, raw

    def view(self, camera: str, view: dict) -> dict:
        result = dict(view)
        pinhole, raw = self.pinhole[camera], self.original[camera]
        for key in ("joints_2d", "wilor_joints_2d"):
            points = np.asarray(view.get(key, []), dtype=float)
            if points.shape == (21, 2):
                result[key] = pixels_to_original(points, pinhole, raw).tolist()
        result["bbox_xyxy"] = box_to_original(view.get("bbox_xyxy", []), pinhole, raw)
        return result

    def native_view(self, camera: str, view: dict | None) -> dict | None:
        if view is None:
            return None
        result = self.view(camera, view)
        points = np.asarray(result.get("wilor_joints_2d", []), dtype=float)
        if points.shape != (21, 2):
            points = np.asarray(result.get("joints_2d", []), dtype=float)
        # Match the existing native-2D drawing threshold, after FOV masking.
        if points.shape != (21, 2) or np.isfinite(points).all(axis=1).sum() < 5:
            return None
        return result

    def fused_points(self, camera: str, points_base: np.ndarray, inference_pixels: np.ndarray) -> np.ndarray:
        raw = self.original[camera]
        points_camera = (np.asarray(points_base) - raw.T_base_camera[:3, 3]) @ raw.T_base_camera[:3, :3]
        mapped, valid = project_points(raw, points_camera)
        valid &= (inside_image(inference_pixels, self.pinhole[camera].image_size)
                  & inside_image(mapped, raw.image_size))
        mapped[~valid] = np.nan
        return mapped
