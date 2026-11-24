"""Native-pixel detector proposals shared by FAH training and deployment evaluation.

One schema_version=1 JSONL record per FrameKey, including frames with proposals=[].
Each proposal has bbox=[x1,y1,x2,y2], score in [0,1] and side=0 (left), 1 (right)
or null (unknown). Preserve detector output order and all post-NMS proposals.
"""
from dataclasses import dataclass
import json
from pathlib import Path
from typing import NamedTuple

import numpy as np


class FrameKey(NamedTuple):
    recording_id: str
    image: str
    camera: str
    sync_index: int

    @classmethod
    def from_row(cls, row):
        strings = [row.get(k) for k in ("recording_id", "image", "camera")]
        sync = row.get("sync_index")
        if any(not isinstance(s, str) or not s.strip() for s in strings):
            raise ValueError("Frame requires non-empty recording_id, image and camera strings")
        if type(sync) is not int or sync < 0:
            raise ValueError("sync_index must be a non-negative integer")
        return cls(*strings, sync)


class Proposal(NamedTuple):
    bbox: np.ndarray
    score: float
    side: int | None


def box_array(values):
    """Validate a collection of native pixel xyxy boxes, including an empty list."""
    boxes = np.asarray(values)
    if boxes.shape == (0,):
        return np.empty((0, 4), dtype=np.float64)
    if boxes.ndim != 2 or boxes.shape[1] != 4 or boxes.dtype.kind not in "iuf":
        raise ValueError("Expected numeric xyxy boxes of shape (N,4)")
    boxes = boxes.astype(np.float64, copy=False)
    if not np.isfinite(boxes).all() or np.any(boxes[:, 2:] <= boxes[:, :2]):
        raise ValueError("Expected finite xyxy boxes of positive width and height")
    return boxes


def _proposal(raw):
    if not isinstance(raw, dict) or not {"bbox", "score", "side"} <= raw.keys():
        raise ValueError("Each proposal requires bbox, score and side (0, 1 or null)")
    if not isinstance(raw["bbox"], list) or any(type(x) not in (int, float) for x in raw["bbox"]):
        raise ValueError("Proposal bbox must be an array of numeric xyxy coordinates")
    box = box_array([raw["bbox"]])[0]
    score, side = raw["score"], raw["side"]
    if type(score) not in (int, float) or not 0 <= score <= 1:
        raise ValueError("score must be a number in [0,1]")
    if side is not None and (type(side) is not int or side not in (0, 1)):
        raise ValueError("side must be 0, 1 or null")
    return Proposal(box, float(score), side)


def _unique_fields(pairs):
    result = dict(pairs)
    if len(result) != len(pairs):
        raise ValueError("Duplicate JSON field")
    return result


def load_proposal_cache(path, required_keys):
    """Validate a complete cache. Missing required frames are errors, distinct from misses."""
    result = {}
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            try:
                row = json.loads(line, object_pairs_hook=_unique_fields)
                if (not isinstance(row, dict) or type(row.get("schema_version")) is not int
                        or row["schema_version"] != 1):
                    raise ValueError("Expected detector cache schema_version=1")
                key = FrameKey.from_row(row)
                if key in result:
                    raise ValueError(f"Duplicate frame {key}")
                if not isinstance(row.get("proposals"), list):
                    raise ValueError("Expected proposals list, use [] for a detector miss")
                result[key] = [_proposal(proposal) for proposal in row["proposals"]]
            except (ValueError, TypeError, OverflowError) as exc:
                raise ValueError(f"{path}:{number}: {exc}") from exc
    missing = set(required_keys).difference(result)
    if missing:
        raise ValueError(f"Cache missing {len(missing)} required frames, including {min(missing)}")
    return result


def box_iou_matrix(gt_boxes, proposal_boxes):
    gt, proposal = box_array(gt_boxes), box_array(proposal_boxes)
    overlap = (np.minimum(gt[:, None, 2:], proposal[:, 2:])
               - np.maximum(gt[:, None, :2], proposal[:, :2])).clip(min=0).prod(axis=2)
    gt_area = (gt[:, 2:] - gt[:, :2]).prod(axis=1)
    proposal_area = (proposal[:, 2:] - proposal[:, :2]).prod(axis=1)
    return overlap / (gt_area[:, None] + proposal_area - overlap)


def match_boxes(gt_boxes, proposal_boxes, min_iou=0.5):
    """Maximize number of pairs over threshold, then total IoU, independent of hand side."""
    return match_box_ious(box_iou_matrix(gt_boxes, proposal_boxes), min_iou)


def match_box_ious(iou, min_iou):
    """Assign a GT-by-proposal IoU matrix. Zero entries exclude unusable pairs."""
    from scipy.optimize import linear_sum_assignment

    if not 0 < min_iou <= 1:
        raise ValueError("min_iou must be in (0,1]")
    eligible = iou >= min_iou
    weight = np.where(eligible, min(iou.shape) + 1 + iou, 0)
    gt, det = linear_sum_assignment(weight, maximize=True)
    return [(int(i), int(j), float(iou[i, j])) for i, j in zip(gt, det) if eligible[i, j]]


@dataclass(frozen=True)
class BoxAugmentation:
    """Probability of taking a matched detector box, or a sampled offset on a miss.

    Explicit offsets (an N×4 array or JSON file) override the empirical train-only
    fallback pool. Rows store normalized centre shift and log width/height ratios.
    Only locally rectifiable matches with valid training targets enter the pool.
    A miss tries up to four distinct offsets on the receiving hand's calibration.
    If those crops cannot be used, that sample receives its validated GT crop.
    """
    cache: str | Path | None = None
    offset_samples: list | np.ndarray | str | Path | None = None
    detector_probability: float = 0.5
    min_iou: float = 0.5
    seed: int = 0

    def __post_init__(self):
        if self.cache is None and self.offset_samples is None:
            raise ValueError("Box augmentation requires a detector cache or explicit offset samples")
        if not 0 < self.detector_probability <= 1 or not 0 < self.min_iou <= 1:
            raise ValueError("detector_probability and min_iou must be in (0,1]")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")

    def offsets(self, measured):
        samples = self.offset_samples
        if isinstance(samples, (str, Path)):
            samples = json.loads(Path(samples).read_text(encoding="utf-8"))
        rows = np.asarray(measured if samples is None else samples)
        if (rows.ndim != 2 or rows.shape[0] == 0 or rows.shape[1] != 4
                or rows.dtype.kind not in "iuf" or not np.isfinite(rows).all()):
            raise ValueError("No valid offsets: supply finite non-empty N×4 samples or usable matched train boxes")
        return rows.astype(np.float64)


def measure_offset(gt, detection):
    extent, det_extent = gt[2:] - gt[:2], detection[2:] - detection[:2]
    shift = ((detection[:2] - gt[:2]) + (det_extent - extent) / 2) / extent
    return np.r_[shift, np.log(det_extent / extent)]


def apply_offset(gt, offset):
    extent = gt[2:] - gt[:2]
    with np.errstate(over="ignore", invalid="ignore"):
        centre = gt[:2] + (0.5 + offset[:2]) * extent
        half = extent * np.exp(offset[2:]) / 2
        shifted = np.r_[centre - half, centre + half]
    return box_array([shifted])[0]


if __name__ == "__main__":
    from .cache_boxes import main
    main()
