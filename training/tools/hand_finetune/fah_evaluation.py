"""Score a standalone FAH-S checkpoint on labeled native KB frames and cached detections."""
import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager, nullcontext
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
import torch

from .crop_geometry import crop_camera
from .fah_experiments import SELECTION_METRIC
from .fah_views import predict_views


TIPS = [4, 8, 12, 16, 20]  # OpenPose order: thumb, index, middle, ring, pinky.
SPATIAL_METRICS = ("wrist_relative_mpjpe_mm", "wrist_error_mm", "wrist_along_ray_signed_mm",
                   "wrist_along_ray_abs_mm", "wrist_perpendicular_mm", "fitted_mesh_mpvpe_mm")
OUTCOMES = ("scored", "scored_2d", "invalid_projection", "missed", "invalid_proposal", "unknown_side",
            "wrong_side", "ambiguous", "duplicate_candidates")
MIN_IOU = 0.5


def _finite(value, shape, name):
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite array of shape {shape}, got {array.shape}")
    return array


def _native_observations(prediction):
    if prediction is None:
        raise ValueError("FAH returned no fitted geometry for a supplied crop")
    uv = np.asarray(prediction["keypoints_2d"], dtype=np.float64)
    if uv.shape != (21, 2):
        raise ValueError(f"Expected 21 predicted native pixel pairs, got {uv.shape}")
    valid = np.isfinite(uv).all(axis=1)
    if not np.isnan(uv[~valid]).all():
        raise ValueError("An undefined native pixel must have two NaN coordinates")
    return uv, valid


def hand_errors(truth, native_pixels, prediction):
    """Native 2D observations, placed joints and fitted MANO vertices, without alignment to labels."""
    joints = _finite(truth["joints"], (21, 3), "label joints")
    vertices = _finite(truth["vertices"], (778, 3), "label mesh")
    pixels = _finite(native_pixels, (21, 2), "label KB projection")
    uv, valid = _native_observations(prediction)
    xyz = _finite(prediction["keypoints_3d"], (21, 3), "predicted joints")
    mesh = _finite(prediction["vertices_mano"], (778, 3), "fitted mesh")
    distance = np.linalg.norm(joints[0])
    if distance < 1e-9:
        raise ValueError("GT wrist lies at the camera centre, so its viewing ray is undefined")
    ray = joints[0] / distance
    wrist_mm = 1000 * (xyz[0] - joints[0])
    along = float(wrist_mm @ ray)
    pixels_error = np.full(21, np.nan)
    pixels_error[valid] = np.linalg.norm(uv[valid] - pixels[valid], axis=1)
    relative = (xyz - xyz[0]) - (joints - joints[0])
    tip_errors = pixels_error[TIPS][valid[TIPS]]
    metrics = dict(joint_2d_px=float(pixels_error[valid].mean()) if valid.any() else None,
                   fingertip_2d_px=float(tip_errors.mean()) if tip_errors.size else None,
                   wrist_relative_mpjpe_mm=1000 * np.linalg.norm(relative, axis=1).mean(),
                   wrist_error_mm=np.linalg.norm(wrist_mm), wrist_along_ray_signed_mm=along,
                   wrist_along_ray_abs_mm=abs(along), wrist_perpendicular_mm=np.linalg.norm(wrist_mm - along * ray),
                   fitted_mesh_mpvpe_mm=1000 * np.linalg.norm(mesh - vertices, axis=1).mean())
    if not all(np.isfinite(v) for v in metrics.values() if v is not None):
        raise FloatingPointError("Nonfinite native hand metric")
    return metrics, pixels_error


def field_region(box, image):
    """Central = GT box centre inside the middle 50% of both native image axes, endpoints included."""
    h, w = image.shape[:2]
    box = np.asarray(box)
    center = (box[:2] + box[2:]) / (2 * np.array([w, h]))
    return "central" if np.all((center >= 0.25) & (center <= 0.75)) else "edge"


class _Summary:
    def __init__(self):
        self.counts = Counter(dict.fromkeys(("labeled", *OUTCOMES), 0))
        self.sums = dict.fromkeys(SPATIAL_METRICS, 0.0)
        self.pixels = np.zeros(21, dtype=np.float64)
        self.pixel_counts = np.zeros(21, dtype=np.int64)
        self.ious = []
        self.scored_correct_side = 0

    def record(self, errors, flags, iou):
        self.counts["labeled"] += 1
        self.counts.update(flags)
        if iou is not None:
            self.ious.append(float(iou))
        if errors is not None:
            self.counts["scored"] += 1
            self.scored_correct_side += int("wrong_side" not in flags)
            metrics, pixels = errors
            for k in self.sums:
                self.sums[k] += metrics[k]
            valid = np.isfinite(pixels)
            self.counts["scored_2d" if valid.all() else "invalid_projection"] += 1
            self.pixels[valid] += pixels[valid]
            self.pixel_counts += valid

    def report(self):
        count = self.counts["scored"]
        total = self.counts["labeled"]
        n_pixels = int(self.pixel_counts.sum())
        n_tips = int(self.pixel_counts[TIPS].sum())
        metrics = dict(joint_2d_px=float(self.pixels.sum() / n_pixels) if n_pixels else None,
                       fingertip_2d_px=float(self.pixels[TIPS].sum() / n_tips) if n_tips else None)
        metrics.update({k: float(v / count) if count else None for k, v in self.sums.items()})
        return dict(counts=dict(self.counts), coverage=count / total if total else None,
                    coverage_2d=self.counts["scored_2d"] / total if total else None,
                    box_recall=(total - self.counts["missed"]) / total if total else None,
                    correct_side_coverage=self.scored_correct_side / total if total else None,
                    metrics=metrics, metric_denominators=dict(joint_2d_px=n_pixels, fingertip_2d_px=n_tips,
                                                               **dict.fromkeys(self.sums, count)),
                    joint_2d_px_by_index=[float(s / n) if n else None for s, n in zip(self.pixels, self.pixel_counts)],
                    joint_2d_count_by_index=self.pixel_counts.tolist(),
                    matched_iou_median=float(np.median(self.ious)) if self.ious else None)


class _Arm:
    def __init__(self):
        self.all = _Summary()
        self.cameras = defaultdict(_Summary)
        self.regions = {name: _Summary() for name in ("central", "edge")}
        self.camera_regions = defaultdict(lambda: {name: _Summary() for name in self.regions})
        self.mesh_ids = set()

    def record(self, camera, region, hand_id, errors, flags=(), iou=None):
        if errors is not None:
            if hand_id in self.mesh_ids:
                raise ValueError("Duplicate labeled hand identity in mesh cohort")
            self.mesh_ids.add(hand_id)
        for group in (self.all, self.cameras[camera], self.regions[region], self.camera_regions[camera][region]):
            group.record(errors, flags, iou)

    def report(self):
        return dict(**self.all.report(), mesh_cohort_sha256=hashlib.sha256(b"".join(sorted(self.mesh_ids))).hexdigest(),
                    by_camera={k: v.report() for k, v in self.cameras.items()},
                    by_region={k: v.report() for k, v in self.regions.items()},
                    by_camera_region={c: {k: v.report() for k, v in rs.items()} for c, rs in self.camera_regions.items()})


def _predict(method, image, boxes, scores, sides, K, distortion):
    """Only image, proposals, handedness and calibration cross the FAH inference boundary."""
    if not boxes:
        return []
    hands, views = predict_views(method.est, image, boxes, scores, [(bool(s),) for s in sides], K, distortion)
    if len(hands) != len(sides) or [h["is_right"] for h in hands] != list(sides):
        raise RuntimeError("FAH changed the number or handedness of input proposals")
    # FAH's pinned package indexes OpenCV's PnP translation as a column.
    # Some OpenCV builds return the same three values as a flat array.
    solve_pnp = cv2.solvePnP

    def column_translation(*args, **kwargs):
        ok, rotation, translation = solve_pnp(*args, **kwargs)
        return ok, rotation, translation.reshape(3, 1) if ok else translation

    with patch.object(cv2, "solvePnP", column_translation):
        method._fit_meshes(hands)
    if not all(h.get("mesh_param") for h in hands):
        raise RuntimeError("FAH did not fit every requested MANO mesh")
    predictions = []
    for hand, view in zip(hands, views, strict=True):
        uv = _finite(hand.get("keypoints_2d_mano", hand["keypoints_2d"]), (21, 2), "FAH 2D observations")
        xyz = _finite(hand.get("keypoints_3d_mano", hand["keypoints_3d"]), (21, 3), "FAH placed joints")
        mesh = _finite(hand["vertices_mano"], (778, 3), "FAH fitted mesh")
        # Rotation into the physical camera is defined even when a direct UV ray has no calibrated native pixel.
        predictions.append(dict(is_right=hand["is_right"], keypoints_2d=view.native_pixels(uv).astype(np.float32),
                                keypoints_3d=(xyz @ view.rotation).astype(np.float32),
                                vertices_mano=(mesh @ view.rotation).astype(np.float32)))
    return predictions


def _proposal_predictions(frame, proposals, predict):
    """Validate each crop using the same KB geometry as inference before batching known sides."""
    K, distortion = frame["K"], frame["distortion"]
    h, w = frame["image"].shape[:2]
    invalid_crops = set()
    for i, proposal in enumerate(proposals):
        box = proposal.bbox
        if np.any(box[2:] <= 0) or np.any(box[:2] >= [w, h]):
            invalid_crops.add(i)
            continue
        try:
            # The invertible ray domain and forward-hemisphere check are independent of resolution.
            crop_camera(proposal.bbox, K, distortion, size=320)
        except ValueError:
            invalid_crops.add(i)
    known = [i for i, p in enumerate(proposals) if p.side is not None and i not in invalid_crops]
    predictions = predict(frame["image"], [proposals[i].bbox for i in known], [proposals[i].score for i in known],
                          [proposals[i].side for i in known], K, distortion)
    predicted = dict(zip(known, predictions, strict=True))
    invalid_projection = {i for i, output in predicted.items() if not _native_observations(output)[1].all()}
    return predicted, invalid_crops, invalid_projection


def _proposal_matches(boxes, proposal_boxes, invalid_crops):
    """Assign usable boxes first, then identify invalid crops that explain the remaining labels."""
    from .boxes import match_boxes

    usable = [i for i in range(len(proposal_boxes)) if i not in invalid_crops]
    matches = {g: (usable[p], iou) for g, p, iou in match_boxes(boxes, [proposal_boxes[i] for i in usable], MIN_IOU)}
    if invalid_crops:
        remaining = [g for g in range(len(boxes)) if g not in matches]
        invalid = sorted(invalid_crops)
        for g, p, iou in match_boxes([boxes[i] for i in remaining], [proposal_boxes[i] for i in invalid], MIN_IOU):
            matches[remaining[g]] = (invalid[p], iou)
    return matches


def score_frames(frames, cache, predict):
    """Run the two protocols independently and associate detections using only box geometry."""
    from .boxes import box_iou_matrix
    from .data import project_native

    gt, detected = _Arm(), _Arm()
    proposals_total = Counter(dict.fromkeys(("total", "unknown_side", "unmatched", "shared_candidates",
                                            "invalid", "invalid_crop", "invalid_projection", "invalid_unmatched"), 0))
    for index in range(len(frames)):
        frame = frames[index]
        key, image, hands = frame["key"], frame["image"], frame["hands"]
        boxes = [h["bbox"] for h in hands]
        sides = [h["side"] for h in hands]
        proposals = cache[key]  # Only an explicit empty proposal list represents a detector miss.
        proposal_boxes = [p.bbox for p in proposals]
        try:
            gt_predictions = predict(image, boxes, [1.] * len(boxes), sides, frame["K"], frame["distortion"])
            predicted, invalid_crops, invalid_projection = _proposal_predictions(frame, proposals, predict)
            invalid = invalid_crops | invalid_projection
            matches = _proposal_matches(boxes, proposal_boxes, invalid_crops)
            candidates = box_iou_matrix(boxes, proposal_boxes) >= MIN_IOU
            shared = candidates.sum(axis=0) > 1  # one proposal overlaps multiple labeled hands
            matched_ids = {p for p, _ in matches.values()}
            proposals_total.update(total=len(proposals), unknown_side=sum(p.side is None for p in proposals),
                                   unmatched=len(proposals) - len(matches), shared_candidates=int(shared.sum()),
                                   invalid=len(invalid), invalid_crop=len(invalid_crops),
                                   invalid_projection=len(invalid_projection), invalid_unmatched=len(invalid - matched_ids))
            for g, (hand, gt_pred) in enumerate(zip(hands, gt_predictions, strict=True)):
                pixels = project_native(hand["joints"], frame["K"], frame["distortion"])
                region = field_region(hand["bbox"], image)
                identity = [key.recording_id, key.camera, key.image, key.sync_index, hand["row"]["label"], int(hand["side"])]
                hand_id = hashlib.sha256(json.dumps(identity, ensure_ascii=False, separators=(",", ":")).encode()).digest()
                gt.record(key.camera, region, hand_id, hand_errors(hand, pixels, gt_pred))
                flags = []
                if candidates[g].sum() > 1:
                    flags.append("duplicate_candidates")
                if flags or np.any(candidates[g] & shared):
                    flags.append("ambiguous")
                p, iou = matches.get(g, (None, None))
                errors = None
                if p is None:
                    flags.append("missed")
                else:
                    if proposals[p].side is None:
                        flags.append("unknown_side")
                    elif proposals[p].side != hand["side"]:
                        flags.append("wrong_side")
                    if p in invalid_crops:
                        flags.append("invalid_proposal")
                    elif proposals[p].side is not None:
                        errors = hand_errors(hand, pixels, predicted[p])
                detected.record(key.camera, region, hand_id, errors, flags, iou)
        except (ValueError, KeyError, FloatingPointError, RuntimeError) as exc:
            raise type(exc)(f"{key}: {exc}") from exc
    if not gt.all.counts["labeled"]:
        raise ValueError("No labeled hands on the requested split")
    deployment = detected.report()
    return dict(gt=gt.report(), detected=deployment, proposals=dict(proposals_total),
                selection_metric=SELECTION_METRIC, selection_metric_mm=deployment["metrics"][SELECTION_METRIC],
                frames=len(frames), min_iou=MIN_IOU, fingertip_indices=TIPS,
                region_rule="central when the labeled box centre lies in [25%,75%] of both image dimensions",
                coverage_rule="scored fitted meshes / labeled hands; known-side matches need a usable input crop",
                coverage_2d_rule="hands with 21 finite native pixel predictions / labeled hands",
                mesh_cohort_rule="SHA256 of sorted concatenated SHA256 bytes of compact UTF-8 JSON arrays "
                                 "[recording_id,camera,image,sync_index,label,physical_side]",
                matching_rule="one-to-one IoU assignment on usable boxes, then on remaining labels and invalid crops",
                ambiguity_rule="flag a label with multiple candidate proposals, or a candidate shared by multiple labels",
                wrist_ray="unit camera-to-GT-wrist vector; positive signed error means predicted farther away")


@contextmanager
def _deployment_method(checkpoint, upstream, release_weights, mano_dir, sam3d, device, batch_size):
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested for FAH validation but unavailable")
    root = Path(upstream).resolve() if upstream else None
    if root and not (root / "fah/__init__.py").is_file():
        raise FileNotFoundError(f"Missing FAH package in {root}")
    if root and str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import fah
    from fah import api
    if root and Path(fah.__file__).resolve().parent.parent != root:
        raise RuntimeError(f"Already imported FAH from {fah.__file__}; requested {root}")
    weights = Path(release_weights or (root / "weights" if root else api.DEFAULT_WEIGHTS)).resolve()
    mano = Path(mano_dir or (root / "data/mano" if root else os.environ.get("MANO_DIR", Path(api.FAH_HOME) / "mano"))).resolve()
    sam = Path(sam3d or (root / "data/sam-3d-body-dinov3" if root else api.DEFAULT_SAM3D)).resolve()
    checkpoint = Path(checkpoint).resolve(strict=True)
    config_path = weights / "fah-s/config.json"
    config = json.loads(config_path.read_text())
    for path in (config_path.parent / config["mesh_fitter"], mano / "MANO_RIGHT.pkl", mano / "MANO_LEFT.pkl",
                 sam / "model_config.yaml", sam / "assets/mhr_model.pt"):
        if not path.is_file():
            raise FileNotFoundError(f"Missing FAH deployment asset {path}")
    cuda = device.type == "cuda"
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if cuda else []
    device_scope = torch.cuda.device(devices[0]) if cuda else nullcontext()
    # The upstream loader reads full encoder/head tensors from finetune_ckpt, including the V4 adapter.
    # Force its torch.load calls to use the tensor-only loader. MANO and SAM-3D are trusted release assets.
    settings = {"MANO_DIR": str(mano), "TORCH_FORCE_WEIGHTS_ONLY_LOAD": "1"}
    precision = torch.get_float32_matmul_precision()
    cudnn_tf32 = torch.backends.cudnn.allow_tf32
    with device_scope, patch.dict(os.environ, settings), torch.random.fork_rng(devices=devices):
        try:
            torch.set_float32_matmul_precision("highest")
            torch.backends.cudnn.allow_tf32 = False
            with patch("bench.methods.fah.build_hand_detector", return_value=SimpleNamespace(names={0: "left", 1: "right"})):
                model = fah.FAH("fah-s", weights=str(weights), sam3d=str(sam), device=str(device), mode="image",
                                finetune_ckpt=str(checkpoint), mesh="mano", tips=False, parts=False, verify=False,
                                fast_det=False, compile=False, parts_in_mask=False, dtype="float32",
                                cuda_graph=False, tf32=False, gpu_frame=False, fast_batch=False)
            estimator = model.method.est
            if (getattr(estimator.student, "name", None) != "dinov3_vits16"
                    or not getattr(estimator.model, "kp_only", False)):
                raise ValueError("Expected a standalone decoder-free FAH-S student checkpoint")
            estimator.max_batch_size = batch_size
            if not model.method.mesh_fit:
                raise ValueError("FAH release configuration must enable the native MANO fitter")
            yield model.method, dict(checkpoint=str(checkpoint), release_weights=str(weights), mano_dir=str(mano),
                                     sam3d=str(sam), backend=str(Path(fah.__file__).resolve()), device=str(device),
                                     dtype="float32", tf32=False, mesh="FAH fitted MANO778 in the physical camera")
        finally:
            torch.set_float32_matmul_precision(precision)
            torch.backends.cudnn.allow_tf32 = cudnn_tf32


def _checked_split(dataset, split, validated_protocol):
    from .fah_experiments import split_protocol

    if split not in ("train", "val", "test"):
        raise ValueError(f"Unknown split {split!r}")
    checked = split_protocol(dataset, fingerprint_inputs=(split,))
    fingerprints = checked["manifests_sha256"]
    if split not in fingerprints:
        raise FileNotFoundError(f"No {split} split declared in {dataset}")
    contents = checked["input_contents_sha256"][split]
    if split == "val" and validated_protocol is not None:
        if validated_protocol.get("dataset") != str(dataset) or validated_protocol.get("eval_split") != split:
            raise ValueError("Prevalidated run does not identify this dataset and evaluation split")
        if (fingerprints != validated_protocol["manifests_sha256"]
                or contents != validated_protocol["input_contents_sha256"].get(split)):
            raise ValueError("Split manifests or validation inputs changed after training preflight")
    return {"holdout": checked["holdout"], "time_field": checked["time_field"],
            "split_manifest_sha256": fingerprints[split], "scored_inputs_sha256": contents}


def _file_identity(path):
    info = path.stat()
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def evaluate_checkpoint(*, checkpoint, dataset, split, detector_cache, upstream=None, release_weights=None,
                        mano_dir=None, sam3d=None, device="cuda", batch_size=4, validated_protocol=None):
    """Evaluate a release/export model.pt with the exact split and proposal cache supplied by the caller."""
    from .boxes import load_proposal_cache
    from .data import EvaluationFrames
    from .fah_experiments import content_digest, deployment_fingerprints, digest

    dataset = Path(dataset).resolve()
    holdout = _checked_split(dataset, split, validated_protocol)
    detector_cache, checkpoint = Path(detector_cache).resolve(), Path(checkpoint).resolve()
    cache_file, checkpoint_file = _file_identity(detector_cache), _file_identity(checkpoint)
    cache_sha256, checkpoint_sha256 = content_digest(detector_cache), digest(checkpoint)
    if split == "val" and validated_protocol is not None and cache_sha256 != validated_protocol["cache_sha256"]:
        raise ValueError("Detector proposal cache changed after training preflight")
    frames = EvaluationFrames(dataset, split)
    cache = load_proposal_cache(detector_cache, frames.keys)
    with _deployment_method(checkpoint, upstream, release_weights, mano_dir, sam3d, device, batch_size) as (method, assets):
        if _file_identity(checkpoint) != checkpoint_file or _file_identity(detector_cache) != cache_file:
            raise ValueError("Checkpoint or detector cache changed while loading")
        deployed = deployment_fingerprints(assets)
        if (split == "val" and validated_protocol is not None and "deployment_sha256" in validated_protocol
                and deployed != validated_protocol["deployment_sha256"]):
            raise ValueError("Deployment assets or FAH backend changed since release validation")

        def predict(image, boxes, scores, sides, K, distortion):
            method.reset()
            return _predict(method, image, boxes, scores, sides, K, distortion)

        result = score_frames(frames, cache, predict)
        # Frames and the MANO fitter load lazily during scoring. Check their recorded content at both boundaries.
        if holdout != _checked_split(dataset, split, validated_protocol) or deployed != deployment_fingerprints(assets):
            raise ValueError("Scored split or deployment assets changed during evaluation")
    return dict(**result, split=split, dataset=str(dataset), detector_cache=str(detector_cache),
                checkpoint_sha256=checkpoint_sha256, cache_sha256=cache_sha256,
                deployment_sha256=deployed, **holdout, **assets)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "dataset", "detector-cache", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], required=True)
    for name in ("upstream", "release-weights", "mano-dir", "sam3d"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    options = vars(parser.parse_args())
    output = options.pop("output")
    if output.exists():
        parser.error(f"Output file already exists: {output}")
    result = evaluate_checkpoint(**options)
    encoded = json.dumps(result, indent=2, allow_nan=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        stream.write(encoded + "\n")
    print(f"Saved {output}; deployment coverage {result['detected']['coverage']:.1%}, "
          f"wrist-relative MPJPE {result['selection_metric_mm']} mm")


if __name__ == "__main__":
    main()
