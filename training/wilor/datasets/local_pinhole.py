"""Export native KB hand boxes into WiLoR's per-box perspective cameras."""
import fcntl
import hashlib
import json
import tempfile
from functools import partial
from pathlib import Path

import cv2
import numpy as np

from tools.hand_finetune.boxes import Proposal, box_iou_matrix, load_proposal_cache, match_box_ious
from tools.hand_finetune.crop_geometry import PerspectiveCrop
from tools.hand_finetune.data import FrameManifest
from wilor.datasets.glove_validation import build_dataset_profile, sha256_file
from wilor.models import resolve_mano_model_path


INPUT_VIEW = "local_pinhole"
SIZE = 256


def local_camera(box, K, distortion):
    view = PerspectiveCrop.from_box(box, K, distortion, SIZE)
    # WiLoR feeds only the central 192 columns to the ViT.
    view.K[0, 0] *= 192 / SIZE
    view.K[1, 1] *= 192 / SIZE
    return view


def bbox_pinhole_camera(box, K, distortion, *, rescale_factor):
    # Match the float32 box and intrinsics read by the reference WiLoR dataset.
    return PerspectiveCrop.from_box(np.asarray(box, dtype=np.float32).reshape(4),
                                    np.asarray(K, dtype=np.float32), distortion, SIZE,
                                    padding=rescale_factor, width_factor=SIZE / 192)


def make_local_label(hand, view, image_shape, mano_model):
    """Rotate metric geometry around the optical center, and MANO around its shaped wrist."""
    raw = hand["native_label"]
    side = hand["side"]
    rotation = view.rotation
    absolute = hand["joints"] @ rotation.T
    mesh = hand["vertices"] @ rotation.T
    wrist_offset = np.asarray(raw["joints_3d"], dtype=np.float64)[0]
    translation = absolute[0] - wrist_offset
    vertex_uv = mesh @ view.K.T
    joint_uv = absolute @ view.K.T
    native_uv = view.native_pixels(joint_uv[:, :2] / joint_uv[:, 2:])
    h, w = image_shape[:2]
    visible = (np.isfinite(native_uv).all(axis=1) & (native_uv >= 0).all(axis=1)
               & (native_uv < [w, h]).all(axis=1))
    label = dict(input_view=INPUT_VIEW, image_space="physical_rectified", camera_model="PINHOLE",
                 side=side, K=view.K.astype("f"), trans=translation.astype("f"),
                 joints_3d=(absolute-translation).astype("f"), vertices=(mesh-translation).astype("f"),
                 joints_2d=(vertex_uv[:, :2] / vertex_uv[:, 2:]).astype("f"),
                 native_joint_visible=visible, bbox=np.array([0, 0, SIZE, SIZE], dtype="f"))
    if raw.get("mano") is not None:
        pose = dict(raw["mano"])
        if "global_orient" in pose:
            # The stored root acts on the canonical MANO model before any handedness reflection.
            if bool(side) != (mano_model == "MANO_RIGHT.pkl"):
                mirror = np.diag([-1., 1., 1.])
                rotation = mirror @ rotation @ mirror
            pose["global_orient"] = (rotation @ np.asarray(pose["global_orient"]).reshape(1, 3, 3)).astype("f")
        label["mano"] = pose
    return label


def detect_with_yolo(weights, device, confidence):
    from ultralytics import YOLO

    model = YOLO(str(weights))
    names = {int(i): str(name).lower() for i, name in model.names.items()}
    sides = {"left": 0, "right": 1, "hand": None}
    if any(name not in sides for name in names.values()):
        raise ValueError(f"Expected a hand-only detector; found classes {names}")

    def detect(image):
        result = model.predict(image, device=device, conf=confidence, iou=0.5, verbose=False)[0].boxes
        return [Proposal(box, float(score), sides[names[int(cls)]]) for box, score, cls in zip(
            result.xyxy.cpu().numpy(), result.conf.cpu().numpy(), result.cls.cpu().numpy())]

    return detect


def match_views(hands, proposals, K, distortion, min_iou, *, camera=local_camera):
    scores = box_iou_matrix([hand["bbox"] for hand in hands], [p.bbox for p in proposals])
    views = {}
    for j, proposal in enumerate(proposals):
        if not (scores[:, j] >= min_iou).any():
            continue
        try:
            view = camera(proposal.bbox, K, distortion)
        except ValueError:
            scores[:, j] = 0
            continue
        views[j] = view
        # Reject unusable edges before one-to-one assignment, allowing another proposal to match.
        for i in np.flatnonzero(scores[:, j] >= min_iou):
            forward = view.rotation[2]
            if (hands[i]["joints"] @ forward).min() <= 0 or (hands[i]["vertices"] @ forward).min() <= 0:
                scores[i, j] = 0
    return [(i, j, iou, views[j]) for i, j, iou in match_box_ious(scores, min_iou)]


def recording_group(row, key):
    # Older merged WiLoR manifests reuse source_group (e.g. processed__both) across recordings.
    if row.get("recording_id") or row.get("source_experiment"):
        return str(row.get("recording_id") or row["source_experiment"])
    if row.get("source_dataset_id") is not None:
        return f"{key.recording_id}/dataset-{row['source_dataset_id']}"
    return key.recording_id


def label_views(hands, K, distortion, *, camera=local_camera):
    for index, hand in enumerate(hands):
        try:
            view = camera(hand["bbox"], K, distortion)
            forward = view.rotation[2]
            if (hand["joints"] @ forward).min() <= 0 or (hand["vertices"] @ forward).min() <= 0:
                raise ValueError("Hand geometry is behind its local camera")
        except ValueError as exc:
            raise ValueError(f"Cannot rectify native label bbox for {hand['row']['label']}: {exc}") from exc
        yield index, hand["bbox"], view, {"bbox_source": "native_label"}


def export_split(frames, root, split, all_rows, detector, cache, min_iou, models, *,
                 label_bboxes=False, bbox_pinhole_rescale=None):
    reference_crop = bbox_pinhole_rescale is not None
    camera = local_camera
    image_flags = cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
    if reference_crop:
        if not 0 < bbox_pinhole_rescale < float("inf"):
            raise ValueError("bbox_pinhole_rescale must be positive and finite")
        camera = partial(bbox_pinhole_camera, rescale_factor=bbox_pinhole_rescale)
        image_flags = cv2.IMREAD_COLOR  # Use the reference dataset's image decoding.
    counts = dict(frames=0, labels=0, detections=0, matched=0)
    with (root / f"{split}.jsonl").open("w") as manifest:
        for key, indices in frames.by_frame.items():
            image = frames.read_image(key, flags=image_flags)
            hands, K, distortion = frames.read_frame_labels(indices, include_native_label=True)
            counts["frames"] += 1
            counts["labels"] += len(hands)
            if label_bboxes:
                views = label_views(hands, K, distortion, camera=camera)
            else:
                proposals = cache[key] if cache is not None else detector(image)
                counts["detections"] += len(proposals)
                views = ((i, proposals[j].bbox, view, dict(
                    detection_index=j, detector_side=proposals[j].side,
                    confidence=proposals[j].score, match_iou=iou))
                    for i, j, iou, view in match_views(hands, proposals, K, distortion, min_iou, camera=camera))
            for i, box, view, provenance in views:
                hand = hands[i]
                side = hand["side"]
                label = make_local_label(hand, view, image.shape, models[("left", "right")[side]])
                if reference_crop:
                    label["bbox"] = view.bounds.astype(np.float32)
                    provenance["pinhole_crop"] = "kb_bbox_pinhole"
                index = counts["matched"]
                stem = f"{split}_{index:08d}"
                image_path, label_path = f"images/{stem}.png", f"labels/{stem}.npy"
                if not cv2.imwrite(str(root / image_path), view.rectify(image, check_view_extent=reference_crop)):
                    raise OSError(f"Cannot write {image_path}")
                np.save(root / label_path, label)
                row = dict(hand["row"], index=stem, source_group=recording_group(hand["row"], key), side=side,
                           native_source_group=hand["row"].get("source_group"),
                           image=image_path, label=label_path, input_view=INPUT_VIEW,
                           camera_model="PINHOLE", image_space="physical_rectified", stored_image_space="physical_rectified",
                           native_image=key.image, native_label=hand["row"]["label"],
                           native_bbox_xyxy=box.tolist(), **provenance,
                           R_local_from_native=view.rotation.tolist(), K_local=view.K.tolist(),
                           K_native=K.tolist(), distortion_native=distortion.tolist())
                line = json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
                manifest.write(line)
                all_rows.write(line)
                counts["matched"] += 1
    if not counts["matched"]:
        raise ValueError(f"No labeled detections matched in {split}: {counts}")
    counts["unmatched_labels"] = counts["labels"] - counts["matched"]
    counts["unmatched_detections"] = 0 if label_bboxes else counts["detections"] - counts["matched"]
    return counts


def export_dataset(source, destination, mano_path, *, weights=None, proposal_cache=None,
                   device="cuda:0", confidence=0.3, min_iou=0.5, label_bboxes=False,
                   bbox_pinhole_rescale=None, cache_inputs=None):
    source = Path(source).expanduser().resolve()
    destination = Path(destination).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"Local view output already exists: {destination}. Reuse it without --native-dataset.")
    if sum((weights is not None, proposal_cache is not None, label_bboxes)) != 1:
        raise ValueError("Select exactly one of --label-bboxes, --detector and --detector-cache")
    if not 0 < min_iou <= 1 or not 0 < confidence <= 1:
        raise ValueError("Matching IoU and detector confidence must be in (0,1]")
    frames = {split: FrameManifest(source, split, strict=False) for split in ("train", "val")}
    overlap = set(frames["train"].keys) & set(frames["val"].keys)
    if overlap:
        raise ValueError(f"Native frame present in both train and val: {min(overlap)}")
    cache, detector, detection = None, None, None
    if proposal_cache is not None:
        cache = load_proposal_cache(proposal_cache, set(frames["train"].keys) | set(frames["val"].keys))
        detection = dict(cache=str(proposal_cache), sha256=sha256_file(proposal_cache))
    elif weights is not None:
        if not Path(weights).is_file():
            raise FileNotFoundError(weights)
        detector = detect_with_yolo(weights, device, confidence)
        detection = dict(weights=str(weights), sha256=sha256_file(weights), confidence=confidence, nms_iou=0.5)
    source_summary = source / "summary.json"
    info = json.loads(source_summary.read_text()) if source_summary.is_file() else {}
    models = info.get("mano_model_by_side", dict(left="MANO_RIGHT.pkl", right="MANO_RIGHT.pkl"))
    if set(models) != {"left", "right"} or not set(models.values()) <= {"MANO_RIGHT.pkl", "MANO_LEFT.pkl"}:
        raise ValueError(f"Unsupported source MANO mapping {models}")
    model_dir = resolve_mano_model_path(mano_path)
    if model_dir.is_file():
        model_dir = model_dir.parent
    assets = {n: {"sha256": sha256_file(model_dir / n)} for n in set(models.values())}
    for name, asset in assets.items():
        expected = info.get("mano_assets", {}).get(name, {}).get("sha256")
        if expected and expected != asset["sha256"]:
            raise ValueError(f"MANO asset differs from the native label export: {name}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent, prefix=".wilor-local-") as work:
        root = Path(work) / "dataset"
        root.mkdir()
        (root / "images").mkdir()
        (root / "labels").mkdir()
        with (root / "index.jsonl").open("w") as all_rows:
            counts = {split: export_split(manifest, root, split, all_rows, detector, cache, min_iou, models,
                                          label_bboxes=label_bboxes, bbox_pinhole_rescale=bbox_pinhole_rescale)
                      for split, manifest in frames.items()}
        summary = dict(input_view=INPUT_VIEW, image_space="physical_rectified", source_dataset=str(source),
                       min_iou=None if label_bboxes else min_iou, detector=detection, splits=counts, mano_model_by_side=models,
                       mano_pose_representation=info.get("mano_pose_representation", "rotation_matrix_full_mean"),
                       mano_assets=assets)
        if bbox_pinhole_rescale is not None:
            summary.update(pinhole_crop="kb_bbox_pinhole", rescale_factor=bbox_pinhole_rescale)
        if label_bboxes:
            summary["bbox_source"] = "native_label"
        (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        profile = build_dataset_profile(root, mano_path)
        if not profile["geometry_validation"]["valid"]:
            raise ValueError(profile["geometry_validation"]["errors"][:5])
        source_profile = source / "dataset_profile.json"
        if source_profile.is_file():
            supervision = json.loads(source_profile.read_text()).get("supervision", {})
            for field in ("global_orient", "hand_pose", "betas"):
                if supervision.get(field, {}).get("available") and not profile["supervision"][field]["available"]:
                    raise ValueError(f"Local labels lost source {field} supervision: {profile['mano_replay']}")
        profile["dataset_root"] = str(destination)
        profile["geometry_validation"]["manifest"] = str(destination / "index.jsonl")
        (root / "dataset_profile.json").write_text(json.dumps(profile, indent=2) + "\n")
        if cache_inputs is not None:
            receipt = {"inputs": cache_inputs, "output": dataset_snapshot(root, exported=True)}
            (root / "export_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        root.rename(destination)
    return summary


def dataset_snapshot(root, *, exported=False):
    """Identify manifests by content and local image/label revisions by filesystem metadata."""
    manifests = ("train.jsonl", "val.jsonl")
    documents = (*manifests, "summary.json", "dataset_profile.json")
    if exported:
        documents += ("index.jsonl",)
    hashes = {name: sha256_file(root / name) for name in documents
              if exported or name in manifests or (root / name).exists()}
    assets = hashlib.sha256()
    seen = set()
    for name in manifests:
        with (root / name).open() as rows:
            for line in rows:
                row = json.loads(line)
                for path in (row["image"], row["label"]):
                    if path not in seen:
                        stat = (root / path).stat()
                        # A replacement or an in-place edit can preserve the size and mtime.
                        revision = [path, stat.st_dev, stat.st_ino, stat.st_size,
                                    stat.st_mtime_ns, stat.st_ctime_ns]
                        assets.update(json.dumps(revision).encode() + b"\n")
                        seen.add(path)
    hashes["referenced_files"] = assets.hexdigest()
    return hashes


def prepare_local_views(config):
    """Publish a complete export once, or reuse it with the same input and output inventory."""
    options = config.local_export
    source, destination = options.native_dataset, config.dataset_root
    wide = config.rescale_factor if config.pinhole_crop == "kb_bbox_pinhole" else None
    label_boxes = options.detector is None and options.detector_cache is None
    source_kind = "native_label" if label_boxes else ("cache" if options.detector_cache else "detector")
    detection_file = options.detector_cache or options.detector
    destination.parent.mkdir(parents=True, exist_ok=True)
    # The sibling lock continues to identify the export across its atomic directory rename.
    with (destination.parent / f".{destination.name}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        inputs = {
            "version": 1, "source": str(source), "dataset": dataset_snapshot(source),
            "bbox_source": source_kind, "pinhole_crop": config.pinhole_crop,
            "bbox_pinhole_rescale": wide,
        }
        if detection_file:
            inputs.update(proposals_sha256=sha256_file(detection_file), min_iou=options.min_iou)
        if options.detector:
            inputs["confidence"] = options.detector_confidence
        if destination.exists():
            message = (f"Local view cache is incomplete or differs from the requested export at {destination}; "
                       "choose a new paths.dataset_root to generate matching views.")
            try:
                receipt = json.loads((destination / "export_receipt.json").read_text())
                if receipt["inputs"] != inputs or receipt["output"] != dataset_snapshot(destination, exported=True):
                    raise ValueError(message)
                summary = json.loads((destination / "summary.json").read_text())
                model_dir = resolve_mano_model_path(config.mano_path)
                if model_dir.is_file():
                    model_dir = model_dir.parent
                for name, asset in summary["mano_assets"].items():
                    if sha256_file(model_dir / name) != asset["sha256"]:
                        raise ValueError(f"{message} MANO asset differs: {name}")
            except (OSError, KeyError, json.JSONDecodeError) as exc:
                raise ValueError(message) from exc
            print(f"Reusing local pinhole views at {destination}", flush=True)
            return summary
        print(f"Exporting local pinhole views from {source} to {destination} (boxes: {source_kind})", flush=True)
        return export_dataset(
            source, destination, config.mano_path, weights=options.detector,
            proposal_cache=options.detector_cache, label_bboxes=label_boxes,
            device=None if config.device == "auto" else config.device,
            confidence=options.detector_confidence, min_iou=options.min_iou,
            bbox_pinhole_rescale=wide, cache_inputs=inputs,
        )
