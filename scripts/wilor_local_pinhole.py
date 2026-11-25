"""WiLoR's training-time per-box perspective camera, with native KB outputs."""
from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np
import torch


class LocalSample(NamedTuple):
    frame_index: int
    detection_index: int
    side: int
    view: Any
    rgb: np.ndarray


def uses_local_pinhole(mode, model, calibration):
    perspective = (getattr(model, "physical_camera_enabled", False)
                   and model.physical_camera_head.camera_model == "physical_perspective")
    compatible = perspective and calibration is not None and calibration.model == "KB"
    if mode == "local-pinhole" and not compatible:
        raise ValueError("local-pinhole requires a physical_perspective WiLoR head and original KB images")
    return mode != "native" and compatible


def rectify_proposal(frame, box, calibration):
    from wilor.datasets.local_pinhole import local_camera
    if frame.shape[1::-1] != tuple(calibration.image_size):
        raise ValueError("Source frame and KB calibration sizes differ")
    try:
        view = local_camera(box, calibration.K, calibration.distortion)
    except ValueError:
        # As in the training exporter, boxes outside the invertible KB domain have no tile.
        return None
    return view, view.rectify(frame)


def infer_boxes(frames, detections, model, cfg, device, calibration,
                save_mano, executor, stats, limit, select, detector_side):
    from tools.hand_finetune.crop_geometry import rgb_chw

    boxes_by_frame = []
    jobs = []
    for fi, detection in enumerate(detections):
        meta = {}
        boxes_by_frame.append(meta)
        if detection.boxes is None or len(detection.boxes) == 0:
            continue
        boxes = detection.boxes.xyxy.detach().cpu().numpy()
        scores = detection.boxes.conf.detach().cpu().numpy()
        sides = detection.boxes.cls.detach().cpu().numpy()
        sources = getattr(detection, "proposal_sources", ["detector"] * len(boxes))
        for idx in select(scores, sides, limit):
            idx = int(idx)
            meta[idx] = dict(bbox_xyxy=boxes[idx].tolist(), confidence=float(scores[idx]),
                             detector_is_right=detector_side(sides[idx]), proposal_source=sources[idx])
            jobs.append((fi, idx, boxes[idx]))

    def rectify(job):
        frame_index, _, box = job
        return rectify_proposal(frames[frame_index], box, calibration)

    samples = []
    for (fi, idx, _), rectified in zip(jobs, executor.map(rectify, jobs)):
        if rectified is None:
            stats["invalid_local_boxes"] += 1
            continue
        view, tile = rectified
        for side in (0, 1):
            samples.append(LocalSample(fi, idx, side, view, rgb_chw(tile, side == 0)))

    records = [[] for _ in frames]
    start = 0
    while start < len(samples):
        chunk = samples[start:start + stats["effective_hypothesis_batch_size"]]
        try:
            result = predict_chunk(chunk, model, cfg, device, save_mano)
        except torch.cuda.OutOfMemoryError:
            if len(chunk) <= 1:
                raise
            stats["effective_hypothesis_batch_size"] = max(1, len(chunk) // 2)
            stats["cuda_oom_retries"] += 1
            continue
        for sample, hand in zip(chunk, result):
            if hand is None:
                stats["invalid_local_projections"] += 1
                continue
            fi, idx, side = sample.frame_index, sample.detection_index, sample.side
            records[fi].append(dict(detection_index=idx, hypothesis_index=idx * 2 + side,
                                    is_right=side, **boxes_by_frame[fi][idx], **hand))
        start += len(chunk)
    return records


def local_input_batch(chunk, cfg, device):
    """Match the physical camera fields and RGB normalization of GloveDataset."""
    views = [sample.view for sample in chunk]
    side = torch.tensor([sample.side for sample in chunk], device=device)
    img = torch.as_tensor(np.stack([sample.rgb for sample in chunk]), device=device).float() / 255.
    n, _, height, width = img.shape
    if height != width or width != cfg.MODEL.IMAGE_SIZE:
        raise ValueError("Local pinhole export and WiLoR input size differ")
    mean = img.new_tensor(cfg.MODEL.IMAGE_MEAN).view(1, 3, 1, 1)
    std = img.new_tensor(cfg.MODEL.IMAGE_STD).view(1, 3, 1, 1)
    return dict(img=(img - mean) / std, side=side,
                 camera_K=img.new_tensor(np.stack([v.K for v in views])),
                 camera_distortion=img.new_zeros(n, 4), crop_box_size=img.new_full((n,), width),
                 crop_transform=torch.eye(2, 3, device=device).expand(n, -1, -1),
                 image_size=img.new_full((n, 2), width),
                 bbox_xyxy=img.new_tensor([0, 0, width, height]).expand(n, -1))


@torch.inference_mode()
def predict_chunk(chunk, model, cfg, device, save_mano):
    views = [sample.view for sample in chunk]
    batch = local_input_batch(chunk, cfg, device)
    img = batch["img"]
    width = img.shape[-1]
    output = model(batch)
    tile_uv = ((output["pred_keypoints_2d"] + .5) * width).cpu().numpy()
    roots = output["pred_root_camera_m"].cpu().numpy()
    local_z = output["pred_keypoints_3d"][..., 2].cpu().numpy()
    clamped = local_z - local_z[:, :1] + roots[:, 2:3] < .01
    mano = dict(output["pred_mano_params"])
    orient = mano["global_orient"].cpu().numpy()
    mano["global_orient"] = img.new_tensor(np.stack([
        view.physical_mano_orientation(orient[i, 0], bool(chunk[i].side)) for i, view in enumerate(views)
    ]))[:, None]
    # MANO rotates about its own wrist. Replay the native parameters to keep the
    # saved translation relative to the MANO origin used by the renderers.
    joints = model.mano(**mano, pose2rot=False).joints.cpu().numpy()
    mano = {k: v.cpu().numpy() for k, v in mano.items()} if save_mano else {}
    result = []
    for i, view in enumerate(views):
        if not chunk[i].side:
            tile_uv[i, :, 0] = width - 1 - tile_uv[i, :, 0]
            joints[i, :, 0] *= -1
        pixels = view.native_pixels(tile_uv[i])
        translation = roots[i] @ view.rotation - joints[i, 0]
        if not np.isfinite(pixels).all() or not np.isfinite(translation).all():
            result.append(None)
            continue
        hand = dict(joints_2d=pixels.tolist(), camera_translation=translation.tolist())
        if clamped[i].any():
            hand["local_depth_clamped_joints"] = np.flatnonzero(clamped[i]).tolist()
        if save_mano:
            hand.update(joints_3d_camera_relative=joints[i].tolist(),
                        mano={k: v[i].tolist() for k, v in mano.items()})
        result.append(hand)
    return result
