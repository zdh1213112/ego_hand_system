#!/usr/bin/env python3
"""Live EGO4 WiLoR/MANO inference with primary stereo and on-demand side views."""

from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import itertools
import os
from pathlib import Path
import subprocess
import time
from datetime import datetime

import cv2
import numpy as np
import orjson
import torch

from camera_models import project_points, unproject_points
from ego_data.calibration import cameras_from_ego_yaml
from ego_four_bridge import DecodedFourReader, FourBridge, LatestFourReader, decode_camera
from ego_live_stabilizer import LiveHandStabilizer
from ego_live_wilor_preview import LivePreview
import wilor_inference as base
import wilor_multiview_inference as multi


ROOT = Path(__file__).resolve().parents[1]
MAIN = ("cam_0", "cam_1")
SIDE = ("cam_2", "cam_3")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bridge", type=Path, default=ROOT / "build/ego_live_bridge")
    parser.add_argument("--sdk-config", type=Path, required=True)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "output/ego_live_wilor" / datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "models/wilor/wilor_final.ckpt")
    parser.add_argument("--model-config", type=Path, default=ROOT / "models/wilor/model_config.yaml")
    parser.add_argument("--detector", type=Path, default=ROOT / "models/wilor/detector0.pt")
    parser.add_argument("--mano-model-dir", type=Path, default=ROOT / "models/mano")
    parser.add_argument("--confidence", type=float, default=0.3)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--rescale-factor", type=float, default=2.0)
    parser.add_argument("--max-detections-per-class", type=int, default=1)
    parser.add_argument("--max-box-area-ratio", type=float, default=0.65,
                        help="reject detector boxes covering too much of a camera image")
    parser.add_argument("--hypotheses", choices=("detector", "dual"), default="detector",
                        help="detector uses its handedness class once; dual runs both sides per box")
    parser.add_argument("--detector-every", type=int, default=12,
                        help="refresh detector every N processed frames while hands are tracked")
    parser.add_argument("--frame-batch", type=int, choices=(1, 2), default=2,
                        help="WiLoR stereo pairs per forward pass; 2 favors throughput")
    parser.add_argument("--crop-device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--side-scan-every", type=int, default=0,
                        help="periodic side-camera discovery; 0 disables periodic scanning")
    parser.add_argument("--side-primary", choices=SIDE, default="cam_2",
                        help="first side view used when primary stereo needs help")
    parser.add_argument("--max-sync-delta-us", type=int, default=20000)
    parser.add_argument("--max-reprojection-px", type=float, default=20.0)
    parser.add_argument("--min-valid-joints", type=int, default=12)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--queue", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--no-preview", action="store_true",
                        help="disable the live hand-pose preview window")
    parser.add_argument("--preview-camera", choices=MAIN, default="cam_0",
                        help="main camera shown in the live preview")
    return parser.parse_args()


def select_hands(predictions):
    """One best detector proposal per handedness, per view."""
    chosen = {}
    for hand in predictions:
        side = int(hand["is_right"])
        if side not in chosen or hand["confidence"] > chosen[side]["confidence"]:
            chosen[side] = hand
    return chosen


class BoxTracker:
    """Short lived hand proposals between detector refreshes."""

    def __init__(self, image_size):
        self.width, self.height = image_size
        self.state = {name: {} for name in MAIN}

    def ready(self):
        return all(self.state[name] for name in MAIN)

    def proposals(self, names):
        return [{"boxes": list(self.state[name].values())} for name in names]

    def update(self, predictions, names, refreshed):
        for name, hands in zip(names, predictions):
            previous = self.state[name]
            current = {}
            for hand in hands:
                side = int(hand["is_right"])
                points = np.asarray(hand["joints_2d"], np.float64)
                if points.shape != (21, 2) or not np.isfinite(points).all():
                    continue
                center = np.median(points, axis=0)
                box = np.asarray(hand["bbox_xyxy"], np.float64)
                if not refreshed and side in previous:
                    shift = np.clip(center - previous[side]["center"], -80, 80)
                    box += np.tile(shift, 2)
                box[[0, 2]] = np.clip(box[[0, 2]], 0, self.width - 1)
                box[[1, 3]] = np.clip(box[[1, 3]], 0, self.height - 1)
                if box[2] <= box[0] or box[3] <= box[1]:
                    continue
                current[side] = {
                    "bbox_xyxy": box.tolist(),
                    "confidence": float(hand["confidence"]),
                    "detector_is_right": side,
                    "source": "tracker",
                    "center": center,
                }
            self.state[name] = current


def triangulate_pair(left_name, left_hand, right_name, right_hand, cameras,
                     max_reprojection_px):
    """Triangulate 21 KB rays together when only two views are selected."""
    names = (left_name, right_name)
    pixels = [np.asarray(hand["joints_2d"], np.float64)
              for hand in (left_hand, right_hand)]
    rays = []
    ray_valid = []
    centers = []
    for name, image_points in zip(names, pixels):
        camera = cameras[name]
        camera_rays, valid = unproject_points(camera, image_points)
        rotation = camera.T_base_camera[:3, :3]
        base_rays = camera_rays @ rotation.T
        rays.append(base_rays)
        ray_valid.append(valid)
        centers.append(camera.T_base_camera[:3, 3])
    first, second = rays
    separation = centers[1] - centers[0]
    cosine = np.einsum("ij,ij->i", first, second)
    denominator = 1.0 - cosine * cosine
    with np.errstate(divide="ignore", invalid="ignore"):
        first_distance = (first @ separation - cosine * (second @ separation)) / denominator
        second_distance = (cosine * (first @ separation) - second @ separation) / denominator
    points = (centers[0] + first_distance[:, None] * first
              + centers[1] + second_distance[:, None] * second) / 2.0
    valid = (ray_valid[0] & ray_valid[1] & (denominator > 1e-5)
             & (first_distance > 0) & (second_distance > 0)
             & np.isfinite(points).all(axis=1))
    valid &= (np.linalg.norm(points - cameras["cam_0"].T_base_camera[:3, 3], axis=1)
              >= 0.15)
    valid &= (np.linalg.norm(points - cameras["cam_0"].T_base_camera[:3, 3], axis=1)
              <= 1.5)
    errors = np.full((21, 2), np.inf)
    for view_index, name in enumerate(names):
        camera = cameras[name]
        projected, projectable = project_points(
            camera, (points[valid] - camera.T_base_camera[:3, 3]) @ camera.T_base_camera[:3, :3],
        ) if valid.any() else (np.empty((0, 2)), np.empty(0, bool))
        if valid.any():
            errors[valid, view_index] = np.where(
                projectable,
                np.linalg.norm(projected - pixels[view_index][valid], axis=1),
                np.inf,
            )
    valid &= np.all(errors <= max_reprojection_px, axis=1)
    points[~valid] = np.nan
    return points, errors, valid


def fuse_side(side, selected, cameras, max_reprojection_px, min_valid_joints):
    observed = {name: hands[side] for name, hands in selected.items() if side in hands}
    result = {
        "is_right": side,
        "views": list(observed),
        "joints_3d_base_m": None,
        "valid_joints": 0,
        "median_reprojection_px": None,
    }
    if len(observed) < 2:
        return result
    if len(observed) == 2:
        (first_name, first_hand), (second_name, second_hand) = observed.items()
        points, pair_errors, valid_mask = triangulate_pair(
            first_name, first_hand, second_name, second_hand,
            cameras, max_reprojection_px,
        )
        valid = int(valid_mask.sum())
        result["valid_joints"] = valid
        if valid >= min_valid_joints:
            result["joints_3d_base_m"] = points.tolist()
            result["median_reprojection_px"] = float(np.median(pair_errors[valid_mask]))
        return result
    candidates = []
    for (first_name, first_hand), (second_name, second_hand) in itertools.combinations(
        observed.items(), 2,
    ):
        points, _pair_errors, valid = triangulate_pair(
            first_name, first_hand, second_name, second_hand,
            cameras, max_reprojection_px,
        )
        if not valid.any():
            continue
        support = np.zeros((21, len(observed)), bool)
        residuals = np.full((21, len(observed)), np.inf)
        for view_index, (name, hand) in enumerate(observed.items()):
            camera = cameras[name]
            pixels = np.asarray(hand["joints_2d"], np.float64)
            projected, projectable = project_points(
                camera,
                (points[valid] - camera.T_base_camera[:3, 3]) @ camera.T_base_camera[:3, :3],
            )
            reprojection = np.linalg.norm(projected - pixels[valid], axis=1)
            residuals[valid, view_index] = np.where(projectable, reprojection, np.inf)
            support[valid, view_index] = projectable & (reprojection <= max_reprojection_px)
        supported = valid & (support.sum(axis=1) >= 2)
        good = int(supported.sum())
        if good == 0:
            continue
        three_view = int((supported & (support.sum(axis=1) >= 3)).sum())
        median_error = float(np.median(residuals[support]))
        baseline = float(np.linalg.norm(
            cameras[first_name].T_base_camera[:3, 3]
            - cameras[second_name].T_base_camera[:3, 3]
        ))
        candidates.append(((good, three_view, -median_error, baseline), points, supported, median_error))
    if candidates:
        _, points, supported, median_error = max(candidates, key=lambda item: item[0])
        valid = int(supported.sum())
        result["valid_joints"] = valid
        if valid >= min_valid_joints:
            points[~supported] = np.nan
            result["joints_3d_base_m"] = points.tolist()
            result["median_reprojection_px"] = median_error
    return result


def main() -> int:
    args = parse_args()
    if (args.max_frames < 0 or args.queue < 1 or args.side_scan_every < 0
            or args.warmup < 0 or args.detector_every < 1):
        raise ValueError("invalid frame, queue, scan, or warmup count")
    if not 0 < args.max_box_area_ratio <= 1:
        raise ValueError("max-box-area-ratio must be in (0, 1]")
    if not torch.cuda.is_available():
        raise RuntimeError("live WiLoR requires CUDA")
    args.output.mkdir(parents=True, exist_ok=True)
    calibration_path = args.calibration or args.output / "ego4_calibration.yaml"
    if not calibration_path.is_file():
        subprocess.run([
            str(args.bridge), "--sdk-config", str(args.sdk_config),
            "--calibration-out", str(calibration_path), "--calibration-only",
        ], check=True)
    cameras = cameras_from_ego_yaml(calibration_path)
    if not set(MAIN + SIDE).issubset(cameras):
        raise ValueError("EGO4 calibration must contain cam_0 through cam_3")

    cv2.setNumThreads(1)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # New hand counts and side-view batches change tensor shapes at runtime.
    # cuDNN autotuning caused 18-second first-use stalls for a four-image YOLO batch.
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda")
    model, config = base.load_wilor_model(args.checkpoint, args.model_config, args.mano_model_dir)
    model = model.to(device).eval().half()
    detector = base.load_detector(args.detector, device)
    print(f"WiLoR loaded; detector={args.detector.name}, hypotheses={args.hypotheses}", flush=True)

    def infer(images, executor, external_boxes=None):
        prep_started = time.perf_counter()
        prepared = multi._prepare_dual_batched(
            images, detector, config, args.confidence, args.iou,
            args.rescale_factor, 4, args.max_detections_per_class,
            preprocess_executor=executor,
            dual_hypothesis=args.hypotheses == "dual",
            max_box_area_ratio=args.max_box_area_ratio,
            external_boxes=external_boxes,
            gpu_crop_device=device if args.crop_device == "cuda" else None,
        )
        prepared_at = time.perf_counter()
        predictions = multi._infer_prepared_batched(
            prepared, model, config, device, batch_size=8, full_output=True,
        )
        return predictions, {
            "detector_and_crop": round((prepared_at - prep_started) * 1000, 2),
            "wilor_forward": round((time.perf_counter() - prepared_at) * 1000, 2),
        }

    bridge = FourBridge(args.bridge, args.sdk_config)
    capture_capacity = max(args.queue, args.frame_batch)
    reader = DecodedFourReader(
        LatestFourReader(bridge, capture_capacity), capacity=args.frame_batch,
    )
    processed = 0
    warmed = False
    tracker = None
    first_timestamp_us = None
    last_timestamp_us = None
    first_sequence = None
    last_sequence = None
    valid_two_hands = 0
    side_frames = 0
    recent_latency = deque(maxlen=300)
    recent_frames = deque()
    stabilizer = LiveHandStabilizer()
    preview_enabled = not args.no_preview and bool(
        os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
    )
    if not args.no_preview and not preview_enabled:
        print("No desktop display found; live preview disabled. JSONL output remains active.",
              flush=True)
    preview = None
    stop_requested = False
    with ThreadPoolExecutor(max_workers=4) as executor, \
            (args.output / "predictions.jsonl").open("wb") as output:
        try:
            while args.max_frames == 0 or processed < args.max_frames:
                next_packet = reader.get(timeout=5.0)
                if next_packet is None:
                    continue
                sequence, packet, frames = next_packet
                if not warmed:
                    warm_frames = [frames[name] for name in MAIN] * args.frame_batch
                    warm_side = [decode_camera(packet, name) for name in SIDE] * args.frame_batch
                    print("Warming main and side WiLoR batches...", flush=True)
                    for _ in range(args.warmup):
                        infer(warm_frames, executor)
                        infer(warm_side, executor)
                    warmed = True
                    if args.warmup:
                        continue
                batch = [(sequence, packet, frames)]
                target_batch = min(
                    args.frame_batch,
                    args.max_frames - processed if args.max_frames else args.frame_batch,
                )
                while len(batch) < target_batch:
                    following = reader.get(timeout=1.0)
                    if following is None:
                        break
                    batch.append(following)
                started = time.perf_counter()
                if tracker is None:
                    tracker = BoxTracker((packet["width"], packet["height"]))
                refresh_detector = processed % args.detector_every == 0 or not tracker.ready()
                images = [item[2][name] for item in batch for name in MAIN]
                external = None if refresh_detector else tracker.proposals(MAIN) * len(batch)
                predictions, main_timing = infer(
                    images, executor, external_boxes=external,
                )
                tracker.update(predictions[-2:], MAIN, refresh_detector)
                main_inferred = time.perf_counter()
                states = []
                for batch_index, (_sequence, packet, _frames) in enumerate(batch):
                    timestamps = packet["timestamp_us"]
                    sync_delta = max(timestamps.values()) - min(timestamps.values())
                    if sync_delta > args.max_sync_delta_us:
                        print(f"warning: four-view timestamp spread {sync_delta} us", flush=True)
                    pair_predictions = predictions[2 * batch_index:2 * batch_index + 2]
                    by_camera = {
                        name: select_hands(hands)
                        for name, hands in zip(MAIN, pair_predictions)
                    }
                    sync_valid = sync_delta <= args.max_sync_delta_us
                    fused = {
                        side: fuse_side(
                            side, by_camera if sync_valid else {}, cameras,
                            args.max_reprojection_px, args.min_valid_joints,
                        )
                        for side in (0, 1)
                    }
                    scan_due = args.side_scan_every and (processed + batch_index) % args.side_scan_every == 0
                    need_side = bool(sync_valid and (scan_due or any(
                        item["joints_3d_base_m"] is None
                        for side, item in fused.items()
                        if side in by_camera[MAIN[0]] or side in by_camera[MAIN[1]]
                    )))
                    states.append({
                        "by_camera": by_camera,
                        "fused": fused,
                        "need_side": need_side,
                        "scan_due": bool(scan_due),
                        "sync_valid": sync_valid,
                        "sync_delta": sync_delta,
                    })
                fused_at = time.perf_counter()
                side_indices = [index for index, state in enumerate(states) if state["need_side"]]
                side_timing = None
                if side_indices:
                    primary = args.side_primary
                    secondary = SIDE[1] if primary == SIDE[0] else SIDE[0]
                    primary_images = [decode_camera(batch[index][1], primary)
                                      for index in side_indices]
                    primary_predictions, primary_timing = infer(primary_images, executor)
                    side_timing = {primary: primary_timing}
                    for side_batch_index, batch_index in enumerate(side_indices):
                        by_camera = states[batch_index]["by_camera"]
                        by_camera[primary] = select_hands(primary_predictions[side_batch_index])
                        states[batch_index]["fused"] = {
                            side: fuse_side(side, by_camera, cameras,
                                            args.max_reprojection_px, args.min_valid_joints)
                            for side in (0, 1)
                        }
                    secondary_indices = [
                        index for index in side_indices
                        if states[index]["scan_due"] or any(
                            item["joints_3d_base_m"] is None
                            for side, item in states[index]["fused"].items()
                            if any(side in hands for hands in states[index]["by_camera"].values())
                        )
                    ]
                    if secondary_indices:
                        secondary_images = [decode_camera(batch[index][1], secondary)
                                            for index in secondary_indices]
                        secondary_predictions, secondary_timing = infer(secondary_images, executor)
                        side_timing[secondary] = secondary_timing
                        for side_batch_index, batch_index in enumerate(secondary_indices):
                            by_camera = states[batch_index]["by_camera"]
                            by_camera[secondary] = select_hands(secondary_predictions[side_batch_index])
                            states[batch_index]["fused"] = {
                                side: fuse_side(side, by_camera, cameras,
                                                args.max_reprojection_px, args.min_valid_joints)
                                for side in (0, 1)
                            }
                for batch_index, (sequence, packet, _frames) in enumerate(batch):
                    state = states[batch_index]
                    by_camera = state["by_camera"]
                    fused = state["fused"]
                    need_side = state["need_side"]
                    ended = time.perf_counter()
                    row = {
                        "sequence": sequence,
                        "timestamp_us": packet["timestamp_us"],
                        "frame_index": packet["index"],
                        "sync_spread_us": state["sync_delta"],
                        "sync_valid": state["sync_valid"],
                        "camera_hands": by_camera,
                        "hands": list(fused.values()),
                        "side_inferred": need_side,
                        "detector_refreshed": refresh_detector,
                        "batch_pairs": len(batch),
                        "latency_ms": round((time.monotonic_ns() - packet["host_ns"]) / 1e6, 2),
                        "timing_ms": {
                            "decode_main": packet["decode_main_ms"],
                            "decode_to_process": round((started * 1e9 - packet["decoded_ns"]) / 1e6, 2),
                            "main_wilor_batch": round((main_inferred - started) * 1000, 2),
                            "main_detector_and_crop_batch": main_timing["detector_and_crop"],
                            "main_forward_batch": main_timing["wilor_forward"],
                            "main_fusion": round((fused_at - main_inferred) * 1000, 2),
                            "side_wilor": side_timing,
                            "total_batch_to_row": round((ended - started) * 1000, 2),
                        },
                    }
                    stabilizer.update_row(row)
                    output.write(orjson.dumps(row, option=orjson.OPT_NON_STR_KEYS) + b"\n")
                    output.flush()
                    camera_timestamp = packet["timestamp_us"]["cam_0"]
                    if first_timestamp_us is None:
                        first_timestamp_us = camera_timestamp
                        first_sequence = sequence
                    last_timestamp_us = camera_timestamp
                    last_sequence = sequence
                    recent_frames.append((camera_timestamp, sequence))
                    while (len(recent_frames) > 2
                           and camera_timestamp - recent_frames[0][0] > 3_000_000):
                        recent_frames.popleft()
                    valid_two_hands += sum(
                        hand["joints_3d_base_m"] is not None for hand in row["hands"]
                    ) == 2
                    side_frames += need_side
                    recent_latency.append(row["latency_ms"])
                    processed += 1
                    if processed % 10 == 0:
                        print(
                            f"frames={processed} capture={reader.sequence} dropped={reader.dropped} "
                            f"batch={len(batch)} last={row['timing_ms']['total_batch_to_row']:.1f}ms "
                            f"latency={row['latency_ms']:.1f}ms "
                            f"valid3d={sum(hand['joints_3d_base_m'] is not None for hand in row['hands'])} "
                            f"side={need_side}", flush=True,
                        )
                    if preview_enabled and batch_index == len(batch) - 1:
                        if preview is None:
                            preview = LivePreview(args.preview_camera,
                                                  cameras[args.preview_camera])
                        recent_span_s = (recent_frames[-1][0] - recent_frames[0][0]) / 1e6
                        output_fps = ((len(recent_frames) - 1) / recent_span_s
                                      if recent_span_s > 0 else 0.0)
                        skipped = (recent_frames[-1][1] - recent_frames[0][1] + 1
                                   - len(recent_frames))
                        if not preview.show():
                            stop_requested = True
                            break
                        preview.submit(packet, _frames, row, output_fps, skipped)
                if stop_requested:
                    break
        except KeyboardInterrupt:
            print("Stopping live WiLoR...", flush=True)
        finally:
            reader.close()
            if preview is not None:
                preview.close()
    if processed:
        span_s = (last_timestamp_us - first_timestamp_us) / 1e6
        output_fps = (processed - 1) / span_s if span_s > 0 else 0.0
        recent = np.asarray(recent_latency, np.float64)
        print(
            f"WiLoR summary: {processed} updates in {span_s:.2f}s = {output_fps:.2f} Hz; "
            f"capture span={last_sequence - first_sequence + 1}, "
            f"skipped={last_sequence - first_sequence + 1 - processed}, "
            f"two-hand 3D={valid_two_hands}/{processed}, side={side_frames}/{processed}, "
            f"last-{len(recent)} latency P50/P95="
            f"{np.percentile(recent, 50):.1f}/{np.percentile(recent, 95):.1f}ms",
            flush=True,
        )
        print(f"Saved {processed} live WiLoR frames to {args.output / 'predictions.jsonl'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
