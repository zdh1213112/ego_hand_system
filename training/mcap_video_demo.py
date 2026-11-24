"""Run fine-tuned WiLoR on one GEN MCAP camera and save an RGB / 3D video."""

from __future__ import annotations

import argparse
from fractions import Fraction
import json
import logging
from pathlib import Path

import av
import cv2
from mcap.reader import NonSeekingReader
from mcap_protobuf.decoder import DecoderFactory
import numpy as np
import torch

from video_demo import choose_device, infer_frame, load_detector
from visualize_hand_3d import H264VideoWriter, HandScene, ROOT, letterbox, put_text, scene_bounds
from wilor.models import load_wilor
from wilor.utils.visualization import LEFT_COLOR, RIGHT_COLOR, draw_hand_overlay


LOG = logging.getLogger("mcap_video_demo")


def camera_topic(camera: str) -> str:
    if not camera.startswith("camera") or not camera[6:].isdigit():
        raise ValueError(f"camera must look like camera2, got {camera!r}")
    return f"/robot0/sensor/{camera}/compressed"


class H264Decoder:
    """Decode the Annex-B H.264 access units stored in a GEN camera topic."""

    def __init__(self):
        self.codec = av.CodecContext.create("h264", "r")
        self.frame_index = 0
        self.decode_errors = 0

    def decode(self, payload: bytes, timestamp_ns: int):
        packet = av.Packet(payload)
        packet.pts = int(timestamp_ns)
        packet.dts = int(timestamp_ns)
        packet.time_base = Fraction(1, 1_000_000_000)
        try:
            frames = self.codec.decode(packet)
        except av.error.InvalidDataError:
            self.decode_errors += 1
            return []
        return self._convert(frames, timestamp_ns)

    def flush(self):
        try:
            frames = self.codec.decode(None)
        except av.error.InvalidDataError:
            self.decode_errors += 1
            return []
        return self._convert(frames, 0)

    def _convert(self, frames, fallback_timestamp: int):
        result = []
        for frame in frames:
            timestamp_ns = int(frame.pts) if frame.pts is not None else int(fallback_timestamp)
            result.append((self.frame_index, timestamp_ns, frame.to_ndarray(format="bgr24")))
            self.frame_index += 1
        return result


def iter_mcap_frames(path: Path, camera: str, max_frames: int | None = None):
    """Yield decoded BGR frames in camera message order."""
    topic = camera_topic(camera)
    decoder = H264Decoder()
    factory = DecoderFactory()
    produced = 0
    with path.open("rb") as stream:
        reader = NonSeekingReader(stream, decoder_factories=[factory])
        for schema, channel, message in reader.iter_messages(topics=[topic], log_time_order=False):
            message_decoder = factory.decoder_for(channel.message_encoding, schema)
            if message_decoder is None:
                raise RuntimeError(f"No protobuf decoder for {topic}")
            proto = message_decoder(message.data)
            if str(proto.format).strip().lower() != "h264":
                raise ValueError(f"{topic} must contain H.264, got {proto.format!r}")
            for item in decoder.decode(bytes(proto.data), int(message.log_time)):
                yield item
                produced += 1
                if max_frames is not None and produced >= max_frames:
                    return
        for item in decoder.flush():
            yield item
            produced += 1
            if max_frames is not None and produced >= max_frames:
                return
    if produced == 0:
        raise RuntimeError(f"No decodable frames found on {topic}")
    if decoder.decode_errors:
        LOG.warning("H.264 decoder ignored %d invalid packets", decoder.decode_errors)


def select_detections(
    boxes: np.ndarray, confidences: np.ndarray, handedness: np.ndarray, limit: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Keep the strongest detections per handedness class."""
    if limit <= 0 or len(boxes) == 0:
        return boxes, confidences, handedness
    keep = []
    for side in sorted(set(int(round(float(value))) for value in handedness)):
        indices = np.flatnonzero(np.rint(handedness).astype(np.int32) == side)
        keep.extend(indices[np.argsort(confidences[indices])[::-1][:limit]].tolist())
    keep = np.asarray(sorted(keep), dtype=np.int64)
    return boxes[keep], confidences[keep], handedness[keep]


def infer_selected_frame(frame, detector, model, model_cfg, device, args):
    """Detect first so duplicate boxes can be filtered before WiLoR inference."""
    detection = detector(frame, conf=args.confidence, iou=args.iou, verbose=False)[0]
    if detection.boxes is None or len(detection.boxes) == 0:
        return []
    boxes = detection.boxes.xyxy.detach().cpu().numpy().astype(np.float32)
    confidences = detection.boxes.conf.detach().cpu().numpy().astype(np.float32)
    handedness = detection.boxes.cls.detach().cpu().numpy().astype(np.float32)
    boxes, confidences, handedness = select_detections(
        boxes, confidences, handedness, args.max_detections_per_class
    )

    class SelectedDetector:
        def __call__(self, *unused_args, **unused_kwargs):
            class Boxes:
                xyxy = torch.from_numpy(boxes)
                conf = torch.from_numpy(confidences)
                cls = torch.from_numpy(handedness)

                def __len__(self):
                    return len(boxes)

            class Result:
                pass

            result = Result()
            result.boxes = Boxes()
            return [result]

    *_, hands, _, _, _ = infer_frame(
        frame, SelectedDetector(), model, model_cfg, device,
        args.confidence, args.iou, args.rescale_factor, args.batch_size, False,
    )
    return hands


def absolute_hands(hands: list[dict]) -> list[tuple[int, np.ndarray]]:
    result = []
    for hand in hands:
        points = np.asarray(hand["joints_3d"], dtype=np.float64)
        translation = np.asarray(hand["camera_translation"], dtype=np.float64)
        if points.shape != (21, 3) or translation.shape != (3,):
            raise ValueError("WiLoR produced invalid 3D joint or camera shapes")
        camera_points = points + translation
        if not np.isfinite(camera_points).all():
            raise ValueError("WiLoR produced non-finite camera-space joints")
        result.append((int(hand["is_right"]), camera_points))
    return result


def visualization_hands(hands: list[dict], coordinate_mode: str) -> list[tuple[int, np.ndarray]]:
    if coordinate_mode == "camera":
        return absolute_hands(hands)
    sides = [int(hand["is_right"]) for hand in hands]
    anchors = [0.14 if side else -0.14 for side in sides]
    if len(set(sides)) < len(sides):
        # Some detectors classify both visible hands as the same side. Preserve
        # their classes, but separate their display wrists in image order.
        order = sorted(range(len(hands)), key=lambda i: (
            sum(hands[i].get("bbox_xyxy", [i, 0, i, 0])[::2]) / 2, i
        ))
        for index, anchor_x in zip(order, np.linspace(-0.14, 0.14, len(hands))):
            anchors[index] = float(anchor_x)
    result = []
    for hand, anchor_x in zip(hands, anchors):
        side = int(hand["is_right"])
        points = np.asarray(hand["joints_3d"], dtype=np.float64)
        if points.shape != (21, 3) or not np.isfinite(points).all():
            raise ValueError("WiLoR produced invalid 3D joints")
        # A monocular weak-perspective camera does not provide metric absolute
        # depth. Fixed per-side wrist anchors keep pose readable and stable.
        anchor = np.array([anchor_x, 0.0, 0.0])
        result.append((side, points - points[:1] + anchor))
    return result


def estimated_fps(records: list[dict], fallback: float) -> float:
    timestamps = np.asarray([record["timestamp_ns"] for record in records], dtype=np.int64)
    if len(timestamps) < 2:
        return fallback
    deltas = np.diff(timestamps)
    deltas = deltas[deltas > 0]
    if not len(deltas):
        return fallback
    value = 1e9 / float(np.median(deltas))
    return value if np.isfinite(value) and 1 <= value <= 240 else fallback


def overlay_predictions(image: np.ndarray, hands: list[dict]) -> np.ndarray:
    """Draw YOLO boxes and projected WiLoR joints on the source camera frame."""
    visual_hands = []
    for hand in hands:
        is_right = bool(hand["is_right"])
        visual_hands.append({
            "bbox": hand["bbox_xyxy"],
            "joints_2d": hand["joints_2d"],
            "is_right": is_right,
            "label": f"{'R' if is_right else 'L'} {float(hand['confidence']):.2f}",
            "pred_color": RIGHT_COLOR if is_right else LEFT_COLOR,
        })
    return draw_hand_overlay(image, visual_hands, show_legend=False)


def render_video(args, records: list[dict], source_fps: float) -> Path:
    output = args.output_dir / f"{args.input.stem}_{args.camera}_pred_3d.mp4"
    temporary = output.with_name(f"{output.stem}.partial.mp4")
    writer = H264VideoWriter(temporary, source_fps, args.width, args.height)
    half, header, footer = args.width // 2, 58, 78
    panel_height = args.height - header - footer
    scene = HandScene(args.width - half, panel_height)
    scene.set_bounds(*scene_bounds([
        visualization_hands(record["hands"], args.coordinate_mode) for record in records
    ]))
    by_index = {record["frame_index"]: record for record in records}
    base_timestamp_ns = int(records[0]["timestamp_ns"])
    model_label = args.model_label or (args.adapter.name if args.adapter is not None else "base WiLoR")
    written = 0
    success = False
    try:
        for frame_index, timestamp_ns, image in iter_mcap_frames(args.input, args.camera, args.max_frames):
            record = by_index.get(frame_index)
            if record is None:
                raise RuntimeError(f"Second MCAP pass produced unexpected frame {frame_index}")
            canvas = np.full((args.height, args.width, 3), 24, dtype=np.uint8)
            annotated = overlay_predictions(image, record["hands"])
            canvas[header:header + panel_height, :half] = letterbox(annotated, half, panel_height)
            canvas[header:header + panel_height, half:] = scene.render(
                visualization_hands(record["hands"], args.coordinate_mode)
            )
            put_text(canvas, f"{args.camera} | YOLO boxes + projected 2D joints",
                     (16, 35), half - 30, 0.60)
            coordinate_title = (
                "camera coordinates" if args.coordinate_mode == "camera"
                else "wrist-relative pose | fixed L/R anchors"
            )
            put_text(canvas, f"WiLoR prediction ({model_label}) | {coordinate_title}",
                     (half + 16, 35), half - 30, 0.60)
            labels = ", ".join(
                f"{'R' if h['is_right'] else 'L'} {h['confidence']:.2f}" for h in record["hands"]
            ) or "No hand detected"
            put_text(canvas, labels, (16, args.height - 48), args.width - 32, 0.60)
            put_text(canvas,
                     f"frame {frame_index} | {(timestamp_ns - base_timestamp_ns) / 1e9:.3f}s"
                     f" | detector: {args.detector.name}"
                     f" | model: {model_label}",
                     (16, args.height - 18), args.width - 32, 0.54)
            writer.write(canvas)
            written += 1
            if written % 100 == 0:
                LOG.info("Rendered %d/%d frames", written, len(records))
        if written != len(records):
            raise RuntimeError(f"Rendered {written} frames, expected {len(records)}")
        success = True
    finally:
        try:
            writer.release()
        except BaseException:
            success = False
            raise
        finally:
            scene.close()
            if not success:
                temporary.unlink(missing_ok=True)
    temporary.replace(output)
    return output


def process(args) -> dict:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions = args.output_dir / f"{args.input.stem}_{args.camera}_predictions.jsonl"
    detector_classes = {0: "left", 1: "right"}
    if args.reuse_predictions:
        if not predictions.is_file():
            raise FileNotFoundError(f"Cannot reuse missing predictions: {predictions}")
        records = [json.loads(line) for line in predictions.read_text(encoding="utf-8").splitlines()
                   if line.strip()]
        if args.max_frames is not None:
            records = records[:args.max_frames]
        LOG.info("Reusing %d saved prediction frames", len(records))
    else:
        device = choose_device(args.device)
        LOG.info(
            "Loading WiLoR base checkpoint%s on %s",
            f" and adapter {args.adapter.name}" if args.adapter is not None else " without adapter",
            device,
        )
        model, model_cfg = load_wilor(
            checkpoint_path=str(args.checkpoint), cfg_path=str(args.model_config),
            mano_path=args.mano_path, adapter_path=args.adapter, init_renderer=False,
        )
        model = model.to(device).eval()
        detector = load_detector(args.detector, device)
        detector_classes = detector.names
        LOG.info("Detector %s: %s", args.detector, detector_classes)
        records = []
        for frame_index, timestamp_ns, image in iter_mcap_frames(args.input, args.camera, args.max_frames):
            hands = infer_selected_frame(image, detector, model, model_cfg, device, args)
            records.append({"frame_index": frame_index, "timestamp_ns": timestamp_ns, "hands": hands})
            if len(records) == 1 or len(records) % 25 == 0:
                LOG.info("Inferred %d frames (frame %d, %d hands)", len(records), frame_index, len(hands))
        partial_predictions = predictions.with_name(f"{predictions.stem}.partial.jsonl")
        partial_predictions.write_text(
            "".join(json.dumps(record, separators=(",", ":")) + "\n" for record in records),
            encoding="utf-8",
        )
        partial_predictions.replace(predictions)
    fps = args.fps or estimated_fps(records, 30.0)
    output = render_video(args, records, fps)
    report = {
        "input": str(args.input), "camera": args.camera,
        "detector": str(args.detector), "detector_classes": detector_classes,
        "checkpoint": str(args.checkpoint),
        "adapter": str(args.adapter) if args.adapter is not None else None,
        "model_label": args.model_label,
        "frames": len(records), "frames_with_hands": sum(bool(r["hands"]) for r in records),
        "hands": sum(len(r["hands"]) for r in records), "fps": fps,
        "coordinate_mode": args.coordinate_mode,
        "confidence": args.confidence, "iou": args.iou,
        "max_detections_per_class": args.max_detections_per_class,
        "rescale_factor": args.rescale_factor,
        "video": str(output), "predictions": str(predictions),
    }
    (args.output_dir / "run_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    LOG.info("Saved %s", output)
    return report


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--camera", default="camera2")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output/mcap_video_results")
    parser.add_argument("--detector", type=Path, default=ROOT / "pretrained_models/best.pt")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "pretrained_models/wilor_final.ckpt")
    parser.add_argument("--model-config", type=Path, default=ROOT / "pretrained_models/model_config.yaml")
    parser.add_argument("--mano-path", type=Path, default=ROOT / "mano_data/mano_v1_2")
    parser.add_argument("--adapter", type=Path,
                        default=ROOT / "output/0905_jhn_stage2_pretrained/best_adapter.pt")
    parser.add_argument("--no-adapter", action="store_true",
                        help="Use the original WiLoR checkpoint without fine-tuned weights.")
    parser.add_argument("--model-label", help="Model name shown in the video and stored in the report.")
    parser.add_argument("--confidence", type=float, default=0.3)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--rescale-factor", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-detections-per-class", type=int, default=1)
    parser.add_argument("--coordinate-mode", choices=("root-relative", "camera"),
                        default="root-relative")
    parser.add_argument("--reuse-predictions", action="store_true",
                        help="Skip inference and rerender the JSONL already in output-dir.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--fps", type=float, help="Override FPS; defaults to MCAP timestamp estimate.")
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--max-frames", type=int)
    args = parser.parse_args(argv)
    for name in ("input", "detector", "checkpoint", "model_config"):
        value = getattr(args, name).expanduser().resolve()
        setattr(args, name, value)
        if not value.is_file():
            parser.error(f"--{name.replace('_', '-')} does not exist: {value}")
    if args.no_adapter:
        args.adapter = None
    else:
        args.adapter = args.adapter.expanduser().resolve()
        if not args.adapter.is_file():
            parser.error(f"--adapter does not exist: {args.adapter}")
    args.mano_path = args.mano_path.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    camera_topic(args.camera)
    if args.input.suffix.lower() != ".mcap":
        parser.error("--input must be a .mcap file")
    if not 0 <= args.confidence <= 1 or not 0 <= args.iou <= 1:
        parser.error("--confidence and --iou must be between 0 and 1")
    if args.batch_size < 1 or args.max_detections_per_class < 0:
        parser.error("batch and detection limits must be non-negative")
    if args.fps is not None and (not np.isfinite(args.fps) or args.fps <= 0):
        parser.error("--fps must be finite and positive")
    if args.width < 960 or args.height < 600 or args.width % 2 or args.height % 2:
        parser.error("video dimensions must be even and at least 960x600")
    if args.max_frames is not None and args.max_frames < 1:
        parser.error("--max-frames must be positive")
    return args


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    process(parse_args())


if __name__ == "__main__":
    main()
