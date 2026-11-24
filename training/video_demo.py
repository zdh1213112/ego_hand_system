"""Run WiLoR on one video or a directory tree containing videos."""

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from wilor.datasets.vitdet_dataset import ViTDetDataset
from wilor.models import load_wilor
from wilor.utils import recursive_to
from wilor.utils.renderer import Renderer, cam_crop_to_full
from wilor.utils.visualization import LEFT_COLOR, RIGHT_COLOR, draw_hand_overlay


LIGHT_PURPLE = (0.25098039, 0.274117647, 0.65882353)
VIDEO_SUFFIXES = {".mkv", ".mp4", ".avi", ".mov", ".webm", ".m4v"}
REPO_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run hand detection and WiLoR reconstruction on videos."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "input_test"
        ),
        help="Input video, or a directory searched recursively for videos.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "output/video_results",
        help="Root directory for rendered videos and JSONL predictions.",
    )
    parser.add_argument(
        "--mano-path",
        type=Path,
        default=Path("mano_data/mano_v1_2"),
        help="MANO_RIGHT.pkl, its directory, or an extracted MANO release.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=REPO_ROOT / "pretrained_models/wilor_final.ckpt",
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        default=REPO_ROOT / "pretrained_models/model_config.yaml",
    )
    parser.add_argument(
        "--detector",
        type=Path,
        default=REPO_ROOT / "pretrained_models/best.pt",
    )
    parser.add_argument(
        "--adapter",
        type=Path,
        default=None,
        help="Optional fine-tuned WiLoR adapter checkpoint.",
    )
    parser.add_argument("--confidence", type=float, default=0.3)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--rescale-factor", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device, for example 'cuda', 'cuda:0', or 'cpu'.",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use WiLoR's FP16, compiled-backbone fast inference mode.",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="Process every Nth source frame; output FPS is adjusted accordingly.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum number of processed frames per video (useful for a quick check).",
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Skip keypoint/mesh overlays but still draw hand detector boxes.",
    )
    parser.add_argument(
        "--render-mesh",
        action="store_true",
        help="Additionally overlay the MANO mesh; off by default.",
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Only save JSONL predictions, without an annotated video.",
    )
    parser.add_argument(
        "--save-meshes",
        action="store_true",
        help="Save every reconstructed hand as an OBJ file.",
    )
    parser.add_argument(
        "--save-vertices",
        action="store_true",
        help="Include all 778 MANO vertices in JSONL (substantially larger files).",
    )
    args = parser.parse_args()
    if args.frame_stride < 1:
        parser.error("--frame-stride must be at least 1")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.max_frames is not None and args.max_frames < 1:
        parser.error("--max-frames must be at least 1")
    return args


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False")
    return device


def load_detector(checkpoint_path: Path, device: torch.device) -> YOLO:
    """Load old Ultralytics checkpoints on PyTorch versions using weights_only."""
    original_torch_load = torch.load

    def trusted_checkpoint_load(*args, **kwargs):
        # detector.pt is the repository's explicitly supplied, trusted checkpoint.
        kwargs.setdefault("weights_only", False)
        return original_torch_load(*args, **kwargs)

    torch.load = trusted_checkpoint_load
    try:
        return YOLO(str(checkpoint_path)).to(device)
    finally:
        torch.load = original_torch_load


def find_videos(input_path: Path) -> Tuple[Path, List[Path]]:
    input_path = input_path.expanduser().resolve()
    if input_path.is_file():
        if input_path.suffix.lower() not in VIDEO_SUFFIXES:
            raise ValueError(f"Unsupported video extension: {input_path}")
        return input_path.parent, [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input does not exist: {input_path}")
    videos = sorted(
        path for path in input_path.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
    )
    if not videos:
        raise FileNotFoundError(f"No videos found under {input_path}")
    return input_path, videos


def load_timestamps(video_path: Path) -> Dict[int, Dict[str, object]]:
    timestamp_path = video_path.with_name("timestamps.csv")
    if not timestamp_path.is_file():
        return {}
    timestamps: Dict[int, Dict[str, object]] = {}
    with timestamp_path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            frame_index = int(row.pop("frame_index"))
            converted: Dict[str, object] = {}
            for key, value in row.items():
                try:
                    converted[key] = int(value)
                except (TypeError, ValueError):
                    converted[key] = value
            timestamps[frame_index] = converted
    return timestamps


def project_points(
    points: np.ndarray,
    camera_translation: np.ndarray,
    focal_length: float,
    image_size: Sequence[float],
) -> np.ndarray:
    width, height = float(image_size[0]), float(image_size[1])
    camera_points = points + camera_translation[None, :]
    depth = np.clip(camera_points[:, 2:3], 1e-8, None)
    xy = camera_points[:, :2] / depth
    xy *= focal_length
    xy += np.array([[width / 2.0, height / 2.0]], dtype=np.float32)
    return xy


def infer_frame(
    frame: np.ndarray,
    detector: YOLO,
    model: torch.nn.Module,
    model_cfg,
    device: torch.device,
    confidence: float,
    iou: float,
    rescale_factor: float,
    batch_size: int,
    fp16: bool,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    List[Dict[str, object]],
    List[np.ndarray],
    List[np.ndarray],
    float,
]:
    detection_result = detector(frame, conf=confidence, iou=iou, verbose=False)[0]
    if detection_result.boxes is None or len(detection_result.boxes) == 0:
        empty_boxes = np.empty((0, 4), dtype=np.float32)
        empty_values = np.empty((0,), dtype=np.float32)
        return empty_boxes, empty_values, empty_values, [], [], [], 0.0

    boxes = detection_result.boxes.xyxy.detach().cpu().numpy().astype(np.float32)
    confidences = detection_result.boxes.conf.detach().cpu().numpy().astype(np.float32)
    handedness = detection_result.boxes.cls.detach().cpu().numpy().astype(np.float32)

    dataset = ViTDetDataset(
        model_cfg,
        frame,
        boxes,
        handedness,
        rescale_factor=rescale_factor,
        fp16=fp16,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    hand_records: List[Optional[Dict[str, object]]] = [None] * len(dataset)
    render_vertices: List[Optional[np.ndarray]] = [None] * len(dataset)
    render_cameras: List[Optional[np.ndarray]] = [None] * len(dataset)
    focal_length = 0.0

    for batch in dataloader:
        batch = recursive_to(batch, device)
        with torch.inference_mode():
            output = model(batch)

        multiplier = 2 * batch["right"] - 1
        pred_cam = output["pred_cam"].clone()
        pred_cam[:, 1] = multiplier * pred_cam[:, 1]
        box_center = batch["box_center"].float()
        box_size = batch["box_size"].float()
        image_size = batch["img_size"].float()
        focal_length = float(
            model_cfg.EXTRA.FOCAL_LENGTH
            / model_cfg.MODEL.IMAGE_SIZE
            * image_size[0].max().item()
        )
        camera_full = (
            cam_crop_to_full(pred_cam, box_center, box_size, image_size, focal_length)
            .detach()
            .cpu()
            .numpy()
        )

        for batch_index in range(batch["img"].shape[0]):
            detection_index = int(batch["personid"][batch_index].item())
            is_right = int(round(float(batch["right"][batch_index].item())))
            side_multiplier = 2 * is_right - 1
            vertices = (
                output["pred_vertices"][batch_index]
                .detach()
                .float()
                .cpu()
                .numpy()
            )
            joints_3d = (
                output["pred_keypoints_3d"][batch_index]
                .detach()
                .float()
                .cpu()
                .numpy()
            )
            vertices[:, 0] *= side_multiplier
            joints_3d[:, 0] *= side_multiplier
            camera_translation = camera_full[batch_index].astype(np.float32)
            joints_2d = project_points(
                joints_3d,
                camera_translation,
                focal_length,
                image_size[batch_index].detach().cpu().numpy(),
            )
            mano_parameters = {
                key: value[batch_index].detach().float().cpu().numpy().tolist()
                for key, value in output["pred_mano_params"].items()
            }
            hand_records[detection_index] = {
                "detection_index": detection_index,
                "bbox_xyxy": boxes[detection_index].tolist(),
                "confidence": float(confidences[detection_index]),
                "handedness": "right" if is_right else "left",
                "is_right": is_right,
                "camera_translation": camera_translation.tolist(),
                "joints_3d": joints_3d.tolist(),
                "joints_2d": joints_2d.tolist(),
                "mano": mano_parameters,
            }
            render_vertices[detection_index] = vertices
            render_cameras[detection_index] = camera_translation

    return (
        boxes,
        confidences,
        handedness,
        [record for record in hand_records if record is not None],
        [vertices for vertices in render_vertices if vertices is not None],
        [camera for camera in render_cameras if camera is not None],
        focal_length,
    )


def make_video_writer(
    path: Path, fps: float, width: int, height: int
) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create output video: {path}")
    return writer


def process_video(
    video_path: Path,
    output_dir: Path,
    detector: YOLO,
    model: torch.nn.Module,
    model_cfg,
    renderer: Optional[Renderer],
    device: torch.device,
    args: argparse.Namespace,
) -> None:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    if source_fps <= 0:
        source_fps = 30.0
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    output_fps = source_fps / args.frame_stride
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / f"{video_path.stem}_wilor.jsonl"
    video_output_path = output_dir / f"{video_path.stem}_wilor.mp4"
    mesh_dir = output_dir / f"{video_path.stem}_meshes"
    if args.save_meshes:
        mesh_dir.mkdir(parents=True, exist_ok=True)

    timestamps = load_timestamps(video_path)
    writer = None if args.no_video else make_video_writer(
        video_output_path, output_fps, width, height
    )
    processed_frames = 0
    source_frame_index = -1
    print(
        f"Processing {video_path} ({width}x{height}, {source_fps:.3f} FPS, "
        f"{total_frames} frames)"
    )

    try:
        with jsonl_path.open("w", encoding="utf-8") as jsonl_file:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                source_frame_index += 1
                if source_frame_index % args.frame_stride != 0:
                    continue
                if args.max_frames is not None and processed_frames >= args.max_frames:
                    break

                (
                    boxes,
                    confidences,
                    handedness,
                    hands,
                    vertices,
                    cameras,
                    focal_length,
                ) = infer_frame(
                    frame,
                    detector,
                    model,
                    model_cfg,
                    device,
                    args.confidence,
                    args.iou,
                    args.rescale_factor,
                    args.batch_size,
                    args.fast,
                )

                if args.save_vertices:
                    for hand, hand_vertices in zip(hands, vertices):
                        hand["vertices"] = hand_vertices.tolist()

                if args.save_meshes and renderer is not None:
                    for hand_index, (hand_vertices, camera, is_right) in enumerate(
                        zip(vertices, cameras, handedness)
                    ):
                        mesh = renderer.vertices_to_trimesh(
                            hand_vertices,
                            camera.copy(),
                            LIGHT_PURPLE,
                            is_right=int(round(float(is_right))),
                        )
                        mesh.export(
                            mesh_dir
                            / f"frame_{source_frame_index:06d}_hand_{hand_index:02d}.obj"
                        )

                if args.render_mesh and renderer is not None and vertices:
                    rgba = renderer.render_rgba_multiple(
                        vertices,
                        cam_t=cameras,
                        render_res=[width, height],
                        is_right=[int(round(float(value))) for value in handedness],
                        mesh_base_color=LIGHT_PURPLE,
                        scene_bg_color=(1, 1, 1),
                        focal_length=focal_length,
                    )
                    rgb = frame[:, :, ::-1].astype(np.float32) / 255.0
                    alpha = rgba[:, :, 3:4]
                    composited = rgb * (1.0 - alpha) + rgba[:, :, :3] * alpha
                    frame = np.clip(composited[:, :, ::-1] * 255.0, 0, 255).astype(
                        np.uint8
                    )

                hands_by_detection = {
                    int(hand["detection_index"]): hand for hand in hands
                }
                visual_hands = []
                for detection_index, (box, confidence, handedness_value) in enumerate(
                    zip(boxes, confidences, handedness)
                ):
                    is_right = bool(round(float(handedness_value)))
                    hand = hands_by_detection.get(detection_index)
                    color = RIGHT_COLOR if is_right else LEFT_COLOR
                    visual_hands.append({
                        "bbox": box,
                        "joints_2d": (
                            hand.get("joints_2d")
                            if hand is not None and not args.no_render else None
                        ),
                        "is_right": is_right,
                        "label": f"{'R' if is_right else 'L'} {float(confidence):.3f}",
                        "pred_color": color,
                    })
                frame = draw_hand_overlay(frame, visual_hands, show_legend=False)
                if writer is not None:
                    writer.write(frame)

                record: Dict[str, object] = {
                    "frame_index": source_frame_index,
                    "processed_frame_index": processed_frames,
                    "source_time_seconds": source_frame_index / source_fps,
                    "num_hands": len(hands),
                    "hands": hands,
                }
                if source_frame_index in timestamps:
                    record.update(timestamps[source_frame_index])
                jsonl_file.write(json.dumps(record, separators=(",", ":")) + "\n")
                jsonl_file.flush()

                processed_frames += 1
                if processed_frames == 1 or processed_frames % 25 == 0:
                    print(
                        f"  {video_path.name}: processed {processed_frames} frames "
                        f"(source frame {source_frame_index})"
                    )
    finally:
        capture.release()
        if writer is not None:
            writer.release()

    print(f"Saved predictions: {jsonl_path}")
    if writer is not None:
        print(f"Saved video:       {video_output_path}")


def main() -> None:
    args = parse_args()
    input_root, videos = find_videos(args.input)
    device = choose_device(args.device)
    print(f"Using device: {device}")
    print(f"Found {len(videos)} video(s)")

    model, model_cfg = load_wilor(
        checkpoint_path=str(args.checkpoint),
        cfg_path=str(args.model_config),
        mano_path=args.mano_path,
        adapter_path=args.adapter,
    )
    if args.fast:
        if device.type != "cuda":
            raise RuntimeError("--fast currently requires a CUDA device")
        torch.set_float32_matmul_precision("high")
        model = model.half()
        model.backbone = torch.compile(model.backbone)
        model.backbone.skip_blocks = True
    model = model.to(device).eval()
    detector = load_detector(args.detector, device)
    needs_renderer = args.render_mesh or args.save_meshes
    renderer = Renderer(model_cfg, faces=model.mano.faces) if needs_renderer else None

    for video_path in videos:
        relative_parent = video_path.parent.relative_to(input_root)
        output_dir = args.output.expanduser().resolve() / relative_parent
        process_video(
            video_path,
            output_dir,
            detector,
            model,
            model_cfg,
            renderer,
            device,
            args,
        )


if __name__ == "__main__":
    main()
