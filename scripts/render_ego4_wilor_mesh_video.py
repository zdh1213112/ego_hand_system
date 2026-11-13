#!/usr/bin/env python3
"""Run the project WiLoR on timestamp-synchronized EGO-4B recordings.

Produces full-length four-view videos, 21-joint overlays, shaded MANO meshes,
and replayable per-view geometry. Uses detector handedness and the project's
batched inference implementation, without dropping WiLoR backbone blocks.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import ctypes
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time
import tempfile

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import wilor_multiview_inference as multi
from normalize_orbbec_ego4_session import CAMERA_SOURCES, synchronize_rows
from normalize_orbbec_session import read_timestamps
from ego_data.dataset import SequentialVideoReader
from render_wilor_predictions import HAND_CONNECTIONS

CAMERAS = tuple(CAMERA_SOURCES)
COLORS = ((245, 205, 65), (65, 170, 255))
FINGER_COLORS = ((95, 110, 255), (80, 205, 255), (110, 240, 100),
                 (255, 210, 65), (235, 110, 235))
LIGHT = np.asarray((-.3, -.5, -.8), np.float32)
LIGHT /= np.linalg.norm(LIGHT)
NATIVE_RENDER = None


def load_native_renderer():
    """Compile the dependency-free renderer once; the portable fallback remains available."""
    global NATIVE_RENDER
    source = Path(__file__).with_name("wilor_mesh_rasterizer.cpp")
    key = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    cache = Path(tempfile.gettempdir()) / "ego4-wilor-mesh-rasterizer"
    cache.mkdir(exist_ok=True)
    library = cache / f"mesh_{key}.so"
    try:
        if not library.exists():
            subprocess.run(["g++", "-O3", "-std=c++17", "-shared", "-fPIC",
                            str(source), "-o", str(library)], check=True, capture_output=True)
        function = ctypes.CDLL(str(library)).render_wilor_mesh
        f32 = np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS")
        function.argtypes = [np.ctypeslib.ndpointer(dtype=np.uint8, flags="C_CONTIGUOUS"),
                             ctypes.c_int, ctypes.c_int, f32, f32, ctypes.c_int,
                             np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS"),
                             ctypes.c_int, f32, ctypes.c_float, ctypes.c_int]
        function.restype = None
        NATIVE_RENDER = function
        print("Mesh renderer: native CPU depth buffer", flush=True)
    except (OSError, subprocess.CalledProcessError) as error:
        print(f"Mesh renderer: portable OpenCV fallback ({error})", flush=True)


def signature(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return dict(path=str(path.resolve()), bytes=path.stat().st_size,
                sha256=digest.hexdigest())


def discover_session(session):
    """Read native partNNNN video/PTS files, preserving their real timestamps."""
    sources, timestamps, sizes = {}, {}, {}
    for camera, (stream, _) in CAMERA_SOURCES.items():
        files = sorted(session.glob(f"part*/*_camera_{stream}_part*.mp4"))
        if not files:
            files = sorted(session.glob(f"*_camera_{stream}.mp4"))
        if not files:
            raise FileNotFoundError(f"missing {camera} video in {session}")
        sources[camera], timestamps[camera] = [], []
        for video in files:
            pts = video.with_name(video.stem + "_pts.csv")
            values = read_timestamps(pts)
            cap = cv2.VideoCapture(str(video))
            try:
                count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
                if not cap.isOpened() or count != len(values):
                    raise ValueError(f"video/PTS count mismatch: {video}: {count}/{len(values)}")
            finally:
                cap.release()
            if camera in sizes and sizes[camera] != size:
                raise ValueError(f"part size mismatch: {video}")
            sizes[camera] = size
            sources[camera].append((video, count))
            timestamps[camera].extend(values)
        if np.any(np.diff(timestamps[camera]) <= 0):
            raise ValueError(f"non-monotonic part timestamps in {session}/{camera}")
    rows = synchronize_rows(timestamps, "camera0", 1_500_000)
    if not rows:
        raise ValueError(f"no synchronized frames in {session}")
    return sources, sizes, rows, {c: len(v) for c, v in timestamps.items()}


class PartReader:
    def __init__(self, files, size):
        self.files, self.size = files, size
        self.part, self.offset, self.reader = 0, 0, None

    def read(self, index):
        while index >= self.offset + self.files[self.part][1]:
            self.offset += self.files[self.part][1]
            self.part += 1
            self.close()
        if self.reader is None:
            self.reader = SequentialVideoReader(self.files[self.part][0], self.size)
        return self.reader.read(index - self.offset)

    def close(self):
        if self.reader is not None:
            self.reader.close()
            self.reader = None


def shaded_mesh(image, vertices, pixels, faces, color, alpha, wire=False):
    """Painter-ordered triangles with camera-space normal lighting."""
    vertices = np.asarray(vertices, np.float32)
    if not np.isfinite(vertices).all() or not np.isfinite(pixels).all():
        return
    if NATIVE_RENDER is not None:
        NATIVE_RENDER(image, image.shape[1], image.shape[0],
                      np.ascontiguousarray(vertices, np.float32),
                      np.ascontiguousarray(pixels, np.float32), len(vertices),
                      np.ascontiguousarray(faces, np.int32), len(faces),
                      np.asarray(color, np.float32), alpha, int(wire))
        return
    triangles = vertices[faces]
    polygons = np.rint(np.clip(pixels[faces], -100000, 100000)).astype(np.int32)
    normal = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    length = np.linalg.norm(normal, axis=1)
    lighting = np.divide(np.abs(normal @ LIGHT), length,
                         out=np.zeros_like(length), where=length > 1e-10)
    shades = np.clip(np.asarray(color) * (.40 + .60 * lighting[:, None]) + 10,
                     0, 255).astype(np.uint8)
    h, w = image.shape[:2]
    keep = ((triangles[:, :, 2].min(1) > .01)
            & (polygons[:, :, 0].max(1) >= 0) & (polygons[:, :, 0].min(1) < w)
            & (polygons[:, :, 1].max(1) >= 0) & (polygons[:, :, 1].min(1) < h)
            & (length > 1e-10))
    indices = np.flatnonzero(keep)
    indices = indices[np.argsort(triangles[indices, :, 2].mean(1))[::-1]]
    layer = image.copy()
    for i in indices:
        cv2.fillConvexPoly(layer, polygons[i], tuple(int(x) for x in shades[i]), cv2.LINE_AA)
    if wire:
        cv2.polylines(layer, list(polygons[indices[::9]]), True,
                      tuple(int(x * .55) for x in color), 1, cv2.LINE_AA)
    cv2.addWeighted(layer, alpha, image, 1 - alpha, 0, image)


def mesh_inset(tile, hands, faces):
    w, h = 242, 172
    panel = np.full((h, w, 3), (28, 33, 41), np.uint8)
    cv2.putText(panel, "3D MANO", (9, 19), cv2.FONT_HERSHEY_SIMPLEX,
                .45, (228, 235, 240), 1, cv2.LINE_AA)
    # A fixed oblique view makes the predicted surface depth visible.
    ay, ax = np.deg2rad(-30), np.deg2rad(18)
    ry = np.array(((np.cos(ay), 0, np.sin(ay)), (0, 1, 0),
                   (-np.sin(ay), 0, np.cos(ay))), np.float32)
    rx = np.array(((1, 0, 0), (0, np.cos(ax), -np.sin(ax)),
                   (0, np.sin(ax), np.cos(ax))), np.float32)
    for hand in hands:
        side = int(hand["is_right"])
        vertices = np.asarray(hand["mano_vertices_camera_relative"], np.float32).copy()
        root = np.asarray(hand["joints_3d_camera_relative"], np.float32)[0]
        vertices = (vertices - root) @ (rx @ ry).T
        # Fixed scale within every recording; no independent per-frame rescaling.
        xy = vertices[:, :2] * 560 + (61 + side * 120, 138)
        vertices[:, 2] += 1
        shaded_mesh(panel, vertices, xy, faces, COLORS[side], 1.0)
        cv2.putText(panel, "L" if side == 0 else "R", (12 + side * 120, h - 9),
                    cv2.FONT_HERSHEY_SIMPLEX, .45, COLORS[side], 1, cv2.LINE_AA)
    tile[8:8+h, tile.shape[1]-w-8:tile.shape[1]-8] = panel


def draw_tile(item, faces, focal, tile_width):
    frame, hands, camera = item
    height = round(frame.shape[0] * tile_width / frame.shape[1])
    tile = cv2.resize(frame, (tile_width, height), interpolation=cv2.INTER_AREA)
    scale = tile_width / frame.shape[1]
    for hand in sorted(hands, key=lambda v: v["camera_translation"][2], reverse=True):
        side = int(hand["is_right"])
        vertices = np.asarray(hand["mano_vertices_camera_relative"], np.float32)
        translated = vertices + np.asarray(hand["camera_translation"], np.float32)
        xy = multi.base.project_points(vertices, np.asarray(hand["camera_translation"]),
                                       focal, np.asarray(frame.shape[1::-1])) * scale
        shaded_mesh(tile, translated, xy, faces, COLORS[side], .53, wire=True)
        points = np.rint(np.asarray(hand["joints_2d"]) * scale).astype(np.int32)
        for i, (a, b) in enumerate(HAND_CONNECTIONS):
            p, q = tuple(points[a]), tuple(points[b])
            if np.max(np.abs(points[[a, b]])) > 100000:
                continue
            cv2.line(tile, p, q, (25, 25, 25), 4, cv2.LINE_AA)
            cv2.line(tile, p, q, FINGER_COLORS[i // 4], 2, cv2.LINE_AA)
        for point in points:
            if np.max(np.abs(point)) < 100000:
                cv2.circle(tile, tuple(point), 3, (250, 250, 250), -1, cv2.LINE_AA)
        box = np.rint(np.asarray(hand["bbox_xyxy"]) * scale).astype(int)
        cv2.rectangle(tile, tuple(box[:2]), tuple(box[2:]), COLORS[side], 1, cv2.LINE_AA)
        text = f"{'L' if side == 0 else 'R'} {hand['confidence']:.2f}"
        cv2.putText(tile, text, (max(5, box[0]), max(23, box[1]-7)),
                    cv2.FONT_HERSHEY_SIMPLEX, .6, COLORS[side], 2, cv2.LINE_AA)
    if tile_width >= 800:
        mesh_inset(tile, hands, faces)
    header = np.full((32, tile_width, 3), (25, 30, 38), np.uint8)
    stream = CAMERA_SOURCES[camera][0]
    label = f"{camera} / {stream}  |  {len(hands)} hands  |  21 joints + MANO mesh"
    cv2.putText(header, label, (12, 22), cv2.FONT_HERSHEY_SIMPLEX,
                .55, (238, 240, 244), 1, cv2.LINE_AA)
    return np.concatenate((header, tile), axis=0)


def open_encoder(path, size, fps):
    # Identical H.264 settings for all sessions allow lossless concatenation.
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
           "-pixel_format", "bgr24", "-video_size", f"{size[0]}x{size[1]}",
           "-framerate", str(fps), "-i", "pipe:0", "-an", "-c:v", "libx264",
           "-threads", "4", "-preset", "veryfast", "-crf", "20",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def process_session(session, output, args, model, config, detector, faces, device, executor):
    target = output / session.parent.name
    target.mkdir(exist_ok=True)
    completed = target / "summary.json"
    if completed.exists():
        result = json.loads(completed.read_text())
        if result.get("run_config") != args.run_signature:
            raise ValueError(f"different run settings in {target}; select a new output")
        print(f"Reuse completed {session.parent.name}", flush=True)
        return result
    sources, sizes, rows, source_counts = discover_session(session)
    full_count = len(rows)
    if args.max_frames:
        rows = rows[:args.max_frames]
    n = len(rows)
    with (target / "synchronized_frames.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if len(set(sizes.values())) != 1:
        raise ValueError("all four cameras must have matching image sizes")
    width, height = sizes["camera0"]
    focal = float(config.EXTRA.FOCAL_LENGTH / config.MODEL.IMAGE_SIZE * max(width, height))
    geometry = {
        "valid": np.zeros((n, 4, 2), bool),
        "confidence": np.zeros((n, 4, 2), np.float16),
        "bbox_xyxy": np.full((n, 4, 2, 4), np.nan, np.float32),
        "camera_translation": np.full((n, 4, 2, 3), np.nan, np.float32),
        "joints_2d": np.full((n, 4, 2, 21, 2), np.nan, np.float32),
        "joints_3d_camera_relative": np.full((n, 4, 2, 21, 3), np.nan, np.float32),
        "vertices_camera_relative": np.full((n, 4, 2, 778, 3), np.nan, np.float16),
    }
    out_height = 2 * (32 + round(height * args.tile_width / width)) + 40
    video = target / "wilor_joints_mesh_4view.mp4"
    partial = target / "wilor_joints_mesh_4view.partial.mp4"
    encoder = open_encoder(partial, (2 * args.tile_width, out_height), args.fps)
    readers = {c: PartReader(sources[c], sizes[c]) for c in CAMERAS}
    runtime = {"effective_hypothesis_batch_size": args.batch_size, "cuda_oom_retries": 0}
    start = time.perf_counter()
    checkpoints = {0, n // 3, n // 2, 2 * n // 3, n - 1}
    print(f"Begin {session.parent.name}: {n}/{full_count} synchronized frames", flush=True)
    stage_seconds = dict(decode=0., detector_crop=0., wilor=0., render_encode=0.)

    def render_chunk(first, chunk, images, predictions):
        # A single rendering worker avoids Python/OpenCV thread contention and
        # runs alongside the following CUDA batch. Keep at most one queued batch.
        render_start = time.perf_counter()
        for j, row in enumerate(chunk):
            group = [draw_tile((images[j*4+i], predictions[j*4+i], c),
                               faces, focal, args.tile_width) for i, c in enumerate(CAMERAS)]
            ordinal = first + j
            mosaic = np.vstack((np.hstack(group[:2]), np.hstack(group[2:])))
            footer = np.full((40, mosaic.shape[1], 3), (20, 25, 32), np.uint8)
            elapsed = (int(row["reference_timestamp_ns"]) - int(rows[0]["reference_timestamp_ns"])) / 1e9
            title = (f"{session.parent.name}  |  {elapsed:.2f}s  |  frame {ordinal+1}/{n}"
                     "  |  detector: 2.0_glove_2906.pt  |  WiLoR")
            cv2.putText(footer, title, (15, 27), cv2.FONT_HERSHEY_SIMPLEX,
                        .6, (230, 235, 240), 1, cv2.LINE_AA)
            frame = np.concatenate((mosaic, footer), axis=0)
            encoder.stdin.write(frame.tobytes())
            if ordinal in checkpoints:
                cv2.imwrite(str(target / f"preview_{ordinal:06d}.jpg"), frame)
            for camera_index in range(4):
                for hand in predictions[j*4+camera_index]:
                    side = int(hand["is_right"])
                    ix = (ordinal, camera_index, side)
                    geometry["valid"][ix] = True
                    for key in ("confidence", "bbox_xyxy", "camera_translation",
                                "joints_2d", "joints_3d_camera_relative"):
                        geometry[key][ix] = hand[key]
                    geometry["vertices_camera_relative"][ix] = hand["mano_vertices_camera_relative"]
        stage_seconds["render_encode"] += time.perf_counter() - render_start

    try:
        with ThreadPoolExecutor(max_workers=1) as render_executor:
            pending = None
            for first in range(0, n, args.frame_batch):
                stage = time.perf_counter()
                chunk = rows[first:first+args.frame_batch]
                images = [readers[c].read(int(row[f"{c}_frame_index"]))
                          for row in chunk for c in CAMERAS]
                stage_seconds["decode"] += time.perf_counter() - stage
                stage = time.perf_counter()
                prepared = multi._prepare_dual_batched(
                    images, detector, config, args.confidence, .5, 2.0,
                    args.workers, 1, preprocess_executor=executor, dual_hypothesis=False)
                stage_seconds["detector_crop"] += time.perf_counter() - stage
                stage = time.perf_counter()
                predictions = multi._infer_prepared_batched(
                    prepared, model, config, device, args.batch_size,
                    runtime_stats=runtime, full_output=True, include_vertices=True)
                stage_seconds["wilor"] += time.perf_counter() - stage
                if pending is not None:
                    pending.result()
                pending = render_executor.submit(render_chunk, first, chunk, images, predictions)
                processed = first + len(chunk)
                if first == 0 or processed % 60 < args.frame_batch or processed == n:
                    seconds = time.perf_counter() - start
                    rate = processed / seconds
                    print(f"{session.parent.name}: {processed}/{n} | {rate:.2f} sync frames/s"
                          f" | ETA {(n-processed)/max(rate, .01)/60:.1f} min", flush=True)
            if pending is not None:
                pending.result()
    finally:
        for reader in readers.values():
            reader.close()
        encoder.stdin.close()
        code = encoder.wait()
    if code:
        raise RuntimeError(f"ffmpeg exited {code}")
    partial.rename(video)
    np.savez_compressed(target / "wilor_geometry.npz", **geometry, faces=faces,
                        camera_ids=np.asarray(CAMERAS), handedness=np.asarray(("left", "right")),
                        reference_timestamp_ns=np.asarray([r["reference_timestamp_ns"] for r in rows], np.int64),
                        focal_length_px=np.asarray(focal), image_size=np.asarray((width, height)))
    detection = geometry["valid"]
    result = dict(source_session=str(session), video=str(video.resolve()),
                  frames=n, full_synchronized_frames=full_count,
                  source_frame_counts=source_counts, fps=args.fps,
                  video_size=[2*args.tile_width, out_height],
                  inference_seconds=time.perf_counter()-start,
                  detected_frames_by_camera=dict(zip(CAMERAS, detection.any(2).sum(0).tolist())),
                  detected_hands_by_camera=dict(zip(CAMERAS, detection.sum((0, 2)).tolist())),
                  max_sync_delta_us=max(abs(float(r[f"{c}_delta_us"])) for r in rows for c in CAMERAS),
                  geometry="wilor_geometry.npz", runtime=runtime, stage_seconds=stage_seconds,
                  mesh_renderer="native_depth_buffer" if NATIVE_RENDER is not None else "opencv_painter",
                  run_config=args.run_signature,
                  reconstruction="independent per-camera WiLoR; detector handedness; original backbone; no NOKOV fitting")
    completed.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--detector", type=Path, default=ROOT/"models/wilor/2.0_glove_2906.pt")
    parser.add_argument("--checkpoint", type=Path, default=ROOT/"models/wilor/wilor_final.ckpt")
    parser.add_argument("--model-config", type=Path, default=ROOT/"models/wilor/model_config.yaml")
    parser.add_argument("--mano-model-dir", type=Path, default=ROOT/"models/mano")
    parser.add_argument("--session-filter", default="*")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--frame-batch", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--tile-width", type=int, default=960)
    parser.add_argument("--confidence", type=float, default=.3)
    parser.add_argument("--fps", type=float, default=30)
    args = parser.parse_args()
    load_native_renderer()
    if min(args.frame_batch, args.batch_size, args.workers, args.tile_width, args.fps) <= 0 or args.max_frames < 0:
        parser.error("batch, workers, tile width and FPS must be positive; max frames must be non-negative")
    args.input, args.output = args.input.resolve(), args.output.resolve()
    for key in ("detector", "checkpoint", "model_config", "mano_model_dir"):
        setattr(args, key, getattr(args, key).resolve())
    if args.output == args.input or args.output in args.input.parents:
        parser.error("output must be separate from input")
    args.output.mkdir(parents=True, exist_ok=True)
    sessions = sorted(args.input.glob(f"{args.session_filter}/EGO-4B*"))
    if not sessions:
        raise FileNotFoundError(f"no EGO-4B sessions in {args.input}")
    settings = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    settings["detector_signature"] = signature(args.detector)
    settings["checkpoint_signature"] = signature(args.checkpoint)
    args.run_signature = settings
    config_path = args.output / "run_config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != settings:
        raise ValueError("output has different run settings; select a new output")
    config_path.write_text(json.dumps(settings, indent=2) + "\n")
    import torch
    cv2.setNumThreads(1)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    device = multi.base.choose_device("cuda")
    model, config = multi.base.load_wilor_model(args.checkpoint, args.model_config, args.mano_model_dir)
    model = model.to(device).eval().half()
    detector = multi.base.load_detector(args.detector, device)
    if detector.names != {0: "left", 1: "right"}:
        raise ValueError(f"expected left/right detector classes; found {detector.names}")
    faces = np.asarray(model.mano.faces, np.int32)
    print(f"Loaded WiLoR + {args.detector.name}; {len(sessions)} sessions", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        results = [process_session(s, args.output, args, model, config, detector,
                                   faces, device, executor) for s in sessions]
    concat = args.output / "concat.txt"
    concat.write_text("".join("file '" + r["video"].replace("'", "'\\''") + "'\n" for r in results))
    combined = args.output / "ego4b_wilor_joints_mesh_all_4view.mp4"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "concat",
                    "-safe", "0", "-i", str(concat), "-c", "copy", "-movflags", "+faststart", str(combined)], check=True)
    play = args.output / "play_loop.sh"
    play.write_text("#!/usr/bin/env bash\nset -euo pipefail\n"
                    "exec env -u LD_LIBRARY_PATH -u QT_PLUGIN_PATH -u QT_QPA_PLATFORM_PLUGIN_PATH "
                    "-u LIBVA_DRIVERS_PATH -u LIBGL_DRIVERS_PATH /usr/bin/vlc "
                    "--no-one-instance --loop --no-video-title-show --avcodec-hw=none --vout=xcb_x11 "
                    + shlex.quote(str(combined)) + "\n")
    play.chmod(0o755)
    report = dict(video=str(combined), sessions=results,
                  total_frames=sum(r["frames"] for r in results), fps=args.fps,
                  duration_seconds=sum(r["frames"] for r in results)/args.fps,
                  loop_player=str(play))
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "sessions"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
