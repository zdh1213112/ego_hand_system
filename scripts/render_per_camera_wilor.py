#!/usr/bin/env python3
"""Render each camera's native WiLoR predictions and a grid of the same images."""
import argparse
from contextlib import ExitStack
import csv
import fcntl
from itertools import islice
import json
from math import ceil, sqrt
import os
from pathlib import Path
import tempfile

import cv2
import numpy as np

from ego_data.dataset import SequentialVideoReader
from fit_shared_wilor_multiview import mano_wrapper_path, signature, validate_synchronized_records
from raw_detector_overlay import draw_detector_boxes, original_detector_boxes
from render_multiview_wilor import _create_video_writer
from render_shared_gem_multiview import draw_shared_skeleton
from wilor_native_mesh_overlay import NativeManoReplay


ROOT = Path(__file__).resolve().parents[1]


def draw_observations(image, camera, record, vertices, replay, counts):
    """Use the same hypothesis for the native skeleton and the native mesh."""
    boxes = original_detector_boxes(record, image.shape[1::-1], detector_only=False)
    by_detection = {}
    for hand in record['hands']:
        by_detection.setdefault(int(hand['detection_index']), []).append(hand)
    for box in boxes:
        candidates = by_detection.get(box['detection_index'], [])
        side = box['detector_is_right']
        matches = [hand for hand in candidates if side is None or hand['is_right'] == side]
        if len(matches) != 1:
            counts['boxes_without_unique_pose'] += 1
            continue
        hand = matches[0]
        side = int(hand['is_right'])
        key = (int(record['source_frame_index']), int(hand['hypothesis_index']))
        counts['mesh_draws'] += int(replay.draw_hand_mesh(image, camera, hand, vertices[key], side))
        pixels = np.asarray(hand['joints_2d'])
        draw_shared_skeleton(image, pixels, np.isfinite(pixels).all(axis=1), side)
        counts['poses'] += 1
    counts['boxes'] += len(boxes)
    for box in boxes:
        candidates = by_detection.get(box['detection_index'], [])
        source = candidates[0].get('proposal_source', 'detector') if candidates else 'detector'
        label = 'DETECTED' if source in ('detector', 'cached_original_detector') else 'PROPOSAL'
        draw_detector_boxes(image, [box], label_prefix=label)


def render(args):
    output = args.output.resolve()
    output.mkdir(exist_ok=True, parents=True)
    directory_fd = os.open(output, os.O_RDONLY | os.O_DIRECTORY)
    try:
        fcntl.flock(directory_fd, fcntl.LOCK_EX)
        render_videos(args, output)
    finally:
        os.close(directory_fd)


def render_videos(args, output):
    dataset, predictions = args.dataset.resolve(), args.predictions.resolve()
    summary = json.loads((predictions / 'summary.json').read_text())
    manifest = json.loads((dataset / 'manifest.json').read_text())
    if Path(summary['dataset']).resolve() != dataset:
        raise ValueError('WiLoR predictions belong to a different dataset')
    if not summary.get('save_mano_params'):
        raise ValueError('Cached WiLoR predictions lack MANO parameters. Run --per-camera-video in a new output directory.')
    cameras = args.cameras or manifest['camera_ids']
    if not cameras or len(set(cameras)) != len(cameras) or not set(cameras) <= set(summary['camera_ids']):
        raise ValueError('expected distinct cameras covered by WiLoR inference')
    with (dataset / 'multiview_frames.csv').open() as stream:
        sync_rows = list(csv.DictReader(stream))
    if args.max_frames < 0:
        raise ValueError('--max-frames must be non-negative')
    rows = sync_rows[:args.max_frames or None]
    if not rows or len(rows) > summary['frame_count']:
        raise ValueError('WiLoR predictions do not cover the requested synchronized frames')

    width, height = manifest['image_size']
    single_size = (width + width % 2, height + height % 2)
    tile_width = min(960, single_size[0])
    tile_height = max(2, 2 * round(tile_width * height / width / 2))
    columns = ceil(sqrt(len(cameras)))
    grid_name = f'native_{len(cameras)}view.mp4'
    sizes = {f'{camera}.mp4': single_size for camera in cameras}
    sizes[grid_name] = (columns * tile_width, ceil(len(cameras) / columns) * tile_height)
    deltas = np.diff([int(row['reference_timestamp_ns']) for row in rows])
    fps = float(1e9 / np.median(deltas[deltas > 0])) if np.any(deltas > 0) else float(manifest['fps'])

    videos = {c: dataset / 'cameras' / c / manifest['storage']['video_filename'] for c in cameras}
    jsonls = {c: predictions / c / 'predictions.jsonl' for c in cameras}
    camera_model = (summary.get('model') or {}).get('camera_model', 'virtual_perspective')
    inputs = [dataset / 'manifest.json', dataset / 'multiview_frames.csv', predictions / 'summary.json',
              args.mano_model_dir / 'MANO_RIGHT.pkl', args.model_config, mano_wrapper_path(),
              *videos.values(), *jsonls.values()]
    if camera_model != 'virtual_perspective':
        inputs.extend(dataset / 'calibration' / f'{c}.json' for c in cameras)
    config = {'version': 1, 'camera_model': camera_model, 'cameras': cameras, 'frame_count': len(rows),
              'codec': args.video_codec, 'inputs': [signature(path) for path in inputs]}
    marker = output / 'summary.json'
    if marker.is_file():
        previous = json.loads(marker.read_text())
        if previous.get('config') == config and all(
            (output / name).is_file() and (output / name).stat().st_size == previous['files'][name]['bytes']
            for name in sizes
        ):
            print(f'[per-camera video] reuse {output}', flush=True)
            return

    replay = NativeManoReplay(summary, (width, height), args.mano_model_dir, args.model_config, args.device)
    counts = {c: dict(boxes=0, poses=0, mesh_draws=0, boxes_without_unique_pose=0) for c in cameras}
    codecs = {}
    with tempfile.TemporaryDirectory(dir=output) as temp:
        stage = Path(temp)
        with ExitStack() as stack:
            streams = {c: stack.enter_context(path.open()) for c, path in jsonls.items()}
            readers, writers = {}, {}
            for c in cameras:
                readers[c] = SequentialVideoReader(videos[c], (width, height))
                stack.callback(readers[c].close)
            for name, size in sizes.items():
                writers[name], codecs[name] = _create_video_writer(stage / name, fps, size, args.video_codec)
                stack.callback(writers[name].release)
            for start in range(0, len(rows), replay.frame_batch):
                batch = rows[start:start + replay.frame_batch]
                records, vertices = {}, {}
                for c in cameras:
                    # Read to the native batch boundary even when --max-frames ends mid-batch.
                    chunk = [json.loads(line) for line in islice(streams[c], replay.frame_batch)]
                    records[c] = {int(r['sync_index']): r for r in chunk}
                    if len(records[c]) != len(chunk):
                        raise ValueError(f'{c}: duplicate sync index in raw predictions')
                    validate_synchronized_records(batch, records[c], c, 'native video')
                    vertices[c] = replay.replay(c, chunk)
                for row in batch:
                    sync = int(row['sync_index'])
                    grid = np.zeros((sizes[grid_name][1], sizes[grid_name][0], 3), dtype=np.uint8)
                    for i, c in enumerate(cameras):
                        image = readers[c].read(int(row[f'{c}_frame_index']))
                        draw_observations(image, c, records[c][sync], vertices[c], replay, counts[c])
                        cv2.putText(image, f'{c} | NATIVE WiLoR | frame {sync}', (8, 24),
                                    cv2.FONT_HERSHEY_SIMPLEX, .6, (80, 230, 80), 2, cv2.LINE_AA)
                        if (width, height) != single_size:
                            image = cv2.copyMakeBorder(image, 0, height % 2, 0, width % 2, cv2.BORDER_CONSTANT)
                        writers[f'{c}.mp4'].write(image)
                        y, x = (i // columns) * tile_height, (i % columns) * tile_width
                        grid[y:y + tile_height, x:x + tile_width] = cv2.resize(image, (tile_width, tile_height))
                    writers[grid_name].write(grid)
                if start // replay.frame_batch % 50 == 0 or start + len(batch) == len(rows):
                    print(f'[per-camera video] {start + len(batch)}/{len(rows)}', flush=True)

        files = {name: {'size': list(size), 'codec': codecs[name], 'bytes': (stage / name).stat().st_size}
                 for name, size in sizes.items()}
        (stage / 'summary.json').write_text(json.dumps(
            {'config': config, 'fps': fps, 'files': files, 'per_camera': counts, 'replay': replay.stats}, indent=2) + '\n')
        # Publish only after every encoder has flushed. A failed publication can be retried.
        marker.unlink(missing_ok=True)
        for name in files:
            (stage / name).replace(output / name)
        (stage / 'summary.json').replace(marker)
    print(f'[per-camera video] saved {output}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('dataset', 'predictions', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--cameras', nargs='+')
    parser.add_argument('--max-frames', type=int, default=0)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--video-codec', choices=('auto', 'h264_nvenc', 'mp4v'), default='auto')
    parser.add_argument('--model-config', type=Path, default=ROOT / 'models/wilor/model_config.yaml')
    parser.add_argument('--mano-model-dir', type=Path, default=ROOT / 'models/mano')
    cv2.setNumThreads(1)
    render(parser.parse_args())
