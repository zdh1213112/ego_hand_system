#!/usr/bin/env python3
"""Convert YOLO left/right detections into GEM-X frame-indexed hand boxes."""
import argparse
import csv
import json
import os
from pathlib import Path

import cv2
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--camera', required=True)
    p.add_argument('--detector', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--max-frames', type=int, default=0)
    p.add_argument('--confidence', type=float, default=0.25)
    p.add_argument('--box-scale', type=float, default=1.3)
    p.add_argument('--image-size', type=int, default=640)
    args = p.parse_args()
    if args.max_frames < 0 or not 0 < args.confidence <= 1 or args.box_scale <= 0:
        p.error('invalid frame count, confidence or box scale')
    os.environ.setdefault('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD', '1')
    os.environ.setdefault('YOLO_CONFIG_DIR', '/tmp/ego-gem-ultralytics')
    from ultralytics import YOLO
    model = YOLO(str(args.detector))
    names = {int(k): str(v).lower() for k, v in model.names.items()}
    if names != {0: 'left', 1: 'right'}:
        raise ValueError(f'expected detector classes 0=left, 1=right; got {names}')
    manifest = json.loads((args.dataset / 'manifest.json').read_text())
    calibration = json.loads((args.dataset / 'calibration' / f'{args.camera}.json').read_text())
    if calibration['model'] != 'PINHOLE':
        raise ValueError('YOLO and GEM must use the same pinhole video and calibration')
    with (args.dataset / 'multiview_frames.csv').open(newline='') as f:
        rows = list(csv.DictReader(f))[:args.max_frames or None]
    indices = [int(row[f'{args.camera}_frame_index']) for row in rows]
    if not indices or any(b <= a for a, b in zip(indices, indices[1:])):
        raise ValueError('expected strictly increasing synchronized frame indices')
    video = args.dataset / 'cameras' / args.camera / manifest['storage']['video_filename']
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise OSError(f'cannot open {video}')
    frames = {}
    current = 0
    try:
        for ordinal, target in enumerate(indices):
            image = None
            while current <= target:
                ok, image = cap.read()
                if not ok:
                    raise OSError(f'{video} ended before frame {target}')
                current += 1
            result = model.predict(image, conf=args.confidence, device=args.device,
                                   imgsz=args.image_size, verbose=False)[0]
            best = {}
            for xyxy, cls, conf in zip(result.boxes.xyxy.cpu().numpy(),
                                       result.boxes.cls.cpu().numpy(),
                                       result.boxes.conf.cpu().numpy()):
                side = names[int(cls)]
                if side in best and best[side]['detection_score'] >= float(conf):
                    continue
                x1, y1, x2, y2 = map(float, xyxy)
                size = max(x2-x1, y2-y1) * args.box_scale
                if size <= 0 or not np.isfinite([x1,y1,x2,y2,size,conf]).all():
                    continue
                best[side] = {'track_id': side, 'side': side,
                              'bbox': [(x1+x2)/2, (y1+y2)/2, size],
                              'detection_score': float(conf)}
            frames[str(target)] = {'instances': list(best.values())}
            if (ordinal+1) % 100 == 0:
                print(f'[YOLO] {args.camera}: {ordinal+1}/{len(indices)}', flush=True)
    finally:
        cap.release()
    args.output.write_text(json.dumps({'intrinsics': calibration['K'], 'frames': frames}))
    print(f'[YOLO] {args.camera}: {len(indices)} frames, {sum(bool(x["instances"]) for x in frames.values())} with hands', flush=True)


if __name__ == '__main__':
    main()
