#!/usr/bin/env python3
"""Compare WiLoR checkpoints on the same single-view ground-truth hand crops."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from itertools import groupby
import json
import os
from pathlib import Path
import time

import cv2
import numpy as np

from evaluate_hand_models import load_chunk, report, save_chunk, write_json
from hand_evaluation.dataset import LabeledDataset, sha256
from hand_evaluation.metrics import ERROR_NAMES, measure

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = dict(input='single raw image, GT bbox and physical side', crop_scale=2.0,
                crop='WiLoR batched OpenCV crop with anti-aliasing', precision='CUDA FP16/AMP',
                placement='calibrated', calibration='original per-recording camera',
                aggregation='per annotated camera/hand; PCK uses all valid GT joints',
                alignment='root-relative subtracts wrist; PA fits similarity transform')


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--checkpoint', type=Path, default=ROOT / 'models/wilor/wilor_final.ckpt')
    p.add_argument('--adapter', action='append', default=[], type=Path)
    p.add_argument('--wilor-root', type=Path, default=ROOT / 'training')
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--chunk-images', type=int, default=48)
    p.add_argument('--workers', type=int, default=6)
    p.add_argument('--max-images', type=int, default=0)
    p.add_argument('--score', help='only score existing predictions for this checkpoint stem')
    p.add_argument('--predict-only', action='store_true')
    args = p.parse_args()
    if min(args.batch_size, args.chunk_images, args.workers) < 1 or args.max_images < 0:
        p.error('invalid batch or selection size')
    return args


def chunks(dataset, size):
    def camera_key(group):
        row = group[0]
        return row.get('calibration_group', row.get('source_dataset_id')), row['camera']
    for _, items in groupby(sorted(dataset.groups, key=camera_key), key=camera_key):
        camera_frames = list(items)
        for start in range(0, len(camera_frames), size):
            yield camera_frames[start:start + size]


def chunk_ids(groups):
    return [row['index'] for group in groups for row in group]


def read_frame(dataset, group):
    labels = [dataset.load(row) for row in group]
    image = cv2.imread(str(dataset.path(group[0]['image'])))
    calibration = labels[0].calibration
    if image is None or tuple(image.shape[1::-1]) != calibration.image_size:
        raise ValueError(f"Image/calibration size mismatch: {group[0]['image']}")
    boxes = [dict(bbox_xyxy=label.bbox.tolist(), confidence=1.0, is_right=label.side,
                  source='ground_truth') for label in labels]
    return image, {'boxes': boxes}, calibration


def predict(args, dataset, batches, variants):
    os.environ['EGO_WILOR_ROOT'] = str(args.wilor_root.resolve())
    import torch
    import wilor_multiview_inference as infer

    device = infer.base.choose_device('cuda')
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    models = {}
    for name, adapter in variants.items():
        model, cfg = infer.base.load_wilor_model(args.checkpoint, ROOT / 'models/wilor/model_config.yaml',
                                               ROOT / 'models/mano', adapter)
        models[name] = (model.to(device=device, dtype=torch.float16).eval(), cfg)
        (args.output / name / 'predictions').mkdir(parents=True, exist_ok=True)

    with ThreadPoolExecutor(max_workers=args.workers) as workers:
        for i, groups in enumerate(batches):
            ids = chunk_ids(groups)
            paths = {name: args.output / name / 'predictions' / f'{i:05d}.npz' for name in variants}
            if all(path.is_file() for path in paths.values()):
                continue
            frames, boxes, calibrations = zip(*workers.map(lambda g: read_frame(dataset, g), groups))
            prepared = infer._prepare_dual_batched(frames, None, cfg, 0.0, 0.0, 2.0, args.workers, 0,
                                                   preprocess_executor=workers, external_boxes=boxes,
                                                   dual_hypothesis=False)
            for name, (model, cfg) in models.items():
                if paths[name].is_file():
                    continue
                records = infer._infer_prepared_batched(
                    prepared, model, cfg, device, args.batch_size, full_output=True,
                    include_vertices=True, calibration=calibrations[0])
                predictions = []
                for group, hands, proposals in zip(groups, records, boxes, strict=True):
                    for j, (row, hand) in enumerate(zip(group, hands, strict=True)):
                        if (hand['detection_index'] != j or hand['is_right'] != row['side'] or
                                not np.allclose(hand['bbox_xyxy'], proposals['boxes'][j]['bbox_xyxy'])):
                            raise RuntimeError(f"GT box/side assignment changed at {row['index']}")
                        translation = np.asarray(hand['camera_translation'], np.float32)
                        predictions.append(dict(joints_2d_px=hand['joints_2d'],
                            joints_camera_m=hand['joints_3d_camera_relative'] + translation,
                            vertices_camera_m=hand['mano_vertices_camera_relative'] + translation))
                save_chunk(paths[name], ids, predictions)
            if i % 10 == 0 or i + 1 == len(batches):
                print(f'Predicted {i+1}/{len(batches)} chunks for all {len(variants)} models', flush=True)


def score(args, dataset, batches, name, identity):
    import torch
    torch.set_num_threads(1)
    args.backend = name
    args.output = args.output / name
    scores, masks = [], []
    errors = {key: [] for key in ERROR_NAMES}
    started = time.monotonic()
    for i, groups in enumerate(batches):
        predictions = load_chunk(args.output / 'predictions' / f'{i:05d}.npz', chunk_ids(groups))
        labels = (dataset.load(row) for group in groups for row in group)
        for label, prediction in zip(labels, predictions, strict=True):
            record, values = measure(label, prediction, placement='calibrated')
            scores.append(record)
            masks.append(label.valid)
            for key in errors:
                errors[key].append(values[key])
        if i % 50 == 0 or i+1 == len(batches):
            print(f'Scored {name} {i+1}/{len(batches)}', flush=True)
    report(args, dataset, dict(model=name, assets=identity['weights']), scores, errors, masks,
           time.monotonic()-started, 0, evaluation_protocol=PROTOCOL,
           limitations=['NOKOV/MANO fitted references; GT boxes and handedness.'])


def main():
    args = arguments()
    dataset = LabeledDataset(args.dataset, max_images=args.max_images)
    variants = {args.checkpoint.stem: None, **{a.stem: a for a in args.adapter}}
    if len(variants) != len(args.adapter) + 1:
        raise ValueError('checkpoint stems must be distinct')
    batches = list(chunks(dataset, args.chunk_images))
    args.output.mkdir(parents=True, exist_ok=True)
    config = args.output / 'run_config.json'
    weights = [args.checkpoint, *args.adapter, ROOT / 'models/wilor/model_config.yaml',
               ROOT / 'models/mano/MANO_RIGHT.pkl', args.wilor_root / 'mano_data/mano_mean_params.npz']
    source_files = [Path(__file__).resolve(), *sorted((ROOT/'scripts/hand_evaluation').glob('*.py')),
                    ROOT/'scripts/evaluate_hand_models.py', ROOT/'scripts/wilor_multiview_inference.py',
                    ROOT/'scripts/wilor_physical_inference.py', ROOT/'scripts/wilor_inference.py',
                    *sorted((args.wilor_root/'wilor').rglob('*.py'))]
    import hashlib
    source_digest = hashlib.sha256()
    for path in source_files:
        source_digest.update(sha256(path).encode())
    identity = dict(dataset=dataset.describe(), weights=[dict(path=str(p.resolve()), sha256=sha256(p)) for p in weights],
                    variants=list(variants), chunks=len(batches), chunk_images=args.chunk_images,
                    batch_size=args.batch_size, source_sha256=source_digest.hexdigest(), protocol=PROTOCOL)
    if config.exists() and json.loads(config.read_text()) != identity:
        raise ValueError('existing output belongs to a different run')
    if not config.exists():
        write_json(config, identity)
    if args.score:
        if args.score not in variants:
            raise ValueError('unknown variant')
        score(args, dataset, batches, args.score, identity)
        return
    predict(args, dataset, batches, variants)
    if not args.predict_only:
        output = args.output
        for name in variants:
            args.output = output
            score(args, dataset, batches, name, identity)


if __name__ == '__main__':
    main()
