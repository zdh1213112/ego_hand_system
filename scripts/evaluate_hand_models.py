#!/usr/bin/env python3
"""Evaluate WiLoR, GEM-X and FAH on labeled WiLoR image exports.

Default protocol: GT boxes + GT physical handedness; raw source images;
single-view inference; shared calibration-only translation placement for the
absolute metric. No label pose, mesh or translation is used by inference.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.dont_write_bytecode = True

import numpy as np

from hand_evaluation.dataset import LabeledDataset, sha256
from hand_evaluation.metrics import ERROR_NAMES, measure, stats, summarize

ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, epilog=(
        'For FAH-S checkpoint evaluation with both single-view and shared MANO scores, '
        'use evaluate_fah_box_protocols.py --box-source detector|gt|both.'))
    p.add_argument('--dataset', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    group = p.add_mutually_exclusive_group()
    group.add_argument('--backend', choices=('wilor', 'gem', 'fah'))
    group.add_argument('--backends', nargs='+', choices=('wilor', 'gem', 'fah'))
    p.add_argument('--device', default='cuda')
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--chunk-images', type=int, default=32)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--crop-scale', type=float, default=2.0, help='WiLoR/GEM crop expansion; FAH rectifies KB hand boxes to local perspective')
    p.add_argument('--max-images', type=int, default=0, help='0 = all selected images; never splits two hands')
    p.add_argument('--image-stride', type=int, default=1)
    p.add_argument('--cameras', nargs='+')
    p.add_argument('--placement', choices=('calibrated', 'native'), default='calibrated')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--score-only', action='store_true', help='score existing numeric prediction chunks without loading models')
    p.add_argument('--inspect-only', action='store_true', help='validate all selected labels/images and write dataset audit')
    p.add_argument('--wilor-checkpoint', type=Path, default=ROOT / 'models/wilor/wilor_final.ckpt')
    p.add_argument('--wilor-config', type=Path, default=ROOT / 'models/wilor/model_config.yaml')
    p.add_argument('--mano-dir', type=Path, default=ROOT / 'models/mano')
    p.add_argument('--gem-root', type=Path, default=ROOT.parent / 'GEM-X')
    p.add_argument('--gem-checkpoint', type=Path)
    p.add_argument('--soma-assets', type=Path)
    p.add_argument('--fah-root', type=Path, default=ROOT.parent / 'FAH')
    p.add_argument('--fah-model', choices=('fah-s', 'fah-b', 'fah', 'fah-h'), default='fah-s')
    p.add_argument('--fah-weights', type=Path)
    p.add_argument('--fah-mano-dir', type=Path)
    p.add_argument('--fah-sam3d', type=Path)
    p.add_argument('--wilor-python', type=Path, default=Path(sys.executable))
    p.add_argument('--gem-python', type=Path, default=Path(sys.executable))
    p.add_argument('--fah-python', type=Path)
    args = p.parse_args(argv)
    if min(args.batch_size, args.chunk_images, args.workers, args.image_stride) < 1 or args.max_images < 0:
        p.error('batch/chunk/workers/stride must be positive; max-images cannot be negative')
    if not np.isfinite(args.crop_scale) or args.crop_scale <= 0:
        p.error('crop-scale must be positive and finite')
    args.backend = args.backend or 'wilor'
    args.gem_checkpoint = args.gem_checkpoint or args.gem_root / 'weights/egohand_s060000.ckpt'
    args.soma_assets = args.soma_assets or args.gem_root / 'assets/soma'
    args.fah_weights = args.fah_weights or args.fah_root / 'weights'
    args.fah_mano_dir = args.fah_mano_dir or args.fah_root / 'data/mano'
    args.fah_sam3d = args.fah_sam3d or args.fah_root / 'data/sam-3d-body-dinov3'
    args.fah_python = args.fah_python or args.fah_root / '.venv/bin/python'
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.expanduser().resolve())
    return args


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def source_fingerprint(args):
    paths = [Path(__file__), *sorted((ROOT / 'scripts/hand_evaluation').glob('*.py')),
             *sorted((ROOT / 'scripts/camera_models').glob('*.py')), ROOT / 'scripts/ego_data/calibration.py']
    if args.backend == 'wilor':
        paths += list((ROOT / 'third_party/WiLoR/wilor').rglob('*.py')) + [ROOT / 'scripts/wilor_inference.py']
    elif args.backend == 'gem':
        paths += list((args.gem_root / 'gem').rglob('*.py')) + [ROOT / 'scripts/gem_multiview_inference.py']
    else:
        from fah_multiview_inference import geometry_sources
        paths += [ROOT / 'scripts/fah_multiview_inference.py', *geometry_sources()]
        paths += list((args.fah_root / 'fah').rglob('*.py')) + list((args.fah_root / 'bench/methods').glob('*.py'))
    return hashlib.sha256(json.dumps([(str(path), sha256(path)) for path in sorted(paths)]).encode()).hexdigest()


def protocol(args):
    return dict(version=1, input='GT bounding boxes and physical handedness',
        inference='single-image; no detector, multiview fusion, temporal smoothing or GT pose fitting',
        image_preprocessing='original exported raster; native model crop policies',
        placement=args.placement,
        absolute_metric=('translation fitted to model-predicted 2D with the actual camera; shape/orientation unchanged; no GT joints used'
                         if args.placement == 'calibrated' else 'native model camera geometry; WiLoR synthetic intrinsics are not calibrated'),
        joint_order='wrist, thumb[4], index[4], middle[4], ring[4], pinky[4]',
        root_alignment='subtract each hand wrist (joint 0); wrist included in mean',
        pa_alignment='per-hand similarity: proper rotation, scale, translation; no reflection',
        mesh_alignment='same joint-derived transform; SOMA/MANO vertex correspondence is unavailable',
        units=dict(geometry='metres', errors_3d='millimetres', errors_2d='source pixels', pck_auc='fraction [0,1]'),
        missing_policy='invalid hands excluded from mean 3D error, retained as failures in PCK/AUC and coverage',
        aggregation='each labeled camera/hand sample has equal weight; not unique physical instants',
        pck_3d_thresholds_mm=[20, 30, 50], auc_3d='exact normalized integral on [0,50] mm',
        pck_2d_thresholds_px=[5, 10, 20])


def inspect(args, dataset):
    import cv2
    started = time.perf_counter()
    for i, group in enumerate(dataset.groups):
        image = cv2.imread(str(dataset.path(group[0]['image'])))
        if image is None or tuple(image.shape[1::-1]) != tuple(dataset.summary['image_size']):
            raise ValueError(f'bad source image: {group[0]["image"]}')
        for row in group:
            dataset.load(row)
        if (i + 1) % 250 == 0:
            print(f'[audit] {i + 1}/{len(dataset.groups)} images validated', flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    result = dict(dataset.describe(), labels_validated=len(dataset.selected), images_decoded=len(dataset.groups),
                  seconds=time.perf_counter() - started, label_geometry='physical camera coordinates = joints_3d + trans',
                  projection_check='stored 778 vertex projections verified with actual lens model; tolerance 0.02 pixels')
    write_json(args.output / 'dataset_audit.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def save_chunk(path, ids, predictions):
    if len(ids) != len(predictions):
        raise ValueError('model returned the wrong number of hands')
    values = dict(ids=np.asarray(ids, np.int64),
                  joints=np.asarray([p['joints_camera_m'] for p in predictions], np.float32),
                  pixels=np.asarray([p['joints_2d_px'] for p in predictions], np.float32),
                  has_mesh=np.asarray(['vertices_camera_m' in p for p in predictions], bool),
                  vertices=np.asarray([p.get('vertices_camera_m', np.full((778, 3), np.nan)) for p in predictions], np.float32))
    if values['joints'].shape != (len(ids), 21, 3) or values['pixels'].shape != (len(ids), 21, 2):
        raise ValueError('bad model output dimensions')
    if values['vertices'].shape != (len(ids), 778, 3):
        raise ValueError('bad MANO mesh dimensions')
    temporary = path.with_suffix('.npz.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **values)
    temporary.replace(path)


def load_chunk(path, ids):
    with np.load(path, allow_pickle=False) as data:
        if not np.array_equal(data['ids'], ids):
            raise ValueError(f'cache sample IDs disagree: {path}')
        for key, shape in (('joints', (len(ids), 21, 3)), ('pixels', (len(ids), 21, 2)),
                           ('vertices', (len(ids), 778, 3)), ('has_mesh', (len(ids),))):
            if data[key].shape != shape:
                raise ValueError(f'bad cache shape {key}: {path}')
        return [dict(joints_camera_m=data['joints'][i], joints_2d_px=data['pixels'][i],
                     **({'vertices_camera_m': data['vertices'][i]} if data['has_mesh'][i] else {})) for i in range(len(ids))]


def report(args, dataset, config, rows, errors, masks, elapsed, reused, *, evaluation_protocol=None, extra=None, limitations=None):
    errors = {name: np.stack(value) for name, value in errors.items()}
    masks = np.stack(masks)
    groups = {}
    for field in ('camera', 'side', 'source_group'):
        groups[field] = {}
        for value in sorted({row[field] for row in rows}):
            indices = [i for i, row in enumerate(rows) if row[field] == value]
            groups[field][str(value)] = summarize([rows[i] for i in indices],
                {name: data[indices] for name, data in errors.items()}, masks[indices])
    result = dict(schema_version=1, backend=args.backend, dataset=dataset.describe(), protocol=evaluation_protocol or protocol(args),
                  model=config.get('model'), assets=config['assets'], runtime=config.get('runtime'),
                  overall=summarize(rows, errors, masks), by=groups,
                  execution=dict(wall_seconds=elapsed, cached_chunks_reused=reused),
                  limitations=['Labels are fitted references: ' + str(dataset.summary.get('supervision_source')),
                    'GT boxes/handedness: detection accuracy is not measured.',
                    'Exported labels already underwent quality/view filtering; unexported frames are outside this benchmark.',
                    'GEM SOMA joints only approximately correspond to MANO; GEM vertex error is unavailable.',
                    'Raw fisheye inference uses each model native camera assumptions; root/PA and absolute errors answer different questions.'])
    if limitations is not None:
        result['limitations'] = limitations
    if extra:
        result.update(extra)
    with (args.output / 'per_sample.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(args.output / 'joint_errors.npz', ids=np.asarray([r['index'] for r in rows]),
                        valid=masks, **errors)
    write_json(args.output / 'summary.json', result)
    render_table(args.output / 'report.html', [result])
    print(json.dumps(dict(backend=args.backend, overall=result['overall']['metrics'],
                         pose_coverage=result['overall']['pose_success'], output=str(args.output)), indent=2), flush=True)
    return result


def render_table(path, reports):
    columns = ['Backend', 'Samples', 'Pose coverage', 'Root MPJPE (mm)', 'PA-MPJPE (mm)',
               'Absolute MPJPE (mm)', '2D error (px)', 'Root PCK@20', 'Root AUC 0–50']
    table = []
    for result in reports:
        overall = result['overall']
        metric = overall['metrics']
        values = [result['backend'], overall['samples'], overall['pose_success']['rate'],
            metric['root_relative_mpjpe_mm']['mean'], metric['pa_mpjpe_mm']['mean'],
            metric['absolute_mpjpe_mm']['mean'], metric['mean_2d_error_px']['mean'],
            overall['root_relative']['pck']['20'], overall['root_relative']['auc_0_50mm']]
        table.append(values)
    formatted = lambda x: 'N/A' if x is None else f'{x:.4f}' if isinstance(x, float) else str(x)
    body = '<tr>' + ''.join(f'<th>{html.escape(x)}</th>' for x in columns) + '</tr>'
    for values in table:
        body += '<tr>' + ''.join(f'<td>{html.escape(formatted(x))}</td>' for x in values) + '</tr>'
    info = html.escape(json.dumps(dict(protocol=reports[0]['protocol'], limitations=reports[0]['limitations'],
                                     dataset=reports[0]['dataset']), ensure_ascii=False, indent=2))
    path.write_text('<!doctype html><meta charset="utf-8"><title>Hand model evaluation</title>'
        '<style>body{font:16px system-ui;margin:32px;color:#172638}table{border-collapse:collapse}td,th{padding:12px;border:1px solid #ccd6df;text-align:right}pre{white-space:pre-wrap;background:#f4f7fa;padding:20px}</style>'
        '<h1>Hand model evaluation</h1><p>' + html.escape(reports[0]['protocol']['input']) + '</p>'
        f'<table>{body}</table><h2>Protocol and dataset</h2><pre>{info}</pre>')
    with path.with_suffix('.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        writer.writerows(table)


def run(args, dataset):
    import cv2
    import torch
    from hand_evaluation.backends import FrameInput, asset_paths, create_backend
    cv2.setNumThreads(args.workers)
    started = time.perf_counter()
    config = dict(schema_version=1, backend=args.backend, dataset_fingerprint=dataset.fingerprint(),
        source_fingerprint=source_fingerprint(args), protocol=protocol(args), batch_size=args.batch_size,
        chunk_images=args.chunk_images, crop_scale=args.crop_scale, device=args.device,
        assets=[dict(path=str(path), sha256=sha256(path), bytes=path.stat().st_size) for path in asset_paths(args)])
    environment = dict(python=sys.version.split()[0], torch=str(torch.__version__), numpy=np.__version__, opencv=cv2.__version__)
    config_path = args.output / 'run_config.json'
    if config_path.exists():
        previous = json.loads(config_path.read_text())
        if any(previous.get(key) != value for key, value in config.items()):
            raise ValueError('saved config/data/model/code differs; use a new output directory')
        if not (args.resume or args.score_only):
            raise FileExistsError('output already has a run; pass --resume or --score-only')
        if not args.score_only and previous.get('environment') != environment:
            raise ValueError('inference environment changed; use a new output directory')
        config = previous
    elif args.score_only:
        raise FileNotFoundError('--score-only requires an existing run_config.json')
    elif args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError('output directory is nonempty without a run_config.json')
    else:
        args.output.mkdir(parents=True, exist_ok=True)
        config['environment'] = environment
        write_json(config_path, config)
    prediction_dir = args.output / 'predictions'
    prediction_dir.mkdir(exist_ok=True)
    backend, reused = None, 0
    scores, masks = [], []
    errors = {name: [] for name in ERROR_NAMES}
    for start in range(0, len(dataset.groups), args.chunk_images):
        groups = dataset.groups[start:start + args.chunk_images]
        labels_by_frame = [[dataset.load(row) for row in group] for group in groups]
        labels = [label for group in labels_by_frame for label in group]
        ids = [label.row['index'] for label in labels]
        path = prediction_dir / f'{start:08d}.npz'
        if path.exists():
            predictions = load_chunk(path, ids)
            reused += 1
        else:
            if args.score_only:
                raise FileNotFoundError(f'missing prediction chunk: {path}')
            if backend is None:
                backend = create_backend(args)
                config.update(model=backend.description, runtime=dict(python=sys.version, torch=torch.__version__,
                    numpy=np.__version__, device_name=torch.cuda.get_device_name(args.device) if args.device.startswith('cuda') else 'CPU'))
                write_json(config_path, config)
            frames = []
            for group in labels_by_frame:
                image = cv2.imread(str(dataset.path(group[0].row['image'])))
                if image is None or tuple(image.shape[1::-1]) != group[0].calibration.image_size:
                    raise ValueError(f'invalid image: {group[0].row["image"]}')
                frames.append(FrameInput(image, np.asarray([label.bbox for label in group], np.float32),
                    np.asarray([label.side for label in group], np.int64), group[0].calibration))
            predicted_frames = backend.predict(frames)
            if len(predicted_frames) != len(frames) or any(len(a) != len(b) for a, b in zip(predicted_frames, labels_by_frame)):
                raise ValueError('backend changed frame/hand ordering or count')
            predictions = [pred for frame in predicted_frames for pred in frame]
            save_chunk(path, ids, predictions)
            # Fresh and resumed scoring must consume exactly the same stored precision.
            predictions = load_chunk(path, ids)
        for label, pred in zip(labels, predictions):
            row, per_joint = measure(label, pred, placement=args.placement)
            scores.append(row)
            masks.append(label.valid)
            for name in ERROR_NAMES:
                errors[name].append(per_joint[name])
        print(f'[{args.backend}] {min(start + args.chunk_images, len(dataset.groups))}/{len(dataset.groups)} images; '
              f'{len(scores)} hands; {time.perf_counter() - started:.1f}s', flush=True)
    report(args, dataset, config, scores, errors, masks, time.perf_counter() - started, reused)


def run_many(args):
    args.output.mkdir(parents=True, exist_ok=True)
    for backend in dict.fromkeys(args.backends):
        child = [str(getattr(args, backend + '_python')), str(Path(__file__).resolve())]
        for key, value in vars(args).items():
            if key in ('backends', 'backend', 'output') or key.endswith('_python') or value is None or value is False:
                continue
            flag = '--' + key.replace('_', '-')
            child += [flag] + ([] if value is True else list(map(str, value)) if isinstance(value, list) else [str(value)])
        child += ['--backend', backend, '--output', str(args.output / backend)]
        subprocess.run(child, check=True, env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'})
    if args.inspect_only:
        return
    reports = [json.loads((args.output / backend / 'summary.json').read_text()) for backend in dict.fromkeys(args.backends)]
    reference = reports[0]
    if any(r['dataset']['fingerprint'] != reference['dataset']['fingerprint'] or r['protocol'] != reference['protocol'] for r in reports):
        raise ValueError('cannot compare different sample sets or evaluation protocols')
    write_json(args.output / 'comparison.json', reports)
    render_table(args.output / 'comparison.html', reports)
    # Also expose a paired comparison on the identical successful hand samples.
    all_errors = {}
    for name in dict.fromkeys(args.backends):
        with np.load(args.output / name / 'joint_errors.npz', allow_pickle=False) as data:
            all_errors[name] = {key: data[key].copy() for key in data.files}
    first = next(iter(all_errors.values()))
    common = np.ones(len(first['ids']), bool)
    for data in all_errors.values():
        if not np.array_equal(data['ids'], first['ids']) or not np.array_equal(data['valid'], first['valid']):
            raise ValueError('comparison prediction IDs/masks differ')
        common &= np.all(np.isfinite(data['root_relative']) | ~data['valid'], axis=1)
    paired = dict(common_pose_samples=int(common.sum()), total_samples=len(common),
                  sample_ids=first['ids'][common].tolist(), metrics={})
    for name, data in all_errors.items():
        paired['metrics'][name] = {key: stats(data[key][common][data['valid'][common]]) for key in ERROR_NAMES}
    write_json(args.output / 'comparison_common_samples.json', paired)


def main():
    args = parse_args()
    if args.backends:
        run_many(args)
        return
    dataset = LabeledDataset(args.dataset, cameras=args.cameras,
                              image_stride=args.image_stride, max_images=args.max_images)
    if args.inspect_only:
        inspect(args, dataset)
    else:
        run(args, dataset)


if __name__ == '__main__':
    main()
