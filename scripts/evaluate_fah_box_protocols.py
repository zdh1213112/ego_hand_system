#!/usr/bin/env python3
"""Evaluate FAH-S on indexed native KB images with detector boxes, labeled boxes, or both."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

import cv2
import numpy as np
import pandas as pd

from ego_hand_inference.ego_data.calibration import CameraCalibration
from evaluate_hand_models import write_json
from hand_evaluation.dataset import LabeledDataset, sha256

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {'detector': 'detector', 'gt': 'gt_box_physical_side'}


def labeled_boxes(dataset, hands, image_size):
    boxes = []
    for hand in hands:
        sample = np.load(dataset.path(hand['label']), allow_pickle=True).item()
        box = np.asarray(sample['bbox'], dtype=float)
        if (box.shape != (4,) or not np.isfinite(box).all() or np.any(box[2:] <= box[:2])
                or np.any(box[:2] < 0) or np.any(box[2:] >= image_size)
                or float(sample['side']) != hand['side']):
            raise ValueError(f"Invalid labeled box or side in {hand['label']}")
        boxes.append(dict(bbox_xyxy=box.tolist(), detector_is_right=hand['side'],
                          confidence=1., source=SOURCES['gt']))
    return boxes


def detector_boxes(detector, paths, camera, image_size):
    import torch

    images = [cv2.imread(str(path)) for path in paths]
    if any(im is None or im.shape[1::-1] != tuple(image_size) for im in images):
        raise ValueError('Detector input image differs from the calibrated raster')
    with torch.inference_mode():
        results = detector(images, conf=.2 if camera == 'camera0' else .3,
                           iou=.5, imgsz=640, half=True, verbose=False)
    return [[dict(bbox_xyxy=box.tolist(), detector_is_right=int(side), confidence=float(score),
                  source=SOURCES['detector'])
             for box, side, score in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy(),
                                         r.boxes.conf.cpu().numpy(), strict=True)] for r in results]


def synchronize(frames, cameras):
    by_sync = defaultdict(dict)
    for frame in frames:
        index, camera = frame['sync_index'], frame['camera']
        if camera in by_sync[index]:
            raise ValueError(f'Duplicate image at {index}/{camera}')
        by_sync[index][camera] = frame
    rows = []
    for index, views in sorted(by_sync.items()):
        stamps = {v['reference_timestamp_ns'] for v in views.values()}
        if len(stamps) != 1:
            raise ValueError(f'Conflicting reference timestamps at {index}')
        stamp = stamps.pop()
        row = dict(sync_index=index, reference_timestamp_ns=stamp)
        for camera in cameras:
            frame = views.get(camera, {})
            row[camera+'_frame_index'] = frame.get('source_frame_index', -1)
            row[camera+'_timestamp_ns'] = frame.get('camera_timestamp_ns', stamp)
        rows.append(row)
    return rows, by_sync


def prepare_group(stage, dataset, recording, frames, mode, detector):
    name = recording['recording']
    calibration_groups = {str(f['calibration_group']) for f in frames}
    if len(calibration_groups) != 1:
        raise ValueError(f'{name} must have one calibration group')
    calibrations = dataset.summary['camera_calibrations_by_source'][calibration_groups.pop()]
    cameras = sorted(calibrations)
    if not {f['camera'] for f in frames}.issubset(cameras):
        raise ValueError(f'{name}: a labeled camera is missing its calibration')
    rows, by_sync = synchronize(frames, cameras)
    if (len(rows) != recording['sync_times'] or len(frames) != recording['images']
            or sum(len(f['hands']) for f in frames) != recording['samples']):
        raise ValueError(f'{name}: exported split and frame counts differ')
    target = stage/name
    normalized = target/'normalized_multiview'
    (normalized/'calibration').mkdir(parents=True)
    size = dataset.summary['image_size']
    for camera in cameras:
        cal = CameraCalibration.from_dict(calibrations[camera])
        if cal.model != 'KB' or cal.image_size != tuple(size):
            raise ValueError('Expected native KB calibration matching the exported images')
        cal.save(normalized/'calibration'/f'{camera}.json')
    with (normalized/'multiview_frames.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    write_json(normalized/'manifest.json', dict(camera_ids=cameras, image_size=size,
        coordinate_system='calibration base frame', indexed_jpegs=str(dataset.root),
        camera_timestamp_semantics='exported camera_timestamp_ns when available, otherwise reference_timestamp_ns'))
    stats = {}
    for camera in cameras:
        path = target/(camera+'.jsonl')
        images = boxes = 0
        with path.open('w') as f:
            for start in range(0, len(rows), 12):
                batch = rows[start:start+12]
                observed = [by_sync[s['sync_index']][camera] for s in batch if camera in by_sync[s['sync_index']]]
                paths = [dataset.path(v['image']) for v in observed]
                if mode == 'gt':
                    proposals = [labeled_boxes(dataset, v['hands'], size) for v in observed]
                else:
                    proposals = detector_boxes(detector, paths, camera, size) if paths else []
                packets = {v['sync_index']: dict(image=v['image'], image_sha256=sha256(p), boxes=b)
                           for v, p, b in zip(observed, paths, proposals, strict=True)}
                images += len(observed)
                boxes += sum(map(len, proposals))
                for s in batch:
                    packet = dict(sync_index=s['sync_index'], source_frame_index=s[camera+'_frame_index'],
                                  timestamp_ns=s[camera+'_timestamp_ns'],
                                  **packets.get(s['sync_index'], dict(image=None, boxes=[])))
                    f.write(json.dumps(packet, allow_nan=False, separators=(',', ':'))+'\n')
        stats[camera] = dict(images=images, boxes=boxes, sha256=sha256(path))
    return dict(cameras=stats, raw=False, sync_count=len(rows), proposal_source=SOURCES[mode])


def prepare_inputs(args, dataset, frames, recordings, mode, fingerprint):
    target = args.output/'inputs'/mode
    groups = [r['recording'] for r in recordings]
    source = dict(dataset=str(dataset.root), groups=groups, proposal_source=SOURCES[mode],
                  dataset_manifest_sha256=sha256(dataset.root/'test.jsonl'), dataset_fingerprint=fingerprint,
                  frames_sha256=sha256(dataset.root/'frames.jsonl'), splits_sha256=sha256(dataset.root/'splits.json'),
                  preparation_sha256=sha256(__file__))
    if mode == 'detector':
        source.update(detector=str(args.detector), detector_sha256=sha256(args.detector),
                      detector_settings=dict(imgsz=640, iou=.5, conf_camera0=.2, conf_other=.3, half=True))
    if target.exists():
        if json.loads((target/'sources.json').read_text()) != source:
            raise ValueError(f'Input configuration changed in {target}. Use a new output directory.')
        return target
    detector = None
    if mode == 'detector':
        from wilor_inference import choose_device, load_detector
        detector = load_detector(args.detector, choose_device('cuda'))
        names = [str(detector.names[i]).lower().replace('_',' ') for i in (0,1)]
        if names not in (['left','right'], ['left hand','right hand']) or len(detector.names) != 2:
            raise ValueError(f'Detector classes must encode left=0 and right=1, got {detector.names}')
    target.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=target.parent) as tmp:
        stage = Path(tmp)/'inputs'
        for recording in recordings:
            name = recording['recording']
            info = prepare_group(stage, dataset, recording, frames[name], mode, detector)
            info.update({k:source[k] for k in ('dataset_manifest_sha256','detector','detector_sha256') if k in source})
            write_json(stage/name/'summary.json', info)
        write_json(stage/'sources.json', source)
        stage.rename(target)
    return target


def run_model(args, inputs, mode, checkpoint, recordings):
    directory = args.output/mode/checkpoint.stem
    def run(python, script, *options):
        cmd = [str(python), str(ROOT/'scripts'/script), *map(str, options)]
        subprocess.run(cmd, cwd=ROOT, check=True, env={**os.environ, 'PYTHONDONTWRITEBYTECODE':'1',
                       'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2'})
    run(args.fah_python, 'predict_detector_fah.py', '--inputs', inputs, '--output', directory,
        '--checkpoint', checkpoint, '--fah-root', args.fah_root)
    common = ['--backend','fah','--dataset',args.dataset,'--output',directory,'--proposal-source',SOURCES[mode]]
    for recording in recordings:
        group = recording['recording']
        stage = directory/group
        shared = stage/'fusion_multiview_shared_fah'
        # Publish only a completed fit. Existing fits go through the fitter's input fingerprint checks.
        with TemporaryDirectory(dir=stage) as tmp:
            fit_output = shared if shared.exists() else Path(tmp)/'fit'
            run(sys.executable, 'fit_shared_wilor_multiview.py', '--dataset', stage/'normalized_multiview',
                '--predictions', stage/'fah_multiview', '--output', fit_output,
                '--mano-model-dir', args.fah_root/'data/mano', '--device', 'cuda',
                '--steps', 100, '--batch-size', 128, '--pose-policy', 'shared-fit')
            if fit_output != shared:
                fit_output.rename(shared)
        run(sys.executable, 'score_indexed_wilor.py', *common, '--group', group)
    run(sys.executable, 'score_indexed_wilor.py', *common, '--aggregate')
    return directory


def export_tables(runs, output):
    summaries, paired = [], []
    for checkpoint, modes in runs.items():
        samples = {}
        for mode, folder in modes.items():
            table = pd.read_csv(folder/'metrics.csv')
            overall = table.loc[(table.dimension=='all') & (table.cohort=='eligible')].copy()
            overall.insert(0, 'box_source', mode)
            overall.insert(0, 'checkpoint', checkpoint)
            summaries.append(overall)
            samples[mode] = pd.read_csv(folder/'per_sample.csv').set_index('index',verify_integrity=True).sort_index()
        if set(samples) == {'detector','gt'}:
            detector, gt = samples['detector'], samples['gt']
            keys = ['source_group','reference_timestamp_ns','camera','source_frame_index','side','image','label']
            pd.testing.assert_frame_equal(detector[keys],gt[keys])
            for method in ('single','shared'):
                prefix = method+'_'
                for field in (c for c in gt if c.startswith(prefix) and c.endswith(('_mm','_px'))):
                    matched = gt[field].notna() & detector[field].notna()
                    a,b = gt.loc[matched,field],detector.loc[matched,field]
                    paired.append(dict(checkpoint=checkpoint,method=method,metric=field[len(prefix):],
                        samples=int(matched.sum()),gt=a.mean(),detector=b.mean(),gt_minus_detector=(a-b).mean()))
    tables = {'metrics.csv': pd.concat(summaries,ignore_index=True)}
    if paired or (output/'paired_box_comparison.csv').exists():
        columns = ['checkpoint','method','metric','samples','gt','detector','gt_minus_detector']
        tables['paired_box_comparison.csv'] = pd.DataFrame(paired,columns=columns)
    for filename, table in tables.items():
        path = output/filename
        temporary = path.with_suffix('.tmp')
        table.to_csv(temporary,index=False)
        temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--box-source',choices=('detector','gt','both'),required=True)
    parser.add_argument('--checkpoint',type=Path,nargs='+',required=True,help='one or more fine-tuned FAH-S .pt files')
    parser.add_argument('--detector',type=Path,default=ROOT/'models/wilor/2.0_glove_wilor_3680_261118.pt')
    parser.add_argument('--fah-root',type=Path,default=ROOT.parent/'FAH')
    parser.add_argument('--fah-python',type=Path)
    args = parser.parse_args()
    for key in ('dataset','output','detector','fah_root'):
        setattr(args,key,getattr(args,key).resolve())
    args.fah_python = (args.fah_python or args.fah_root/'.venv/bin/python').absolute()
    checkpoints = [p.resolve(strict=True) for p in args.checkpoint]
    if len({p.stem for p in checkpoints}) != len(checkpoints):
        parser.error('Checkpoint filenames must have distinct stems')
    dataset = LabeledDataset(args.dataset)
    recordings = json.loads((args.dataset/'splits.json').read_text())['recordings']
    frames = defaultdict(list)
    with (args.dataset/'frames.jsonl').open() as f:
        for frame in map(json.loads,f):
            frames[frame['source_group']].append(frame)
    modes = ('detector','gt') if args.box_source=='both' else (args.box_source,)
    args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'.evaluation.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        runs = defaultdict(dict)
        fingerprint = dataset.fingerprint()
        for mode in modes:
            inputs = prepare_inputs(args,dataset,frames,recordings,mode,fingerprint)
            for checkpoint in checkpoints:
                runs[checkpoint.stem][mode] = run_model(args,inputs,mode,checkpoint,recordings)
        export_tables(runs,args.output)

if __name__=='__main__':
    main()
