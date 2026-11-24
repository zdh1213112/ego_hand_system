#!/usr/bin/env python3
"""Run the existing shared fitters on available labeled multiview JPGs.

This GT-box benchmark is explicit about missing/unexported camera views. It
does not fabricate videos or claim to run the original detector recording.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import os
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True

import numpy as np

from evaluate_hand_models import ROOT, parse_args as parse_model_args, source_fingerprint, write_json
from hand_evaluation.dataset import LabeledDataset, sha256


def parse_args(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--max-sync-frames', type=int, default=0)
    parser.add_argument('--sync-stride', type=int, default=1)
    parser.add_argument('--shared-steps', type=int, default=100)
    parser.add_argument('--shared-batch-size', type=int, default=32)
    parser.add_argument('--pose-policy', choices=('native-wrist', 'shared-fit'),
                        help='WiLoR shared articulation policy (default native-wrist)')
    parser.add_argument('--pose-switch-margin-px', type=float, default=1.,
                        help='native source retention margin in pixels (default 1, 0 disables)')
    parser.add_argument('--no-native-recovery', action='store_true')
    parser.add_argument('--no-pose-transition', action='store_true')
    parser.add_argument('--only-fit', action='store_true')
    extra, remaining = parser.parse_known_args(argv)
    args = parse_model_args(remaining)
    if args.max_images or args.image_stride != 1 or args.score_only or args.inspect_only:
        raise ValueError('use --max-sync-frames/--sync-stride; score existing results with evaluate_shared_hand_models.py')
    if extra.max_sync_frames < 0 or min(extra.sync_stride, extra.shared_steps, extra.shared_batch_size) < 1:
        raise ValueError('invalid synchronized sampling/shared fitting options')
    if not np.isfinite(extra.pose_switch_margin_px) or extra.pose_switch_margin_px < 0:
        raise ValueError('--pose-switch-margin-px must be finite and nonnegative')
    for key, value in vars(extra).items():
        setattr(args, key, value)
    args.export_shared_parameters = True
    return args


def select_synchronized(dataset, stride, limit):
    sources = {(r.get('source_dataset_id'), r.get('source_group')) for r in dataset.selected}
    if len(sources) != 1 or dataset.summary['image_space'] != 'native_raw':
        raise ValueError('expected one native-raw labeled recording')
    sync_times = {}
    for row in dataset.selected:
        sync = int(row['sync_index'])
        stamp = int(row['reference_timestamp_ns'])
        if sync in sync_times and sync_times[sync] != stamp:
            raise ValueError('one sync_index maps to different timestamps')
        sync_times[sync] = stamp
    if len(set(sync_times.values())) != len(sync_times):
        raise ValueError('duplicate synchronized timestamps')
    selected = sorted(sync_times)[::stride][:limit or None]
    selected_set = set(selected)
    dataset.groups = [g for g in dataset.groups if int(g[0]['sync_index']) in selected_set]
    dataset.selected = [r for group in dataset.groups for r in group]
    return selected


def normalized_metadata(dataset, syncs, output):
    root = output / 'normalized_multiview'
    root.mkdir(parents=True, exist_ok=True)
    (root / 'calibration').mkdir(exist_ok=True)
    cameras = sorted({r['camera'] for r in dataset.selected})
    if len(cameras) < 2:
        raise ValueError('shared fitting needs at least two camera views')
    grouped = {}
    for group in dataset.groups:
        row = group[0]
        key = int(row['sync_index']), row['camera']
        if key in grouped:
            raise ValueError('duplicate image at synchronized camera/frame')
        grouped[key] = row
    times = {int(row['sync_index']): int(row['reference_timestamp_ns']) for row in dataset.selected}
    rows = []
    for sync in syncs:
        row = dict(sync_index=sync, reference_timestamp_ns=times[sync])
        for camera in cameras:
            label_row = grouped.get((sync,camera))
            row[f'{camera}_frame_index'] = int(label_row['source_frame_index']) if label_row else -1
            # Original per-camera exposure timestamps are not available in this export.
            row[f'{camera}_timestamp_ns'] = times[sync]
        rows.append(row)
    for camera in cameras:
        write_json(root / 'calibration' / f'{camera}.json', dataset.summary['camera_calibrations'][camera])
    with (root / 'multiview_frames.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    write_json(root / 'manifest.json', dict(schema_version=1, dataset_type='normalized_multiview',
        camera_ids=cameras, image_size=dataset.summary['image_size'],
        coordinate_system='label-export camera calibration base frame; metres',
        storage=dict(kind='labeled_image_metadata_only', source=str(dataset.root)),
        evaluation_input=dict(protocol='GT boxes and physical handedness; available labeled views only',
            label_dataset=str(dataset.root), label_dataset_fingerprint=dataset.fingerprint(),
            unavailable_view_policy='empty observations; no synthetic pixels, pose or fallback',
            synchronization='exported sync_index and reference_timestamp_ns; original source frame indices retained',
            camera_timestamp_semantics='reference timestamp copied for interface compatibility; not measured per-camera exposure time')))
    return root, rows, cameras


def to_plain(value):
    if isinstance(value, dict):
        return {k:to_plain(v) for k,v in value.items()}
    if hasattr(value, 'detach'):
        return value.detach().cpu().numpy().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def infer_and_export(args, dataset, syncs):
    import cv2
    import torch
    from hand_evaluation.backends import FrameInput, asset_paths, create_backend
    config = dict(stage='labeled_shared_native_inference', pose_backend=args.backend,
        dataset_fingerprint=dataset.fingerprint(), source_fingerprint=source_fingerprint(args),
        bridge_code_sha256=sha256(Path(__file__)), batch_size=args.batch_size, chunk_images=args.chunk_images,
        crop_scale=args.crop_scale, device=args.device,
        environment=dict(python=sys.version.split()[0],torch=str(torch.__version__),numpy=np.__version__,opencv=cv2.__version__),
        assets=[dict(path=str(p), sha256=sha256(p)) for p in asset_paths(args)])
    root = args.output
    native = root / f'{args.backend}_multiview'
    config_path = root / 'run_config.json'
    if config_path.exists():
        if not args.resume or json.loads(config_path.read_text()) != config:
            raise ValueError('existing native inference config differs or --resume missing')
        if (native / 'summary.json').is_file():
            print(f'[shared benchmark] reuse native parameters: {native}',flush=True)
            return root / 'normalized_multiview', native
    elif root.exists() and any(root.iterdir()):
        raise FileExistsError('nonempty experiment output without run_config.json')
    root.mkdir(parents=True,exist_ok=True)
    write_json(config_path,config)
    cache=root/'native_cache';cache.mkdir(exist_ok=True)
    normalized, sync_rows, cameras = normalized_metadata(dataset,syncs,root)
    cv2.setNumThreads(args.workers)
    records={camera:{int(r['sync_index']):dict(sync_index=int(r['sync_index']),
        source_frame_index=int(r[f'{camera}_frame_index']),timestamp_ns=int(r[f'{camera}_timestamp_ns']),hands=[])
        for r in sync_rows} for camera in cameras}
    soma={camera:[] for camera in cameras}
    backend=None
    for start in range(0,len(dataset.groups),args.chunk_images):
        groups=dataset.groups[start:start+args.chunk_images]
        ids=[[r['index'] for r in group] for group in groups]
        path=cache/f'{start:08d}.pt'
        if path.is_file():
            saved=torch.load(path,map_location='cpu',weights_only=False)
            if saved['ids']!=ids:
                raise ValueError('native cache sample IDs differ')
            predictions=saved['predictions']
        else:
            if backend is None:
                backend=create_backend(args)
            frames=[]
            for group in groups:
                labels=[dataset.load(row) for row in group]
                image=cv2.imread(str(dataset.path(group[0]['image'])))
                if image is None or tuple(image.shape[1::-1])!=labels[0].calibration.image_size:
                    raise ValueError('invalid input image')
                frames.append(FrameInput(image,np.asarray([l.bbox for l in labels],np.float32),
                    np.asarray([l.side for l in labels],np.int64),labels[0].calibration))
            predictions=backend.predict(frames)
            if len(predictions)!=len(groups) or any(len(a)!=len(b) for a,b in zip(predictions,groups)):
                raise ValueError('native model changed labeled frame/hand order')
            temp=path.with_suffix('.pt.tmp')
            torch.save(dict(ids=ids,predictions=predictions),temp);temp.replace(path)
        for group,frame in zip(groups,predictions):
            first=group[0];camera=first['camera'];sync=int(first['sync_index'])
            instances=[]
            for detection,(row,pred) in enumerate(zip(group,frame)):
                pixels=np.asarray(pred['joints_2d_px'])
                if pixels.shape!=(21,2) or not np.isfinite(pixels).all():
                    continue
                side=int(row['side']);hypothesis=2*detection+side
                bbox=np.load(dataset.path(row['label']),allow_pickle=True).item()['bbox']
                hand=dict(detection_index=detection,hypothesis_index=hypothesis,is_right=side,
                    detector_is_right=side,confidence=1.,bbox_xyxy=np.asarray(bbox).tolist(),
                    proposal_source='gt_box_physical_side',joints_2d=pixels.tolist())
                if args.backend=='gem':
                    parameters={name:torch.as_tensor(value).detach().cpu() for name,value in pred['soma_parameters'].items()}
                    instances.append(dict(track_id=hypothesis,side='right' if side else 'left',parameters=parameters))
                else:
                    hand['mano']=to_plain(pred['mano'])
                records[camera][sync]['hands'].append(hand)
            if args.backend=='gem':
                soma[camera].append(dict(frame_id=int(first['source_frame_index']),instances=instances))
        print(f'[shared native {args.backend}] {min(start+args.chunk_images,len(dataset.groups))}/{len(dataset.groups)} images',flush=True)
    del backend
    gc.collect()
    if torch.cuda.is_available():torch.cuda.empty_cache()
    native.mkdir(exist_ok=True)
    for camera in cameras:
        directory=native/camera;directory.mkdir(exist_ok=True)
        with (directory/'predictions.jsonl').open('w') as stream:
            for row in records[camera].values():
                stream.write(json.dumps(row,allow_nan=False)+'\n')
        if args.backend=='gem':
            torch.save(dict(format='gem_pose_predictions_v1',frames=soma[camera]),directory/'gem_pose.pt')
    write_json(native/'summary.json',dict(stage='labeled_shared_native_inference',save_mano_params=args.backend!='gem',
        camera_ids=cameras,frame_count=len(syncs),image_count=len(dataset.groups),sample_count=len(dataset.selected),
        input_protocol='GT boxes and physical handedness; labeled views only'))
    return normalized,native


def fit(args, normalized, native):
    shared=args.output/f'fusion_multiview_shared_{args.backend}'
    if args.backend=='gem':
        command=[str(args.gem_python),str(ROOT/'scripts/fit_shared_gem_multiview.py'),
                 '--gem-root',str(args.gem_root),'--soma-assets',str(args.soma_assets)]
    else:
        command=[str(args.wilor_python),str(ROOT/'scripts/fit_shared_wilor_multiview.py'),
                 '--mano-model-dir',str(args.fah_mano_dir if args.backend=='fah' else args.mano_dir),
                 '--pose-policy','shared-fit' if args.backend=='fah' else (args.pose_policy or 'native-wrist')]
        if args.backend == 'wilor' and args.pose_policy != 'shared-fit':
            command += ['--pose-switch-margin-px', str(args.pose_switch_margin_px)]
            if args.no_native_recovery:
                command.append('--no-native-recovery')
            if args.no_pose_transition:
                command.append('--no-pose-transition')
    command+=['--dataset',str(normalized),'--predictions',str(native),'--native-predictions',str(native),
        '--output',str(shared),'--device',args.device,'--steps',str(args.shared_steps),'--batch-size',str(args.shared_batch_size)]
    subprocess.run(command,check=True,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})


def main():
    args=parse_args()
    if args.backends:
        root=args.output
        for name in dict.fromkeys(args.backends):
            command=[str(getattr(args,name+'_python')),str(Path(__file__).resolve())]
            for key,value in vars(args).items():
                if key in ('backends','backend','output','only_fit','export_shared_parameters') or value is None or value is False:
                    continue
                command+=['--'+key.replace('_','-')]+([] if value is True else list(map(str,value)) if isinstance(value,list) else [str(value)])
            command+=['--backend',name,'--output',str(root/name),'--only-fit']
            subprocess.run(command,check=True,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
        if not args.only_fit:
            command=[str(args.wilor_python),str(ROOT/'scripts/evaluate_shared_hand_models.py'),
                '--dataset',str(args.dataset),'--output',str(root/'evaluation'),'--runs',
                *[f'{name}={root/name}' for name in dict.fromkeys(args.backends)]]
            if args.cameras:command+=['--cameras',*args.cameras]
            if args.resume:command+=['--resume']
            subprocess.run(command,check=True,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
        return
    dataset=LabeledDataset(args.dataset,cameras=args.cameras)
    syncs=select_synchronized(dataset,args.sync_stride,args.max_sync_frames)
    normalized,native=infer_and_export(args,dataset,syncs)
    fit(args,normalized,native)
    if not args.only_fit:
        command=[str(args.wilor_python),str(ROOT/'scripts/evaluate_shared_hand_models.py'),
            '--dataset',str(args.dataset),'--output',str(args.output/'evaluation'),'--run',str(args.output),'--backend',args.backend]
        if args.cameras:command+=['--cameras',*args.cameras]
        if args.resume:command+=['--resume']
        subprocess.run(command,check=True,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})


if __name__=='__main__':
    main()
