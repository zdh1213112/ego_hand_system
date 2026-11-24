"""Score the saved, accepted shared MANO/SOMA model in every labeled camera.

No per-camera fitting, model selection using GT, or fallback is performed here.
"""
from __future__ import annotations

from collections import Counter
import csv
from dataclasses import replace
import json
from pathlib import Path
import time

import numpy as np

from ego_hand_inference.camera_models import project_points
from ego_hand_inference.ego_data.calibration import CameraCalibration
from gem_multiview_inference import SOMA_TO_21
from hand_evaluation.dataset import Label, array, sha256
from hand_evaluation.metrics import ERROR_NAMES, measure, summarize


def base_to_camera(points, calibration):
    return (np.asarray(points) - calibration.T_base_camera[:3, 3]) @ calibration.T_base_camera[:3, :3]


def camera_to_base(points, calibration):
    return np.asarray(points) @ calibration.T_base_camera[:3, :3].T + calibration.T_base_camera[:3, 3]


def read_jsonl(path):
    rows = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = int(row['sync_index'])
        if key in rows:
            raise ValueError(f'duplicate sync_index in {path}: {key}')
        rows[key] = row
    return rows


def resolve_shared(run, backend, normalized=None):
    run = Path(run).resolve()
    if (run / 'shared_pose.pt').is_file():
        shared = run
        normalized = Path(normalized).resolve() if normalized else run.parent / 'normalized_multiview'
    else:
        candidates = sorted(p for p in run.glob(f'*_shared_{backend}') if (p / 'shared_pose.pt').is_file())
        if len(candidates) != 1:
            raise ValueError(f'{run}: expected one *_shared_{backend}/shared_pose.pt; found {len(candidates)}; specify the shared folder directly')
        shared = candidates[0]
        normalized = Path(normalized).resolve() if normalized else run / 'normalized_multiview'
    if not (normalized / 'manifest.json').is_file():
        raise FileNotFoundError(f'normalized dataset missing: {normalized}; use --normalized-dataset')
    root_config = shared.parent / 'run_config.json'
    if root_config.is_file():
        cfg = json.loads(root_config.read_text())
        actual = cfg.get('pose_backend')
        actual = actual.get('name') if isinstance(actual, dict) else actual
        if actual and actual != backend:
            raise ValueError(f'backend mismatch: {backend} vs {actual}')
    return shared, normalized


class SharedResult:
    def __init__(self, run, backend, normalized=None):
        import torch
        self.shared, self.normalized = resolve_shared(run, backend, normalized)
        self.backend = backend
        self.manifest = json.loads((self.normalized / 'manifest.json').read_text())
        self.summary = json.loads((self.shared / 'summary.json').read_text())
        root_config = self.shared.parent / 'run_config.json'
        self.experiment_config = json.loads(root_config.read_text()) if root_config.is_file() else None
        if self.summary.get('skeleton_and_mesh_share_model') is not True or self.summary.get('independent_view_fallback') is not False:
            raise ValueError('expected shared skeleton/mesh without independent-view fallbacks')
        self.calibrations = {camera: CameraCalibration.load(self.normalized / 'calibration' / f'{camera}.json')
                             for camera in self.manifest['camera_ids']}
        with (self.normalized / 'multiview_frames.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        self.by_time, self.by_sync = {}, {}
        for row in rows:
            sync, stamp = int(row['sync_index']), int(row['reference_timestamp_ns'])
            if sync in self.by_sync or stamp in self.by_time:
                raise ValueError('duplicate synchronized index or timestamp')
            self.by_time[stamp] = row
            self.by_sync[sync] = row
        self.data = torch.load(self.shared / 'shared_pose.pt', map_location='cpu', weights_only=False)
        expected = 'shared_gem_soma_v1' if backend == 'gem' else 'shared_wilor_mano_v1'
        if self.data.get('format') != expected:
            raise ValueError(f'expected {expected}, got {self.data.get("format")}')
        if self.data.get('coordinate_system') != self.manifest.get('coordinate_system'):
            raise ValueError('shared model and normalized dataset coordinate systems differ')
        if backend == 'gem' and tuple(self.data.get('soma_to_21_indices', ())) != SOMA_TO_21:
            raise ValueError('unknown SOMA -> 21 joint mapping')
        accepted = read_jsonl(self.shared / 'accepted.jsonl')
        self.rejected = []
        path = self.shared / 'rejected.jsonl'
        if path.exists():
            self.rejected = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        self.models, self.frames = {}, set()
        for frame in self.data['frames']:
            sync = int(frame['sync_index'])
            if sync in self.frames or sync not in self.by_sync:
                raise ValueError('duplicate or unknown shared frame')
            self.frames.add(sync)
            accepted_hands = accepted.get(sync, {}).get('hands', [])
            accepted_sides = [int(h['side']) for h in accepted_hands]
            if len(accepted_sides) != len(set(accepted_sides)):
                raise ValueError('duplicate accepted side')
            by_side = {int(h['side']): h for h in accepted_hands}
            seen = set()
            for model in frame['hands']:
                side = int(model['side'])
                if side not in (0, 1) or side in seen or side not in by_side:
                    raise ValueError('duplicate/invalid shared side or model not in accepted.jsonl')
                seen.add(side)
                if int(model.get('sync_index', sync)) != sync:
                    raise ValueError('model/frame sync_index mismatch')
                support = list(model['supporting_cameras'])
                if len(set(support)) < 2 or not set(support) <= self.calibrations.keys():
                    raise ValueError('shared model must have at least two calibrated supporting views')
                if backend == 'gem':
                    joints = array(model['joints_soma_base_m'], (25, 3), 'shared SOMA joints')[list(SOMA_TO_21)]
                else:
                    joints = array(model['joints_base_m'], (21, 3), 'shared MANO joints')
                vertices = np.asarray(model['vertices_base_m'], dtype=float)
                if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
                    raise ValueError('invalid shared vertices')
                if backend != 'gem' and vertices.shape != (778, 3):
                    raise ValueError('MANO shared mesh must have 778 vertices')
                np.testing.assert_allclose(joints, array(by_side[side]['joints_base_m'], (21, 3), 'accepted joints'), atol=1e-7, rtol=0)
                if set(by_side[side]['supporting_cameras']) != set(support):
                    raise ValueError('accepted/shared support views differ')
                if not model.get('parameters'):
                    raise ValueError('shared model parameters missing')
                for name, value in model['parameters'].items():
                    if not np.isfinite(np.asarray(value)).all():
                        raise ValueError(f'invalid shared parameter {name}')
                self.models[sync, side] = dict(joints=joints, vertices=vertices, support=support)
            if seen != set(by_side):
                raise ValueError('accepted/shared hand coverage differs')
        if not set(accepted) <= self.frames:
            raise ValueError('accepted rows missing from shared_pose.pt')
        if self.summary.get('accepted_hand_count') != len(self.models):
            raise ValueError('summary/shared model count differs')

    def fingerprint(self):
        paths = [self.shared / name for name in ('shared_pose.pt', 'accepted.jsonl', 'summary.json', 'run_config.json')]
        paths += [self.normalized / name for name in ('manifest.json', 'multiview_frames.csv')]
        paths += [self.normalized / 'calibration' / f'{c}.json' for c in self.calibrations]
        if (self.shared.parent / 'run_config.json').is_file():
            paths.append(self.shared.parent / 'run_config.json')
        return [dict(path=str(p), sha256=sha256(p), bytes=p.stat().st_size) for p in paths]

    def match(self, label):
        camera = label.row['camera']
        if camera not in self.calibrations:
            raise ValueError(f'labeled camera {camera} absent from normalized calibration')
        source_cal, gt_cal = self.calibrations[camera], label.calibration
        if source_cal.model != gt_cal.model or source_cal.image_size != gt_cal.image_size:
            raise ValueError('prediction/label lens model or raster differs')
        for name, tolerance in (('K', 1e-3), ('distortion', 1e-7), ('T_base_camera', 1e-6)):
            if not np.allclose(getattr(source_cal, name), getattr(gt_cal, name), atol=tolerance, rtol=0):
                raise ValueError(f'prediction/label {name} differs for {camera}; explicit coordinate conversion is required')
        if 'reference_timestamp_ns' not in label.row:
            raise ValueError('labels need reference_timestamp_ns; sync_index alone cannot identify a recording')
        match = self.by_time.get(int(label.row['reference_timestamp_ns']))
        if match is None:
            return None, None, 'not_in_run'
        if int(match[f'{camera}_frame_index']) != int(label.row['source_frame_index']):
            raise ValueError(f'{camera}: source frame differs at timestamp {label.row["reference_timestamp_ns"]}')
        sync = int(match['sync_index'])
        if sync not in self.frames:
            return sync, None, 'not_processed'
        model = self.models.get((sync, label.side))
        return sync, model, 'accepted' if model is not None else 'rejected_or_no_shared_hand'

    def prediction(self, label, model):
        if model is None:
            return dict(joints_camera_m=np.full((21, 3), np.nan), joints_2d_px=np.full((21, 2), np.nan))
        points = base_to_camera(model['joints'], self.calibrations[label.row['camera']])
        pixels, valid = project_points(label.calibration, points)
        pixels[~valid] = np.nan
        pred = dict(joints_camera_m=points, joints_2d_px=pixels)
        if self.backend != 'gem':
            pred['vertices_camera_m'] = base_to_camera(model['vertices'], self.calibrations[label.row['camera']])
        return pred


def shared_protocol():
    return dict(version=1, input='saved accepted multiview shared models, matched by reference timestamp and source frame',
        inference='one shared model per physical hand/frame; no independent-view fallback',
        placement='saved_shared_geometry', absolute_metric='direct shared base -> calibrated camera transformation; no translation refit',
        joint_order='wrist, thumb[4], index[4], middle[4], ring[4], pinky[4]',
        root_alignment='subtract each hand wrist (joint 0); wrist included in mean',
        pa_alignment='per-hand similarity: proper rotation, scale, translation; no reflection',
        mesh_alignment='same joint-derived transform; SOMA/MANO vertex correspondence is unavailable',
        pixel_metric='project final shared 3D through actual lens model; compare with label joint projections in ALL labeled views',
        units=dict(geometry='metres', errors_3d='millimetres', errors_2d='source pixels', pck_auc='fraction [0,1]'),
        missing_policy='rejected/missing/unprocessed hand frames remain in coverage/PCK/AUC denominators',
        aggregation='primary: each labeled camera/hand sample; secondary: one sample per labeled timestamp/physical side',
        pck_3d_thresholds_mm=[20, 30, 50], auc_3d='exact normalized integral on [0,50] mm', pck_2d_thresholds_px=[5, 10, 20])


def evaluate_shared(args, dataset, source):
    from evaluate_hand_models import report, write_json
    started = time.perf_counter()
    sources = {(r.get('source_dataset_id'), r.get('source_group')) for r in dataset.selected}
    if len(sources) != 1:
        raise ValueError('one shared recording must be evaluated against one label source at a time')
    config = dict(stage='shared_multiview_evaluation', backend=args.backend, dataset_fingerprint=dataset.fingerprint(),
                  assets=source.fingerprint(), protocol=shared_protocol(),
                  model=dict(shared_geometry='SOMA' if args.backend == 'gem' else 'MANO', run=str(source.shared)),
                  gt_consistency_mm=args.gt_consistency_mm,
                  evaluation_code_sha256={p.name: sha256(p) for p in (
                      Path(__file__), Path(__file__).with_name('metrics.py'), Path(__file__).with_name('dataset.py'),
                      Path(__file__).parents[1] / 'evaluate_hand_models.py',
                      Path(__file__).parents[1] / 'evaluate_shared_hand_models.py')})
    config_path = args.output / 'run_config.json'
    if config_path.exists():
        if not args.resume or json.loads(config_path.read_text()) != config:
            raise ValueError('output exists with different config or without --resume; use a new evaluation output')
    elif args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError('nonempty evaluation output without run_config.json')
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(config_path, config)
    scores, masks, errors = [], [], {name: [] for name in ERROR_NAMES}
    physical, max_gt_difference = {}, 0.
    matched = 0
    for i, row in enumerate(dataset.selected):
        label = dataset.load(row)
        sync, model, status = source.match(label)
        matched += sync is not None
        pred = source.prediction(label, model)
        score, per_joint = measure(label, pred, placement='native')
        score.update(shared_sync_index=sync, shared_status=status,
                     supporting_view_count=len(model['support']) if model else 0,
                     evaluated_view_is_supporting=bool(model and row['camera'] in model['support']))
        scores.append(score)
        masks.append(label.valid)
        for name in ERROR_NAMES:
            errors[name].append(per_joint[name])
        key = int(row['reference_timestamp_ns']), label.side
        base_gt = camera_to_base(label.joints, label.calibration)
        if key in physical:
            existing = physical[key]
            difference = float(np.max(np.linalg.norm(base_gt - existing['gt'], axis=1)))
            max_gt_difference = max(max_gt_difference, difference)
            if difference > args.gt_consistency_mm / 1000:
                raise ValueError(f'cross-view label geometry differs by {difference * 1000:.5f} mm; cannot deduplicate hand/frame labels')
            if not np.array_equal(existing['mask'], label.valid):
                raise ValueError('cross-view label joint masks differ; cannot deduplicate hand/frame labels')
        else:
            base_label = replace(label, row={**row, 'camera': 'base'}, joints=base_gt,
                                 vertices=camera_to_base(label.vertices, label.calibration), pixels=np.full((21, 2), np.nan))
            base_pred = dict(joints_camera_m=model['joints'] if model else np.full((21, 3), np.nan), joints_2d_px=np.full((21, 2), np.nan))
            if model and args.backend != 'gem':
                base_pred['vertices_camera_m'] = model['vertices']
            physical_score, physical_errors = measure(base_label, base_pred, placement='native')
            physical[key] = dict(gt=base_gt, score=physical_score, errors=physical_errors, mask=label.valid)
        if (i + 1) % 1000 == 0:
            print(f'[shared evaluation {args.backend}] {i + 1}/{len(dataset.selected)} labeled views', flush=True)
    if not matched:
        raise ValueError('no timestamp overlap between this shared run and the labeled recording')
    physical_rows = list(physical.values())
    unique = summarize([r['score'] for r in physical_rows],
                       {name: [r['errors'][name] for r in physical_rows] for name in ERROR_NAMES},
                       [r['mask'] for r in physical_rows])
    unique.pop('pixel')
    unique['metrics'].pop('mean_2d_error_px')
    unique['aggregation'] = 'one deterministic reference label per timestamp/side; base coordinates; no repeated camera weighting'
    return report(args, dataset, config, scores, errors, masks, time.perf_counter() - started, 0,
        evaluation_protocol=shared_protocol(),
        extra=dict(unique_hand_frames=unique, shared_evaluation=dict(
            matched_label_views=matched, status_counts=dict(Counter(row['shared_status'] for row in scores)),
            shared_run=str(source.shared), normalized_dataset=str(source.normalized),
            upstream_summary=source.summary, experiment_config=source.experiment_config,
            input_provenance=dict(source=source.manifest.get('source'), calibration=source.manifest.get('calibration'),
                                  evaluation_input=source.manifest.get('evaluation_input')),
            max_cross_view_gt_joint_difference_mm=max_gt_difference * 1000,
            gt_consistency_tolerance_mm=args.gt_consistency_mm)),
        limitations=['Labels are fitted references: ' + str(dataset.summary.get('supervision_source')),
            'Only exported labeled views are scored, including views not selected by the shared fitter.',
            'Detector/GT-box provenance is recorded from the upstream run; coverage is shared reconstruction coverage.',
            'GEM shares SOMA parameters, not MANO; its vertex error against MANO is unavailable.',
            'Missing shared results are failures, never replaced with per-camera predictions.'])
