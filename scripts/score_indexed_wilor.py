"""Pair single-camera and shared MANO scores by official sample ID and box protocol."""
import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from camera_models import project_points
from evaluate_detected_wilor import native_meshes, select_by_side
from fit_shared_wilor_multiview import load_mano
from hand_evaluation.dataset import LabeledDataset, sha256
from hand_evaluation.metrics import ERROR_NAMES, bbox_iou, measure
from hand_evaluation.shared import SharedResult, camera_to_base


METRICS = ('absolute_mpjpe_mm', 'root_relative_mpjpe_mm', 'pa_mpjpe_mm', 'wrist_error_mm',
           'mean_2d_error_px', 'absolute_mpvpe_mm', 'root_relative_mpvpe_mm', 'pa_mpvpe_joint_aligned_mm')
METHODS = ('single','shared')


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def audit_index(dataset):
    split = read_json(dataset.root/'splits.json')
    fingerprint = sha256(dataset.root/'index.jsonl')
    assert sha256(dataset.root/split['manifest']) == fingerprint
    assert len(dataset.selected) == len(dataset.rows) == split['samples']['test']
    ids = {}
    frames = set()
    with (dataset.root/'frames.jsonl').open() as stream:
        for row in map(json.loads,stream):
            assert row['image'] not in frames
            frames.add(row['image'])
            for hand in row['hands']:
                key = hand['index']
                assert key not in ids
                ids[key] = (row['source_group'], row['sync_index'], row['camera'], row['reference_timestamp_ns'],
                            row['source_frame_index'], row['image'], hand['side'], hand['label'])
    for row in dataset.rows:
        assert ids.pop(row['index']) == tuple(row[k] for k in ('source_group','sync_index','camera',
                    'reference_timestamp_ns','source_frame_index','image','side','label'))
    assert not ids and len(frames) == len(dataset.groups)
    return split, fingerprint


def load_group(args, dataset, group, rows, source):
    root = args.output/group
    native = root / f'{args.backend}_multiview'
    summary = read_json(native/'summary.json')
    assert summary['model_load_sha256'] == sha256(args.output/'model_load.json')
    model = read_json(args.output/'model_load.json')
    if model.get('proposal_source','detector') != args.proposal_source:
        raise ValueError('Scoring protocol differs from the saved prediction protocol')
    by_cam = defaultdict(dict)
    for row in rows:
        by_cam[row['camera']][row['sync_index']] = row['image']
    counts, images, observed = Counter(),Counter(),Counter()
    detections = {}
    max_delta_ns = {}
    prediction_hashes = {}
    for camera, info in summary['cameras'].items():
        path = native/camera/'predictions.jsonl'
        assert sha256(path) == info['sha256']
        prediction_hashes[camera] = info['sha256']
        deltas = []
        with path.open() as stream:
            for row, (index, sync) in zip(map(json.loads,stream),source.by_sync.items(),strict=True):
                assert row['sync_index'] == index and row['timestamp_ns'] == int(sync[camera+'_timestamp_ns'])
                assert row['source_frame_index'] == int(sync[camera+'_frame_index'])
                expected = by_cam[camera].get(index)
                assert expected == row['image']
                if not expected:
                    assert not row['hands']
                    continue
                assert sha256(dataset.path(expected)) == row['image_sha256']
                deltas.append(abs(row['timestamp_ns']-int(sync['reference_timestamp_ns'])))
                images[index] += 1
                counts[camera] += 1
                selected = select_by_side(row['hands'])
                for side,h in selected.items():
                    assert h['proposal_source'] == args.proposal_source and h['detector_is_right'] == h['is_right'] == side
                    observed[index,side] += 1
                    detections[index,camera,side] = h
        assert counts[camera] == info['images']
        max_delta_ns[camera] = max(deltas, default=None)
    assert len(images) == len(source.by_sync)
    evidence = dict(proposal_source=args.proposal_source, image_counts=dict(counts), timestamp_max_abs_delta_ns=max_delta_ns,
                    prediction_sha256=prediction_hashes, sync_times=len(images),
                    indexed_cameras_per_sync=dict(Counter(images.values())),
                    detector_views_per_sync_side=dict(Counter(observed[s,k] for s in images for k in (0,1))),
                    source_files=source.fingerprint())
    return detections, images, observed, evidence


def load_fah_meshes(native):
    """FAH's pixel-aligned native surface differs from a replay of its MANO seed."""
    meshes = {}
    for camera, info in read_json(native/'summary.json')['cameras'].items():
        path = native/camera/'meshes.npz'
        if sha256(path) != info['mesh_sha256']:
            raise ValueError(f'{camera}: unexpected native mesh archive')
        with np.load(path) as data:
            vertices = data['vertices_camera_m']
            if vertices.shape != (info['detections'],778,3) or not np.isfinite(vertices).all():
                raise ValueError(f'{camera}: invalid FAH vertices')
            for frame, hyp, side, mesh in zip(data['source_frame_indices'],data['hypothesis_indices'],
                                              data['is_right'],vertices,strict=True):
                key = (camera,int(frame),int(hyp),int(side))
                if key in meshes:
                    raise ValueError(f'Duplicate native mesh {key}')
                meshes[key] = mesh
    return meshes


def score_group(args, group, dataset, group_rows):
    root = args.output/group
    source = SharedResult(root,args.backend)
    view_ids = {(int(frame['sync_index']),int(model['side'])):
                {c:int(v['detection_index']) for c,v in model['views'].items() if v['supported']}
                for frame in source.data['frames'] for model in frame['hands']}
    hands, images, observed, evidence = load_group(args,dataset,group,group_rows,source)
    rejected_fit = {(r['sync_index'],r['side']) for r in source.rejected if 'side' in r}
    if args.backend == 'fah':
        archived = load_fah_meshes(root/'fah_multiview')
    else:
        layer = load_mano(Path(__file__).resolve().parents[1]/'models/mano')
    gt_base = {}
    gt_masks = {}
    max_gt_delta = 0.
    max_native_projection_delta = 0.
    projection_checks = 0
    scores, valid_masks, errors = [], [], defaultdict(list)
    for start in range(0,len(group_rows),64):
        labels = [dataset.load(r) for r in group_rows[start:start+64]]
        matches = []
        for label in labels:
            r = label.row
            h = hands.get((r['sync_index'], r['camera'], label.side))
            if args.proposal_source == 'gt_box_physical_side':
                if h is None or not np.array_equal(h['bbox_xyxy'],label.bbox) or h['confidence'] != 1:
                    raise ValueError(f"Prediction lacks the exact indexed GT box for {r['index']}")
            overlap = bbox_iou(h['bbox_xyxy'],label.bbox) if h else None
            status = 'no_same_side_box' if h is None else 'matched' if overlap >= .5 else 'box_iou_below_0.5'
            matches.append((h if status == 'matched' else None,status,overlap))
        if args.backend == 'fah':
            meshes = {j: archived[label.row['camera'],label.row['source_frame_index'],
                                  hand['hypothesis_index'],label.side]
                      for j,(label,(hand,_,_)) in enumerate(zip(labels,matches)) if hand is not None}
        else:
            meshes = native_meshes([m[0] for m in matches],layer)
        for j,(label,(hand,status,overlap)) in enumerate(zip(labels,matches,strict=True)):
            r = label.row
            sync, shared, matched = source.match(label)
            assert sync == r['sync_index'] and sync in source.frames
            key = sync,label.side
            gt = camera_to_base(label.joints,label.calibration)
            difference = np.linalg.norm(gt - gt_base.setdefault(key,gt),axis=1).max()
            max_gt_delta = max(max_gt_delta, float(difference)*1000)
            assert max_gt_delta <= .1
            np.testing.assert_array_equal(label.valid,gt_masks.setdefault(key,label.valid))
            count = observed[key]
            if shared:
                shared_status = 'accepted'
            elif images[sync] < 2:
                shared_status = 'fewer_than_two_indexed_views'
            elif count < 2:
                shared_status = 'fewer_than_two_detector_views'
            elif key in rejected_fit:
                shared_status = 'fit_quality_rejected'
            else:
                shared_status = 'no_consistent_association'
            single = dict(joints_camera_m=np.full((21,3),np.nan),joints_2d_px=np.full((21,2),np.nan))
            if hand:
                single = dict(joints_camera_m=np.asarray(hand['joints_3d_camera_relative'])+hand['camera_translation'],
                              joints_2d_px=hand['joints_2d'],vertices_camera_m=meshes[j])
                check_points = single['joints_camera_m'].copy()
                if args.backend == 'wilor':
                    check_points[:,2] = np.maximum(check_points[:,2],.01)
                projected, valid = project_points(label.calibration,check_points)
                if valid.any():
                    delta = np.linalg.norm(projected[valid]-np.asarray(hand['joints_2d'])[valid],axis=1)
                    max_native_projection_delta = max(max_native_projection_delta,float(delta.max()))
                    projection_checks += int(valid.sum())
            result = {k:r[k] for k in ('index','source_group','sync_index','reference_timestamp_ns','camera',
                                       'source_frame_index','image','label','side')}
            result.update(indexed_views=images[sync], detector_views=count, single_status=status, detection_iou=overlap,
                          proposal_source=args.proposal_source,
                          shared_status=shared_status, supporting_views=len(shared['support']) if shared else 0,
                          evaluated_view_supports=bool(shared and r['camera'] in shared['support']),
                          single_detection_id=hand['detection_index'] if hand else None,
                          shared_support_detection_id=view_ids.get(key,{}).get(r['camera']))
            result['same_support_detection'] = (hand is not None and result['shared_support_detection_id'] ==
                                                hand['detection_index'])
            for name, pred in zip(METHODS,(single,source.prediction(label,shared)),strict=True):
                score, per_joint = measure(label,pred,placement='native')
                for metric in METRICS:
                    result[f'{name}_{metric}'] = score[metric]
                for metric in ERROR_NAMES:
                    errors[name+'_'+metric].append(per_joint[metric].astype(np.float32))
            result['paired'] = all(result[n+'_'+k] is not None for n in METHODS for k in METRICS[:3])
            for k in METRICS[:3]:
                result['delta_'+k] = result['shared_'+k] - result['single_'+k] if result['paired'] else None
            scores.append(result)
            valid_masks.append(label.valid)
        if start % 2048 == 0:
            print(f'{group} scored {len(scores)}/{len(group_rows)}',flush=True)
    table = pd.DataFrame(scores)
    table.to_csv(root/'per_sample.csv',index=False)
    np.savez_compressed(root/'joint_errors.npz',ids=table['index'].to_numpy(),valid=np.stack(valid_masks),**errors)
    evidence.update(max_cross_camera_gt_disagreement_mm=max_gt_delta, eligible=len(table), unique_physical_hands=len(gt_base),
                    native_forward_projected_joints=projection_checks, max_native_projection_delta_px=max_native_projection_delta,
                    paired=int(table.paired.sum()), per_sample_sha256=sha256(root/'per_sample.csv'),
                    joint_errors_sha256=sha256(root/'joint_errors.npz'))
    write_json(root/'score_audit.json',evidence)


def subsets(table):
    yield 'all','all',table
    for col in ('source_group','camera','side','indexed_views','detector_views','supporting_views','evaluated_view_supports',
                'same_support_detection'):
        for name, selected in table.groupby(col):
            yield col,str(name),selected
    for (group, camera), selected in table.groupby(['source_group','camera']):
        yield 'group_camera',f'{group}/{camera}',selected


def aggregate(args, groups, fingerprint):
    chunks, arrays = [], defaultdict(list)
    for group in groups:
        root = args.output/group
        evidence = read_json(root/'score_audit.json')
        if evidence.get('proposal_source','detector') != args.proposal_source:
            raise ValueError(f'{group}: source mismatch in scores')
        assert sha256(root/'per_sample.csv') == evidence['per_sample_sha256']
        assert sha256(root/'joint_errors.npz') == evidence['joint_errors_sha256']
        table = pd.read_csv(root/'per_sample.csv')
        with np.load(root/'joint_errors.npz') as archive:
            np.testing.assert_array_equal(archive['ids'],table['index'])
            for name in archive.files:
                if name != 'ids':
                    arrays[name].append(archive[name])
        chunks.append(table)
    table = pd.concat(chunks,ignore_index=True)
    arrays = {k:np.concatenate(v) for k,v in arrays.items()}
    assert not table['index'].duplicated().any()
    declared = read_json(args.dataset/'splits.json')['samples']['test']
    assert len(table) == declared and table['index'].nunique() == declared
    table.to_csv(args.output/'per_sample.csv',index=False)
    index_fields = ['index','source_group','sync_index','reference_timestamp_ns','camera','source_frame_index','side','image']
    table.loc[table.paired,index_fields].to_csv(args.output/'paired_ids.csv',index=False)
    metrics, deltas, coverage, failures, physical = [],[],[],[],[]
    for dim, key, selected in subsets(table):
        paired = selected[selected.paired]
        ids = dict(dimension=dim,subset=key)
        unique = selected.drop_duplicates(['source_group','sync_index','side'])
        coverage.append(dict(**ids,eligible=len(selected),physical_hands=len(unique),paired=len(paired),
                             single_valid=int(selected.single_absolute_mpjpe_mm.count()),
                             shared_valid=int(selected.shared_absolute_mpjpe_mm.count()),
                             indexed_views_lt2=int((selected.indexed_views<2).sum()),
                             detector_views_lt2=int((selected.detector_views<2).sum()),
                             paired_rate=len(paired)/len(selected)))
        for m in METHODS:
            for status,count in selected[m+'_status'].value_counts().items():
                failures.append(dict(**ids,method=m,status=status,samples=int(count)))
            for cohort, frame in [('eligible',selected),('paired',paired)]:
                row = dict(**ids,method=m,cohort=cohort,denominator=len(frame))
                row['valid'] = int(frame[m+'_absolute_mpjpe_mm'].count())
                row['coverage'] = row['valid']/len(frame) if len(frame) else None
                for k in METRICS:
                    x = frame[m+'_'+k].dropna()
                    row[k] = x.mean()
                    row[k+'_median'] = x.median()
                    row[k+'_p95'] = x.quantile(.95)
                mask = arrays['valid'][frame.index]
                for k in ERROR_NAMES:
                    e = arrays[m+'_'+k][frame.index][mask]
                    limit = 10 if k=='pixel' else 20
                    row[f'{k}_pck{limit}'] = np.mean(e<=limit) if len(e) else None
                metrics.append(row)
        for status,count in unique.shared_status.value_counts().items():
            physical.append(dict(**ids,shared_status=status,physical_hands=int(count)))
        for k in METRICS:
            diff = (paired['shared_'+k]-paired['single_'+k]).dropna()
            deltas.append(dict(**ids,metric=k,n=len(diff),mean_delta=diff.mean(),median_delta=diff.median(),
                               p05_delta=diff.quantile(.05),p95_delta=diff.quantile(.95),
                               improved=int((diff<0).sum()),worsened=int((diff>0).sum()),ties=int((diff==0).sum())))
    for name, records in [('metrics',metrics),('paired_deltas',deltas),('coverage',coverage),('statuses',failures),
                          ('physical_hand_statuses',physical)]:
        pd.DataFrame(records).to_csv(args.output/(name+'.csv'),index=False)
    paired_hands = table.loc[table.paired].groupby(['source_group','sync_index','side']).agg(
        paired_camera_labels=('index','size'),
        **{n+'_'+k:(n+'_'+k,'mean') for n in METHODS for k in METRICS[:3]})
    paired_hands.to_csv(args.output/'paired_physical_hands.csv')
    # An empty matched cohort has undefined means/quantiles, but its zero coverage is reportable.
    def nullable(frame):
        return frame.astype(object).where(frame.notna(), None)
    physical_means = paired_hands.drop(columns='paired_camera_labels').mean()
    write_json(args.output/'summary.json',dict(manifest_sha256=fingerprint,groups=groups,proposal_source=args.proposal_source,
               all_labels=coverage[0],metrics=nullable(pd.DataFrame(metrics[:4])).to_dict('records'),
               paired_physical_hands=len(paired_hands), equal_physical_hand_means=nullable(physical_means).to_dict(),
               deltas=nullable(pd.DataFrame(deltas[:len(METRICS)])).to_dict('records'),statuses=[r for r in failures if r['dimension']=='all'],
               physical_statuses=[r for r in physical if r['dimension']=='all']))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--backend',choices=('wilor','fah'),default='wilor')
    p.add_argument('--proposal-source',choices=('detector','gt_box_physical_side'),default='detector')
    p.add_argument('--group',help='score just this recording')
    p.add_argument('--aggregate',action='store_true',help='aggregate completed recording scores')
    args = p.parse_args()
    torch.set_num_threads(2)
    dataset = LabeledDataset(args.dataset)
    split, fingerprint = audit_index(dataset)
    groups = [g['recording'] for g in split['recordings']]
    if args.aggregate:
        aggregate(args,groups,fingerprint)
    else:
        if args.group not in groups:
            p.error('an official recording name is required')
        rows = [r for r in dataset.selected if r['source_group'] == args.group]
        score_group(args,args.group,dataset,rows)


if __name__ == '__main__':
    main()
